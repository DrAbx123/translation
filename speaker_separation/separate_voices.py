#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音分离脚本 - 将混合音频中的不同说话人声音分离成独立音轨
即使多人同时说话也能分开

使用 pyannote-audio 的 SpeechSeparation pipeline
"""

import torch
import numpy as np
import os

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INPUT_AUDIO = "input.mp3"
OUTPUT_DIR = "speaker_separation/output"
# hahmadraz/sepformer-libri4mix 需要 48kHz
SAMPLE_RATE = 48000


def load_audio(file_path):
    """加载音频文件并转换为正确格式"""
    import subprocess
    import tempfile
    import soundfile as sf

    # 用 ffmpeg 转换（支持各种格式，确保单声道 16kHz）
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        print(f"    转换音频格式...")
        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",  # 单声道
            "-y", temp_path
        ], check=True, capture_output=True)

        audio, _ = sf.read(temp_path)
        return audio.astype(np.float32)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def audio_to_pyannote_format(audio_array, sample_rate=16000):
    """将音频数组转换为 pyannote 内存格式"""
    waveform = torch.from_numpy(audio_array).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (samples,) -> (1, samples)
    return {"waveform": waveform, "sample_rate": sample_rate}


def main():
    print("=" * 60)
    print("语音分离 - 分离混合语音中的不同说话人")
    print("（可以处理多人同时说话的情况）")
    print("=" * 60)

    if not os.path.exists(INPUT_AUDIO):
        raise FileNotFoundError(f"找不到音频文件: {INPUT_AUDIO}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 加载音频
    print("\n[1] 加载音频...")
    audio = load_audio(INPUT_AUDIO)
    duration = len(audio) / SAMPLE_RATE
    print(f"    时长: {duration/60:.1f} 分钟 ({duration:.1f} 秒)")

    # 检查是否有语音分离模型
    print("\n[2] 加载语音分离模型...")

    try:
        from pyannote.audio import Pipeline as PyannotePipeline

        # 尝试加载语音分离 pipeline
        # 注意：这需要额外的模型权限
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speech-separation-ami-1.0",
            token=HF_TOKEN
        )

        if device == "cuda":
            pipeline = pipeline.to(torch.device(device))

        print("    ✓ 语音分离模型加载成功")

    except Exception as e:
        print(f"    ⚠ 语音分离模型加载失败: {e}")
        print("\n    尝试使用备选方案...")

        # 备选方案：使用 speechbrain 的分离模型
        try:
            from speechbrain.inference.separation import SepformerSeparation

            # 使用4人分离模型
            # hahmadraz/sepformer-libri4mix: 4人分离, 48kHz
            MODEL_NAME = "hahmadraz/sepformer-libri4mix"
            EXPECTED_SPEAKERS = 4  # 这个模型固定输出4个声源

            print(f"    加载 SpeechBrain SepFormer 模型: {MODEL_NAME}")
            print(f"    (此模型固定分离 {EXPECTED_SPEAKERS} 个声源，采样率 48kHz)")
            separator = SepformerSeparation.from_hparams(
                source=MODEL_NAME,
                savedir=f"speaker_separation/pretrained_models/{MODEL_NAME.replace('/', '_')}",
                run_opts={"device": device}  # 使用 GPU
            )

            print(f"    ✓ SpeechBrain 分离模型加载成功 (运行于 {device})")

            # 使用 SpeechBrain 进行分离
            print("\n[3] 分离语音...")

            import soundfile as sf
            import gc

            # 分段处理配置
            CHUNK_DURATION = 60  # 3分钟
            OVERLAP_DURATION = 30  # 30秒重叠用于拼接
            MAX_SPEAKERS = 4  # 最多4个人

            print(f"    音频分段处理（每段 {CHUNK_DURATION} 秒，重叠 {OVERLAP_DURATION} 秒）...")

            # 先处理第一小段确定有多少声源
            test_chunk = audio[:int(10 * SAMPLE_RATE)]  # 只用10秒测试
            audio_tensor = torch.from_numpy(test_chunk).float().unsqueeze(0).to(device)
            est_sources = separator.separate_batch(audio_tensor)
            # sepformer-libri3mix 固定输出3个声源，但可以根据能量过滤静音声源
            num_sources = min(est_sources.shape[2], MAX_SPEAKERS)  # shape是 (batch, time, num_sources)
            print(f"    模型输出 {est_sources.shape[2]} 个声源，将处理 {num_sources} 个")

            del audio_tensor, est_sources, test_chunk
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 初始化每个声源的音频列表
            all_sources = [[] for _ in range(num_sources)]

            # 分段处理
            chunk_start = 0
            chunk_idx = 0
            total_chunks = int(np.ceil(duration / (CHUNK_DURATION - OVERLAP_DURATION)))

            while chunk_start < duration:
                chunk_end = min(chunk_start + CHUNK_DURATION, duration)

                start_sample = int(chunk_start * SAMPLE_RATE)
                end_sample = int(chunk_end * SAMPLE_RATE)
                chunk_audio = audio[start_sample:end_sample]

                print(f"    处理第 {chunk_idx+1}/{total_chunks} 段: {chunk_start/60:.1f}-{chunk_end/60:.1f} 分钟...")

                audio_tensor = torch.from_numpy(chunk_audio).float().unsqueeze(0).to(device)
                est_sources = separator.separate_batch(audio_tensor)

                # 计算保留部分
                # sepformer 输出 shape: (batch, time, num_sources)
                if chunk_idx == 0:
                    # 第一段：保留到重叠点
                    if chunk_end < duration:
                        keep_end = int((CHUNK_DURATION - OVERLAP_DURATION/2) * SAMPLE_RATE)
                        for i in range(num_sources):
                            all_sources[i].append(est_sources[0, :keep_end, i].cpu().numpy())
                    else:
                        # 只有一段
                        for i in range(num_sources):
                            all_sources[i].append(est_sources[0, :, i].cpu().numpy())
                elif chunk_end >= duration:
                    # 最后一段：跳过重叠前半
                    skip_samples = int(OVERLAP_DURATION/2 * SAMPLE_RATE)
                    for i in range(num_sources):
                        all_sources[i].append(est_sources[0, skip_samples:, i].cpu().numpy())
                else:
                    # 中间段：跳过两端重叠
                    skip_samples = int(OVERLAP_DURATION/2 * SAMPLE_RATE)
                    keep_end = int((CHUNK_DURATION - OVERLAP_DURATION/2) * SAMPLE_RATE)
                    for i in range(num_sources):
                        all_sources[i].append(est_sources[0, skip_samples:keep_end, i].cpu().numpy())

                # 清理显存
                del audio_tensor, est_sources, chunk_audio
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                chunk_start = chunk_end - OVERLAP_DURATION
                if chunk_start >= duration - OVERLAP_DURATION:
                    break
                chunk_idx += 1

            # 合并并保存
            print("\n[4] 保存分离的音轨...")

            for i in range(num_sources):
                combined = np.concatenate(all_sources[i])

                # 归一化
                max_val = np.abs(combined).max()
                if max_val > 0:
                    combined = combined / max_val * 0.9

                output_file = os.path.join(OUTPUT_DIR, f"speaker_{i+1}.wav")
                sf.write(output_file, combined, SAMPLE_RATE)

                seg_duration = len(combined) / SAMPLE_RATE
                print(f"    保存: {output_file} ({seg_duration/60:.1f} 分钟)")

            print("\n" + "=" * 60)
            print("完成！")
            print(f"输出目录: {OUTPUT_DIR}/")
            print("=" * 60)
            return

        except Exception as e2:
            print(f"    ⚠ SpeechBrain 也失败了: {e2}")
            print("\n    尝试第三个方案：使用 demucs...")

            # 第三个备选：demucs（主要用于音乐分离，但对人声也有效）
            try:
                import demucs.separate
                print("    demucs 可用，但主要用于音乐分离")
            except:
                pass

            print("\n" + "=" * 60)
            print("错误：无法加载任何语音分离模型")
            print("\n可能的解决方案：")
            print("1. 安装 speechbrain: pip install speechbrain")
            print("2. 或者使用说话人分离（不能处理重叠语音）:")
            print("   python split_speakers.py")
            print("=" * 60)
            return

    # 使用 pyannote 语音分离
    print("\n[3] 分离语音...")

    # 分段处理长音频
    CHUNK_DURATION = 60  # 语音分离更占资源，用更短的分段

    if duration > CHUNK_DURATION:
        print(f"    音频较长，分段处理（每段 {CHUNK_DURATION} 秒）...")

        all_sources = {}  # speaker_id -> list of audio chunks

        chunk_start = 0
        chunk_idx = 0

        while chunk_start < duration:
            chunk_end = min(chunk_start + CHUNK_DURATION, duration)

            start_sample = int(chunk_start * SAMPLE_RATE)
            end_sample = int(chunk_end * SAMPLE_RATE)
            chunk_audio = audio[start_sample:end_sample]

            print(f"    处理第 {chunk_idx+1} 段: {chunk_start/60:.1f}-{chunk_end/60:.1f} 分钟...")

            audio_input = audio_to_pyannote_format(chunk_audio, SAMPLE_RATE)
            result = pipeline(audio_input)

            # 收集分离结果
            for speaker_id, source_audio in result.items():
                if speaker_id not in all_sources:
                    all_sources[speaker_id] = []
                all_sources[speaker_id].append(source_audio)

            # 清理显存
            del audio_input, result
            if device == "cuda":
                torch.cuda.empty_cache()

            chunk_start = chunk_end
            chunk_idx += 1

        # 合并各段
        print("\n[4] 保存分离的音轨...")
        import soundfile as sf

        for speaker_id, chunks in all_sources.items():
            combined = np.concatenate(chunks)

            # 归一化
            max_val = np.abs(combined).max()
            if max_val > 0:
                combined = combined / max_val * 0.9

            output_file = os.path.join(OUTPUT_DIR, f"{speaker_id}.wav")
            sf.write(output_file, combined, SAMPLE_RATE)

            seg_duration = len(combined) / SAMPLE_RATE
            print(f"    保存: {output_file} ({seg_duration/60:.1f} 分钟)")

    else:
        # 短音频直接处理
        audio_input = audio_to_pyannote_format(audio, SAMPLE_RATE)
        result = pipeline(audio_input)

        print("\n[4] 保存分离的音轨...")
        import soundfile as sf

        for speaker_id, source_audio in result.items():
            # 归一化
            max_val = np.abs(source_audio).max()
            if max_val > 0:
                source_audio = source_audio / max_val * 0.9

            output_file = os.path.join(OUTPUT_DIR, f"{speaker_id}.wav")
            sf.write(output_file, source_audio, SAMPLE_RATE)

            seg_duration = len(source_audio) / SAMPLE_RATE
            print(f"    保存: {output_file} ({seg_duration/60:.1f} 分钟)")

    print("\n" + "=" * 60)
    print("完成！")
    print(f"输出目录: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
