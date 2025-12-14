#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人分离脚本 - 将音频按说话人切分成多个文件
"""

import torch
import numpy as np
import os
from collections import defaultdict

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INPUT_AUDIO = "input.mp3"
OUTPUT_DIR = "speaker_audio"
SAMPLE_RATE = 16000


def load_audio(file_path):
    """加载音频文件"""
    import soundfile as sf
    
    # 尝试直接读取
    try:
        audio, sr = sf.read(file_path)
        if sr != SAMPLE_RATE:
            # 需要重采样
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        return audio
    except:
        pass
    
    # 用 ffmpeg 转换（支持 mp3 等格式）
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    try:
        subprocess.run([
            "ffmpeg", "-i", file_path, 
            "-ar", str(SAMPLE_RATE), 
            "-ac", "1",  # 单声道
            "-y", temp_path
        ], check=True, capture_output=True)
        
        audio, _ = sf.read(temp_path)
        return audio
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
    print("说话人分离 - 将音频按说话人切分")
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
    
    # 加载说话人分离模型
    print("\n[2] 加载说话人分离模型...")
    from pyannote.audio import Pipeline as PyannotePipeline
    
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    ).to(torch.device(device))
    
    # 分段处理长音频
    CHUNK_DURATION = 300  # 5分钟一段
    
    all_segments = []
    
    if duration > CHUNK_DURATION:
        print(f"\n[3] 分段处理音频（每段 {CHUNK_DURATION} 秒）...")
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
            
            # 收集这段的结果（调整时间偏移）
            for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True):
                all_segments.append({
                    "start": turn.start + chunk_start,
                    "end": turn.end + chunk_start,
                    "speaker": speaker
                })
            
            # 清理显存
            del audio_input, result
            if device == "cuda":
                torch.cuda.empty_cache()
            
            chunk_start = chunk_end
            chunk_idx += 1
    else:
        print("\n[3] 分析说话人...")
        audio_input = audio_to_pyannote_format(audio, SAMPLE_RATE)
        result = pipeline(audio_input)
        
        for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True):
            all_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
    
    # 清理模型
    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 统计说话人
    speaker_times = defaultdict(float)
    for seg in all_segments:
        speaker_times[seg["speaker"]] += seg["end"] - seg["start"]
    
    print(f"\n[4] 检测到 {len(speaker_times)} 个说话人:")
    for speaker in sorted(speaker_times.keys()):
        time = speaker_times[speaker]
        print(f"    {speaker}: {time/60:.1f} 分钟")
    
    # 按说话人合并音频片段并输出
    print(f"\n[5] 按说话人切分并保存音频...")
    import soundfile as sf
    
    for speaker in sorted(speaker_times.keys()):
        # 收集这个说话人的所有片段
        speaker_segments = [s for s in all_segments if s["speaker"] == speaker]
        speaker_segments.sort(key=lambda x: x["start"])
        
        # 合并音频
        speaker_audio = []
        for seg in speaker_segments:
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = int(seg["end"] * SAMPLE_RATE)
            speaker_audio.append(audio[start_sample:end_sample])
        
        if speaker_audio:
            combined_audio = np.concatenate(speaker_audio)
            
            # 保存
            output_file = os.path.join(OUTPUT_DIR, f"{speaker}.wav")
            sf.write(output_file, combined_audio, SAMPLE_RATE)
            
            seg_duration = len(combined_audio) / SAMPLE_RATE
            print(f"    保存: {output_file} ({seg_duration/60:.1f} 分钟, {len(speaker_segments)} 个片段)")
    
    # 保存分段信息（方便后续使用）
    import json
    segments_file = os.path.join(OUTPUT_DIR, "segments.json")
    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_duration": duration,
            "speakers": list(speaker_times.keys()),
            "speaker_durations": dict(speaker_times),
            "segments": all_segments
        }, f, ensure_ascii=False, indent=2)
    print(f"    保存分段信息: {segments_file}")
    
    print("\n" + "=" * 60)
    print("完成！")
    print(f"输出目录: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
