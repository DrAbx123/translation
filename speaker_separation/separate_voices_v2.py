#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音分离脚本 v2 - 支持任意数量的说话人

思路：
1. 用 pyannote diarization 识别所有说话人和他们说话的时间段
2. 提取每个人的纯净片段，建立声纹(speaker embedding)
3. 对于重叠语音片段，用分离模型分开
4. 用声纹匹配把分离的声音归类到对应的说话人
5. 最后为每个说话人生成完整的音轨
"""

import torch
import numpy as np
import os
from collections import defaultdict

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INPUT_AUDIO = "input.mp3"
OUTPUT_DIR = "speaker_separation/output"
SAMPLE_RATE = 16000  # pyannote 和声纹模型都用 16kHz
MAX_SPEAKERS = 4  # 限制最大说话人数量


def load_audio(file_path):
    """加载音频文件"""
    import subprocess
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        print("    转换音频格式...")
        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-y", temp_path
        ], check=True, capture_output=True)

        audio, _ = sf.read(temp_path)
        return audio.astype(np.float32)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    print("=" * 60)
    print("语音分离 v2 - 支持任意数量的说话人")
    print("=" * 60)

    if not os.path.exists(INPUT_AUDIO):
        raise FileNotFoundError(f"找不到音频文件: {INPUT_AUDIO}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 加载音频
    print("\n[1] 加载音频...")
    audio = load_audio(INPUT_AUDIO)
    duration = len(audio) / SAMPLE_RATE
    print(f"    时长: {duration/60:.1f} 分钟 ({duration:.1f} 秒)")

    # Step 1: 说话人分离 (Diarization)
    print("\n[2] 说话人分离 (识别谁在什么时候说话)...")

    from pyannote.audio import Pipeline as PyannotePipeline

    # 使用 pyannote-audio 4.0 的最新模型
    diarization_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN
    )
    if device == "cuda":
        diarization_pipeline = diarization_pipeline.to(torch.device(device))

    # 运行 diarization，限制最大说话人数
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    print(f"    限制最大说话人数: {MAX_SPEAKERS}")
    diarization_result = diarization_pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        max_speakers=MAX_SPEAKERS
    )

    # pyannote-audio 4.0 返回 DiarizeOutput，需要通过 speaker_diarization 属性访问
    if hasattr(diarization_result, 'speaker_diarization'):
        diarization = diarization_result.speaker_diarization
    else:
        diarization = diarization_result

    # 收集每个说话人的片段
    speaker_segments = defaultdict(list)  # speaker_id -> [(start, end), ...]

    # 获取所有说话人
    speakers = list(diarization.labels())
    print(f"    检测到 {len(speakers)} 个说话人: {speakers}")

    if len(speakers) == 0:
        print("    ⚠ 未检测到任何说话人，退出")
        return

    # 分析每个时间点有几个人说话
    for segment, track, speaker in diarization.itertracks(yield_label=True):
        speaker_segments[speaker].append((segment.start, segment.end))

    # Step 2.5: 用 Segmentation 模型获取帧级置信度
    print("\n[2.5] 分析每个片段的置信度...")

    from pyannote.audio import Model as PyannoteModel
    from pyannote.audio import Inference as PyannoteInference

    # 加载 segmentation 模型获取帧级概率
    seg_model = PyannoteModel.from_pretrained(
        "pyannote/segmentation-3.0",
        token=HF_TOKEN
    )
    if device == "cuda":
        seg_model = seg_model.to(torch.device(device))

    # 使用滑动窗口模式，避免内存不足
    # window="sliding" 使用默认的10s窗口，step=默认的滑动步长
    seg_inference = PyannoteInference(seg_model, window="sliding", batch_size=8)

    # 获取帧级 segmentation 输出 (每帧每个潜在说话人的概率)
    segmentation_output = seg_inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    # segmentation_output 是 SlidingWindowFeature，形状 (num_frames, num_classes)

    seg_data = segmentation_output.data  # numpy array
    seg_frames = seg_data.shape[0]
    seg_resolution = duration / seg_frames  # 每帧多少秒

    print(f"    Segmentation 输出: {seg_data.shape} (帧数 x 说话人槽位)")
    print(f"    帧分辨率: {seg_resolution*1000:.1f}ms")

    # 分析每个 diarization 片段的置信度
    low_confidence_segments = []  # 存储置信度低的片段

    # 创建帧级的"可疑多人说话"mask，用于后续判断纯净语音
    suspected_overlap_mask = np.zeros(seg_frames, dtype=bool)

    print("\n    --- 各说话人片段置信度分析 ---")
    for speaker in speakers:
        segments = speaker_segments[speaker]
        total_duration_spk = sum(end - start for start, end in segments)

        segment_confidences = []
        for start, end in segments:
            # 找到对应的帧范围
            start_frame = int(start / seg_resolution)
            end_frame = int(end / seg_resolution)
            start_frame = max(0, min(start_frame, seg_frames - 1))
            end_frame = max(0, min(end_frame, seg_frames - 1))

            if end_frame > start_frame:
                # 获取这个时间段的概率
                segment_probs = seg_data[start_frame:end_frame]  # (frames, classes)

                # 计算该片段的置信度：
                # 1. 最大概率的平均值（主说话人有多确定）
                max_probs = np.max(segment_probs, axis=1)
                avg_max_prob = np.mean(max_probs)

                # 2. 检测是否有多人同时说话（第二高概率也很高）
                sorted_probs = np.sort(segment_probs, axis=1)[:, ::-1]  # 降序
                second_probs = sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else np.zeros(len(sorted_probs))
                avg_second_prob = np.mean(second_probs)

                # 3. 判断是否可能是多人说话（被错误标记为单人）
                # 如果第二高概率 > 0.3，说明可能有另一个人也在说话
                overlap_ratio = np.mean(second_probs > 0.3)  # 有多少帧可能是多人

                segment_confidences.append({
                    'start': start,
                    'end': end,
                    'main_confidence': avg_max_prob,
                    'second_speaker_prob': avg_second_prob,
                    'potential_overlap_ratio': overlap_ratio
                })

                # 标记低置信度片段（置信度低 或 多人说话概率高）
                if avg_max_prob < 0.7 or overlap_ratio > 0.3:
                    low_confidence_segments.append({
                        'speaker': speaker,
                        'start': start,
                        'end': end,
                        'main_confidence': avg_max_prob,
                        'overlap_ratio': overlap_ratio
                    })
                    # 标记这个时间段为可疑多人说话区域
                    suspected_overlap_mask[start_frame:end_frame] = True

        # 输出该说话人的置信度统计
        if segment_confidences:
            avg_conf = np.mean([s['main_confidence'] for s in segment_confidences])
            avg_overlap = np.mean([s['potential_overlap_ratio'] for s in segment_confidences])
            low_conf_count = sum(1 for s in segment_confidences if s['main_confidence'] < 0.7)
            high_overlap_count = sum(1 for s in segment_confidences if s['potential_overlap_ratio'] > 0.3)

            print(f"    {speaker}: 总时长 {total_duration_spk:.1f}s, {len(segments)} 片段")
            print(f"      平均置信度: {avg_conf:.2f} | 潜在多人比例: {avg_overlap*100:.1f}%")
            if low_conf_count > 0:
                print(f"      ⚠ {low_conf_count} 个片段置信度 < 0.7")
            if high_overlap_count > 0:
                print(f"      ⚠ {high_overlap_count} 个片段可能包含多人说话")

    # 输出被自动标记为混合的片段
    suspected_duration = np.sum(suspected_overlap_mask) * seg_resolution
    print(f"\n    自动标记为混合的时长: {suspected_duration:.1f}s ({suspected_duration/duration*100:.1f}%)")

    if low_confidence_segments:
        print(f"    共 {len(low_confidence_segments)} 个片段被标记 (置信度<0.7 或 多人概率>30%):")
        # 按置信度排序，显示最可疑的
        low_confidence_segments.sort(key=lambda x: x['main_confidence'])
        for i, seg in enumerate(low_confidence_segments[:10]):  # 最多显示10个
            print(f"      {i+1}. [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']} - 置信度:{seg['main_confidence']:.2f}, 多人:{seg['overlap_ratio']*100:.0f}%")
        if len(low_confidence_segments) > 10:
            print(f"      ... 还有 {len(low_confidence_segments) - 10} 个片段")
    else:
        print("    ✓ 所有片段置信度良好，无需额外标记")

    # 找出重叠片段
    print("\n[3] 分析重叠语音...")

    # 创建时间线，标记每个时间点的说话人
    timeline_resolution = 0.05  # 50ms 精度
    timeline_length = int(duration / timeline_resolution) + 1
    timeline = [set() for _ in range(timeline_length)]

    for speaker, segments in speaker_segments.items():
        for start, end in segments:
            start_idx = int(start / timeline_resolution)
            end_idx = int(end / timeline_resolution)
            for i in range(start_idx, min(end_idx + 1, timeline_length)):
                timeline[i].add(speaker)

    # 扩展重叠区域的安全边界（前后各扩展 0.3s）
    # 防止 diarization 边界误差导致把重叠音频当成纯净
    SAFETY_MARGIN = int(0.3 / timeline_resolution)  # 0.3s 安全边界

    overlap_mask = [len(t) > 1 for t in timeline]
    expanded_overlap_mask = overlap_mask.copy()

    for i in range(len(overlap_mask)):
        if overlap_mask[i]:
            # 向前扩展
            for j in range(max(0, i - SAFETY_MARGIN), i):
                expanded_overlap_mask[j] = True
            # 向后扩展
            for j in range(i + 1, min(len(overlap_mask), i + SAFETY_MARGIN + 1)):
                expanded_overlap_mask[j] = True

    # 统计重叠情况
    overlap_count = sum(1 for t in timeline if len(t) > 1)
    overlap_duration = overlap_count * timeline_resolution
    print(f"    重叠语音时长: {overlap_duration:.1f} 秒 ({overlap_duration/duration*100:.1f}%)")

    # 找出连续的重叠片段
    overlap_regions = []
    in_overlap = False
    overlap_start = 0
    overlap_speakers = set()

    for i, speakers_at_time in enumerate(timeline):
        if len(speakers_at_time) > 1:
            if not in_overlap:
                in_overlap = True
                overlap_start = i * timeline_resolution
                overlap_speakers = speakers_at_time.copy()
            else:
                overlap_speakers.update(speakers_at_time)
        else:
            if in_overlap:
                overlap_end = i * timeline_resolution
                overlap_regions.append((overlap_start, overlap_end, list(overlap_speakers)))
                in_overlap = False
                overlap_speakers = set()

    if in_overlap:
        overlap_regions.append((overlap_start, duration, list(overlap_speakers)))

    # 把低置信度片段也加入重叠区域（当作混合音频处理）
    for seg in low_confidence_segments:
        # 找出这个时间段涉及的所有说话人
        start_idx = int(seg['start'] / timeline_resolution)
        end_idx = int(seg['end'] / timeline_resolution)
        involved = set()
        for j in range(start_idx, min(end_idx + 1, timeline_length)):
            involved.update(timeline[j])
        if len(involved) == 1:
            # 单人片段但置信度低，说明可能漏检了其他人，加入所有可能的说话人
            involved = set(speakers)
        overlap_regions.append((seg['start'], seg['end'], list(involved)))

    # 合并重叠的区域
    if overlap_regions:
        overlap_regions.sort(key=lambda x: x[0])
        merged_regions = []
        current_start, current_end, current_speakers = overlap_regions[0]
        current_speakers = set(current_speakers)

        for start, end, spks in overlap_regions[1:]:
            if start <= current_end:  # 有重叠，合并
                current_end = max(current_end, end)
                current_speakers.update(spks)
            else:
                merged_regions.append((current_start, current_end, list(current_speakers)))
                current_start, current_end = start, end
                current_speakers = set(spks)
        merged_regions.append((current_start, current_end, list(current_speakers)))
        overlap_regions = merged_regions

    print(f"    重叠片段数量: {len(overlap_regions)} (含低置信度片段)")

    # Step 2: 提取声纹
    print("\n[4] 提取每个说话人的声纹...")

    try:
        from speechbrain.inference.speaker import SpeakerRecognition  # type: ignore
    except ImportError:
        from speechbrain.pretrained import SpeakerRecognition  # type: ignore

    speaker_encoder = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="speaker_separation/pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    speaker_embeddings = {}  # speaker_id -> embedding

    for speaker in speakers:
        # 收集该说话人的纯净片段（不与其他人重叠的部分）
        clean_segments = []
        for start, end in speaker_segments[speaker]:
            # 检查这个片段是否有重叠
            start_idx = int(start / timeline_resolution)
            end_idx = int(end / timeline_resolution)

            clean_start = None
            for i in range(start_idx, min(end_idx + 1, timeline_length)):
                # 使用扩展的重叠mask，更保守地判断纯净语音
                # 同时检查置信度mask，低置信度片段也不算纯净
                current_time = i * timeline_resolution
                seg_frame_idx = min(int(current_time / seg_resolution), seg_frames - 1)
                is_suspected = suspected_overlap_mask[seg_frame_idx]

                is_clean = (len(timeline[i]) == 1) and (not expanded_overlap_mask[i]) and (not is_suspected)
                if is_clean:  # 只有一个人说话、不在重叠安全区内、且置信度高
                    if clean_start is None:
                        clean_start = current_time
                else:
                    if clean_start is not None:
                        clean_end = current_time
                        if clean_end - clean_start >= 0.5:  # 至少 0.5 秒
                            clean_segments.append((clean_start, clean_end))
                        clean_start = None

            if clean_start is not None:
                clean_end = min(end_idx + 1, timeline_length) * timeline_resolution
                if clean_end - clean_start >= 0.5:
                    clean_segments.append((clean_start, clean_end))

        if clean_segments:

            clean_audio = []
            total_len = 0
            for start, end in clean_segments:
                if total_len >= 300:
                    break
                start_sample = int(start * SAMPLE_RATE)
                end_sample = int(end * SAMPLE_RATE)
                segment_audio = audio[start_sample:end_sample]
                clean_audio.append(segment_audio)
                total_len += end - start

            if clean_audio:
                combined = np.concatenate(clean_audio)
                audio_tensor = torch.from_numpy(combined).float().unsqueeze(0)
                embedding = speaker_encoder.encode_batch(audio_tensor)
                speaker_embeddings[speaker] = embedding.squeeze().cpu()
                print(f"    {speaker}: 提取了 {total_len:.1f}s 纯净语音的声纹")
        else:
            print(f"    ⚠ {speaker}: 没有找到纯净片段，将使用混合音频")

    # 检查是否所有说话人都没有声纹
    if not speaker_embeddings:
        print("\n    ⚠ 警告：所有说话人都没有足够的纯净语音提取声纹")
        print("    将跳过声纹匹配，重叠部分直接复制混合音频")

    # Step 3: 为每个说话人生成音轨
    print("\n[5] 生成每个说话人的音轨...")

    # 初始化每个说话人的音轨
    speaker_audio = {speaker: np.zeros_like(audio) for speaker in speakers}

    # 处理非重叠部分 - 直接复制原音频
    for speaker in speakers:
        total_copied = 0
        for start, end in speaker_segments[speaker]:
            # 遍历这个片段的每个时间点，找出非重叠的部分
            i = int(start / timeline_resolution)
            end_i = int(end / timeline_resolution)

            clean_start = None
            while i <= end_i and i < timeline_length:
                current_time = i * timeline_resolution

                # 使用扩展的重叠mask，更保守地判断纯净语音
                # 同时检查置信度mask，低置信度片段也不算纯净
                seg_frame_idx = min(int(current_time / seg_resolution), seg_frames - 1)
                is_suspected = suspected_overlap_mask[seg_frame_idx]

                is_clean = (len(timeline[i]) == 1) and (not expanded_overlap_mask[i]) and (not is_suspected)
                if is_clean:  # 只有这一个人说话、不在重叠安全区内、且置信度高
                    if clean_start is None:
                        clean_start = max(start, current_time)
                else:  # 有重叠、在安全区内、或置信度低
                    if clean_start is not None:
                        # 保存之前的纯净片段
                        clean_end = current_time
                        s = int(clean_start * SAMPLE_RATE)
                        e = int(clean_end * SAMPLE_RATE)
                        if e > s:
                            speaker_audio[speaker][s:e] = audio[s:e]
                            total_copied += (e - s) / SAMPLE_RATE
                        clean_start = None
                i += 1

            # 处理最后一段纯净片段
            if clean_start is not None:
                clean_end = end
                s = int(clean_start * SAMPLE_RATE)
                e = int(clean_end * SAMPLE_RATE)
                if e > s:
                    speaker_audio[speaker][s:e] = audio[s:e]
                    total_copied += (e - s) / SAMPLE_RATE

        print(f"    {speaker}: 复制了 {total_copied:.1f}s 纯净语音")

    # Step 4: 处理重叠片段
    if overlap_regions and len(speakers) > 1 and any(speaker in speaker_embeddings for speaker in speakers):
        print(f"\n[6] 处理 {len(overlap_regions)} 个重叠片段...")

        # 加载分离模型
        try:
            from speechbrain.inference.separation import SepformerSeparation  # type: ignore
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation  # type: ignore

        # 根据最大同时说话人数选择模型
        max_overlap = max(len(s) for _, _, s in overlap_regions)
        print(f"    最大同时说话人数: {max_overlap}")

        # 使用降级的 SpeechBrain 0.5.16 可以支持 4 人模型
        if max_overlap <= 2:
            sep_model = "speechbrain/sepformer-wsj02mix"
            sep_sr = 8000
        elif max_overlap <= 3:
            sep_model = "speechbrain/sepformer-libri3mix"
            sep_sr = 8000
        else:
            sep_model = "hahmadraz/sepformer-libri4mix"
            sep_sr = 48000
            if max_overlap > 4:
                print("    ⚠ 超过4人同时说话，将用声纹匹配尽可能分离")

        separator = SepformerSeparation.from_hparams(
            source=sep_model,
            savedir=f"speaker_separation/pretrained_models/{sep_model.replace('/', '_')}",
            run_opts={"device": device}
        )

        # 处理每个重叠片段
        import torchaudio.functional as F

        # 分离模型的最大处理长度（秒），超过这个长度需要分块
        MAX_SEP_DURATION = 10.0  # 10秒，避免显存溢出

        for idx, (start, end, involved_speakers) in enumerate(overlap_regions):
            start_sample = int(start * SAMPLE_RATE)
            end_sample = int(end * SAMPLE_RATE)
            overlap_audio = audio[start_sample:end_sample]
            segment_duration = end - start

            # 太短的重叠片段，直接给所有涉及的人都复制一份
            if segment_duration < 0.01:  # 小于100ms的片段太短，无法有效分离
                for speaker in involved_speakers:
                    speaker_audio[speaker][start_sample:end_sample] = overlap_audio
                print(f"    短重叠片段 {idx+1}: {start:.2f}s-{end:.2f}s -> 复制给所有 {len(involved_speakers)} 人")
                continue

            print(f"    处理重叠片段 {idx+1}/{len(overlap_regions)}: {start:.1f}s-{end:.1f}s ({len(involved_speakers)}人)")

            # 如果片段太长，需要分块处理
            if segment_duration > MAX_SEP_DURATION:
                print(f"      片段较长({segment_duration:.1f}s)，分块处理...")
                # 分块，每块最多 MAX_SEP_DURATION 秒
                chunk_samples = int(MAX_SEP_DURATION * SAMPLE_RATE)
                num_chunks = int(np.ceil(len(overlap_audio) / chunk_samples))

                # 为每个说话人初始化分离结果
                separated_results = {speaker: np.zeros_like(overlap_audio) for speaker in involved_speakers}

                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_samples
                    chunk_end = min((chunk_idx + 1) * chunk_samples, len(overlap_audio))
                    chunk_audio = overlap_audio[chunk_start:chunk_end]

                    # 重采样到分离模型需要的采样率
                    chunk_tensor = torch.from_numpy(chunk_audio).float().unsqueeze(0)
                    if sep_sr != SAMPLE_RATE:
                        chunk_tensor = F.resample(chunk_tensor, SAMPLE_RATE, sep_sr)

                    # 分离
                    with torch.no_grad():
                        separated = separator.separate_batch(chunk_tensor.to(device))

                    # 清理显存
                    del chunk_tensor
                    if device == "cuda":
                        torch.cuda.empty_cache()

                    num_sources = separated.shape[2]
                    chunk_assigned = set()

                    # 重采样回原采样率并匹配说话人
                    for src_idx in range(num_sources):
                        src_audio = separated[0, :, src_idx].cpu()
                        if sep_sr != SAMPLE_RATE:
                            src_audio = F.resample(src_audio.unsqueeze(0), sep_sr, SAMPLE_RATE).squeeze(0)
                        src_audio_np = src_audio.numpy()

                        # 确保长度匹配
                        target_len = chunk_end - chunk_start
                        if len(src_audio_np) > target_len:
                            src_audio_np = src_audio_np[:target_len]
                        elif len(src_audio_np) < target_len:
                            src_audio_np = np.pad(src_audio_np, (0, target_len - len(src_audio_np)))

                        # 声纹匹配（太短的跳过，后面统一处理）
                        MIN_AUDIO_LEN = int(0.1 * SAMPLE_RATE)  # 0.1秒
                        if len(src_audio) < MIN_AUDIO_LEN:
                            continue

                        src_embedding = speaker_encoder.encode_batch(src_audio.unsqueeze(0))
                        src_embedding = src_embedding.squeeze().cpu()

                        best_speaker = None
                        best_score = -1
                        for speaker in involved_speakers:
                            if speaker in chunk_assigned:
                                continue
                            if speaker in speaker_embeddings:
                                score = torch.nn.functional.cosine_similarity(
                                    src_embedding.unsqueeze(0),
                                    speaker_embeddings[speaker].unsqueeze(0)
                                ).item()
                                if score > best_score:
                                    best_score = score
                                    best_speaker = speaker

                        if best_speaker:
                            separated_results[best_speaker][chunk_start:chunk_end] = src_audio_np
                            chunk_assigned.add(best_speaker)
                            print(f"        块{chunk_idx+1} 声源{src_idx+1} -> {best_speaker} (相似度: {best_score:.2f})")

                    # 未分配的说话人复制混合音频
                    unassigned_chunk = set(involved_speakers) - chunk_assigned
                    if unassigned_chunk:
                        print(f"        块{chunk_idx+1} 未分配: {unassigned_chunk} -> 复制混合")
                    for speaker in unassigned_chunk:
                        separated_results[speaker][chunk_start:chunk_end] = chunk_audio

                    del separated

                # 写入最终结果
                for speaker in involved_speakers:
                    speaker_audio[speaker][start_sample:end_sample] = separated_results[speaker]
                print(f"      分块完成，共 {num_chunks} 块，已分配给 {len(involved_speakers)} 人")
                continue

            # 正常处理（短片段）
            # 重采样到分离模型需要的采样率
            overlap_tensor = torch.from_numpy(overlap_audio).float().unsqueeze(0)
            if sep_sr != SAMPLE_RATE:
                overlap_tensor = F.resample(overlap_tensor, SAMPLE_RATE, sep_sr)

            # 分离
            with torch.no_grad():  # 节省显存
                separated = separator.separate_batch(overlap_tensor.to(device))

            # 清理显存
            del overlap_tensor
            if device == "cuda":
                torch.cuda.empty_cache()

            num_sources = separated.shape[2]

            # 记录哪些说话人已被分配
            assigned_speakers = set()

            # 重采样回原采样率并匹配说话人
            for src_idx in range(num_sources):
                src_audio = separated[0, :, src_idx].cpu()

                if sep_sr != SAMPLE_RATE:
                    src_audio = F.resample(src_audio.unsqueeze(0), sep_sr, SAMPLE_RATE).squeeze(0)

                src_audio_np = src_audio.numpy()

                # 确保长度匹配
                target_len = end_sample - start_sample
                if len(src_audio_np) > target_len:
                    src_audio_np = src_audio_np[:target_len]
                elif len(src_audio_np) < target_len:
                    src_audio_np = np.pad(src_audio_np, (0, target_len - len(src_audio_np)))

                # 计算这个分离出的声音与各个说话人的相似度
                # 音频太短无法提取声纹时，记录下来最后统一处理
                MIN_AUDIO_LEN = int(0.1 * SAMPLE_RATE)  # 至少0.1秒才能做声纹匹配
                if len(src_audio) < MIN_AUDIO_LEN:
                    # 太短无法匹配，但不跳过，后面统一处理
                    continue

                src_embedding = speaker_encoder.encode_batch(src_audio.unsqueeze(0))
                src_embedding = src_embedding.squeeze().cpu()

                best_speaker = None
                best_score = -1

                # 只匹配尚未分配的说话人
                for speaker in involved_speakers:
                    if speaker in assigned_speakers:
                        continue
                    if speaker in speaker_embeddings:
                        score = torch.nn.functional.cosine_similarity(
                            src_embedding.unsqueeze(0),
                            speaker_embeddings[speaker].unsqueeze(0)
                        ).item()
                        if score > best_score:
                            best_score = score
                            best_speaker = speaker

                # 即使相似度很低也要分配，只要有最佳匹配
                if best_speaker:
                    speaker_audio[best_speaker][start_sample:end_sample] = src_audio_np
                    assigned_speakers.add(best_speaker)
                    print(f"      声源 {src_idx+1} -> {best_speaker} (相似度: {best_score:.2f})")

            del separated

            # 如果有说话人没分到声源，所有涉及的人都复制混合音频
            unassigned = set(involved_speakers) - assigned_speakers
            if unassigned:
                print(f"      ⚠ 无法完全分配，所有 {len(involved_speakers)} 人都复制混合音频")
                for speaker in involved_speakers:
                    speaker_audio[speaker][start_sample:end_sample] = overlap_audio

            # 定期清理显存
            if device == "cuda" and idx % 50 == 0:
                torch.cuda.empty_cache()

    # Step 5: 保存结果
    print("\n[7] 归一化并保存分离的音轨...")
    import soundfile as sf

    def normalize_audio(audio_data, target_rms=0.1):
        """
        RMS 归一化 - 保证整体响度一致，不受单个峰值影响
        """
        # 只计算有声音部分的 RMS
        non_silent = np.abs(audio_data) > 0.005
        if np.sum(non_silent) > 0:
            current_rms = np.sqrt(np.mean(audio_data[non_silent] ** 2))
            if current_rms > 0:
                # RMS 归一化
                audio_data = audio_data * (target_rms / current_rms)

        # 限幅防止爆音（软限幅）
        max_val = np.abs(audio_data).max()
        if max_val > 0.95:
            audio_data = audio_data * (0.95 / max_val)

        return audio_data

    for speaker in speakers:
        audio_data = speaker_audio[speaker]

        # RMS 归一化
        audio_data = normalize_audio(audio_data)

        output_file = os.path.join(OUTPUT_DIR, f"{speaker}.wav")
        sf.write(output_file, audio_data.astype(np.float32), SAMPLE_RATE)

        # 计算实际说话时长
        non_zero = np.abs(audio_data) > 0.01
        speaking_duration = np.sum(non_zero) / SAMPLE_RATE
        print(f"    保存: {output_file} (说话时长: {speaking_duration/60:.1f} 分钟)")

    print("\n" + "=" * 60)
    print("完成！")
    print(f"输出目录: {OUTPUT_DIR}/")
    print(f"共分离出 {len(speakers)} 个说话人的音轨")
    print("=" * 60)


if __name__ == "__main__":
    main()
