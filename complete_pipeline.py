#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的端到端流程：从混杂语音到中文字幕
====================================

核心思路：
1. 检测音频中是否有语音（VAD）
2. 使用 pyannote diarization 初步分析说话人和时间
3. 识别混杂片段（多人同时说话）
4. 对混杂片段使用源分离模型分离
5. 对每个说话人的音频（纯净+分离后的）进行转录
6. 合并所有转录结果，按时间排序
7. 翻译为中文
8. 生成统一的字幕文件

支持场景：
- 无人说话 → 生成空字幕
- 单人说话 → 直接转录翻译
- 多人轮流说话 → 说话人分离 + 转录翻译
- 多人同时说话 → 源分离 + 分别转录 + 翻译
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import whisperx
import soundfile as sf
import tempfile
import gc
from datetime import timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 修复 PyTorch 与 pyannote 兼容性
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=False, **pickle_load_args):
    weights_only = False
    if hasattr(f, 'read') and hasattr(f, 'seek'):
        try:
            pos = f.tell()
            header = f.read(4)
            f.seek(pos)
            if header[:2] == b'PK':
                content = f.read()
                f.seek(pos)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    result = _original_torch_load(tmp_path, map_location=map_location,
                                                  pickle_module=pickle_module,
                                                  weights_only=weights_only,
                                                  **pickle_load_args)
                    return result
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        except Exception:
            if hasattr(f, 'seek'):
                try:
                    f.seek(pos)
                except:
                    pass
    return _original_torch_load(f, map_location=map_location,
                                pickle_module=pickle_module,
                                weights_only=weights_only,
                                **pickle_load_args)

torch.load = _patched_torch_load

# 导入翻译模型
from transformers import MarianMTModel, MarianTokenizer

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SAMPLE_RATE = 16000

# 全局模型缓存
translation_model = None
translation_tokenizer = None


def format_timestamp(seconds):
    """SRT 时间格式"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def load_and_normalize_audio(file_path: str) -> np.ndarray:
    """加载音频并转换为标准格式"""
    import subprocess
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    try:
        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",  # 单声道
            "-y", temp_path
        ], check=True, capture_output=True, text=True)
        
        audio, _ = sf.read(temp_path)
        return audio.astype(np.float32)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def detect_speech_activity(audio: np.ndarray) -> bool:
    """检测是否有语音活动"""
    # 简单的能量检测
    energy = np.sqrt(np.mean(audio ** 2))
    return energy > 0.01  # 阈值


def perform_diarization(audio: np.ndarray, device: str) -> Tuple[List[Dict], List[str]]:
    """
    说话人分离 - 识别谁在什么时候说话
    返回：(说话人片段列表, 说话人列表)
    """
    print("  🎯 执行说话人分离...")
    
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        ).to(torch.device(device))
        
        # 准备输入
        waveform = torch.from_numpy(audio).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        
        # 执行分离
        diarization = pipeline(audio_input)
        
        # 提取片段
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        
        speakers = list(set(seg["speaker"] for seg in segments))
        
        print(f"  ✓ 检测到 {len(speakers)} 个说话人")
        
        # 清理
        del pipeline, audio_input, diarization
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return segments, speakers
        
    except Exception as e:
        print(f"  ⚠ 说话人分离失败: {e}")
        # 返回整段作为单一说话人
        duration = len(audio) / SAMPLE_RATE
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": duration}], ["SPEAKER_00"]


def find_overlapping_segments(segments: List[Dict], timeline_resolution: float = 0.05) -> List[Dict]:
    """
    找出有多人同时说话的片段
    返回重叠片段列表，每个包含 start, end, speakers
    """
    if len(segments) < 2:
        return []
    
    # 计算音频总时长
    duration = max(seg["end"] for seg in segments)
    timeline_length = int(duration / timeline_resolution) + 1
    timeline = [set() for _ in range(timeline_length)]
    
    # 填充时间线
    for seg in segments:
        start_idx = int(seg["start"] / timeline_resolution)
        end_idx = int(seg["end"] / timeline_resolution)
        speaker = seg["speaker"]
        for i in range(start_idx, min(end_idx + 1, timeline_length)):
            timeline[i].add(speaker)
    
    # 找出重叠区域
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
                overlap_regions.append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "speakers": list(overlap_speakers)
                })
                in_overlap = False
                overlap_speakers = set()
    
    if in_overlap:
        overlap_regions.append({
            "start": overlap_start,
            "end": duration,
            "speakers": list(overlap_speakers)
        })
    
    return overlap_regions


def separate_mixed_audio(audio: np.ndarray, overlap_region: Dict, device: str) -> Dict[str, np.ndarray]:
    """
    使用源分离模型分离混杂音频
    返回：{speaker: separated_audio}
    """
    start = overlap_region["start"]
    end = overlap_region["end"]
    speakers = overlap_region["speakers"]
    num_speakers = len(speakers)
    
    # 提取混杂片段
    start_sample = int(start * SAMPLE_RATE)
    end_sample = int(end * SAMPLE_RATE)
    mixed_audio = audio[start_sample:end_sample]
    
    # 如果片段太短，直接返回所有人共享
    if len(mixed_audio) < 0.1 * SAMPLE_RATE:
        return {speaker: mixed_audio for speaker in speakers}
    
    try:
        # 根据说话人数选择分离模型
        if num_speakers <= 2:
            from speechbrain.inference.separation import SepformerSeparation
            separator = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir="pretrained_models/sepformer-wsj02mix",
                run_opts={"device": device}
            )
            sep_sr = 8000
        elif num_speakers <= 3:
            from speechbrain.inference.separation import SepformerSeparation
            separator = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-libri3mix",
                savedir="pretrained_models/sepformer-libri3mix",
                run_opts={"device": device}
            )
            sep_sr = 8000
        else:
            # 4人或更多，使用4人模型
            from speechbrain.inference.separation import SepformerSeparation
            separator = SepformerSeparation.from_hparams(
                source="hahmadraz/sepformer-libri4mix",
                savedir="pretrained_models/sepformer-libri4mix",
                run_opts={"device": device}
            )
            sep_sr = 48000
        
        # 重采样
        import torchaudio.functional as F
        audio_tensor = torch.from_numpy(mixed_audio).float().unsqueeze(0)
        if sep_sr != SAMPLE_RATE:
            audio_tensor = F.resample(audio_tensor, SAMPLE_RATE, sep_sr)
        
        # 分离
        with torch.no_grad():
            separated = separator.separate_batch(audio_tensor.to(device))
        
        # 重采样回原始采样率
        num_sources = separated.shape[2]
        separated_audios = []
        for i in range(num_sources):
            src = separated[0, :, i].cpu()
            if sep_sr != SAMPLE_RATE:
                src = F.resample(src.unsqueeze(0), sep_sr, SAMPLE_RATE).squeeze(0)
            separated_audios.append(src.numpy())
        
        # 分配给说话人（简单策略：按顺序）
        result = {}
        for i, speaker in enumerate(speakers):
            if i < len(separated_audios):
                result[speaker] = separated_audios[i]
            else:
                result[speaker] = mixed_audio  # 如果分离的源不够，用混合音频
        
        # 清理
        del separator, audio_tensor, separated
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"    ⚠ 分离失败: {e}，使用混合音频")
        return {speaker: mixed_audio for speaker in speakers}


def create_speaker_audio_tracks(audio: np.ndarray, 
                                diarization_segments: List[Dict],
                                overlap_regions: List[Dict],
                                speakers: List[str],
                                device: str) -> Dict[str, np.ndarray]:
    """
    为每个说话人创建完整的音频轨道
    """
    print("  🎵 创建说话人音频轨道...")
    
    # 初始化每个说话人的音轨
    speaker_tracks = {speaker: np.zeros_like(audio) for speaker in speakers}
    
    # 创建时间线标记哪些是重叠区域
    timeline_resolution = 0.01
    timeline_length = int(len(audio) / SAMPLE_RATE / timeline_resolution) + 1
    is_overlap = [False] * timeline_length
    
    for overlap in overlap_regions:
        start_idx = int(overlap["start"] / timeline_resolution)
        end_idx = int(overlap["end"] / timeline_resolution)
        for i in range(start_idx, min(end_idx + 1, timeline_length)):
            is_overlap[i] = True
    
    # 处理非重叠片段 - 直接复制
    for seg in diarization_segments:
        speaker = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        start_sample = int(start * SAMPLE_RATE)
        end_sample = int(end * SAMPLE_RATE)
        
        # 检查这段是否完全不重叠
        start_idx = int(start / timeline_resolution)
        end_idx = int(end / timeline_resolution)
        has_overlap = any(is_overlap[i] for i in range(start_idx, min(end_idx + 1, timeline_length)))
        
        if not has_overlap:
            # 纯净片段，直接复制
            speaker_tracks[speaker][start_sample:end_sample] = audio[start_sample:end_sample]
    
    # 处理重叠片段 - 使用分离
    for i, overlap in enumerate(overlap_regions):
        print(f"    处理重叠片段 {i+1}/{len(overlap_regions)}: {overlap['start']:.1f}s-{overlap['end']:.1f}s")
        
        separated = separate_mixed_audio(audio, overlap, device)
        
        start_sample = int(overlap["start"] * SAMPLE_RATE)
        for speaker, sep_audio in separated.items():
            end_sample = start_sample + len(sep_audio)
            if end_sample <= len(speaker_tracks[speaker]):
                speaker_tracks[speaker][start_sample:end_sample] = sep_audio
    
    print(f"  ✓ 完成 {len(speakers)} 个说话人的音轨")
    return speaker_tracks


def transcribe_speaker_audio(audio: np.ndarray, speaker: str, device: str, compute_type: str) -> List[Dict]:
    """
    转录单个说话人的音频
    """
    # 保存为临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    try:
        sf.write(temp_path, audio, SAMPLE_RATE)
        
        # 使用 WhisperX 转录
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        audio_data = whisperx.load_audio(temp_path)
        
        result = model.transcribe(audio_data, batch_size=16, language="ja")
        
        # 添加说话人标签
        for seg in result["segments"]:
            seg["speaker"] = speaker
        
        # 清理
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return result["segments"]
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def load_translation_model():
    """加载翻译模型"""
    global translation_model, translation_tokenizer
    
    if translation_model is None:
        print("  📚 加载翻译模型...")
        model_name = "Helsinki-NLP/opus-mt-ja-zh"
        translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
        translation_model = MarianMTModel.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            translation_model = translation_model.to("cuda")
        
        print("  ✓ 翻译模型加载完成")


def translate_text(text: str) -> str:
    """翻译文本"""
    global translation_model, translation_tokenizer
    
    if not text or not text.strip():
        return text
    
    try:
        if translation_model is None:
            load_translation_model()
        
        inputs = translation_tokenizer(text, return_tensors="pt", padding=True, 
                                       truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            translated = translation_model.generate(**inputs, max_length=512)
        
        translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text.strip()
        
    except Exception as e:
        print(f"  ⚠ 翻译出错: {e}")
        return text


def generate_srt(segments: List[Dict], output_file: str):
    """生成 SRT 字幕"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            if not seg.get('text', '').strip():
                continue
            
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            
            speaker = seg.get('speaker', 'SPEAKER_00')
            zh_text = seg.get('zh_translation', seg['text'])
            ja_text = seg['text']
            
            f.write(f"[{speaker}] {zh_text}\n")
            f.write(f"({ja_text})\n")
            f.write("\n")


def complete_pipeline(input_audio: str,
                     output_srt: str = "output.srt",
                     output_json: str = "output.json"):
    """
    完整的端到端处理流程
    """
    print("=" * 70)
    print("完整的语音到中文字幕流程")
    print("=" * 70)
    print()
    
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"音频文件不存在: {input_audio}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"设备: {device}")
    print(f"计算类型: {compute_type}")
    print()
    
    # ========== 步骤 1: 加载音频 ==========
    print("[1/7] 📂 加载音频...")
    audio = load_and_normalize_audio(input_audio)
    duration = len(audio) / SAMPLE_RATE
    print(f"  时长: {duration/60:.1f} 分钟")
    print()
    
    # ========== 步骤 2: 检测语音活动 ==========
    print("[2/7] 🔍 检测语音活动...")
    has_speech = detect_speech_activity(audio)
    
    if not has_speech:
        print("  ⚠ 未检测到语音，生成空字幕")
        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write("")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)
        return
    
    print("  ✓ 检测到语音活动")
    print()
    
    # ========== 步骤 3: 说话人分离 ==========
    print("[3/7] 👥 说话人分离...")
    diarization_segments, speakers = perform_diarization(audio, device)
    print()
    
    # ========== 步骤 4: 识别混杂片段 ==========
    print("[4/7] 🔎 识别混杂语音片段...")
    overlap_regions = find_overlapping_segments(diarization_segments)
    
    if overlap_regions:
        overlap_duration = sum(r["end"] - r["start"] for r in overlap_regions)
        print(f"  发现 {len(overlap_regions)} 个混杂片段，总时长 {overlap_duration:.1f}秒")
    else:
        print("  ✓ 无混杂片段，所有人都是轮流说话")
    print()
    
    # ========== 步骤 5: 创建说话人音轨（包含分离） ==========
    print("[5/7] 🎼 创建分离的音频轨道...")
    speaker_tracks = create_speaker_audio_tracks(
        audio, diarization_segments, overlap_regions, speakers, device
    )
    print()
    
    # ========== 步骤 6: 转录每个说话人 ==========
    print("[6/7] 🎙️  转录各说话人音频...")
    all_segments = []
    
    for speaker in speakers:
        print(f"  转录 {speaker}...")
        segments = transcribe_speaker_audio(
            speaker_tracks[speaker], speaker, device, compute_type
        )
        all_segments.extend(segments)
        print(f"    ✓ {len(segments)} 个片段")
    
    # 按时间排序
    all_segments.sort(key=lambda x: x["start"])
    print(f"  ✓ 总共 {len(all_segments)} 个片段")
    print()
    
    # ========== 步骤 7: 翻译 ==========
    print("[7/7] 🌐 翻译为中文...")
    load_translation_model()
    
    for i, seg in enumerate(all_segments, 1):
        if i % 10 == 0 or i == 1:
            print(f"  进度: {i}/{len(all_segments)}")
        
        ja_text = seg["text"]
        zh_text = translate_text(ja_text)
        seg["zh_translation"] = zh_text
    
    print("  ✓ 翻译完成")
    print()
    
    # ========== 保存结果 ==========
    print("💾 保存结果...")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON: {output_json}")
    
    generate_srt(all_segments, output_srt)
    print(f"  ✓ SRT: {output_srt}")
    print()
    
    # ========== 摘要 ==========
    print("=" * 70)
    print("✅ 处理完成！")
    print("=" * 70)
    print(f"说话人数: {len(speakers)}")
    print(f"混杂片段: {len(overlap_regions)}")
    print(f"字幕片段: {len(all_segments)}")
    print(f"输出文件:")
    print(f"  📄 {output_srt}")
    print(f"  📊 {output_json}")
    print()
    
    # 显示预览
    if all_segments:
        print("前 3 条字幕:")
        print("-" * 70)
        for i, seg in enumerate(all_segments[:3], 1):
            print(f"{i}. [{seg['speaker']}] {format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
            print(f"   原文: {seg['text']}")
            print(f"   中文: {seg['zh_translation']}")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="完整的语音到中文字幕流程")
    parser.add_argument("--input", "-i", default="input.mp3", help="输入音频文件")
    parser.add_argument("--output-srt", "-s", default="output.srt", help="输出 SRT 文件")
    parser.add_argument("--output-json", "-j", default="output.json", help="输出 JSON 文件")
    
    args = parser.parse_args()
    
    try:
        complete_pipeline(args.input, args.output_srt, args.output_json)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
