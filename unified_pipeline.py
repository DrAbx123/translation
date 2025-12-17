#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的语音到中文翻译完整流程
支持多种场景：
1. 有人说话 / 没有人说话
2. 单人说话 / 多人说话（自动检测）
3. 混杂语音 / 纯净语音
4. 使用大模型和专门模型处理各种情况
"""

import os
import json
import torch
import whisperx
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple
import gc
import warnings
warnings.filterwarnings('ignore')

# 修复 PyTorch 2.8 与 pyannote-audio 的兼容性
import tempfile
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=False, **pickle_load_args):
    """修复 pyannote 模型加载问题"""
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
                    except (OSError, IOError):
                        pass
        except Exception:
            if hasattr(f, 'seek'):
                try:
                    f.seek(pos)
                except (IOError, OSError):
                    pass
    return _original_torch_load(f, map_location=map_location,
                                pickle_module=pickle_module,
                                weights_only=weights_only,
                                **pickle_load_args)

torch.load = _patched_torch_load

# 导入翻译相关
from transformers import MarianMTModel, MarianTokenizer, pipeline
from huggingface_hub import InferenceClient

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
USE_QWEN = os.environ.get("USE_QWEN", "false").lower() == "true"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 使用稳定版本，支持推理API，性能和准确性平衡较好

# 全局翻译模型
translation_model = None
translation_tokenizer = None
qwen_client = None


class AudioAnalysisResult:
    """音频分析结果"""
    def __init__(self):
        self.has_speech = False
        self.num_speakers = 0
        self.speaker_segments = []
        self.silence_segments = []
        self.is_mixed_audio = False  # 是否为混杂音频
        self.audio_type = "unknown"  # silence, single_speaker, multi_speaker


def format_timestamp(seconds):
    """将秒数转换为 SRT 时间格式"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def detect_voice_activity(audio_file: str, device: str = "cpu") -> Tuple[bool, List[Dict]]:
    """
    使用 VAD (Voice Activity Detection) 检测音频中是否有语音
    返回: (是否有语音, 语音片段列表)
    """
    print("  🔍 检测语音活动...")
    
    try:
        # 使用 WhisperX 自带的 Silero VAD
        import torch
        from whisperx.vad import load_vad_model, merge_chunks
        
        vad_model = load_vad_model(device)
        audio = whisperx.load_audio(audio_file)
        
        # 获取 VAD 片段
        vad_segments = vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size=30,
            onset=0.5,
            offset=0.363,
        )
        
        has_speech = len(vad_segments) > 0
        
        if has_speech:
            print(f"  ✓ 检测到语音，共 {len(vad_segments)} 个活跃片段")
        else:
            print(f"  ⚠ 未检测到语音活动")
        
        return has_speech, vad_segments
        
    except Exception as e:
        print(f"  ⚠ VAD 检测失败: {e}，假设有语音")
        return True, []


def transcribe_audio(audio_file: str, device: str, compute_type: str, batch_size: int = 16) -> Dict:
    """
    使用 WhisperX 进行语音识别
    自动处理任意语言（主要支持日语到中文）
    """
    print("  🎙️  加载 Whisper large-v3 模型...")
    
    model = whisperx.load_model(
        "large-v3",
        device,
        compute_type=compute_type,
        vad_method="silero"  # 使用 Silero VAD
    )
    
    audio = whisperx.load_audio(audio_file)
    
    # 首先尝试自动检测语言
    print("  🌐 检测音频语言...")
    result = model.transcribe(audio, batch_size=batch_size)
    detected_lang = result.get("language", "ja")
    print(f"  ✓ 检测到语言: {detected_lang}")
    
    # 如果不是日语，重新转录指定语言
    if detected_lang != "ja":
        print(f"  🔄 使用检测到的语言重新转录...")
        result = model.transcribe(audio, batch_size=batch_size, language=detected_lang)
    else:
        result = model.transcribe(audio, batch_size=batch_size, language="ja")
    
    print(f"  ✓ 转录完成，识别到 {len(result['segments'])} 个片段")
    
    # 清理资源
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result, audio, detected_lang


def align_timestamps(segments: List[Dict], audio, language_code: str, device: str) -> Dict:
    """对齐时间戳以提高精度"""
    print("  ⏱️  对齐时间戳...")
    
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )
    
    print("  ✓ 时间戳对齐完成")
    
    # 清理资源
    del model_a
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result


def perform_speaker_diarization(audio_file: str, audio, device: str, 
                                min_speakers: Optional[int] = None,
                                max_speakers: Optional[int] = None) -> Tuple[pd.DataFrame, int]:
    """
    执行说话人分离
    支持自动检测说话人数量或指定范围
    """
    print("  👥 进行说话人分离...")
    
    if min_speakers or max_speakers:
        range_info = f"（范围: {min_speakers or '自动'}-{max_speakers or '自动'}）"
    else:
        range_info = "（自动检测）"
    print(f"      {range_info}")
    
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        
        diarize_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        ).to(torch.device(device))
        
        # 准备音频输入
        waveform = torch.from_numpy(audio).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": 16000}
        
        # 执行分离
        kwargs = {}
        if min_speakers is not None:
            kwargs['min_speakers'] = min_speakers
        if max_speakers is not None:
            kwargs['max_speakers'] = max_speakers
        
        diarization = diarize_pipeline(audio_input, **kwargs)
        
        # 提取说话人片段
        diarize_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarize_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        # 统计说话人数量
        speakers = set(seg["speaker"] for seg in diarize_segments)
        num_speakers = len(speakers)
        
        print(f"  ✓ 检测到 {num_speakers} 个说话人: {', '.join(sorted(speakers))}")
        
        # 转换为 DataFrame
        diarize_df = pd.DataFrame(diarize_segments)
        
        # 清理资源
        del diarize_pipeline, audio_input, diarization
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return diarize_df, num_speakers
        
    except Exception as e:
        print(f"  ⚠ 说话人分离失败: {e}")
        print(f"  继续处理，将所有片段标记为单一说话人")
        # 返回空的 DataFrame 和 1 个说话人
        return pd.DataFrame(), 1


def load_translation_model():
    """加载本地翻译模型"""
    global translation_model, translation_tokenizer
    
    if translation_model is None:
        print("  📚 加载翻译模型 (Helsinki-NLP/opus-mt-ja-zh)...")
        model_name = "Helsinki-NLP/opus-mt-ja-zh"
        translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
        translation_model = MarianMTModel.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            translation_model = translation_model.to("cuda")
        
        print("  ✓ 翻译模型加载完成")


def translate_with_local_model(text: str) -> str:
    """使用本地模型翻译"""
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


def translate_with_qwen(text: str, context: str = "") -> str:
    """使用 Qwen 模型翻译"""
    global qwen_client
    
    if not text or not text.strip():
        return text
    
    try:
        if qwen_client is None:
            qwen_client = InferenceClient(api_key=HF_TOKEN)
        
        system_prompt = "你是专业的日中翻译专家。请直接翻译日语到中文，保持自然流畅的口语化表达。只输出中文翻译，不要添加解释。"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请翻译：{text}"}
        ]
        
        response = qwen_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        
        translated = response.choices[0].message.content.strip()
        
        # 清理可能的引号
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1]
        if translated.startswith('「') and translated.endswith('」'):
            translated = translated[1:-1]
        
        return translated
        
    except Exception as e:
        print(f"  ⚠ Qwen 翻译出错: {e}，回退到本地模型")
        return translate_with_local_model(text)


def translate_text(text: str, use_qwen: bool = False, context: str = "") -> str:
    """统一的翻译接口"""
    if use_qwen:
        return translate_with_qwen(text, context)
    else:
        return translate_with_local_model(text)


def analyze_audio_type(segments: List[Dict], num_speakers: int) -> str:
    """
    分析音频类型
    返回: silence, single_speaker, multi_speaker
    """
    if not segments or len(segments) == 0:
        return "silence"
    elif num_speakers <= 1:
        return "single_speaker"
    else:
        # 检查是否为混杂音频（多人在同一时间段说话）
        # 首先按时间排序
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        # 检查是否有重叠的说话片段
        for i in range(len(sorted_segments) - 1):
            if sorted_segments[i]['end'] > sorted_segments[i + 1]['start']:
                return "mixed_multi_speaker"
        return "multi_speaker"


def merge_short_segments(segments: List[Dict], min_duration: float = 1.0, max_gap: float = 2.0) -> List[Dict]:
    """
    合并过短的片段
    
    Args:
        segments: 片段列表
        min_duration: 最小片段时长（秒）
        max_gap: 同一说话人片段之间的最大间隔（秒），超过此间隔不合并
    """
    if not segments:
        return segments
    
    merged = []
    current = None
    
    for seg in segments:
        duration = seg['end'] - seg['start']
        
        if current is None:
            current = seg.copy()
            current['texts'] = [seg['text']]
        elif (duration < min_duration and 
              current.get('speaker') == seg.get('speaker') and
              seg['start'] - current['end'] <= max_gap):  # 检查时间间隔
            # 合并到当前片段
            current['end'] = seg['end']
            current['texts'].append(seg['text'])
            current['text'] = ' '.join(current['texts'])
        else:
            # 完成当前片段
            if 'texts' in current:
                del current['texts']
            merged.append(current)
            current = seg.copy()
            current['texts'] = [seg['text']]
    
    if current:
        if 'texts' in current:
            del current['texts']
        merged.append(current)
    
    return merged


def generate_srt(segments: List[Dict], output_file: str):
    """生成 SRT 字幕文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        srt_index = 1
        for seg in segments:
            # 跳过空片段
            if not seg.get('text', '').strip():
                continue
            
            f.write(f"{srt_index}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            
            speaker = seg.get('speaker', 'SPEAKER_00')
            zh_text = seg.get('zh_translation', seg['text'])
            ja_text = seg['text']
            
            f.write(f"[{speaker}] {zh_text}\n")
            f.write(f"({ja_text})\n")
            f.write("\n")
            
            srt_index += 1
    
    print(f"✓ 字幕文件已生成: {output_file}")


def process_audio(audio_file: str,
                 output_srt: str = "output.srt",
                 output_json: str = "output.json",
                 use_qwen: bool = False,
                 force_num_speakers: Optional[int] = None):
    """
    完整的音频处理流程
    
    参数:
        audio_file: 输入音频文件路径
        output_srt: 输出 SRT 字幕文件路径
        output_json: 输出 JSON 数据文件路径
        use_qwen: 是否使用 Qwen 模型翻译（否则使用本地模型）
        force_num_speakers: 强制指定说话人数量（None 则自动检测）
    """
    print("=" * 70)
    print("统一语音到中文翻译流程")
    print("=" * 70)
    
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 16 if device == "cuda" else 4
    
    print(f"✓ 设备: {device}")
    print(f"✓ 计算类型: {compute_type}")
    print(f"✓ 翻译模式: {'Qwen 大模型' if use_qwen else '本地 Helsinki-NLP'}")
    print()
    
    # ========== 步骤 1: 检测语音活动 ==========
    print("[1/6] 检测语音活动")
    has_speech, vad_segments = detect_voice_activity(audio_file, device)
    print()
    
    if not has_speech:
        print("⚠ 音频中没有检测到语音，生成空字幕")
        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write("")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("✓ 处理完成（空结果）")
        return
    
    # ========== 步骤 2: 语音识别 ==========
    print("[2/6] 语音识别")
    result, audio, detected_lang = transcribe_audio(audio_file, device, compute_type, batch_size)
    print()
    
    if not result['segments']:
        print("⚠ 未识别到任何内容，生成空字幕")
        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write("")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("✓ 处理完成（空结果）")
        return
    
    # ========== 步骤 3: 时间戳对齐 ==========
    print("[3/6] 时间戳对齐")
    result = align_timestamps(result["segments"], audio, detected_lang, device)
    print()
    
    # ========== 步骤 4: 说话人分离 ==========
    print("[4/6] 说话人分离")
    
    min_speakers = force_num_speakers if force_num_speakers else None
    max_speakers = force_num_speakers if force_num_speakers else None
    
    diarize_df, num_speakers = perform_speaker_diarization(
        audio_file, audio, device,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    # 分配说话人标签
    if not diarize_df.empty:
        result = whisperx.assign_word_speakers(diarize_df, result)
    else:
        # 如果分离失败，给所有片段添加默认说话人
        for seg in result["segments"]:
            if "speaker" not in seg:
                seg["speaker"] = "SPEAKER_00"
    
    # 分析音频类型
    audio_type = analyze_audio_type(result["segments"], num_speakers)
    print(f"  📊 音频类型: {audio_type}")
    print()
    
    # ========== 步骤 5: 优化片段 ==========
    print("[5/6] 优化片段")
    result["segments"] = merge_short_segments(result["segments"], min_duration=1.0)
    print(f"  ✓ 优化后共 {len(result['segments'])} 个片段")
    print()
    
    # ========== 步骤 6: 翻译 ==========
    print("[6/6] 翻译为中文")
    
    if use_qwen:
        print("  使用 Qwen 模型翻译...")
    else:
        load_translation_model()
        print("  使用本地模型翻译...")
    
    total_segments = len(result["segments"])
    for i, seg in enumerate(result["segments"], 1):
        if i % 10 == 0 or i == 1:
            print(f"  进度: {i}/{total_segments} ({i*100//total_segments}%)")
        
        ja_text = seg["text"]
        zh_text = translate_text(ja_text, use_qwen=use_qwen)
        seg["zh_translation"] = zh_text
    
    print("  ✓ 翻译完成")
    print()
    
    # ========== 保存结果 ==========
    print("💾 保存结果...")
    
    # 保存 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    print(f"✓ JSON 文件: {output_json}")
    
    # 生成 SRT
    generate_srt(result["segments"], output_srt)
    
    # ========== 显示摘要 ==========
    print()
    print("=" * 70)
    print("✅ 处理完成！")
    print("=" * 70)
    print(f"音频类型: {audio_type}")
    print(f"说话人数: {num_speakers}")
    print(f"字幕片段: {len(result['segments'])}")
    print(f"输出文件:")
    print(f"  📄 字幕文件: {output_srt}")
    print(f"  📊 JSON 数据: {output_json}")
    print()
    
    # 显示示例
    if result["segments"]:
        print("前 3 条字幕预览:")
        print("-" * 70)
        for i, seg in enumerate(result["segments"][:3], 1):
            speaker = seg.get('speaker', 'UNKNOWN')
            print(f"\n{i}. [{speaker}] {format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
            print(f"   原文: {seg['text']}")
            print(f"   中文: {seg['zh_translation']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="完整的语音到中文翻译流程，支持多种场景"
    )
    parser.add_argument(
        "--input", "-i",
        default="input.mp3",
        help="输入音频文件路径"
    )
    parser.add_argument(
        "--output-srt", "-s",
        default="output.srt",
        help="输出 SRT 字幕文件路径"
    )
    parser.add_argument(
        "--output-json", "-j",
        default="output.json",
        help="输出 JSON 数据文件路径"
    )
    parser.add_argument(
        "--use-qwen", "-q",
        action="store_true",
        help="使用 Qwen 大模型翻译（需要 HF_TOKEN）"
    )
    parser.add_argument(
        "--num-speakers", "-n",
        type=int,
        default=None,
        help="强制指定说话人数量（默认自动检测）"
    )
    
    args = parser.parse_args()
    
    try:
        process_audio(
            audio_file=args.input,
            output_srt=args.output_srt,
            output_json=args.output_json,
            use_qwen=args.use_qwen,
            force_num_speakers=args.num_speakers
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
