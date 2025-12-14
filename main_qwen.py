#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日语音频处理完整版：使用最佳模型组合
- Whisper large-v3: 最高精度语音识别
- Pyannote speaker-diarization-3.1: 最新说话人分离
- Qwen3-32B: 千问翻译
"""

# ============================================================
# 修复 PyTorch 2.8 与 pyannote-audio 的兼容性问题
# 问题根源：lightning_fabric 使用 fsspec 打开文件，传入 torch.load 时
#          文件对象没有 name 属性，导致 PyTorch 无法正确检测 ZIP 格式
# 解决方案：在 torch.load 调用前确保文件对象有正确的属性
# ============================================================
import torch
import zipfile
import io
import os
import tempfile

_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=False, **pickle_load_args):
    """修复 pyannote 模型加载问题"""
    # 强制 weights_only=False
    weights_only = False

    # 如果是文件对象，检查是否是 ZIP 格式并正确处理
    if hasattr(f, 'read') and hasattr(f, 'seek'):
        try:
            pos = f.tell()
            header = f.read(4)
            f.seek(pos)

            # 检查是否是 ZIP 格式 (PK\x03\x04)
            if header[:2] == b'PK':
                # 读取全部内容到内存
                content = f.read()
                f.seek(pos)

                # 使用临时文件方式加载（更可靠）
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
        except Exception as e:
            # 出错时重置位置并尝试原始方式
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
# ============================================================

import json
import shutil
import whisperx
import pandas as pd
from datetime import timedelta
from huggingface_hub import InferenceClient
import gc
from typing import List, Dict, Any
import re
import soundfile as sf

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INPUT_AUDIO = "input.mp3"
OUTPUT_SRT = "output_qwen.srt"
OUTPUT_JSON = "output_qwen.json"

# 检查点文件配置
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_STEP1 = os.path.join(CHECKPOINT_DIR, "step1_transcribe.json")
CHECKPOINT_STEP2 = os.path.join(CHECKPOINT_DIR, "step2_align.json")
CHECKPOINT_STEP3 = os.path.join(CHECKPOINT_DIR, "step3_diarize.json")
CHECKPOINT_STEP4 = os.path.join(CHECKPOINT_DIR, "step4_personas.json")
CHECKPOINT_STEP5 = os.path.join(CHECKPOINT_DIR, "step5_merged.json")
CHECKPOINT_STEP6 = os.path.join(CHECKPOINT_DIR, "step6_translated.json")

# 千问模型配置 - 使用 Qwen3 30B MoE (HuggingFace Inference Providers)
QWEN_MODEL = "Qwen/Qwen3-30B-A3B"  # Qwen3 MoE: 30B总参数，3B激活，支持推理模式


def ensure_checkpoint_dir():
    """确保检查点目录存在"""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"✓ 创建检查点目录: {CHECKPOINT_DIR}")


def save_checkpoint(filepath: str, data: Any, step_name: str):
    """保存检查点"""
    ensure_checkpoint_dir()
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 检查点已保存: {filepath}")


def load_checkpoint(filepath: str, step_name: str) -> Any:
    """加载检查点，如果存在的话"""
    if os.path.exists(filepath):
        print(f"  📂 发现已有检查点: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  ⏭️  跳过 {step_name}，使用缓存结果")
        return data
    return None


def clear_checkpoints():
    """清除所有检查点（用于强制重新运行）"""
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"✓ 已清除所有检查点")


def format_timestamp(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"




def clean_repetitive_text(text: str, min_repeat: int = 3) -> str:
    """
    清理 Whisper hallucination 产生的重复文本
    例如: "いやいやいやいや..." -> "いや"
    """
    if not text:
        return text

    # 检测并清理重复模式 (1-10个字符的重复)
    for pattern_len in range(1, 11):
        # 匹配连续重复 min_repeat 次以上的模式
        pattern = r'(.{' + str(pattern_len) + r'})\1{' + str(min_repeat - 1) + r',}'
        match = re.search(pattern, text)
        if match:
            repeated_unit = match.group(1)
            # 只保留一次重复单元（或根据上下文保留合理次数）
            text = re.sub(pattern, repeated_unit, text)

    # 清理日语常见的重复语气词
    common_patterns = [
        (r'(いや){3,}', 'いや'),
        (r'(そう){3,}', 'そう'),
        (r'(ね){3,}', 'ね'),
        (r'(よ){3,}', 'よ'),
        (r'(な){4,}', 'な'),
        (r'(けい){3,}', 'けい'),
        (r'(初めて){3,}', '初めて'),
        (r'(もらって){3,}', 'もらって'),
        (r'(そうなのか){3,}', 'そうなのか'),
        (r'(あった){3,}', 'あった'),
    ]

    for pattern, replacement in common_patterns:
        text = re.sub(pattern, replacement, text)

    return text.strip()


def call_qwen(messages: List[Dict[str, str]],
              model: str = None,
              max_tokens: int = 800,
              temperature: float = 0.2) -> str:
    """调用千问模型进行翻译"""
    if model is None:
        model = QWEN_MODEL

    try:
        client = InferenceClient(api_key=HF_TOKEN)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠ 调用千问失败: {e}")
        return None


def build_speaker_persona(samples_ja: str) -> Dict[str, Any]:
    """为说话人构建人设风格"""
    prompt = [
        {"role": "system", "content": "你是资深日中本地化译者。输出必须是严格 JSON 格式。"},
        {"role": "user", "content": f"""分析下面这位说话人的日语台词风格：

{samples_ja}

请推断说话人的风格特征，用于保持翻译一致性。

输出 JSON 格式，包含字段：
- tone: 语气特征（简短描述）
- politeness: 礼貌程度（简短描述）
- preferred_pronouns: 常用人称代词
- style_notes: 风格注意事项（数组）

只输出 JSON，不要其他内容。
"""}
    ]

    try:
        out = call_qwen(prompt, model=QWEN_MODEL, max_tokens=500, temperature=0.2)
        if out:
            # 尝试提取 JSON
            json_match = re.search(r'\{.*\}', out, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
    except Exception as e:
        print(f"⚠ 构建人设失败: {e}")

    return {"tone": "中性", "politeness": "普通", "preferred_pronouns": "", "style_notes": []}


def translate_text(ja_text: str, persona: Dict[str, Any] = None) -> str:
    """使用千问翻译日语文本"""
    if not ja_text or not ja_text.strip():
        return ja_text

    # 构建提示词
    if persona and persona.get("tone"):
        persona_info = f"""
角色风格：
- 语气：{persona.get('tone', '自然')}
- 礼貌度：{persona.get('politeness', '普通')}
- 人称：{persona.get('preferred_pronouns', '根据上下文')}
"""
        style_notes = persona.get('style_notes', [])
        if style_notes:
            persona_info += "- 风格注意：" + "、".join(style_notes[:3])
    else:
        persona_info = ""

    prompt = [
        {"role": "system", "content": "你是专业的日译中翻译，擅长自然流畅的口语化翻译。"},
        {"role": "user", "content": f"""请将下面的日语翻译成中文。

要求：
1. 保持口语化和自然流畅
2. 符合中文表达习惯
3. 不要添加解释或注释
{persona_info}

日语原文：
{ja_text}

只输出中文翻译：
"""}
    ]

    translated = call_qwen(prompt, max_tokens=800, temperature=0.2)

    if translated:
        # 清理可能的多余内容
        translated = translated.strip()
        # 移除可能的引号
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1]
        if translated.startswith('「') and translated.endswith('」'):
            translated = translated[1:-1]
        return translated
    else:
        return ja_text  # 翻译失败返回原文


def generate_srt(segments, output_file):
    """生成 SRT 字幕文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")

            speaker = seg.get('speaker', 'SPEAKER_00')
            zh_text = seg.get('zh_translation', seg['text'])
            ja_text = seg['text']

            # 双语字幕格式
            f.write(f"[{speaker}] {zh_text}\n")
            f.write(f"({ja_text})\n")
            f.write("\n")

    print(f"✓ 字幕文件已生成: {output_file}")


def merge_short_segments(segments: List[Dict], min_duration: float = 1.5) -> List[Dict]:
    """合并过短的片段，避免字幕闪烁"""
    if not segments:
        return segments

    merged = []
    current = None

    for seg in segments:
        duration = seg['end'] - seg['start']

        if current is None:
            current = seg.copy()
            current['texts'] = [seg['text']]
        elif duration < min_duration and current['speaker'] == seg.get('speaker', 'UNKNOWN'):
            # 合并到当前片段
            current['end'] = seg['end']
            current['texts'].append(seg['text'])
            current['text'] = ' '.join(current['texts'])
            if 'words' in seg:
                current.setdefault('words', []).extend(seg['words'])
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


def process_audio(audio_file, force_restart=False):
    """完整处理流程

    Args:
        audio_file: 音频文件路径
        force_restart: 是否强制重新开始（忽略检查点）
    """
    print("=" * 70)
    print("日语音频处理 - 使用最佳模型组合")
    print("=" * 70)

    if force_restart:
        clear_checkpoints()
        print()

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 1 if device == "cuda" else 1  # 降低显存占用

    print(f"✓ 设备: {device}")
    print(f"✓ 计算类型: {compute_type}")
    print()

    # ========== 步骤 1: 使用 Whisper large-v3 转录 ==========
    print("[1/6] 🎙️  使用 Whisper large-v3 进行语音识别...")

    cached_result = load_checkpoint(CHECKPOINT_STEP1, "语音识别")
    if cached_result is not None:
        result = {"segments": cached_result}
    else:
        print("      (这是最高精度的模型，首次使用会下载，请耐心等待)")

        model = whisperx.load_model(
            "large-v3",
            device,
            compute_type=compute_type,
            vad_method="silero"  # 使用 Silero VAD 进行语音活动检测
        )
        audio = whisperx.load_audio(audio_file)

        result = model.transcribe(audio, batch_size=batch_size, language="ja")
        print(f"✓ 转录完成，识别到 {len(result['segments'])} 个片段")

        # 清理重复文本 (Whisper hallucination)
        print("  清理重复文本...")
        for seg in result['segments']:
            seg['text'] = clean_repetitive_text(seg['text'])

        # 保存检查点
        save_checkpoint(CHECKPOINT_STEP1, result['segments'], "语音识别")

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    print()

    # ========== 步骤 2: 对齐时间戳 ==========
    print("[2/6] ⏱️  对齐时间戳...")

    cached_result = load_checkpoint(CHECKPOINT_STEP2, "时间戳对齐")
    if cached_result is not None:
        result = {"segments": cached_result}
    else:
        # 需要重新加载音频（如果从检查点恢复）
        if 'audio' not in dir() or audio is None:
            audio = whisperx.load_audio(audio_file)

        model_a, metadata = whisperx.load_align_model(language_code="ja", device=device)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
        print("✓ 时间戳对齐完成")

        # 保存检查点
        save_checkpoint(CHECKPOINT_STEP2, result['segments'], "时间戳对齐")

        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    print()

    # ========== 步骤 3: 说话人分离 ==========
    print("[3/6] 👥 进行说话人分离...")

    cached_result = load_checkpoint(CHECKPOINT_STEP3, "说话人分离")
    if cached_result is not None:
        result = {"segments": cached_result["segments"]}
        speakers = set(cached_result.get("speakers", []))
        print(f"✓ 已缓存 {len(speakers)} 个说话人: {', '.join(sorted(speakers))}")
    else:
        #try:
            # 需要重新加载音频（如果从检查点恢复）
            if 'audio' not in dir() or audio is None:
                audio = whisperx.load_audio(audio_file)

            # 使用 pyannote.audio 4.0 的新 API（不用 whisperx 的旧封装）
            print("      正在加载说话人分离模型...")
            from pyannote.audio import Pipeline as PyannotePipeline

            # 分段处理配置 - 用时间换空间
            CHUNK_DURATION = 180  # 每段5分钟（秒）
            OVERLAP_DURATION = 60  # 重叠10秒，确保说话人连续性
            SAMPLE_RATE = 16000

            audio_duration = len(audio) / SAMPLE_RATE
            print(f"      音频总时长: {audio_duration/60:.1f} 分钟")

            diarize_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN
            ).to(torch.device(device))

            # 辅助函数：将 numpy array 转为 pyannote 期望的字典格式（绕过 torchcodec）
            def audio_to_pyannote_format(audio_array, sample_rate=16000):
                """将音频数组转换为 pyannote 内存格式，避免使用 torchcodec"""
                waveform = torch.from_numpy(audio_array).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # (samples,) -> (1, samples)
                return {"waveform": waveform, "sample_rate": sample_rate}

            all_diarize_segments = []

            if audio_duration > CHUNK_DURATION:
                # 长音频分段处理
                print(f"      音频较长，将分段处理（每段{CHUNK_DURATION}秒）...")

                chunk_start = 0
                chunk_idx = 0

                while chunk_start < audio_duration:
                    chunk_end = min(chunk_start + CHUNK_DURATION, audio_duration)

                    # 提取音频片段
                    start_sample = int(chunk_start * SAMPLE_RATE)
                    end_sample = int(chunk_end * SAMPLE_RATE)
                    chunk_audio = audio[start_sample:end_sample]

                    print(f"      处理第 {chunk_idx+1} 段: {chunk_start/60:.1f}-{chunk_end/60:.1f} 分钟...")

                    # 直接用内存格式，不写临时文件（绕过 torchcodec）
                    audio_input = audio_to_pyannote_format(chunk_audio, SAMPLE_RATE)
                    diarization = diarize_pipeline(audio_input)

                    # pyannote 4.0 返回 DiarizeOutput，需要访问 speaker_diarization 属性
                    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                        all_diarize_segments.append({
                            "start": turn.start + chunk_start,
                            "end": turn.end + chunk_start,
                            "speaker": speaker
                        })

                    # 清理GPU内存
                    del audio_input, diarization
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                    # 移动到下一段（减去重叠部分）
                    chunk_start = chunk_end - OVERLAP_DURATION
                    if chunk_start >= audio_duration - OVERLAP_DURATION:
                        break
                    chunk_idx += 1

                # 合并重叠区域的说话人段（简单策略：保留较早的）
                all_diarize_segments.sort(key=lambda x: x["start"])

                diarize_segments = all_diarize_segments
            else:
                # 短音频直接处理
                print("      正在分析说话人...")
                audio_input = audio_to_pyannote_format(audio, SAMPLE_RATE)
                diarization = diarize_pipeline(audio_input)

                # pyannote 4.0 返回 DiarizeOutput，需要访问 speaker_diarization 属性
                diarize_segments = []
                for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                    diarize_segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
                del audio_input, diarization

            # 清理 pipeline 释放显存
            del diarize_pipeline
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 转换为 DataFrame（whisperx.assign_word_speakers 需要 DataFrame）
            diarize_df = pd.DataFrame(diarize_segments)

            # 分配说话人到片段
            result = whisperx.assign_word_speakers(diarize_df, result)
            print("✓ 说话人分离完成")

            speakers = set()
            for seg in result["segments"]:
                if "speaker" in seg:
                    speakers.add(seg["speaker"])
            print(f"✓ 检测到 {len(speakers)} 个说话人: {', '.join(sorted(speakers))}")

            # 保存检查点（包含说话人列表）
            checkpoint_data = {
                "segments": result["segments"],
                "speakers": list(speakers)
            }
            save_checkpoint(CHECKPOINT_STEP3, checkpoint_data, "说话人分离")

            # 清理模型
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        #except Exception as e:
         #   print(f"⚠ 说话人分离失败: {e}")
            print("  将继续处理但不区分说话人（所有片段标记为 SPEAKER_00）")

            # 为所有片段添加默认说话人
            # for seg in result["segments"]:
            #     if "speaker" not in seg:
            #         seg["speaker"] = "SPEAKER_00"

            # speakers = {"SPEAKER_00"}

            # 保存检查点
            checkpoint_data = {
                "segments": result["segments"],
                "speakers": list(speakers)
            }
            save_checkpoint(CHECKPOINT_STEP3, checkpoint_data, "说话人分离")
    print()

    # ========== 步骤 4: 构建说话人人设 ==========
    print("[4/6] 🎭 分析说话人风格...")

    cached_result = load_checkpoint(CHECKPOINT_STEP4, "风格分析")
    if cached_result is not None:
        personas = cached_result
    else:
        personas = {}
        by_speaker = {}

        for seg in result["segments"]:
            speaker = seg.get("speaker", "SPEAKER_00")
            by_speaker.setdefault(speaker, []).append(seg["text"])

        for speaker, texts in by_speaker.items():
            if len(texts) >= 3:  # 至少有3句话才分析
                sample = "\n".join(texts[:10])  # 取前10句
                print(f"  分析 {speaker} 的风格...")
                personas[speaker] = build_speaker_persona(sample)
            else:
                personas[speaker] = {}

        print("✓ 风格分析完成")

        # 保存检查点
        save_checkpoint(CHECKPOINT_STEP4, personas, "风格分析")
    print()

    # ========== 步骤 5: 优化片段 ==========
    print("[5/6] 📝 优化字幕片段...")

    cached_result = load_checkpoint(CHECKPOINT_STEP5, "片段优化")
    if cached_result is not None:
        result["segments"] = cached_result
    else:
        result["segments"] = merge_short_segments(result["segments"], min_duration=1.5)
        print(f"✓ 优化后共 {len(result['segments'])} 个片段")

        # 保存检查点
        save_checkpoint(CHECKPOINT_STEP5, result["segments"], "片段优化")
    print()

    # ========== 步骤 6: 使用千问翻译 ==========
    print(f"[6/6] 🌐 使用千问 ({QWEN_MODEL}) 翻译...")

    cached_result = load_checkpoint(CHECKPOINT_STEP6, "翻译")
    if cached_result is not None:
        result["segments"] = cached_result
    else:
        print(f"      共需翻译 {len(result['segments'])} 个片段")

        for i, seg in enumerate(result["segments"], 1):
            ja_text = seg["text"]
            speaker = seg.get("speaker", "SPEAKER_00")
            persona = personas.get(speaker, {})

            if i % 10 == 0 or i == 1:
                print(f"  翻译进度: {i}/{len(result['segments'])} ({i*100//len(result['segments'])}%)")

            zh_text = translate_text(ja_text, persona)
            seg["zh_translation"] = zh_text
            seg["persona"] = persona

        print("✓ 翻译完成")

        # 保存检查点
        save_checkpoint(CHECKPOINT_STEP6, result["segments"], "翻译")
    print()

    # ========== 保存结果 ==========
    print("💾 保存结果...")

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    print(f"✓ JSON 文件: {OUTPUT_JSON}")

    generate_srt(result["segments"], OUTPUT_SRT)

    print()
    print("=" * 70)
    print("✅ 处理完成！")
    print("=" * 70)
    print(f"输出文件:")
    print(f"  📄 字幕文件: {OUTPUT_SRT}")
    print(f"  📊 JSON 数据: {OUTPUT_JSON}")
    print()

    # 显示示例
    print("前 3 条字幕预览:")
    print("-" * 70)
    for i, seg in enumerate(result["segments"][:3], 1):
        speaker = seg.get('speaker', 'UNKNOWN')
        print(f"\n{i}. [{speaker}] {format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        print(f"   日: {seg['text']}")
        print(f"   中: {seg['zh_translation']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="日语音频处理")
    parser.add_argument("--force", "-f", action="store_true",
                        help="强制重新运行，忽略所有检查点")
    parser.add_argument("--clear", "-c", action="store_true",
                        help="仅清除检查点，不运行处理")
    args = parser.parse_args()

    try:
        if args.clear:
            clear_checkpoints()
            print("✓ 检查点已清除")
        else:
            process_audio(INPUT_AUDIO, force_restart=args.force)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
