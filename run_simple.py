#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日语音频处理：快速测试版本
使用更小的模型，更快速
"""

import os
import json
import torch
import whisperx
from datetime import timedelta
import gc

# 修复 PyTorch 2.8 兼容性问题
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.base import ContainerMetadata
import omegaconf
torch.serialization.add_safe_globals([
    ListConfig, DictConfig, ContainerMetadata,
    omegaconf.nodes.AnyNode,
    omegaconf.nodes.BooleanNode,
    omegaconf.nodes.IntegerNode,
    omegaconf.nodes.FloatNode,
    omegaconf.nodes.StringNode,
    omegaconf.nodes.EnumNode,
])

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INPUT_AUDIO = "input.mp3"
OUTPUT_SRT = "output_simple.srt"
OUTPUT_JSON = "output_simple.json"


def format_timestamp(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def simple_translate(text):
    """简单的占位翻译（实际使用时需要替换）"""
    # 这里使用简单的标记，实际项目中应该调用真正的翻译服务
    return f"[翻译] {text}"


def generate_srt(segments, output_file):
    """生成 SRT 字幕文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")

            # 显示说话人和文本
            speaker = seg.get('speaker', 'SPEAKER_00')
            text = seg['text']
            zh_text = seg.get('zh_translation', simple_translate(text))

            f.write(f"[{speaker}] {zh_text}\n")
            f.write(f"({text})\n")  # 原文
            f.write("\n")

    print(f"✓ 字幕文件已生成: {output_file}")


def process_audio_simple(audio_file):
    """简化的处理流程 - 使用 base 模型"""
    print("=" * 60)
    print("开始处理音频文件 (简化版本)...")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 16 if device == "cuda" else 4

    print(f"设备: {device}")
    print(f"计算类型: {compute_type}")
    print()

    # ========== 步骤 1: 转录（使用 base 模型，更快） ==========
    print("[1/4] 加载 Whisper base 模型并转录音频...")
    print("提示: 使用 base 模型以加快速度，如需更高精度请使用 large-v3")

    # 使用 base 模型，更快
    model = whisperx.load_model("base", device, compute_type=compute_type, vad_method="silero")
    audio = whisperx.load_audio(audio_file)

    # 指定语言为日语
    result = model.transcribe(audio, batch_size=batch_size, language="ja")
    print(f"✓ 转录完成，识别到 {len(result['segments'])} 个片段")
    print()

    # 释放内存
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ========== 步骤 2: 对齐时间戳 ==========
    print("[2/4] 对齐时间戳...")
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
    print()

    # 释放内存
    del model_a
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ========== 步骤 3: 说话人分离 ==========
    print("[3/4] 进行说话人分离...")
    try:
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("✓ 说话人分离完成")

        # 统计说话人数量
        speakers = set()
        for seg in result["segments"]:
            if "speaker" in seg:
                speakers.add(seg["speaker"])
        print(f"✓ 检测到 {len(speakers)} 个说话人: {', '.join(sorted(speakers))}")
    except Exception as e:
        print(f"⚠ 说话人分离失败: {e}")
        print("  继续处理，但不会区分说话人...")
    print()

    # ========== 步骤 4: 生成输出 ==========
    print("[4/4] 生成输出文件...")

    # 添加简单的翻译标记
    for seg in result["segments"]:
        seg["zh_translation"] = simple_translate(seg["text"])

    # 保存 JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    print(f"✓ JSON 文件已生成: {OUTPUT_JSON}")

    # 生成 SRT
    generate_srt(result["segments"], OUTPUT_SRT)

    print()
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - 字幕文件: {OUTPUT_SRT}")
    print(f"  - JSON 数据: {OUTPUT_JSON}")
    print()
    print("注意: 此版本使用占位翻译，如需真实翻译请：")
    print("  1. 使用 main_local.py (本地翻译模型)")
    print("  2. 或配置 main.py 使用 OpenAI API")
    print()

    # 显示前几条结果
    print("前 3 条字幕预览:")
    print("-" * 60)
    for i, seg in enumerate(result["segments"][:3], 1):
        speaker = seg.get('speaker', 'UNKNOWN')
        print(f"{i}. [{speaker}] {format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        print(f"   日语: {seg['text']}")
        print()


if __name__ == "__main__":
    try:
        process_audio_simple(INPUT_AUDIO)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
