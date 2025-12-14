#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日语音频处理：语音识别 + 说话人分离 + 中文翻译 + 字幕生成
本地版本 - 使用 HuggingFace 免费翻译模型
"""

import os
import json
import torch
import whisperx
from datetime import timedelta
from transformers import MarianMTModel, MarianTokenizer
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
OUTPUT_SRT = "output.srt"
OUTPUT_JSON = "output.json"

# 全局翻译模型（延迟加载）
translation_model = None
translation_tokenizer = None


def format_timestamp(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def load_translation_model():
    """加载翻译模型（Helsinki-NLP 的日语到中文模型）"""
    global translation_model, translation_tokenizer

    if translation_model is None:
        print("加载翻译模型 (Helsinki-NLP/opus-mt-ja-zh)...")
        model_name = "Helsinki-NLP/opus-mt-ja-zh"
        translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
        translation_model = MarianMTModel.from_pretrained(model_name)

        # 如果有 GPU，移到 GPU
        if torch.cuda.is_available():
            translation_model = translation_model.to("cuda")

        print("✓ 翻译模型加载完成")


def translate_text(text):
    """使用本地模型翻译文本"""
    global translation_model, translation_tokenizer

    if not text or not text.strip():
        return text

    try:
        # 确保模型已加载
        if translation_model is None:
            load_translation_model()

        # 准备输入
        inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 移到相同设备
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 生成翻译
        with torch.no_grad():
            translated = translation_model.generate(**inputs, max_length=512)

        # 解码
        translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)

        return translated_text.strip()

    except Exception as e:
        print(f"⚠ 翻译出错: {e}")
        return text  # 返回原文


def generate_srt(segments, output_file):
    """生成 SRT 字幕文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")

            # 显示说话人和翻译后的文本
            speaker = seg.get('speaker', 'SPEAKER_00')
            zh_text = seg.get('zh_translation', seg['text'])

            f.write(f"[{speaker}] {zh_text}\n")
            f.write(f"({seg['text']})\n")  # 原文
            f.write("\n")

    print(f"✓ 字幕文件已生成: {output_file}")


def process_audio(audio_file):
    """主处理流程"""
    print("=" * 60)
    print("开始处理音频文件...")
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

    # ========== 步骤 1: 加载音频并转录 ==========
    print("[1/6] 加载 Whisper 模型并转录音频...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
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
    print("[2/6] 对齐时间戳...")
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
    print("[3/6] 进行说话人分离...")
    try:
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

        # 可以指定说话人数量范围，如果不确定可以不指定
        diarize_segments = diarize_model(audio)
        # 或者指定范围: diarize_model(audio, min_speakers=2, max_speakers=4)

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

    # ========== 步骤 4: 加载翻译模型 ==========
    print("[4/6] 加载翻译模型...")
    load_translation_model()
    print()

    # ========== 步骤 5: 翻译为中文 ==========
    print("[5/6] 翻译为中文...")
    for i, seg in enumerate(result["segments"]):
        ja_text = seg["text"]
        print(f"  [{i+1}/{len(result['segments'])}] 翻译: {ja_text[:30]}...")
        zh_text = translate_text(ja_text)
        seg["zh_translation"] = zh_text
    print("✓ 翻译完成")
    print()

    # ========== 步骤 6: 生成输出文件 ==========
    print("[6/6] 生成输出文件...")

    # 保存 JSON 格式（包含所有信息）
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    print(f"✓ JSON 文件已生成: {OUTPUT_JSON}")

    # 生成 SRT 字幕
    generate_srt(result["segments"], OUTPUT_SRT)

    print()
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - 字幕文件: {OUTPUT_SRT}")
    print(f"  - JSON 数据: {OUTPUT_JSON}")
    print()

    # 显示前几条结果示例
    print("前 3 条字幕预览:")
    print("-" * 60)
    for i, seg in enumerate(result["segments"][:3], 1):
        speaker = seg.get('speaker', 'UNKNOWN')
        print(f"{i}. [{speaker}] {format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        print(f"   日语: {seg['text']}")
        print(f"   中文: {seg['zh_translation']}")
        print()


if __name__ == "__main__":
    try:
        process_audio(INPUT_AUDIO)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
