#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日语音频处理：语音识别 + 说话人分离 + 中文翻译 + 字幕生成
"""

import os
import json
import torch
import whisperx
from datetime import timedelta
from openai import OpenAI
import gc

# 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INPUT_AUDIO = "input.mp3"
OUTPUT_SRT = "output.srt"
OUTPUT_JSON = "output.json"

# OpenAI 客户端（用于翻译，你也可以换成其他翻译服务）
# 这里使用兼容 OpenAI API 的服务
client = OpenAI(
    api_key="sk-xxx",  # 替换为你的 API key
    base_url="https://api.openai.com/v1"  # 或使用其他兼容服务
)


def format_timestamp(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def translate_text(text, source_lang="ja", target_lang="zh"):
    """使用 GPT 翻译文本"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"你是一个专业的{source_lang}到{target_lang}翻译专家。请直接翻译以下文本，不要添加任何解释。保持口语化和自然。"},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"翻译出错: {e}")
        return text  # 翻译失败时返回原文


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
    print("[1/5] 加载 Whisper 模型并转录音频...")
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
    print("[2/5] 对齐时间戳...")
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
    print("[3/5] 进行说话人分离...")
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

    # ========== 步骤 4: 翻译为中文 ==========
    print("[4/5] 翻译为中文...")
    for i, seg in enumerate(result["segments"]):
        ja_text = seg["text"]
        print(f"  [{i+1}/{len(result['segments'])}] 翻译: {ja_text[:30]}...")
        zh_text = translate_text(ja_text, "ja", "zh")
        seg["zh_translation"] = zh_text
    print("✓ 翻译完成")
    print()

    # ========== 步骤 5: 生成输出文件 ==========
    print("[5/5] 生成输出文件...")

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
