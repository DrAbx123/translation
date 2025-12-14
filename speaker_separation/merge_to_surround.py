#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将4个分离的说话人音轨合成为4声道环绕声文件
左前(FL) - 左后(RL) - 右前(FR) - 右后(RR)
"""

import numpy as np
import soundfile as sf
import os

# 配置
INPUT_DIR = "speaker_separation/output"
OUTPUT_FILE = "speaker_separation/output/combined_surround.wav"


def main():
    print("=" * 60)
    print("合成4声道环绕声")
    print("声道分配: 左前(FL) - 右前(FR) - 左后(RL) - 右后(RR)")
    print("=" * 60)

    # 查找所有说话人音频文件
    speaker_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.startswith("SPEAKER_") and f.endswith(".wav")
    ])

    if len(speaker_files) == 0:
        print("错误: 没有找到说话人音频文件")
        print(f"请先运行 separate_voices_v2.py 生成分离音轨")
        return

    print(f"\n找到 {len(speaker_files)} 个说话人音轨:")
    for f in speaker_files:
        print(f"  - {f}")

    # 加载所有音频
    audios = []
    sample_rate = None
    max_length = 0

    for f in speaker_files:
        filepath = os.path.join(INPUT_DIR, f)
        audio, sr = sf.read(filepath)
        if sample_rate is None:
            sample_rate = sr
        audios.append(audio)
        max_length = max(max_length, len(audio))
        print(f"  加载: {f} ({len(audio)/sr:.1f}s)")

    # 确保所有音频长度相同
    for i in range(len(audios)):
        if len(audios[i]) < max_length:
            audios[i] = np.pad(audios[i], (0, max_length - len(audios[i])))

    # 创建4声道数组
    # 声道顺序: FL(左前), FR(右前), RL(左后), RR(右后)
    num_channels = 4
    combined = np.zeros((max_length, num_channels), dtype=np.float32)

    # 分配说话人到声道
    # 如果有4个说话人，每人一个声道
    # 如果少于4个，循环分配
    # 如果多于4个，前4个分别放，其余混合到各声道

    channel_names = ["左前(FL)", "右前(FR)", "左后(RL)", "右后(RR)"]

    print(f"\n声道分配:")
    for i, f in enumerate(speaker_files[:4]):
        channel_idx = i
        combined[:, channel_idx] = audios[i]
        print(f"  {channel_names[channel_idx]}: {f}")

    # 如果少于4个说话人，其他声道保持静音
    if len(speaker_files) < 4:
        print(f"\n注意: 只有 {len(speaker_files)} 个说话人，部分声道为静音")

    # 如果多于4个说话人，将多余的混合进去
    if len(speaker_files) > 4:
        print(f"\n警告: 有 {len(speaker_files)} 个说话人，多余的将混合到现有声道")
        for i in range(4, len(speaker_files)):
            channel_idx = i % 4
            combined[:, channel_idx] += audios[i] * 0.5  # 降低混合音量
            print(f"  {speaker_files[i]} 混合到 {channel_names[channel_idx]}")

    # 归一化，防止爆音
    max_val = np.abs(combined).max()
    if max_val > 0:
        combined = combined / max_val * 0.9

    # 保存为4声道WAV文件
    sf.write(OUTPUT_FILE, combined, sample_rate)

    duration = max_length / sample_rate
    print(f"\n保存: {OUTPUT_FILE}")
    print(f"时长: {duration/60:.1f} 分钟")
    print(f"声道数: {num_channels}")
    print(f"采样率: {sample_rate} Hz")

    # 同时生成一个立体声版本（左右各混合2个声道）
    stereo_file = OUTPUT_FILE.replace("_surround.wav", "_stereo.wav")
    stereo = np.zeros((max_length, 2), dtype=np.float32)

    # 左声道 = 左前 + 左后
    stereo[:, 0] = combined[:, 0] * 0.7 + combined[:, 2] * 0.5
    # 右声道 = 右前 + 右后
    stereo[:, 1] = combined[:, 1] * 0.7 + combined[:, 3] * 0.5

    # 归一化
    max_val = np.abs(stereo).max()
    if max_val > 0:
        stereo = stereo / max_val * 0.9

    sf.write(stereo_file, stereo, sample_rate)
    print(f"\n额外生成立体声版本: {stereo_file}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
