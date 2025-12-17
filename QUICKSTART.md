# 快速开始指南

## 🚀 5分钟上手

### 1️⃣ 安装依赖

```bash
# 安装 FFmpeg（必需）
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2️⃣ 配置 Token

```bash
# 获取 HuggingFace Token
# 访问: https://huggingface.co/settings/tokens
# 创建一个 read 权限的 token

# 设置环境变量
export HF_TOKEN="your-token-here"
```

### 3️⃣ 运行处理

```bash
# 准备音频文件（命名为 input.mp3 或使用 --input 指定）
# 运行完整流程
python complete_pipeline.py --input input.mp3

# 等待处理完成...
# 输出: output.srt (字幕) 和 output.json (数据)
```

## 💡 使用场景

### 场景 1: 单人清晰录音

```bash
# 使用快速版本即可
python main_local.py
```

⏱️ 处理时间: 5分钟音频约 2-3 分钟

### 场景 2: 多人对话（轮流说话）

```bash
# 使用统一流程
python unified_pipeline.py --input conversation.mp3
```

⏱️ 处理时间: 5分钟音频约 3-5 分钟

### 场景 3: 混杂语音（多人同时说话）

```bash
# 必须使用完整流程
python complete_pipeline.py --input meeting.mp3
```

⏱️ 处理时间: 5分钟音频约 5-12 分钟

### 场景 4: 需要高质量翻译

```bash
# 方案 A: 使用 Qwen 大模型（免费但需 Token）
python main_qwen.py

# 方案 B: 使用 OpenAI API（付费）
# 编辑 main.py 设置 API key
python main.py
```

## 📄 输出示例

### SRT 字幕文件 (output.srt)

```srt
1
00:00:01,500 --> 00:00:03,200
[SPEAKER_00] 你好，今天天气真好
(こんにちは、今日はいい天気ですね)

2
00:00:03,500 --> 00:00:05,800
[SPEAKER_01] 是啊，我们出去走走吧
(そうですね、出かけましょう)
```

### JSON 数据文件 (output.json)

```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 1.5,
    "end": 3.2,
    "text": "こんにちは、今日はいい天気ですね",
    "zh_translation": "你好，今天天气真好"
  }
]
```

## ❓ 常见问题

### Q: 显示 "GPU 内存不足"

```bash
# 解决方案 1: 使用 CPU（会很慢）
export CUDA_VISIBLE_DEVICES=""
python complete_pipeline.py --input input.mp3

# 解决方案 2: 处理更短的音频片段
ffmpeg -i long_audio.mp3 -ss 00:00:00 -t 00:05:00 -c copy short_audio.mp3
```

### Q: 下载模型太慢

```bash
# 使用国内镜像（中国用户）
export HF_ENDPOINT="https://hf-mirror.com"
python complete_pipeline.py --input input.mp3
```

### Q: 说话人识别不准确

```bash
# 手动指定说话人数量
python unified_pipeline.py --input input.mp3 --num-speakers 3
```

### Q: 翻译质量不好

```bash
# 使用更好的翻译模型
python main_qwen.py  # Qwen 大模型

# 或
python main.py  # OpenAI API（需付费）
```

## 📊 性能参考

| 音频时长 | 说话人 | 混杂程度 | GPU (3090) | CPU |
|---------|-------|---------|-----------|-----|
| 5分钟   | 1人   | 无      | 2-3分钟   | 20-30分钟 |
| 5分钟   | 2人   | 少量    | 4-5分钟   | 40-60分钟 |
| 5分钟   | 4人   | 较多    | 8-12分钟  | 80-120分钟 |

## 🔧 高级设置

### 批量处理

```bash
# 创建批处理脚本
for file in audio/*.mp3; do
    name=$(basename "$file" .mp3)
    python complete_pipeline.py \
        --input "$file" \
        --output-srt "output/${name}.srt" \
        --output-json "output/${name}.json"
done
```

### 自定义输出

```bash
# 指定输出路径
python complete_pipeline.py \
    --input my_audio.mp3 \
    --output-srt results/my_subtitle.srt \
    --output-json results/my_data.json
```

## 📚 更多文档

- [完整使用指南](USAGE_GUIDE.md) - 详细的使用说明
- [技术总结](SOLUTION_SUMMARY.md) - 技术细节和算法说明
- [README](README.md) - 项目概述

## 🆘 获取帮助

遇到问题？

1. 查看 [USAGE_GUIDE.md](USAGE_GUIDE.md) 的故障排查部分
2. 检查依赖是否正确安装: `pip list`
3. 确认 FFmpeg 已安装: `ffmpeg -version`
4. 提交 Issue 到 GitHub

## 🎉 开始使用

```bash
# 一键运行完整流程
python complete_pipeline.py --input your_audio.mp3

# 就这么简单！
```

祝你使用愉快！🚀
