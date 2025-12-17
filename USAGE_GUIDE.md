# 使用指南

## 快速开始

### 1. 安装依赖

```bash
# 安装 FFmpeg（必需）
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载并添加到 PATH

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 配置 HuggingFace Token

```bash
# 访问 https://huggingface.co/settings/tokens 创建 token
export HF_TOKEN="your-huggingface-token"
```

### 3. 准备音频文件

将你的音频文件放在项目目录下，命名为 `input.mp3`（或使用 `--input` 参数指定）

### 4. 运行处理

```bash
# 使用完整流程（推荐）
python complete_pipeline.py --input input.mp3

# 或使用统一流程
python unified_pipeline.py --input input.mp3
```

## 详细使用场景

### 场景 1: 处理单人说话的纯净音频

```bash
# 使用简单流程即可
python main_local.py
```

**输入**: 一个人清晰说话的音频
**输出**: 
- `output.srt` - 带时间戳的中文字幕
- `output.json` - 包含原文和翻译的完整数据

### 场景 2: 处理多人轮流说话的音频

```bash
# 使用统一流程
python unified_pipeline.py --input input.mp3
```

**适用**: 对话、访谈、会议等场景
**特点**: 自动识别说话人，分别标注

### 场景 3: 处理混杂语音（多人同时说话）

```bash
# 必须使用完整流程
python complete_pipeline.py --input input.mp3
```

**适用**: 
- 嘈杂的会议
- 多人讨论
- 包含背景对话的音频

**处理过程**:
1. 识别混杂片段
2. 使用源分离模型分离各个说话人
3. 分别转录每个说话人
4. 合并结果

### 场景 4: 未知说话人数量

```bash
# 自动检测
python complete_pipeline.py --input input.mp3
```

系统会自动检测说话人数量（1-4人）

### 场景 5: 需要高质量翻译

```bash
# 使用 Qwen 模型
export HF_TOKEN="your-token"
python main_qwen.py
```

或者

```bash
# 使用 OpenAI API（需要付费）
# 编辑 main.py 设置 API key
python main.py
```

## 输出格式

### SRT 字幕格式

```
1
00:00:01,500 --> 00:00:03,200
[SPEAKER_00] 你好，今天天气真好
(こんにちは、今日はいい天気ですね)

2
00:00:03,500 --> 00:00:05,800
[SPEAKER_01] 是啊，我们出去走走吧
(そうですね、出かけましょう)
```

### JSON 格式

```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 1.5,
    "end": 3.2,
    "text": "こんにちは、今日はいい天気ですね",
    "zh_translation": "你好，今天天气真好"
  },
  {
    "speaker": "SPEAKER_01",
    "start": 3.5,
    "end": 5.8,
    "text": "そうですね、出かけましょう",
    "zh_translation": "是啊，我们出去走走吧"
  }
]
```

## 常见问题

### Q: GPU 内存不足怎么办？

A: 完整流程需要较多显存。如果遇到 OOM：

1. 使用较小的 batch size
2. 处理较短的音频片段
3. 使用 CPU 模式（会慢很多）

```bash
# 强制使用 CPU
export CUDA_VISIBLE_DEVICES=""
python complete_pipeline.py --input input.mp3
```

### Q: 说话人分离效果不好？

A: 可能的原因：

1. 音频质量差（噪音太多）
2. 说话人声音相似
3. 背景音乐或噪音干扰

**解决方案**:
- 使用音频预处理（降噪）
- 尝试不同的分离模型
- 手动调整参数

### Q: 翻译质量不满意？

A: 尝试以下方案：

1. 使用 Qwen 模型（更好的翻译质量）
2. 使用 OpenAI API（最佳质量，需付费）
3. 自定义翻译提示词

### Q: 处理时间太长？

A: 优化建议：

1. 使用 GPU（比 CPU 快 10-20 倍）
2. 使用较小的 Whisper 模型（large-v3 → medium）
3. 跳过源分离（如果没有混杂语音）
4. 使用本地翻译模型而非 API

### Q: 检测到的说话人数量不对？

A: 

```bash
# 手动指定说话人数量
python unified_pipeline.py --input input.mp3 --num-speakers 3
```

### Q: 如何处理非日语音频？

A: 修改代码中的语言参数：

```python
# 在 complete_pipeline.py 中
result = model.transcribe(audio_data, batch_size=16, language="en")  # 改为英语

# 同时需要修改翻译模型
model_name = "Helsinki-NLP/opus-mt-en-zh"  # 英译中模型
```

## 性能参考

基于 NVIDIA RTX 3090（24GB）测试：

| 音频时长 | 说话人数 | 混杂片段 | 处理时间 | 显存占用 |
|---------|---------|---------|---------|---------|
| 5 分钟 | 1 人 | 无 | 2-3 分钟 | 4-6 GB |
| 5 分钟 | 2 人 | 少量 | 4-5 分钟 | 6-8 GB |
| 5 分钟 | 4 人 | 较多 | 8-12 分钟 | 10-15 GB |
| 30 分钟 | 2 人 | 中等 | 20-30 分钟 | 8-12 GB |

CPU 模式大约慢 10-20 倍。

## 高级用法

### 批量处理多个文件

```bash
# 创建批处理脚本
for file in audio/*.mp3; do
    output_name=$(basename "$file" .mp3)
    python complete_pipeline.py \
        --input "$file" \
        --output-srt "output/${output_name}.srt" \
        --output-json "output/${output_name}.json"
done
```

### 自定义模型

```python
# 在 complete_pipeline.py 中修改

# 使用更小的 Whisper 模型（更快但精度稍低）
model = whisperx.load_model("medium", device, compute_type=compute_type)

# 使用不同的翻译模型
model_name = "Helsinki-NLP/opus-mt-ja-zh"  # 日译中
# 或
model_name = "Helsinki-NLP/opus-mt-jap-zho"  # 另一个日译中模型
```

### 保存中间结果

```python
# 在处理流程中添加检查点
# 保存分离后的音频
for speaker, audio_track in speaker_tracks.items():
    sf.write(f"checkpoints/{speaker}.wav", audio_track, SAMPLE_RATE)

# 保存未翻译的转录
with open("checkpoints/transcriptions.json", "w") as f:
    json.dump(all_segments, f, ensure_ascii=False, indent=2)
```

## 故障排查

### 模型下载问题

如果模型下载失败：

```bash
# 设置镜像（中国用户）
export HF_ENDPOINT="https://hf-mirror.com"

# 或手动下载模型到指定目录
# 然后在代码中指定本地路径
```

### 音频格式不支持

```bash
# 使用 FFmpeg 转换
ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

### ImportError

```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt

# 或单独安装问题包
pip install --upgrade whisperx pyannote.audio speechbrain
```

## 贡献和反馈

遇到问题或有改进建议？欢迎提交 Issue 或 Pull Request！
