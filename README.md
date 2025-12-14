# 日语音频处理工具

自动处理日语音频，提取语音、区分说话人、翻译成中文并生成字幕。

## 功能特性

✅ **语音识别**: 使用 WhisperX 进行高精度日语语音识别
✅ **说话人分离**: 自动识别并区分不同的说话人
✅ **中文翻译**: 将日语内容翻译成中文
✅ **字幕生成**: 生成带时间戳的 SRT 字幕文件
✅ **JSON 输出**: 保存完整的结构化数据

## 安装步骤

### 1. 安装 FFmpeg

Windows 用户：
```powershell
# 使用 chocolatey
choco install ffmpeg

# 或从官网下载: https://ffmpeg.org/download.html
```

### 2. 安装 Python 依赖

```powershell
# 确保在项目目录
cd e:\Coding\translation

# 安装依赖
pip install -r requirements.txt

# 或者手动安装核心包
pip install whisperx torch torchaudio transformers pyannote.audio soundfile
```

### 3. 配置 HuggingFace Token

1. 访问 https://huggingface.co/settings/tokens
2. 创建一个访问令牌（需要 read 权限）
3. 在脚本中替换 `HF_TOKEN` 的值

## 使用方法

### 方案一：使用本地翻译模型（推荐，免费）

```powershell
python main_local.py
```

这个版本使用 Helsinki-NLP 的免费翻译模型，无需 API 密钥。

### 方案二：使用 OpenAI API（需要付费）

1. 编辑 `main.py`，设置你的 OpenAI API key：
```python
client = OpenAI(
    api_key="sk-your-api-key",
    base_url="https://api.openai.com/v1"
)
```

2. 运行：
```powershell
python main.py
```

## 输出文件

运行后会生成以下文件：

- **output.srt**: SRT 格式字幕文件，包含说话人标识、中文翻译和日语原文
- **output.json**: JSON 格式的完整数据，包含所有时间戳和说话人信息

### SRT 字幕格式示例

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

## 自定义配置

你可以在脚本中修改以下参数：

```python
# 输入/输出文件名
INPUT_AUDIO = "input.mp3"
OUTPUT_SRT = "output.srt"
OUTPUT_JSON = "output.json"

# Whisper 模型选择 (tiny, base, small, medium, large-v3)
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

# 说话人数量范围（如果已知）
diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=4)
```

## 常见问题

### Q: GPU 内存不足怎么办？

A: 可以降低 batch_size 或使用更小的模型：
```python
batch_size = 4  # 降低批次大小
model = whisperx.load_model("medium", device, compute_type="int8")
```

### Q: 说话人分离效果不好？

A: 尝试指定说话人数量范围：
```python
diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=3)
```

### Q: 翻译质量不够好？

A: 本地模型的翻译质量有限，建议使用 OpenAI API 或其他商业翻译服务。

### Q: 没有 GPU 能用吗？

A: 可以，但会慢很多。脚本会自动检测并使用 CPU 模式。

## 技术栈

- **WhisperX**: 语音识别和时间对齐
- **Pyannote Audio**: 说话人分离
- **Transformers**: 翻译模型
- **PyTorch**: 深度学习框架

## 许可证

本项目仅供学习和研究使用。请确保：
- 遵守各个模型的使用条款
- 不要用于商业用途（除非获得相应授权）
- HuggingFace Token 和 API Key 请妥善保管

## 更新日志

### v1.0 (2025-12-15)
- 初始版本
- 支持日语语音识别
- 支持说话人分离
- 支持中文翻译
- 生成 SRT 字幕
