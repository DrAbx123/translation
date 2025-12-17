# 日语音频处理工具

自动处理日语音频，提取语音、区分说话人、翻译成中文并生成字幕。

## 功能特性

✅ **语音识别**: 使用 WhisperX 进行高精度日语语音识别
✅ **说话人分离**: 自动识别并区分不同的说话人
✅ **混杂语音处理**: 使用源分离模型处理多人同时说话的情况
✅ **完整端到端流程**: 从原始音频到最终中文字幕一键完成
✅ **中文翻译**: 将日语内容翻译成中文
✅ **字幕生成**: 生成带时间戳的 SRT 字幕文件
✅ **JSON 输出**: 保存完整的结构化数据

## 支持的场景

- ✅ 无人说话的片段（自动检测并跳过）
- ✅ 单人说话的纯净音频
- ✅ 多人轮流说话（通过说话人分离处理）
- ✅ 多人同时说话的混杂音频（通过源分离模型处理）
- ✅ 自动检测说话人数量（1-4人，可扩展）

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

### 🌟 推荐：完整的端到端流程（complete_pipeline.py）

这是最新的完整解决方案，能处理所有场景，包括混杂语音：

```bash
# 基本用法
python complete_pipeline.py --input input.mp3

# 指定输出文件
python complete_pipeline.py --input audio.mp3 --output-srt result.srt --output-json result.json
```

**功能特点：**
- ✅ 自动检测语音活动（无语音自动跳过）
- ✅ 自动识别混杂语音片段
- ✅ 使用源分离模型分离多人同时说话的音频
- ✅ 分别转录每个说话人
- ✅ 自动翻译为中文
- ✅ 生成统一的字幕文件

### 方案二：统一流程（unified_pipeline.py）

适用于简单场景（无混杂语音或混杂程度较低）：

```bash
python unified_pipeline.py --input input.mp3
```

### 方案三：使用本地翻译模型（main_local.py）

不需要处理混杂语音的简单场景：

```bash
python main_local.py
```

这个版本使用 Helsinki-NLP 的免费翻译模型，无需 API 密钥。

### 方案四：使用 Qwen 大模型翻译（main_qwen.py）

最高质量的翻译，但需要 HuggingFace Token：

```bash
export HF_TOKEN="your-huggingface-token"
python main_qwen.py
```

### 方案五：使用 OpenAI API（main.py）

需要付费，需要设置 API key：

```bash
# 编辑 main.py 设置你的 API key，然后运行
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

## 各方案对比

| 方案 | 适用场景 | 混杂语音处理 | 翻译质量 | 速度 | 成本 |
|------|---------|------------|---------|------|------|
| **complete_pipeline.py** | 所有场景（推荐） | ✅ 完整支持 | 良好 | 较慢 | 免费 |
| unified_pipeline.py | 简单场景 | ⚠️ 基础支持 | 良好 | 快 | 免费 |
| main_qwen.py | 需要高质量翻译 | ⚠️ 基础支持 | 优秀 | 中等 | 免费（需HF） |
| main_local.py | 简单场景 | ❌ 不支持 | 良好 | 快 | 免费 |
| main.py | 需要API翻译 | ❌ 不支持 | 优秀 | 快 | 付费 |

## 工作流程说明

### complete_pipeline.py 完整流程

1. **加载音频** - 自动转换为标准格式（16kHz, 单声道）
2. **检测语音活动** - 使用 VAD 检测是否有人说话
3. **说话人分离** - 使用 Pyannote 识别谁在什么时候说话
4. **识别混杂片段** - 自动检测多人同时说话的时间段
5. **源分离** - 对混杂片段使用 SepFormer 模型分离各个说话人
6. **创建音轨** - 为每个说话人生成完整的音频轨道
7. **转录** - 使用 WhisperX 分别转录每个说话人
8. **翻译** - 将所有日语文本翻译为中文
9. **生成字幕** - 输出 SRT 和 JSON 格式

### 关键技术

- **Whisper large-v3**: 最高精度的语音识别
- **Pyannote 3.1**: 最新的说话人分离模型
- **SepFormer**: 声源分离模型（支持2-4人）
- **Helsinki-NLP**: 日译中翻译模型

## 自定义配置

### 环境变量

```bash
# HuggingFace Token（用于下载模型）
export HF_TOKEN="your-token"

# 是否使用 Qwen 翻译（unified_pipeline.py）
export USE_QWEN="true"
```

### 命令行参数

```bash
# complete_pipeline.py
python complete_pipeline.py \
  --input audio.mp3 \
  --output-srt result.srt \
  --output-json result.json

# unified_pipeline.py
python unified_pipeline.py \
  --input audio.mp3 \
  --use-qwen \
  --num-speakers 4
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
