# 语音评分系统使用说明

## 概述

`inference.py` 是一个语音评分整合系统，集成了三个核心功能：

1. **语音质量评分** — 使用 CNN-LSTM 模型对语音进行质量评估（1.0-6.0 分）
2. **语音转文字** — 使用 faster-whisper 将语音转换为文本
3. **内容评分** — 使用 Ollama LLM 对回答内容进行评分（0-10 分）

## 环境依赖

```bash
pip install torch torchaudio faster-whisper requests
```

## 所需模型文件

| 模型         | 默认路径                                             | 说明                                   |
| ------------ | ---------------------------------------------------- | -------------------------------------- |
| wav2vec 模型 | `/root/autodl-tmp/models/checkpoints/best_model.pth` | 语音质量评分模型                       |
| Whisper 模型 | `wav2text/models/whisper-base.en`                    | 语音转文字模型                         |
| Ollama 服务  | `http://localhost:11434`                             | 需要运行 Ollama 并拉取 `qwen3:8b` 模型 |

## 使用方式

### 方式一：命令行调用

```bash
python inference.py \
    --audio /path/to/audio.wav \
    --transcript /path/to/transcript.txt \
    --task /path/to/task.txt \
    --wav2vec-model /path/to/best_model.pth \
    --whisper-model /path/to/whisper-base.en \
    --ollama-host http://localhost:11434 \
    --ollama-model qwen3:8b
```

**参数说明：**

| 参数              | 必需 | 默认值                     | 说明                   |
| ----------------- | ---- | -------------------------- | ---------------------- |
| `--audio`         | 是   | `test_data/test_audio.wav` | 考生语音文件路径       |
| `--transcript`    | 是   | `test_data/听力原文.txt`   | 听力原文文件路径       |
| `--task`          | 是   | `test_data/任务描述.txt`   | 任务要求文件路径       |
| `--wav2vec-model` | 否   | 见上表                     | wav2vec 模型路径       |
| `--whisper-model` | 否   | 见上表                     | Whisper 模型路径       |
| `--ollama-host`   | 否   | `http://localhost:11434`   | Ollama 服务地址        |
| `--ollama-model`  | 否   | `qwen3:8b`                 | Ollama 模型名称        |
| `--output-json`   | 否   | 无                         | 结果输出 JSON 文件路径 |

### 方式二：作为模块导入

```python
from inference import SpeechScoringSystem

# 初始化系统
system = SpeechScoringSystem(
    wav2vec_model_path="/path/to/best_model.pth",
    whisper_model_path="/path/to/whisper-base.en",
    ollama_host="http://localhost:11434",
    ollama_model="qwen3:8b"
)

# 单个文件评估
result = system.evaluate(
    audio_path="audio.wav",
    transcript_path="transcript.txt",
    task_path="task.txt"
)

print(f"语音评分: {result.audio_score}")
print(f"转写文本: {result.transcribed_text}")
print(f"内容评分: {result.llm_score}")
```

### 方式三：快速评估函数

```python
from inference import quick_evaluate

result = quick_evaluate(
    audio_path="audio.wav",
    transcript_path="transcript.txt",
    task_path="task.txt",
    wav2vec_model_path="/path/to/best_model.pth",
    whisper_model_path="/path/to/whisper-base.en"
)
```

### 方式四：批量评估

```python
from inference import SpeechScoringSystem

system = SpeechScoringSystem(...)

results = system.evaluate_batch(
    audio_paths=["audio1.wav", "audio2.wav", "audio3.wav"],
    transcript_path="transcript.txt",
    task_path="task.txt",
    output_json="results.json"  # 可选，自动保存结果
)
```

## 输出结果

### 控制台输出示例

```
============================================================
语音评分结果
============================================================
1. 语音质量评分: 4.25 / 6.0
2. 转文字结果:
   The student's response about the topic...
3. 内容评分 (LLM): 7.5 / 10.0
============================================================
```

### JSON 输出格式

```json
{
  "audio_score": 4.25,
  "transcribed_text": "The student's response...",
  "llm_score": 7.5,
  "audio_score_detail": {
    "raw_score": 4.253,
    "clamped_score": 4.25,
    "device": "cuda"
  },
  "llm_score_detail": "7.5"
}
```

## 输入文件格式

- **音频文件**：支持 `.wav`、`.mp3`、`.flac` 格式
- **听力原文**：纯文本 `.txt` 文件，支持 UTF-8/GBK 编码
- **任务描述**：纯文本 `.txt` 文件，描述学生需要完成的任务

## 注意事项

通过 [ModelScope](https://modelscope.cn/models/modelscope/ollama-linux?utm_source=chatgpt.com) 下载 Ollama Linux 客户端：

```
pip install modelscope
modelscope download --model=modelscope/ollama-linux --local_dir ./ollama-linux --revision v0.14.1
```

1. 首次运行前请确保 Ollama 服务已启动：`ollama serve`
2. 需要预先拉取 LLM 模型：`ollama pull qwen3:8b`
3. 音频最大处理长度为 30 秒，超出部分会被截断
4. GPU 可用时自动使用 CUDA 加速，否则使用 CPU
5. 若需要访问UI界面，请确保api成功加载，需要在先在一个终端uvicorn api_server:app --host 127.0.0.1 --port 8000 再新开一个终端启动./cloudflared tunnel --url http://127.0.0.1:8000

## UI界面展示

![A46YJ FEU)({F N1_`TO)EV](https://github.com/user-attachments/assets/19054845-2195-48cc-86b4-d3a94e8e8e8c)

![{%W`LGUVVJP8VW4~6N0 5DS](https://github.com/user-attachments/assets/f3755f9c-2a5c-4309-a5d1-a605983ff03f)
