#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式语音评分系统
==================

功能：
1. 显示题目和听力原文
2. 播放题目音频（可选）
3. 录制考生回答
4. 自动评分并输出结果

使用方式：
    python interactive_speech_scoring.py
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import wave
import tempfile
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

# 检查并安装必要的库
def check_dependencies():
    """检查并提示安装必要的依赖"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import torchaudio
    except ImportError:
        missing.append("torchaudio")
    
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    if missing:
        print("缺少以下依赖库，请安装：")
        print(f"  pip install {' '.join(missing)}")
        if "pyaudio" in missing:
            print("\n注意：pyaudio 可能需要额外步骤安装：")
            print("  Ubuntu/Debian: sudo apt-get install portaudio19-dev python3-pyaudio")
            print("  macOS: brew install portaudio && pip install pyaudio")
            print("  Windows: pip install pyaudio")
        return False
    return True


# ============================================================================
# 录音模块
# ============================================================================

class AudioRecorder:
    """
    音频录制器
    支持按键开始/停止录音
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 format_type: int = None):
        """
        Args:
            sample_rate: 采样率
            channels: 声道数
            chunk_size: 缓冲区大小
            format_type: 音频格式（pyaudio 格式常量）
        """
        import pyaudio
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format_type = format_type or pyaudio.paInt16
        
        self.pyaudio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None
    
    def start_recording(self):
        """开始录音"""
        self.frames = []
        self.is_recording = True
        
        self.stream = self.pyaudio.open(
            format=self.format_type,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("\n🎤 录音中... 按 Enter 键停止录音")
        
        # 在后台线程中录音
        def record_loop():
            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    if self.is_recording:
                        print(f"录音错误: {e}")
                    break
        
        self.record_thread = threading.Thread(target=record_loop)
        self.record_thread.start()
    
    def stop_recording(self) -> list:
        """停止录音并返回音频帧"""
        self.is_recording = False
        
        if self.record_thread:
            self.record_thread.join(timeout=1)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        return self.frames
    
    def save_to_wav(self, filename: str, frames: list = None) -> str:
        """
        保存录音到 WAV 文件
        
        Args:
            filename: 输出文件名
            frames: 音频帧（如果为 None，使用最近录制的）
        
        Returns:
            保存的文件路径
        """
        import pyaudio
        
        if frames is None:
            frames = self.frames
        
        if not frames:
            raise ValueError("没有录音数据可保存")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format_type))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        return filename
    
    def close(self):
        """释放资源"""
        if self.stream:
            self.stream.close()
        self.pyaudio.terminate()


# ============================================================================
# 从原文件导入的评分模块（简化版）
# ============================================================================

import torch
import torch.nn as nn
import torchaudio


@dataclass
class ScoringResult:
    """评分结果数据类"""
    audio_score: float
    transcribed_text: str
    llm_score: float
    audio_score_detail: Optional[Dict[str, Any]] = None
    llm_score_detail: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "audio_score": self.audio_score,
            "transcribed_text": self.transcribed_text,
            "llm_score": self.llm_score,
            "audio_score_detail": self.audio_score_detail,
            "llm_score_detail": self.llm_score_detail
        }


class CNN_LSTM_Regressor(nn.Module):
    """CNN-LSTM 回归模型"""
    
    def __init__(self, 
                 n_mels: int = 128,
                 cnn_channels: list = None,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.3,
                 fc_hidden_size: int = 128,
                 fc_dropout: float = 0.3):
        super(CNN_LSTM_Regressor, self).__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in cnn_channels:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels
        
        self.freq_dim_after_cnn = n_mels // (2 ** len(cnn_channels))
        self.cnn_out_features = cnn_channels[-1] * self.freq_dim_after_cnn
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        lstm_output_size = lstm_hidden_size * 2
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * freq)
        
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        score = self.fc(pooled)
        
        return score


class AudioScorer:
    """语音质量评分器"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.sample_rate = 16000
        self.max_length = 30
        self.max_samples = self.sample_rate * self.max_length
        self.n_mels = 128
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=self.n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> CNN_LSTM_Regressor:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = CNN_LSTM_Regressor()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            pad_length = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db
    
    def score(self, audio_path: str) -> Tuple[float, Dict]:
        mel_spec_db = self.preprocess_audio(audio_path)
        mel_spec_db = mel_spec_db.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(mel_spec_db)
            score = output.item()
        
        score = max(1.0, min(6.0, score))
        return score, {"raw_score": output.item(), "clamped_score": score}


class SpeechToText:
    """语音转文字模块"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.compute_type = "float16"
            else:
                self.device = "cpu"
                self.compute_type = "int8"
        else:
            self.device = device
            self.compute_type = "float16" if device != "cpu" else "int8"
        
        self.model = self._load_model()
    
    def _load_model(self):
        from faster_whisper import WhisperModel
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Whisper 模型路径不存在: {self.model_path}")
        
        return WhisperModel(
            model_size_or_path=self.model_path,
            device=self.device,
            compute_type=self.compute_type
        )
    
    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=4,
            vad_filter=True
        )
        return " ".join([seg.text.strip() for seg in segments]).strip()


class LLMScorer:
    """LLM 内容评分器"""
    
    PROMPT_TEMPLATE = """You are an expert English evaluator. Score the student's answer from 0 to 10.

Criteria:
1. Relevance to the question
2. Correct use of information from reference
3. Completeness and clarity
4. Use of important keywords (not simple copying)

Question: {question}
Reference: {reference}
Student Answer: {answer}

Provide:
1. Score (0-10)
2. Brief feedback in Chinese (2-3 sentences)

Format your response as:
分数: [number]
评价: [feedback]"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.host = host.rstrip("/")
        self.model = model
    
    def score(self, question: str, reference: str, answer: str) -> Tuple[float, str]:
        import requests
        
        prompt = self.PROMPT_TEMPLATE.format(
            question=question.strip(),
            reference=reference.strip(),
            answer=answer.strip()
        )
        
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 500}
                },
                timeout=180
            )
            r.raise_for_status()
            raw_output = r.json().get("response", "").strip()
            
            # 提取分数
            m = re.search(r"分数[：:]\s*(\d+(?:\.\d+)?)", raw_output)
            if m:
                score = float(m.group(1))
                score = max(0, min(10, score))
            else:
                m = re.search(r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", raw_output)
                score = float(m.group(1)) if m else 5.0
            
            return score, raw_output
            
        except Exception as e:
            return 5.0, f"评分失败: {e}"


# ============================================================================
# 交互式主程序
# ============================================================================

class InteractiveSpeechScoring:
    """交互式语音评分系统"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: 配置字典，包含模型路径等
        """
        self.config = config or {}
        self.audio_scorer = None
        self.speech_to_text = None
        self.llm_scorer = None
        self.recorder = None
    
    def print_header(self):
        """打印欢迎信息"""
        print("\n" + "="*60)
        print("        🎓 交互式英语口语评分系统")
        print("="*60)
        print("\n本系统将对您的口语回答进行以下评估：")
        print("  1. 语音质量评分 (1.0-6.0)")
        print("  2. 语音转文字")
        print("  3. 内容评分 (0-10)")
        print("\n" + "-"*60)
    
    def get_input_files(self) -> Tuple[str, str]:
        """获取输入文件路径"""
        print("\n📁 请输入文件路径（直接回车使用默认路径）：\n")
        
        default_transcript = self.config.get("default_transcript", "test_data/听力原文.txt")
        default_task = self.config.get("default_task", "test_data/任务描述.txt")
        
        # 听力原文
        while True:
            transcript_path = input(f"听力原文文件路径 [{default_transcript}]: ").strip()
            if not transcript_path:
                transcript_path = default_transcript
                print(f"  → 使用默认路径: {transcript_path}")
            
            if os.path.exists(transcript_path):
                break
            print(f"  ❌ 文件不存在，请重新输入")
        
        # 任务要求
        while True:
            task_path = input(f"任务要求文件路径 [{default_task}]: ").strip()
            if not task_path:
                task_path = default_task
                print(f"  → 使用默认路径: {task_path}")
            
            if os.path.exists(task_path):
                break
            print(f"  ❌ 文件不存在，请重新输入")
        
        return transcript_path, task_path
    
    def read_file(self, path: str) -> str:
        """读取文本文件"""
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    
    def display_question(self, transcript: str, task: str):
        """显示题目"""
        print("\n" + "="*60)
        print("📝 题目信息")
        print("="*60)
        
        print("\n【听力原文】")
        print("-"*40)
        # 显示原文（如果太长则截断）
        if len(transcript) > 500:
            print(transcript[:500] + "...")
            print(f"\n(原文共 {len(transcript)} 字符，已截断显示)")
        else:
            print(transcript)
        
        print("\n【任务要求】")
        print("-"*40)
        print(task)
        
        print("\n" + "="*60)
    
    def play_audio_prompt(self):
        """提示播放音频（如果有的话）"""
        audio_path = input("\n🔊 题目音频路径 (直接回车跳过): ").strip()
        
        if audio_path and os.path.exists(audio_path):
            print(f"\n请播放音频文件: {audio_path}")
            print("（本系统不支持自动播放，请使用系统播放器）")
            input("播放完成后按 Enter 继续...")
        elif audio_path:
            print(f"  ⚠️ 文件不存在: {audio_path}")
    
    def record_answer(self) -> str:
        """录制考生回答"""
        print("\n" + "="*60)
        print("🎤 录音环节")
        print("="*60)
        
        print("\n准备好后，按 Enter 开始录音...")
        input()
        
        # 初始化录音器
        self.recorder = AudioRecorder(sample_rate=16000)
        
        # 开始录音
        self.recorder.start_recording()
        
        # 等待用户按 Enter 停止
        input()
        
        # 停止录音
        frames = self.recorder.stop_recording()
        
        if not frames:
            print("❌ 录音失败，没有捕获到音频")
            return None
        
        # 计算录音时长
        duration = len(frames) * self.recorder.chunk_size / self.recorder.sample_rate
        print(f"\n✅ 录音完成！时长: {duration:.1f} 秒")
        
        # 保存到临时文件
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        audio_path = os.path.join(temp_dir, f"recording_{timestamp}.wav")
        
        self.recorder.save_to_wav(audio_path, frames)
        print(f"📁 录音已保存: {audio_path}")
        
        self.recorder.close()
        
        return audio_path
    
    def initialize_models(self):
        """初始化评分模型"""
        print("\n" + "="*60)
        print("⚙️ 初始化评分模型")
        print("="*60)
        
        # 直接使用配置中的路径
        wav2vec_path = self.config.get("wav2vec_model_path")
        whisper_path = self.config.get("whisper_model_path")
        ollama_host = self.config.get("ollama_host", "http://localhost:11434")
        ollama_model = self.config.get("ollama_model", "qwen3:8b")
        
        print(f"\n模型配置:")
        print(f"  wav2vec:  {wav2vec_path}")
        print(f"  whisper:  {whisper_path}")
        print(f"  ollama:   {ollama_host} / {ollama_model}")
        print()
        
        print("[1/3] 加载语音评分模型...")
        try:
            self.audio_scorer = AudioScorer(wav2vec_path)
            print("  ✅ 语音评分模型加载成功")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            self.audio_scorer = None
        
        print("[2/3] 加载语音转文字模型...")
        try:
            self.speech_to_text = SpeechToText(whisper_path)
            print("  ✅ 语音转文字模型加载成功")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            self.speech_to_text = None
        
        print("[3/3] 初始化 LLM 评分器...")
        self.llm_scorer = LLMScorer(ollama_host, ollama_model)
        print(f"  ✅ LLM 评分器就绪 (模型: {ollama_model})")
    
    def evaluate(self, audio_path: str, transcript: str, task: str) -> dict:
        """执行评估"""
        print("\n" + "="*60)
        print("📊 正在评估...")
        print("="*60)
        
        result = {
            "audio_score": None,
            "transcribed_text": "",
            "llm_score": None,
            "llm_feedback": ""
        }
        
        # Step 1: 语音质量评分
        print("\n[1/3] 语音质量评分...")
        if self.audio_scorer:
            try:
                score, detail = self.audio_scorer.score(audio_path)
                result["audio_score"] = score
                print(f"  ✅ 语音评分: {score:.2f} / 6.0")
            except Exception as e:
                print(f"  ❌ 评分失败: {e}")
        else:
            print("  ⚠️ 语音评分模块未加载")
        
        # Step 2: 语音转文字
        print("\n[2/3] 语音转文字...")
        if self.speech_to_text:
            try:
                text = self.speech_to_text.transcribe(audio_path)
                result["transcribed_text"] = text
                print(f"  ✅ 转写完成 ({len(text)} 字符)")
                print(f"\n  转写结果：")
                print(f"  {text[:200]}{'...' if len(text) > 200 else ''}")
            except Exception as e:
                print(f"  ❌ 转写失败: {e}")
        else:
            print("  ⚠️ 语音转文字模块未加载")
        
        # Step 3: LLM 内容评分
        print("\n[3/3] LLM 内容评分...")
        if self.llm_scorer and result["transcribed_text"]:
            try:
                score, feedback = self.llm_scorer.score(task, transcript, result["transcribed_text"])
                result["llm_score"] = score
                result["llm_feedback"] = feedback
                print(f"  ✅ 内容评分: {score:.1f} / 10.0")
            except Exception as e:
                print(f"  ❌ 评分失败: {e}")
        elif not result["transcribed_text"]:
            print("  ⚠️ 无转写文本，跳过内容评分")
        
        return result
    
    def display_result(self, result: dict):
        """显示最终结果"""
        print("\n" + "="*60)
        print("📋 评估结果")
        print("="*60)
        
        print("\n┌─────────────────────────────────────────────────────────┐")
        
        # 语音质量评分
        if result["audio_score"] is not None:
            score = result["audio_score"]
            bar = "█" * int(score) + "░" * (6 - int(score))
            print(f"│ 语音质量评分: {score:.2f} / 6.0  [{bar}]")
        else:
            print(f"│ 语音质量评分: 未评估")
        
        # 内容评分
        if result["llm_score"] is not None:
            score = result["llm_score"]
            bar = "█" * int(score) + "░" * (10 - int(score))
            print(f"│ 内容评分:     {score:.1f} / 10.0 [{bar}]")
        else:
            print(f"│ 内容评分:     未评估")
        
        print("└─────────────────────────────────────────────────────────┘")
        
        # 转写文本
        if result["transcribed_text"]:
            print("\n【您的回答（转写文本）】")
            print("-"*40)
            print(result["transcribed_text"])
        
        # LLM 反馈
        if result["llm_feedback"]:
            print("\n【评价与建议】")
            print("-"*40)
            # 提取评价部分
            feedback = result["llm_feedback"]
            if "评价:" in feedback or "评价：" in feedback:
                match = re.search(r"评价[：:]\s*(.+)", feedback, re.DOTALL)
                if match:
                    print(match.group(1).strip())
                else:
                    print(feedback)
            else:
                print(feedback)
        
        print("\n" + "="*60)
    
    def run(self):
        """运行交互式评分系统"""
        self.print_header()
        
        # 检查依赖
        if not check_dependencies():
            return
        
        try:
            # 获取输入文件
            transcript_path, task_path = self.get_input_files()
            
            # 读取文件内容
            transcript = self.read_file(transcript_path)
            task = self.read_file(task_path)
            
            # 显示题目
            self.display_question(transcript, task)
            
            # 播放音频提示
            self.play_audio_prompt()
            
            # 初始化模型
            self.initialize_models()
            
            # 录制回答
            audio_path = self.record_answer()
            
            if not audio_path:
                print("\n❌ 录音失败，程序退出")
                return
            
            # 评估
            result = self.evaluate(audio_path, transcript, task)
            
            # 显示结果
            self.display_result(result)
            
            # 询问是否保存
            save = input("\n是否保存结果到文件？(y/n): ").strip().lower()
            if save == 'y':
                output_path = f"result_{int(time.time())}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✅ 结果已保存到: {output_path}")
            
            # 清理临时文件
            if audio_path and os.path.exists(audio_path) and "temp" in audio_path.lower():
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            print("\n感谢使用！再见 👋\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断，程序退出")
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse
    
    # ========== 默认配置（可直接修改此处） ==========
    DEFAULT_WAV2VEC_MODEL = "/root/autodl-tmp/models/checkpoints/best_model.pth"
    DEFAULT_WHISPER_MODEL = "/root/autodl-tmp/models/whisper-base.en"
    DEFAULT_OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_OLLAMA_MODEL = "qwen3:8b"
    DEFAULT_TRANSCRIPT = "test_data/听力原文.txt"
    DEFAULT_TASK = "test_data/任务描述.txt"
    # ================================================
    
    parser = argparse.ArgumentParser(description="交互式语音评分系统")
    parser.add_argument("--wav2vec-model", type=str, default=DEFAULT_WAV2VEC_MODEL,
                        help="wav2vec 模型路径")
    parser.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help="Whisper 模型路径")
    parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST,
                        help="Ollama 服务地址")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL,
                        help="Ollama 模型名称")
    parser.add_argument("--transcript", type=str, default=DEFAULT_TRANSCRIPT,
                        help="听力原文文件路径")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="任务要求文件路径")
    
    args = parser.parse_args()
    
    config = {
        "wav2vec_model_path": args.wav2vec_model,
        "whisper_model_path": args.whisper_model,
        "ollama_host": args.ollama_host,
        "ollama_model": args.ollama_model,
        "default_transcript": args.transcript,
        "default_task": args.task
    }
    
    system = InteractiveSpeechScoring(config)
    system.run()


if __name__ == "__main__":
    main()