#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音评分整合系统
================

功能：
1. 对考生语音进行 wav2vec 模型打分（语音质量评估）
2. 将语音转换为文字（使用 faster-whisper）
3. 结合听力原文和任务要求，使用大模型对内容进行打分

输入：
- wav_path: 考生语音文件路径 (.wav/.mp3/.flac)
- transcript_path: 听力原文文件路径 (.txt)
- task_path: 任务要求文件路径 (.txt)

输出：
- audio_score: 语音打分 (1.0-6.0)
- transcribed_text: 转文字结果
- llm_score: 大模型打分 (0-10)

使udio用方式：
    python speech_scoring_system.py \
        --audio /path/to/audio.wav \
        --transcript /path/to/transcript.txt \
        --task /path/to/task.txt

或者作为模块导入：
    from speech_scoring_system import SpeechScoringSystem
    
    system = SpeechScoringSystem(
        wav2vec_model_path="/path/to/wav2vec/best_model.pth",
        whisper_model_path="/path/to/whisper-base.en",
        ollama_host="http://localhost:11434",
        ollama_model="qwen3:8b"
    )
    result = system.evaluate(wav_path, transcript_path, task_path)
"""

from __future__ import annotations

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchaudio


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ScoringResult:
    """评分结果数据类"""
    audio_score: float          # 语音质量评分 (1.0-6.0)
    transcribed_text: str       # 语音转文字结果
    llm_score: float            # 大模型内容评分 (0-10)
    audio_score_detail: Optional[Dict[str, Any]] = None  # 语音评分详情
    llm_score_detail: Optional[str] = None               # LLM 原始输出
    
    def to_dict(self) -> Dict:
        return {
            "audio_score": self.audio_score,
            "transcribed_text": self.transcribed_text,
            "llm_score": self.llm_score,
            "audio_score_detail": self.audio_score_detail,
            "llm_score_detail": self.llm_score_detail
        }
    
    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"语音评分结果\n"
            f"{'='*60}\n"
            f"1. 语音质量评分: {self.audio_score:.2f} / 6.0\n"
            f"2. 转文字结果:\n   {self.transcribed_text[:200]}{'...' if len(self.transcribed_text) > 200 else ''}\n"
            f"3. 内容评分 (LLM): {self.llm_score:.1f} / 10.0\n"
            f"{'='*60}"
        )


# ============================================================================
# 模块1: CNN-LSTM 语音评分模型 (基于 wav2vec 特征)
# ============================================================================

class CNN_LSTM_Regressor(nn.Module):
    """
    CNN-LSTM 回归模型 - 用于语音质量评分
    
    架构:
        Input: Mel-Spectrogram [batch, 1, n_mels, time]
        ↓
        CNN 层 (提取局部特征)
        ↓
        LSTM 层 (时序建模)
        ↓
        全局池化
        ↓
        全连接层 (回归)
        ↓
        Output: Score [batch, 1]
    """
    
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
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        lstm_output_size = lstm_hidden_size * 2
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # CNN 特征提取
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # 重塑为 LSTM 输入格式
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * freq)
        
        # LSTM 时序建模
        lstm_out, _ = self.lstm(x)
        
        # 全局池化
        pooled = torch.mean(lstm_out, dim=1)
        
        # 回归输出
        score = self.fc(pooled)
        
        return score


class AudioScorer:
    """
    语音质量评分器
    使用预训练的 CNN-LSTM 模型对语音进行评分
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 sample_rate: int = 16000,
                 max_length: int = 30,
                 n_mels: int = 128,
                 n_fft: int = 400,
                 hop_length: int = 160):
        """
        Args:
            model_path: 模型权重文件路径
            device: 设备 ("auto", "cuda", "cpu")
            sample_rate: 采样率
            max_length: 最大音频长度（秒）
            n_mels: Mel 频率数量
            n_fft: FFT 窗口大小
            hop_length: 帧移
        """
        # 设备设置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 音频参数
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = sample_rate * max_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Mel-Spectrogram 变换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        print(f"[AudioScorer] 模型已加载，设备: {self.device}")
    
    def _load_model(self, model_path: str) -> CNN_LSTM_Regressor:
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = CNN_LSTM_Regressor(
            n_mels=self.n_mels,
            cnn_channels=[32, 64, 128],
            lstm_hidden_size=128,
            lstm_num_layers=2,
            lstm_dropout=0.3,
            fc_hidden_size=128,
            fc_dropout=0.3
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """预处理音频文件"""
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 重采样
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 裁剪或填充到固定长度
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            pad_length = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        # 提取 Mel-Spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # 归一化
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db
    
    def score(self, audio_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        对音频进行评分
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            score: 评分 (1.0-6.0)
            detail: 详细信息
        """
        # 预处理
        mel_spec_db = self.preprocess_audio(audio_path)
        mel_spec_db = mel_spec_db.unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(mel_spec_db)
            score = output.item()
        
        # 限制在有效范围内
        score = max(1.0, min(6.0, score))
        
        detail = {
            "raw_score": output.item(),
            "clamped_score": score,
            "device": str(self.device)
        }
        
        return score, detail


# ============================================================================
# 模块2: 语音转文字 (faster-whisper)
# ============================================================================

class SpeechToText:
    """
    语音转文字模块
    使用 faster-whisper 进行音频转写
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 compute_type: str = "float16",
                 language: str = "en"):
        """
        Args:
            model_path: Whisper 模型路径
            device: 设备 ("auto", "cuda", "cpu")
            compute_type: 计算类型 ("float16", "int8", "float32")
            language: 语言代码
        """
        self.model_path = model_path
        self.language = language
        
        # 设置设备和计算类型
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                self.compute_type = compute_type
            else:
                self.device = "cpu"
                self.compute_type = "int8"
        else:
            self.device = device
            self.compute_type = compute_type if device != "cpu" else "int8"
        
        # 加载模型
        self.model = self._load_model()
        
        print(f"[SpeechToText] 模型已加载，设备: {self.device}, 计算类型: {self.compute_type}")
    
    def _load_model(self):
        """加载 Whisper 模型"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("请安装 faster-whisper: pip install faster-whisper")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Whisper 模型路径不存在: {self.model_path}")
        
        model = WhisperModel(
            model_size_or_path=self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            download_root=None
        )
        
        return model
    
    def transcribe(self, audio_path: str) -> str:
        """
        转写音频文件
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            transcribed_text: 转写文本
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        try:
            segments, _ = self.model.transcribe(
                audio_path,
                language=self.language,
                beam_size=4,
                vad_filter=True,
                vad_parameters={"threshold": 0.6},
                without_timestamps=True
            )
            
            transcribed_text = " ".join([seg.text.strip() for seg in segments]).strip()
            return transcribed_text
            
        except Exception as e:
            print(f"[SpeechToText] 转写失败: {e}")
            return ""


# ============================================================================
# 模块3: LLM 内容评分 (Ollama)
# ============================================================================

class LLMScorer:
    """
    LLM 内容评分器
    使用 Ollama 对学生回答进行内容评分
    """
    
    PROMPT_TEMPLATE = """You are an expert English evaluator. I will provide a question, a reference passage, and a student's answer.
Please score the student's answer from 0 to 10 based on:
1. Relevance to the question.
2. Whether the student correctly uses information from the reference passage.
3. Completeness and clarity of logic.
4. Use of important keywords/ideas from the reference (do not reward simple copying).

Here are the inputs:
Question: {question}
Reference Passage: {reference}
Student Answer: {answer}

Please provide the score (a number from 0 to 10).
"""
    
    def __init__(self, 
                 host: str = "http://localhost:11434",
                 model: str = "qwen3:8b",
                 timeout: int = 180):
        """
        Args:
            host: Ollama 服务地址
            model: 模型名称
            timeout: 请求超时时间（秒）
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        
        print(f"[LLMScorer] Ollama 配置 - Host: {self.host}, Model: {self.model}")
    
    @staticmethod
    def normalize_whitespace(s: str) -> str:
        """规范化空白字符"""
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n"))
        return s.strip()
    
    def build_prompt(self, question: str, reference: str, answer: str) -> str:
        """构建评分 prompt"""
        return self.PROMPT_TEMPLATE.format(
            question=self.normalize_whitespace(question),
            reference=self.normalize_whitespace(reference),
            answer=self.normalize_whitespace(answer)
        )
    
    def _call_ollama(self, prompt: str) -> str:
        """调用 Ollama API"""
        try:
            import requests
        except ImportError:
            raise ImportError("请安装 requests: pip install requests")
        
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 500,
            },
        }
        
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API 调用失败: {e}")
    
    @staticmethod
    def extract_score(text: str) -> float:
        """从模型输出中提取分数"""
        t = text.strip()
        m = re.search(r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", t)
        if not m:
            raise ValueError(f"无法从模型输出中提取分数: {text!r}")
        
        score_str = m.group(1)
        v = float(score_str)
        if v < 0 or v > 10:
            raise ValueError(f"分数超出范围 0-10: {v}")
        
        return v
    
    def score(self, question: str, reference: str, answer: str) -> Tuple[float, str]:
        """
        对学生回答进行评分
        
        Args:
            question: 任务/问题
            reference: 听力原文/参考文本
            answer: 学生回答
        
        Returns:
            score: 评分 (0-10)
            raw_output: 模型原始输出
        """
        prompt = self.build_prompt(question, reference, answer)
        raw_output = self._call_ollama(prompt)
        
        try:
            score = self.extract_score(raw_output)
        except ValueError as e:
            print(f"[LLMScorer] 评分提取失败: {e}")
            score = 5.0  # 默认中间分数
        
        return score, raw_output


# ============================================================================
# 主系统类
# ============================================================================

class SpeechScoringSystem:
    """
    语音评分整合系统
    
    整合三个模块：
    1. AudioScorer: 语音质量评分
    2. SpeechToText: 语音转文字
    3. LLMScorer: LLM 内容评分
    """
    
    def __init__(self,
                 wav2vec_model_path: str,
                 whisper_model_path: str,
                 ollama_host: str = "http://localhost:11434",
                 ollama_model: str = "qwen3:8b",
                 device: str = "auto"):
        """
        Args:
            wav2vec_model_path: wav2vec 模型路径
            whisper_model_path: Whisper 模型路径
            ollama_host: Ollama 服务地址
            ollama_model: Ollama 模型名称
            device: 设备 ("auto", "cuda", "cpu")
        """
        print("\n" + "="*60)
        print("初始化语音评分系统")
        print("="*60 + "\n")
        
        # 初始化各模块
        print("[1/3] 加载语音评分模型...")
        self.audio_scorer = AudioScorer(
            model_path=wav2vec_model_path,
            device=device
        )
        
        print("[2/3] 加载语音转文字模型...")
        self.speech_to_text = SpeechToText(
            model_path=whisper_model_path,
            device=device
        )
        
        print("[3/3] 初始化 LLM 评分器...")
        self.llm_scorer = LLMScorer(
            host=ollama_host,
            model=ollama_model
        )
        
        print("\n" + "="*60)
        print("系统初始化完成！")
        print("="*60 + "\n")
    
    @staticmethod
    def read_text_file(file_path: str) -> str:
        """读取文本文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    
    def evaluate(self, 
                 audio_path: str, 
                 transcript_path: str, 
                 task_path: str) -> ScoringResult:
        """
        对考生语音进行完整评估
        
        Args:
            audio_path: 考生语音文件路径
            transcript_path: 听力原文文件路径
            task_path: 任务要求文件路径
        
        Returns:
            ScoringResult: 评分结果
        """
        print("\n" + "-"*60)
        print("开始评估...")
        print("-"*60)
        
        # 验证文件存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 读取文本文件
        transcript_text = self.read_text_file(transcript_path)
        task_text = self.read_text_file(task_path)
        
        print(f"\n[输入信息]")
        print(f"  音频文件: {audio_path}")
        print(f"  听力原文: {len(transcript_text)} 字符")
        print(f"  任务要求: {len(task_text)} 字符")
        
        # Step 1: 语音质量评分
        print("\n[Step 1/3] 语音质量评分...")
        audio_score, audio_detail = self.audio_scorer.score(audio_path)
        print(f"  ✓ 语音评分: {audio_score:.2f} / 6.0")
        
        # Step 2: 语音转文字
        print("\n[Step 2/3] 语音转文字...")
        transcribed_text = self.speech_to_text.transcribe(audio_path)
        print(f"  ✓ 转写完成: {len(transcribed_text)} 字符")
        if transcribed_text:
            preview = transcribed_text[:100] + "..." if len(transcribed_text) > 100 else transcribed_text
            print(f"  预览: {preview}")
        
        # Step 3: LLM 内容评分
        print("\n[Step 3/3] LLM 内容评分...")
        if transcribed_text:
            llm_score, llm_raw = self.llm_scorer.score(
                question=task_text,
                reference=transcript_text,
                answer=transcribed_text
            )
        else:
            llm_score = 0.0
            llm_raw = "转写失败，无法评分"
        print(f"  ✓ 内容评分: {llm_score:.1f} / 10.0")
        print("suggesions")
        print(llm_raw)
        
        # 构建结果
        result = ScoringResult(
            audio_score=audio_score,
            transcribed_text=transcribed_text,
            llm_score=llm_score,
            audio_score_detail=audio_detail,
            llm_score_detail=llm_raw
        )
        
        print("\n" + "-"*60)
        print("评估完成！")
        print("-"*60)
        
        return result
    
    def evaluate_batch(self,
                       audio_paths: list,
                       transcript_path: str,
                       task_path: str,
                       output_json: str = None) -> list:
        """
        批量评估多个音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            transcript_path: 听力原文文件路径（所有音频共用）
            task_path: 任务要求文件路径（所有音频共用）
            output_json: 输出 JSON 文件路径（可选）
        
        Returns:
            results: 评分结果列表
        """
        results = []
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}] 处理: {os.path.basename(audio_path)}")
            
            try:
                result = self.evaluate(audio_path, transcript_path, task_path)
                results.append({
                    "audio_path": audio_path,
                    "status": "success",
                    "result": result.to_dict()
                })
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                results.append({
                    "audio_path": audio_path,
                    "status": "error",
                    "error": str(e)
                })
        
        # 保存结果
        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_json}")
        
        return results


# ============================================================================
# 简化版评估函数（不需要实例化类）
# ============================================================================

def quick_evaluate(
    audio_path: str,
    transcript_path: str,
    task_path: str,
    wav2vec_model_path: str,
    whisper_model_path: str,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "qwen3:8b"
) -> ScoringResult:
    """
    快速评估单个音频文件
    
    Args:
        audio_path: 考生语音文件路径
        transcript_path: 听力原文文件路径
        task_path: 任务要求文件路径
        wav2vec_model_path: wav2vec 模型路径
        whisper_model_path: Whisper 模型路径
        ollama_host: Ollama 服务地址
        ollama_model: Ollama 模型名称
    
    Returns:
        ScoringResult: 评分结果
    """
    system = SpeechScoringSystem(
        wav2vec_model_path=wav2vec_model_path,
        whisper_model_path=whisper_model_path,
        ollama_host=ollama_host,
        ollama_model=ollama_model
    )
    
    return system.evaluate(audio_path, transcript_path, task_path)


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="语音评分整合系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python speech_scoring_system.py \\
      --audio /path/to/audio.wav \\
      --transcript /path/to/transcript.txt \\
      --task /path/to/task.txt \\
      --wav2vec-model /path/to/wav2vec/best_model.pth \\
      --whisper-model /path/to/whisper-base.en

输出示例:
  1. 语音质量评分: 4.25 / 6.0
  2. 转文字结果: "The student's response..."
  3. 内容评分 (LLM): 7.5 / 10.0
        """
    )
    
    # 必需参数
    parser.add_argument("--audio",default="test_data/test_audio.wav", type=str,
                        help="考生语音文件路径")
    parser.add_argument("--transcript",default="test_data/听力原文.txt", type=str,
                        help="听力原文文件路径")
    parser.add_argument("--task",default="test_data/任务描述.txt", type=str,
                        help="任务要求文件路径")
    
    # 模型路径
    parser.add_argument("--wav2vec-model", type=str, 
                        default="/root/autodl-tmp/models/checkpoints/best_model.pth",
                        help="wav2vec 模型路径")
    parser.add_argument("--whisper-model", type=str,
                        default="/root/autodl-tmp/models/whisper-base.en",
                        help="Whisper 模型路径")
    
    # Ollama 配置
    parser.add_argument("--ollama-host", type=str,
                        default="http://localhost:11434",
                        help="Ollama 服务地址")
    parser.add_argument("--ollama-model", type=str,
                        default="qwen3:8b",
                        help="Ollama 模型名称")
    
    # 输出选项
    parser.add_argument("--output-json", type=str,
                        help="输出 JSON 文件路径（可选）")
    
    args = parser.parse_args()
    
    # 创建系统并评估
    try:
        system = SpeechScoringSystem(
            wav2vec_model_path=args.wav2vec_model,
            whisper_model_path=args.whisper_model,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model
        )
        
        result = system.evaluate(
            audio_path=args.audio,
            transcript_path=args.transcript,
            task_path=args.task
        )
        
        # 打印结果
        print(result)
        
        # 保存 JSON（如果指定）
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output_json}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())