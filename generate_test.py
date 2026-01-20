#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据生成工具
================

生成用于测试语音评分系统的模拟数据：
1. 合成 WAV 音频文件（使用 TTS 或生成静音/噪声）
2. 生成任务描述 TXT 文件
3. 生成听力原文 TXT 文件

使用方式：
    python generate_test_data.py --output-dir ./test_data
    
或者在 Python 中导入：
    from generate_test_data import generate_test_data
    paths = generate_test_data(output_dir="./test_data")
"""

import os
import argparse
import struct
import math
import random
from pathlib import Path
from typing import Dict, Optional


# ============================================================================
# 示例数据
# ============================================================================

# 托福/雅思风格的任务描述示例
SAMPLE_TASKS = [
    """Using points and examples from the lecture, explain how the professor describes the relationship between sleep and memory consolidation.""",
    
    """Summarize the points made in the lecture, being sure to explain how they challenge the claims made in the reading passage.""",
    
    """Using the examples from the lecture, explain two ways that animals have adapted to survive in extreme environments.""",
    
    """The professor describes the process of photosynthesis. Explain the main steps involved and their significance.""",
    
    """Using points and details from the lecture, explain the professor's opinion about social media's impact on communication.""",
]

# 听力原文示例
SAMPLE_TRANSCRIPTS = [
    """Today I want to talk about something that affects all of us: sleep and its relationship to memory. 
You might think that sleep is just a time when your brain shuts down, but actually, it's quite the opposite. 
During sleep, especially during what we call REM sleep, your brain is incredibly active. 
It's processing all the information you took in during the day and deciding what to keep and what to discard.

Research has shown that students who get adequate sleep after learning new material perform significantly better on tests 
than those who stay up all night studying. This is because sleep helps consolidate memories - 
it moves information from short-term to long-term storage.

There are two main phases where this happens. First, during deep sleep, your brain replays the day's events, 
strengthening neural connections. Then, during REM sleep, your brain integrates this new information 
with existing knowledge, making creative connections you might not make while awake.""",

    """Let me tell you about some fascinating adaptations that animals have developed to survive in extreme environments.

First, let's consider the Arctic fox. This remarkable creature has several adaptations for the cold. 
Its fur changes color with the seasons - white in winter for camouflage in snow, and brown in summer. 
More importantly, it has an incredibly efficient circulatory system that keeps its core body temperature stable 
even when external temperatures drop to minus 50 degrees Celsius.

Another example is the camel, perfectly adapted for desert life. Contrary to popular belief, 
camels don't store water in their humps - they store fat. This fat can be metabolized for both energy and water. 
Camels can also tolerate body temperature fluctuations that would be dangerous for other mammals, 
reducing the need to sweat and thus conserving water.""",

    """The process of photosynthesis is fundamental to life on Earth. Let me walk you through the main steps.

First, in the light-dependent reactions, chlorophyll in the plant's leaves absorbs sunlight. 
This energy is used to split water molecules into hydrogen and oxygen. 
The oxygen is released as a byproduct - this is actually where most of Earth's atmospheric oxygen comes from.

The hydrogen ions and electrons from water are then used to create ATP and NADPH, 
which are energy-carrying molecules. These molecules move to the second phase: the light-independent reactions, 
also known as the Calvin cycle.

In the Calvin cycle, the plant uses the ATP and NADPH to convert carbon dioxide from the air into glucose. 
This glucose is then used by the plant for energy and growth, or stored as starch for later use.""",
]

# 模拟学生回答（用于参考，实际测试时会用语音转文字的结果）
SAMPLE_STUDENT_ANSWERS = [
    """The professor explains that sleep is very important for memory. During sleep, the brain processes information 
and moves it from short-term to long-term memory. There are two phases: deep sleep and REM sleep. 
In deep sleep, the brain replays events, and in REM sleep, it connects new information with old knowledge.""",
    
    """Animals adapt to extreme environments in different ways. The Arctic fox changes its fur color 
and has a special blood system to stay warm. Camels store fat in their humps for energy and water, 
and they can handle temperature changes that other animals cannot.""",
    
    """Photosynthesis has two main parts. First, plants use sunlight to split water and make oxygen. 
Then they use the energy to turn carbon dioxide into glucose in the Calvin cycle. 
Plants use glucose for energy and growth.""",
]


# ============================================================================
# WAV 文件生成工具
# ============================================================================

def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 16000, 
                       amplitude: float = 0.5) -> bytes:
    """
    生成正弦波音频数据
    
    Args:
        frequency: 频率 (Hz)
        duration: 持续时间 (秒)
        sample_rate: 采样率
        amplitude: 振幅 (0.0-1.0)
    
    Returns:
        音频数据字节
    """
    num_samples = int(sample_rate * duration)
    audio_data = []
    
    for i in range(num_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency * t)
        # 转换为 16-bit PCM
        sample = int(value * 32767)
        sample = max(-32768, min(32767, sample))
        audio_data.append(struct.pack('<h', sample))
    
    return b''.join(audio_data)


def generate_speech_like_audio(duration: float, sample_rate: int = 16000) -> bytes:
    """
    生成类似语音的音频（多个频率叠加 + 调制）
    
    Args:
        duration: 持续时间 (秒)
        sample_rate: 采样率
    
    Returns:
        音频数据字节
    """
    num_samples = int(sample_rate * duration)
    audio_data = []
    
    # 人声基频范围
    base_frequencies = [150, 200, 250, 300]  # 模拟人声的基频
    
    for i in range(num_samples):
        t = i / sample_rate
        value = 0
        
        # 叠加多个频率
        for j, freq in enumerate(base_frequencies):
            # 添加频率调制（模拟语音变化）
            freq_mod = freq * (1 + 0.1 * math.sin(2 * math.pi * 3 * t))
            # 添加幅度调制（模拟语音节奏）
            amp_mod = 0.3 * (1 + 0.5 * math.sin(2 * math.pi * 4 * t))
            value += amp_mod * math.sin(2 * math.pi * freq_mod * t) / len(base_frequencies)
        
        # 添加一些随机噪声（模拟气息声）
        value += random.uniform(-0.05, 0.05)
        
        # 转换为 16-bit PCM
        sample = int(value * 32767 * 0.8)
        sample = max(-32768, min(32767, sample))
        audio_data.append(struct.pack('<h', sample))
    
    return b''.join(audio_data)


def generate_silence(duration: float, sample_rate: int = 16000) -> bytes:
    """
    生成静音音频
    
    Args:
        duration: 持续时间 (秒)
        sample_rate: 采样率
    
    Returns:
        音频数据字节
    """
    num_samples = int(sample_rate * duration)
    return b'\x00\x00' * num_samples


def create_wav_file(audio_data: bytes, output_path: str, 
                    sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16):
    """
    创建 WAV 文件
    
    Args:
        audio_data: 音频数据字节
        output_path: 输出文件路径
        sample_rate: 采样率
        channels: 声道数
        bits_per_sample: 位深度
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(audio_data)
    file_size = 36 + data_size
    
    with open(output_path, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', file_size))
        f.write(b'WAVE')
        
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # audio format (PCM)
        f.write(struct.pack('<H', channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))
        
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(audio_data)


# ============================================================================
# 测试数据生成函数
# ============================================================================

def generate_test_data(
    output_dir: str = "./test_data",
    audio_duration: float = 10.0,
    audio_type: str = "speech_like",
    task_index: int = 0,
    sample_rate: int = 16000,
    custom_task: Optional[str] = None,
    custom_transcript: Optional[str] = None
) -> Dict[str, str]:
    """
    生成完整的测试数据集
    
    Args:
        output_dir: 输出目录
        audio_duration: 音频时长（秒）
        audio_type: 音频类型 ("speech_like", "sine", "silence")
        task_index: 使用第几个示例任务 (0-4)
        sample_rate: 采样率
        custom_task: 自定义任务描述（可选）
        custom_transcript: 自定义听力原文（可选）
    
    Returns:
        包含生成文件路径的字典
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件路径
    audio_path = output_dir / "test_audio.wav"
    task_path = output_dir / "任务描述.txt"
    transcript_path = output_dir / "听力原文.txt"
    
    print(f"\n{'='*60}")
    print("生成测试数据")
    print(f"{'='*60}\n")
    
    # 1. 生成音频文件
    print(f"[1/3] 生成音频文件...")
    print(f"  类型: {audio_type}")
    print(f"  时长: {audio_duration} 秒")
    print(f"  采样率: {sample_rate} Hz")
    
    if audio_type == "speech_like":
        audio_data = generate_speech_like_audio(audio_duration, sample_rate)
    elif audio_type == "sine":
        audio_data = generate_sine_wave(440, audio_duration, sample_rate)
    elif audio_type == "silence":
        audio_data = generate_silence(audio_duration, sample_rate)
    else:
        raise ValueError(f"未知音频类型: {audio_type}")
    
    create_wav_file(audio_data, str(audio_path), sample_rate)
    print(f"  ✓ 已保存: {audio_path}")
    
    # 2. 生成任务描述
    print(f"\n[2/3] 生成任务描述...")
    task_text = custom_task if custom_task else SAMPLE_TASKS[task_index % len(SAMPLE_TASKS)]
    with open(task_path, 'w', encoding='utf-8') as f:
        f.write(task_text)
    print(f"  ✓ 已保存: {task_path}")
    print(f"  内容预览: {task_text[:80]}...")
    
    # 3. 生成听力原文
    print(f"\n[3/3] 生成听力原文...")
    transcript_text = custom_transcript if custom_transcript else SAMPLE_TRANSCRIPTS[task_index % len(SAMPLE_TRANSCRIPTS)]
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    print(f"  ✓ 已保存: {transcript_path}")
    print(f"  内容预览: {transcript_text[:80]}...")
    
    # 返回路径
    result = {
        "audio_path": str(audio_path),
        "task_path": str(task_path),
        "transcript_path": str(transcript_path),
        "output_dir": str(output_dir)
    }
    
    print(f"\n{'='*60}")
    print("测试数据生成完成！")
    print(f"{'='*60}")
    print(f"\n文件列表:")
    print(f"  1. 音频文件: {audio_path}")
    print(f"  2. 任务描述: {task_path}")
    print(f"  3. 听力原文: {transcript_path}")
    
    print(f"\n使用示例:")
    print(f"  python speech_scoring_system.py \\")
    print(f"      --audio {audio_path} \\")
    print(f"      --transcript {transcript_path} \\")
    print(f"      --task {task_path}")
    
    return result


def generate_multiple_test_sets(
    output_dir: str = "./test_data",
    num_sets: int = 3,
    audio_duration: float = 10.0
) -> list:
    """
    生成多组测试数据
    
    Args:
        output_dir: 输出根目录
        num_sets: 生成组数
        audio_duration: 每个音频的时长
    
    Returns:
        所有生成文件路径的列表
    """
    results = []
    output_dir = Path(output_dir)
    
    for i in range(num_sets):
        set_dir = output_dir / f"test_set_{i+1:02d}"
        result = generate_test_data(
            output_dir=str(set_dir),
            audio_duration=audio_duration,
            task_index=i,
            audio_type="speech_like"
        )
        results.append(result)
    
    print(f"\n\n{'='*60}")
    print(f"共生成 {num_sets} 组测试数据")
    print(f"{'='*60}")
    
    return results


# ============================================================================
# 使用真实 TTS 生成音频（可选，需要额外依赖）
# ============================================================================

def generate_tts_audio(
    text: str,
    output_path: str,
    engine: str = "gtts"
) -> bool:
    """
    使用 TTS 引擎生成真实语音（需要网络和额外依赖）
    
    Args:
        text: 要转换的文本
        output_path: 输出文件路径
        engine: TTS 引擎 ("gtts", "pyttsx3")
    
    Returns:
        是否成功
    """
    if engine == "gtts":
        try:
            from gtts import gTTS
            import subprocess
            
            # 生成 MP3
            mp3_path = output_path.replace('.wav', '.mp3')
            tts = gTTS(text=text, lang='en')
            tts.save(mp3_path)
            
            # 转换为 WAV（需要 ffmpeg）
            subprocess.run([
                'ffmpeg', '-i', mp3_path, 
                '-ar', '16000', '-ac', '1',
                '-y', output_path
            ], check=True, capture_output=True)
            
            # 删除临时 MP3
            os.remove(mp3_path)
            
            print(f"  ✓ TTS 音频已生成: {output_path}")
            return True
            
        except ImportError:
            print("  ✗ 需要安装 gtts: pip install gtts")
            return False
        except Exception as e:
            print(f"  ✗ TTS 生成失败: {e}")
            return False
    
    elif engine == "pyttsx3":
        try:
            import pyttsx3
            import wave
            
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            print(f"  ✓ TTS 音频已生成: {output_path}")
            return True
            
        except ImportError:
            print("  ✗ 需要安装 pyttsx3: pip install pyttsx3")
            return False
        except Exception as e:
            print(f"  ✗ TTS 生成失败: {e}")
            return False
    
    else:
        print(f"  ✗ 未知 TTS 引擎: {engine}")
        return False


def generate_test_data_with_tts(
    output_dir: str = "./test_data",
    task_index: int = 0,
    answer_text: Optional[str] = None
) -> Dict[str, str]:
    """
    使用 TTS 生成包含真实语音的测试数据
    
    Args:
        output_dir: 输出目录
        task_index: 任务索引
        answer_text: 自定义学生回答文本（用于 TTS）
    
    Returns:
        包含生成文件路径的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_path = output_dir / "test_audio.wav"
    task_path = output_dir / "任务描述.txt"
    transcript_path = output_dir / "听力原文.txt"
    
    print(f"\n{'='*60}")
    print("生成测试数据 (TTS 模式)")
    print(f"{'='*60}\n")
    
    # 1. 生成任务描述
    print(f"[1/3] 生成任务描述...")
    task_text = SAMPLE_TASKS[task_index % len(SAMPLE_TASKS)]
    with open(task_path, 'w', encoding='utf-8') as f:
        f.write(task_text)
    print(f"  ✓ 已保存: {task_path}")
    
    # 2. 生成听力原文
    print(f"\n[2/3] 生成听力原文...")
    transcript_text = SAMPLE_TRANSCRIPTS[task_index % len(SAMPLE_TRANSCRIPTS)]
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    print(f"  ✓ 已保存: {transcript_path}")
    
    # 3. 使用 TTS 生成音频
    print(f"\n[3/3] 使用 TTS 生成音频...")
    text_to_speak = answer_text if answer_text else SAMPLE_STUDENT_ANSWERS[task_index % len(SAMPLE_STUDENT_ANSWERS)]
    
    tts_success = generate_tts_audio(text_to_speak, str(audio_path), engine="gtts")
    
    if not tts_success:
        print("  降级为合成音频...")
        audio_data = generate_speech_like_audio(10.0, 16000)
        create_wav_file(audio_data, str(audio_path), 16000)
        print(f"  ✓ 已保存合成音频: {audio_path}")
    
    return {
        "audio_path": str(audio_path),
        "task_path": str(task_path),
        "transcript_path": str(transcript_path),
        "output_dir": str(output_dir)
    }


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成语音评分系统的测试数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成单组测试数据
  python generate_test_data.py --output-dir ./test_data
  
  # 生成多组测试数据
  python generate_test_data.py --output-dir ./test_data --num-sets 5
  
  # 指定音频时长和类型
  python generate_test_data.py --output-dir ./test_data --duration 15 --audio-type sine
  
  # 使用 TTS 生成真实语音（需要安装 gtts）
  python generate_test_data.py --output-dir ./test_data --use-tts
        """
    )
    
    parser.add_argument("--output-dir", type=str, default="./test_data",
                        help="输出目录 (默认: ./test_data)")
    parser.add_argument("--num-sets", type=int, default=1,
                        help="生成测试集数量 (默认: 1)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="音频时长（秒）(默认: 10)")
    parser.add_argument("--audio-type", type=str, default="speech_like",
                        choices=["speech_like", "sine", "silence"],
                        help="音频类型 (默认: speech_like)")
    parser.add_argument("--task-index", type=int, default=0,
                        help="任务索引 0-4 (默认: 0)")
    parser.add_argument("--use-tts", action="store_true",
                        help="使用 TTS 生成真实语音")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="采样率 (默认: 16000)")
    
    args = parser.parse_args()
    
    if args.num_sets > 1:
        # 生成多组测试数据
        generate_multiple_test_sets(
            output_dir=args.output_dir,
            num_sets=args.num_sets,
            audio_duration=args.duration
        )
    elif args.use_tts:
        # 使用 TTS 生成
        generate_test_data_with_tts(
            output_dir=args.output_dir,
            task_index=args.task_index
        )
    else:
        # 生成单组测试数据
        generate_test_data(
            output_dir=args.output_dir,
            audio_duration=args.duration,
            audio_type=args.audio_type,
            task_index=args.task_index,
            sample_rate=args.sample_rate
        )
    
    return 0


if __name__ == "__main__":
    exit(main())