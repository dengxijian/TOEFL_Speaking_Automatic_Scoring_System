#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用 Ollama 批量对考生转写打分（只输出0-10）。

目录示例：
050/
  任务描述.txt
  听力原文.txt
  考生语音转文本/
    1_01.mp3.txt
    1_01.mp3_评分.txt   (可选：脚本默认跳过“_评分.txt”文件)

依赖：
  pip install requests
Ollama：
  确保 ollama serve 正在运行，并且已 pull 对应模型（如 gemma2:9b / qwen2.5:7b 等）
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import requests


# -----------------------------
# 1) 改这里：数据目录（050这一层）
# -----------------------------
DATA_DIR = Path("../autodl_tmp/candidate_text/050")

TASK_FILE = "任务描述.txt"
REF_FILE = "听力原文.txt"
STUDENT_DIR = "考生语音转文本"


# -----------------------------
# 2) 改这里：Ollama 配置
# -----------------------------
OLLAMA_HOST = "http://localhost:11434"  # 服务器上跑就改成 http://<ip>:11434
OLLAMA_MODEL = "modelscope.cn/Qwen/Qwen3-8B-GGUF:latest"             # 你本地实际存在的模型名，如 "gemma2:9b"


PROMPT_TEMPLATE = """You are an expert English evaluator. I will provide a question, a reference passage, and a student's answer.
Please score the student's answer from 0 to 10 based on:
1. Relevance to the question.
2. Whether the student correctly uses information from the reference passage.
3. Completeness and clarity of logic.
4. Use of important keywords/ideas from the reference (do not reward simple copying).

Output:
- A numerical score.
- A short explanation of strengths and weaknesses.
- Which key points from the reference were correctly covered and which were missing.

Here are the inputs:
Question: {question}
Reference Passage: {reference}
Student Answer: {answer}

Please provide the score only.
"""


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n"))
    return s.strip()


def list_student_answer_files(student_dir: Path) -> List[Path]:
    files = sorted(student_dir.glob("*.mp3.txt"))
    files = [p for p in files if not p.name.endswith("_评分.txt")]
    return files


def build_prompt(question: str, reference: str, answer: str) -> str:
    return PROMPT_TEMPLATE.format(
        question=normalize_whitespace(question),
        reference=normalize_whitespace(reference),
        answer=normalize_whitespace(answer),
    )


def ollama_generate(prompt: str) -> str:
    """
    调用 Ollama /api/generate，返回 response 文本。
    """
    url = OLLAMA_HOST.rstrip("/") + "/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # 这些参数用于降低“输出多余解释”的概率
        "options": {
            "temperature": 0,
            "num_predict": 16,  # 只需要一个分数，输出越短越好
        },
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def extract_score_only(text: str) -> str:
    """
    从模型输出中提取 0-10 分数（支持 7 / 7.5 / 7/10 / Score: 7）
    """
    t = text.strip()
    m = re.search(r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", t)
    if not m:
        raise ValueError(f"无法从模型输出中提取分数: {text!r}")

    score_str = m.group(1)
    v = float(score_str)
    if v < 0 or v > 10:
        raise ValueError(f"分数超出范围0-10: {v}")

    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return score_str


def main():
    task_path = DATA_DIR / TASK_FILE
    ref_path = DATA_DIR / REF_FILE
    student_dir = DATA_DIR / STUDENT_DIR

    question = read_text(task_path)
    reference = read_text(ref_path)

    if not student_dir.exists():
        raise FileNotFoundError(f"目录不存在: {student_dir}")

    answer_files = list_student_answer_files(student_dir)
    if not answer_files:
        raise FileNotFoundError(f"未找到考生转写文件: {student_dir}/**/*.mp3.txt")

    out_path = DATA_DIR / "scores.jsonl"

    for ans_file in answer_files:
        answer = read_text(ans_file)
        prompt = build_prompt(question, reference, answer)

        raw = ollama_generate(prompt)
        score = extract_score_only(raw)

        # 终端只打印分数（满足“Please provide the score only.”）
        print(score)

        # 同步落盘记录（便于对齐文件与分数）
        record = {
            "file": str(ans_file.relative_to(DATA_DIR)),
            "score": score,
            "raw_model_output": raw,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
