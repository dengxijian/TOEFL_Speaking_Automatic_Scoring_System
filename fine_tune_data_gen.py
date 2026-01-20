import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from modelscope import snapshot_download
except ImportError:
    snapshot_download = None  # 仅当需要下载模型时才依赖 modelscope

ENCODINGS = ("utf-8", "utf-8-sig", "gbk")

def read_text(path: Path) -> str:
    """容错读取文本文件，去掉首尾空白"""
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def iter_problem_dirs(root: Path) -> Iterable[Path]:
    """找到含有考生语音转文本子目录的题目文件夹"""
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "考生语音转文本").exists():
            yield p

def build_entry(
    problem_dir: Path,
    base_name: str,
    task_text: str,
    reference_text: str,
    answer_text: str,
    score_text: str,
) -> Dict[str, object]:
    """构建一条微调样本"""
    input_text = (
        f"任务描述：{task_text}\n"
        f"听力原文：{reference_text}\n"
        f"考生回答：{answer_text}"
    ).strip()

    return {
        "instruction": "评分",
        "input": input_text,
        "output": score_text.strip(),
        "meta": {
            "problem_id": problem_dir.name,
            "audio_file": base_name,
        },
    }

def load_dataset(data_root: Path) -> List[Dict[str, object]]:
    """按文件结构加载全部评分样本"""
    samples: List[Dict[str, object]] = []
    for problem_dir in iter_problem_dirs(data_root):
        task_file = problem_dir / "任务描述.txt"
        reference_file = problem_dir / "听力原文.txt"
        student_dir = problem_dir / "考生语音转文本"

        if not task_file.exists() or not reference_file.exists():
            print(f"[skip] 缺少任务或听力文件: {problem_dir}")
            continue

        task_text = read_text(task_file)
        reference_text = read_text(reference_file)

        for score_file in sorted(student_dir.glob("*_评分.txt")):
            base = score_file.name.removesuffix("_评分.txt")
            answer_file = student_dir / f"{base}.txt"
            if not answer_file.exists():
                print(f"[skip] 找不到对应转写: {answer_file}")
                continue

            answer_text = read_text(answer_file)
            score_text = read_text(score_file)
            samples.append(
                build_entry(
                    problem_dir,
                    base,
                    task_text,
                    reference_text,
                    answer_text,
                    score_text,
                )
            )
    return samples

def save_jsonl(samples: List[Dict[str, object]], output: Path) -> None:
    """保存为 JSONL"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[done] 写入 {len(samples)} 条 -> {output}")


def maybe_download_model(args: argparse.Namespace) -> Optional[str]:
    """可选模型下载，避免默认阻塞"""
    if not args.download_model:
        return None
    if snapshot_download is None:
        raise RuntimeError("需要先安装 modelscope 才能下载模型")
    model_dir = snapshot_download(
        args.model_id,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    print(f"[model] 下载完成: {model_dir}")
    return model_dir



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建评分微调数据集")
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/candidate_text",
        type=Path,
        help="数据集根目录（包含001/002/...题目文件夹）",
    )
    parser.add_argument(
        "--output",
        default="micro_tune_dataset.jsonl",
        type=Path,
        help="JSONL 输出路径",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="需要时下载 Qwen 模型（可选）",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="master")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_download_model(args)
    samples = load_dataset(args.data_root)
    save_jsonl(samples, args.output)


if __name__ == "__main__":
    main()