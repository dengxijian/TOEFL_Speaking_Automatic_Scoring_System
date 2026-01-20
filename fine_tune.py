#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

PROMPT_TPL = "Instruction: {instruction}\nInput:\n{inp}\nAnswer:"


def build_prompt(example: Dict) -> str:
    return PROMPT_TPL.format(instruction=example["instruction"], inp=example["input"])


def load_ds(path: Path) -> Dataset:
    return Dataset.from_json(str(path))


def preprocess_dataset(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    def _convert(example: Dict) -> Dict:
        prompt = build_prompt(example)
        answer = example["output"] + tokenizer.eos_token
        full_text = prompt + " " + answer

        tokenized = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), len(tokenized["input_ids"]))

        labels: List[int] = tokenized["input_ids"].copy()
        for i in range(prompt_len):
            labels[i] = -100  # mask prompt part

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    return ds.map(_convert, remove_columns=ds.column_names)


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def setup_wandb(args: argparse.Namespace) -> Optional[object]:
    """可选初始化 wandb，未安装时给出提示。"""
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "wandb 未安装，运行 pip install wandb 或去掉 --wandb 开关。"
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config={
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    )
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen 0.6B on scoring data.")
    parser.add_argument("--data", type=Path, default=Path("micro_tune_dataset.jsonl"))
    parser.add_argument("--model-id", type=str, default="/root/autodl-tmp/Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/Qwen/qwen-0.6b-lora1"))
    parser.add_argument("--epochs", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # wandb
    parser.add_argument("--wandb", action="store_true", help="开启 wandb 可视化")
    parser.add_argument("--wandb-project", type=str, default="qwen-lora-scorer")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    wandb_run = setup_wandb(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=get_dtype(),
    )
    model.config.use_cache = False

    lora_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_cfg)

    raw_ds = load_ds(args.data)
    ds = preprocess_dataset(raw_ds, tokenizer, args.max_length)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        report_to=["wandb"] if args.wandb else ["none"],
        run_name=args.wandb_run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Done. LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()