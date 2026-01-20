# main.py
import os
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg

if __name__ == '__main__':
    config = load_config("config.yaml")
    preload = bool(config.get("preload", False))
    preload_path = config.get("preload_path", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_ID = config.get("model_name","Qwen/Qwen3-Embedding-0.6B")
    MODEL_NAME = MODEL_ID.split("/")[-1]

    if preload:
        if not preload_path:
            raise ValueError("配置 preload 为 True，但未提供 preload_path")
        base_path = Path(preload_path)
        target_dir = base_path / MODEL_NAME

        model_files_present = any((target_dir / fname).exists() for fname in ["config.json", "tokenizer.json", "pytorch_model.bin", "model.safetensors"])
        if model_files_present:
            print(f"Found existing model files in {target_dir}, skipping download.")
        else:
            base_path.mkdir(parents=True, exist_ok=True)
            print(f"Downloading model {MODEL_ID} into {target_dir} ...")
            try:
                snapshot_download(repo_id=MODEL_ID, local_dir=str(target_dir), resume_download=True)
            except Exception as e:
                raise RuntimeError(f"snapshot_download 失败: {e}")
            print("Download finished.")

        # now load from target_dir
        print(f"Loading model & tokenizer from local path: {target_dir}")
        tokenizer = AutoTokenizer.from_pretrained(target_dir, padding_side="left", trust_remote_code=True)
        model = AutoModel.from_pretrained(target_dir, trust_remote_code=True)
    else:
        print(f"Loading model & tokenizer from Hugging Face Hub: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", trust_remote_code=True)
        if device.type == "cuda":
            model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        else:
            model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

    model.to(device)
    print(f"Model loaded. Device: {device}. Model dtype: {next(model.parameters()).dtype}")

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    input_texts = queries + documents

    max_length = 8192
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)

    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T)
    print(scores.tolist())
