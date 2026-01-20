
# Hugging Face Login
```
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGING_FACE_HUB_TOKEN="<hf Token>" 
```
## Download
```
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir /root/autodl-tmp --local-dir-use-symlinks False
```
在example下运行示例代码：
```
cd ~/codes/text_judgement/example
python3 sentence_transformer.py
```