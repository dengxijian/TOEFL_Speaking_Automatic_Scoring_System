# whisper-base.en 模型文件获取与配置说明
由于 `model.bin` 文件体积约 1.1GB，超出 GitHub 单个文件 100MB 的上传限制，因此未提交至仓库。请按照以下步骤获取并配置该文件：

## 一、文件下载
### 方式 1：官方源下载（推荐）
访问 Hugging Face 官方仓库下载核心权重文件：
- 下载地址：https://huggingface.co/openai/whisper-base.en/resolve/main/pytorch_model.bin
- 备用地址（镜像加速）：https://hf-mirror.com/openai/whisper-base.en/resolve/main/pytorch_model.bin

### 方式 2：命令行一键下载（Python 环境）
若本地已安装 Python，可执行以下命令直接下载到指定目录（无需手动改路径）：
```powershell
# 安装依赖（首次执行）
pip install huggingface-hub

# 下载并自动重命名为 model.bin（适配本项目路径）
python -c "
from huggingface_hub import hf_hub_download
import os

# 配置镜像加速（国内用户建议开启）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载文件
hf_hub_download(
    repo_id='openai/whisper-base.en',
    filename='pytorch_model.bin',
    local_dir='./Data_process/models/whisper-base.en',
    local_dir_use_symlinks=False
)

# 重命名为 model.bin（适配项目代码读取逻辑）
os.rename(
    './Data_process/models/whisper-base.en/pytorch_model.bin',
    './Data_process/models/whisper-base.en/model.bin'
)
print('模型文件下载并重命名完成！')
"
