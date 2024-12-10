# Fine-tuning StarCoder2-3b for cmake applications


## 使用过程

### 1.首先安装conda环境
打开终端，下载 Miniconda 安装包：
``` bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

```
运行安装脚本(注意最后添加环境变量时选择yes，全部输yes)
```
bash Miniconda3-latest-Linux-x86_64.sh
```
完成安装后，执行：
```
source ~/.bashrc
```

### 2.创建conda环境并进入

```
conda create -n cmake python=3.10 && conda activate cmake
```

### 3.创建一个cmake文件夹（最好创建在挂载盘）

先cmake里拷入所需文件夹

### 4.安装环境
首先添加代理(每次打开cmake环境都得添加)
```
 
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"
```

执行安装命令
```
pip install -r requirements.txt
```

登录huggingface
```
huggingface-cli login
# 需要先获取模型许可
# 需要在huggingface上注册并获取token
```
安装Git LFS
```
# 使用目录下压缩包
tar -xzvf git-lfs-linux-amd64-v3.6.0.tar.gz
# 安装
cd git-lfs-3.6.0/

sudo ./install.sh
# 初始化

git lfs install

#验证
git lfs version
```
配置显卡算力
```
#根据显卡类型调整
export TORCH_CUDA_ARCH_LIST="8.9" 
```

升级所有依赖到最新
```
pip install --upgrade numpy

pip install --upgrade -r requirements.txt

pip install --upgrade deepspeed transformers

```

运行项目
```
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=8 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json

#--nproc_per_node后的数字根据显卡数量变化
# 等待他下载模型和数据，使用代理后速度还行
```

### 5.监控参数
新建会话

```
watch -n 1 nvidia-smi
# 监控显卡参数

export HF_HOME=~/shared-nvme/huggingface

source ~/.bashrc   # 或者 source ~/.zshrc

echo $HF_HOME

python generate.py --model_id /home/pod/shared-nvme/cmake/data/starchat-alpha

pip uninstall deepspeed

pip install deepspeed==0.15.4





