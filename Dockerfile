# 使用更精简的基础镜像（如果可能），但这里我们保留原样，因为您需要完整的开发环境
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量，只设置一次
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# 阶段 1: 安装系统依赖和 Python 环境
# 将所有 apt 和 pip 基础设置合并到一个 RUN 指令中
# ==============================================================================
RUN \
    # 替换为国内源以加速
    sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    \
    # 更新并安装系统依赖
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3.10 \
        python3.10-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
        tmux && \
    \
    # 创建 python 软链接
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    \
    # 升级 pip 并设置国内源
    python3 -m pip install --upgrade pip setuptools wheel && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    \
    # [关键] 清理 apt 缓存，减小这一层的体积
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 修复 libcuda.so.1 链接问题
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

# 设置工作目录
WORKDIR /app

# ==============================================================================
# 阶段 2: 安装 Python 依赖
# 利用 Docker 缓存机制：先只复制 requirements.txt 并安装，
# 这样如果 requirements.txt 没有变化，这一层就可以被缓存
# ==============================================================================
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# 阶段 3: 安装 PaddlePaddle 和 PaddleX
# 将所有 paddle 相关的安装合并到一个 RUN 指令中
# ==============================================================================
RUN \
    # [关键] 使用 --no-cache-dir 避免 pip 缓存占用空间
    pip install --no-cache-dir paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
    \
    # 克隆 PaddleX 仓库
    git clone https://github.com/PaddlePaddle/PaddleX.git && \
    cd PaddleX && \
    \
    # 以可编辑模式安装 PaddleX 及其依赖
    pip install --no-cache-dir -e ".[base]" && \
    pip install --no-cache-dir -e ".[ocr]" && \
    \
    # 安装 PaddleX 的组件
    paddlex --install PaddleTS && \
    paddlex --install PaddleOCR && \
    paddlex --install PaddleDetection && \
    paddlex --install PaddleSeg && \
    \
    # [关键] 清理：安装完成后可以删除克隆下来的源码仓库以减小体积
    cd .. && rm -rf PaddleX

# ==============================================================================
# 阶段 4: 复制应用程序代码
# 将代码复制放在最后，这样修改代码不会导致前面的依赖安装步骤重新运行
# ==============================================================================
COPY main.py ./
COPY apps/ ./apps/
COPY dataset/ ./dataset/
COPY doc/ ./doc/
COPY model/ ./model/

# 暴露端口并设置默认命令
EXPOSE 37700-37900
CMD ["bash", "-c", "tail -f /dev/null"]