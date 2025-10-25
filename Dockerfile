# 使用精简的基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.cargo/bin:${PATH}"

# ==============================================================================
# 阶段 1: 安装系统依赖、Python 和 uv
# ==============================================================================
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
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
        tmux \
        curl \
        ca-certificates && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    # 安装 uv
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    # 配置 uv 使用国内镜像
    uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 修复 libcuda.so.1 链接问题
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

# 设置工作目录
WORKDIR /app

# ==============================================================================
# 阶段 2: 使用 uv 安装 Python 依赖
# ==============================================================================
# 复制依赖文件
COPY requirements.txt ./

# 使用 uv 安装基础依赖（速度比 pip 快很多）
RUN uv pip install --system --no-cache -r requirements.txt

# ==============================================================================
# 阶段 3: 安装 PaddlePaddle 和 PaddleX
# ==============================================================================
# 安装 PaddlePaddle
RUN uv pip install --system --no-cache paddlepaddle-gpu==3.0.0 \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 安装 PaddleX
RUN git clone https://github.com/PaddlePaddle/PaddleX.git /tmp/PaddleX && \
    cd /tmp/PaddleX && \
    uv pip install --system --no-cache -e ".[base]" && \
    uv pip install --system --no-cache -e ".[ocr]" && \
    paddlex --install PaddleTS && \
    paddlex --install PaddleOCR && \
    paddlex --install PaddleDetection && \
    paddlex --install PaddleSeg && \
    # 清理不必要的文件
    rm -rf .git .github tests docs examples && \
    cd /app

# ==============================================================================
# 阶段 4: 复制应用代码
# ==============================================================================
COPY main.py ./
COPY apps/ ./apps/
COPY dataset/ ./dataset/
COPY doc/ ./doc/
COPY model/ ./model/
COPY shell/ ./shell/

# 暴露端口并设置默认命令
EXPOSE 37700-37900
CMD ["bash", "-c", "tail -f /dev/null"]