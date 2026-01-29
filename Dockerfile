# 使用精简的基础镜像
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# 设置环境变量
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"
    # UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# ==============================================================================
# 阶段 1: 安装系统依赖和 Python
# ==============================================================================
# 同样注意：Github Actions 环境下使用 aliyun 可能会变慢
# RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
#     sed -i 's@http://security.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        supervisor \
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
        nano \
        fonts-wqy-microhei \
        fonts-wqy-zenhei \
        ca-certificates && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 修复 libcuda.so.1 链接问题
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

# 设置工作目录
WORKDIR /app

# ==============================================================================
# 阶段 2: 使用 uv 安装 Python 依赖
# ==============================================================================
COPY requirements.txt ./

# 使用 uv 安装基础依赖
RUN uv pip install --system --no-cache -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# ==============================================================================
# 阶段 3: 安装 PaddlePaddle 和 PaddleX
# ==============================================================================

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN uv pip install --system --no-cache paddlepaddle-gpu==3.1.1 \
     --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

RUN uv pip install --system langchain==0.3.27

RUN git clone --depth 1 --branch v3.3.9 https://github.com/PaddlePaddle/PaddleX.git /tmp/PaddleX && \
    cd /tmp/PaddleX && \
    uv pip install --system --no-cache -e ".[base]" && \
    uv pip install --system --no-cache -e ".[ocr]" && \
    uv run paddlex --install PaddleTS && \
    uv run paddlex --install PaddleOCR && \
    uv run paddlex --install PaddleDetection && \
    uv run paddlex --install PaddleSeg && \
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
COPY utils/ ./utils/

RUN chmod +x /app/shell/start_all_apps.sh /app/shell/start_app.sh
RUN mkdir -p /app/logs

EXPOSE 37700-37900
CMD ["bash", "/app/shell/start_all_apps.sh"]