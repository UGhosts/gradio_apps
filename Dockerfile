FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update && \
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
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --upgrade pip setuptools wheel


WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

COPY requirements.txt ./
COPY main.py ./
COPY apps/ ./apps/
COPY dataset/ ./dataset/
COPY doc/ ./doc/
COPY model/ ./model/

RUN git clone https://github.com/PaddlePaddle/PaddleX.git

WORKDIR /app/PaddleX

RUN pip install -e ".[base]"

RUN paddlex --install PaddleTS

WORKDIR /app

EXPOSE 37700-37900
CMD ["bash", "-c", "tail -f /dev/null"]