#!/bin/bash

# 定义你的Python应用文件名
APP_FILE="${1:-apps/main.py}"
# 定义端口，默认为7860
PORT="${2:-7860}"
# 定义重启信号文件名，必须和Python脚本中的一致
RESTART_SIGNAL_FILE=".restart_signal"

# 启动一个无限循环
while true; do
    echo "========= 正在启动 Gradio 应用... ========="
    
    # 运行你的Python应用
    /nfs/miniconda3/envs/gradio/bin/python "$APP_FILE" --port "$PORT"
    
    # Python应用退出后，检查是否存在重启信号文件
    if [ -f "$RESTART_SIGNAL_FILE" ]; then
        echo "========= 检测到重启信号，正在重启应用... ========="
        # 删除信号文件，为下一次重启做准备
        rm "$RESTART_SIGNAL_FILE"
        # 短暂等待，防止CPU占用过高
        sleep 1
    else
        echo "========= 应用正常退出，关闭包装脚本。 ========="
        # 如果没有信号文件，说明是手动关闭，跳出循环
        break
    fi
done