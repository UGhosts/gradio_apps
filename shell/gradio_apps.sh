#!/bin/bash

# 定义默认值
APP_FILE="/home/software/gradio_apps/main.py"
APP=""
PORT=""
# RESTART_SIGNAL_FILE=".restart_signal"

# 解析命令行参数
while getopts ":f:a:p:" opt; do
  case $opt in
    f) APP_FILE="$OPTARG"
    ;;
    a) APP="$OPTARG"
    ;;
    p) PORT="$OPTARG"
    ;;
    \?) echo "无效的选项 -$OPTARG" >&2
    ;;
  esac
done

shift $((OPTIND-1))
RESTART_SIGNAL_FILE=".restart_signal_${APP}"
# 检查APP是否已设置
if [ -z "$APP" ]; then
    echo "错误：必须指定应用名。请使用 -a <应用名> 参数。"
    exit 1
fi

# 启动一个无限循环
while true; do
    echo "========= 正在启动 Gradio 应用... ========="
    # 运行你的Python应用
    if [ -n "$PORT" ]; then
        /nfs/miniconda3/envs/gradio/bin/python "$APP_FILE" "$APP" "$PORT"
    else
        /nfs/miniconda3/envs/gradio/bin/python "$APP_FILE" "$APP"
    fi
    
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