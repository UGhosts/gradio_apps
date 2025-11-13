#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取项目根目录（脚本所在目录的上一级）
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 定义默认值
APP_FILE="${PROJECT_ROOT}/main.py"
APP=""
PORT=""

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
        exit 1
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

# 检查应用文件是否存在
if [ ! -f "$APP_FILE" ]; then
    echo "错误：找不到应用文件: $APP_FILE"
    exit 1
fi

# 启动计数器
RESTART_COUNT=0
MAX_RESTARTS=10  # 最大重启次数，防止无限重启
RESTART_WINDOW=60  # 时间窗口（秒）
LAST_START_TIME=$(date +%s)

# 启动一个无限循环
while true; do
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - LAST_START_TIME))
    
    # 如果在时间窗口内，增加重启计数
    if [ $TIME_DIFF -lt $RESTART_WINDOW ]; then
        ((RESTART_COUNT++))
    else
        # 重置计数器
        RESTART_COUNT=1
    fi
    
    LAST_START_TIME=$CURRENT_TIME
    
    # 检查是否超过最大重启次数
    if [ $RESTART_COUNT -gt $MAX_RESTARTS ]; then
        echo "========= 错误：应用 $APP 在 ${RESTART_WINDOW}秒内重启次数过多（${RESTART_COUNT}次），停止重启。 ========="
        exit 1
    fi
    
    if [ $RESTART_COUNT -gt 1 ]; then
        echo "========= 重启 #$RESTART_COUNT: 正在启动 Gradio 应用 $APP ... ========="
    else
        echo "========= 正在启动 Gradio 应用 $APP ... ========="
    fi
    
    # 运行你的Python应用
    if [ -n "$PORT" ]; then
        /home/software/gradio_apps/temp/.venv/bin/python "$APP_FILE" "$APP" "$PORT"
        # python "$APP_FILE" "$APP" "$PORT"
    else
        /home/software/gradio_apps/temp/.venv/bin/python "$APP_FILE" "$APP"
        # python "$APP_FILE" "$APP"
    fi
    
    EXIT_CODE=$?
    
    # Python应用退出后，检查是否存在重启信号文件
    if [ -f "$RESTART_SIGNAL_FILE" ]; then
        echo "========= 检测到重启信号，正在重启应用 $APP ... ========="
        # 删除信号文件，为下一次重启做准备
        rm "$RESTART_SIGNAL_FILE"
        # 短暂等待，防止CPU占用过高
        sleep 1
    else
        if [ $EXIT_CODE -eq 0 ]; then
            echo "========= 应用 $APP 正常退出（退出码: $EXIT_CODE），关闭包装脚本。 ========="
        else
            echo "========= 应用 $APP 异常退出（退出码: $EXIT_CODE）========="
            # 异常退出时等待一段时间再重启，防止频繁崩溃
            if [ $RESTART_COUNT -gt 3 ]; then
                WAIT_TIME=$((RESTART_COUNT * 2))
                echo "========= 等待 ${WAIT_TIME}秒 后重启... ========="
                sleep $WAIT_TIME
            else
                sleep 2
            fi
            continue
        fi
        # 如果没有信号文件且正常退出，跳出循环
        break
    fi
done