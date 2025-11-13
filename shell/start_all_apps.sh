#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取项目根目录（脚本所在目录的上一级）
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志目录（基于项目根目录）
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# 应用配置文件路径（基于项目根目录）
APP_FILE="${PROJECT_ROOT}/main.py"

# 单个应用启动脚本路径（在当前 shell 目录）
WRAPPER_SCRIPT="${SCRIPT_DIR}/start_app.sh"

# 定义要启动的应用列表（格式：应用名:端口）
# 如果不需要指定端口，只写应用名即可
declare -a APPS=(
    "crew_rul:37700"
    "cwru_cls:37701"
    "ele_metric:37702"
    "industrial_meter:37703"
    "air_compressor_cls:37704"
    # 添加更多应用...
)

# 存储所有应用的 PID
declare -A APP_PIDS

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}收到终止信号，正在停止所有应用...${NC}"
    for app_name in "${!APP_PIDS[@]}"; do
        pid=${APP_PIDS[$app_name]}
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${BLUE}停止应用: $app_name (PID: $pid)${NC}"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    
    # 等待所有进程退出
    sleep 2
    
    # 强制杀死仍在运行的进程
    for app_name in "${!APP_PIDS[@]}"; do
        pid=${APP_PIDS[$app_name]}
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}强制停止应用: $app_name (PID: $pid)${NC}"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    
    echo -e "${GREEN}所有应用已停止${NC}"
    exit 0
}

# 注册信号处理
trap cleanup SIGTERM SIGINT SIGQUIT

# 启动单个应用
start_app() {
    local app_config="$1"
    local app_name=""
    local port=""
    
    # 解析应用配置
    if [[ "$app_config" == *":"* ]]; then
        app_name="${app_config%%:*}"
        port="${app_config##*:}"
    else
        app_name="$app_config"
    fi
    
    local log_file="$LOG_DIR/${app_name}.log"
    
    echo -e "${BLUE}启动应用: $app_name${NC}"
    if [ -n "$port" ]; then
        echo -e "${BLUE}  端口: $port${NC}"
    fi
    echo -e "${BLUE}  日志: $log_file${NC}"
    
    # 启动应用（后台运行）
    if [ -n "$port" ]; then
        bash "$WRAPPER_SCRIPT" -f "$APP_FILE" -a "$app_name" -p "$port" > "$log_file" 2>&1 &
    else
        bash "$WRAPPER_SCRIPT" -f "$APP_FILE" -a "$app_name" > "$log_file" 2>&1 &
    fi
    
    local pid=$!
    APP_PIDS[$app_name]=$pid
    
    # 等待一小段时间，检查应用是否成功启动
    sleep 2
    
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}✓ 应用 $app_name 启动成功 (PID: $pid)${NC}\n"
        return 0
    else
        echo -e "${RED}✗ 应用 $app_name 启动失败${NC}"
        echo -e "${YELLOW}  查看日志: tail -f $log_file${NC}\n"
        unset APP_PIDS[$app_name]
        return 1
    fi
}

# 主函数
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    开始启动所有 Gradio 应用${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    # 检查包装脚本是否存在
    if [ ! -f "$WRAPPER_SCRIPT" ]; then
        echo -e "${RED}错误：找不到应用启动脚本: $WRAPPER_SCRIPT${NC}"
        exit 1
    fi
    
    # 检查主应用文件是否存在
    if [ ! -f "$APP_FILE" ]; then
        echo -e "${RED}错误：找不到主应用文件: $APP_FILE${NC}"
        exit 1
    fi
    
    local success_count=0
    local fail_count=0
    
    # 启动所有应用
    for app in "${APPS[@]}"; do
        if start_app "$app"; then
            ((success_count++))
        else
            ((fail_count++))
        fi
        
        # 稍微间隔一下，避免同时启动太多进程
        sleep 1
    done
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}启动完成${NC}"
    echo -e "${GREEN}  成功: $success_count 个应用${NC}"
    if [ $fail_count -gt 0 ]; then
        echo -e "${RED}  失败: $fail_count 个应用${NC}"
    fi
    echo -e "${GREEN}========================================${NC}\n"
    
    # 如果没有任何应用启动成功，退出
    if [ $success_count -eq 0 ]; then
        echo -e "${RED}所有应用启动失败，退出...${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}所有应用正在后台运行...${NC}"
    echo -e "${BLUE}按 Ctrl+C 停止所有应用${NC}\n"
    
    # 监控所有应用状态
    while true; do
        sleep 10
        
        for app_name in "${!APP_PIDS[@]}"; do
            pid=${APP_PIDS[$app_name]}
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${YELLOW}警告: 应用 $app_name (PID: $pid) 已停止${NC}"
                echo -e "${YELLOW}  查看日志: tail -f $LOG_DIR/${app_name}.log${NC}"
                unset APP_PIDS[$app_name]
            fi
        done
        
        # 如果所有应用都停止了，退出
        if [ ${#APP_PIDS[@]} -eq 0 ]; then
            echo -e "${RED}所有应用都已停止，退出...${NC}"
            exit 1
        fi
    done
}

# 运行主函数
main