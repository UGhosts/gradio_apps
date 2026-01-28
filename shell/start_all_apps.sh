#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 日志目录
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# 配置文件路径
APP_FILE="${PROJECT_ROOT}/main.py"
WRAPPER_SCRIPT="${SCRIPT_DIR}/start_app.sh"
SUPERVISOR_CONF="${SCRIPT_DIR}/supervisord.conf"

# === 你的应用配置列表 (保持不变) ===
declare -a APPS=(
    "crew_rul:37700"
    "cwru_cls:37701"
    "ele_metric:37702"
    "industrial_meter:37703"
    "air_compressor_cls:37704"
    "mp_ocr:37705"
    "wendu_prd:37706"
    "zhoucheng_cls:37707"
    "dianji_cls:37708"
    "belt_det:37709"
    "driver_belt_rul:37710"
    "chilun_cls:37711"
    "gangyin_ocr:37712"
    "tujiaoji_cls:37713"
    "tubuji_com:37714"
    "duanmo_prd:37715"
    "centrifugal_fault_det:37716"
    "fan_noise_flask:37717"
    "jiaobanji_prd:37718"
    "press_machine_cls:37719"
)

# === 生成 Supervisor 配置文件 ===

echo "正在生成 Supervisor 配置..."

# 1. 写入全局配置
cat > "$SUPERVISOR_CONF" <<EOF
[unix_http_server]
file=/var/run/supervisor.sock   ; socket 文件路径
chmod=0700                       ; socket 权限

[supervisord]
logfile=${LOG_DIR}/supervisord.log ; 主日志文件
pidfile=/var/run/supervisord.pid ; pid 文件
nodaemon=true                    ; 【关键】在前台运行，防止 Docker 退出
user=root                        ; 运行用户

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock ; 连接 socket

EOF

# 2. 循环写入每个应用的配置
for app_config in "${APPS[@]}"; do
    # 解析应用名和端口
    if [[ "$app_config" == *":"* ]]; then
        app_name="${app_config%%:*}"
        port="${app_config##*:}"
    else
        app_name="$app_config"
        port=""
    fi
    
    # 构造启动命令 (保留你原来的 start_app.sh 调用逻辑)
    # 注意：这里我们让 supervisor 直接调用 bash 去执行你的 wrapper 脚本
    if [ -n "$port" ]; then
        COMMAND="bash ${WRAPPER_SCRIPT} -f ${APP_FILE} -a ${app_name} -p ${port}"
    else
        COMMAND="bash ${WRAPPER_SCRIPT} -f ${APP_FILE} -a ${app_name}"
    fi

    # 追加到配置文件
    cat >> "$SUPERVISOR_CONF" <<EOF

[program:${app_name}]
command=${COMMAND}
directory=${PROJECT_ROOT}
autostart=true
autorestart=true
startretries=3
stopasgroup=true
killasgroup=true
stderr_logfile=${LOG_DIR}/${app_name}.err.log
stdout_logfile=${LOG_DIR}/${app_name}.out.log
EOF

done

echo "配置生成完毕: $SUPERVISOR_CONF"

# === 启动 Supervisor ===
# 这将接管当前进程，并根据上面生成的配置启动所有应用
echo "正在启动 Supervisor..."
exec supervisord -c "$SUPERVISOR_CONF"