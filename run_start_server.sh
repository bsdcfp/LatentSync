
#!/bin/bash

# 最简化的 Docker 主进程脚本

SCRIPT="./start_server.sh"

# 信号处理
cleanup() {
    echo "Shutting down..."
    [[ -n "$PID" ]] && kill "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null || true
    exit 0
}

trap cleanup TERM INT

# 循环启动
while true; do
    echo "Starting server..."
    chmod +x "$SCRIPT"
    "$SCRIPT" &
    PID=$!
    
    # 等待进程结束
    wait "$PID"
    
    echo "Server exited, restarting in 3 seconds..."
    sleep 3
done
