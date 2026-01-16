export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 确保使用 roll 环境的 Python 与 Ray
export PATH="/data/share/projects/quanshu/envs/roll/bin:${PATH}"

# 1. 清理环境
echo "[1/5] Cleaning up previous processes..."
pkill -u quanshu -f "ray"
pkill -u quanshu -f "python.*ray"
pkill -u quanshu -f "python examples/start_agentic_pipeline.py" || true
docker ps -a | grep osworld | awk '{print $1}' | xargs -r docker rm -f


# 2. 网络设置
echo "[2/5] Setting up network..."
export NCCL_SOCKET_IFNAME=bond0.45
export GLOO_SOCKET_IFNAME=bond0.45
export TP_SOCKET_IFNAME=bond0.45
export MASTER_ADDR=10.10.41.33
: ${MASTER_PORT:=29650}
export MASTER_PORT
export RAY_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"
export AWS_REGION=us-east-1
export AWS_SUBNET_ID=dummy_subnet
export AWS_SECURITY_GROUP_ID=dummy_sg

# 3. 实验与日志设置
export WANDB_INIT_TIMEOUT=300
# export WANDB_MODE=offline  # 如需离线模式，取消注释

TS=$(date +%Y%m%d-%H%M%S)
EXP_NAME="agentic_pipeline_osworld_vnc-${TS}"
LOG_DIR="./output/logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

export HYDRA_RUN_DIR="${LOG_DIR}"
export OSWORLD_SERVER_LOG_DIR="${LOG_DIR}"

# 4. 配置参数（可根据需要修改）

# OSWorld 配置：控制使用哪个配置文件
CONFIG_PATH="${OSWORLD_CONFIG_PATH:-qwen3-8B-OSWorld}"
CONFIG_NAME="${OSWORLD_CONFIG_NAME:-agent_val_osworld_PPO_2_linux}"

# Hydra 参数覆盖（仅保留动态参数，其他参数请在配置文件中修改）
HYDRA_OVERRIDES="exp_name=${EXP_NAME} hydra.run.dir=${LOG_DIR}"

# 允许通过命令行传递临时参数覆盖（例如：bash run_osworld.sh max_steps=50）
if [ "$#" -gt 0 ]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES} $*"
fi
export HYDRA_OVERRIDES

echo "[4/5] Configuration completed"
echo "[5/5] Starting pipeline..."

# 5. 启动任务
CMD="/data/share/projects/quanshu/envs/roll/bin/python examples/start_agentic_pipeline.py \
  --config_path ${CONFIG_PATH} \
  --config_name ${CONFIG_NAME}"

nohup $CMD > "${LOG_DIR}/terminal_output.log" 2>&1 &
PID=$!

echo "✅ OSWorld Pipeline started! PID: $PID"
echo "📄 Logs: tail -f ${LOG_DIR}/terminal_output.log"
echo "🔍 Check status: ps aux | grep $PID"

