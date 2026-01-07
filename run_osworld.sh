#!/bin/bash
# ==========================================
# OSWorld Agentic Pipeline 启动脚本
# ==========================================
# 用法（按需选择配置）：
#   OSWORLD_CONFIG_NAME=agent_val_osworld          # Linux 1VM（默认）
#   OSWORLD_CONFIG_NAME=agent_val_osworld_2vm      # Linux 2VM
#   OSWORLD_CONFIG_NAME=agent_val_osworld_windows  # Windows
# 如需改目录：OSWORLD_CONFIG_PATH=其他目录名（默认 qwen2.5-vl-7B-OSWorld）
# 示例：Linux 1VM：OSWORLD_CONFIG_NAME=agent_val_osworld bash run_osworld.sh
# 示例：Linux 2VM：OSWORLD_CONFIG_NAME=agent_val_osworld_2vm_linux bash run_osworld.sh
# 示例：Windows：OSWORLD_CONFIG_NAME=agent_val_osworld_windows bash run_osworld.sh

# export LLM_JUDGE_MODEL="gpt-4o" 
# export OPENAI_API_KEY=sk-UvQ4LjDbNt1FQKhJolXPZ9VTGtMPMfx0lXjetCtdjFmHKleZ
# export OPENAI_BASE_URL=https://api3.xhub.chat/v1
# OSWORLD_CONFIG_NAME=agent_val_osworld_2vm_linux bash run_osworld.sh

# # 查看 response_level 计算
# grep "Response Level Reward Calculation" /data/share/projects/quanshu/ROLL/output/logs/agentic_pipeline_osworld_vnc-*/terminal_output.log
# # 查看模型更新
# grep "Updating Actor Model\|Actor Model Updated" /data/share/projects/quanshu/ROLL/output/logs/agentic_pipeline_osworld_vnc-*/terminal_output.log
# # 查看训练 reward 统计
# grep "Response Level Rewards Stats" /data/share/projects/quanshu/ROLL/output/logs/agentic_pipeline_osworld_vnc-*/terminal_output.log

# pytohn /data/share/projects/quanshu/scripts/analysis/analyze_training_log.py /data/share/projects/quanshu/ROLL/output/logs/agentic_pipeline_osworld_vnc-20260104-211309
# ==========================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 确保使用 roll 环境的 Python 与 Ray
export PATH="/data/share/projects/quanshu/envs/roll/bin:${PATH}"

# ==========================================
# 1. 清理环境
# ==========================================
echo "[1/5] Cleaning up previous processes..."
pkill -u quanshu -f "ray"
pkill -u quanshu -f "python.*ray"
pkill -u quanshu -f "python examples/start_agentic_pipeline.py" || true
docker ps -a | grep osworld | awk '{print $1}' | xargs -r docker rm -f


# ==========================================
# 2. 网络设置
# ==========================================
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

# ==========================================
# 3. 实验与日志设置
# ==========================================
export WANDB_INIT_TIMEOUT=300
# export WANDB_MODE=offline  # 如需离线模式，取消注释

TS=$(date +%Y%m%d-%H%M%S)
EXP_NAME="agentic_pipeline_osworld_vnc-${TS}"
LOG_DIR="./output/logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

export HYDRA_RUN_DIR="${LOG_DIR}"
export OSWORLD_SERVER_LOG_DIR="${LOG_DIR}"

# ==========================================
# 4. 配置参数（可根据需要修改）
# ==========================================

# OSWorld 配置：控制使用哪个配置文件
# - agent_val_osworld: Linux 1VM（默认）
# - agent_val_osworld_2vm: Linux 2VM
# - agent_val_osworld_windows: Windows
CONFIG_PATH="${OSWORLD_CONFIG_PATH:-qwen2.5-vl-7B-OSWorld}"
CONFIG_NAME="${OSWORLD_CONFIG_NAME:-agent_val_osworld}"

# Hydra 参数覆盖（仅保留动态参数，其他参数请在配置文件中修改）
# 说明：
#   - exp_name 和 hydra.run.dir: 必须保留（动态生成，每次运行都不同）
#   - max_steps、gpu_memory_utilization 等：请在配置文件中修改
HYDRA_OVERRIDES="exp_name=${EXP_NAME} hydra.run.dir=${LOG_DIR}"

# 允许通过命令行传递临时参数覆盖（例如：bash run_osworld.sh max_steps=50）
if [ "$#" -gt 0 ]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES} $*"
fi
export HYDRA_OVERRIDES

echo "[4/5] Configuration completed"
echo "[5/5] Starting pipeline..."

# ==========================================
# 5. 启动任务
# ==========================================
CMD="/data/share/projects/quanshu/envs/roll/bin/python examples/start_agentic_pipeline.py \
  --config_path ${CONFIG_PATH} \
  --config_name ${CONFIG_NAME}"

nohup $CMD > "${LOG_DIR}/terminal_output.log" 2>&1 &
PID=$!

echo "✅ OSWorld Pipeline started! PID: $PID"
echo "📄 Logs: tail -f ${LOG_DIR}/terminal_output.log"
echo "🔍 Check status: ps aux | grep $PID"

