# ==========================================
# 用法（按需选择配置）：
#   OSWORLD_CONFIG_NAME=agent_val_osworld          # Linux 1VM（默认）
#   OSWORLD_CONFIG_NAME=agent_val_osworld_2vm      # Linux 2VM
#   OSWORLD_CONFIG_NAME=agent_val_osworld_windows  # Windows
# 如需改目录：OSWORLD_CONFIG_PATH=其他目录名（默认 qwen2.5-vl-7B-OSWorld）
# 示例：Linux 1VM：OSWORLD_CONFIG_NAME=agent_val_osworld bash run_osworld.sh
# 示例：Linux 2VM：OSWORLD_CONFIG_NAME=agent_val_osworld_2vm bash run_osworld.sh
# 示例：Windows：OSWORLD_CONFIG_NAME=agent_val_osworld_windows bash run_osworld.sh
# ==========================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 确保使用 roll 环境的 Python 与 Ray
export PATH="/data/share/projects/quanshu/envs/roll/bin:${PATH}"

# ==========================================
# 1. 清理环境
# ==========================================
echo "[1/4] Cleaning up previous processes..."
pkill -u quanshu -f "ray"
pkill -u quanshu -f "python.*ray"
pkill -u quanshu -f "python examples/start_agentic_pipeline.py" || true
docker ps -a | grep osworld | awk '{print $1}' | xargs -r docker rm -f

# ==========================================
# 2. 网络设置
# ==========================================
echo "[2/4] Setting up network..."
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
# 3. 配置与日志
# ==========================================
# 保持 WandB 离线，防止网络报错
# export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=300

TS=$(date +%Y%m%d-%H%M%S)
EXP_NAME="agentic_pipeline_osworld_vnc-${TS}"
LOG_DIR="./output/logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

# 让 Hydra 和 OSWorld server 使用同一运行目录
export HYDRA_RUN_DIR="${LOG_DIR}"
export OSWORLD_SERVER_LOG_DIR="${LOG_DIR}"

# ROLL/OSWorld 配置：在这里统一定义 config_path / config_name，
# 下面既用于启动命令，也用于自动解析 VM HTTP 配置
# 如需 Windows 配置，使用：export OSWORLD_CONFIG_NAME=agent_val_osworld_windows
CONFIG_PATH="${OSWORLD_CONFIG_PATH:-qwen2.5-vl-7B-OSWorld}"
CONFIG_NAME="${OSWORLD_CONFIG_NAME:-agent_val_osworld}"

# 自动解析配置里的 path_to_vm，便于日志提示
# 例如 path_to_vm: js2.blockelite.cn:12923:12976 -> IP=js2.blockelite.cn, HTTP_PORT=12976
VM_INFO=$(/data/share/projects/quanshu/envs/roll/bin/python - << 'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

config_path = os.environ.get("OSWORLD_CONFIG_PATH", "qwen2.5-vl-7B-OSWorld")
config_name = os.environ.get("OSWORLD_CONFIG_NAME", "agent_val_osworld")

root = Path("/data/share/projects/quanshu/ROLL/examples")
yaml_path = root / config_path / f"{config_name}.yaml"
if not yaml_path.exists():
    raise SystemExit(0)

cfg = OmegaConf.load(str(yaml_path))
try:
    path_to_vm = cfg.custom_envs.OSWorld.env_config.path_to_vm
except Exception:
    raise SystemExit(0)

if not path_to_vm:
    raise SystemExit(0)

parts = str(path_to_vm).split(":")
if len(parts) == 3:
    host, _vnc_port, http_port = parts
elif len(parts) == 2:
    host, http_port = parts
else:
    raise SystemExit(0)

host = host.strip()
http_port = str(http_port).strip()
if not host or not http_port:
    raise SystemExit(0)

print(f"{host} {http_port}")
PY
)

if [ -n "${VM_INFO}" ]; then
  OSWORLD_VM_IP_PARSED=$(echo "${VM_INFO}" | awk '{print $1}')
  OSWORLD_VM_PORT_PARSED=$(echo "${VM_INFO}" | awk '{print $2}')
  if [ -n "${OSWORLD_VM_IP_PARSED}" ] && [ -n "${OSWORLD_VM_PORT_PARSED}" ]; then
    export OSWORLD_VM_IP="${OSWORLD_VM_IP_PARSED}"
    export OSWORLD_VM_PORT="${OSWORLD_VM_PORT_PARSED}"
    echo "[INFO] Parsed OSWorld VM from config: OSWORLD_VM_IP=${OSWORLD_VM_IP}, OSWORLD_VM_PORT=${OSWORLD_VM_PORT}"
  fi
fi

echo "[3/4] Starting run: ${EXP_NAME}"

HYDRA_OVERRIDES="exp_name=${EXP_NAME} hydra.run.dir=${LOG_DIR} max_steps=10 actor_infer.strategy_args.strategy_config.gpu_memory_utilization=0.15"

if [ "$#" -gt 0 ]; then
  EXTRA_OVERRIDES="$*"
  HYDRA_OVERRIDES="${HYDRA_OVERRIDES} ${EXTRA_OVERRIDES}"
fi
export HYDRA_OVERRIDES

# ==========================================
# 4. 启动任务
# ==========================================

# 1 默认 CMD（CONFIG_NAME 控制 Linux/Windows/2VM）
CMD="/data/share/projects/quanshu/envs/roll/bin/python examples/start_agentic_pipeline.py \
  --config_path ${CONFIG_PATH} \
  --config_name ${CONFIG_NAME}"

nohup $CMD > "${LOG_DIR}/terminal_output.log" 2>&1 &
PID=$!

echo "✅ OSWorld Pipeline started! PID: $PID"
echo "📄 Logs: tail -f ${LOG_DIR}/terminal_output.log"
echo "🔍 Check status: ps aux | grep $PID"

