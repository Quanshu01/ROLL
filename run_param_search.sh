#!/bin/bash
# ============================================================
# PPO 参数验证脚本
# 使用方法:
#   bash run_param_search.sh [实验编号] [可选参数覆盖]
#
# 示例:
#   bash run_param_search.sh 1              # 运行实验组1 (100条数据)
#   bash run_param_search.sh 2              # 运行实验组2 (100条数据)
#   bash run_param_search.sh full1          # 运行完整验证1 (500条数据)
#   bash run_param_search.sh custom actor_train.training_args.learning_rate=2e-6
#
# 替换训练数据:
#   bash run_param_search.sh custom task_config_path=evaluation_examples/your_data.json max_steps=50
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/data/share/projects/quanshu/envs/roll/bin:${PATH}"

# 网络配置
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
export WANDB_INIT_TIMEOUT=300

# 清理环境
echo "[1/5] Cleaning up previous processes..."
pkill -u $(whoami) -f "ray" || true
pkill -u $(whoami) -f "python.*ray" || true
pkill -u $(whoami) -f "python examples/start_agentic_pipeline.py" || true
docker ps -a | grep osworld | awk '{print $1}' | xargs -r docker rm -f || true

# 配置文件
CONFIG_PATH="qwen3-8B-OSWorld"
CONFIG_NAME="agent_val_osworld_PPO_param_search"

# ============================================================
# 预定义的参数搜索组合
# ============================================================
EXP_ID="${1:-1}"
TS=$(date +%Y%m%d-%H%M%S)

case $EXP_ID in
    # ========== 阶段1: 快速筛选（100条数据）==========
    # 100条 / rollout_batch_size(40) × 2 epoch = 5步
    # Train和Val复用VM，全部40台用于训练
    "1")
        # 实验组1: 基线 - 保守学习率
        EXP_NAME="param_search_exp1_conservative_lr-${TS}"
        OVERRIDES="max_steps=10 \
                   task_config_path=evaluation_examples/test_genearl_harm_200.json \
                   actor_train.training_args.learning_rate=5e-7 \
                   critic.training_args.learning_rate=5e-6 \
                   ppo_epochs=1"
        ;;
    "2")
        # 实验组2: 基线 - 标准学习率
        EXP_NAME="param_search_exp2_standard_lr-${TS}"
        OVERRIDES="max_steps=10 \
                   task_config_path=evaluation_examples/test_genearl_harm_200.json \
                   actor_train.training_args.learning_rate=1e-6 \
                   critic.training_args.learning_rate=1e-5 \
                   ppo_epochs=1"
        ;;
    "3")
        # 实验组3: 较大学习率 + ppo_epochs=2
        EXP_NAME="param_search_exp3_higher_lr_ppo2-${TS}"
        OVERRIDES="max_steps=10 \
                   task_config_path=evaluation_examples/test_genearl_harm_200.json \
                   actor_train.training_args.learning_rate=2e-6 \
                   critic.training_args.learning_rate=1e-5 \
                   ppo_epochs=2"
        ;;
    "4")
        # 实验组4: 激进学习率
        EXP_NAME="param_search_exp4_aggressive_lr-${TS}"
        OVERRIDES="max_steps=10 \
                   actor_train.training_args.learning_rate=5e-6 \
                   critic.training_args.learning_rate=2e-5 \
                   ppo_epochs=2"
        ;;

    # ========== 阶段2: 完整验证（500条数据）==========
    # 500条 / rollout_batch_size(40) × 2 epoch = 25步
    # 注意: 需要替换 task_config_path 为500条数据的配置文件
    "full1")
        # 完整验证1: 推荐配置
        EXP_NAME="param_search_full1_recommended-${TS}"
        OVERRIDES="max_steps=25 \
                   task_config_path=evaluation_examples/test_genearl_harm_500.json \
                   actor_train.training_args.learning_rate=1.5e-6 \
                   critic.training_args.learning_rate=1e-5 \
                   ppo_epochs=2 \
                   init_kl_coef=0.01 \
                   entropy_loss_coef=0.01"
        ;;
    "full2")
        # 完整验证2: 较大学习率
        EXP_NAME="param_search_full2_higher_lr-${TS}"
        OVERRIDES="max_steps=25 \
                   task_config_path=evaluation_examples/test_genearl_harm_500.json \
                   actor_train.training_args.learning_rate=2e-6 \
                   critic.training_args.learning_rate=1e-5 \
                   ppo_epochs=2 \
                   init_kl_coef=0.01"
        ;;
    "full3")
        # 完整验证3: 更多PPO epochs
        EXP_NAME="param_search_full3_ppo4-${TS}"
        OVERRIDES="max_steps=25 \
                   task_config_path=evaluation_examples/test_genearl_harm_500.json \
                   actor_train.training_args.learning_rate=1e-6 \
                   critic.training_args.learning_rate=1e-5 \
                   ppo_epochs=4 \
                   init_kl_coef=0.02"
        ;;

    # ========== 自定义实验 ==========
    "custom")
        EXP_NAME="param_search_custom-${TS}"
        shift  # 移除第一个参数
        OVERRIDES="$*"
        ;;
    *)
        echo "未知实验编号: $EXP_ID"
        echo "可用选项:"
        echo "  快速筛选 (100条): 1, 2, 3, 4"
        echo "  完整验证 (500条): full1, full2, full3"
        echo "  自定义: custom [参数覆盖]"
        echo ""
        echo "替换数据示例:"
        echo "  bash run_param_search.sh custom task_config_path=evaluation_examples/your_data.json max_steps=50"
        exit 1
        ;;
esac

LOG_DIR="./output/logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

export HYDRA_RUN_DIR="${LOG_DIR}"
export OSWORLD_SERVER_LOG_DIR="${LOG_DIR}"

# 构建完整的 Hydra overrides
FULL_OVERRIDES="exp_name=${EXP_NAME} hydra.run.dir=${LOG_DIR} ${OVERRIDES}"
export HYDRA_OVERRIDES="${FULL_OVERRIDES}"

echo "============================================================"
echo "🚀 PPO 参数搜索实验"
echo "============================================================"
echo "实验编号: $EXP_ID"
echo "实验名称: $EXP_NAME"
echo "配置文件: ${CONFIG_PATH}/${CONFIG_NAME}"
echo "参数覆盖: ${OVERRIDES}"
echo "日志目录: ${LOG_DIR}"
echo "============================================================"

CMD="/data/share/projects/quanshu/envs/roll/bin/python examples/start_agentic_pipeline.py \
  --config_path ${CONFIG_PATH} \
  --config_name ${CONFIG_NAME}"

echo "[5/5] Starting experiment..."
nohup $CMD > "${LOG_DIR}/terminal_output.log" 2>&1 &
PID=$!

echo ""
echo "✅ 实验已启动! PID: $PID"
echo "📄 实时日志: tail -f ${LOG_DIR}/terminal_output.log"
echo "📊 WandB: https://wandb.ai/your-project/roll-agentic"
echo "🔍 检查状态: ps aux | grep $PID"
