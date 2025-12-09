#!/bin/bash

# ========================================================
# pFL-FOOGD 自动化实验脚本 - GPU 1 (v2.1)
# 运行后4个实验：真实中差异(1.0)和理想对照(10.0)的FOOGD和Baseline
# ========================================================

# 设置使用显卡1
export CUDA_VISIBLE_DEVICES=1

# 1. 基础配置 (请根据服务器实际情况调整)
DATA_ROOT="./Plankton_OOD_Dataset"
N_CLIENTS=5
ROUNDS=100
EPOCHS=3                # 修改为3，与client.py中的local_epochs默认值一致
BATCH_SIZE=64
MODEL="densenet121"     # 使用densenet121以节省显存
SEED=2025               # 固定随机种子，确保所有实验的数据划分完全一致！

# 创建日志目录
mkdir -p logs

echo "========================================================"
echo "开始运行 pFL-FOOGD 实验组 - GPU 1 (共 2 组)"
echo "开始时间: $(date)"
echo "随机种子: $SEED (保证数据划分一致性)"
echo "使用显卡: GPU 1 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "========================================================"

# 定义单次实验运行函数
run_experiment() {
    local ALPHA=$1
    local USE_FOOGD=$2
    local EXP_NAME=$3
    local DESC=$4

    echo ""
    echo "--------------------------------------------------------"
    echo "正在运行: $EXP_NAME"
    echo "场景说明: $DESC"
    echo "配置: Alpha=$ALPHA | Use FOOGD=$USE_FOOGD"
    echo "使用显卡: GPU 1"
    echo "--------------------------------------------------------"

    # 构建基础命令
    # 注意：这里假设你已经修改了 train_federated.py，增加了 --seed 参数
    IMG_SIZE=299

    CMD="python train_federated.py \
        --data_root $DATA_ROOT \
        --n_clients $N_CLIENTS \
        --alpha $ALPHA \
        --communication_rounds $ROUNDS \
        --local_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --image_size $IMG_SIZE \
        --model_type $MODEL \
        --seed $SEED \
        --compute_aug_features \
        --freeze_bn \
        --output_dir ./experiments/$EXP_NAME"

    # 根据开关添加 --use_foogd 参数
    # 注意：这里假设你已经将 train_federated.py 中 use_foogd 的 default 改为了 False
    if [ "$USE_FOOGD" = "true" ]; then
        CMD="$CMD --use_foogd"
    fi

    # 执行命令并保存日志
    # 使用 tee 同时在屏幕显示和写入文件，方便实时查看进度
    echo "执行命令: $CMD"
    $CMD 2>&1 | tee "logs/${EXP_NAME}.log"

    echo ">>> 实验 $EXP_NAME 完成！"
}

# ================= GPU 1 实验队列 (共2组) =================

# --- 第3组：中等差异 (Alpha=5.0) ---
# 意义：模拟中等程度的异质性，介于强异质性和IID之间。
run_experiment 5.0 "true"  "alpha5.0_with_foogd" "中等异质性 (With FOOGD)"
run_experiment 5.0 "false" "alpha5.0_no_foogd"   "中等异质性 (Baseline)"

echo "========================================================"
echo "GPU 1 上的 2 组实验已全部完成！请检查 logs/ 目录下的日志文件。"
echo "结束时间: $(date)"
echo "========================================================"