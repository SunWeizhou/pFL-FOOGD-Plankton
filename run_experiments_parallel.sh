#!/bin/bash

# ========================================================
# pFL-FOOGD 并行实验启动脚本 (v2.2)
# 同时在两张显卡上运行所有6个实验
# ========================================================

echo "========================================================"
echo "开始并行运行 pFL-FOOGD 全面对照实验 (共 6 组)"
echo "开始时间: $(date)"
echo "========================================================"

# 创建日志目录
mkdir -p logs

echo ""
echo "启动 GPU 0 上的实验 (前4组: alpha=0.1, 0.5)..."
echo "在后台运行: ./run_experiments_gpu0.sh"
./run_experiments_gpu0.sh > logs/gpu0_parallel.log 2>&1 &
GPU0_PID=$!
echo "GPU 0 进程ID: $GPU0_PID"

echo ""
echo "启动 GPU 1 上的实验 (后2组: alpha=5.0)..."
echo "在后台运行: ./run_experiments_gpu1.sh"
./run_experiments_gpu1.sh > logs/gpu1_parallel.log 2>&1 &
GPU1_PID=$!
echo "GPU 1 进程ID: $GPU1_PID"

echo ""
echo "========================================================"
echo "所有实验已在后台启动！"
echo "GPU 0 (进程 $GPU0_PID): 运行alpha=0.1和0.5的实验 (4组)"
echo "GPU 1 (进程 $GPU1_PID): 运行alpha=5.0的实验 (2组)"
echo ""
echo "查看GPU 0日志: tail -f logs/gpu0_parallel.log"
echo "查看GPU 1日志: tail -f logs/gpu1_parallel.log"
echo "查看所有实验日志: ls -la logs/*.log"
echo ""
echo "等待所有实验完成..."
echo "========================================================"

# 等待两个进程完成
wait $GPU0_PID
GPU0_STATUS=$?
wait $GPU1_PID
GPU1_STATUS=$?

echo ""
echo "========================================================"
echo "所有实验已完成！"
echo "GPU 0 退出状态: $GPU0_STATUS"
echo "GPU 1 退出状态: $GPU1_STATUS"
echo "结束时间: $(date)"
echo "========================================================"

# 检查是否有错误
if [ $GPU0_STATUS -eq 0 ] && [ $GPU1_STATUS -eq 0 ]; then
    echo "✅ 所有实验成功完成！"
    exit 0
else
    echo "❌ 部分实验失败，请检查日志文件："
    echo "   GPU 0: logs/gpu0_parallel.log"
    echo "   GPU 1: logs/gpu1_parallel.log"
    exit 1
fi