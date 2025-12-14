# Taxonomy-Guided Federated Learning for Robust Plankton Recognition (TR-FedRoD)


## �� 项目简介 (Introduction)

本项目提出了一种**基于生物分类学引导的联邦学习框架 (TR-FedRoD)**，专为海洋浮游生物识别场景设计。针对海洋监测中面临的**非独立同分布 (Non-IID)**、**长尾分布 (Long-tailed Distribution)** 以及**新物种 (Out-of-Distribution, OOD)** 频发的挑战，我们实现了一个有机统一的解决方案。

**核心创新点：**

1.  **TR-FedRoD (Taxonomy-Regularized FedRoD)**: 在经典的 FedRoD 算法基础上，引入了**生物分类树 (Taxonomy Tree)** 作为先验知识。通过在通用头 (Generic Head) 上施加层级正则化损失 (Hierarchical Loss)，强迫模型学习具有生物学语义的特征空间，显著提升了**尾部稀有物种**的识别精度。
2.  **协同 OOD 检测 (Synergistic OOD Detection)**: 集成了 **FOOGD** (Federated OOD Generalization/Detection) 模块。得益于 TR-FedRoD 优化后的特征空间（类内更紧凑，类间更分离），基于 Score Matching 的 OOD 检测器能够更精准地拒识未知物种。

-----

## ��️ 系统架构 (Architecture)

本框架包含三个核心组件，协同工作以实现稳健的浮游生物识别：

1.  **Backbone (DenseNet169)**: 共享特征提取器，由 Head-G 的层级损失引导，提取具有生物分类学意义的特征。
2.  **Dual-Head Classifier**:
      * **Generic Head (Head-G)**: 负责学习通用的生物学知识，受 `TaxonomyLoss` 约束。
      * **Personalized Head (Head-P)**: 负责适应各海域（Client）特定的数据分布，解决 Non-IID 问题。
3.  **FOOGD Module**: 一个基于能量/分数的生成模型，用于在测试阶段检测未知物种（OOD）。

-----

## �� 项目结构 (Structure)

```text
TR-FedRoD-Plankton/
├── data/                   # 数据集目录
│   ├── D_ID_train/         # 分布内 (ID) 训练数据
│   ├── D_ID_test/          # 分布内 (ID) 测试数据
│   ├── D_Near_test/        # 近似 OOD 测试数据 (Near-OOD)
│   └── D_Far_test/         # 远距离 OOD 测试数据 (Far-OOD)
├── models.py               # 模型定义 (DenseNet, FedRoD, TaxonomyLoss)
├── client.py               # 联邦客户端逻辑 (本地训练, 开关控制)
├── server.py               # 联邦服务端逻辑 (参数聚合, 全局评估)
├── data_utils.py           # 数据加载与分类树矩阵构建 (build_taxonomy_matrix)
├── train_federated.py      # 训练主入口脚本
├── requirements.txt        # 依赖库
└── README.md               # 项目说明文档
```

-----

## �� 快速开始 (Quick Start)

### 1\. 环境准备

推荐使用 Conda 创建虚拟环境：

```bash
conda create -n plankton_fl python=3.8
conda activate plankton_fl
pip install -r requirements.txt
```

### 2\. 数据准备

请确保您的数据按照以下结构放置在 `data/` 目录下（基于 DYB-PlanktonNet 格式）：

  * **54 类 ID 物种**: 包含 *Acartia*, *Polychaeta*, *Calanoid* 等。
  * **Near OOD**: 形态相似但未见过的物种（如各类幼体）。
  * **Far OOD**: 非生物颗粒、气泡等干扰项。

### 3\. 运行实验

本项目支持通过命令行参数灵活切换算法模式，便于进行消融实验 (Ablation Study)。

#### ✅ 运行本文提出的方法 (TR-FedRoD)

启用分类树引导 (`--use_taxonomy`) 和 OOD 检测模块 (`--use_foogd`)：

```bash
python train_federated.py \
    --algorithm fedrod \
    --use_taxonomy \
    --use_foogd \
    --alpha 0.1 \
    --n_clients 3 \
    --epochs 50 \
    --gpu 0
```

  * `--use_taxonomy`: 开启层级正则化，优化长尾识别。
  * `--alpha 0.1`: 模拟极度 Non-IID 的海域分布差异。

#### ⚖️ 运行基线方法 (Standard FedRoD)

关闭分类树引导，仅使用标准的 FedRoD + FOOGD：

```bash
python train_federated.py \
    --algorithm fedrod \
    --use_foogd \
    --alpha 0.1 \
    --gpu 0
```

#### �� 运行 FedAvg 基线

```bash
python train_federated.py --algorithm fedavg --use_foogd --alpha 0.1
```

-----

## �� 评估指标 (Evaluation)

模型训练完成后，会自动在测试集上评估以下关键指标：

1.  **Personalized Accuracy (Head-P)**: 衡量模型在各海域本地部署时的识别精度（**核心指标**）。
2.  **Tail-Class Accuracy**: 专门评估样本稀缺类别（长尾部分）的识别率，用于验证 Taxonomy 的作用。
3.  **OOD Detection AUROC**: 衡量模型区分已知物种和未知物种的能力。
4.  **Hierarchical Error**: 预测错误在生物分类树上的平均距离（验证模型是否犯了“离谱”的错误）。

-----

## �� 方法论细节 (Methodology)

### Taxonomy-Aware Loss

为了缓解长尾分布带来的偏差，我们在通用头 (Head-G) 上引入了层级代价矩阵 $\mathbf{M}$：

$$
\mathcal{L}_{total} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda \sum_{k} \hat{p}_k \cdot \mathbf{M}_{y,k}
$$

其中 $\mathbf{M}_{y,k}$ 定义了真实类别 $y$ 与预测类别 $k$ 之间的生物学距离（如同属距离为1，跨门距离为5）。这迫使模型即使预测错误，也倾向于预测为生物学上相近的物种。

### FOOGD for OOD Detection

我们利用 **Langevin Dynamics** 和 **Score Matching** 在特征空间中估计数据密度。由于 Taxonomy Loss 使得 ID 数据的特征分布更加紧凑（Compactness），FOOGD 能更敏锐地检测出落在低密度区域的 OOD 样本。

-----



