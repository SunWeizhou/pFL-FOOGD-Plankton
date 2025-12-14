## �� 生物分类学引导模块 (Taxonomy-Aware Module) 技术说明

### 1. 动机 (Motivation)

在海洋浮游生物识别的联邦学习任务中，单纯依赖数据驱动（Data-driven）的深度学习模型面临着三大核心挑战，引入生物分类学先验（Taxonomy Prior）正是为了解决这些问题：

* **缓解长尾分布 (Mitigating Long-tailed Distribution)**:
    海洋生物数据呈极端的长尾分布。稀有物种（Tail Classes）样本极少，导致模型难以学习其特征。生物分类树提供了一种**结构化先验 (Structural Prior)**。通过层级关系，稀有物种（如某特定种的桡足类）可以“借用”同属或同科中常见物种（Head Classes）的特征信息，从而在样本稀缺的情况下也能维持较好的分类边界。

* **重塑特征空间以增强 OOD 检测 (Reshaping Feature Space for OOD Detection)**:
    传统的交叉熵损失（Cross-Entropy Loss）只关注将不同类别分开，而不关心它们“怎么分”。这可能导致特征空间中，生物学上完全不同的物种（如水母和虾）距离很近。引入分类学约束后，模型被迫学习**语义结构化 (Semantically Structured)** 的特征空间：同科同属的样本在特征空间中聚类更紧密（Compactness），不同门的样本分得更开。这种清晰的类内分布显著降低了 OOD 检测模块（FOOGD）的难度，使其更容易识别出落在低密度区域的未知物种。

* **提升联邦全局模型的泛化性 (Improving Generalization of Generic Model)**:
    在 FedRoD 架构中，通用头（Generic Head）旨在学习跨海域的通用知识。生物分类学关系是**跨海域不变 (Invariant)** 的真理。利用这一不变性作为正则化项，可以防止模型过度拟合某些海域的特有噪声（如光照、背景），迫使 Backbone 提取出更本质的生物学特征。

### 2. 生物分类代价矩阵构建 (Taxonomy Matrix Construction)

为了将生物学知识注入神经网络，我们构建了一个 $C \times C$ 的**层级代价矩阵 (Hierarchical Cost Matrix)** $\mathbf{M}$，其中 $C=54$ 为类别总数。

矩阵中的元素 $\mathbf{M}_{ij}$ 表示将第 $i$ 类（真实标签）错分为第 $j$ 类（预测标签）时的**生物学代价**。我们依据生物分类阶元（Taxonomy Ranks：界、门、纲、目、科、属、种）定义了如下距离度量 $D(i, j)$：

* **Level 0 (同种, Species)**: $D=0$。
    * *说明*: 预测完全正确，无惩罚。
* **Level 1 (同属/形态极似, Genus)**: $D=1.0$。
    * *例子*: *Acartia sp.A* (纺锤水蚤A) vs. *Acartia sp.B* (纺锤水蚤B)。
    * *逻辑*: 这种错误在生物学上是可以接受的“软错误”，惩罚较小。
* **Level 2 (同目/同纲, Order/Class)**: $D=2.0$。
    * *例子*: *Calanoid* (哲水蚤) vs. *Oithona* (剑水蚤)。两者都属于桡足类 (Copepods)，形态有共性，惩罚中等。
* **Level 3 (同门不同纲, Phylum)**: $D=3.0$。
    * *例子*: *Copepod* (桡足类) vs. *Shrimp* (虾类)。两者虽同属节肢动物门，但形态差异明显，惩罚较大。
* **Level 4 (跨门/完全不同, Kingdom/Phylum)**: $D=5.0$。
    * *例子*: *Acartia* (节肢动物) vs. *Jellyfish* (刺胞动物) 或 *Noctiluca* (原生动物)。
    * *逻辑*: 这是严重的性质错误，必须施加最大惩罚，强迫模型避免此类预测。

最终构建的矩阵 $\mathbf{M}$ 是一个对称矩阵，对角线元素为 0。

### 3. Loss 的用法与集成 (Loss Formulation & Integration)

我们将上述矩阵封装为 **Taxonomy-Regularized Loss (层级正则化损失)**，并将其专门应用于 FedRoD 的 **通用头 (Generic Head)**。

#### 数学公式
假设模型的预测输出（经过 Softmax 后）为概率分布向量 $\mathbf{p} \in \mathbb{R}^C$，真实标签为 $y$。总损失函数定义为：

$$
\mathcal{L}_{total} = \mathcal{L}_{CE}(\mathbf{p}, y) + \lambda \cdot \mathcal{L}_{Taxonomy}(\mathbf{p}, y)
$$

其中，层级正则化项 $\mathcal{L}_{Taxonomy}$ 计算的是**预期生物学代价 (Expected Biological Cost)**：

$$
\mathcal{L}_{Taxonomy}(\mathbf{p}, y) = \sum_{k=1}^{C} \mathbf{p}_k \cdot \mathbf{M}_{y, k}
$$

* $\mathbf{p}_k$: 模型预测样本属于第 $k$ 类的概率。
* $\mathbf{M}_{y, k}$: 真实类别 $y$ 与预测类别 $k$ 之间的生物学距离。
* $\lambda$: 正则化系数（在实验中设为 0.5），用于平衡分类准确度和层级结构约束。

**直观解释**: 该 Loss 惩罚模型将概率分配给那些与真实类别在生物树上距离较远的类别。例如，如果真实标签是“虾”，模型把 0.4 的概率给了“水母”（距离5.0），那么 Loss 会瞬间增加 $0.4 \times 5.0 = 2.0$，迫使模型修正参数。

#### 在 FedRoD 中的集成策略
在我们的 TR-FedRoD 框架中，采用了**解耦训练 (Decoupled Training)** 策略：

1.  **通用头 (Head-G)**: 使用 **$\mathcal{L}_{total}$ (CE + Taxonomy)** 进行训练。
    * *目的*: Head-G 负责学习符合生物学逻辑的通用特征。由于梯度会回传给 Backbone，这意味着 Backbone 被迫提取出能区分“门”和“纲”的结构化特征。
2.  **个性化头 (Head-P)**: 仅使用标准的 **$\mathcal{L}_{CE}$** 进行训练。
    * *目的*: Head-P 需要保持灵活性，以拟合本地海域特定的数据分布（如特定的光照、特定的优势物种）。由于 Backbone 已经被 Head-G 优化得很好（特征空间结构清晰），Head-P 能够更高效地进行微调，从而在长尾和 Non-IID 场景下实现更高的精度。