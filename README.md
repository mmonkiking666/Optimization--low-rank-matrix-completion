# Optimization--low-rank-matrix-completion
> 基于凸松弛（核范数）的大规模矩阵填充算法实现 - MovieLens 10M 数据集

## 1. 项目简介 (Introduction)

本项目旨在解决大规模推荐系统中的矩阵填充（Matrix Completion）问题。核心目标是预测用户对未观看电影的评分。

不同于传统的非凸分解方法（如 ALS 或梯度下降），本项目采用了**凸优化**的方法，通过将矩阵的**秩（Rank）**松弛为**核范数（Nuclear Norm）**，将原本 NP-Hard 的非凸问题转化为凸问题求解。项目在 **MovieLens 10M** 数据集上进行了完整验证，并使用了 **5-Fold Cross Validation (5倍交叉验证)** 作为评价标准。

## 2. 问题描述 (Problem Formulation)

### 2.1 原始问题
矩阵填充的原始目标是寻找一个秩最小的矩阵 $X$，使其在观测位置 $\Omega$ 上与真实评分矩阵 $M$ 一致：

$$
\min_{X} \text{rank}(X) \quad \text{s.t.} \quad P_{\Omega}(X) = P_{\Omega}(M)
$$

由于秩函数 $\text{rank}(X)$ 是非凸且不连续的，直接求解该问题是 NP-Hard 的。

### 2.2 凸松弛 (Convex Relaxation)
根据 *Candes & Recht* 等人的理论，**核范数（Nuclear Norm, $\|X\|_*$）** 是秩函数的最佳凸包（Convex Envelope）。因此，我们将问题转化为以下凸优化形式：

$$
\min_{X} \frac{1}{2} \|P_{\Omega}(X - M)\|_F^2 + \lambda \|X\|_*
$$

其中：
* $\|X\|_* = \sum \sigma_i(X)$ （奇异值之和）。
* $\lambda$ 是正则化参数，用于控制低秩程度。

## 3. 数据集 (Dataset)

使用 **MovieLens 10M/100k** 数据集 [GroupLens Research]。
* 
### 3.1 基本统计
* **原始数据**：71,567 用户, 10,681 电影, 10,000,054 条评分。
* **预处理后**：**69,878** 有效评分用户（即至少有过一次评分行为的用户）。
* **评分范围**：0.5 - 5.0（步长 0.5）。

### 3.2 数据稀疏性分析 (Sparsity Analysis)
推荐系统的数据通常具有极高的稀疏性。本项目的矩阵规模与其稀疏度计算如下：

* **矩阵全空间 (Total Entries)**:
  $$\text{Users} \times \text{Movies} = 69,878 \times 10,681 \approx \mathbf{7.46 \times 10^8} \text{ (7.46亿个格子)}$$
* **实际观测值 (Observed Ratings)**:
  $$\approx \mathbf{1.00 \times 10^7} \text{ (1000万条评分)}$$
* **稀疏度计算 (Sparsity Calculation)**:
  
  $$\text{Density} = \frac{10,000,054}{746,366,918} \approx \mathbf{1.34\\%}$$
  
  $$\text{Sparsity} = 100\\% - 1.34\\% = \mathbf{98.66\\%}$$

**结论**：矩阵中 **约 98.7% 的元素为空**。这意味着对于绝大多数“用户-电影”对，我们都不知道其评分。这正是本项目需要利用凸优化算法（核范数最小化）从仅有的 1.3% 信息中恢复低秩结构的原因。
## 4. 方法与实现 (Methodology)

### 4.1 核心算法：Soft-Impute
为了求解上述核范数正则化问题，使用了 **Soft-Impute** 算法 (Mazumder et al., 2010)。该算法通过迭代奇异值阈值（Iterative Singular Value Thresholding）进行更新：

1.  **SVD 分解**: 对当前填充矩阵进行奇异值分解。
2.  **奇异值收缩 (Shrinkage)**: $S_{\lambda}(\sigma) = \max(\sigma - \lambda, 0)$。
3.  **矩阵重构**: 利用收缩后的奇异值重构低秩矩阵。
4.  **观测值回填**: 保持已知评分不变，仅更新未知位置。

### 4.2 关键优化策略 (Optimizations)

由于数据集较大($70k \times 10k$)，直接计算会导致严重的内存溢出（Memory Error）。本项目实施了以下工程优化：

1.  **去均值化 (Mean Centering)**:
    * 在训练前减去全局平均分，使数据中心化。
2.  **内存压缩**:
    * 数据类型转换为 `float32` / `int32`，内存占用减半。
    * 使用二进制 `.npz` 格式存储稀疏结构，大幅提升加载速度。
    * 在迭代过程中使用 `gc.collect()` 显式管理内存。
3.  **部分 SVD (Partial SVD)**:
    * 使用 `scipy.sparse.linalg.svds` 仅计算前 $k$ 个奇异值（Rank Limit），避免全量分解。

## 5. 实验过程与参数调整 (Experiments)

### 5.1 超参数设置

经过多次实验调优，最终选定的最佳参数如下：

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| **Lambda ($\lambda$)** | **5** | 正则化系数。配合去均值化数据，较小的 $\lambda$ 保留了更多细节。 |
| **Rank Limit** | **20** | SVD截断维数。平衡了计算速度与模型表达能力。 |
| **Iterations** | **15** | 迭代次数。保证算法充分收敛。 |

### 5.2 调参记录
* *初始尝试*：初始 $\lambda=15$，未去均值。结果 RMSE $\approx$ 1.83 (严重欠拟合，预测值趋向于0)。
* *改进尝试*：引入去均值化，保持 $\lambda=15$。结果 RMSE $\approx$ 1.05。
* *最终优化*：去均值化 + 降低 $\lambda$ 至 5 + 增加迭代次数。结果 RMSE $\approx$ 0.84。

## 6. 实验结果 (Results)

采用 **5-Fold Cross Validation (5倍交叉验证)** 评估模型性能，评价指标为 RMSE (Root Mean Square Error)。

### 6.1 详细数据表

| 折数 (Fold) | RMSE | 耗时 (Seconds) |
| :--- | :--- | :--- |
| Fold 1 | 0.8418 | 77.06s |
| Fold 2 | 0.8413 | 91.41s |
| Fold 3 | 0.8403 | 84.30s |
| Fold 4 | 0.8409 | 89.56s |
| Fold 5 | 0.8407 | 103.07s |
| **平均 (Average)** | **0.8410** | **-** |

### 6.2 结果可视化

下图展示了 5 次交叉验证的 RMSE 分布，极小的方差证明了模型的鲁棒性。

![RMSE Analysis](交叉验证RMSE.png)


## 7. 参考文献 (References)

1.  **Mazumder, R., Hastie, T., & Tibshirani, R.** (2010). *Spectral Regularization Algorithms for Learning Large Incomplete Matrices*. Journal of Machine Learning Research.
2.  **Ge, R., Lee, J. D., & Ma, T.** (2016). *Matrix Completion has No Spurious Local Minimum*. NIPS.
3.  **Ge, R., Jin, C., & Zheng, Y.** (2017). *No Spurious Local Minima in Nonconvex Low Rank Problems: A Unified Geometric Analysis*. ICML.
4.  **Hastie, T., et al.** (2015). *Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares*. JMLR.
5.  **Koren, Y., Bell, R., & Volinsky, C.** (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
