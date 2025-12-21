# Low-Rank Matrix Completion: Convex vs. Non-Convex Optimization
> 基于凸松弛（核范数）与非凸优化（ALS+谱初始化）的大规模矩阵填充算法对比研究 - MovieLens 10M 数据集

## 1. 项目背景与简介 (Introduction)

推荐系统的核心挑战之一是**矩阵填充（Matrix Completion）**，即利用极其稀疏的用户评分数据预测缺失的评分。本项目基于 **MovieLens 10M** 数据集，深入探讨并实现了解决该问题的两种主流数学优化范式：

1.  **凸优化方法 (Convex Approach)**: 基于核范数松弛 (Nuclear Norm Relaxation) 的 Soft-Impute 算法。该方法理论基础扎实，具有全局最优解。
2.  **非凸优化方法 (Non-Convex Approach)**: 基于谱初始化 (Spectral Initialization) 和交替最小化 (ALS) 的矩阵分解。该方法近年来因其计算效率高且无虚假局部极小值（No Spurious Local Minima）而受到关注。

本项目不仅复现了相关算法，还通过 **5-Fold Cross Validation** 进行了严谨的对比实验。结果表明，非凸方法在适当的初始化策略下，在计算速度和预测精度上均优于传统的凸松弛方法。

## 2. 数据集与预处理 (Dataset & Preprocessing)

本项目使用 **MovieLens 10M/100k** 数据集 [GroupLens Research]。

### 2.1 数据规模与稀疏性
* **用户数 (Users)**: 69,878
* **电影数 (Movies)**: 10,681
* **总评分数**: $\approx 1.00 \times 10^7$
* **矩阵总容量**: $\approx 7.46 \times 10^8$
* **稀疏度 (Sparsity)**: **98.66%** 的元素为空。这意味着我们仅拥有 1.34% 的信息来恢复整个矩阵。

### 2.2 数据预处理 (Preprocessing)
为了提升算法收敛速度和预测准确性，我们实施了以下预处理：
1.  **去均值化 (Mean Centering)**: 计算全局平均评分，将训练数据中心化($R_{ij} \leftarrow R_{ij} - \mu$)。这消除了数据的偏移量，使零点成为更有意义的参考点。
2.  **内存优化**: 使用 `.npz` 二进制格式存储稀疏矩阵，并将数据精度压缩为 `float32`，成功在普通单机环境下处理了大规模矩阵。

## 3. 方法一：凸优化 (Convex Optimization)

### 3.1 理论模型
原始的秩最小化问题是 NP-Hard 的。根据 Candès & Recht 的理论，**核范数(Nuclear Norm)** 是秩函数的最佳凸包。我们将问题转化为以下凸优化形式：

$$
\min_{X} \frac{1}{2} \lVert P_{\Omega}(X - M) \rVert_F^2 + \lambda \lVert X \rVert_*
$$

其中：
* $\lVert X \rVert_* = \sum \sigma_i(X)$ 是核范数（奇异值之和）。
* $\lambda$ 是正则化参数。

### 3.2 算法实现：Soft-Impute
结合参考文献 1 的成果，使用 **Soft-Impute** 算法 (Mazumder et al., 2010)，通过迭代奇异值阈值 (SVT) 求解：
1.  **SVD 分解**: 对当前填充矩阵进行奇异值分解。
2.  **收缩 (Shrinkage)**: 将奇异值减去 $\lambda$，小于 0 的截断为 0。
3.  **重构**: 利用收缩后的奇异值重构低秩矩阵。

### 3.3 实验结果 (5-Fold CV)
* **参数设置**: $\lambda=5$, Rank Limit=20, Iterations=15

| Fold | RMSE | Time (s) |
| :--- | :--- | :--- |
| Fold 1 | 0.8418 | 77.06s |
| Fold 2 | 0.8413 | 91.41s |
| Fold 3 | 0.8403 | 84.30s |
| Fold 4 | 0.8409 | 89.56s |
| Fold 5 | 0.8407 | 103.07s |
| **Average** | **0.8410** | **~89s** |

![Convex Results](交叉验证RMSE.png)
### 3.4 算法改进 (使用随机SVD方法替换SVD)
* **参数设置**: $\lambda=5$, Rank Limit=20, Iterations=15

| Fold | RMSE | Time (s) |
| :--- | :--- | :--- |
| Fold 1 | 0.8420 | 76.42s |
| Fold 2 | 0.8416 | 78.51s |
| Fold 3 | 0.8407 | 79.13s |
| Fold 4 | 0.8414 | 72.07s |
| Fold 5 | 0.8409 | 44.03s |
| **Average** | **0.8413** | **~70s** |

### 使用随机SVD方法改进后，在平均RMSE基本不变的情况下，时间快了20s，性能得到提升。
![RSVD Convex Results](RSVD交叉验证RMSE.png)


---

## 4. 方法二：非凸优化 (Non-Convex Optimization)

### 4.1 理论模型
直接对低秩因子矩阵 $U \in \mathbb{R}^{m \times r}$ 和 $V \in \mathbb{R}^{n \times r}$ 进行优化。虽然目标函数非凸，但通过特定的正则化和初始化，可以保证良好的收敛性：

$$
\min_{U, V} \sum_{(i,j) \in \Omega} (R_{ij} - U_i V_j^T)^2 + \lambda (\lVert U \rVert_F^2 + \lVert V \rVert_F^2)
$$

### 4.2 算法实现：Smart Init + ALS
结合参考文献 2, 3, 4 的最新研究成果：
1.  **谱初始化 (Spectral Initialization)**:
    * 对零填充矩阵进行 SVD，取前 $r$ 个奇异向量初始化 $U$ 和 $V$。
    * **作用**: 确保初始点落在全局最优解的“吸引域（Basin of Attraction）”内，避开鞍点。
2.  **交替最小化 (ALS)**:
    * 固定 $U$ 优化 $V$，再固定 $V$ 优化 $U$。每一步都是一个凸的岭回归（Ridge Regression）问题。

### 4.3 实验结果 (5-Fold CV)
* **参数设置**: Rank=10, $\lambda=10.0$, Iterations=6

| Fold | RMSE | Time (s) |
| :--- | :--- | :--- |
| Fold 1 | 0.8031 | 20.37s |
| Fold 2 | 0.8027 | 30.13s |
| Fold 3 | 0.8023 | 19.97s |
| Fold 4 | 0.8030 | 19.88s |
| Fold 5 | 0.8024 | 20.10s |
| **Average** | **0.8027** | **~22s** |

![Non-Convex Results](交叉验证RMSE（Non-Convex）.png)

---

## 5. 综合对比分析 (Comparison & Discussion)

通过对比两种方法的实验数据，我们得出以下关键结论：

### 5.1 性能对比表

| 比较维度 | 凸优化 (Soft-Impute) | 非凸优化 (Non-Convex ALS) | 结论 |
| :--- | :--- | :--- | :--- |
| **预测误差 (RMSE)** | 0.8410 | **0.8027** | 非凸方法精度提升约 **4.6%** |
| **计算速度 (平均/Fold)** | ~89 秒 | **~22 秒** | 非凸方法快约 **4 倍** |
| **内存消耗** | 较高 (需存储 $70k \times 10k$ 稠密矩阵) | **极低** (仅存因子矩阵 $U, V$) | 非凸方法更适合超大规模数据 |

### 5.2性能对比图
1.  **迭代收敛曲线图**

![Comparison Figure1](迭代收敛曲线图.png)

2.  **预测值分布直方图**

![Comparison Figure2](预测值分布直方图.png)


3.  **奇异值分布谱**

![Comparison Figure4](奇异值分布谱.png)

### 5.3 深度分析：为什么非凸方法更好？

1.  **偏差 (Bias) 问题**:
    * **凸方法**: 核范数正则化通过“软阈值”操作（Soft Thresholding）将所有奇异值都减去了 $\lambda$。这虽然实现了低秩，但也人为地压缩了主要成分的能量，导致预测值普遍偏低（Bias）。
    * **非凸方法**: 直接拟合数据，仅通过 Frobenius 范数约束幅度，没有强制改变奇异值的分布结构，因此在拥有足够数据时，偏差更小。

2.  **计算复杂度**:
    * **凸方法**: 每次迭代需要计算 $SVD(X)$，即使是 Partial SVD，其复杂度也接近 $O(mnk)$。
    * **非凸方法**: ALS 的每一步只需解线性方程组，且秩 $r$ 通常很小（本实验为10），计算极其高效。

3.  **优化景观 (Optimization Landscape)**:
    * 近年来 Ge et al. (2016, 2017) 的研究证明，在满足一定条件（如 RIP 性质或足够多的观测样本）下，低秩矩阵分解的非凸目标函数**没有虚假局部极小值**，且所有的鞍点都有负曲率方向。这意味着配合**谱初始化**，简单的梯度下降或 ALS 也能收敛到全局最优解。

### 5.3 内存消耗与空间复杂度分析 (Memory & Space Complexity)

除了预测精度和计算速度，**内存占用**是大规模矩阵填充在实际工程应用中的另一大瓶颈。两种方法在空间复杂度上存在本质区别：

| 方法 | 空间复杂度 (Space Complexity) | 实际内存峰值 (Estimated Peak RAM) | 说明 |
| :--- | :--- | :--- | :--- |
| **凸优化 (Soft-Impute)** | **$O(m \times n)$** | **~3.0 GB** | 算法迭代过程中需要维护稠密的重构矩阵 $X$ 或其残差矩阵，内存随数据维度平方级增长。 |
| **非凸优化 (ALS)** | **$O((m + n) \times r)$** | **< 100 MB** | 仅需存储两个低秩因子矩阵 $U$ 和 $V$。内存消耗极低，随维度线性增长。 |

**详细分析**：
1.  **凸优化 (Convex)**:
    * 为了进行奇异值分解 (SVD)，Soft-Impute 算法通常需要处理大小为 $69878 \times 10681$ 的稠密矩阵。
    * 即使使用 `float32` (4 bytes) 存储，仅存储该矩阵就需要：
        $$69,878 \times 10,681 \times 4 \text{ bytes} \approx \mathbf{2.85 \text{ GB}}$$
    * 这就是为什么在代码实现中必须极其小心地管理内存（使用 `gc.collect`）并进行数据类型压缩的原因，否则极易触发 `MemoryError`。

2.  **非凸优化 (Non-Convex)**:
    * ALS 算法利用了矩阵分解的特性，从未显式构建完整的 $m \times n$ 矩阵。
    * 当 Rank $r=10$ 时，只需存储：
        $$(69,878 + 10,681) \times 10 \times 4 \text{ bytes} \approx \mathbf{3.1 \text{ MB}}$$
      
3.  **结论**：非凸方法的内存效率是凸方法的 **近 1000 倍**，这使其能够轻松扩展到亿级用户规模的工业界推荐系统中。 
## 6. 参考文献 (References)

1.  **Mazumder, R., Hastie, T., & Tibshirani, R.** (2010). *Spectral Regularization Algorithms for Learning Large Incomplete Matrices*. Journal of Machine Learning Research.
2.  **Ge, R., Lee, J. D., & Ma, T.** (2016). *Matrix Completion has No Spurious Local Minimum*. NIPS.
3.  **Ge, R., Jin, C., & Zheng, Y.** (2017). *No Spurious Local Minima in Nonconvex Low Rank Problems: A Unified Geometric Analysis*. ICML.
4.  **Hastie, T., Mazumder, R., Lee, J. D., & Zadeh, R.** (2015). *Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares*. JMLR.
5.  **Koren, Y., Bell, R., & Volinsky, C.** (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
