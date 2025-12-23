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
### 4.4 进阶实验：良态景观理论验证与梯度下降的局限性

为了验证非凸优化领域的“良态景观（Benign Landscape）”理论 [Ge et al., 2016]，我们设计了一个基于 **随机初始化 (Random Init)** 和 **扰动梯度下降 (Perturbed Gradient Descent)** 的对比实验。

#### 4.4.1 实验目的与设置
理论指出，低秩矩阵分解的优化景观中没有虚假局部极小值，且鞍点具有负曲率。这意味着不依赖精心设计的谱初始化，仅凭随机初始化配合鞍点逃离机制（Saddle-point Escaping）也能收敛。

* **算法:** 全量梯度下降 (Full GD) + 梯度裁剪 (Gradient Clipping) + 噪声注入。
* **初始化:** 随机高斯分布 $N(0, 1/\sqrt{r})$，模拟无先验知识场景。
* **参数:** $\text{Rank}=10, \text{LR}=0.0001, \text{Clip Norm}=100.0$。
    * *注：为了防止上一轮实验中出现的梯度爆炸问题，我们大幅降低了学习率并增加了梯度裁剪。*

#### 4.4.2 实验结果 (30 Iterations)
运行日志显示，在强约束条件下，算法成功避免了数值发散，但陷入了极慢的收敛区域：

| Iteration | Status | RMSE | Raw Grad Norm | Time |
| :--- | :--- | :--- | :--- | :--- |
| Iter 01 | Normal GD | 1.1053 | 22279.0 | 1.52s |
| Iter 10 | Normal GD | 1.1050 | 22171.5 | 1.44s |
| Iter 20 | Normal GD | 1.1048 | 22052.5 | 1.49s |
| Iter 30 | Normal GD | **1.1045** | 21934.0 | 1.42s |

* **最终 RMSE:** 1.1045 (远差于 ALS 的 0.8027)
* **梯度状态:** `Raw Grad` 维持在 $2.2 \times 10^4$ 量级，远高于噪声注入阈值，说明模型并未卡在鞍点，而是在一个非常陡峭的峡谷中以极小的步长缓慢下降。

#### 4.4.3 深度讨论：GD vs. ALS
本次实验深刻揭示了理论与工程实践的差距：

1.  **收敛效率差异:**
    * **ALS (交替最小二乘):** 利用了问题的子结构（固定一侧即为凸问题），每一步都能求得闭式解（Closed-form Solution），相当于在优化曲面上“大步跳跃”，仅需 6 步即可收敛到 RMSE 0.80。
    * **GD (梯度下降):** 面对大规模稀疏矩阵，全量梯度的数值极其巨大。为了保证数值稳定性（不发生梯度爆炸），必须将学习率设得极小（$10^{-4}$ 级别）。这导致模型如蜗牛般爬行，收敛效率极低。

2.  **良态景观的现实意义:**
    理论上的“良态”保证了**最终**能找到全局最优，但并未保证**收敛速度**。实验证明，虽然随机初始化没有导致算法卡死，但在同样的计算时间预算下，ALS 的效率比经过精细调参（Clipping+LR Decay）的 GD 高出几个数量级。

**结论：** 在大规模推荐系统场景下，尽管良态景观在数学上有较好的理论保证，但 **ALS 在工程实现上具有压倒性的鲁棒性和效率优势**。


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

![Comparison Figure1](rmse_convergence.png)
* 从迭代收敛曲线图可以看出，非凸方法的收敛明显更快；且收敛的RMSE更低
2.  **预测值分布直方图**

![Comparison Figure2](error_distribution_comparison.png)
* 从预测值分布直方图可以看出，非凸方法的预测精度更高，偏差更小。

3.  **奇异值分布谱**

![Comparison Figure3](spectrum_comparison.png.png)
* 从奇异值分布谱可以看出，初始观测矩阵的奇异值分布确实是近似低秩的，这也进一步证明了截断操作的合理性。（由于初始矩阵规模太大无法直接进行全SVD，故采用稀疏SVD）
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
      
3.  **结论**：非凸方法的内存效率是凸方法的 **近 1000 倍**，这使其能够轻松扩展到亿级用户规模的电影评分推荐系统中。

## 6. 扩展研究：MovieLens 20M 数据集 (Extended Research on ML-20M)

为了验证算法在大规模数据下的扩展性与鲁棒性，我们将实验规模扩大至 **MovieLens 20M** 数据集。该实验深刻揭示了凸优化方法在内存与计算上的瓶颈，以及非凸方法在巨大数据规模下的压倒性优势。

### 6.1 数据集规格与挑战
* **用户数 (Users):** 138,493
* **电影数 (Movies):** 26,744
* **总评分数:** $\approx 2.00 \times 10^7$
* **挑战:**
    * **维度爆炸:** 稠密矩阵大小为 $138,493 \times 26,744$。若使用 `float32` 存储，仅构建该矩阵就需要约 **14.8 GB** 内存，这已逼近普通单机（16GB RAM）的物理极限。
    * **计算墙:** 标准 SVD 在此维度下的计算成本呈立方级增长。

### 6.2 实验结果 (Train/Test Split)
考虑到 20M 数据集的计算成本，本环节采用 **80% 训练 / 20% 测试** 的划分方式进行评估。

| 算法类型 | 具体方法 | 迭代次数 | RMSE (误差) | 总耗时 (Time) | 内存瓶颈 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **凸优化** | Standard Soft-Impute | 5 | 0.8803 | **13,825.3s** (~3.8小时) | **极高** (需频繁 Swap 交换内存) |
| **凸优化** | Soft-Impute + **RSVD** | 5 | 0.8817 | **3,883.5s** (~1.1小时) | 高 (仍需构建稠密矩阵) |
| **非凸优化** | **Non-Convex ALS** | 5 | **0.7943** | **93.6s** (~1.5分钟) | **极低** (纯稀疏运算) |

### 6.3 结果深度剖析

**1. 凸方法的“内存墙”与效率崩塌**
* **标准 Soft-Impute:** 运行耗时近 4 小时。虽然代码中尝试进行内存监控，但由于需要构建 14GB+ 的稠密矩阵 $Z$，导致操作系统频繁进行磁盘交换，计算效率急剧下降。
* **RSVD (随机SVD) 的优化:** 引入 Randomized SVD 后，时间缩短至约 1 小时，速度提升约 **3.5倍**。但算法依然受限于 $Z$ 矩阵的重构步骤，内存占用并未本质减少。

**2. 非凸方法的“降维打击”**
* **速度奇迹:** 非凸 ALS 方法仅耗时 **93.6秒**，相比标准凸方法快了 **147倍**，相比 RSVD 优化版快了 **41倍**。这得益于其从未构建稠密矩阵，始终在稀疏格式下运算。
* **精度更优:** 在 20M 数据集上，非凸方法的 RMSE (**0.7943**) 依然显著低于凸方法 (**0.88xx**)，再次印证了在数据量充足时，非凸模型对数据的拟合能力更强。

**3. 结论**
从 10M 到 20M 的跨越证明：**随着数据规模的增长，非凸优化方法的优势从“量变”转为“质变”。** 对于实际使用的大规模用户推荐系统，基于 ALS 的非凸分解是唯一具备可行性的单机解决方案。
## 7. 参考文献 (References)

1.  **Mazumder, R., Hastie, T., & Tibshirani, R.** (2010). *Spectral Regularization Algorithms for Learning Large Incomplete Matrices*. Journal of Machine Learning Research.
2.  **Ge, R., Lee, J. D., & Ma, T.** (2016). *Matrix Completion has No Spurious Local Minimum*. NIPS.
3.  **Ge, R., Jin, C., & Zheng, Y.** (2017). *No Spurious Local Minima in Nonconvex Low Rank Problems: A Unified Geometric Analysis*. ICML.
4.  **Hastie, T., Mazumder, R., Lee, J. D., & Zadeh, R.** (2015). *Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares*. JMLR.
5.  **Koren, Y., Bell, R., & Volinsky, C.** (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
