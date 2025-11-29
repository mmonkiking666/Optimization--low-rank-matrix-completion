import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold
import time
import gc


# ==========================================
# 核心类：非凸矩阵分解 (Non-Convex Matrix Factorization)
# ==========================================
class NonConvexMatrixFactorization:
    def __init__(self, n_users, n_items, rank=10, lambda_reg=0.1):
        self.n_users = n_users
        self.n_items = n_items
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.U = None
        self.V = None

    def smart_initialization(self, train_matrix):
        """
        参考文献 2: Smart Initialization (Spectral Initialization)
        利用 SVD 分解零填充矩阵来寻找好的初始点
        """
        print("    > [Step 1] 执行 Smart Initialization (谱初始化)...")
        # 1. 对稀疏矩阵进行 SVD (即使缺失值视为0)
        # 注意：这里需要转换为 float 类型进行计算
        u, s, vt = svds(train_matrix, k=self.rank)

        # 2. 按照平方根分配奇异值 (Ge et al. 建议)
        s_sqrt = np.sqrt(np.diag(s))

        # 3. 初始化 U 和 V
        # U: (n_users, rank), V: (n_items, rank)
        self.U = np.dot(u, s_sqrt)
        self.V = np.dot(vt.T, s_sqrt)

    def alternating_refinement(self, train_csr, train_csc, max_iter=10):
        """
        参考文献 4: Alternating Minimization (ALS)
        交替最小化 refinement
        """
        print(f"    > [Step 2] 开始 Alternating Refinement (ALS), Rank={self.rank}...")

        n_users, n_items = train_csr.shape
        eye = np.eye(self.rank) * self.lambda_reg

        for it in range(max_iter):
            start_t = time.time()

            # --- 1. 固定 V，优化 U ---
            # 对于每个用户 u: U_u = (V_idx.T * V_idx + lambda*I)^-1 * (V_idx.T * R_u)
            # 这是一个岭回归问题
            for u in range(n_users):
                # 获取用户 u 评分过的电影索引和评分
                start_idx = train_csr.indptr[u]
                end_idx = train_csr.indptr[u + 1]

                if start_idx == end_idx: continue  # 该用户无评分

                movie_indices = train_csr.indices[start_idx:end_idx]
                ratings = train_csr.data[start_idx:end_idx]

                # 取出对应的 V 行
                V_sub = self.V[movie_indices, :]

                # 计算闭式解: (V.T * V + lambda * I) * U.T = V.T * Y
                # A = V_sub.T @ V_sub + eye # 这样写比较慢
                # Y = V_sub.T @ ratings
                # self.U[u] = np.linalg.solve(A, Y)

                # 稍微优化一点的写法
                A = np.dot(V_sub.T, V_sub) + eye
                Y = np.dot(V_sub.T, ratings)
                self.U[u] = np.linalg.solve(A, Y)

            # --- 2. 固定 U，优化 V ---
            # 对于每个电影 i: V_i = (U_idx.T * U_idx + lambda*I)^-1 * (U_idx.T * R_i)
            for i in range(n_items):
                start_idx = train_csc.indptr[i]
                end_idx = train_csc.indptr[i + 1]

                if start_idx == end_idx: continue

                user_indices = train_csc.indices[start_idx:end_idx]
                ratings = train_csc.data[start_idx:end_idx]

                U_sub = self.U[user_indices, :]

                A = np.dot(U_sub.T, U_sub) + eye
                Y = np.dot(U_sub.T, ratings)
                self.V[i] = np.linalg.solve(A, Y)

            print(f"        Iter {it + 1}/{max_iter} 完成 (耗时 {time.time() - start_t:.1f}s)")

    def predict(self, user_indices, item_indices):
        """预测评分: U_i * V_j.T"""
        pred = np.sum(self.U[user_indices] * self.V[item_indices], axis=1)
        return pred


# ==========================================
# 主程序
# ==========================================
def main():
    print("1. 正在加载数据 compressed_matrix.npz ...")
    loaded = np.load('compressed_matrix.npz')
    row = loaded['row']
    col = loaded['col']
    data = loaded['data']
    shape = tuple(loaded['shape'])

    print(f"   数据加载完毕: {shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    # 全局索引
    all_idx = np.arange(len(data))

    fold = 1
    for train_idx, test_idx in kf.split(all_idx):
        print(f"\n=== Fold {fold}/5 ===")
        t0 = time.time()

        # 准备训练数据
        train_row = row[train_idx]
        train_col = col[train_idx]
        raw_train_data = data[train_idx]

        # === 关键步骤：去均值化 ===
        global_mean = np.mean(raw_train_data)
        train_data_centered = raw_train_data - global_mean

        # 构造稀疏矩阵 (CSR 格式用于快速按行切片，CSC 格式用于快速按列切片)
        train_coo = coo_matrix((train_data_centered, (train_row, train_col)), shape=shape)
        train_csr = train_coo.tocsr()
        train_csc = train_coo.tocsc()

        # === 实例化与运行 ===
        # 参数设置：
        # Rank: 10 (参考文献建议低秩)
        # Lambda: 10.0 (ALS通常需要比Soft-Impute更大的正则化)
        model = NonConvexMatrixFactorization(shape[0], shape[1], rank=10, lambda_reg=10.0)

        # 1. Smart Initialization (谱初始化)
        # 直接传入 train_csr (scipy svds 支持稀疏矩阵)
        model.smart_initialization(train_csc.astype(float))  # 转换为float避免svds报错

        # 2. Alternating Refinement (ALS)
        # 迭代 5 次即可，ALS 收敛通常比梯度下降快
        model.alternating_refinement(train_csr, train_csc, max_iter=6)

        # === 预测与评估 ===
        test_user = row[test_idx]
        test_movie = col[test_idx]
        true_val = data[test_idx]

        # 预测偏差值
        pred_bias = model.predict(test_user, test_movie)

        # 加回均值
        pred_val = pred_bias + global_mean
        pred_val = np.clip(pred_val, 0.5, 5.0)

        rmse = np.sqrt(np.mean((true_val - pred_val) ** 2))
        print(f"Fold {fold} RMSE: {rmse:.4f} (总耗时: {time.time() - t0:.2f}s)")
        rmse_scores.append(rmse)

        # 显式清理内存
        del model, train_csr, train_csc, train_coo
        gc.collect()

        fold += 1

    print("\n" + "=" * 30)
    print(f"5-Fold CV Average RMSE: {np.mean(rmse_scores):.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()