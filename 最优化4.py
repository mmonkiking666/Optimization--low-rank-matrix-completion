import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold
import time
import gc


# lambda_val 默认值调为 5，rank 保持 20
def soft_impute_fast(train_row, train_col, train_data, shape, lambda_val=5, rank=20):
    # 构造稀疏矩阵
    M_sparse = coo_matrix((train_data, (train_row, train_col)), shape=shape, dtype=np.float32)

    # 转换为稠密矩阵
    Z = M_sparse.toarray()

    # 立即释放稀疏矩阵内存
    del M_sparse
    gc.collect()

    # 迭代次数设为 15，让结果更精准
    for _ in range(15):
        U, s, Vt = svds(Z, k=rank)

        # 奇异值收缩
        s_shrink = np.maximum(s - lambda_val, 0)

        # 原地更新 Z
        # Z = U * s_shrink * Vt
        Z[:] = np.dot(U * s_shrink, Vt)

        # 恢复观测值 (这一步保证了已知数据不动，只猜测未知数据)
        Z[train_row, train_col] = train_data

    return Z


def main():
    print("正在加载 compressed_matrix.npz ...")
    loaded = np.load('compressed_matrix.npz')
    row = loaded['row']
    col = loaded['col']
    data = loaded['data']
    shape = tuple(loaded['shape'])

    print(f"数据加载完毕: {shape}")

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

        # === 修改点3：去均值化 (Mean Centering) ===
        # 计算当前训练集的平均分
        global_mean = np.mean(raw_train_data)
        # 将训练数据中心化 (变成 -1.5 到 +1.5 之间)
        train_data_centered = raw_train_data - global_mean

        try:
            # 运行算法 (传入中心化后的数据)
            # lambda_val 设为 5，适合中心化后的数据
            Z_hat_bias = soft_impute_fast(
                train_row, train_col, train_data_centered, shape,
                lambda_val=5,
                rank=20
            )

            # 预测与评估
            test_row = row[test_idx]
            test_col = col[test_idx]
            true_val = data[test_idx]  # 真实值保持原始分

            # 获取预测的“偏差值”
            pred_bias = Z_hat_bias[test_row, test_col]

            # === 修改点4：加回平均分 ===
            pred_val = pred_bias + global_mean

            # 截断 (限制在 0.5 - 5.0 之间)
            pred_val = np.clip(pred_val, 0.5, 5.0)

            rmse = np.sqrt(np.mean((true_val - pred_val) ** 2))
            print(f"Fold {fold} RMSE: {rmse:.4f} (耗时: {time.time() - t0:.2f}s)")
            rmse_scores.append(rmse)

            # 清理大矩阵
            del Z_hat_bias
            gc.collect()

        except MemoryError:
            print("❌ 内存不足，跳过此 Fold。建议减少 rank 或使用部分数据。")
            break

        fold += 1

    if rmse_scores:
        print("\n" + "=" * 30)
        print(f"最终结果: 5-Fold CV RMSE = {np.mean(rmse_scores):.4f}")
        print("=" * 30)


if __name__ == "__main__":
    main()