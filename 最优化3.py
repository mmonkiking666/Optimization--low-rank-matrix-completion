import pandas as pd
import numpy as np


def convert_to_npz():
    print("1. 正在读取 ratings_clean.csv ...")
    # 读取刚才生成的清洗后的 csv
    df = pd.read_csv('ratings_clean.csv')

    print("2. 正在转换数据格式...")
    # 重新映射 ID 到 0 ~ N-1
    user_ids = df['UserID'].unique()
    movie_ids = df['MovieID'].unique()

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    movie_map = {mid: i for i, mid in enumerate(movie_ids)}

    # 转换为更省内存的数组
    row = df['UserID'].map(user_map).values.astype(np.int32)
    col = df['MovieID'].map(movie_map).values.astype(np.int32)
    data = df['Rating'].values.astype(np.float32)

    shape = (len(user_ids), len(movie_ids))

    print(f"3. 矩阵维度: {shape}")
    print("4. 正在保存为 compressed_matrix.npz ...")

    # 保存为压缩二进制
    np.savez_compressed(
        'compressed_matrix.npz',
        row=row, col=col, data=data, shape=shape
    )
    print("✅ 转换完成！文件名为: compressed_matrix.npz")
    print("这个文件非常小（约30-40MB），便于传输或快速加载。")


if __name__ == "__main__":
    convert_to_npz()