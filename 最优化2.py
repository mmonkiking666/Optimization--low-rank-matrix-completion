import pandas as pd

def clean_file():
        print("正在读取 ratings.dat ...")
        # 1. 读取数据，只取前三列 (UserID, MovieID, Rating)
        df = pd.read_csv(
            'C:/Users/22939/Desktop/pythonProject1/ratings.dat',
            sep='::',
            engine='python',
            header=None,
            usecols=[0, 1, 2],  # 只取前3列，忽略第4列 Timestamp
            names=['UserID', 'MovieID', 'Rating']
        )

        print(f"读取完成，共有 {len(df)} 条数据。")

        # 2. 保存为新的 CSV 文件
        # index=False 表示不保存行号，进一步省空间
        output_file = 'ratings_clean.csv'
        print(f"正在保存为 {output_file} ...")

        df.to_csv(output_file, index=False)

        print("✅ 处理完成！")
        print("处理后的文件已经保存为 ratings_clean.csv ")

if __name__ == "__main__":
        clean_file()