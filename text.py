import pandas as pd
import numpy as np

# 1. 读原始数据
df = pd.read_csv("2_Iris.csv")   # 换成你的文件名

# 2. 固定随机种子，保证每次生成的缺失位置一致
rng = np.random.default_rng(42)

# 3. 每列设定缺失比例
missing_ratio = {
    "SepalLengthCm": 0.05,
    "SepalWidthCm":  0.08,
    "PetalLengthCm": 0.10,
    "PetalWidthCm":  0.06,
    "Species":       0.04,
}

# 4. 逐列随机置 NaN
for col, ratio in missing_ratio.items():
    n = len(df)
    n_missing = int(n * ratio)
    idx = rng.choice(n, size=n_missing, replace=False)
    df.loc[idx, col] = np.nan

# 5. 查看缺失情况
print(df.isnull().sum())
print(df.isnull().mean().round(3))

# 6. 保存
df.to_csv("2_Iris_missing_value.csv", index=False)