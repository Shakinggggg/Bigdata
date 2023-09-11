import numpy as np

# 判断矩阵
matrix = np.array([
    [1, 3, 2, 2, 2, 3, 3, 2, 1],
    [1/3, 1, 1/2, 1/2, 1/2, 2/3, 1, 1/2, 1/3],
    [1/2, 2, 1, 1, 1, 2/3, 1, 1/2, 1/2],
    [1/2, 2, 1, 1, 1, 2/3, 1, 1/2, 1/2],
    [1/2, 2, 1, 1, 1, 2/3, 1, 1/2, 1/2],
    [1/3, 3/2, 3/2, 3/2, 3/2, 1, 2/3, 3/2, 1],
    [1/3, 1, 1, 1, 1, 3/2, 1, 1, 1/2],
    [1/2, 2, 2, 2, 2, 2/3, 1, 1, 1/2],
    [1, 3, 2, 2, 2, 1/2, 2, 2, 1]
])

# 计算特征向量
eigenvalues, eigenvectors = np.linalg.eig(matrix)
# 获取最大特征值的索引
max_eigenvalue_index = np.argmax(eigenvalues)

# 获取对应的特征向量
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

# 归一化特征向量以获得权重
weights = max_eigenvector / sum(max_eigenvector)
for i, weight in enumerate(weights):
    print(f"Dimension {i + 1} Weight: {weight:.2f}")
# 计算一致性指标CI
n = len(matrix)  # 判断矩阵的维度
average_max_eigenvalue = sum(eigenvalues) / n
CI = (average_max_eigenvalue - n) / (n - 1)

# 随机一致性指标RI
RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

# 计算一致性比率CR
CR = CI / RI[n - 1]

# 打印CR值
print(f"Consistency Ratio (CR): {CR:.4f}")

# 判断CR是否小于阈值
threshold = 0.1
if CR < threshold:
    print("The consistency of the judgment matrix is acceptable.")
else:
    print("The consistency of the judgment matrix needs to be reassessed.")
