import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
# 数据标准化（零均值单位方差）
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_standardized = (data - mean) / std
    
    # 计算协方差矩阵（特征维度间协方差）
    cov_matrix = np.cov(data_standardized.T)  # 转置使特征作为行
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 按特征值降序排列索引
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # 选择前k个主成分
    top_k_indices = sorted_indices[:k]
    principal_components = eigenvectors[:, top_k_indices]
    
    return principal_components


def main():
    # 示例输入
    data = np.array([[1, 2], [3, 4], [5, 6]])
    k = 1
    
    # 执行 PCA
    principal_components = pca(data, k)
    
    print("主成分：")
    print(np.round(principal_components, 4))  # 四舍五入到小数点后四位
    return


if __name__ == '__main__':
    main()