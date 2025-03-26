import numpy as np

def calculate_standardized_covariance_matrix(vectors):
    data = np.array(vectors)
    
    # 1. 标准化数据
    mean = np.mean(data, axis=0)  # 计算每列均值
    std = np.std(data, axis=0, ddof=1)  # 样本标准差（分母n-1）
    
    # 处理标准差为0的情况：避免除以零
    std_adj = np.where(std == 0, 1.0, std)  # 将0标准差替换为1（避免除零错误）
    x = (data - mean) / std_adj
    
    # 2. 计算协方差矩阵
    cov_matrix = np.cov(x.T)  # 转置后每行表示一个特征
    
    # 将标准差为0的列对应的协方差设为NaN或0
    zero_std_mask = (std == 0)
    cov_matrix[zero_std_mask, :] = 0  # 该列与其他列的协方差为0
    cov_matrix[:, zero_std_mask] = 0  # 其他列与该列的协方差为0
    
    return cov_matrix
def main():
    # 示例输入
    vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    # 执行标准化协方差矩阵计算
    covariance_matrix = calculate_standardized_covariance_matrix(vectors)
    
    print("标准化协方差矩阵：")
    print(covariance_matrix)
    return

if __name__ == '__main__':
    main()