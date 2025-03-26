import numpy as np

def divide_on_feature(X, feature_i, threshold):
    # feature_i（整数）：表示用于划分的特征索引。
    # 这个表示根据哪一个数据划分，比如3,10,feature是0就划分成小的，1就话分成大的
    s1 = []
    s2 = []
    for x in X:
        if x[feature_i]>=threshold:
            s1.append(x)
        else:
            s2.append(x)
    return [np.array(s1), np.array(s2)]  # 这里为什么一定要转换成numpy的数组?因为X就是numpy的类型
def main():
    # 示例数据集
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
    feature_i = 0
    threshold = 5
    
    # 执行数据集划分
    subsets = divide_on_feature(X, feature_i, threshold)
    
    print("Subset 1:")
    print(subsets[0])
    print("\nSubset 2:")
    print(subsets[1])
    return

if __name__ == '__main__':
    main()