import numpy as np

def average_pooling(image, pool_size, stride):
    h,w = image.shape
    nh = (h-pool_size[0])//stride+1
    nw = (w-pool_size[1])//stride+1
    res = np.zeros((nh,nw))
    for i in range(nh):
        for j in range(nw):
            # 这里推一下res[i][j]对应image[i][j]哪四个像素
            # 怎么算池化的初始点。
            # 对于池化后的img[i][j],第(i+1)个相当于右移移stride*i次
            # 第j+1个相当于下移j*stride
            # 所以初始点是i*stride,j*stride
            a = stride*i
            b = stride*j
            res[i][j] = np.mean(image[a:a+pool_size[0],b:b+pool_size[1]])
    return res

def main():
    # 示例图像
    image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    # 池化窗口的尺寸和步长
    pool_size = (2, 2)
    stride = 2

    # 执行平均池化
    pooled_image = average_pooling(image, pool_size, stride)
    print(pooled_image)

if __name__ == '__main__':
    main()