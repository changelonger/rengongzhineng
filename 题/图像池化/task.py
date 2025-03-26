import numpy as np

def average_pooling(image, pool_size, stride):
    image_height,image_width = image.shape[0],image.shape[1]
    height = (image_height-pool_size[0])//stride+1
    width = (image_width-pool_size[1])//stride+1
    print(height,width)
    # img = [[0]*height for _ in range(width)] ,不好，不如numpy
    img = np.zeros((height,width)) # 这里传元组，不然报错
    for i in range(height):
        for j in range(width):
            # 怎么算池化的初始点。
            # 对于池化后的img[i][j],第(i+1)个相当于右移移stride*i次
            # 第j+1个相当于下移j*stride
            # 所以初始点是i*stride,j*stride
            a = stride*i
            b = stride*j
            # img[i][j] = np.mean(image[a:a+stride][b:b+stride])
            img[i][j] = np.mean(image[a:a+stride,b:b+stride])
            # print(i,j)
    return img

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