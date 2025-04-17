import numpy as np

def mean_filter(image, kernel_size):
    h,w = image.shape
    r = kernel_size//2
    res = np.empty((h,w))
    for i in range(0,h):
        for j in range(0,w):
            # 上下左右四个位置
            # 判断是否超过边界，然后给范围
            u = i-r if i>=r else 0
            d = i+r if i+r<h else h-1  # 这里可以取到
            l = j-r if j>=r else 0
            ri = j+r if j+r<h else h-1
            num = (d-u+1)*(ri-l+1)
            # print(i,j,f'对应的坐标范围是:{u}:{d+1},{l}:{ri+1}')
            # print(i,j,f'对应的数量是{num}')
            # print(i,j,f'对应的数是{image[u:d+1,l:ri+1]}')
            # print(i,j,f'对应的和是{np.sum((image[u:d+1,l:ri+1]))}')
            
            res[i][j] = np.sum((image[u:d+1,l:ri+1]))/num
    return res
def main():
    # 示例图像数据，一个4x4的numpy数组
    image_data = np.array([[10, 20, 30, 40],
                        [50, 60, 70, 80],
                        [90, 100, 110, 120],
                        [130, 140, 150, 160]])

    # 应用均值滤波
    filtered_image = mean_filter(image_data, 3)  # 使用3x3的滤波器
    print("原始图像:\n", image_data)
    print("滤波后的图像:\n", filtered_image)
if __name__ == '__main__':
    main()

