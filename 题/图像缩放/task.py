import numpy as np

def nearest_neighbor_interpolation(image, new_width, new_height):
    h,w = image.shape
    sh,sw  = h/new_height,w/new_width
    res = np.empty((new_height,new_width),dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            # print(i,j)
            ii = int(i*sh)
            jj = int(j*sw)
            res[i][j] = image[ii][jj]
    return res

def main():
    #5X5 的图像
    image = np.array([[56, 23, 15, 1, 3],
                      [65, 32, 78, 255, 0], 
                      [12, 45, 62, 1, 128],
                      [255, 0, 0,0,1],
                      [1, 128, 255,0,1]], dtype=np.uint8)
    #放大为 10X10
    resized_image = nearest_neighbor_interpolation(image, 10, 10)
    print(resized_image)

if __name__ == '__main__':
    main()

