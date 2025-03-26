import numpy as np

def laplacian_sharpen(image, alpha):
    laplacian = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])
    
    height, width = image.shape
    padded_img = np.pad(image,pad_width=1,mode='edge')
    sharpened = np.zeros_like(image, dtype=np.float32)
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            region = padded_img[i-1:i+2, j-1:j+2]
            lap_value = np.sum(region * laplacian)
            sharpened[i-1, j-1] = image[i-1, j-1] + alpha * lap_value
    sharpened = np.clip(sharpened, 0, 255).astype(image.dtype)
    return sharpened

def main():
    # 示例 5x5 的图像
    image = np.array([[100, 120, 150, 140, 130],
                      [115, 130, 160, 150, 140],
                      [130, 140, 170, 160, 150],
                      [145, 150, 180, 170, 160],
                      [160, 170, 200, 190, 180]], dtype=np.uint8)
    
    # 对图像进行锐化，alpha 设为 1.5
    sharpened_image = laplacian_sharpen(image, 1.5)

    print("锐化后的图像：")
    print(sharpened_image)

if __name__ == '__main__':
    main()