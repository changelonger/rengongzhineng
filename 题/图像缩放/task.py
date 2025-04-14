import numpy as np
def nearest_neighbor_interpolation(image, new_width, new_height):
    old_height, old_width = image.shape
    sx = old_width / new_width
    sy = old_height / new_height
    res = np.empty((new_height, new_width), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            res[y][x] = image[int(sy*y)][int(sx*x)]

    return res

def main():
    image = np.array([[56, 23, 15, 1, 3],
                      [65, 32, 78, 255, 0], 
                      [12, 45, 62, 1, 128],
                      [255, 0, 0,0,1],
                      [1, 128, 255,0,1]], dtype=np.uint8)
    resized_image = nearest_neighbor_interpolation(image, 10, 10)
    
    #img_resize = cv2.resize(image, (10, 10), interpolation=cv2.INTER_NEAREST)
    print(resized_image.shape)
    print(resized_image)

if __name__ == '__main__':
    main()
    