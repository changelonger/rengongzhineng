#task-start
import numpy as np 
import pandas as pd 
import json
import cv2
import os
import torch
import torchvision.transforms as transforms
def img_processor(data_path, dst_size = (224,224)):
    Image_std = [0.229, 0.224, 0.225]
    Image_mean = [0.485, 0.456, 0.406]
    _std = np.array(Image_std).reshape((1,1,3))
    _mean = np.array(Image_mean).reshape((1,1,3))
    image_src = cv2.imread(data_path)
    #TODO
    # 调整当前image_src的尺寸
    image = cv2.resize(image_src,(256,256))
    
    # 计算图像调整的初始像素及结束像素值，注意opencv的像素点位置坐标为先行后列(y,x)且初始为0
    startx = (image.shape[1] - dst_size[1])//2 # (256-224)//2 = 16
    starty = (image.shape[0] - dst_size[0])//2 
    endx = startx + dst_size[1] # 16 + 224 = 240
    endy = starty + dst_size[0]

    # 使用索引提取的方式完成图像截取[16,16:240,240]
    image = image_src[starty:endy,startx:endx]

    # 进行标准化

    image = (image - _mean)/ _std

    # ！！！一定要用处理后的image保存替换原图像，可能绝大多数未能通过都在这一步，编题人就不能说清楚要求吗，
    # 提交了50+次不通过 T_T
    cv2.imwrite(data_path,image)

    return image_src, image, (startx,starty)


def simple_generator(data_list, json_file, dst_size = (224, 224)):
    with open(json_file, 'r') as f:
        data = json.load(f)
    folder_map = {v[0]: (int(k), v[1]) for k,v in data.items()}

    for img_path in data_list:
        image_src, image, (startx,starty) = img_processor(img_path, dst_size)
        label = folder_map[img_path.split('/')[-2]]
        yield image_src, image, label, (startx,starty)

def main():
    Image_path = 'Imagedata/images'
    Json_path = 'Imagedata/image_class_index.json'

    data_list = []
    for dirname, _, filenames in os.walk(Image_path):
        if os.path.basename(dirname).startswith('n'):
            for filename in filenames:
                data_list.append(os.path.join(dirname, filename))

    # 创建生成器
    generator = simple_generator(data_list, Json_path)

    # 查看示例，检查图像、标签等属性的正确性
    num_samples = 5
    for _ in range(num_samples):
        image_src, image, label, (startx,starty) = next(generator)
        print("SrcImage shape:", image_src.shape)
        print("Image shape:", image.shape)
        print("Label:", label)
        print("startx and starty:",(startx,starty))

if __name__ == '__main__':
    main()
#task-end