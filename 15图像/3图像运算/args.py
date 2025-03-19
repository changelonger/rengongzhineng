import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
# 使用 add_argument 添加命令行参数，
# 这里和之前学习的有点不同，这个函数多了一个 action = "append" 参数，
# 这个参数将允许我们多次使用 --image 命令读取多张图片。
# 然后 parse_args 将命令解析并存储到列表中。
ap = argparse.ArgumentParser()
ap.add_argument('--image',action = 'append',type = str,help = 'image2.path')

args = ap.parse_args()
"""
接下来我们就可以使用从命令行获取到的图片路径读取图片了。
首先我们用 if 语句判断是否获取到图片，如果我们获取到了图片路径，
则 args.image 为真并执行后面的代码。
然后我们再次用 if 语句判断是否获取到了多张图片路径，
本次实验我们将用到两张图片，所以这里会获得 image[0] 和 image[1] 两张图片。
如果只获取一张图片路径，则将只读取 args.image[0] 这一张图片。
"""
if args.image:
    if len(args.image) > 1:
        image1 = cv2.imread(args.image[0])
        image2 = cv2.imread(args.image[1])
    else:
        image = cv2.imread(args.image[0])