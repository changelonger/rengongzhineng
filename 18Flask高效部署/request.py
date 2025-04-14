import requests
import numpy as np

# 初始化一个 200 个样本数量的随机图片数据
input_data = {"inputs": np.empty((200, 3, 224, 224)).tolist()}

r = requests.post('http://localhost:5000/predict', json=input_data)
print(r.json())