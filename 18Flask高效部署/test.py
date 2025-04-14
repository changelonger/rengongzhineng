from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

# app = Flask(__name__)
# # 加载预先训练好的 ONNX 模型
# session = ort.InferenceSession('mobilenetv2.onnx')

# # 定义路由'/predict'，并允许 POST 请求
# @app.route("/predict", methods=['POST'])
# def predict():
#     # 从请求中获取 JSON 格式的数据，并从中提取出输入数据
#     inputs = request.get_json()['inputs']
#     # 使用 ONNX 会话对象执行模型推理，并获取输出结果
#     output = session.run(None, {"input": inputs})[0]
#     # 将输出结果转换为列表，并封装成 JSON 格式返回给客户端
#     label = np.argmax(output, axis=1).tolist()
#     return jsonify({'label': label})


# app.run()

from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

session = ort.InferenceSession('mobilenetv2.onnx')
# 定义批次大小
BATCH_SIZE = 32

def batch_predict(inputs):
    # 初始化一个空列表来收集所有的预测结果
    output = []
    # 遍历输入数据的索引，步长为 BATCH_SIZE
    for i in range(0, len(inputs), BATCH_SIZE):
        # 使用切片来获取当前批次的输入数据
        batch_inputs = inputs[i:i+BATCH_SIZE]
        batch_outputs = session.run(None, {"input": batch_inputs})[0]
        # 将当前批次的预测结果添加到总结果列表中
        output.extend(batch_outputs)
    # 返回所有批次的预测结果
    return output

@app.route("/predict", methods=['POST'])
def predict():

    inputs = request.get_json()['inputs']
    output = batch_predict(inputs)
    label = np.argmax(output, axis=1).tolist()
    return jsonify({'label': label})


app.run()