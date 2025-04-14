# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def welcome_lanqiao():
#     return "欢迎参加蓝桥杯"

# if __name__ == '__main__':
#     # 开启调试模式（开发时使用，生产环境务必关闭）
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# import onnxruntime as ort

# app = Flask(__name__)
# # 加载预先训练好的 ONNX 模型
# session = ort.InferenceSession(r'D:\CODE\rengongzhineng\Falsk部署\deploy.onnx')

# @app.route("/")
# def welcome_lanqiao():
#     return "欢迎参加蓝桥杯"

# # 定义路由'/predict'，并允许 POST 请求
# @app.route("/predict", methods=['POST'])
# def predict():
#     # 从请求中获取 JSON 格式的数据，并从中提取出输入数据
#     inputs = request.get_json()['inputs']
#     # 使用 ONNX 会话对象执行模型推理，并获取输出结果
#     output = session.run(None, {"input": inputs})[0]
#     # 将输出结果转换为列表，并封装成 JSON 格式返回给客户端
#     return jsonify({'output': output.tolist()})


# app.run()

from flask import Flask, request, jsonify
import onnxruntime as ort
import threading
app = Flask(__name__)
session = ort.InferenceSession(r'D:\CODE\rengongzhineng\Falsk部署\deploy.onnx')
model_lock = threading.Lock()  # 线程锁

@app.route('/')
def welcome_lanqiao():
    return '欢迎参加蓝桥杯'

@app.route('/predict',methods = ['POST'])
def predict():
    global session
    inputs = request.json['inputs']
    with model_lock:
        output = session.run(None,{'input':inputs})[0]
    return jsonify({'output':output.tolist(),'model_path':session._model_path})

@app.route('/update_model',methods = ['POST'])
def update_model():
    global session
    new_model_path = request.json['model_path']

    with model_lock:
        session = ort.InferenceSession(new_model_path)
    return '模型更新成功'
