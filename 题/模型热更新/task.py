from threading import Lock
from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model_lock = Lock()
# Flask 应用在启动时，读取 `svm_1.pkl` 作为初始模型。
current_model = 'svm_1.pkl'
with open(current_model,'rb') as f:
    model = pickle.load(f)
    
@app.route('/predict', methods=['POST'])
def predict():
    with model_lock:
        inputs = request.get_json()["input"]
        output = model.predict(inputs).tolist()
    return jsonify({'output':output,'current_model':current_model})
    
@app.route('/update', methods=['POST'])
def update():
    with model_lock:
        new_model = request.get_json()['model']
        if new_model in ['svm_1.pkl','svm_2.pkl','svm_3.pkl']:
            global current_model,model
            current_model = new_model
            with open(current_model,'rb') as f:
                model = pickle.load(f)
            return current_model,200
        return 'Invalid model',400


if __name__ == '__main__':
    app.run()