#task-start
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)
model = torch.jit.load('ner.pt')
model.eval()
index2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}

# 创建处理函数。这个函数接收一个包含多个句子的列表（每个句子都是一个整数列表），然后用模型处理这些输入，最后将处理结果转换成我们需要的格式。
def process(inputs):
    results = []  # 这个列表将存储所有句子的处理结果

    # 用模型进行预测
    outputs = model(torch.tensor(inputs)).detach().numpy()

    # 对于每一条输出结果，转换为实体标注列表。
    for output in outputs:
        output_labels = [index2label[o] for o in output]  # 将类别编号转换为标签

        # 为每一条句子创建一个实体列表，
        entities = []
        # 遍历标签，寻找实体的起始和结束位置
        for i in range(len(output_labels)):
            if output_labels[i].startswith('B-'):  # 找到一个实体的开始
                j = i + 1
                while j < len(output_labels) and output_labels[j].startswith('I-'):  # 找到实体的结束位置
                    j += 1
                # 检查实体是否包含多于一个标签，如果不是，则忽略
                if j - i > 1:
                    entities.append({
                        'start': i,  # 实体的开始位置
                        'end': j - 1,  # 实体的最后一个位置
                        'label': output_labels[i][2:]  # 实体的类型
                    })
        results.append(entities)  # 将这个句子的实体列表添加到处理结果中

    # 最后返回处理结果
    return results


@app.route('/ner', methods=['POST'])
def ner():

    data = request.get_json()
    inputs = data['inputs']
    outputs = process(inputs)
    return jsonify(outputs)


if __name__ == '__main__':
    app.run(debug=True)
#task-end