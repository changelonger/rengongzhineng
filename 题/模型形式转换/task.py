#task-start
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512, num_classes=2):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):

        embedded = self.embedding(text)
        packed_output, (hidden, cell) = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


def convert():
    """
    ：PyTorch的模型参数（权重和偏置）通过 state_dict 保存，而模型结构由代码中的类定义决定。当保存模型时，通常只保存参数（而非整个模型对象）。因此，加载时需先创建一个与保存时结构​​完全一致​​的模型实例，再将参数加载到该实例中。
    """
    model = TextClassifier()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    x = torch.ones([256,1],dtype=int)
    onnx_model = torch.onnx.export(model,x,'text_classifier.onnx',input_names=['input'],output_names=['output'])

def inference(model_path, input):
    # TODO
    input = np.array(input,dtype=np.int64)
    input = np.pad(input,(256-len(input),0))
    data = np.reshape(input,(-1,1))
    model1 = ort.InferenceSession(model_path)
    result = model1.run(input_feed={'input':data},output_names=['output'])[0]
    return result


def main():
    convert()
    result = inference('/home/project/text_classifier.onnx', [101, 304, 993, 108,102])
    print(result)


if __name__ == '__main__':
    main()
#task-end