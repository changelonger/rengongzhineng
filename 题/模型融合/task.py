import numpy as np
import tensorflow as tf
import pickle
import jieba
import os
os.environ["OMP_NUM_THREADS"] = "1"
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def text_to_sequence(text, file="vectorizer.pkl"):
    # 第一步加载文件
    with open(file,'rb') as f:
        vectorizer = pickle.load(f)
        vocab = vectorizer.vocabulary_
    # 词表得到单词->索引
    word2idx = {word:idx for idx,word in enumerate(vocab)}
    # 分割
    words = jieba.cut(text)
    seq = [word2idx.get(word,0) for word in words]
    # jieba.cut会返回一个只能迭代一次的对象
    seq2 = seq[:100]+[0]*(100-len(seq))
    seq3 = seq[:50]+[0]*(50-len(seq))
    seq1 = vectorizer.transform([' '.join(words)])
    return seq1, seq2, seq3
def predict_text(seq1, seq2, seq3, model1='classifier_model1.pkl', model2='classifier_model2.h5',model3='classifier_model3.h5'):
    #TODO 
    # 加载分类模型1
    with open(model1, 'rb') as file:
        classifier1 = pickle.load(file)
    # 加载分类模型2
    classifier2 = tf.keras.models.load_model(model2)
    # 加载分类模型3
    classifier3 = tf.keras.models.load_model(model3)
    
    pred1 = classifier1.predict(seq1)[0] # 获取类别为1的概率值
    
    sequence2 = np.array([seq2])
    pred2 = int(classifier2.predict(sequence2)[0][0]>= 0.5)
    sequence3 = np.array([seq3])
    pred3 = int(classifier3.predict(sequence3)[0][0]>= 0.5)
    print(pred1,pred2,pred3)
    votes = [pred1, pred2, pred3]
    ensemble_pred = int(sum(votes) >= 2)  # 投票法，判定为1的个数大于等于2则预测为1，否则预测为0
    
    return ensemble_pred

def main():
    input_text = "自由多么快乐."
    seq1, seq2, seq3 = text_to_sequence(input_text)
    prediction = predict_text(seq1, seq2, seq3)
    print(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()