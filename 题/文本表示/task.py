import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

w2v_file_path = "word2vec_model.bin"
W2V_MODEL= Word2Vec.load(w2v_file_path)
W2V_SIZE = 100

def get_w2v(word):
    if word in W2V_MODEL.wv:
        return W2V_MODEL.wv[word]
    else:
        return None



def get_sentence_vector(sentence):
    valid_vectors = []
    for word in sentence:
        word_vec = get_w2v(word)
        if word_vec is not None:
            valid_vectors.append(word_vec)

    if not valid_vectors:
        #返回形状为（100，）的全零向量
        return np.zeros(W2V_SIZE)
    else:
        #计算有效词向量的平均值
        return np.mean(valid_vectors,axis=0)

def get_similarity(array1, array2):
    array1_2d = np.reshape(array1, (1, -1))
    array2_2d = np.reshape(array2, (1, -1))
    similarity = cosine_similarity(array1_2d, array2_2d)[0][0]
    return similarity

def main():

    # 测试两个句子
    sentence1 = '我不喜欢看新闻。'
    sentence2 = '我觉得新闻不好看。'
    sentence_split1 = jieba.lcut(sentence1)
    sentence_split2 = jieba.lcut(sentence2)
    # 获取句子的句向量
    sentence1_vector = get_sentence_vector(sentence_split1)
    sentence2_vector = get_sentence_vector(sentence_split2)
    # 计算句子的相似度
    similarity = get_similarity(sentence1_vector, sentence2_vector)
    print(similarity) 

if __name__ == '__main__':
    main()