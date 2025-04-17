import jieba
from gensim.models import Word2Vec

def process_dataset(file_path):
    train_data = []
    with open(file_path,"r") as f:
        for sentences in f.readlines():
            for s in sentences.split("。"):
                train_data.append(jieba.lcut(s.strip()))
    return train_data

def train_w2v_model(train_data, model_path="word2vec_model.bin"):
    model = Word2Vec(train_data, vector_size=100, workers=1,min_count=1)
    model.save(model_path)

def calculate_similarity(word1, word2, model_path="word2vec_model.bin"):
    model = Word2Vec.load(model_path)
    if word1 not in model.wv.key_to_index or word2 not in model.wv.key_to_index:
        return None
    similarity = model.wv.similarity(word1, word2)
    return similarity
def main():
    train_file_path = "w2v_train_data.txt"
    train_data = process_dataset(train_file_path)
    train_w2v_model(train_data)
    word1,word2 = "春节","假期"
    similarity = calculate_similarity(word1, word2)
    print(similarity)

if __name__ == '__main__':
    main()