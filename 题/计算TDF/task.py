from collections import Counter
import math
import jieba

def calc_tfidf(words, corpus):
    #TODO
    word_counter = Counter(words)
    tfidf = {}
    total_docs = len(corpus)

    for word in set(words):
        tf = word_counter[word] / len(words)
        doc_count = 0
        for doc in corpus:
            if word in doc:
                doc_count += 1
        idf = math.log(total_docs / (doc_count+1))
        tfidf[word] = tf * idf
    return tfidf


def main():
    # 读取输入原始文本 corpus
    with open('input_a.txt', 'r', encoding='UTF-8') as f:
        corpus = f.readlines()
    # 进行分词得到新的 corpus
    corpus = [jieba.lcut(para) for para in corpus]
    # 打开输出文件
    with open('output_a.txt', 'w', encoding='UTF-8') as f:
        # 对每个词进行 TF-IDF 计算
        for i, words in enumerate(corpus, 1):
            tfidf = calc_tfidf(words, corpus)

            # 写入到文件中
            f.write(f"第{i}段文本：\n")
            top_words = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:2]
            for j, (word, score) in enumerate(top_words, 1):
                if j != len(top_words):
                    f.write(f"{word} {score:.5f}\n")
                else:
                    f.write(f"{word} {score:.5f}")
            if i != len(corpus):
                f.write("\n")

if __name__ == '__main__':
    main()