import jieba
# 评论数据
comments = [
    {"text": "这个电影真是好看了，剧情精彩，演员演技出众，特效惊人，yyds！", "language": "zh"},
    {"text": "真的很失望，剧情太拖沓了，演员演技平平，特效一般般。", "language": "zh"},
    {"text": "这本书写得棒，内容深入浅出，观点独到！", "language": "zh"},
    {"text": "这文章实在是糟糕，内容浅薄，观点站不住脚啊。", "language": "zh"},
    {"text": "I just watched the new movie and it blew my mind ! The plot twists were unexpected and kept me on the edge of my seat . The performances were outstanding , especially the lead actor . I would highly recommend it !", "language": "en"},
    {"text": "The food at this restaurant is absolutely delicious ! The flavors are well-balanced and the presentation is top-notch . The service is also excellent . I'll definitely be coming back !", "language": "en"},
    {"text": "I'm really impressed with this photo . The composition is stunning , and the lighting is perfect . It captures the beauty of the scenery beautifully . Great job , photographer !", "language": "en"}
]
# comments=[{'text':'这个电影着实精彩','language':'zh'}]

# 英文单词与其还原后的词形字典
word_lemma = {'watched': 'watch', 'blew': 'blow', 'twists': 'twist', 'were': 'be', 'kept': 'keep', 'me': 'i', 'performances': 'performance', 'is': 'be', 'flavors': 'flavor', 'are': 'be', 'coming': 'come', 'captures': 'capture'}

# 定义中文文本处理函数
def process_zh(text):
    seg_tokens = jieba.lcut(text)
    return seg_tokens

# 定义英文文本处理函数
def process_en(text):
    text_lower = text.lower()
    text_tokens = text_lower.split(" ")
    lem_tokens = [word_lemma[i] if i in word_lemma else i for i in text_tokens]
    return lem_tokens

# 读取 word2id.txt 文件并转换为字典
def read_word2id(file_path):
    word2id = {}
    with open(file_path,'r',encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        word,idx  = line.split()
        # print(word),idx
        word2id[word] = int(idx)
    return word2id
# 将分词后的列表转换为固定长度的 id 列表
def tokens_to_ids(tokens, word2id, max_len=10, unk='<unk>', pad='<pad>'):
    res = []
    for word in tokens:
        if word in word2id.keys():
            res.append(word2id[word])
        else:
            res.append(word2id[unk])
    res = res[0:max_len]+[word2id[pad]] * (max_len - len(res))
    return res

# 获取输入文本的 id 列表
def get_input_ids(word2id_file_path):
    res = []
    word2id = read_word2id(word2id_file_path)
    for i in comments:
        tokens = []
        if i["language"] == "zh":
            tokens = process_zh(i["text"])  
        elif i["language"] == "en":
            tokens = process_en(i["text"])  
        else:
            print("error input language")
        ids = tokens_to_ids(tokens, word2id)
        res.append(ids)
    return res

def main():
    word2id_file_path = 'word2id.txt'
    res = get_input_ids(word2id_file_path)
    print(res)
if __name__ == '__main__':
    main()

