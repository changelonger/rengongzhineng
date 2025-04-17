def forward_segment(text, dictionary, max_len=5):
    #TODO
    segments = []  # 存储分词结果的列表
    text_length = len(text)  # 待分词文本的长度
    i = 0  # 当前位置指针
    while i < text_length:
        matched = False  # 匹配标志
        # 从最大词长开始递减尝试匹配
        for j in range(i + max_len, i, -1):
            if j > text_length:
                continue  # 避免索引越界
            word = text[i:j]  # 截取长度为 j-i 的词
            if word in dictionary:  # 如果截取的词在词典中存在
                segments.append(word)  # 将词添加到分词结果列表中
                matched = True  # 设置匹配标志为 True
                i = j  # 更新下一个位置
                break
        # 如果没有匹配到词，则将当前字符视为单个字符的词
        if not matched:
            segments.append(text[i])
            i += 1
    return segments