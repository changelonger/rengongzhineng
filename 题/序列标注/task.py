import pandas as pd
data = pd.read_json('ner.json')
res = []
for i in data.index:
    temp = {}
    text = data.iloc[i]['text']
    text_id = data.iloc[i]['text_id']
    temp['text'] = text
    temp['text_id'] = text_id
    temp['label'] = ['O' for _ in range(len(text))]
    for ann in  data.iloc[i]['ann']:
        start,end = ann['start'],ann['end']
        label = ann['label']
        temp['label'][start] = 'B-'+label
        temp['label'][start+1:end] = ['I-'+label]*(end-start-1)
    res.append(temp)
pd_res = pd.DataFrame(res)
pd_res = pd.DataFrame(res)
pd_res.to_json('ner_processed.json', orient='records', force_ascii=False,indent=2)  # 添加indent提高可读性
