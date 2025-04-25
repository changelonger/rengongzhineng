from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split()# 用空格分词
    )
with open('news_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    train_data = [line.strip() for line in lines]
with open('label_newstrain.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    train_label = [line.strip() for line in lines]
train_feature = vectorizer.fit_transform(train_data)
tarin_x,test_x,train_y,test_y = train_test_split(train_feature,train_label,test_size=0.2)
clf =RandomForestClassifier(n_estimators=50)
clf.fit(tarin_x,train_y)
pre = clf.predict(test_x)
print(accuracy_score(test_y,pre))
with open('news_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    test_data =  [line.strip() for line in lines]
    test_feature = vectorizer.transform(test_data)

pre = clf.predict(test_feature)

with open('pred_test.txt','w') as f:
    for i in pre:
        f.write(f'{i}\n')
