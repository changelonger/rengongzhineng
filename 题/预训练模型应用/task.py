import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.metrics import accuracy_score  # 计算准确度
from sklearn.model_selection import train_test_split

def get_feature(arr):
    res = []
    for i in arr:
        res.append(i)
    return np.array(res)
resnet = pd.read_json('resnet_train.json')
inception = pd.read_json('inception_train.json')
xception  = pd.read_json('xception_train.json')
test = pd.read_json('test.json')
x1,y1 = resnet.loc['feature'].values,resnet.loc['label'].values
x2,y2 = inception.loc['feature'].values,inception.loc['label'].values
x3,y3 = xception.loc['feature'].values,xception.loc['label'].values
test = get_feature(test.loc['feature'].values)

# 现在得到了特征和标签，但是每一个特征是List形式，转换成numpy
feature = np.concatenate([get_feature(x1),get_feature(x2),get_feature(x3)],axis=1)
label  = y1.astype(int)
# print(feature.shape,label.shape) # (400,6144),(400,1)
# print(test.shape)  # 1600,6144
x_train,x_test,y_train,y_text = train_test_split(feature,label,test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
score = accuracy_score(y_text,clf.predict(x_test))
# print(score)   # 0.9878
pre = clf.predict(test)
id = pd.read_json('test.json').columns.tolist()
# print(id)
# print(pre)
res = pd.DataFrame({'id':id,'label':pre})
# print(id)
res.to_csv('result.csv',index= False)