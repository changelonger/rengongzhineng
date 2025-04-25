import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split 
train_data ,test_data= pd.read_csv('songs_train.csv'),pd.read_csv('songs_test.csv')

train_label = train_data['popularity'] 
train_feature = train_data.drop('popularity',axis=1) 
test_label = train_data['popularity'] 
test_feature = test_data.drop('popularity',axis=1) 
train_x,test_x,train_y,test_y = train_test_split(train_feature,train_label,test_size=0.2,random_state=42) 
Fun = LinearRegression() 
Fun.fit(train_x,train_y)
pre_y = Fun.predict(test_x)
print(r2_score(test_y,pre_y))
pre = Fun.predict(test_feature) 
test_data['popularity'] = pre 
test_data.to_csv('songs_testout.csv')