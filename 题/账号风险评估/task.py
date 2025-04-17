import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import pickle

def load_data(file_path):
    data = pd.read_csv(file_path)
    y = data['target']
    X = data.drop('target', axis=1)
    return X, y

def preprocess_data(X_train, X_test):
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    return X_train, X_test

def logistic_regression_model(X_train, y_train, X_test, y_test, dist):
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    with open(dist, 'wb') as f:
        pickle.dump(clf, f) #
    return f1_score(y_test,clf.predict(X_test))


def main():
    X, y = load_data('data_user.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test = preprocess_data(X_train, X_test)
    f1 = logistic_regression_model(X_train, y_train, X_test, y_test,'./lr_model.pkl')
    print('%.2f'% f1)

if __name__ == '__main__':
    main()