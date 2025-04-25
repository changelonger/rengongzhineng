import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
def standardization(X):
    mean = np.mean(X,axis=0,keepdims=True)
    std = np.std(X,axis=0,keepdims=True)
    return (X-mean)/(std)


def f1_score(y_test, y_pred):
    
    true_positives = sum((y_test == 1) & (y_pred == 1))
    false_positives = sum((y_test == 0) & (y_pred == 1))
    false_negatives = sum((y_test == 1) & (y_pred == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return f1


def train():

    file_path = 'classification_data.csv'
    data = pd.read_csv(file_path)

    X = data.drop('target', axis=1).values
    y = data['target'].values

    X = standardization(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return (X_train, X_test), (y_train, y_test), y_pred, model


def main():

    (X_train, X_test), (y_train, y_test), y_pred, model = train()
    print('F1 Score: {:.2f}'.format(f1_score(y_test, y_pred)))
    # 预期输出 F1 Score: 0.94


if __name__ == '__main__':
    main()