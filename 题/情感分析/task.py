import jieba
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_dataset(file_path):
    x, y = [], [] 
    with open(file_path, "r", encoding='utf-8') as f:
        for i in f.readlines():
            i_list = i.strip().split(",")
            x.append(i_list[2]) 
            y.append(int(i_list[1]))
    return x, y

def preprocess_text(text):
    s = ' '.join(jieba.cut(text,cut_all=False,))
    return s

def fit_classifier(x, y, classifier_path="classifier_model.pkl"):
    classifier = LogisticRegression()
    classifier.fit(x,y)
    with open(classifier_path, 'wb') as file:
        pickle.dump(classifier,file)

def test_classifier(test_x, test_y, classifier_path="classifier_model.pkl"):
    with open(classifier_path, 'rb') as file:
        classifier = pickle.load(file)
    predict = classifier.predict(test_x)
    return accuracy_score(test_y,predict)

def main():
    train_x, train_y = load_dataset("sentiment_analysis_train.txt")
    test_x, test_y = load_dataset("sentiment_analysis_test.txt")

    train_x = [preprocess_text(text) for text in train_x]
    test_x = [preprocess_text(text) for text in test_x]

    vectorizer = TfidfVectorizer(max_features=1000)
    train_x_features = vectorizer.fit_transform(train_x)
    test_x_features = vectorizer.transform(test_x)

    fit_classifier(train_x_features, train_y)
    acc = test_classifier(test_x_features, test_y)
    print("Accuracy:", acc)

if __name__ == '__main__':
    main()