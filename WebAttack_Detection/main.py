import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

# parsing : 파싱을 진행하는 함수
def parsing(path):  
    with open(path, 'r', encoding='utf-8') as f:  
        train = []
        para = ""
        while True:
            l = f.readline()  

            # 파일을 전부 읽으면 읽기 중단
            if not l:
                break  

            if l != "\n":
              if (l[:3] == 'GET') : 
                para += l
              elif (l[:4] == 'POST') : 
                para += l

            else:
                if para != '':
                    # Method가 POST인 경우 예외적으로 바디까지 가져옴
                    if para[:4] == 'POST':  
                        para += f.readline()

                    train.append(para)
                    para = ""
    return train


# dataset : 데이터셋 생성 - 파싱한 데이터와 라벨 생성
def dataset(path, mod='train'):  
    # mod에 따라 train을 가져올지 test 데이터를 가져올지 결정
    x = parsing(f'{path}norm_{mod}.txt')  
    # 정상 라벨 0 을 정상 데이터 개수 만큼 생성
    y = [0] * len(x)  
    x += parsing(f'{path}anomal_{mod}.txt')
    # 비정상 라벨 1을 비정상 데이터 개수 만큼 생성
    y += [1] * (len(x) - len(y))
    return x, y


# vectorize : 문장을 벡터로 만듦 - 빈도수 측정하는 tf-idf 사용 
def vectorize(train_x, test_x):
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)
    train_vec = tf.transform(train_x)
    test_vec = tf.transform(test_x)
    return train_vec, test_vec


# train : 훈련 모델(분류 모델 사용) 
# randomForest, logisticRegression, decesionTree, Perceptron, supportVectorMachine
def train(train_vec, train_y):
    # # Random Forest
    # rf = RandomForestClassifier()
    # rf.fit(train_vec, train_y)
    # accuracy: 0.9640619848334981
    # f1_score: 0.9568914376112321

    # rf2 = RandomForestClassifier(n_estimators=500)
    # rf2.fit(train_vec, train_y)
    # accuracy: 0.9640619848334981
    # f1_score: 0.9568999604586792


    # # Logistic Regression
    # lr = LogisticRegression()
    # lr.fit(train_vec, train_y)
    # accuracy: 0.9751895812726673
    # f1_score: 0.9692197566213314

    # # Decesion Tree
    # dt = DecisionTreeClassifier()
    # dt.fit(train_vec, train_y)
    # accuracy: 0.9660402242004615
    # f1_score: 0.9591026404605917

    # # Perceptron
    # perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=1)
    # perceptron.fit(train_vec, train_y)
    # accuracy: 0.9923343224530168
    # f1_score: 0.9905131082321739

    # Linear SVM
    linear_svm = LinearSVC(C=1)
    linear_svm.fit(train_vec, train_y)
    # accuracy: 0.9943949884602704
    # f1_score: 0.9930682976554537
    
    return linear_svm


# test : 입력 받은 테스트와 모델로 테스트를 실시합니다
def test(test_y, test_vec, rf):  
    pred = rf.predict(test_vec)
    print("accuracy: ", accuracy_score(test_y, pred))
    print("f1_score: ", f1_score(test_y, pred))
    return pred


def run():
    train_x, train_y = dataset('./', 'train')
    test_x, test_y = dataset('./', 'test')

    train_vec, test_vec = vectorize(train_x, test_x)
    rf = train(train_vec, train_y)
    pred = test(test_y, test_vec, rf)

    tf = TfidfVectorizer()
    tf = tf.fit(train_x)
    
    # print("단어의 수 : ",len(tf.vocabulary_)) # parsing()을 통해 82681 -> 33625
    # print(tf.transform(train_x)[0])


if __name__ == "__main__":
    run()
