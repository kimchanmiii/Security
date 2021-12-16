# 정보보호와 시스템보안
## Project 1: WEBATTACK DETECTION

### 팀원 소개

``` 
소프트웨어학부 20191574 김찬미 
https://github.com/kimchanmiii
```

```
소프트웨어학부 20191686 최혜원
https://github.com/Hyewon0223
```

```
정보보호암호수학과 20172245 박동현
https://github.com/ehdgus12
```
---
## CSIC 2010 데이터셋
### 프로젝트 설명
> 스페인 온라인 쇼핑몰의 http request 로그이다. 로그들 중엔 정상적인 로그들과 비정상적인 로그들이 모두 존재한다. 정상 데이터와 비정상 데이터를 구분하는 것이 이번 프로젝트의 목표이다. 이를 하기 위해서는 2개의 방법을 사용한다. 첫번째는 txt 파일에서 불필요한 단어들을 읽지 않아 속도와 정확도를 높여준다. 두번째 방법은 머신러닝을 이용하여 여러 모델을 학습시킨 후 accuracy와 f1_score가 가장 높은 모델을 찾는다. 
---
### 1. 불필요한 단어 줄이기
> txt 파일에서 정상 GET 로그들과 비정상 GET 로그들을 분석해본 결과 첫 줄만 비교해도 정상/비정상을 구분하는 것이 가능하다고 판단하였다. POST 로그는 첫줄과 body 부분이 정상/비정상 로그를 구분하는 핵심 부분이라고 판단하였다. 구분하는 데에 핵심 부분이라고 생각된 부분만 읽고 저장할 수 있도록 파싱하는 코드를 고쳐 주어 1차적으로 속도와 정확도를 높였다. 그 결과 82,681개였던 단어가 33,625개로 줄어들어 정확도는 약 44% -> 약 96%, f1_score는 약 59% -> 약 95%까지 향상 시켰다.
- 단어줄이기 코드
```python
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
                #처음 시작단어 3글자가 GET이면 
                if (l[:3] == 'GET') :
                    para += l
                #처음 시작단어 4글자가 GET이면
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
```
---
### 2. 학습 모델 변경하기
> 주어진 코드 파일에서는 random forest라는 모델을 훈련시켜 정상/비정상 로그를 구분하였다. 이 모델로는 정확도가 약 96%, f1_score는 95%가 나왔다. 정확도와 f1_score를 더 높이기 위해 여러 모델을 학습시켜 비교하여 가장 좋은 모델을 찾아내었다. 

#### 베이스 모델. random forest
> random forest는 같은 데이터에 대해 decision tree를 여러 개 만들어 그 결과를 종합해 예측 성능을 높이는 기법이다. 
- random forest 코드
```python
def train(train_vec, train_y):
    rf = RandomForestClassifier()
    rf.fit(train_vec, train_y)

    return rf

    # 결과값
    # accuracy: 0.9640619848334981
    # f1_score: 0.9568914376112321
```

#### 모델 1. random forest (n_estimators 인자 추가)
> 제공해준 random forest 모델에서 n_estimators라는 인자를 추가한 모델이다. n_estimators는 모델에서 사용할 트리의 개수, 즉 학습시 생성할 트리 개수를 의미한다. 이를 추가하면 반복횟수를 늘려 데이터들을 더 많이 학습시킨다. 이 모델을 학습시킨 결과 정확도와 f1_score 약간의 향상만 있었다. 
- random forest (n_estimators) 코드
```python
def train(train_vec, train_y):
    rf2 = RandomForestClassifier(n_estimators=500)
    rf2.fit(train_vec, train_y)

    return rf2

    # 결과값
    # accuracy: 0.9640619848334981
    # f1_score: 0.9568999604586792
```

#### 모델 2. Logistic Regression
> Logistic Regression은 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0에서 1사이의 값으로 예측하고 그 확률에 따라 가능성이 더 높은 범주에 속하는 것으로 분류해주는 지도 학습 알고리즘이다. 이는 이진 분류를 수행하는 데에 사용되어 데이터 샘플을 1 또는 0 클래스 둘 중 어디에 속하는지 예측해준다. 이 모델을 실행시킨 결과 정확도는 약 97%, f1_score는 약 96%까지 향상되었다. 
- Logistic Regression 코드
```python
def train(train_vec, train_y):
    lr = LogisticRegression()
    lr.fit(train_vec, train_y)

    return lr

    # 결과값
    # accuracy: 0.9751895812726673
    # f1_score: 0.9692197566213314
```

#### 모델 3. Decision Tree
> decision tree는 분류와 회귀 모두 가능한 지도 학습 모델 중 하나로 특정 기준에 따라 데이터를 구분하는 모델이다. 이 모델을 학습시킨 결과 정확도는 약 96%, f1_score는 약 95%까지 향상되었다.
- Decesion Tree 코드
```python
def train(train_vec, train_y):
    dt = DecisionTreeClassifier()
    dt.fit(train_vec, train_y)

    return dt

    # 결과값
    # accuracy: 0.9660402242004615
    # f1_score: 0.9591026404605917
```

#### 모델 4. Perceptron
> perceptron 모델은 초기형태의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘이다. 학습할 때 주어진 데이터를 학습하고 에러가 발생한 데이터에 기반하여 Weight(가중치) 값을 기존에서 새로운 Weight 값으로 업데이트 시켜주면서 학습을 하게 된다. 이 모델을 학습시킨 결과 정확도는 약 99%, f1_score는 약 99%까지 향상되었다.
- Perceptron 코드
```python
def train(train_vec, train_y):
    perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=1)
    perceptron.fit(train_vec, train_y)

    return perceptron

    # 결과값
    # accuracy: 0.9923343224530168
    # f1_score: 0.9905131082321739
```

#### 모델 5. Linear SVM
> SVM(Support Vector Machine)은 주어진 데이터가 어느 카테고리에 속할지 이진 선형 분류 모델이다. 두 데이터 집합을 나누는 결정 경계의 마진이 최대화 되도록 학습한다. 이때, 마진이란, 결정 경계와 데이터의 거리를 의미한다. SVM은 답러닝 이전에 가장 일반적이고 뛰어난 성능을 보였던 지도학습 모델이다. 이 모델을 학습시킨 결과 정확도는 약 99%, f1_score는 약 99%까지 향상되었다.
- Linear SVM 코드
```python
def train(train_vec, train_y):
    linear_svm = LinearSVC(C=1)
    linear_svm.fit(train_vec, train_y)

    return linear_svm

    # 결과값
    # accuracy: 0.9943949884602704
    # f1_score: 0.9930682976554537
```

__위의 다섯가지 모델을 모두 비교 분석해본 결과 Linear SVM 모델이 정확도 99.44%, f1_score 99.31%로 가장 높았다. 따라서 최종적으로 Linear SVM 모델을 선택하였다.__
