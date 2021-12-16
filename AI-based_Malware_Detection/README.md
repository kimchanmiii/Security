# 정보보호와 시스템보안
## Project 2 : AI-based Malware Detection

## 팀원 소개

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
## * 참고 사항 *
```
본 readme파일에는 코드에 대한 설명이 포함되어 있지 않습니다. 
코드에 대한 설명은 AI_malware_detection.ipynb 파일에 주석으로 써놓았으니 참고 부탁드립니다.
```
---

## 프로젝트 목표 및 진행 단계
```
PE파일이 주어졌을 때, 주어진 파일이 악성파일인지 정상파일인지 판별하는 이진 분류기 모델을 만드는 것이 목표이다.
```
1. 특징 추출 및 전처리 : 제공한 PESTUDDIO, EMBER, PEMINER의 JSON 파일이 각 파일의 특징에 해당됨(특징 추출)
- 추출한 특징을 가공해서 고정크기의 특징벡터를 생성(전처리) 
    - 특징 벡터의 모든 원소는 정수 또는 실수 형태여야 함
- 파일(JSON) -> 고정된 특징 벡터 -> 머신러닝 모델 (악성 : 1, 정상 : 0)
2. 학습 : 생성한 특징 벡터를 사용해 해당 파일이 악성/정상임을 판별하는 이진 분류 머신러닝 모델 만들기
- 지도학습 방법으로 머신러닝 모델을 학습할 것
    - (데이터(특징 벡터), 정답)을 모델에게 제공해 학습을 진행하는 방법
    - 분류(classification), 회귀(regression) 문제에 많이 사용
- 학습데이터를 구성하면 train 함수를 사용해 머신러닝 모델 학습 가능
3. 성능평가
- 학습한 모델이 잘 학습 되었는지 evaluate 함수를 사용해 확인
- 검증데이터를 예측한 정확도를 계속 높여주는 작업을 반복
    - 모델의 하이퍼 파라미터 조정 (Grid Search, Random Search)
    - 다른 특징 벡터 추출
- 최대한 높인 정확도(accuracy)를 근거로 테스트 데이터를 예측
---
## 특징 추출 방식
### PEMINER 
- 전체 데이터 사용 

### EMBER 
- 기존에 있던 특징들을 수정 
- 새로운 특징을 찾아 vector에 하나씩 추가해보며 정확도가 비교적 높은 것과 낮은 것을 구분하여 최적의 조합을 찾음 
- ** 해당 특징을 추출한 이유에 대해서는 코드에 주석으로 달아놓음

### PESTUDIO 
- 데이터들을 확인하며 특징들을 추출 
- 여러가지의 특징을 찾아 vector에 하나씩 추가해보며 정확도가 비교적 높은 것과 낮은 것을 구분하여 최적의 조합을 찾음 
- ** 해당 특징을 추출한 이유에 대해서는 코드에 주석으로 달아놓음

### 앙상블 
- 여러가지의 모델을 모두 사용하여 비교해보고 최고의 정확도가 나온 앙상블을 선택하여 사용함 

### 테스트 데이터 
- 학습된 모델과 테스트 데이터를 앙상블 한 결과를 csv 파일로 생성함
---
## Feature Selection
> 빈배열로 학습시킨 결과와 특징을 찾아 vector에 추가해보면서 정확도가 높아진 것과 떨어진 것을 구분해 정확도가 증가한 특징들을 찾아내었다. 데이터셋을 다운로드 한 후, 모델 학습을 시키기 전에 특징 추출과정은 필수적으로 해야한다.
> - 결과 엑셀 정리 : https://docs.google.com/spreadsheets/d/1wCgR1ketaOvxqovktrqcTlO3uvL_FQ8i7W0P02fu-ZE/edit#gid=2014606618

### EMBER Features 
```
EMBER 코드는 scikit-learn 도구를 사용하여 각 특징을 벡터로 해시하는 방법을 의미한다.
```
#### 대표 Feature 설명
- byte histogram (histogram) : 각 바이트의 발생 횟수에 대한 단순 게산
- byte entropy histogram (byteentropy) : sliding window entropy 계산
- section information (section) : 섹션의 이름, 크기, 엔트로피 및 각 섹션에 대해 주어진 기타 정보를 가진 모든 섹션의 목록
- import information (imports) : import한 기능 이름과 가져온 각 라이브러리
- export information (exports) : export한 기능 이름
- string information (strings) : 문자열 수, 평균 길이, 문자 histogram, URLs, MZ 헤더 등과 같은 다양한 패턴과 일치하는 문자열의 수
- general information (general) : import, export, symbol의 수와 파일의 relocations, resources, signature이 있는지 여부에 대한 수
- header information (header) : 파일이 컴파일된 시스템에 대한 세부 정보. 링커, 이미지, 운영 체제 버전 등등

> 정확도가 높아진 특징들
> - get_general_file_info
> - get_histogram_max
> - get_byteennum
> - get_byteentropy (가장 높음)
> - get_string_max
> - get_section_number
> - get_entropy_max
> - get_bigentropy_len
> - get_datadirnum

<img width="1146" alt="스크린샷 2021-12-16 오후 3 55 07" src="https://user-images.githubusercontent.com/54922827/146322844-c84b3301-efd6-492b-8223-0c2148bebb6a.png">

### PEMINER Features 
```
PESTUDIO 코드는 import된 함수들 중 특히 악성코드들이 주로 사용하는 것들을 블랙리스트로 구분한다.
```
#### 대표 Feature 설명 
- indicators: 해당 프로그램이 하는 행위에 대해서 위험 심각도에 따라 분류되어 표시된다.
- sections: PE 구조에서 각각의 섹션을 보여준다.
- imports: imports 되는 함수. 악성코드에서 자주 사용하는 API에 대한 blacklist를 제공하고, blacklist에 따라서 API가 분류된다.
- resources: 리소스 파일이 존재할 경우 출력한다.
- strings: 파일 내의 문자열 정보를 의미한다.

> 정확도가 높아진 특징들
> - overview_entropy
> - import_blacklist
> - size
> - entropy
> - get_string_leng
> - string_size_max
> - get_tls_callbacks (가장 높음)
> - get_overlay
> - get_string_network
> - get_indicator_ave

![스크린샷 2021-12-16 오후 5 17 07](https://user-images.githubusercontent.com/54922827/146334429-c4eec299-3e97-4615-8d93-a02987daaa98.png)

## Feature 조합
> 정확도가 높아진 것들은 추가하고 떨어진 것들은 제거하는 방식으로 여러가지 feature들을 조합해본 결과 EMBER와 PESTUDIO 모델을 빈 배열로 돌린 기준보다 정확도가 증가한 특징들로 조합했을 때 가장 높은 정확도가 보여 그 조합으로 선택했다.

최종 선택한 조합
- EMBER 
    - get_general_file_info
    - get_histogram_max
    - get_byteennum
    - get_byteentropy
    - get_string_max
    - get_section_number
    - get_entropy_max
    - get_bigentropy_len
    - get_datadirnum
- PESTUDIO
    - overview_entropy
    - import_blacklist
    - size
    - entropy
    - get_string_leng
    - string_size_max
    - get_tls_callbacks (가장 높음)
    - get_overlay
    - get_indicators_ave
    - get_string_network

![스크린샷 2021-12-16 오후 5 59 58](https://user-images.githubusercontent.com/54922827/146340412-c3d259fb-1e3a-4db0-8fb2-8b2c1f534e94.png)

## Model Selection 
> 8가지 모델을 사용하여 비교해보고 최고의 정확도가 나온 앙상블을 선택하여 사용하였다. 

> 8가지 모델 정확도 평균 비교 
> - Random Forest : 0.9505260234 (1위)
> - Decision Tree : 0.9262057018
> - Light GBM : 0.9501780702 (2위)
> - SVM : 0.8269947368
> - Logistic Regression : 0.8224523392
> - KNN : 0.9042396199
> - ADABOOST : 0.8999877193
> - MLP : 0.757224269

![스크린샷 2021-12-16 오후 6 00 46](https://user-images.githubusercontent.com/54922827/146340501-a9eeb1bb-f8d7-4b3a-ba75-cdfe11b0fb38.png)

> 결과적으로 모든 특징들에 있어서 Random Forest와 Light GBM이 꾸준한 높은 정확도를 보여 RF와 LGB를 선택하여 사용하였다.

### 선택한 모델 설명
1. Random Forest
> 랜덤 포레스트는 결정 트리를 기반으로 하는 알고리즘이다. 랜덤 포레스트는 여러 개의 결정 트리 분류기가 배깅을 기반으로 각자의 데이터를 샘플링 하여 학습을 수행한 후에 최종적으로 보팅을 통해 예측 결정을 하게 된다. 랜덤 포레스트는 부트스트래핑(bootstrapping) 방식으로 분할 하여 중첩되게 샘플링이 된다.

2. Light GBM
> Light GBM은 gradient boosting 프레임워크로 Tree 기반 학습 알고리즘이다. Light GBM은 Tree가 수직적으로 확장되는 반면에 다른 알고리즘은 Tree가 수평적으로 확장된다. 즉 Light GBM은 leaf-wise 인 반면 다른 알고리즘은 level-wise 이다. 동일한 leaf를 확장할 때, leaf-wise 알고리즘은 level-wise 알고리즘보다 더 많은 loss, 손실을 줄일 수 있다.
> Light GBM 모델은 대용량 데이터 처리가 가능하고, 다른 모델보다 더 적은 메모리를 사용해 빠르다는 장점을 가지고 있다. 또한 GPU까지 지원해주기 때문에 다른 앙상블 모델보다 인기를 끌고 있다. 

## Result 
1. 분류 모델 정확도 비교
- 8가지의 분류 모델을 비교해본 결과 RandomForest와 LGBM 모델 조합이 정확도가 가장 높았다. 
2. feature 조합 비교 
- feature 조합으로는 정확도가 기준보다 증가한 feature들의 조합이 정확도가 가장 높이 나왔다. 
``` 
최종 정확도 : 0.957
``` 

