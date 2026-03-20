<img width="303" height="143" alt="image" src="https://github.com/user-attachments/assets/ae5c4c9c-2c00-47d6-9853-9df15b11d344" /># BERT 기반 한국어 영화 리뷰 감성 분류 모델 개발

## 0. 프로젝트 소개 및 목차
**(1) 개요**: 네이버 영화 플랫폼에서 수집된 리뷰 데이터를 활용해 각 리뷰의 감성과 그 강약을 분류하는 텍스트 모델을 개발하는 프로젝트. 이 프로젝트의 주요 기술적 과제는 다음과 같습니다.
- 텍스트 전처리 및 정규화
- 토큰화 및 토크나이저 수정
- 분류 모델 설계 및 학습
- 평가 지표에 따른 성능 최적화

**(2) 목표**: 영화 리뷰 데이터를 라벨에 맞춰 감성 분류하기 (기간: 10월 20일~10월 30일, 약 2주)
  
**(3) 사용 데이터셋**: {“ID”, “review”, “label”, “type”}으로 이루어진 영화 리뷰 텍스트 데이터  
- 전체 데이터 개수: 279,650
- “type”: original (사람 데이터), augment(생성 데이터). 데이터 비율=1:1
- “label”: 0 (강한 부정), 1 (약한 부정), 2 (약한 긍정), 3 (강한 긍정). 데이터 비율=41:10:35:14

**(4) 규칙**: 외부 데이터 활용 금지, 아래의 허용된 모델 외 사전훈련 모델 사용 금지
- klue/roberta-base
- klue/bert-base
- kykim/bert-kor-base
- beomi/kcbert-base
- monologg/koelectra-base-v3-discriminator

**(5) 코드 구조**
```plaintext
C:.
|   config.yaml
|   ensemble.ipynb
|   main.ipynb
|
\---utils
        dataset.py
        ensemble.py
        loss.py
        model.py
        optimizer.py
        preprocessing.py
        setting.py
        tokenizer.py
        train.py
        utils.py
```

**(6) 목차:**
1. [BERT 논문 정독](#1-bert-논문-정독)  
2. [베이스라인 코드 실행](#2-베이스라인-코드-실행)  
3. [EDA 및 희귀 클래스 데이터 보강](#3-eda-및-희귀-클래스-데이터-보강)  
4. [데이터 전처리 보강 및 토큰 추가, 교정기 도입](#4-데이터-전처리-보강-및-토큰-추가-교정기-도입)  
5. [모델 성능 실험](#5-모델-성능-실험)  
6. [손실 함수 실험](#6-손실-함수-실험)  
7. [모델 앙상블](#7-모델-앙상블)  
8. [최종 결과](#8-최종-결과)  
9. [프로젝트 소감](#9-프로젝트-소감)

## 1. BERT 논문 정독
**(1) BERT 논문 정독**

일단 BERT에 대해 아는 것이 없었습니다. 그래서 논문([링크](https://arxiv.org/abs/1810.04805))을 읽으며 우선 BERT를 이해하기로 했습니다.

BERT는 언어 표현을 학습하는 모델로, 나중에 사람들이 파인튜닝해 쓸 것을 염두에 두고 설계 및 사전학습된, 범용적인 모델이었습니다. Transformer의 Encoder를 여러 층 쌓은 구조를 가지고 있으며 Masked Language Model (줄여서 MLM)과 Next Sentence Prediction (줄여서 NSP) 두 태스크를 사용해 멀티 태스크 러닝으로 훈련되었습니다. 가장 왼쪽의 모델 입력단에 [CLS]라는 스페셜 토큰을 입력으로 넣고 그로부터 얻은 출력 임베딩 벡터를 가지고 NSP 태스크를 수행함으로써 모델이 텍스트 내 문장 간 관계를 이해하도록 유도하고, 그 외 입력으로부터 얻은 출력으로는 MLM 태스크를 수행해 모델이 문장 내 단어 간 관계와 의미를 학습하도록 했습니다.

그래서 이번 프로젝트에서 제가 해야 할 일은 정리하면, NSP 태스크로 훈련된 출력단에 Classifier head를 얹고 [CLS] 토큰에 대한 출력 임베딩 벡터를 4가지 감성으로 분류하는 것이었습니다. 

**(2) 베이스라인 코드 실행**

HuggingFace의 Transformers 라이브러리에 있는 AutoModelForSequenceClassification을 사용해 BERT 분류 모델을 구현했습니다. 우선 베이스라인 코드를 실행해 모델을 파인튜닝하면서 적절한 Batch size, Learning rate, Max Seq Length를 탐색하기로 했습니다. 다행히 BERT 논문에 파인튜닝 관련 하이퍼파라미터 설정에 대한 내용이 있어 적절한 값을 찾는 데 오래 걸리진 않았습니다. 아래는 찾아낸 하이퍼파라미터를 바탕으로 klue/bert-base를 파인튜닝한 결과입니다.

| Dropout | Batch size | Learning rate | Max Seq Length | Val Acc. | Training time (L4 GPU) |
|:-------:|:----------:|:-------------:|:--------------:|:--------:|:----------------------:|
|X|32|2e-5|64|80.26%|23분(5.8분/epoch)|
|O|32|2e-5|64|80.26%|23분(5.8분/epoch)|

Dropout 적용은 필수인 것 같습니다. 상세한 설정은 config.yaml 파일 참고 바랍니다.

## 2. 베이스라인 코드 실행
ㅇㅇ

## 3. EDA 및 희귀 클래스 데이터 보강
ㅇㅇ

## 4. 데이터 전처리 보강 및 토큰 추가, 교정기 도입
ㅇㅇ

## 5. 모델 성능 실험
ㅇㅇ

## 6. 손실 함수 실험
ㅇㅇ

## 7. 모델 앙상블
ㅇㅇ

## 8. 최종 결과
ㅇㅇ

## 9. 프로젝트 소감
ㅇㅇ
