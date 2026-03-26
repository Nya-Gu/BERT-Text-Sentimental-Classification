# BERT 기반 한국어 영화 리뷰 감성 분류 모델 개발

## 0. 프로젝트 소개 및 목차
**(1) 개요**: 네이버 영화 플랫폼에서 수집된 리뷰 데이터를 활용해 각 리뷰의 감성과 그 강약을 분류하는 텍스트 모델을 개발하는 프로젝트입니다. 이 프로젝트의 주요 기술적 과제는 다음과 같습니다.
- 텍스트 전처리 및 정규화
- 토큰화 및 토크나이저 수정
- 분류 모델 설계 및 학습
- 평가 지표에 따른 성능 최적화

**(2) 목표**: 영화 리뷰 데이터를 감성에 따라 4가지 클래스로 분류하기
  
**(3) 사용 데이터셋**: {“ID”, “review”, “label”, “type”}으로 이루어진 영화 리뷰 텍스트 데이터  
- 전체 데이터 개수: 279,650
- “type”: original (실제 리뷰 데이터), augment(LLM 생성 데이터). 데이터 비율 = 100 : 100
- “label”: 0 (강한 부정), 1 (약한 부정), 2 (약한 긍정), 3 (강한 긍정). 데이터 비율 = 41: 10 : 35 : 14

**(4) 규칙**: 외부 데이터 활용 금지, 아래 모델 외 사용 금지
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
4. [데이터 전처리 보강](#4-데이터-전처리-보강)
5. [토큰 추가, 교정기 시험](#5-토큰-추가-교정기-시험)
6. [모델 성능 실험](#6-모델-성능-실험)  
7. [손실 함수 실험](#7-손실-함수-실험)  
8. [모델 앙상블](#8-모델-앙상블)  
9. [최종 결과](#9-최종-결과)  
10. [프로젝트 소감](#10-프로젝트-소감)

## 1. BERT 논문 정독

일단 BERT에 대해 아는 것이 없었습니다. 그래서 논문([링크](https://arxiv.org/abs/1810.04805))을 읽으며 우선 BERT를 이해하기로 했습니다.

BERT는 언어 표현을 학습하는 모델로, 나중에 파인튜닝해 쓸 것을 염두에 두고 설계 및 사전학습된, 범용적인 모델이었습니다. Transformer의 Encoder를 여러 층 쌓은 구조를 가지고 있으며 Masked Language Model (줄여서 MLM)과 Next Sentence Prediction (줄여서 NSP) 두 태스크를 사용해 멀티 태스크 러닝으로 훈련되었습니다. 가장 왼쪽의 모델 입력단에 [CLS]라는 스페셜 토큰을 입력으로 넣고 그로부터 얻은 출력 임베딩 벡터를 가지고 NSP 태스크를 수행함으로써 모델이 텍스트 내 문장 간 관계를 이해하도록 유도하고, 그 외 입력으로부터 얻은 출력으로는 MLM 태스크를 수행해 모델이 문장 내 단어 간 관계와 의미를 학습하도록 유도했습니다.

그래서 이번 프로젝트에서 제가 해야 할 일은 정리하면, NSP 태스크로 훈련된 출력단에 Classifier head를 얹고 [CLS] 토큰에 대한 출력 임베딩 벡터를 4가지 감성으로 분류하는 것이었습니다. 

## 2. 베이스라인 코드 실행

HuggingFace의 Transformers 라이브러리에 있는 AutoModelForSequenceClassification을 사용해 BERT 분류 모델을 구현했습니다. 우선 베이스라인 코드를 실행해 모델을 파인튜닝하면서 적절한 Batch size, Learning rate, Max Seq Length를 탐색하기로 했습니다. 다행히 BERT 논문에 파인튜닝과 관련해 하이퍼파라미터 설정에 대한 내용이 있어 적절한 값을 찾는 데 오래 걸리지 않았습니다. 아래는 찾아낸 하이퍼파라미터를 바탕으로 klue/bert-base를 파인튜닝한 결과입니다.

| Dropout | Batch size | Learning rate | Max Seq Length | Val Acc. | Training time (L4 GPU) |
|:-------:|:----------:|:-------------:|:--------------:|:--------:|:----------------------:|
|X|32|2e-5|64|80.26%|23분(5.8분/epoch)|
|O|32|2e-5|64|**80.71%**|23분(5.8분/epoch)|

Dropout 적용은 필수인 것 같습니다. 상세한 설정은 config.yaml 파일 참고 바랍니다.

## 3. EDA 및 희귀 클래스 데이터 보강

EDA를 통해 데이터의 클래스 불균형을 확인했습니다.

<img width="780" height="335" alt="image" src="https://github.com/user-attachments/assets/e7797941-cea8-439b-9c82-21bbf3448526" />

0, 2번 클래스에 비해 1, 3번 클래스에 대한 모델의 분류 성능이 낮았고 이를 해결하기 위해 1, 3번 클래스에 한해서 LLM 생성 데이터로 데이터 증강을 했습니다. 그 결과 모델 성능이 80.71%에서 81.00%로 0.29% 개선되었습니다.

| Class | Original(증강 전) | LLM | Original+LLM(증강 후) |
|:----------:|:----:|:-----:|:-----:|
|0 (강한 부정)|40.79%|-|32.24%|
|1 (약한 부정)|9.73%|9.73%|16.41%|
|2 (약한 긍정)|35.55%|-|27.82%|
|3 (강한 긍정)|13.93%|13.93%|23.53%|

| Data | Val Acc. | Training Time (L4 GPU) |
|:----:|:--------:|:----------------------:|
|Original|80.71%|23분(5.8분/epoch)|
|Original+LLM (only rare class)|**81.00%**|29분(7.3분/epoch)|

Original 데이터와 LLM 생성 데이터의 분포가 상당히 달랐음에도 데이터 증강이 효과가 있었습니다. LLM 데이터는 텍스트 길이가 긴 편이고 Max Seq Length=64로 설정했기 때문에 뒷내용이 많이 잘렸을 것입니다. 하지만 앞내용만으로도 충분히 클래스 예측이 가능했던 것으로 보입니다.

<img width="780" height="580" alt="image" src="https://github.com/user-attachments/assets/e782685c-5da0-41a3-931c-4da6cd84e689" />

참고로 모든 클래스를 LLM 데이터로 증강하는 경우 모델 성능은 80.64%로 하락했습니다.

## 4. 데이터 전처리 보강

(1) 중복값 처리

EDA를 통해 이상 데이터에서 중복 리뷰가 큰 비중을 차지하고 있음을 확인했습니다.

<img width="377" height="916" alt="image" src="https://github.com/user-attachments/assets/86f6eb19-c9ab-4fc5-93bf-cadaf5407240" />

중복 리뷰를 살피니 내용은 동일하지만 평점이 서로 다른 경우가 다수 존재했습니다. 이에 대한 처리 방안을 고민하며 여러 차례 실험을 진행했고, 최종적으로는 중복 리뷰를 하나만 남기고 나머지는 삭제하되, 남은 리뷰의 평점은 중복된 리뷰들 중 가장 자주 등장한 값으로 설정하기로 했습니다.

(2) 데이터 전처리 로직 보강

예측에 실패한 데이터를 분석한 결과, 분류하기 쉬워 보이는 문장에서도 모델이 자주 오분류하는 것이 확인되었습니다. 이는 전처리 과정에서 문장이 충분히 정제되지 않아 모델이 해당 데이터를 제대로 이해하고 분류하지 못한 것이라고 판단했습니다. 이에 따라 데이터 전처리 과정을 보강하기로 했고, 그 결과 모델 성능이 81.00%에서 81.09%로 0.09% 개선되었습니다. (utils/preprocessing.py 내 TextPreprocessingPipeline 클래스 참고)

(3) 교정 함수 제작

이후 모델이 제대로 분류하지 못하는 데이터를 거듭해서 살핀 결과, 문장을 인코딩 했을 때 [UNK] 토큰을 포함하고 있으면 모델이 해당 문장을 분류하는 실패하는 걸 다수 관찰할 수 있었습니다. 이에 간단한 교정 함수를 도입하였고, 학습 데이터 내 [UNK] 토큰이 등장하는 데이터의 비율을 5.96% (8000개)에서 0.48% (675개)로 낮춰 모델 성능을 81.09%에서 81.17%로 0.08% 개선할 수 있었습니다. (utils/preprocessing.py 내 Proofreading 함수 참고)

| 전처리 | Val Acc. |
|:------:|:--------:|
|Baseline|81.00%|
|Baseline 전처리 보강|81.09%|
|Baseline 전처리 보강 + Proofreading 함수 추가|**81.17%**|

## 5. 토큰 추가, 교정기 시험

(1) 토큰 추가

학습 데이터에서 자주 등장하지만 Tokenizer의 Vocab에는 없는 단어의 경우 Vocab에 새 토큰으로 등록을 했습니다. 그 결과 모델 성능이 81.17%에서 81.20%로 0.03% 향상되었습니다.

- 새로 추가한 토큰: 'oo', 'ㅡㅅㅡ', 'ㅗ', '^^', '용두사미', '망작', '꿀잼', '노잼', '남주'

(2) Hanspell 교정 시험 

여전히 [UNK] 토큰이 등장하는 데이터를 분석한 결과, 문장 전체를 [UNK] 토큰으로 치환하는 경우가 다수 확인되었습니다. 특히 띄어쓰기가 전혀 없는 데이터가 그러했고, 이러한 문법적 오류를 교정하기 위해 Hanspell 교정기를 도입하기로 했습니다.

Hanspell을 사용하고 [UNK] 토큰으로 처리되는 문장이 줄어들었습니다. 하지만 모델 성능은 오히려 하락했습니다. 이는 현재 쓰고 있는 klue/bert-base 모델 뿐만 아니라 kykim/bert-base를 비롯한 다른 모델에서도 일관되게 관찰되는 현상이었습니다. 따라서 Hanspell 교정기는 사용하지 않기로 했습니다.

| 모델 | 추가 전처리 | Val Acc. |
|:------:|:--------:|:---------:|
|klue/bert-base|-|81.17%|
|klue/bert-base|+새 토큰 추가|**81.20%**|
|klue/bert-base|+새 토큰 추가, Hanspell 교정|80.72%|

## 6. 모델 성능 실험

아래를 기본 설정으로 하여 이번 프로젝트에서 사용이 허가된 다섯 모델을 풀-파인튜닝했습니다. 
- LR = 2e-5
- Batch = 64
- Max_seq_length = 64
- Data: Original + LLM (rare class)
- Preprocessing:
  - TextPreprocessingPipeline 클래스 보강
  - Proofreading 함수 추가
  - 새 Token 추가

| 모델 | 사전훈련에 쓰인 데이터 | Vocab 사이즈 | Val Acc. |
|:------:|:--------------------:|:------------:|:--------:|
|klue/bert-base|종합 데이터|32000|81.20%|
|kykim/bert-base|블로그, 댓글, 리뷰|42000|82.35%|
|klue/roberta-base|종합 데이터|32000|81.78%|
|beomi/kcbert-base|뉴스 댓글, 대댓글|30000|80.38%|
|monologg/koelectra-base-v3-discriminator|뉴스, 위키, 나무위키|35000|81.42%|

1. 논문([링크](https://arxiv.org/abs/1905.05583))을 참고해 일부 레이어만 파인튜닝하거나 LoRA를 사용해 파인튜닝해봤지만 풀-파인튜닝보다 성능이 좋게 나오지 않았습니다.

2. 위의 논문에 따르면 Further Pretraining(파인튜닝에 사용할 데이터와 유사한 도메인의 데이터로 사전훈련으로 수행하는 것)을 수행한 뒤 파인튜닝을 하면 더 좋은 성능을 볼 수 있다고 합니다. kykim/bert-kor-base 모델이 다른 모델보다 좋은 성능을 낸 것도 이와 같은 이유가 아닐까 싶습니다. kykim/bert-kor-base 모델은 현 프로젝트에서 사용하고 있는 리뷰와 유사한 데이터(블로그, 댓글, 리뷰)로 사전훈련되었기 때문에 다른 모델에 비해 리뷰 데이터을 이해하는 데 '익숙'하고, 그로 인해 성능이 더 좋게 나온 것으로 보입니다.

## 7. 손실 함수 실험

리뷰의 긍정과 부정을 분류하는 건 쉽지만 그것의 강도, 다시 말해 강약까지 모델이 분류하는 건 쉽지 않을 것입니다. 실제로 제가 직접 분류해봤지만 강약까지 맞추기는 어려웠습니다. 따라서 모델이 자신의 예측을 너무 확신하지 않도록 smoothing loss를 사용하고 이를 통해 모델의 일반화 성능을 향상시키기로 했습니다. smoothing loss와 함께 여러 손실 함수를 테스트한 결과, smoothing factor=0.1을 사용할 때 모델 성능이 가장 좋게 나왔습니다.

| 모델 | 손실 함수 | Val Acc. |
|:----:|:---------:|:--------:|
|klue/bert-base| Smooth CE 0.05 | 81.07% |
|klue/bert-base| Smooth CE 0.10 | **81.22%** |
|klue/bert-base| Smooth CE 0.10 + Class weight | 80.61% |
|klue/bert-base| Focal | 80.97% |

이후 검증, 훈련 데이터를 9 대 1로 분할하고 Smoothing loss를 사용해 다섯 모델을 재학습시켰습니다.

| 모델 | Val Acc. |
|:----:|:--------:|
|klue/bert-base|81.70%|
|kykim/bert-base|82.84%|
|klue/roberta-base|82.50%|
|beomi/kcbert-base|80.91%|
|monologg/koelectra-base-v3-discriminator|81.46%|

## 8. 모델 앙상블

다음으로 앙상블을 진행했습니다. Soft Voting, Hard Voting, Stacking 세 가지를 수행했습니다.

| 방법 | Val Acc. | 세부 사항 |
|:----:|:---------:|:---------|
|Soft Voting|**84.122%**|모델별 가중치<br>Klue/bert-base: 0.1<br>kykim/bert-kor-base: 0.6<br>klue/roberta-base: 0.6<br>beomi/kcbert-base: 0.3<br>monologg/koelectra-base-v3-discriminator: 0.3|
|Hard Voting|83.61%|-|
|Stacking (meta learning)|83.98%|MLP_Classifier(in_size = 20, out_size = 4, inner_width  = 32, depth = 1, dropout = 0.1)|

앙상블을 하는 건 이번이 처음이었습니다. 시작하기 전에는 Stacking이 Soft Voting의 상위 호환인 기법이라고 생각하고 가장 성능이 좋게 나올 거라고 생각했습니다. 하지만 결과는 그렇지 않았고, 그 이유는 나름 고민해봤습니다. Soft Voting은 최고 성능을 낳는 가중치 조합을 직접적으로 찾는 방법인 반면, Stacking은 목적 함수(CE)를 최소화하는 가중치를 찾는 간접적인 방법이기에 애초에 Stacking이 Soft Voting의 상위 호환인 기법이 아니었던 것 같습니다. 따라서 성능이 항상 더 좋게 나올 이유가 없었던 것 같습니다.

MLP Classifier 외에 LightGBM 같은 Tree 기법이나 랜덤 서치를 사용해봤으나 모두 성능이 Soft Voting에 미치지 못했습니다. Tree 기반 Stacking은 아직 익숙하지 않은 탓에 성능이 다소 낮게 나온 측면도 있을 거라고 생각합니다.

추가로 위의 정석적인 앙상블 기법 외에 제 맘대로 모델을 앙상블 해봤습니다. 긍정/부정을 분류하는 모델과 감정의 강/약을 분류하는 모델 등을 학습시키고 다양한 방식으로 조합해봤지만 성능은 좋지 않았습니다. Soft Voting, Hard Voting 같은 기법이 괜히 정석인 게 아닌 모양입니다.

앙상블에서 어느 모델이 가장 성능에 크게 기여하고 있는지 한번 확인해봤습니다. 각 모델이 앙상블에 쓰이지 않았을 때 앙상블 성능이 얼마나 하락하는지를 기록했습니다.

| 모델 | Val Acc. | 세부 사항 |
|:----:|:--------:|:---------:|
|All|84.129%|best weight: (0.1, 0.6, 0.6, 0.3, 0.3)|
|- klue/bert-base|84.087% (0.042%⬇️)|best weight: (X, 1.0, 1.0, 0.4, 0.5)|
|- kykim/bert-base|83.829% (0.300%⬇️)|best weight: (0.4, X, 0.6, 0.5, 0.2)|
|- klue/roberta-base|83.712% (0.417%⬇️)|best weight: (0.8, 0.6, X, 0.7, 0.4)|
|- beomi/kcbert-base|84.012% (0.117%⬇️)|best weight: (0.4, 0.5, 0.5, X, 0.1)|
|- monologg/koelectra-base-v3-discriminator|84.071% (0.058%⬇️)|best weight: (0.9, 0.8, 1.0, 0.3, X)|

위 결과를 토대로 앙상블 성능에 가장 크게 기여한 모델을 순서대로 나열하면 다음과 같습니다.
1. klue/roberta-base
2. kykim/bert-kor-base
3. Klue/bert-base
4. beomi/kcbert-base
5. koelectra

## 9. 최종 결과

최종 성능은 Soft Voting을 사용한 **84.122%** 입니다.
각 모델의 Confusion matrix는 아래와 같습니다. 행을 기준으로 normalize 되어있으며 행은 정답 클래스, 열은 모델의 예측 클래스입니다.

<img width="1434" height="173" alt="image" src="https://github.com/user-attachments/assets/c122e498-1db0-4069-ace8-3a367ef926dd" />

## 10. 프로젝트 소감

수행한 실험을 정리해서 적고 보니 내용이 별로 없어 프로젝트가 하루 이틀만에 끝났을 것 같습니다만, 실제로는 여기에 기록되지 않은 많은 삽질과 뻘짓이 있었고 프로젝트를 수행하는 데 열흘 가량 걸렸습니다. 이게 제 첫 프로젝트라서 더 오래 걸렸던 것 같습니다..

이번 프로젝트에서 Transformer와 Tokenizer를 다뤄야 한다는 걸 들었을 때 솔직히 자신 없었습니다. 익숙하지 않고 아는 게 없으니까 두려웠던 것 같습니다. 하지만 막상 해보니 할 만 했습니다. 덕분에 Transformer와 Tokenizer에 대해 막연히 가지고 있던 두려움을 없애고 새로운 지식과 경험을 얻을 수 있었습니다.

이상입니다. 여기까지 제 프로젝트를 봐주셔서 감사합니다.

좋은 하루 되시길 바랍니다.

