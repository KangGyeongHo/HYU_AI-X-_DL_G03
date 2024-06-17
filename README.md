# HYU_AI-X_DL_G03
# TITLE
## AI 생성 이미지 분류 모델
[설명영상 YouTube 링크](https://www.youtube.com/)
## 목차
- [MEMBERS](#MEMBERS)
- [PROPOSAL](#PROPOSAL)
- [DATASETS](#DATASETS)
- [METHODOLOGY](#METHODOLOGY)
- [EVALUATION&ANALYSIS](#EVALUATION&ANALYSIS)
- [RELATED_WORKS](#RELATED_WORKS)
- [CONCLUSION](#CONCLUSION)

## MEMBERS
이름|학과|학번|역할|이메일
---|---|---|---|---|
강경호|건축학부|2017025023|조장,Datasets,python코딩,블로그 작성|kgh7296@naver.com
신승민|성악과|2018031830|YouTube 영상 녹화 및 게시|fjdkslafj@naver.com
이상훈|융합전자공학부|2017028268|ResNet50, ViT 이론 설명 작성|tan981212@naver.com
조성재|데이터사이언스학부|2024002778|영상 대본 작성, 블로그 초안|csjchris08@hanyang.ac.kr
## PROPOSAL
### MOTIVATION
최근 생성형AI 분야가 급속도로 발전하여 이미지, 딥페이크 영상을 넘어, 아예 AI로 제작된 영상에 대한 소식이 들려오고 있다. 
실제 이미지와 구분하기 힘든 수준으로 올라온 AI 생성 이미지를 판별하는 모델을 탐색하고 검증하는 것을 목표로 프로젝트를 시작했다.
### GOAL
AI 생성 이미지와 실제 이미지 데이터셋을 이용하여 ResNet50 모델을 검증해보고, AI 생성 영상 판별로 확장하는 방법을 탐구한다.

## DATASETS
### DATASET LINK
캐글에서 찾은 ai생성/실물 사과 이미지 데이터셋을 이용하여 모델을 학습시킨다.

[Dataset of AI Generated Fruits and Real Fruits](https://www.kaggle.com/datasets/osmankagankurnaz/dataset-of-ai-generated-fruits-and-real-fruits?resource=download)

## METHODOLOGY
    Explaining your choice of algorithms (methods)
ResNet50

ResNet은 2014년에 나왔음에도 불구하고 현재까지도 대표적인 CNN구조로 꼽히고 있다. 먼저 ResNet의 가장 기초적인 구조는 Residual Block이다. ResNet은 블록 단위로 층을 쌓는데 그 구조는 아래와 같다.

![ResNet50-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/6bd2b889-8a95-4664-bd09-9eabf84cc095)

Conv 층을 통과한 F(X) 과 Conv 층을 통과하지 않은 X를 더하는 과정을 Residual Mapping이라고 한다. 위 Residual Block이 여러 개 쌓여서 나온 CNN 모델을 Residual Network(ResNet)이라고 부른다. 모델명에 붙은 숫자는 층의 개수를 의미한다. 즉, ResNet18은 18개의 층이 있다는 소리이고 ResNet50은 50개의 층이 있다는 의미이다. 다른 CNN 종류인 AlexNet이나 GoogLeNet이 가지고 있는 문제점은 바로 층이 깊어짐에 따라 발생하는 과적합과 기울기 소멸 문제이다. 연산 중에 의미 없는 파라미터의 수가 발생하면 연쇄적으로 다음 층에 영향을 미치기 때문에 이를 해결할 방법을 강구해야만 했다. ResNet은 잔차 연결(Skip Connection)을 사용하는 Residual Block을 제안하면서 층이 깊어짐에 따른 과적합이나 기울기 소멸 문제를 해결해버렸다.
ImageNet 데이터에 맞춰진 ResNet은 7 X 7 필터를 사용하는 합성곱과 3 X 3 Max Pooling을 사용한다. ResNet50은 아래의 표에서 50-layer부분에 해당한다.

![ResNet50-2](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/956ed96a-5207-4bdb-84b1-b0eff37b2e7e)

이러한 층 방식을 통해 학습시키고, 마지막에 테스트를 걸쳐 모델을 평가할 수 있다.
    Explaining features (if any)
이번 프로젝트에서 사용할 데이터셋은 캐글에서 찾은 AI 생성 과일과 실제 과일 이미지 데이터셋이다. 이 데이터셋은 총 306개의 파일로 구성되어 있으며, AI 생성 사과와 실제 사과 이미지를 포함한다. 데이터셋은 녹색 사과와 빨간 사과로 나뉘며, 각 카테고리는 오버헤드 샷과 측면 샷으로 세분화되어 있다. 구체적으로, AI 생성 녹색 사과와 빨간 사과 각각 75개씩, 실제 녹색 사과 77개와 빨간 사과 79개로 구성되어 있다.
데이터 전처리 과정에서는 각 이미지를 로드하고 라벨을 설정한 후, 층화 샘플링을 통해 각 클래스가 균형 있게 분포되도록 했다. 데이터셋을 학습용, 검증용, 테스트용으로 분할하여 모델의 성능을 평가할 수 있도록 준비했다.
이미지|오버헤드|측면|
---|---|---|
AI-Green|![green-apple-ohs-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/d1d483f4-afa1-425f-b4d2-b228ca8dcd58)|![green-apple-ss-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/a88f8a7c-54d9-4e74-b548-e0c58f1987aa)|
AI-Red|![red-apple-ohs-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/2fd24c7a-d775-46ad-96ba-ec075566beb7)|![red-apple-ss-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/cca52f95-eaf6-4ede-8973-eacaad49eeb8)|
REAL-Green|![green-apple-ohs-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/bc44c26f-b101-40b7-a186-6559d192d100)|![green-apple-ss-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/2a7da03e-7059-464d-8507-19e1a0f61073)|
REAL-Red|![red-apple-ohs-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/0decfabf-f504-41e5-bef0-53e5911e9c32)|![red-apple-ss-1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/d0ddf7e3-c844-465e-a86e-7da2400b761b)|

## EVALUATION&ANALYSIS
Graphs, tables, any statistics (if any)
```
(py311) PS C:\Users\220102KANG\Desktop\건축학부\2024-1\AIX딥러님> python AppleResNet50.py
cuda
Epochs 1 train loss 0.0533266332550127 val 0.9221311475409836 val loss 0.041465121892190746 acc 1.0
Epochs 2 train loss 0.0406652322558106 val 1.0 val loss 0.04069767363609806 acc 1.0
Epochs 3 train loss 0.04054069714468034 val 1.0 val loss 0.040602121622331684 acc 1.0
Epochs 4 train loss 0.04001102413310379 val 1.0 val loss 0.04050751078513361 acc 1.0
Epochs 5 train loss 0.04060975498840457 val 0.9959016393442623 val loss 0.040809701527318644 acc 1.0
Epochs 6 train loss 0.03999627625844518 val 1.0 val loss 0.04048849882618073 acc 1.0
Epochs 7 train loss 0.039921326837578756 val 1.0 val loss 0.040462009368404266 acc 1.0
Epochs 8 train loss 0.04062260821705959 val 0.9959016393442623 val loss 0.04064356319365963 acc 1.0
Epochs 9 train loss 0.04047424724844635 val 0.9959016393442623 val loss 0.040491112778263706 acc 1.0
Epochs 10 train loss 0.039891102641332346 val 1.0 val loss 0.040480745415533745 acc 1.0
Epochs 11 train loss 0.03991010711818445 val 1.0 val loss 0.04046798713745609 acc 1.0
Epochs 12 train loss 0.039844179617576914 val 1.0 val loss 0.040466620076087215 acc 1.0
Epochs 13 train loss 0.03985497709669051 val 1.0 val loss 0.040462729430967764 acc 1.0
Epochs 14 train loss 0.04010496996953839 val 1.0 val loss 0.04047364092642261 acc 1.0
Epochs 15 train loss 0.03985128893715437 val 1.0 val loss 0.04045112094571514 acc 1.0
Epochs 16 train loss 0.040009948318121866 val 1.0 val loss 0.04046387345560135 acc 1.0
Epochs 17 train loss 0.03985562432007712 val 1.0 val loss 0.0404480763020054 acc 1.0
Epochs 18 train loss 0.03983433898843703 val 1.0 val loss 0.04044248404041413 acc 1.0
Epochs 19 train loss 0.03984034464496081 val 1.0 val loss 0.04044307720276617 acc 1.0
Epochs 20 train loss 0.0399602397543485 val 1.0 val loss 0.040439993143081665 acc 1.0
Epochs 21 train loss 0.039815275517643474 val 1.0 val loss 0.04044123714970004 acc 1.0
Epochs 22 train loss 0.03983171589550425 val 1.0 val loss 0.040444351973072175 acc 1.0
Epochs 23 train loss 0.03983920421756682 val 1.0 val loss 0.04043821942421698 acc 1.0
Epochs 24 train loss 0.039814882102559825 val 1.0 val loss 0.04043723787030866 acc 1.0
Epochs 25 train loss 0.03986351143141262 val 1.0 val loss 0.040438818354760445 acc 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        14
           1       1.00      1.00      1.00        17

    accuracy                           1.00        31
   macro avg       1.00      1.00      1.00        31
weighted avg       1.00      1.00      1.00        31
```
ResNet50 모델의 학습 및 검증 결과를 보면, 학습 정확도와 검증 정확도가 매우 높은 수준을 유지하고 있다. 이는 모델이 데이터셋을 잘 학습했음을 의미한다.

![Figure_1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/e47d0002-623f-463a-becb-1590db9fdd6e)

![Figure_3](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/dc69c90a-c201-489a-ab42-7e09258b72b1)

아래 이미지는 모델이 예측한 결과를 보여준다. 각 이미지 위에는 모델이 예측한 라벨과 실제 라벨이 표시되어 있으며, 예측된 라벨과 실제 라벨이 모두 일치하는 것을 확인할 수 있다.
또한, 혼동 행렬을 통해 모델의 예측이 얼마나 정확한지 시각적으로 확인할 수 있다. 이 모델은 모든 테스트 데이터에 대해 정확한 예측을 보여주고 있다.

![Figure_2](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/2055329a-b01e-4937-974a-15ac6f409b33)

## RELATED_WORKS
    Tools, libraries, blogs, or any documentation that you have used to do this project.
Tool로는 Python을 사용하였으며, 이 프로젝트를 위해 사용된 라이브러리는 다음과 같다.
Pandas: 데이터 처리 및 분석을 위해 사용. 데이터 프레임을 통해 데이터셋을 효율적으로 관리하고 조작할 수 있다.
NumPy: 수치 연산을 효율적으로 수행하기 위해 사용.
Matplotlib 및 Seaborn: 데이터 시각화를 위해 사용. 데이터의 분포를 시각적으로 확인하고 분석할 수 있다.
OpenCV 및 PIL: 이미지 처리 및 전처리를 위해 사용.
Scikit-learn: 데이터 분할 및 성능 평가를 위해 사용.
PyTorch: 모델 학습 및 평가를 위해 사용. PyTorch를 통해 ResNet50 모델을 구현 및 학습.

추가적으로, ViT(Vision Transformer) 모델도 관련 연구로서 함께 소개하고자 한다.
Vision Transformer는 Ai/ML의 자연어 처리 분야에서 각광받고 있는 Transformer 구조를 Language가 아닌 Vision 영역에 적용한 구조이다. 기본적인 ViT의 구조는 아래와 같다.
![ViT1](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/4965ba7b-2b17-4f29-a33c-fdbdfda145cb)
ViT를 진행하는 순서는 크게 Image Patch, Patch Embedding, Position Embedding, Transformer Encoder로 이루어져 있다. 먼저 Image Patch는 이미지를 여러 개의 patch로 나누어서 embedding layer에 들어갈 vector를 만들어내는 과정이다. 위의 과정에서 만들어진 patch들을 patch embedding에 집어넣는 과정이 Patch Embedding이다. 이를 좀 더 자세히 설명하면, input vector 들이 Layernorm을 지나 linear layer를 지난다.
![ViT2](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/c03665fb-2f8d-48e3-b0aa-9f2a864f8b72)
위 그림에서는 linear layer의 dimension 이 3이라고 하면, 이제 4x768 -> 4x3으로 바뀌어 중간 output 이 나오고 이게 layernorm을 지나 최종 embedding vectors 가 생성이 된다. 이제 앞서 나눈 patch embedding 들을 Transformer에 집어넣을 때, patch 가 원래 이미지 어느 위치에 있던 간에 한꺼번에 다 같이 집어넣는다.
![ViT3](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/c92ee7a9-e6c0-41e5-842b-e4aabee9ca62)
그렇기 때문에, 좀 더 높은 성능을 위해서 각 patch에 원래 이미지의 위치 정보를 추가해야 한다. 그를 위해 필요한 것이 Position embedding이다. 위에서 생성된 Final Embedding Vectors 들은 Dropout에 지나가고 다음 Transformer Block 여러 개를 지난 후에 최종적으로 MLP head를 지나 최종 classification output 이 나온다.
![ViT4](https://github.com/KangGyeongHo/HYU_AI-X-_DL_G03/assets/168617928/f150016a-e279-4898-af65-3113f6303c0f)
위에서 Transformer Block을 자세히 그리면 위 그림과 같다. 앞서 만든 CLS token까지 해서 5개의 embedding vector 들(5x3)이 LayerNorm을 지나 MSA(Multi-head Self Attention) block을 지나게 된다. 위 그림에선 head 개수는 3개 head_dimension(내부에서 쓰이는 hidden layer dimension) 은 4로 표현했다. 즉, 5x3 이었던 input 이 MSA 안에서 3x4의 Query, Key, Value Weight 들을 지나서 Attention 값을 추출하는 계산을 거쳐 총 3개의 5x4 matrix 가 생성이 된다. (head 개수가 3개이므로) 이 3개의 matrix를 concat 해서 5x12 matrix를 만들고 이를 hidden linear layer을 거쳐 원래의 dimension이었던 3으로 또 추출된다. 즉 5x3의 Attention output 이 생성된다. 후로, 이제 MLP block을 거치게 된다. 5x3 Attention output 이 LayerNorm 지나고 또 Linear Layer에 들어간다. 여기서 MLP_dimension 은 4로 표현했다. 그래서 해당 Linear Layer를 거쳐 5x4 로 생성되고, GELU + Dropout을 지나고 다시 Linear Layer를 통해 5x3의 matrix 로 돌아온다. 그리고 최종적으로 Dropout을 거쳐 Transformer의 Output 이 출력된다. 이런 MSA+MLP 로 이루어진 Transformer Layer 가 N번 반복되어서 지나가고 마지막에 MLP Head를 거쳐 최종 Classification Output 이 나타나는 구조가 ViT(Vision Transformer)이다.

다음과 같은 블로그와 문서들을 참고하였습니다:
- [ResNet50 모델 원본](https://www.kaggle.com/code/a3amat02/ai-vs-real-apple-resnet50-100-accuracy#DataFrame-creation)
- [ResNet50 설명 블로그](https://blog.naver.com/wooy0ng/222653802427)
- [Vision Transformer 설명 블로그](https://mishuni.tistory.com/137)
- [Vision Transformer 논문](https://arxiv.org/abs/2010.11929v2)

## CONCLUSION
이 프로젝트를 통해 AI 생성 이미지와 실제 이미지를 구분하는 모델을 성공적으로 검증하였으며, 나아가 AI 생성 영상까지 판별할 수 있는 확장 가능성을 탐구할 것이다. ResNet50 모델의 성능을 통해 AI 생성 이미지와 실제 이미지를 정확히 구분할 수 있었으며, 이를 기반으로 다양한 응용 분야에서 최적의 모델을 선택할 수 있는 기준을 마련하고자 한다. 향후 연구 방향으로는 더 큰 데이터셋을 사용하여 모델의 일반화 능력을 향상시키고, 다양한 생성형 AI 모델을 활용하여 성능을 더욱 개선할 수 있을 것이다. 최근 Stability.ai 사의 SVD-xt에 대해 알아봤는데, 영상의 경우 프레임 단위로 이미지를 분할하여 동일한 ResNet 또는 ViT 모델로 학습시켜 판별할 수 있을 것 같다. 
