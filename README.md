# HYU_AI-X_DL_G03
# TITLE
## AI 이미지/영상 판별 (임시)
## 목차
- [PRELIMINARY_RESEARCH](#PRELIMINARY_RESEARCH)
- [MEMBERS](#MEMBERS)
- [PROPOSAL](#PROPOSAL)
- [DATASETS](#DATASETS)
- [METHODOLOGY](#METHODOLOGY)
- [EVALUATION&ANALYSIS](#EVALUATION&ANALYSIS)
- [RELATED_WORKS](#RELATED_WORKS)
- [CONCLUSION](#CONCLUSION)

## PRELIMINARY_RESEARCH
    
### 강경호

[깃허브 서식 쓰는 법](https://docs.github.com/ko/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

[AI생성 이미지 판별 대회 리뷰 - YouTube 영상](https://youtu.be/fmiDrEZ9vyw?si=fJafWSkHFMEq1z_h)

[OpenAI SORA 관련 - YouTube 영상](https://youtu.be/S_QfBYnGKnc?si=RwRCt_ph4-BJJ7BR)

[AI Generated Video Detection 주제 최신 논문 업데이트 중인 깃허브 블로그](https://github.com/chenhaoxing/Awesome-AI-Generated-Video-Detection?tab=readme-ov-file)

[시공간적 이상 학습을 통한 AI 생성 비디오 감지 논문](https://arxiv.org/abs/2403.16638)

[Beyond Deepfake Images: Detecting AI-Generated Videos-딥페이크관련](https://arxiv.org/abs/2404.15955)

[WinstonAI-AI 탐지기, 자칭 99.98% 정확성](https://gowinston.ai/ai-video-detection/)

의견: 아직 신뢰할 수준의 ai-생성 영상 탐지기가 있는 것 같지는 않음. 딥페이크 판독은 자료들이 더 많긴 하지만 경과는 크게 다르지는 않은 듯. 정지 이미지의 경우는 찾아보지 않아서 모르겠음. 아래는 ChatGPT4o의 방법 제안.
>How to detect ai generated video using AI/ML/DL? please in Korean

```
AI/ML/DL을 사용하여 AI 생성 비디오를 감지하는 방법은 다음과 같습니다:

특징 추출:

이미지 및 영상 특징 추출: AI 생성 비디오는 실제 비디오와는 다른 특징을 가질 수 있습니다. CNN(Convolutional Neural Network)을 사용하여 비디오의 프레임에서 주요 특징을 추출할 수 있습니다.
움직임 분석: RNN(Recurrent Neural Network)이나 LSTM(Long Short-Term Memory) 모델을 사용하여 비디오의 시간적 흐름과 움직임 패턴을 분석할 수 있습니다. AI 생성 비디오는 자연스럽지 않은 움직임을 보일 수 있습니다.
딥페이크 감지 모델:

ResNet, VGGNet 등 CNN 기반 모델: 딥페이크를 감지하기 위해 특별히 훈련된 CNN 모델을 사용할 수 있습니다. 이러한 모델은 이미지 내의 미세한 변화를 감지할 수 있습니다.
EfficientNet, Xception: 최신 모델을 사용하면 감지 성능이 향상될 수 있습니다. 이들은 더 적은 파라미터로 높은 성능을 발휘합니다.
데이터셋 활용:

딥페이크 데이터셋 사용: 훈련을 위해 공개된 딥페이크 데이터셋(예: FaceForensics++, DeepFake Detection Challenge Dataset)을 사용할 수 있습니다. 이러한 데이터셋은 다양한 AI 생성 비디오 샘플을 포함하고 있어 모델 훈련에 유용합니다.
진짜 및 가짜 비디오 데이터셋: 실제 비디오와 AI 생성 비디오를 포함한 데이터셋을 수집하여 모델을 훈련시킵니다.
전처리 및 증강:

데이터 전처리: 영상 데이터를 정규화하고, 크기 조정 및 정렬을 수행합니다.
데이터 증강: 학습 데이터를 증대하여 모델의 일반화 성능을 향상시킵니다. 예를 들어, 회전, 뒤집기, 밝기 조절 등을 통해 다양한 데이터를 생성합니다.
모델 훈련:

전이 학습(Transfer Learning): 미리 훈련된 모델을 사용하여 새로운 데이터셋에 맞게 재훈련합니다. 이는 훈련 시간을 단축하고 성능을 향상시킬 수 있습니다.
교차 검증: 다양한 검증 기법을 사용하여 모델의 성능을 평가하고 과적합을 방지합니다.
후처리 및 검증:

결과 해석: 모델의 예측 결과를 해석하고, 신뢰도 높은 검출을 위해 후처리 단계를 추가할 수 있습니다. 예를 들어, 예측 결과를 기반으로 가짜 비디오일 확률을 계산합니다.
테스트 및 평가: 훈련된 모델을 실제 데이터에 적용하여 성능을 평가합니다. 정확도, 정밀도, 재현율, F1 점수 등을 사용하여 모델의 성능을 측정합니다.
위의 방법들을 결합하여 AI/ML/DL 기술을 사용한 AI 생성 비디오 감지 시스템을 구축할 수 있습니다. 기술이 발전함에 따라 더 정교하고 정확한 감지 방법이 개발되고 있습니다.
```


### 신승민

### 이상훈
[메타에서 개발한 AI 워터마킹 시스템](https://www.aitimes.kr/news/articleView.html?idxno=29175)

[AI가 생성한 텍스트의 판별과 관련된 기사](https://www.ciokorea.com/news/275659)

[한국 기업 딥브레인AI와 경찰청이 협력해 영상, 이미지, 음성을 판별하는 모델 제작, 기존의 서양인 기반 데이터 이외에도 한국인 및 아시아인의 데이터도 500만건 이상 대입해, 신뢰도 올렸다.](https://www.aitimes.kr/news/articleView.html?idxno=30624)

[딥페이크 관련 이미지를 지역화 된 표정 변화를 바탕으로 높은 확률로 판별해 내는 모델 개발](https://arxiv.org/pdf/2103.08134)

의견: 저도 경호님 의견과 마찬가지로 아직 신뢰도가 높은 AI 생성물 판독기는 없는 것 같습니다. 일단 지금부터 생성되는 AI의 결과물은 워터마크와 함께 사용해 예방하고, 이미 생성되어있는 것들은 이미지를 지역화 시켜 CNN으로 판독한다면 그래도 좀 높은 신뢰도를 가지고 판별할 수 있을 것 같습니다. 그리고 텍스트, 이미지, 영상, 음성 등 생성형 AI의 결과물의 종류도 다양한데, 어느쪽을 집중적으로 생각해 볼 지도 정해봐야 할 것 같습니다.
### 조성재
[https://www.pcmag.com/how-to/how-to-detect-ai-generated-video-footage](https://www.pcmag.com/how-to/how-to-detect-ai-generated-video-footage)

[https://www.foolproofme.org/articles/975-how-to-recognize-ai-generated-images-and-videos-artificial-intelligence](https://www.foolproofme.org/articles/975-how-to-recognize-ai-generated-images-and-videos-artificial-intelligence)

[중복,https://arxiv.org/abs/2404.15955](https://arxiv.org/abs/2404.15955)

[중복,https://gowinston.ai/ai-video-detection/](https://gowinston.ai/ai-video-detection/)



## MEMBERS
이름|학과|학번|역할|이메일
---|---|---|---|---|
강경호|건축학부|2017025023|조장,|kgh7296@naver.com
신승민|성악과|0000000000|1|
이상훈|융합전자공학부|0000000000|2|
조성재|데이터사이언스학부|0000000000|3|
## PROPOSAL
### MOTIVATION
최근 생성형AI 분야가 급속도로 발전하여 이미지, 딥페이크 영상을 넘어, 아예 AI로 제작된 영상에 대한 소식이 들려오고 있다. 사람도 구분하기 힘든 수준으로 올라온 AI 이미지를 판별해주는 모델을 탐색하고 검증하는 것을 목표로 프로젝트를 시작했다.
### GOAL


## DATASETS
### DATASET LINK


### PREPROCESSING
## METHODOLOGY
    Explaining your choice of algorithms (methods)
    Explaining features (if any)
## EVALUATION&ANALYSIS
    Graphs, tables, any statistics (if any)
## RELATED_WORKS
    Tools, libraries, blogs, or any documentation that you have used to do this project.
## CONCLUSION

