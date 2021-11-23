# Inception, VGGNet, ResNet (Chapter 9)

Assign: Anonymous
Due Date: November 18, 2021
Status: Completed

# Inception Model

### 개요

- **병렬 처리** 합성곱 신경망 구조인 인셉션 모듈을 반복적으로 활용하여 신경망의 규모를 크게 늘린 모델
- 다양한 크기의 커널을 이용한 합성곱 계층을 병렬로 이용해서 **앙상블 효과** 기대
    - 한 가지 분석방법에 의존하기보다 여러 방법으로 분석하고 그 결과를 종합
- Inception model-v3 는 2014년 이미지넷 경진대회 ILSVRC(ImageNet Large Scale Visual Recognition Challenge) 우승

## Inception Model-v1 (GoogLeNet)

<img width="594" alt="스크린샷_2021-11-14_오후_5 17 17" src="https://user-images.githubusercontent.com/54128055/143047400-8f4a4a46-19f9-4818-8a3e-28758234c37b.png">

- 이미지에 표현되어 있지 않지만, 각 Convolution의 출력은 복수의 채널을 가지며 각 분기(branch)의 처리 결과들을 별도의 채널에 차례로 나열해서 합병 (concat)
    - 각 갈래의 처리 결과는 채널 축을 제외한 나머지 세 축에 대해 같은 형태를 가져야 함
    - 따라서, SAME 패딩을 적용하고, stride를 [1 , 1]로 설정
    - 마지막 분기의 3 x 3 Max pooling 은 입력과 같은 채널 수를 갖고, 나머지 Convolution의 결과 채널 수가 더해지기 때문에, 출력 채널 수는 입력 채널의 수보다 커짐

### 1 x 1 Convolution

- 합성곱 연산에서 하나의 출력 픽셀을 계산하는 데 입력 채널 전체가 반영된다
- 1 x 1 Convolution 은 입력 채널 전체와 출력 채널 전체를 연결하는 구조로, 각 픽셀 위치에 대해 출력 채널별로 입력 채널 정보를 종합한 내용 생성
    - Inception model 에서 Feature Map 의 수를 줄여, 연산량을 줄이는 목적으로 사용
    - ex) 480장의 14 x 14 사이즈 Feature Map (14 x 14 x 480)에 48개의 (5 x 5 x 480) 커널로 Convolution 연산을 수행하면, 48장의 Feature Map (14 x 14 x 48) 이 생성됨
        - 연산량은 (14 x 14 x 48) x (5 x 5 x 480) = 약 112.9M
    - 하지만, 16개의 (1 x 1 x 480) 커널로 Convolution 연산을 수행하면, Feature Map (14 x 14 x 16)이 생성됨. 이 후, 48개의 (5 x 5 x 16) 커널로 Convolution 연산을 수행하면 Feature Map (14 x 14 x 48) 로 동일한 사이즈의 Feature Map 을 얻을 수 있음
        - 연산량은 (14 x 14 x 16) x (1 x 1 x 480) + (14 x 14 x 48) x (5 x 5 x 16) = 약 5.3M

([https://bskyvision.com/539](https://bskyvision.com/539))

<img width="615" alt="스크린샷_2021-11-14_오후_5 54 24" src="https://user-images.githubusercontent.com/54128055/143048442-f3c37d6d-f95f-4b77-bb33-9dfdaaec9b52.png">

**개선된 Inception 모듈 vs 기본형 Inception 모듈**

- 기본적인 구조 및 분기의 역할은 비슷
- 1 x 1 Convolution 연산 추가
    - 각 픽셀 위치별로 채널 정보를 종합하는 단계를 두어 품질을 높이려는 의도
    - Max pooling 전후의 1 x 1 Convolution 을 통해 분기의 채널 수 변경
- 5 x 5 혹은 7 x 7 Convolution 을 n개의 3 x 3 Convolution 으로 대체 (출력 크기는 동일)
    - 5 x 5 Convolution 연산은 25회 곱셈연산을 수행하지만, 3 x 3 Convolution 연산을 두 번 수행하면 (3 x 3) + (3 x 3) = 18회 곱셈연산을 수행
    - 나아가서, n x n 커널을 1 x n 과 n x 1 Convolution 으로 변환하면 연산량을 더 줄일 수 있음
        - n x n 커널의 연산량은 $n^2$ 이지만, 1 x n 과 n x 1 커널로 대체하면 연산량은 $2n$
    - **연산량의 절감을 통한 학습 속도의 개선 목적**

## Inception Model-v2 & Inception Model-v3

논문명 : Rethinking the Inception Architecture for Computer Vision

([https://deep-learning-study.tistory.com/517](https://deep-learning-study.tistory.com/517))

- convolution 분해를 활용해서 연산량이 최소화 되는 방향으로 모델의 크기를 키우는데 집중
    - 성능을 올리기 위해 모델 크기를 증가시키면 연산량이 증가하게 되고, mobile과 같은 제한된 메모리에서 활용해야 할때 단점으로 작용

## Techniques

1. **더 작은 합성곱으로 분해 (Factorization into smaller convolutions)**
    
    <img width="322" alt="스크린샷_2021-11-18_오후_4 30 27" src="https://user-images.githubusercontent.com/54128055/143047916-266bd991-9640-4cf6-a35e-89be57fb7d67.png">
    
    - 큰 커널을 n개의 3 x 3 커널로 분해하여 연산량과 parameter 의 수를 줄임 (출력 크기 동일)
    - 앞서 언급한 내용 참조
    - Linear vs ReLU
        
        <img width="444" alt="스크린샷_2021-11-18_오후_4 32 27" src="https://user-images.githubusercontent.com/54128055/143047773-6bf75975-bce5-4756-b923-48263d843563.png">
        
        첫 번째 3 x 3 convolution 연산 후 Linear activation 적용, 두 번째는 ReLU activation 적용한 모델과 두 개의 3 x 3 convolution 연산 후 ReLU activation 적용한 모델의 성능을 비교한 그래프
        
        두 번 모두 ReLU 를 사용했을 경우 정확도가 더 높았으며, 추가적으로 Batch normalization 을 사용했을 때 더 높은 정확도를 얻음
        

1. **비대칭 합성곱 분해 (Asymmetric Convolutions)**
    
    <img width="432" alt="스크린샷_2021-11-18_오후_4 37 29" src="https://user-images.githubusercontent.com/54128055/143047957-0401b6c8-5fc8-4df2-bbca-a24668c4c190.png">
    
    - 3 x 3 convolution 을 1 x 3 convolution, 3 x 1 convolution 으로 분해
        - 저자의 실험에 따르면, 2 x 2 convolution 으로 분해하는 것보다 비대칭 분해가 성능이 더 좋음
        - 비대칭 분해의 경우 약 33%의 연산량 절감 / 2 x 2 convolution 은 약 11% 연산량 절감
    
    if) n x n convolution 을 n x 1 , 1 x n convolution 으로 분해하면?
    
    <img width="286" alt="스크린샷_2021-11-18_오후_5 07 09" src="https://user-images.githubusercontent.com/54128055/143048019-6e5dc2ba-197e-4a36-8a12-fc4d031f8ce1.png">
    
    - 7 x 7 convolution 을 7 x 1 , 1 x 7 convolution 분해한 형태
    - 실험에서 Feature Map size 가 12 - 20 일 때 효과가 좋았음
        - Inception model-v2 에서 Feature Map size $\fallingdotseq$ 17 구간에 해당 Inception module 사용

1. **보조분류기 (Utility of Auxiliary Classifiers)**
    
    <img width="645" alt="스크린샷_2021-11-18_오후_5 14 21" src="https://user-images.githubusercontent.com/54128055/143048079-ac7309b4-3639-4565-ae20-0f38d2fd9067.png">
    
    - 해당 그림은 Liwei Wang 의 논문에 사용된 실험용 DNN의 구조, Supervision 이라는 보조분류기 사용
    - Auxiliary Classifier 를 통해 Vanishing Gradient 문제 해결
        - 그림의 X4 위치에서 보조분류기와 최종 출력의 back propagation 결과 결합
        - 두 back propagation 이 더해지기 때문에 Gradient 가 작아지는 문제를 피할 수 있다
    - Auxiliary Classifier 의 적용 위치는 저자가 Iteration을 통해 Gradient 의 변화를 관찰하고 배치
        
        <img width="635" alt="스크린샷_2021-11-18_오후_5 20 08" src="https://user-images.githubusercontent.com/54128055/143048138-414fbfee-15df-4819-9283-5dbedaf6a4ff.png">

        
    - 좌측 그림은 Auxiliary Classifier 미적용 / Iteration 이 증가할수록 Gradient 가 현저하게 감소
    - 우측 그림은 convolution X4 에서 Gradient 에 대한 그래프
        - 파란 점선은 보조분류기 미적용, 빨간 실선은 보조분류기 적용
    - 이 후 2015년 GoogLeNet 의 저자가 발표한 논문에 Auxiliary Classifier 언급
        - Auxiliary Classifier 는 **Regularizer** 와 같은 역할
        - Batch Normalize or Drop-out Layer 를 갖고 있는 경우 성능이 더 좋음
    - Auxiliary Classifier 는 학습을 도와주는 역할, 학습된 모델을 사용할 때는 Auxiliary Classifier 제거
        
        
2. **효율적인 그리드 크기 축소 (Efficient Grid Size Reduction)**
    - 일반적인 CNN 신경망은 Feature Map 의 사이즈를 줄이기 위해서 pooling 연산 사용
    - Pooling 연산을 먼저 하고 Inception Module 을 적용하면 Feature Map 의 사이즈가 줄어들어 연산량이 감소하지만, 정보의 손실에 의해 신경망의 표현력 (representation) 도 감소
        
        ![스크린샷_2021-11-18_오후_8 18 50](https://user-images.githubusercontent.com/54128055/143048197-837794dd-3414-43d1-ab7b-6dd8f2fa0005.png)
        
    - Representational Bottlenet 을 피하기 위해 필터 수를 증가시킴
        - stride 2 인 pooling layer 와 convolution layer 를 병렬로 사용
        - 정보의 손실없이 연산량 감소
        
        ![스크린샷_2021-11-18_오후_8 29 48](https://user-images.githubusercontent.com/54128055/143048646-10d02c7a-f9a1-4f81-9b09-7401fca4b2f7.png)
        

## Inception Model-v2

- 42 Layers 신경망, 연산량은 GoogLeNet(Inception Model-v1)보다 2.5배 많고, VGGNet과 비슷
- 각 Inception Module 에서 convolution 연산은 0-padding 적용, 그 외 convolution layer 에는 미적용
- Inception Model-v2 는 3가지 모듈 사용 (Factorization, Asymmetric, Expanded Filter)
    
    ![스크린샷_2021-11-18_오후_8 37 17](https://user-images.githubusercontent.com/54128055/143048731-66da6db7-bb46-40e3-a361-b52ad04915ed.png)
    
    ![스크린샷_2021-11-18_오후_8 37 28](https://user-images.githubusercontent.com/54128055/143048773-fd4b963a-44ff-46cc-a4ec-5fc293e6e986.png)
    
    ![스크린샷_2021-11-18_오후_8 37 39](https://user-images.githubusercontent.com/54128055/143048795-d12ca6c8-7d1e-4193-87e1-a245f3cc336e.png)
    

- 참고
    
    ![스크린샷_2021-11-18_오후_8 42 29](https://user-images.githubusercontent.com/54128055/143048842-7cb88122-3040-4830-a32f-ce59d503b2cb.png)

    
    Label smoothing 이 왜 정답에 대한 확신을 감소시키는 건지 모르겠어연
    
    과신뢰도 방지
    
    너무 확신을 가지고 틀린다? > Robust
    

## Inception Model-v3

- Inception Model-v2 구조에서 Batch Normalized Auxiliary Classifier, RMSProp, Label Smoothing + Factorized 7 x 7 을 적용한 모델

## VGG-19

- Layer를 Deep하게 쌓기 위해 3x3커널만 사용 > 파라미터 수 감소, 연산량 감소 + ReLU 적용 횟수 증가
    - Network의 Depth가 정확도에 미치는 영향 연구 목적
    - Inception model은 앙상블 효과를 기대하고 다양한 커널을 사용한 것과 비교됨
    - 이 후, Plain-34 모델을 통해 Layer의 수를 34개 까지 늘림
    - ResNet의 저자는 Depth의 영향에 대해 연구하다가 단순히 Depth를 늘렸을 때, 정확도 감소 현상 발견
    

![스크린샷_2021-11-18_오후_8 49 26](https://user-images.githubusercontent.com/54128055/143048884-442c3fef-47ae-4587-9e83-1b647db049de.png)


### VGG-19 모델 구성

1. 3 x 3 convolution layers 뒤에 Max pooling layer 배치, 이 과정을 5회 반복
2. Feature Map 이 큰 모델 초기 모듈에는 convolution layer 를 2개만 배치하고 뒤에는 4개씩 배치
3. 각 모듈의 첫 convolution layer 는 채널 > 해상도 수를 반으로 줄이고 나머지 layer 는 유지 > 책내용
    - 채널 x 2 , 해상도 / 2 아닌가?
4. 이미지의 해상도는 pooling layer 에서 가로, 세로 각각 절반으로 줄어듦 (전체의 1/4)
5. 모듈은 진행될수록 압축적이고 추상적인 정보가 만들어짐
6. 폭이 4,096 인 Fully Connected Layer 3개 사용 (출력계층을 고려하면 총 4개)
    - why 4,096? [https://bskyvision.com/504](https://bskyvision.com/504) 경험법칙?
    - 합성곱 계층을 연달아 배치하듯 지역적으로 수집된 정보를 종합하는 과정도 반복이 필요하다고 생각?

Plain-34 는 VGG-19 와 동일한 구조에서 3 x 3 convolution layer 를 33개로 늘리고 Fully Connected Layer 는 출력계층을 제외하고 제거한 모델

### ResNet

- Residual 계층 추가
- Convolution 2개의 계층 사이에 Input x를 더하는 연산 수행
    - 기존 Convolution은 x를 통해 y를 추정하는 목적함수 $H(x)$를 구하는 과정
    - ResNet은 Layer의 입력을 Layer의 출력에 연결하는 skip connection 사용
        - $F(x) + x$ 를 최소화하는 것이 목적
        - 기존 Conv 연산은 $H(x)$ 를 구하는 것이 목적이고 ResNet은 $F(x)$를 0으로 만드는 것이 목적
            - $H(x) = F(x) + x$ 라고 할 때, $F(x) = H(x) - x$ 가 됨
            - $x$는 현시점에서 변할 수 없는 값이므로 $F(x)$를 0에 가깝게 만드는 것이 목적
            - $F(x)$가 0이 되면 출력과 입력이 모두 $x$로 같아지게 된다
            - $F(x) = H(x) - x$이므로 $F(x)$를 최소로 해준다는 것은 $H(x) - x$를 최소로 해주는 것과 동일한 의미를 지닌다. 여기서 $H(x) - x$를 **잔차(residual)**라고 한다. 즉, 잔차를 최소로 해주는 것이므로 ResNet이란 이름이 붙게 된다.
            

[https://m.blog.naver.com/laonple/220764986252](https://m.blog.naver.com/laonple/220764986252)

[https://bskyvision.com/504](https://bskyvision.com/504)
