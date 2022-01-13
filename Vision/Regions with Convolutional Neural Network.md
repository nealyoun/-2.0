# Regions with Convolutional Neural Network

reference : [https://ganghee-lee.tistory.com/35](https://ganghee-lee.tistory.com/35)

---

# Introduction

 

### Computer Vision 분류

1. Classification : Single Object 에 대한 Classification
2. Object Detection
    1. Object Localization : Single Object 에 대해 Bounding Box Regression 으로 Object 의 위치를 찾고**(Localization) + Classification**
    2. Object Detection : Multiple Object 에 대해 **Localization + Classification** 수행
3. Image Segmentation : Object 의 위치를 Bounding Box 가 아닌 Object 의 edge 로 찾음
4. Visual relationship

***Object Detection 은 1-stage detector 와 2-stage detector 가 있음***

**2-stage detector**

![스크린샷 2021-12-07 오후 3.58.28.png](Regions%20with%20Convolutional%20Neural%20Network%204bf480aba85443b0b9ae5ce196df2389/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-12-07_%EC%98%A4%ED%9B%84_3.58.28.png)

Selective Search, Region proposal network 와 같은 알고리즘이나 네트워크를 통해 Object가 있을 만한 영역 (RoI ; Region of Interest) 추출

추출된 각 역영들을 Convolution Network 를 통해 Classification, Bounding Box Regression 수행

Deep Learning Models : R-CNN, Fast R-CNN, Faster R-CNN, etc.

**1-stage detector**

![스크린샷 2021-12-07 오후 4.01.20.png](Regions%20with%20Convolutional%20Neural%20Network%204bf480aba85443b0b9ae5ce196df2389/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-12-07_%EC%98%A4%ED%9B%84_4.01.20.png)

RoI 영역을 추출하지 않고, 전체 Image 에 대해 Convolution Network 로 Classification, Bounding Box Regression 수행

2-stage detector 에 비해 Multiple Object Image (noise) 에 대해서 정확도는 떨어지지만,

모델이 간단하고 쉬운만큼 속도가 빠른 장점 (fps)

Deep Learning Models : YOLO

- Object Detection 에서 Region Proposal 을 하는 이유
    
    안녕하십니까,
    
    오브젝트 detection이 classification 대비 대표적인 어려운점이
    
    1. 이미지 내에 여러개의 오브젝트가 있으며 이들을 모두 detect 해야함.
    2. 오브젝트가 이미지내에 어떤 위치에 있는지 찾아야 함.
    3. 이미지내에 동일한 오브젝트가 여러개 있을시 이들을 모두 찾고 위치까지 찾아야 함.
    
    일반적으로 region proposal을 적용하지 않고, 이미지 내에서 ground truth bounding box를 기반으로 아무리 회귀 loss함수를 잘 만들어서  Detect를 하려고 해도 잘 안됩니다.
    
    특히 여러개의 오브젝트가 한 이미지에 있을 때 아무리 loss함수를 잘 만들어도 딥러닝에서 이를 정확히 찾아내기가 어렵습니다. 서로 다른 이미지에는 여러개의 서로 다른 오브젝트들이 서로 다른 위치에 놓여 있기 때문에 loss 함수가 쉽게 수렴해서 최적으로 오브젝트 detect를 해주기 어렵습니다.
    
    그래서 object가 있을 만한 위치를 먼저 찾아줍니다. 이 오브젝트가 있을 만한 힌트를 먼저 가진 다음에 (즉 딥러닝 학습시 오브젝트에 대한 제약 조건을 강화해서) 이를 기반으로 bounding box regression등의 학습을 진행하여 성능을 향상 시킵니다.
    
    object 가 있을 것으로 예상되는 regions(bounding boxes) 에서 다시 한번 위치를 알기 위한 regression 이 이루어진다
    

# R-CNN

- Object Detection 을 Deep Learning 에 처음 적용한 모델
- Region Proposal : Object 가 있을만한 위치 예측
    - ex. Sliding Window → window 의 크기, Object 의 크기 등 다양한 Issue
    - 자동으로 Object 가 있을만한 위치를 찾아주는 알고리즘 → Selective Search
    

![스크린샷 2021-12-06 오후 9.47.28.png](Regions%20with%20Convolutional%20Neural%20Network%204bf480aba85443b0b9ae5ce196df2389/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-12-06_%EC%98%A4%ED%9B%84_9.47.28.png)

### R-CNN process

- ImageNet 이미지를 통해 Feature Extractor (CNN model) 를 **pre-train**
- Object Detection Target Data 로 Convolution Model **fine-tuning**
    - CNN - Ground Truth 와 RoI 를 비교해서 IoU 가 0.5 이상인 경우 해당 Class 로, 아닌 경우 Background 로 fine-tuning
    - SVM Classifier - Ground Truth 로만 학습하되 0.3 IoU 이하는 Background 로 설정, 그 이상이지만 Ground Truth 가 아니면 무시
- **Selective Search** 를 통해 Regional proposal output 약 2,000개 추출 (**Image cropping**)
- 추출한 Regional proposal output 을 모두 동일 Input size (224x224) 로 **Image warping**
    - CNN 모델에서 Convolution Layer 는 Input size 가 고정되지 않지만, FC layer 의 Input size 는 고정이므로 Convolution Layer 에 동일한 Input size 로 입력
        - FC Layer 의 Input size 가 동일해야 하는 이유
            
            입력 사이즈가 3이고 히든레이어의 노드수가 4 일떄 이때 총 가중치 수는 12개이다.
            
            그러나 입력사이즈가 3이 아닌 4가 들어왔다면, 총 가중치의 수는 16이 된다. 그러면 기존에 없던 가중치들이 필요하기 때문에 문제가 될 수 있다 .그러면 기존에 없던 가중치들이 필요하기 때문에 문제가 될 수 있다 .
            
            fully connected layer의 출력 neuron 수를 정하고, 이를 모델의 layer에 붙일 때 이전 layer의 출력 결과 크기 즉, fully connected layer의 입력 크기는 고정됩니다. 이미지 별로 입력 크기가 동적으로 달라진다던가 할 수 없습니다.
            
            이것은 입력값과 neuron에 따라 선형대수 식이 맞아 돌아가야 하기 때문에 고정된 입력 사이즈가 보장되어야 합니다.
            
    - but. Image warping 으로 해상도가 무너지는 현상 발생
- 2,000개의 warped Image 를 각각 **CNN 모델에 입력**
    - CNN 모델은 AlexNet 구조를 거의 그대로 사용
- 각각의 Conv. 결과 (4,096-dimentional feature vector) 에 대해 **Classification / Localization** 수행

### Bounding Box Regression

Selective Search 로 만든 Bounding Box 에서 Localization error 를 줄이기 위해 Regression 수행

![스크린샷 2021-12-07 오후 5.14.45.png](Regions%20with%20Convolutional%20Neural%20Network%204bf480aba85443b0b9ae5ce196df2389/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-12-07_%EC%98%A4%ED%9B%84_5.14.45.png)

**Notations**

$p_i$ : Selective Search 로 만든 Bounding Box 의 중심 $x,\;y$ 좌표, $width, \; height$

**Ground truth 에 대한 예측값**

$\hat g_x = p_wd_x(p)+p_x$

$\hat g_y = p_hd_y(p)+p_y$ 

$\hat g_w = p_wexp(d_w(p))$

$\hat g_h = p_hexp(d_h(p))$

**Ground Truth : $d_i(p)$**

$t_x=(g_x-p_x)/p_w$

$t_y=(g_y-p_y)/p_h$

$t_w=log(g_w/p_w)$

$t_h=log(g_h/p_h)$

**Loss Function**

$$
L_{reg}=\sum_{i\in \{ x,y,w,h \}}(t_i-d_i(p))^2+\lambda||w||^2
$$

- Selective Search 의 중심좌표와 Ground Truth 의 중심좌표 사이 거리가 짧을 수록 좋음

- $d_x(P)$ 의 형태 / width, height 의 예측 값에 $exp$ 사용하는 이유
    
    R-CNN논문에 따르면 $d_x(p)$는 일반적인 선형함수로 되어 있습니다. 아마도 다음과 같은 형태로 추정됩니다.
    
    $d_x(p) = w_0x_0 + w_1x_1$
    
    그리고 논문에 따르면 $w$ 와 $h$ 에 $exp$ 를 적용한 이유는 넓이와 높이 값은 좌표값$(x,y)$ 와 스케일이 달라서 $exp$ 를 적용했습니다.
    
- Bounding Box Regression 의 Loss Function 에서 L2 norm 을 사용하는 이유
    
    학습 데이터만을 기반으로 하다보면 너무 학습 데이터에 최적화된 모델이 만들어진다. 학습 데이터가 아닌 다른 데이터 세트에서는 오히려 예측효율이 많이 떨어지는 오버피팅 현상이 나타날 수 있는데
    
    $L1, L2$ 규제(Regularization) 학습의 예측 오류가 최소화 되는 방향성을 어느정도 방해함으로써 방지
    

### R-CNN 한계

- SoftMax 대신 SVM Classifier 적용
    - why? 당시에 CNN fine-tuning 을 위한 학습 데이터가 많지 않아서 SoftMax 로 성능을 내기 어려웠음
    - SVM 은 CNN 으로부터 추출된 각각의 feature vector 들의 점수를 Class 별로 매기고, 객체인지 아닌지, 객체라면 어떤 객체인지 등을 판별하는 역할을 하는 Classifier
    - **Feature 를 다루는 단계에서만 딥러닝 사용**
- 복잡한 Architecture 와 train process → R-CNN modules : Region proposal + CNN + SVM
    - ***SVM, Bounding Box Regression 에서 학습한 결과가 CNN Model 을 업데이트 시키지 못함***
- 학습 시간 및 추론 시간이 오래 걸림 → Selective Search 를 통해 뽑아낸 이미지 각각에 대해 CNN 적용
    - 한 장의 이미지를 Object Detection 하는데 50초 소요 (**0.02 fps**)
    - e.g. Model 이 50초에 1장 Detection 을 수행한다면, 1초에 1/50 장 Detection 수행
        
        따라서, $1fps = {Image = 1 \over seconds = 50} = {1\over50} = 0.02fps$
        

- **R-CNN 이후 Object Detection 의 연구 방향성**
    - Deep Learning 기반 Object Detection 의 성능 입증
    - Region Proposal 기반 성능 입증
    - Detection 수행 시간을 줄이고 복잡하게 분리된 개별 아키텍처를 통합할 수 있는 방안 연구