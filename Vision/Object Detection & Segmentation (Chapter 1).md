# Object Detection & Segmentation (Chapter 1)

Reference: https://blog.naver.com/cjh226/220809060935

## Object Detection과 Segmentation의 이해

![Untitled](https://user-images.githubusercontent.com/54128055/140332678-6a976884-b52d-4321-a03d-eaed828e3633.png)

- 2012 년 ImageNet (Image Classification Competition) 에서 우승한 **AlexNet**의 등장 이전까지는 고도화된 Algorithm을 통해 Image Classification과 Object Detection 수행
- AlextNet은 **CNN**을 기반으로 한 딥러닝 모델을 이용해서 mAP (Object Detection 에서 사용되는 모델의 성능 평가 지표) 가 70% 후반 정도의 성능을 보여줌

### Object Detection 과제의 분류

- **Localization** : 하나의 Object 위치를 Bounding Box로 지정해서 Detecting
- **Object Detection** : 여러 Object들에 대한 위치를 Bounding Box로 지정해서 Detecting
- **Segmentation** : Detection보다 발전된 형태로 Pixel Level Detection 수행

![Untitled 1](https://user-images.githubusercontent.com/54128055/140332734-f7b7e521-3068-43e5-8009-61f42a9d4a3f.png)

Localization과 Detection은 Bounding Box Regression (Box의 좌표 예측) 과 Classification 두 개의 문제가 결합되어 있는 형태

Detection은 여러 Object를 이미지 내 임의 위치에서 찾기 때문에 Localization에 비해 상대적으로 여러 문제를 해결해야 함

### Object Detection의 Issues

- Classification과 Regression을 동시에 수행 : Total Loss의 설정 및 최적화
- 다양한 크기와 유형의 Object 존재 : 크기, 색상 등 특징이 서로 다른 Object가 섞여있는 이미지에서 Detect해야 됨
- Detecting Speed : 실시간 영상 기반에서 Detect해야 하는 요구사항 증대
- 명확하지 않은 이미지 : 이미지의 많은 부분이 배경으로 구성되고 Object의 비중이 낮은 경우
- Dateset의 부족 : Annotation을 만들어야 하므로 Training Dataset를 생성하기 어렵고, Training 가능한 Dataset의 부족

### Object Detection의 주요 구성요소

![Untitled 2](https://user-images.githubusercontent.com/54128055/140332767-3ba9012a-0ea9-46c0-b20a-b1c31d7801f8.png)

**Region Proposal** : Object가 위치할 영역을 추정 (ex. 뒤에서 다룰 Selective Search, RPN)

**Deep Learning Network 구성** : Feature Extraction 을 통해 이미지의 일부를 추출해 추상적 이미지를 가진 Feature Map을 도출하고 FPN 과정을 거쳐 Feature를 식별한 후 Classification 및 Bounding Box Regression 수행

![Untitled 3](https://user-images.githubusercontent.com/54128055/140332808-809a1de0-2ce1-450c-b815-d535bea46f55.png)

- Feature Extraction & Feature Pyramid Network
    
    **Feature Extraction**
    
    ![Untitled 4](https://user-images.githubusercontent.com/54128055/140332834-4cc8ddc1-7aba-4c7f-ae1a-b84e5c39d3be.png)
    
    CNN 모델은 다음과 같이 Feature Extraction과 Classification으로 구성된다
    
    Feature Extraction : Input Date의 고유한 특징을 찾는 단계
    
    - Convolution Matrix를 이용해서 원하는 특징(Invariance)을 두드러지게 함
    - Kernel과 Pooling(Sub-Sampling) 방법으로 이뤄짐
    - ex. 학습된 Kernel을 통해 Edge detection을 수행할 수 있다
    
    ![Untitled 5](https://user-images.githubusercontent.com/54128055/140332881-4d5a4a42-9999-4a80-8e37-9ba8e9fe7f09.png)
    
    - Kernel은 parameter(weight)이고, Pooling은 아니다
    - Feature Extraction의 결과 값은 Activation Func. 을 통해 fully connected Neural Net 에 입력된다
    
    ( [https://blog.naver.com/cjh226/220809060935](https://blog.naver.com/cjh226/220809060935) )
    
    **FPN (Feature Pyramid Network)**
    
    - 다양한 크기의 Object를 Detect하기 위해 이미지 자체의 크기를 Resizing > 메모리 및 시간 측면에서 비효율적 (ex. 뒤에서 다룰 Sliding Window, Image Scaling or FPH 등)
    - 추후에 다룰 예정이므로 현재 Chapter에서는 skip
        
        참고 링크 : ( [https://eehoeskrap.tistory.com/300](https://eehoeskrap.tistory.com/300) )
        

### Object Localization의 개요

**step**

1. Input : 원본 Image와 Annotation (Object의 Bounding Box 좌표) 입력
2. Feature Extraction을 통해 이미지를 학습해서 Classification을 수행
3. 동시에 Bounding Box Regression을 통해 Object를 Detect

![Untitled 6](https://user-images.githubusercontent.com/54128055/140332926-d1d86f26-1d61-4868-8a51-3c912e7c0898.png)

![Untitled 7](https://user-images.githubusercontent.com/54128055/140333057-5f775186-c90d-440b-8f9f-5d8f6a83c1f7.png)

![Untitled 8](https://user-images.githubusercontent.com/54128055/140333092-000068fb-cfca-4806-8975-7feef91a02ff.png)

### Challenge on Object Detection

![Untitled 9](https://user-images.githubusercontent.com/54128055/140333132-51d4ea68-b760-4244-b815-0e0a6d47bf8e.png)

다양한 Object가 존재하는 이미지의 어느 위치에서 Object를 Detect할 것인가?

- 이미지 내 Object가 많으면 Feature Map에 여러 요소가 포함됨
    - Localization과 달리 Bounding Box만 모델에 넣어서는 Inference가 어려움
    - ex. 사람의 경우 Feature 가 유사한데 Feature Map에 Feature가 다수 위치할 경우 Bounding Box가 Feature를 제대로 인식하지 못하는 문제 발생
    

**Object가 있을만한 위치를 예측하고 (Region Proposal), Object Detection (Network) 을 수행**

### Sliding Window

- Window가 왼쪽 상단부터 오른쪽 하단으로 이동하면서 Object Detection 수행
    - Opt1. 다양한 형태의 Window를 각각 sliding
    - Opt2. Window Scale은 고정하고, 다양한 Scale의 이미지에 Window sliding 적용
    
    ![Untitled 10](https://user-images.githubusercontent.com/54128055/140333194-faa5ba85-1605-4b34-9026-819e6d4a8a2e.png)
    
    ![Untitled 11](https://user-images.githubusercontent.com/54128055/140333233-c29ea785-a8cb-4845-8a5f-ba4aa92eacc9.png)
    
    - 이미지가 Window보다 크면 Target Object를 Detect하지 못하거나, 이미지가 작아질 경우 Target Object 외 다른 Object가 포함되어 성능이 떨어짐
- Object Detection의 초기 기법으로 활용
    - Object가 없는 영역도 sliding하기 때문에 수행 시간이 오래 걸리고 검출 성능이 낮음
- Region Proposal 기법의 등장으로 활용도는 떨어졌지만, Object Detection 발전을 위한 기술적 토대 제공

### Region Proposal - Selective Search

- 이미지 전체에 Feature Extraction 적용
- Color, Texture, Size, Shape 등에 따라 유사한 Region을 계층적으로 Grouping해서 계산
    1. Selective Search 알고리즘 첫 단계에서는 pixel Intensity에 기반한 **graph-based segment 기법**에 따라 Over Segmentation 수행
    2. 개별 Segment된 모든 부분들을 Bounding Box로 만들어서 Region Proposal 리스트에 추가
    3. Color, Texture, Size, Shape 등에 따라 유사도가 높은 Segment들을 Grouping 후 Region Proposal 리스트에 추가
    4. 2, 3번 과정 반복

![Untitled 12](https://user-images.githubusercontent.com/54128055/140333288-c8497128-1bd3-4327-a13f-d9e4470e92a5.png)

**빠른 Detection과 높은 Recall 예측 성능을 동시에 만족하는 알고리즘**
