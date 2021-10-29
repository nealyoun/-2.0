# Transfer Learning & Adam Algorithm(Chapter 6)

Assign: Anonymous
Due Date: November 4, 2021
Reference: https://hiddenbeginner.github.io/deeplearning/2019/09/22/optimization_algorithms_in_deep_learning.html
Status: Completed

## 6.1 오피스31 데이터셋과 다차원 분류

- 오피스31 데이터셋 : 컴퓨터 비전 분야에서 전이학습 연구용으로 구축된 표준 벤치마크 데이터셋
    - 사무용품 이미지 4,652장으로 구성
    - 이중 레이블링으로 구성
        - 수집방법 3가지 도메인 : amazon(웹), dSLR(디지털 카메라 촬영), webcam(웹캠 촬영)
        - 31가지 품목 : 배낭, 자전거, 파일캐비닛, 헤드폰, 키보드 등

![Untitled](https://user-images.githubusercontent.com/54128055/139453442-6584c2f6-9487-4782-892a-0ce92327eb80.png)

### 전이학습 (Transfer Learning)

한 도메인에서 학습시킨 결과를 다른 도메인에 활용하여 학습 효과를 높이는 학습 기법

- 전이학습을 하는 이유
    - 레이블링 작업은 많은 경우 데이터 수집 이상의 시간과 인력 소모
    - 레이블링 없이 신경망을 학습시킬 수 있는 방법이 없음

**if)** amazon 도메인 데이터들은 품목 레이블링이 완료 되었고, 이를 이용해 준수한 모델을 만들었다.

dSLR 도메인 데이터들은 레이블링이 되어있지 않다면?

 > amazon 도메인 모델의 학습 정보를 활용해서 dSLR 도메인에서 더 나은 성능을 얻을 수 있다.

**if)** 2차원 분류를 (도메인, 품목) 순서쌍 레이블링을 통해 1차원 분류로 차원을 축소하면?

3 가지 도메인 X 31가지 품목 = 93가지 순서쌍에 대한 Multiclass 분류 문제가 된다

출력이 과도하게 커지고, 도메인과 품목 특성을 따로 포착하기 어려워 학습 성과가 떨어질 우려가 있다

## 6.2 딥러닝에서 복합 출력의 학습법

딥러닝에서 퍼셉트론은 학습 과정에 따라 역할이 달라진다. 퍼셉트론의 구조가 아니라 퍼셉트론의 파라미터(weight, bias 등)값 구성이 변한다.

- **동일한 구조의 신경망을 다양한 용도에 이용하려면, 후처리 과정에서의 처리 방법 변경**
    
    **후처리 과정** : 순전파를 통해 얻은 출력으로부터 손실함수(Cost func.)를 계산. 이 후, 출력 각 성분의 손실 기울기를 계산해서 신경망 역전파 처리 과정에 시동을 걸어주는 과정
    
    ( output layer의 처리)
    

### 복합 출력의 처리도 후처리 과정의 변화를 통해 해결

- 신경망 회로에서 알맞은 크기의 출력 벡터 생성
- ex) 오피스31 데이터셋에서는 Input vector의 크기는 이미지의 픽셀 수이고, output vector의 크기는 3(도메인) + 31(품목) = 34 로 지정하면 된다

**if)** 복합출력 대신, 출력별로 별도의 신경망을 구성하면?

출력 계층만 고려하면 신경망을 분리해도 무관

은닉 계층의 퍼셉트론은 모든 출력 계층 퍼셉트론에 영향을 주고 역전파를 받기 때문에 신경망을 분리하면 큰 차이가 생긴다

+ 출력의 공통특성이 있을 수 있는 것을 고려하면 복합 출력으로 처리하는 것이 바람직

### 손실함수 및 역전파 처리

레이블 정보 $y_1$, $y_2$ 와 복합 출력의 두 성분 $output_1$, $output_2$ 를 비교해서 $L_1$, $L_2$ 계산

$$⁍$$

전체 손실 함숫값은 각 손실값의 합으로 정의, $L$를 최소화하는 것은 $L_1$, $L_2$를 함께 줄이는 과정

- $Loss$ 의 편미분
    - $L_1$, $L_2$ 는 서로에게 상수로 간주됨
    - $output_1$의 손실 기울기는 $y_1$ 과 후처리 과정을 통해 $L_1$ 을 이용해서 계산 ( $output_2$ 동일)
    - 즉, 기존의 방법 그대로 손실 기울기를 각각 구하면 된다
    - 역전파 과정에서 은닉 계층 퍼셉트론 단계에서 서로 연관되어 반영됨
    
- 역전파 처리
    - 출력이 복잡하면 출력 정보를 세부적인 요소로 나누어볼 수 있다
    - 회귀 분석, 이진 판단, 선택 분류 중 한 가지를 도출할 수 있고, 손실 기울기를 계산
    - 손실 기울기들을 복합 출력의 형태에 맞춰 모아 신경망 출력 계층에 전달 후 역전파 수행
    

## 6.3  복합 출력을 위한 MlpModel 클래스와 Dataset 클래스의 역할

(코드에 대한 내용이라서 Skip)

## 6.4 아담 알고리즘 (Adaptive Moments Algorithm)

**파라미터에 적용되는 학습률(Learning Rate)을 파라미터 별로 동적으로 조절해 경사하강법(Gradient descent)의 동작을 보완하고 학습 품질을 높이는 방법**

- 모멘텀(moments) : 최근 파라미터값의 변화 추세를 나타내는 정보
    
    Momentum(관성) 방법은 미끄럼틀을 타고 평평한 지점까지 내려왔을때, 가속도에 의해 바로 멈추지 않듯이 gradient가 0에 가까워 졌을 때 바로 멈추지 않는다. Momentum은 현재 파라미터를 업데이트할 때, 이전 gradient들도 계산에 포함해 문제를 해결한다
    
    ![Untitled 1](https://user-images.githubusercontent.com/54128055/139453555-a685535f-6eab-4e23-a31f-fb1a890c600d.png)
    
    Momentum은 위의 식과 같이 이전 gradient의 영향력을 일정 비율 감소시켜 반영한다
    
    ![Untitled 2](https://user-images.githubusercontent.com/54128055/139453594-83a53f04-2fbd-4a72-b580-630b5eaffdee.png)
    
    전체 식으로 표현하면 위의 식과 같다. 보통 $gamma$ 값은 0.9를 사용한다
    
    **Momentum은 gradient descent를 사용할 때, Local Minimum에 빠질 위험을 줄여준다**
    
    ([https://hiddenbeginner.github.io/deeplearning/2019/09/22/optimization_algorithms_in_deep_learning.html](https://hiddenbeginner.github.io/deeplearning/2019/09/22/optimization_algorithms_in_deep_learning.html))
    

- 주의할 사항
    - 아담 알고리즘은 모멘텀 정보와 함께 2차 모멘텀 정보까지 활용
    - 모멘텀 정보와 2차 모멘텀 정보는 개별 파라미터 수준에서 따로 계산되고 관리되기 때문에 파라미터 관리에 필요한 메모리 소비량이 3배로 늘어남 (+ 계산량도 증가)
    - Learning Rate를 파라미터별로 보정해 Gradient descent보다 품질 저하가 적다
    - 다만, 적절한 Learning Rate를 찾는 노력이 완전히 없어지지 않는다
