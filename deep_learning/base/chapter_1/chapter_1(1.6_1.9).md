# chapter_1

## 1.7 편미분과 손실 기울기의 계산

**딥러닝의 신경망들은 편미분이 가능한 함수만 손실 함수 계산 과정에 이용되도록 설계 되었다.**

**즉, 모든 경우에서 역전파 처리가 가능하다.**

편미분의 chain rule로 인해 Back Propagation 과정서의 cost function에 직간접적으로 영향을 미친 모든 성분에 대하여 손실 함수의 기울기를 계산

![Untitled 0](https://user-images.githubusercontent.com/54128055/126657097-7803d25d-48a1-487d-85c2-3fe4fbfa92ec.png)

Back Propagation: foward propagation의 역순으로 진행 

![Untitled 1](https://user-images.githubusercontent.com/54128055/126657099-e95fc628-0a1a-4ba6-aeb4-ccce32c50d22.png)

### 입력이 다수인 경우의 처리

y = f(x1, x2, x3, ... xn) 의 경우

![Untitled 2](https://user-images.githubusercontent.com/54128055/126657104-d1b1950b-b1f2-4448-8c67-728becca9946.png)

- 각 입력 성분 별로 따로 처리
- 파라미터에 해당되는 입력 성분의 경우 learning rate를 곱하여 값을 수정

![Untitled 3](https://user-images.githubusercontent.com/54128055/126657106-a7249dda-320a-42ff-850e-6d6edb84133a.png)

![Untitled 4](https://user-images.githubusercontent.com/54128055/126657111-2537d743-4252-40bd-acb6-78291d21eeec.png)

### 출력이 중복으로 이용되는 경우

즉, y가 y = y1 = y2 = y3 ... = yn

출력의 손실기울기들을 합산하고 해당 값을 이용해 입력에 대한 손실 기울기를 계산

![Untitled 5](https://user-images.githubusercontent.com/54128055/126657116-ae180eae-814a-4f60-af2f-cfcd12ac3d82.png)

### 학습률: &alpha;

가중치 혹은 편향 즉 파라미터 성분은 해당 성분의 손실 기울기에 학습률을 곱한 값을 빼줌으로써 해당 값을 업데이트한다.

![Untitled 3](https://user-images.githubusercontent.com/54128055/126657106-a7249dda-320a-42ff-850e-6d6edb84133a.png)

위 식의 &alpha; 는 학습 속도를 조절하는 hyperparameter로서 너무 크거나 작지 않게 설정하는 것이 중요

신경망 학습 결과가 만족스럽지 않을 경우 가장 먼저 조정해야하는 hyperparameter이다.

## 1.7 Hyperparameter

- learning rate, epoch, mini-batch size와 같이 딥러닝 모델의 구조 혹은 학습 과정에 영향을 미치는 상수값.
- 딥러닝 알고리즘이 실행되는 동안은 값이 변하지 않는 상수
- 하지만 만족스러운 학습 결과를 얻기 위해 개발자가 끊임없이 값을 조정해가며 실험해야한다
- **parameter는 학습을 통해 적당한 값을 향해가지만, hyperparameter는 학습 진행을 위해 개발자가 미리 정해주는 값**
