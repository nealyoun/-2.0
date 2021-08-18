# chapter_2

# Classification (binary)

### Dataset

1. Pulsar Dataset
* dataset uploaded in kaggle & https://github.com/KONANtechnology/Academy.ALZZA/tree/master/data/chap02

1. Feature
    1. Mean of the integrated profile
    2. Standard deviation of the integrated profile
    3. Excess kurtosis of the integrated profile
    4. Skewness of the integrated profile
    5. Mean of the DM-SNR curve
    6. Standard deviation of the DM-SNR curve
    7. Excess kurtosis of the DM-SNR curve
    8. Skewness of the DM-SNR curve

- Binary Classification - 두 가지 중 하나로 답하는 문제

    but. 가중치와 편향을 이용하는 퍼셉트론의 선형 연산은 기본적으로 두 가지 값으로 결과를 제한할 수 없다.

- sol_1. 선형 연산 결과가 임계치를 넘는지에 따라 두 가지 값 중 하나를 출력 하는 방법

    but. 미분이 불가능하여 학습이 어렵다.

- sol_2. 신경망은 확률에 해당하는 값을 추정. 이 값이 1에 가까우면 True, 0에 가까우면 False

    but. 0과 1사이의 확률값으로 출력 범위를 제한하는 것 또한 불가능

**∴** 범위에 제한이 없는 실수 값을 생성하고 이를 확률값의 성질에 맞게 변환해주는 비선형 함수

→ **Sigmoid**

하지만 시그모이드 함수를 이용해 신경망 출력을 확률로 해석했지만 어떻게 학습?

딥러닝에서는 값이 0 이상이면서 확실해질수록 작아지는 성질이 있는 손실 함수를 정의해야함

그래서 나온게 Cross Entropy, 항상 양수이고 두 확률 분포가 비슷해질수록 값이 작아지는 성질 덕분에 Sigmoid 함수의 Cross Entropy값을 손실 함수로 정의하여 학습이 가능하게 됨

범위에 제한 없는 임의의 실수 값을 입력으로 받아 확률값의 범위에 해당하는 0과 1 사이의 값을 출력하는 함수(σ(x))

sigmoid 함수는 입력 x가 어떤 확률 값의 logit 표현이라고 간주

logit이란 실제 표현하려는 값을 로그값으로 대신 나타낸 것
