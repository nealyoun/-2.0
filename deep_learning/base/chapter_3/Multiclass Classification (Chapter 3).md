# Multiclass Classification (Chapter 3)

Assign: Anonymous
Due Date: September 13, 2021
Reference: https://m.blog.naver.com/dlgjsdn999/221960033232
Status: Completed

## 3.2 선택 분류 문제의 신경망 처리

- 선택 분류 문제는 몇 가지 정해진 후보 가운데 하나를 골라 답하는 문제 (Multi Class)
- 선택 분류 신경망은 이진 판단처럼 각 후보 항목의 로짓값을 추정하도록 구성되어 있음
    - $logit$ 은 상대적인 가능성을 $Log$ 를 이용해서 나타낸 값
- 퍼셉트론 하나가 후보 하나에 대한 로짓값 출력 → 후보 수만큼의 퍼셉트론 필요

*선택 분류에서의 로짓값의 표현 대상은 **각 후보 항목을 답으로 추정할 확률***

if) A 항목 $logit$ $3$ , B 항목 $logit$ $1$ 이라면

- A를 답으로 추정할 확률이 B보다 $e^{(3-1)} = e^2 = 7.39$ 배 높음

*하나의 로짓값 만으로는 의미가 없고 **로짓값 사이의 차이가 중요!***

### Multiclass Classification 에서 Cross Entropy 계산

1. 복수의 Class 에 대한 $logit$ 벡터를 확률 분포 벡터로 변환 → SoftMax Function
2. 추정한 확률분포와 정답 확률 분포 사이의 Cross Entropy 계산

## 3.3 SoftMax Function

- SoftMax 는 $logit$ 벡터를 확률 분포 벡터로 변환해주는 비선형 함수
    - SoftMax 의 출력은 각 Class 가 정답일 확률로 구성된 벡터
    - 벡터의 각 성분은 $[0, 1]$
    - 각 클래스가 정답일 확률의 합(i.e. 전체 벡터 성분의 합)은 1

- 굳이 확률 분포를 따질 필요가 없다!!!
    - 각 클래스에 해당할 확률을 계산하지 않고, 벡터에서 $logit$ 이 최대인 항 선택
        - 각 항목에 대한 $logit$ 이 크면 확률값 $e^{logit}$ 도 크기 때문
        - Loss Function 도 확률을 계산하지 않고 $logit$ 으로 계산 가능 (순전파는 $logit$ 으로만 처리 가능)
    - $logit$ 은 $[-inf, inf]$ 인데 확률은 $[0, 1]$ **(로짓을 확률 대신 쓸 수 있는 이유 ??)**

![Untitled](Multiclass%20Classification%20(Chapter%203)%20b4b1ed298b2946ecb1e5b3fec5caa156/Untitled.png)

**SoftMax Function 을 통해 확률분포 벡터를 구하는 경우**

 

1. 확률 분포를 눈으로 확인하고 싶을 때
2. Backpropagation 에서 Loss Function 의 편미분을 구할 때

### SoftMax Function

입력 $logit$ 벡터의 각 성분에 대해 전체 대비 비율을 통해 확률(상대적 가능성)로 변환

$$y_i = {e^{x_i}\over e^{x_1}+...+e^{x_n}}$$

**if)** $logit$ 벡터가 $(2.0, \;1.0, \;1.2, \;0.7)$ 이면,

$e^2 : e^1 : e^{1.2} : e^{0.7} = 7.39 : 2.72 : 3.32 : 2.01$

${e^{x_i}\over e^2 + e^1 + e^{1.2} + e^{0.7}}$ 이렇게 SoftMax 를 적용하면 $0.479 : 0.176 : 0.125 : 0.130$

**실제 소프트맥스 값을 계산할때는 $x_i$ 중 최대값 $x_k$ 를 이용해 변형식 이용**

- 분자와 분모를 동시에 $e^{x_k}$ 로 나눠서 ****정규화

$$y_i={e^{x_i-x_k} \over e^{x_1-x_k}+...+e^{x_n-x_k}}$$

- $logit$ 은 $[-inf, inf]$ 이기 때문에 $e^{x_i}$ 를 계산할 때 오버플로우 발생 우려
    - 모든 값의 범위가 $0 ≤ e^{x_i-x_k} ≤ 1$ 가 되기 때문에 오버플로우 방지
- 혹은, 모든 값이 아주 작은 값을 가지면 분모가 0에 너무 가까워져 불능 발생 우려
    - $e^{x_k-x_k}=e^0=1$ 항이 분모에 존재하기 때문에 불능 방지

## 3.4 SoftMax 의 편미분

- 벡터를 입력받아 벡터를 출력하기 때문에 벡터 성분 간 **편미분이 다대다 관계**여서 계산과정이 굉장히 복잡
- 역전파에서 편미분 대상은 Loss Function 인 Cross Entropy 여서 SoftMax 의 편미분을 사용하지 않음
    - SoftMax 에 대한 이해를 돕기 위해 소개

**편미분 계산 과정**

- 입력 벡터의 모든 $x_i$ 가 출력 벡터의 모든 $y_i$ 에 영향을 미치므로 모든 $(x_i, y_i)$ 쌍의 편미분 계산
- 모든 입출력 벡터의 모든 원소 쌍에 대한 ${\partial y_i \over \partial x_i}$ 값들로 구성된 2차원 행렬 구함 → 야코비 행렬
- 딥러닝 알고리즘은 미니배치 단위로 한꺼번에 여러 개의 데이터 처리 → 3차원 텐서 형태로 계산

$i \ne j$ 인 경우) $y_i={e^{x_j} \over e^{x_1}+...+e^{x_n}}$ 에서 $x_i$ 와 관련 있는 항은 분모에 있는 $e^{x_i}$ 밖에 없음

$${\partial y_i \over \partial x_i}=-{e^{x_j}(e^{x_1}+...+e^{x_n})^{'} \over (e^{x_1}+...+e^{x_n})^2}=-{e^{x_j}e^{x_i} \over (e^{x_1}+...+e^{x_n})^2}$$

$$\;=-{e^{x_i} \over e^{x_1}+...+e^{x_n}}{e^{x_j} \over e^{x_1}+...+e^{x_n}}= -y_iy_j$$

$i = j$ 인 경우) $(x_i, y_i)$ 쌍에 대한 편미분 계산

$${\partial y_i \over \partial x_i}= e^{x_i}{1 \over e^{x_1}+...+e^{x_n}}-e^{x_i}{(e^{x_1}+...+e^{x_n})^{'} \over (e^{x_1}+...+e^{x_n})^2}$$

$$={e^{x_i}(e^{x_1}+...+e^{x_n})-e^{x_i}(e^{x_1}+...+e^{x_n})^{'} \over (e^{x_1}+...+e^{x_n})^2}={e^{x_i}(e^{x_1}+...+e^{x_n})-e^{x_i}e^{x_i} \over (e^{x_1}+...+e^{x_n})^2}$$

$$={e^{x_i} \over e^{x_1}+...+e^{x_n}}(1-{e^{x_i} \over e^{x_1}+...+e^{x_n}})=y_i-y_iy_i=y_i-y_i^2$$

$i \ne j$ 일 때의 처리 결과에 $y_i$ 를 더해주면 $i = j$ 인 경우의 편미분 값이므로

***모든 경우에 대해 ${\partial y_i \over \partial x_i} = -y_iy_j$ 로 계산한 후, 각 $i$ 에 대해 ${\partial y_i \over \partial x_i} = -y_iy_i=-y_i^2$ 값에 $y_i$ 만 더해줘서 보정***

## 3.5 SoftMax Cross Entropy

### Cross Entropy

$logit$ 벡터 $a_1,...,a_n$ 과 정답 벡터 $y_1,...,y_n$ 이 주어졌을 때,

$P$ : 정답 레이블 / $Q$ : SoftMax Function 이 적용된 신경망의 출력

$$H(P,Q)=-\sum p_i \;log\;(q_i+\varepsilon)$$

- 정답 벡터는 One-Hot Encoding 으로 입력되어 정답만 1이고 나머지는 0인 형태
- $P = 0\;or\;1$ 이어서 $-inf \le log P \le 0$
- $0\le Q \le 1$ 이므로 $log$ 를 취하기 유리. 단, $log0$ 은 정의될 수 없기 때문에 아주 작은 양수를 더해줌

## 3.6 SoftMax Cross Entropy 의 편미분

$p_i = 0\;or\;1$, $q_i = {e^{x_i} \over e^{x_1}+...+e^{x_n}}$ 일 때, 

$${\partial H \over \partial x_i}=-{\partial \over \partial x_i}\sum^n_{k=0} p_k\;log\;q_k=-\sum^n_{k=0} p_k\; {\partial \over \partial x_i}\; log\;q_k = -\sum^n_{k=0}p_k{1 \over q_k}{\partial q_k \over \partial x_i}$$

에서 ${\partial q_k \over \partial x_i}$ 는 SoftMax 의 편미분이므로,

$${\partial H \over \partial x_i}= -{p_i \over q_i}{\partial q_i \over \partial x_i} -\sum^n_{k\ne i} {p_k \over q_k}{\partial q_k \over \partial x_i} = -{p_i(q_i-q_i^2) \over q_i}-\sum_{k\ne i}{p_k(-q_iq_k) \over q_k}$$

$$=p_iq_i-p_i+\sum_{k\ne i}p_kq_i=q_i\sum^n_{k=1}p_k - p_i$$

전체 확률값의 합은 1 이므로 $\sum^n_{k=1}p_k=1$

$${\partial H \over \partial x_i}=q_i-p_i$$

## 3.7 Sigmoid 와 SoftMax 의 관계

- Sigmoid 는 두 후보를 갖는 SoftMax 의 입력 벡터 $x_1,x_2$ 에서 $x_2 = 0$ 으로 설정해 입력 변수를 하나로 줄인 함수에 해당
    - $x_2$ 를 $0$ 으로 고정시키더라도 $logit$ 이기 때문에 계산에 문제가 없으며, 전체 확률의 합은 $1$ 이기 때문에 참일 확률 $P_T$ 만 알면 거짓일 확률도 알 수 있어서 $x_2$ 를 고정시켜도 문제 없음
- SoftMax 에서 총 합이 $1$ 임을 이용하여 $logit$ 항을 줄일 수 있음
    - 한 개의 $logit \to 0$  but. 코드가 복잡해짐
    - 입출력이 하나인 Sigmoid 를 이용할 뿐 Multiclass Classification 에서 $logit$ 항을 줄이는 방식은 거의 사용하지 않음
- Multiclass Classification 에서도 Sigmoid 를 사용해도 무관하나, 각 클래스에 대한 확률이 구해질 뿐 그 합이 $1$이라는 제약은 사라짐 (각 클래스에 속할 확률 ≠ 여러 클래스 중 한 클래스에 속할 확률)
but. 가장 큰 확률을 구해도 되고 학습이 성공적이면 특정 항목에 대한 값은 1 나머지는 0으로 수렴하기는 함