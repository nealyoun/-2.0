# Binary Classification (Chapter 2.5~)

Assign: TAEHO KIM
Attachment: https://www.notion.so/80ca5b1901f848ae9772cb7d1db57260
Due Date: August 26, 2021
Status: Next Up

**2.5 확률분포의 추정과 교차 엔트로피**

정보 엔트로피 : 하나의 확률 분포가 갖는 불확실성 혹은 정보량을 정량적으로 계산

교차 엔트로피 : 두 가지 확률 분포가 얼마나 비슷한지 수치로 표현

정보 엔트로피 : 정보량의 기댓값

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled.png)

확률  (사건)를 갖는 항의 정보량  (확률값)으로 표현할 수 있다

(다시 말해, 각 사건의 발생 확률에 따라 정보량의 가중평균을 구한다)

교차 엔트로피 : 정보량을 제공하는 확률 분포와 가중평균 계산에 사용되는 확률 분포를 서로 다르게 설정한 채 정보량의 기댓값을 구하는 방식

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%201.png)

q1 ... qi **의 확률 분포 Q에 따른 정보량을 갖는 사건이 p**1 ... pi **의 확률 분포 P에 따라 일어날 때 정보량의 가중평균으로 정의한다**

- **교차 엔트로피의 특징**
1. 교차 엔트로피는 두 확률 분포가 같은 내용을 가질 때 해당 확률 분포의 정보 엔트로피 값과 같아진다
2. 언제나 H(P,Q)≥H(P,P) 가 성립한다
3. 위 특징으로 인해 두 확률 분포가 닮아갈수록 값이 작아지기 때문에 두 확률 분포가 서로 얼마나 다른 지 나타내는 정량적 지표 역할을 한다

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%202.png)

철수의 추정에 대한 교차 엔트로피가 깡통로봇 보다 정보 엔트로피 값에 가깝기 때문에 철수의 추정이 더 정확하다

추정이 정확해질 수록 교차 엔트로피 값은 정보 엔트로피 값에 수렴한다

**2.6 딥러닝 학습에서의 교차 엔트로피**

- **학습 과정에서의 문제**
1. 확률분포 Q를 정확하게 알지 못하는 상황이라서 Q와 P의 교차 엔트로피를 계산할 수 없다
2. 학습에 이용되는 데이터들은 각각 다른 입력과 그에 따른 출력이라서 확률분포 Q는 고정된 확률 분포가 아니라 입력에 따라 그때 그때 달라지는 조건부 확률 분포이다

- **해결책**

모범답안으로 주어지는 레이블 정보를 입력에 따른 조건부 확률 분포로 P를 재해석하고 이를 이용해 교차 엔트로피를 계산한다 ( ex. 그림 인식 문제에서 주어진 그림이 고양이라면, 확률 값 1 부여 > 레이블 정보를 추정 확률 분포가 접근해가야 할 학습 목표, 입력에 대한 출력의 조건부 확률 분포로 간주해서 교차 엔트로피 계산)

입력 데이터와 입력에 대한 출력값의 조건부 확률분포 사이에는 **어떤 상관관계 패턴**이 숨어 있을 것이고, 학습을 반복하면 이 패턴이 신경망에 반영되면서 각 입력에 알맞은 조건부 확률분포를 적절하게 만들어 낼 것으로 기대 > 이후에는 새로운 입력에 대한 성능도 기대

- **이진 판단 문제에서는 답 자체보다 답이 나오는 확률 분포가 핵심**

**2.7 시그모이드 교차 엔트로피와 편미분**

교차 엔트로피 (P: 정답 레이블 /  Q: 신경망의 출력 값)

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%203.png)

에서 학습 데이터의 정답으로  z 가 주어졌을 때 , 신경망은 로짓값 x  를 출력했다.

z 는 대부분의 데이터 셋에서 0 or 1로 주어지기 때문에 Pt = z, Pf = 1-z  로 정의할 수 있다

Q_F=1-Q_T=〖1- σ(x)〗_T이므로, 다음과 같이 재정의된다

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%204.png)

위 식을 정리하면,

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%205.png)

이와 같이 교차 엔트로피를 구할 수 있다

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%206.png)

각 P 확률에 대해 다시 정리하면,

z = 0 일 때, H=x+log⁡(1+e^(-x))

z = 1일 때, H=log⁡(1+e^(-x))

log⁡(1+e^(-x) )= -log⁡〖σ(x)〗이므로, 다시 정리하면 다음과 같다

z = 0 일 때, H=x-log⁡σ(x)

z = 1 일 때, H= -log⁡〖σ(x)〗

- **교차 엔트로피 편미분**

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%207.png)

이므로

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%208.png)

정리하면

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%209.png)

**2.8 계산 값 폭주 문제와 시그모이드 관련 함수의 안전한 계산법**

수학적 정의에 따라 코드를 작성해 교차 엔트로피 값을 계산하다 보면 값이 발산하는 오류가 종종 발생한다

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%2010.png)

시그모이드 함수는 위와 같고,  e^(-x)는 다음 그래프와 같은 형태를 갖는다

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%2011.png)

따라서 x 값이 음수일 때, σ(x)=1/Inf 의 형태를 띄게 된다

프로그램 내에서는 교차 엔트로피 값이 NaN으로 지정되고 이후 처리에 문제가 발생한다

- **해결책**

값이 음수일 때, 교차 엔트로피 식을  로 재정의해서 사용한다

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%2012.png)

단, 딥러닝 알고리즘 내에서 조건문을 이용해 처리할 경우, 연산의 효율을 크게 떨어뜨린다

![Untitled](Binary%20Classification%20(Chapter%202%205~)%203823cef490c64429a51baedcc3dea6aa/Untitled%2013.png)

위 정의식을 딥러닝 알고리즘에 사용한다