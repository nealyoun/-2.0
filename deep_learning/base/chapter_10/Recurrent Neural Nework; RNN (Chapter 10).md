# Recurrent Neural Nework; RNN (Chapter 10)

Assign: Anonymous
Due Date: November 21, 2021
Status: Completed

순환 신경망은 MLP의 은닉 계층 가운데 일부를 순환 계층(recurrent layer)으로 대체하여 만든 신경망
순환 계층은 시계열(time series data)를 시간대별로 반복 처리
→ 어떤 시간대에 출력된 내용을 다음 시간대에 다시 입력의 일부로 활용 
*(fully connected layer 대비 획기적으로 줄어든 규모의 파라미터를 반복활용 가능)*

## 10.1 시계열 데이터

- 시계열 데이터는 일정시간 간격으로 배치된 같은 데이터들의 열(sequence)
e.g. 주가 예측, 경제학 분야 등
→ 시계열 데이터를 통해 시간대를 넘나드는 다양한 패턴이 숨어있기 마련이라 내용을 분석하고 이해하고 그 이해를 바탕으로 데이터의 분포 특성과 유용한 패턴을 찾아 활용하기는 쉽지 않지만 RNN 으로 해결할 대안으로 각광받음
- 멀티미디어 데이터, 주식 시세 등 시계열 로그 기록 뿐만 아니라 말이나 글도 시계열 데이터가 될 수 있음
Language model, speech recognition, conversation model, image captioning 등

## 10.2 순환 계층과 순환 벡터의 활용

- 순환 신경망은 은닉 계층 안에 하나 이상의 순환 계층을 갖는 신경망 구조이며 h(책 A)는 퍼셉트론 하나가 아닌 여러 퍼셉트론으로 구성된 퍼셉트론 계층
- 순환 계층 h에서 나와 다시 h로 향하는 화살표는 h의 출력이 다시 h로 입력되는 것을 표현하는데
이를 순환 벡터(recurrent vector)라고 함
→ 순환 벡터는 동시점 입출력이 이루어지는 되먹임(feedback) 입력이 아니라 어떤 시간대 출력 후 다음 시간대의 입력으로 이용되는 지연 입력이다. (현재 state가 다음 state에 영향을 끼치는 구조)
- 입력 벡터는 시간 대에 따른 내용은 변할지언정 벡터의 형태와 벡터 각 성분의 역할은 같다.(계층 h의 반복 사용, 파라미터 값 동일) → 한 장치를 반복적으로 이용하면 학습 효과 유지 및 데이터 마다 시간대 길이가 달라질 수 있는 시계열 데이터셋을 유연하게 처리할 수 있다.

![Untitled](Recurrent%20Neural%20Nework;%20RNN%20(Chapter%2010)%206d0295b16f63450c951671683cc0d8fe/Untitled.png)

- 순환 계층 h 계산식은 아래와 같다.
(ht : new state / fw(활성화함수 with w) / ht-1 : old state / xt : input)

![Untitled](Recurrent%20Neural%20Nework;%20RNN%20(Chapter%2010)%206d0295b16f63450c951671683cc0d8fe/Untitled%201.png)

![Untitled](Recurrent%20Neural%20Nework;%20RNN%20(Chapter%2010)%206d0295b16f63450c951671683cc0d8fe/Untitled%202.png)

### * static RNN vs. dynamic RNN

- 실제 들어오는 데이터의 길이는 가변적이기 때문에 발생하는 처리 과정으로 static과 dynamic으로 나뉘게 되는데 아래 그림에서 보면 hello / hi / why 로 데이터 길이가 다르게 전달

![Untitled](Recurrent%20Neural%20Nework;%20RNN%20(Chapter%2010)%206d0295b16f63450c951671683cc0d8fe/Untitled%203.png)

- static RNN :
    
    ![Untitled](Recurrent%20Neural%20Nework;%20RNN%20(Chapter%2010)%206d0295b16f63450c951671683cc0d8fe/Untitled%204.png)
    
    - 문장의 길이(seq_len)가 고정 되있음 (패딩으로)
    → 이 경우 이러한 padding을 넣어도, 각 모델에 있는 weight에 의해서 어떠한 값이 나오게 됨
    but 나오는 값 때문에 우리의 loss함수가 헷갈려할 수 있어 결과가 좋지 않을 수도 있습니다.
    - 보통 별도의 padding이나 truncating이 필요

- dynamic RNN (deprecated) :
    
    ![Untitled](Recurrent%20Neural%20Nework;%20RNN%20(Chapter%2010)%206d0295b16f63450c951671683cc0d8fe/Untitled%205.png)
    
    - 문장의 길이(seq_len)만큼 recurrent가 일어남
    → 문자열 개수 만큼 값을 줘서 올바른 loss가 나오도록 유도
    - 별도로 각 문장마다 길이가 얼마인지를 명시해줘야함

## 10.3 순환 계층의 입출력 형태

- 순환 계층은 입력만 시계열(e.g. sentiment analysis) , 출력만 시계열, 입출력 모두 시계열로 이뤄지고 
입력만 데이터인 경우 시계열 데이터 분석이라고 하고 출력만 시계열인 경우 시계열 데이터 생성이라고 함
- 입출력이 모두 시계열 데이터인 응용에서 입출력 시간대 축이 서로 다른 의미라면 하나의 순환 계층에서 처리하기 곤란? 
—> 입력을 분석하는 순환 계층과 출력을 생성하는 순환 계층을 따로 만들어 연결해야 함 (인코더 - 디코더)단 Encode, Decode란 말부터 이해해보자.

![Untitled](Recurrent%20Neural%20Nework;%20RNN%20(Chapter%2010)%206d0295b16f63450c951671683cc0d8fe/Untitled%206.png)

먼저, **Encode란 말은 '암호(code)화 하는 행위'** 라고 할 수 있고, **Decode는 당연히 암호(code)를 푸는 행위**다.

그럼 RNN에서 Encode, Decode는 무슨 의미를 가질까? 위 그림에서 'I love you'의 입력은 RNN을 거치면서 암호화 된다. 이**혹은 thought vector)**라 한다. 그리고 **Decoder**는 Context vector를 보고 '나는 널 사랑해'의 형태로 **암호를 풀어야 하는 입장**이다.

이렇게 Encode-Decode 과정으로 문장을 번역했을 때 2가지 문제가 발생한다.

- 어찌됐든 1개의 벡터로 문장을 표현하기 때문에 **정보의 손실**이 발생한다.(CNN에서 convolution을 통해 down sampling 되는 것과 비슷한 느낌)
- LSTM에서 어느정도 해결했다곤 하지만, RNN의 고질적인 문제인 **vanishing gradient**는 여전히 존재한다.

위 같은 문제점 때문에, 기계 번역 분야에서 입력 문장이 길며 길 수록 품질이 떨어지는 문제가 나타났다. 이것을 보정하기 위해 제안된 기법이 바로 **Attention**

## 10.4 순환 계층을 위한 시계열 데이터의 표현

- 순환 계층이 처리할 시계열 데이터를 다루기 위해서 시간 축 하나를 더 가져야 하며 뿐만 아니라 시간 축에 대한 길이, 시간대 수가 데이터마다 달라질 수 있음 (미니배치에서도)
- 시계열 데이터를 전달할 때 텐서의 차원도 하나 늘어나야되며 데이터의 길이를 표현 가능해야 함
- 미니배치에서 시간대 수(데이터 길이) 를 나타내는 방법으로 데이터 길이 정보를 별도의 벡터로 담아 전달할 수 있음
- 앞서 입출력이 시계열이고 시간 축의 의미가 서로 다르다면 순환신경망으로만 처리가 어려움 (인코더 디코더에서 처리)

- 딥러닝에서 미니배치 단위로 처리가 이루어질 때 미니배치 안의 데이터들은 각각 길이가 다른 시계열 데이터들 일 수 있다.

![Untitled](Recurrent%20Neural%20Nework;%20RNN%20(Chapter%2010)%206d0295b16f63450c951671683cc0d8fe/Untitled%207.png)

- 경기 결과 하나를 저장하는 데 크기 4인 (경기일자, 상대국가, 득점, 실점) 벡터 공간이 필요하다.
(but 나라 별 치른 경기 수가 서로 다름)
- [7,4] 텐서 공간을 활용하는 것이 좋지만 팀 별로 달라지는 시간 축 크기를 늘려 [8,4]로 바꾸어 주어 총 [32,8,4] 형태의 공간으로 나타낼 수 있음 
(그러나 첫 번째 행의 나머지 세 열 원소는 낭비되는데 시계열 데이터는 다소의 메모리를 낭비를 감수해야 함)