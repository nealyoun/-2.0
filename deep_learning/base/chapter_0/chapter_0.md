# chapter_0

----

## A.I. ML & Deep Learning

가장 큰 개념은 인공지능. 즉 ML 과 Deep Learning은 인공지능의 부분 집합. 여기서 deep learning은 machine learning의 부분집합.

ML과 Deep Learning은 인공지능의 "지식 기반 접근"과 "데이터 기반 접근" 중 당연히 데이터 기반 접근 방식이다. (Machine Learning: 프로그램이 직접 데이터를 분석하여 데이터에 내재되어 있는 규칙이나 패턴을 포착하여 문제를 해결)

## 신경세포, Neuron

- 인공신경망(Artificial Neural Network)은 동물의 신경세포의 뉴런을 흉내내어 고안한 퍼셉트론(perceptron) 단위로 구성

<img width="500" alt="Screen_Shot_2021-07-08_at_2 19 42_PM" src="https://user-images.githubusercontent.com/54128055/124869553-570aa480-dffc-11eb-8e03-137ac1bd5bcc.png">


## 인공 신경망의 기본 유닌, Perceptron

- input x1, ... x(n)은 다른 뉴런들로부터 전달되는 전기 신호에 해당
- weight w1, ...w(n)은 뉴런 견결 부위에 형성된 시냅스의 발달 정도에 해당
- x0 와 w0 는 흔히 bias라 불리는 요소
- x(i)w(i) 값들을 합산하는 &Sigma; 처리는 각 전기 신호들이 뉴런 세포체 안에서 합해지는 과정
- f() 는 Activation Function(활성화 함수)로 비선형 함수에 해당

<img width="500" alt="Screen_Shot_2021-07-08_at_2 23 49_PM" src="https://user-images.githubusercontent.com/54128055/124869564-5b36c200-dffc-11eb-8803-3463eb99e144.png">

뉴런의 세포체가 단순히 전기 신호들의 합을 나름의 처리를 통해 출력으로 삼듯이, perceptron의 activation function 또한 &Sigma;로부터 구한 값에 비선형 함수를 적용하여 perceptron 단위에서 최종 출력을 결정한다.

이처럼 구성된 퍼셉트론을 통해 다양한 구조의 신경망을 구성하고 가충치와 편향값을 알맞게 조정하며 문제를 해결하는 것이다.

해당 책에서 다루는 수학 분야는 선형대수학, 미분적분학, 확률통계론으로 Deep Learning 알고리즘을 보다 더  깊게 이해하기 위한 base 지식이라 생각된다.
