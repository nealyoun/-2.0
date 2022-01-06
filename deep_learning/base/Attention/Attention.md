# Attention

Assign: Anonymous
Due Date: January 6, 2022
Status: Completed

### Sequence-to-Sequence (seq2seq)

입력된 sequence 로부터 다른 도메인의 sequence 를 출력하는 다양한 분야에서 사용되는 모델

- Encoder-Decoder Model
- e.g. 챗봇, 기계 번역(Machine Translate)

**Machine Translate structure**

- encoder, decoder module (RNN architectur) 로 구성
- encoder : 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤, 모든 단어 정보들을 압축한 벡터 생성 (context vector)
- decoder : context vector 를 encoder 로부터 전달받아 번역된 단어를 한 개씩 순차적으로 출력

![Untitled](Attention%209dd9aa92d1a04a178be0304d4781273d/Untitled.png)

**process**

- 입력 문장에 대한 tokenized word 각각을 encoder 의 RNN 셀의 각 시점에 입력
- encoder RNN 셀의 마지막 시점의 hidden state (context vector) 를 decoder (RNN Language Model) RNN 셀로 넘겨줌
- context vector 는 decoder RNN 셀의 첫 hidden state 에 사용

![Untitled](Attention%209dd9aa92d1a04a178be0304d4781273d/Untitled%201.png)

참고!

- <s> 는 <sos>, <bos>, <Go> 로 사용하기도 함
- Affine Layer 는 hidden state 를 Input 으로 받아 분류 갯수로 출력해주는 Feedforward Network

***Sequence 길이와 순서를 자유롭게 해서 서로 다른 도메인으로 출력을 변환하는 task 에 이상적인 모델!***

**seq2seq 모델의 한계**

- 하나의 고정된 크기의 벡터를 생성하고 모든 정보를 압축하기 때문에 정보의 손실 발생
    - Input data 가 길어지면 성능이 크게 저하됨
    - Decoder 의 Input 으로 Encoder 의 마지막 RNN 셀의 hidden state 만 사용
- RNN 의 고질적인 문제 Vanishing Gradient Problem 존재

---

# Attention Mechanism

Decoder 에서 출력을 생성하는 매 time step 마다 Encoder 의 hidden state 를 다시 한 번 참고

참고하는 비율을 해당 time step 에서 생성하는 출력과 연관이 있는 부분을 Attention score 를 이용해서 판단

**“한 time step 에서 RNN 셀의 hidden state 에는 직전 시점에 입력된 단어에 대한 정보가 많을 것이다”**

e.g. dot-product, scaled dot-product, general, concat(Bahdanau), location-base, self etc.

![Untitled](Attention%209dd9aa92d1a04a178be0304d4781273d/Untitled%202.png)

### **Attention Function**

***Attention(Q, K, V) = Attention Value*** (Dictionary(Key-Value) 자료형)

- Q (Query) : t 시점의 Decoder 셀의 hidden state (s)
- K (Keys) : 모든 시점의 Encoder 셀의 hidden state (h)
- V (Values) : 모든 시점의 Encoder 셀의 hidden state (a)

![Untitled](Attention%209dd9aa92d1a04a178be0304d4781273d/Untitled%203.png)

### Dot-Product Attention

**Attention score**

현재 Decoder 의 출력 생성에 어느 Encoder 의 hidden state 가 얼만큼 필요한지 나타내는 지표

- Decoder 의 hidden state 와 Encoder 의 모든 RNN 셀의 hidden state 의 내적 수행
- t 시점 Decoder 의 hidden state : $s_t$
- Encoder 의 hidden state : $h_i$ 일 때,
- $Attention\ score(s_t, h_i) = s_t^Th_i$
    
    ![Untitled](Attention%209dd9aa92d1a04a178be0304d4781273d/Untitled%204.png)
    
- 내적 결과 값에 softmax 함수를 적용해서 Attention Distribution 생성
- $e^t=[s^T_th_1,...,s^T_th_N]$ 에 대해, $\alpha^t = softmax(e^t)$

![Untitled](Attention%209dd9aa92d1a04a178be0304d4781273d/Untitled%205.png)

- Attention Distribution 은 **각 Encoder hidden state 의 중요도** 의미

***context vector = Attention Value*** $a_t = \sum^N_{i=1}\alpha^t_ih_i$

![Untitled](Attention%209dd9aa92d1a04a178be0304d4781273d/Untitled%206.png)

**Attention 에서 context vector 는 Encoder 의 문맥을 포함하고 있음**

(seq2seq 의 context vector 는 Encoder 의 마지막 hidden state)

**output of Decoder**

- Decoder 의 hidden state 와 Attention Value 를 concatenate 한 뒤,
    
    Dense Layer 에 연결하고 tanh func. 적용
    
    ![Untitled](Attention%209dd9aa92d1a04a178be0304d4781273d/Untitled%207.png)
    
    - $\tilde s_t = tanh(W_c[a_t;s_t]+b_c)$
- $\tilde s_t$ 를 출력층의 Input 으로 사용해서 예측 벡터 생성
    - $\hat y_t = softmax(W_y\tilde s_t+b_y)$

[https://www.youtube.com/watch?v=WsQLdu2JMgI&t=189](https://www.youtube.com/watch?v=WsQLdu2JMgI&t=189)

[밑바닥부터 이해하는 어텐션 메커니즘(Attention Mechanism) (tistory.com)](https://glee1228.tistory.com/3)