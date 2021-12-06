# LSTM & (GRU)

Assign: Anonymous
Due Date: December 2, 2021
Reference: https://wikidocs.net/book/2155
Status: Completed

**RNN issue:**

1. rnn 계층에서는 순환 벡터 혹은 그 기울기 정보에 정보의 소멸 및 폭주 현상 발생 (Vanishing & Exploding Gradient)
2. 정보의 장거리 전달의 어려움 (Long Term Dependencies)
    
    → 위 문제를 해결하기 위해 lstm 계층을 이용했으며 이를 **LSTM 신경망** 이라고 부름

## Vanishing & Exploding Gradient

점점 더 많은 계층을 갖는 신경망일수록 기울기의 소멸 및 폭주 문제 심각

- **Vanishing Gradient**: 역전파 과정에서 계층을 거칠수록 손실 기울깃값이 점점 작아져 초반 계층의 학습이 이루어지지 않는 경우 → 학습 진척이 어려움
- **Exploding Gradient**: 계층을 거칠수록 손실 기울기가 점차 커지면서 가중치들이 비정상적으로 크게 업데이트 되는 경우 → 한 번만 일어나도 학습된 파라미터값이 크게 훼손됨
    
    ![Untitled](https://user-images.githubusercontent.com/54128055/144828309-899e51b8-b90d-4da5-9062-4c02920c80df.png)
    
    - *chain rule을 통해 기울기를 구할 때 곱해지는 값(a ~ f)이 1보다 작으면 곱할수록 점점 작아지게 된다. 즉, 기울기는 매우 작은 값을 갖게되며 이에 학습률을 곱해 weight를 업데이트하게 되면 위처럼 0.3 → 0.29로 되는 것 처럼 weight는 거의 변하지 않는다. 이를 멈춤 상태라 하며 weight는 최적의 값에 도달하지 못한다.*
    - *chain rule에서 1보다 큰 값이 곱해지면서 weight 값이 발산하게된다. 이또한 weight는 최적의 값에 도달하지 못한다.*
- 해당 문제를 완화하는 다양한 방법론과 기법이 연구되면서 요즘은 손실기울기의 소멸 및 폭주 문제는 그다지 심각한 문제로 거론되진 않음
    - 해당 문제는 문제를 일으킬 소지가 있는 특정 데이터를 다른 데이터와 혼합해서 처리하는 mini-batch
    - 수렴하지 않는 활성화 함수 사용 (LeakyReLU)
    - Gradient Clipping (Exploding gradient를 완화)
    - Weight 초기화
    - Batch Normalization
- 시계열 데이터에는 **기억** 효과가 반영되며, vanishing or exploding gradient 문제로 인해 먼 시간대 사이의 패턴을 포착하기 어려움. 즉, **Long-Term Dependency (장기 의존성),** 시퀀스 길이가 길어질수록 과거의 정보를 전달하지 못하는 한계점 존재

### RNN's Vanishing & Exploding Gradient

이전 시간대에서 전달된 순환 벡터를 입력의 일부로 사용

다음 시간대로 전달할 순환 벡터 생성

- 각 시간대 데이터에 대해 동일한 내용의 가중치 행렬을 반복적으로 선형 연산에 이용
    - 가중치 행렬 내용은 학습 진행에 따라 지속적으로 수정되지만, 하나의 미니배치 데이터를 처리하는 동안은 가중치가 변함없이 일정학게 유지
- 따라서 동일한 내용의 가중치가 각 시간대에 대해 반복 이용되는 상황을 초래

### 순환 벡터 원소 값의 문제점

- 순환 벡터 원소에 곱해지는 가중치 행렬의 원소가 모든 시간대에 걸쳐 같은 위치의 원소
- 하나의 미니배치 데이터를 처리하는 동안 가중치 원소의 값은 일정
- 반복적으로 같은 값을 곱하는 문제
    - 절대값이 1보다 큰 값이 반복해 곱해지면 절대값이 무한히 커짐
    - 절대값이 1보다 작은 값이 반복해 곱해지면 절대값이 0으로 수렴

---

잠깐 알고 가자!

### Gate

- 게이트는 데이터의 흐름을 제어하는 역할
    - 즉, LSTM에서의 게이트는 ‘열기/닫기’ 뿐 아니라, 어느 정도 열지를 조절 하는 역할
- 열기/닫기의 정도를 0.0 ~ 1.0의 실수로 표현
- 해당 값이 다음으로 넘어가는 데이터의 양을 결정
- **‘게이트의 열림 정도’ 또한 데이터로부터 자동으로 학습**

## LSTM의 구조와 동작 방식

*** 해당 notation은 위키독스의 *"딥 러닝을 이용한 자연어 처리 입문"* 을 따랐음 *****

- RNN의 문제점을 해결하기 위해 셀 구조를 변형시킨 것이 LSTM
    
    ![Untitled 1](https://user-images.githubusercontent.com/54128055/144828313-64214e86-5c23-4bda-a1be-3f620193a5e4.png)
    
    ![Untitled 2](https://user-images.githubusercontent.com/54128055/144828314-442d5982-6e3d-4104-be2e-accc3f1854b2.png)
    
    - RNN의 hidden state에 cell-state를 추가한 구조
    - RNN hidden state: $h_t = tanh(W_{x}x_{t} + W_{h}h_{t-1} + b)$
- 계속 곱해서 문제가 발생하기 때문에 셀 구조를 더하는 걸로 바꾸며 이를 **cell state** 라고 한다
- cell state는 컨베이어 벨트에 비유되는데, 전체 chain을 따라 직진하며 약간의 작은 선형 상호작용이 있을 뿐이다

![Untitled 3](https://user-images.githubusercontent.com/54128055/144828315-95083698-ade6-4da6-b08a-b5df71341b06.png)

- LSTM hidden state:
    - **forget gate**
        
        ![Untitled 4](https://user-images.githubusercontent.com/54128055/144828317-a41d47ef-c7f1-421d-8319-c7be9c7c79db.png)
        
        기억을 삭제하기 위한 게이트. 현재 시점 $t$의 $x$ 값과 이전 시점 $t-1$의 hidden state가 시그모이드 함수를 지난다
        
        $f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)$
        
        $**f_t$는 0~1 사이의 값을 가지며, 0에 가까울수록 정보가 많이 삭제된 것이고 1에 가까울수록 정보를 온전히 기억한 것이다**
        
    
    - **input gate**
        
        ![Untitled 5](https://user-images.githubusercontent.com/54128055/144828319-a83bae17-edf8-4cc3-aa26-40777ac2e23f.png)
        
        input gate는 현재의 정보를 기억하기 위한 게이트. 현재 시점 $t$의 $x$값과 input gate로 이어지는 가중치 $W_{xi}$를 곱한 값과 이전 시점 $t-1$의 hidden state가 input gate로 이어지는 가중치 $W_{hi}$를 곱한 값을 더하여 시그모이드 함수를 지난다
        
        $i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$
        
        현재 시점 $t$의 $x$ 값과 input gate로 이어지는 $W_{xi}$를 곱한 값과 이전 시점 $t-1$의 hidden state가 input gate로 이어지는 가중치 $W_{hg}$를 곱한 값을 더하여 $tanh$ 함수를 지난다
        
        $g_t = tanh(W_{xg}x_t + W_{hg}h_{t-1} + b_g)$
        
        $**i_t$ 는 0~1 사이의 값, $g_t$는 -1~1 사이의 값 두 개가 나오게 된다. 이 두 값을 가지고 이번에 선택된 기억할 정보의 양을 정한다**
        
    - **cell state (long term, 장기 상태), cell memory**
        
        ![Untitled 6](https://user-images.githubusercontent.com/54128055/144828321-628a0eae-ef82-4bda-b89d-1df6fc1649f1.png)
        
        cell state를 LSTM에서는 장기 상태라고 부른다. 우선 forget gate에서 일부 기억을 잃은 상태이다
        
        input gate에서 구한 $i_{t}$, $g_{t}$, 이 두 값에 대해서 entrywise product를 진행한다 (같은 크기의 두 행렬이 있을때 같은 위치의 성분끼리 곱하는 것). 이 과정이 이번에 선택된 "기억할" 값이다
        
        input gate에서 선택된 기억을 forget gate의 결괏값과 합한 것이 현재 시점 $t$의 cell state이며, 이는 다음 $t+1$ 시점의 LSTM 셀로 넘겨진다
        
        **forget gate의 출력값 $f_{t}$가 0 이라면, 이전 시점의 cell state ($C_{t-1}$)은 현재 시점의 cell state 값을 결정하기 위한 영향력이 0이 된다.** 즉, 오직 input gate의 결과만이 현재 시점의 cell state 값을 결정하게 된다. **반대로 input gate의 출력값 $i_{t}$가 0 이라면, 현재 시점의 cell state 값은 오직 이전 시점의 cell state ($C_{t-1}$) 의 값에 의해 결정된다**
        
        따라서 forget gate는 이전 시점의 입력을 얼마나 반영할지를, input gate는 현재 시점의 입력을 얼마나 반영할지 결정하게 된다
        
        cell state의 업데이트는 각 gate의 결과를 더함으로써 진행되기 때문에 이는 시퀀스가 길어도 gradient, 즉 오차를 상대적으로 잘 전파하는 역할을 수행한다
        
        $C_t = f_t \circ C_{t-1} + i_t \circ g_t$
        
    
    - **output gate & hidden state (short term, 단기 상태)**
        
        ![Untitled 7](https://user-images.githubusercontent.com/54128055/144828323-eea545b3-85b3-4740-99a5-21aae815f351.png)
        
        output gate는 현재 시점 $t$의 $x$값과 이전 시점 $t-1$의 hidden state가 sigmoid 함수를 통과한다. 해당 결괏값은 현재 시점 $t$의 hidden state를 결정한다
        
        **hidden state를 단기 상태라고도 하는데, hidden state는 장기 상태의 값이 $tanh$ 함수를 통과하여 -1~1 사이의 값을 갖게 된다. 이는 output gate의 값과 연산되어 값이 걸러지는 효과를 발생한다**
        
        $o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_t)$
        
        $h_t = o_t \circ tanh(c_t)$
        

## Hyperbolic Tangent

$tanh x = \frac{e^x - e^x}{e^x + e^x} = \frac{e^{2x} - 1}{e^{2x} + 1}$

![Untitled 8](https://user-images.githubusercontent.com/54128055/144828324-849bdc59-a0cf-4d3d-9edf-85e433619db9.png)

**sigmoid 함수**

- sigmoid는 (0, 0.5)에 대해 대칭
- $x$ 값이 커지면 $y$ 값은 1, $x$ 값이 작아지면 $y$ 값은 0에 수렴
- 함수의 출력 값은 0 ~ 1 사이로 다른 값과 곱해질 경우 "전부 차단"에서 "전부 통과"까지를 양극단으로 하면서 값의 일부만을 통과하게끔 일종의 밸브 역할로 gate라 불림

**hyperbolic tangent 함수**

- 원점에 대해 대칭
- $x$ 값이 커지면 $y$ 값은 1, $x$ 값이 작아지면 $y$ 값은 -1에 수렴
- sigmoid 함수의 그래프를 위아래 방향으로 두 배 늘리고, 좌우 방향으로는 절반으로 줄였으며 중심점을 원점으로 옮긴 모양
    
    $tanh x = \frac{e^{2x} - 1}{e^{2x} + 1} = \frac{e^{2x} + 1 - 2}{e^{2x} + 1} = 1 - \frac{2}{e^{2x} + 1} = 1 - 2\sigma(2x)$
    
    $\frac{\partial}{\partial x}tanh x = \frac{\partial}{\partial x}(1-\frac{2}{e^{2x} + 1}) = \frac{2(e^{2x} + 1)'}{(e^{2x} + 1)^2} = \frac{4e^{2x}}{(e^{2x} + 1)^2} = \frac{(e^{2x} + 1)^2 - (e^{2x} - 1)^2}{(e^{2x} + 1)^2} = 1 - tanh^2x$
    
- 함수의 출력 값은 -1 ~ 1 사이의 범위로 상태 벡터나 순환 벡터를 계산하는 과정에서 값의 합산 및 감산 또한 가능해지는 효과를 얻음

## LSTM backpropagation

reference: [https://ratsgo.github.io/natural language processing/2017/03/09/rnnlstm/](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)

- $dy_t$를 시작으로 순차적으로 backword process 진행
- $df_t, di_t, dg_t, do_t$를 구하기까지의 프로세스는 RNN과 유사

![Untitled 9](https://user-images.githubusercontent.com/54128055/144828327-f89be5b0-8841-40ce-bf70-0514175f1efd.png)

- $dH_t$ 를 구하는 과정이 LSTM backward pass의 핵심
    - $H_t$ 는 $f_t, i_t,o_t, g_t$ 로 구성된 행렬
    - 따라서 각각의 해당하는 gradient를 합쳐서(merge) $dH_t$
    - $f_t, i_t,o_t$ 의 활성화 함수는 sigmoid, $g_t$ 의 활성화 함수는 $tanh$ 이기 때문에 각각의 활성화 함수에 대한 local gradient를 구하고 각각의 흘러들어온 gradient를 곱함
        
        ![Untitled 10](https://user-images.githubusercontent.com/54128055/144828329-a266e8f9-6a29-4e1e-b92b-546538402913.png)
        
- 위에서 $df_t, di_t, dg_t, do_t$를 합쳐서 구한 $dH_t$는 다시 RNN과 같은 방식으로 backpropagate

![Untitled 11](https://user-images.githubusercontent.com/54128055/144828331-fec88432-8c20-4270-a79e-12c5fb9ea7d0.png)

---

## GRU

- **Gated Recurrent Unit, GRU**
- LSTM에서 수학적으로 수식을 소거하여 나온 모델
    - 그에 따라 연산량 또한 많이 줄어듬 → weight 갯수가 줄었다는 의미
    - LSTM에 준하는 성능 제공
    - LSTM 셀의 간소화된 버전
- 3개의 gate 가 있었던 LSTM과 달리 **update gate, reset gate**로 gate의 수를 2개로 줄임
    
    ![Untitled 12](https://user-images.githubusercontent.com/54128055/144828333-b692491a-8ca9-4212-b540-43664b040fa7.png)
    
    - $r_t = \sigma(W^T_{xr}\cdot x_t + W^T_{hr}\cdot h_{t-1} + b_r )$
    - $z_t = \sigma(W^T_{xz}\cdot x_t + W^T_{hz}\cdot h_{t-1} + b_z )$
    - $g_t = tanh(W^T_{xg}\cdot x_t + W^T_{hg}\cdot (r_t \otimes h_{t-1}) + b_g )$
    - $h_t = z_t \otimes h_{t-1} + (1 - z_t) \otimes g_t$
        
        $\otimes :$  tensor product
        
        - $z_t$가 forget, input gate를 모두 제어한다. $z_t$가 1을 출력하면 forget gate가 열리고 input gate가 닫힌다. 반대로 $z_t$가 0 일 경우 forget gate가 닫히고 input gate가 열린다
        - 이전 $t-1$의 기억이 저장 될때 마다 time-step $t$ 의 입력은 삭제된다
        - GRU 셀은 output gate가 없어 전체 상태 벡터가 time-step 마다 출력되며, 이전 상태의 어느 부분이 출력될지 제어하는 새로운 gate controller인 $r_t$가 있다
        
        reference: [https://excelsior-cjh.tistory.com/185](https://excelsior-cjh.tistory.com/185)
        

**Reset Gate**

- reset gate를 구하는 공식은 위 그림에서 $r_t$식에 해당
- 이전 시점의 hidden state와 현 시점의 $x$를 시그모이드 함수 통과
- 결과값은 0~1 사이의 값을 가지며 이전 hidden state의 값을 얼마나 반영할 것인지 정함
- reset gate의 출력값이 그대로 사용되는 것이 아니라 $g_t$ 를 도출할 때 활용
    - 전 시점의 hidden state에 reset gate를 곱하여 계산

**Update Gate**

- update gate는 LSTM의 input, forget gate와 비슷한 역할
- 과거와 현재의 정보를 각각 얼마나 반영할지에 대한 비율을 정함
- $z_t$ 통해 구한 결과 z는 현재 정보를 얼마나 사용할지 반영
- $h_t$의 $(1-z_t)$는 과거 정보에 대해서 얼마나 사용할지 반영
- 각 역할을 LSTM의 input, forget gate로 볼 수 있고 최종적으로 $h_t$를 통해 현 시점의 hidden state 도출

### LSTM vs. GRU

- deep learning 모델 특성상 데이터가 많으면 모델의 용량을 키워야함, 즉 weight 와 bias 의 개수가 많으면 많을 수록 좋다
    - 반대로 데이터가 적으면 parameter의 개수가 적으면 좋음
- 사람들의 다양한 실험을 통해 데이터가 많으면 LSTM이 GRU 보다 우수한 성능을 보임
    - Google 번역기 논문에 담겨진 내용에 따르면 LSTM 과 GRU를 번갈아가며 전부 실험한 결과 LSTM이 항상 우수

```python
# ex) binary model
model = Sequential()
model.add(Embedding(5000,128))
model.add(GRU(256)) #SimpleRNN, LSTM
model.add(Dense(1, activation = "sigmoid"))
```

---

## Bidirectional Recurrent Neural Network

![Untitled 13](https://user-images.githubusercontent.com/54128055/144828334-9da699e5-802f-410c-8f3f-632e57957233.png)

- 역방향으로 입력을 참고하는 RNN을 추가하여 양방향으로 만든 것
- 주황색 기존의 RNN 모델, 초록색은 text를 뒤에서 앞으로 읽은 모델
    - ex) $y_3$의 경우
        - $x_1, x_2, x_3$ 를 읽은 RNN의 hidden state를 출력층으로 보내는 동시에 $x_4, x_3$ 를 뒤에서 부터 읽은 RNN의 hidden state 또한 출력층으로 보낸다
- Bidirectional RNN은 앞의 문맥뿐만 아니라 뒤의 문맥까지 참고할 수 있다는 이점을 가짐

```markdown
Exercise is very effective at [ ] belly fat.
1) Reducing
2) increasing
3) multiplying
```
