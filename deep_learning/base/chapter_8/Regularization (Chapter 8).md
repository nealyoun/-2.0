# Regularization (Chapter 8)

Assign: Anonymous
Attachment: Regularization%20(Chapter%208)%20ff10c16b02d54c54a431c977820ab6b8/Regularization.pdf
Due Date: November 14, 2021
Status: Completed

1. Regularization
• 일부러 적당한 제약을 가해 학습을 방해하는 기법
• 학습을 방해하는 이유는 과적합 현상을 막기 위함
2. Method
• L2 Loss
• L1 Loss
• Dropout
• Noise Injection
• Batch Normalization

# BackGround

1. Underfitting
• 학습용 예제 자체를 제대로 풀지 못하는 현상
• 문제의 난이도에 비해 모델 용량이 부족할 때 발생
• 해결방안은 모델 용량, 학습 횟수, 데이터 양, 품질을 확인해보고 수정
• 어떻게 대처해야 할지 알기 쉽지만, 문제 해결을 위해 늘리는 것은 쉽지 않음
2. Overfitting
• 전체적인 문제 특성을 파악하지 못한 채 학습 데이터 자체의 사소한 특성을 외워 버리는 바람에 나타나는 현상
• Train loss는 감소하는 추세를 보이지만 Validation loss 는 감소하다가 증가하는 추세를 보임
• 문제의 난이도에 비해 모델 용량이 크면서 데이터가 부족할 때 주로 발생
• 해결 방안은 양질의 데이터를 충분히 확보하는 것이지만, 현실적으로 힘든 경우가 많음
• 따라서 학습 과정에 적당한 활용하는 것이 현실적
• 앞서 언급했던 방법들을 사용하는데 L2 & L1 Loss는 기존 계층들의 순전파 및 역전파 처리 방식을 변형시키는 형태로 처리, 나머지는 새로운
계층으로 추가

# Method

1. L2 Loss
• L2 페널티 값을 손실 함수에 더해주는 정규화 기법
• 절댓값이 큰 파라미터에 대해 제약을 걸어 발산을 막고 되도록 작은 파라미터들로 문제를 풀도록 유도
• 하지만 파라미터 절댓값을 줄여주지는 않음
    
    ![Untitled](https://user-images.githubusercontent.com/54128055/142767164-b79ddde8-bbc7-4824-ae7f-0ef2f743b860.png)
    
    • 이에 따라 역전파 처리도 수정 -> 각각의 파라미터 𝑤𝑖값이 수정될 때 L2 손실을 고려하지 않고 기존의 손실 기울기에 𝜆𝑤𝑖값만 더해서 처리
    
    ![Untitled 1](https://user-images.githubusercontent.com/54128055/142767143-e83d5321-86c4-4d7d-9c1b-327f6b122b6d.png)
    
    • 그렇다고 파라미터가 0으로 수렴하는 것이 아닌, L2 loss를 상쇄하는 방향으로 학습이 변화
    • 이러한 경향이 파라미터값의 감소와 균형을 이루며 악영향을 미치지 않는 선에서 작은 값들로 모임
    • L2 loss의 영향을 받는 가중치 파라미터에는 fully connected layer의 가중치 행렬은 물론 conv layer의 kernel도 포함. 다만 bias는 제외시키는
    데 어느 정도의 크기가 나와야 하는 출력의 경우 bias가 그 역할을 해주기 때문
    • Validation set에는 필요하지 않음
    
2. L1 Loss
• L1 페널티 값을 손실 함수에 더해주는 정규화 기법
    
    ![Untitled 2](https://user-images.githubusercontent.com/54128055/142767176-c5a91fe2-46d0-4843-b097-df20b4698430.png)
    
    • 역전파 처리도 수정 -> 각각의 파라미터 𝑤𝑖값이 수정될 때 L1 손실을 고려하지 않고 기존의 손실 기울기에 𝛼sign(𝑤𝑖)값만 더해서 처리
    
    ![Untitled 3](https://user-images.githubusercontent.com/54128055/142767193-30917378-5e37-408b-88e8-24bb2a8347a0.png)
    
    • L2처럼 자기 값에서 일정한 비율을 덜어내는 것이 아니라 일률적으로 정해진 값을 덜어내는 것
    • 따라서 큰 값을 갖는 파라미터는 미미한 영향을 주지만 작은 파라미터에 대해서는 치명적, 따라서 0으로 수렴하는 파라미터를 양산
    • 하지만 마찬가지로 문제 풀이에 꼭 필요한 일부에 대해서는 L1 loss를 상쇄하는 방향으로 학습
    • L1 loss의 영향을 받는 가중치 파라미터에는 fully connected layer의 가중치 행렬은 물론 conv layer의 kernel도 포함. 다만 bias는 제외 시키는데 어느 정도의 크기가 나와야 하는 출력의 경우 bias가 그 역할을 해주기 때문
    
3. Dropout
• 입력 또는 어떤 계층의 출력을 다음 계층에서 모두 이용하지 않고 일부만 이용하면서 신경망을 학습시키는 regularization 기법
    
    ![Untitled 4](https://user-images.githubusercontent.com/54128055/142767212-47da9a10-bfc3-42bd-aa76-49cbaa511db9.png)
    
    출처:bit/ly/2Pmhq9i
    
    • 다만 그림을 보고 계산량이 줄어든다고 생각하면 안됨. 실제 처리과정은 점선으로 된 노드들이 0으로 존재하기 때문에 처리과정은 동일하며,
    오히려 난수 함수를 통한 노드 선정, 순전파와 역전파에서 0으로 마스킹 처리까지 하면 계산량은 오히려 증가
    • Dropout은 보통 하나의 독립된 계층 형태로 이용, drop ratio라는 하이퍼파라미터로 n%확률로 노드를 날리도록 설정
    • 단 n%만큼 버리는게 아닌 n%확률임을 유의
    • 살아남은 원소들이 100%의 역할을 하기위해 100/(100-n) 배로 증폭
    • 역전파에서 dropout layer는 순전파 처리에 참여했던 성분들에만 손실 기울기 전달하고 나머지는 0으로 처리, 이때 순전파에서 곱했던 배율을
    손실 기울기에도 똑같이 곱해 전달. 해당 처리를 위해 dropout layer는 각 성분의 사용 여부를 나타내는 마스크 정보를 보조 정보로 유지
    • 학습 시점에만 적용(is_running 플래그 정보를 참조하여 시점 구별)
    • 장점으로는 같은 문제를 반복해 풀더라도 그 처리를 담당하는 퍼셉트론 그룹을 매번 다르게 배당하여 패턴 형태로 단순 암기하는 것을 예방
    -> 문제 해결을 위한 패턴 처리 능력을 특정 요소에 집중시키는 대신 여러 요소에 골고루 퍼뜨리는 효과, local minimum에 갇히는 현상 방지
    • Batch Norm과 함께 가장 효과가 큰 기법
    
4. Noise Injection
• 두 계층 사이에 Noise Injection layer를 삽입하여 이 계층이 아래 계층의 출력에 적당한 형태의 잡음을 추가하여 위 계층에 전달하는 정규화 기법
• Noise가 주입되면 그만큼 학습이 더뎌지지만 더욱 robust한 학습이 이루어짐 -> test data의 다양한 noise에 대해 robust
• Noise는 입력과 무관히 더해진 것 뿐, noise 후 output과 original input의 편미분은 1 이기 때문에 학습의 대상이 아니기에 역전파 처리의 대상이 아님
• 매번 다른 noise 주입으로 모델의 과적합 방지
5. Batch Normalization
• BN은 mini-batch 내의 데이터들에 대해 벡터 성분별로 정규화를 수행하는 방식
• Normalization은 선형 변환을 가해 평균 0, 표준 편차 1의 분포로 만들어주는 처리 (2가지 방식이 있음)
• Normalization은 입력 성분 간의 분포 차이로 인한 가중치 학습의 불균형을 방지하기 위해 도입
• 중간 layer들은 계속 다른 input값을 받기 때문에 적어도 평균 분산은 유지시켜 주자는 생각에서 시작
• BN의 주목할 점은 보통 preprocessing 단계에서 input에 대해서만 정규화시켜 주었는데, 이를 mini-batch data를 대상으로 삼아 hidden node까지
normalization 하는 것으로 확장
• [m,n] 형태의 입력이 주어졌을 때, 배치 정규화는 크기가 n인 벡터 성분 각각이 저마다의 독립된 속성을 표현하는 것으로 보고 벡터 성분별로 그룹 n개를
만들어 각 그룹 안에 있는 데이터 값 m개를 normalization.
• 평균을 0으로 만들기 때문에 편향을 제거한 분포를 만들기 때문에 악영향을 미칠 가능성 또한 존재, 이에 따라 BN에서는 데이터 정규화를 통해 얻은 값
에 다시 scale factor라는 파라미터 값을 곱하고 shift factor라는 파라미터 값을 더함 (감마와 베타)
• 이 파라미터들은 처음에 각각 1,0으로 초기화후 학습
• Mini-batch가 너무 작으면 효과가 미미함
- 왜 mini-batch에 대해 정규화?
a. 전체 데이터셋이라고 해도 실세계에선 mini-batch
b. 은닉 계층이 생성하는 은닉 벡터에는 적용 불가
c. 매번 달라지는 mini-batch의 평균과 분산 이용이 일종의 잡음 주입 같은 효과 (Q. 근데 그걸 정규화 시켜서 날리지 않음?)
d. 드롭아웃과는 달리 모든 데이터를 학습에 충분히 이용
