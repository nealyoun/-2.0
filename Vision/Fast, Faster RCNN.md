# Fast, Faster RCNN

Assign: Anonymous
Due Date: 2021/12/14
Status: Completed

## 주요 특징

- **SPP Layer를 RoI Pooling Layer로 변환**
    
    ![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled.png)
    
    - **ex)** channel 이 256 으로 가정,
        
        fixed length: 16 x 256, 4 x 256, 256 → (16+4+1) x 256
        
- End-to-End Network Learning (RoI Proposal 제외)
    - Deep Learning Network에 포함되어 있지 않던 SVM을 Softmax으로 변환하면서 Deep Learning Network 안으로 포함 시킴
    - Multi-task loss 함수로 Classification과 Regression을 함께 최적화

# RoI Pooling

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%201.png)

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%202.png)

- 위의 Feature Map에서 Selective Search를 통해 8x4 RoI를 추출
- 매핑 시 일반적으로 Max Pooling을 적용
    
    ![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%203.png)
    
- 가령 2x2로 pooling을 한다면
    
    | 29 | 31 |
    
    | 30 | 28 |
    
     
    

### Fast RCNN - RoI Pooling

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%204.png)

- 일반적으로 Pool 크기를 7x7로 설정
- 14x7 사이즈는 7x7 pooling 으로 딱 떨어짐
- 8x4 사이즈의 경우에는 7x7 pool 사이즈와 정수형으로 떨어지지 않음
    - image resize 메서드를 이용
    - 혹은 보간법을 이용하여 pool size와 떨어지게 보정
- SPP와는 다르게 Feature Map이 가령 256 이라면 RoI Pooling 또한 256개
    - RoI pooling은 면적적인 사이즈만 고정, channel depth는 유지
    - 따라서, batch size = 4, pool size = 7 x 7 이라면
        
        (4, 2000, 7, 7, 256)
        
    - FC Layer 통과 후 SVM이 아닌, Softmax 함수에 의해서 분류

- Multi-task Loss function: $L(p, u, t^u, v) = L_{cls}(p,u) + \lambda[u\geq 1]L_{loc}(t^u,v)$
    
    ![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%205.png)
    

---

# Faster RCNN

- Fast RCNN의 경우 Selective Search 를 통한 RoI Proposal은 Deep Learning Network에 미포함
- Faster RCNN의 경우 Selective Search를 통한 RoI Proposal 구조를 RPN(Region Proposal Network)로 별도로 구성
    - Faster RCNN = RPN + Fast RCNN

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%206.png)

- 기존의 Selective Search를 Neural Network 구조로 변경; selective search를 RPN으로 대체
    - RPN을 통해 object가 있을 만한 위치를 탐색
    - Back-propagation시 RPN에도 적용
    - GPU 사용으로 빠른 학습 즉, inference 가 줄어듬
    - End to End Network 학습이 가능
    

## Region Proposal Network issues

- Selective Search는 특징들의 segmentation을 통해 2000개의 위치를 추정
- Selective Search 수준의 Region Proposal을 위해 **Anchor Box** 개념을 도입
    - Anchor Box: Object 가 있는지 없는지의 후보 Box

### Anchor Box

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%207.png)

Object의 형태가 모두 동일하지 않고 다르기 때문에 총 9개의 Anchor box를 생성 (3개의 서로 다른 크기, 3개의 서로 다른 ratio로 구성)

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%208.png)

- 가령 위와 같은 image에 가로가 긴 Anchor box만을 이용한다면, 차량 보다 높이 솟아 있는 image 속 사람은 detect하지 못한다. 즉, 다양한 형태의 object들을 모두 찾아내기 위해 총 9개의 Anchor box를 이용한다.
    - IOU를 통해 Anchor box 내에 object가 얼마나 들어와있는지 판단하게 되는데, 가로로 긴 Anchor box만을 이용한다면 해당 image 속의 사람과 IOU가 겹치는 Anchor box를 찾을 수 없음
    

### Anchor box mapping

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%209.png)

- 원본 image width = 800, 원본 image height = 600
- 원본 image를 backbone을 통해 1/16 크기의 Feature map으로 down sampling
    - Feature map width: 800/16 = 50, Feature map height: 600/16 = 38
- Grid Point가 feature map의 width와 height에 따라 생성, 해당 이미지의 경우 가로로 50개, 세로로 38개, 총 1900개의 grid point 생성
- 각 Grid Point 별로 Anchor box 생성
    
    ![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%2010.png)
    
    - Anchor box는 9개로 구성되므로 Anchor box의 총 개수는 1900 * 9 = 17100
        
        ![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%2011.png)
        

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%2012.png)

***Anchor Box 한계점 Reference:* [https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=infoefficien&logNo=221229808775](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=infoefficien&logNo=221229808775)**

### Positive, Negative Anchor Box

Ground Truth Bounding Box와 겹치는 IOU 값에 따라 Anchor Box를 Positive 혹은 Negative로 분류

- IOU가 가장 높은 Anchor는 Positive
- IOU가 0.7 이상이면 Postive
- IOU가 0.3보다 작으면 Negative
- IOU가 0.3과 0.7 사이면 학습에서 제외

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%2013.png)

- 예측 B.Box와 Positive Anchor Box의 좌표 (거리) 차이는 Ground Truth와 Positive Anchor Box와의 좌표 차이와 최대한 동일하게 될 수 있도록 regression 학습

### RPN Loss Function

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%2014.png)

$p_i:$  Anchor i가 오브젝트일 예측 확률
$p^*_i:$  Anchor i의 Ground truth Object 여부(Positive 1, Negative 0)

$t_i:$  Anchor i와 예측 좌표 차이(x, y, w, h)

$t^*_i:$  Anchor i와 Ground Truth 좌표 차이(x, y, w, h)

$N_{cls}:$  미니 배치에 따른 정규화 값(256)

$N_{box}:$  박스 개수 정규화 값 (최대 2400)

$\lambda:$  밸런싱 값: 10

![Untitled](Fast,%20Faster%20RCNN%200f835daec8be474aba2be88368936677/Untitled%2015.png)

논문에 기재된 3가지의 학습 방법 (*Alternative training, Approximate joint training, Non-approximate joint training*) 중 최종적으로 사용한 학습 방법: *"Alternative training"*

1. RPN 을 먼저 학습
2. Fast RCNN Classification/Regression 학습
3. RPN 을 Fine Tuning
4. Fast RCNN Fine Tuning