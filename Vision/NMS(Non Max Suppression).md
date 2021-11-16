# NMS(Non Max Suppression)

- Object Detection 알고리즘들은:
    - 정확하게 Object 가 있는 위치를 반환하는 것이 아닌 Object가 있을 만한 곳을 반환하는 것
    - Object가 있는 곳을 놓치지 않기 위함
    
- Non-Max Suppression:
    - Max가 아닌 것에 대해 Suppress 시키는 것
    - 동일한 물체를 가리키는 여러 박스의 중복을 제거하는 것이 목적
    - Detected 된 Object의 B.Box중에 비슷한 위치에 있는 box를 제거하고 가장 적합한 box를 선택하는 기법
        
        <img width="530" alt="Screen_Shot_2021-11-16_at_8 01 04_PM" src="https://user-images.githubusercontent.com/54128055/142005285-421bcb10-dbec-44bd-be74-cfdac0e26964.png">
        
        특정 클래스나 특정 박스들에 대해서 수행하는 작업이 아닌, **하나의 Detection 장면에서 모든 Bounding Box에 대해 수행하는 작업**
        
        1. Detected 된 B.Box별로 특정 Confidence threshold 이하 B.box는 먼저 제거 (ex: confidence score < 0.5)
            - 모든 박스를 보며, Confidence가 일정 수준 이하인 박스들에 대해 일차적으로 필터링을 거치는 것
        2. 가장 높은 confidence score를 가진 box 순으로 내림차순 정렬 후
            - 높은 confidence score를 가진 box와 다른 box를 모두 조사하여 IoU가 특정 threshold 이상인 box를 모두 제거 ( ex: IoU Threshold > 0.4)
            - 쉽게 떠올리면 $O(N^2)$ 알고리즘이다. 2중 for문
            - IoU가 일정 수준 이상이라면, 두 박스는 서로 같은 물체를 가리키는 것이라고 판단하여 상대적으로 Confidence가 낮은 박스를 제거
            
            위의 로직을 모든 box에 순차적으로 적용
            
        3. 남아 있는 box만 선택
    - **Confidence Score Threshold 가 높을 수록** 많은 값들이 필터링 된다
        
        **IoU Threshold가 낮을 수록** 많은 Box가 제거됨
