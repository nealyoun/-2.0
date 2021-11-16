# Evaluation Metrice - IoU

## Object Detection Evaluation Metrice

### IoU: Intersection over Union

- 모델이 예측한 결과와 실측(ground truth) Box가 얼마나 정확하게 겹치는지가를 나타내는 지표
- 분자: Ground Truth 와 predicted Bounding Box의 intersection
- 분모: round Truth 와 predicted Bounding Box의 Union
- 완벽하게 detect 했다면 intersection과 union의 면적이 거의 대등하므로 $IoU \fallingdotseq 1$
    - 즉, IoU가 0 이면 교집합이 없다는 의미고, IoU가 1이면 두 박스가 완전히 일치하는 것
        
        ![Screen Shot 2021-11-14 at 5.42.53 PM.png](Evaluation%20Metrice%20-%20IoU%20fd7f73181c58418988bec4d7c4b3120d/Screen_Shot_2021-11-14_at_5.42.53_PM.png)
        
        - Pascal VOC 기준
        - $IoU < 0.5$  → Poor Detection
        - $IoU \fallingdotseq 0.75$ → Good Detection
        - $IoU > 0.90$  → Excellent Detection

*PASCAL VOC 같은 경우 IoU ≥ 0.5이면 예측 성공으로 보고, MS COCO 같은 경우 IoU 기준을 점점 높여가면서 진행*