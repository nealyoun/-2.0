# OpenCV
---
## Python 기반 주요 image library

- PIL (python image library)
    - 주로 이미지 처리만을 위해 사용
    - 처리 성능이 상대적으로 느림
- Scikit Image
    - 파이썬 기반의 computer vision 기능 제공
    - Scipy, Numpy
- OpenCV
    - 가장 주로 사용되는 computer vision library
    - 컴퓨터 비전 기능 일반화에 크게 기여
    - C++ 기반이나 Python 도 지원 (Java, C# 등 다양한 언어 지원)
    - 방대한 CV 관련 라이브러리와 손쉬운 인터페이스 제공
    

## OpenCV image load

- imread()를 이용한 이미지 로딩 (imread()는 파일을 읽어 numpy array로 변환)
    
    주의할 점: 
    
    - **OpenCV를 이용하여 이미지를 불러오면 RGB 형태가 아닌 BGR 형태로 로딩**
    - cvtColor()를 이용하여 BGR을 RGB로 변환
        
        <img width="502" alt="Screen_Shot_2021-12-01_at_3 51 23_PM" src="https://user-images.githubusercontent.com/54128055/144253616-307b8add-a8d0-4935-8de6-8d92f7260b8a.png">
        
    - imwrite()를 이용하여 파일 쓰기 기능 제공. 하지만 imread()로 인해 BGR형태로 되어 있는 이미지 배열을 다시 RGB 형태로 변환하여 저장하는 것을 주의
    - OpenCV Windows Frame 인터페이스
        - Window Frame 생성이 가능한 GUI 개발 환경에서만 가능 (Windows GUI, Linux X-windows)
        - cv2.imshow()
        - cv2.waitKey()
        - cv2.destroyAllWindows()
        - Jupyter 기반에서는 이미지 배열의 시각화에 matplotlib를 사용 (위 함수들은 Jupyter에서 제공되지 않음)

## OpenCV video

- OpenCV의 VideoCapture()클래스는 동영상을 개별 frame으로 하나씩 read하는 기능을 제공
    - VideoCapture 객체는 생성 인조로 입력 video 파일 위치를 받아 생성
        
        ```python
        cap = cv2.VideoCapture("video_input_path")
        ```
        
    - VideoCapture 객체는 입력 video 파일의 다양한 속성을 불러 올 수 있음
        
        ```python
        # 영상 Frame width
        cap.get(cap.CAP_PROP_FRAME_WIDTH)
        
        # 영상 Frame height
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # 영상 FPS
        cap.get(cv2.CAP_PROP_FPS)
        ```
        
    - VideoCapture 객체의 read()는 마지막 Frame까지 차례로 Frame을 읽음
        
        ```python
        while True:
        	hasFrame, img_frame = cap.read()
        	if not hasFrame:
        		break
        ```
        

- VideoWriter는 VideoCapture로 읽어들인 개별 frame을 동영상 파일로 write하는 기능을 제공
    - VideoWriter 객체는 write할 동영상 파일 위치, Encoding 코덱 유형, write fps 수치, frame 크기를 생성자로 입력 받아 이들 값에 따른 동영상 Write 수행
    - 특정 포맷으로 동영상 Encoding 가능 (DIVX, XVID, MJPG, X264, WMV1, WMV2)
        
        ```python
        cap = cv2.VideoCapture("video_input_path")
        
        codec = cv2.VideoWriter_fourcc(*"XVID")
        
        vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 아래와 같은 포맷으로 동영상을 encoding하여 write
        vid_writer = cv2.VideoWriter("video_input_path", codec, vid_fps, vid_size)
        ```
