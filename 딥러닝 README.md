# 2022150477 김영훈 readme

- `C:\Users\yumye\Downloads\딥러닝_CLIP.ipynb`
- `C:\Users\yumye\Downloads\딥러닝_OWLv2.ipynb`
- `C:\Users\yumye\Downloads\딥러닝_SAM.ipynb`

위의 세 파일 모두 Google Colab 환경을 기준으로 작성되어 있으며, 이미지 1장을 업로드한 뒤 사전학습 비전 모델을 사용해 서로 다른 작업을 수행합니다.

## 1. 노트북 개요

### 1) CLIP 노트북
- 파일: `딥러닝_CLIP.ipynb`
- 사용 모델: `openai/clip-vit-base-patch32`
- 목적: 이미지와 여러 텍스트 후보를 비교해 가장 잘 맞는 라벨을 선택하는 이미지 분류
- 입력:
  - 업로드한 이미지 1장
  - 후보 라벨 목록 (`candidate_labels`)
- 출력:
  - 각 라벨별 확률
  - 가장 높은 확률의 예측 결과
  - 예측 결과가 제목으로 표시된 이미지

현재 예제 라벨:
- `a laptop`
- `a tablet`
- `a desktop computer`
- `a phone`
- `a projector screen`
- `a classroom`

### 2) OWLv2 노트북
- 파일: `딥러닝_OWLv2.ipynb`
- 사용 모델: `google/owlv2-base-patch16-ensemble`
- 목적: 사용자가 입력한 텍스트 질의에 해당하는 객체를 이미지에서 찾아 바운딩 박스로 표시
- 입력:
  - 업로드한 이미지 1장
  - 탐지할 텍스트 질의 (`texts`)
  - 탐지 임계값 (`detection_threshold`)
- 출력:
  - 탐지된 객체의 라벨, 점수, 좌표
  - 객체 위치가 표시된 시각화 결과

현재 예제 질의:
- `laptop`

### 3) SAM 노트북
- 파일: `딥러닝_SAM.ipynb`
- 사용 모델: `facebook/sam-vit-base`
- 목적: 사용자가 지정한 좌표를 기준으로 이미지의 특정 객체를 분할
- 입력:
  - 업로드한 이미지 1장
  - 포인트 좌표 (`input_points`)
  - 포인트 라벨 (`input_labels`)
- 출력:
  - 예측된 마스크 후보 중 최고 점수 결과
  - 원본 이미지 위에 segmentation mask를 덧씌운 시각화

현재 예제 포인트:
- 좌표: `[200, 550]`
- 라벨: `1` (관심 객체 포인트)

## 2. 공통 실행 흐름

세 노트북은 아래와 같은 공통 구조를 가집니다.

1. 필요한 라이브러리 설치
2. PyTorch 및 Hugging Face 관련 모듈 import
3. GPU 사용 가능 여부 확인
4. Colab에서 이미지 업로드
5. 모델과 프로세서 로드
6. 입력값 설정
7. 추론 실행
8. 결과 출력 및 시각화

## 3. 설치 패키지

### CLIP / SAM
```bash
pip install transformers torch torchvision pillow matplotlib
```

### OWLv2
```bash
pip install transformers torch torchvision pillow matplotlib timm
```

## 4. 실행 환경

- 권장 환경: Google Colab
- 권장 하드웨어: GPU
- 프레임워크:
  - `PyTorch`
  - `transformers`
  - `Pillow`
  - `matplotlib`

노트북 내부에서 아래 코드로 GPU 사용 여부를 확인합니다.

```python
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")
```

## 5. 노트북별 핵심 차이

### CLIP
- 이미지 전체를 하나의 의미 단위로 보고 텍스트 후보들과의 유사도를 비교합니다.
- 여러 후보 중 무엇과 가장 가까운지 판단하는 데 적합합니다.
- 객체 위치를 박스로 찾거나 픽셀 단위로 분할하지는 않습니다.

### OWLv2
- 텍스트로 지정한 객체를 이미지 안에서 찾아 위치를 반환합니다.
- "무엇이 있는가"뿐 아니라 "어디에 있는가"까지 확인할 수 있습니다.
- Open-vocabulary detection이므로 사전에 고정된 클래스 목록 없이 텍스트 질의로 대상을 지정할 수 있습니다.

### SAM
- 분류나 탐지보다 더 세밀한 픽셀 단위 분할에 초점이 맞춰져 있습니다.
- 사용자가 점 하나를 찍어주면 해당 위치의 객체 마스크를 생성합니다.
- 객체 경계 추출이나 영역 분리에 적합합니다.

## 6. 입력값 수정 포인트

### CLIP에서 분류 라벨 바꾸기
```python
candidate_labels = [
    "a cat",
    "a dog",
    "a person",
    "a car"
]
```

### OWLv2에서 탐지 대상 바꾸기
```python
texts = [["person"]]
detection_threshold = 0.40
```

여러 질의를 시도하고 싶다면 다음처럼 바꿀 수 있습니다.

```python
texts = [["person", "laptop", "chair"]]
```

### SAM에서 분할 위치 바꾸기
```python
input_points = [[[200, 550]]]
input_labels = [[[1]]]
```

이미지를 먼저 격자와 함께 띄운 뒤, 원하는 객체 위 좌표를 직접 골라 수정하면 됩니다.

## 7. 결과 해석

### CLIP 결과 해석
- 각 라벨에 대해 softmax 확률이 출력됩니다.
- 가장 높은 확률의 라벨이 최종 예측 결과입니다.
- 후보 라벨을 어떻게 작성하느냐에 따라 결과가 달라질 수 있습니다.

### OWLv2 결과 해석
- 각 탐지 결과에 대해 점수와 박스 좌표가 출력됩니다.
- 점수가 낮으면 `detection_threshold`를 조정해볼 수 있습니다.
- 박스가 표시된 위치를 보고 텍스트 질의가 적절했는지 판단할 수 있습니다.

### SAM 결과 해석
- 여러 마스크 후보 중 가장 높은 IoU score를 가진 결과를 선택합니다.
- 점을 찍은 위치가 객체 중심과 너무 멀면 원하는 분할이 나오지 않을 수 있습니다.
- 포인트를 바꾸어 다시 시도하면 결과가 개선될 수 있습니다.

## 8. 추천 사용 순서

하나의 이미지에 대해 세 모델을 함께 학습하거나 비교하려면 다음 순서가 자연스럽습니다.

1. `CLIP`으로 이미지의 전체 의미를 분류한다.
2. `OWLv2`로 원하는 객체의 위치를 찾는다.
3. `SAM`으로 해당 객체를 정밀하게 분할한다.

즉, 세 노트북은 다음처럼 연결해서 이해할 수 있습니다.

- `CLIP`: 이 이미지가 무엇인가?
- `OWLv2`: 그 객체가 어디에 있는가?
- `SAM`: 그 객체의 정확한 영역은 어디까지인가?

## 9. 한계 및 주의사항

- Colab 세션이 초기화되면 설치한 패키지와 업로드한 이미지가 사라집니다.
- GPU가 없으면 모델 로딩과 추론 속도가 느릴 수 있습니다.
- CLIP은 후보 라벨 기반이므로 라벨 설계가 중요합니다.
- OWLv2는 질의어가 너무 모호하면 탐지가 부정확할 수 있습니다.
- SAM은 포인트 좌표 선택이 결과 품질에 큰 영향을 줍니다.

## 10. 요약

이 세 노트북은 서로 다른 비전 태스크를 다룹니다.

- `딥러닝_CLIP.ipynb`: 이미지 분류
- `딥러닝_OWLv2.ipynb`: 텍스트 기반 객체 탐지
- `딥러닝_SAM.ipynb`: 포인트 기반 객체 분할

따라서 같은 이미지를 대상으로 해도,
"무엇인지 분류", "어디 있는지 탐지", "정확히 어느 영역인지 분할"이라는 서로 다른 관점에서 결과를 확인할 수 있습니다.
