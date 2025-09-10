**FastAPI**와 **YOLO**를 활용한 음식 객체 탐지 시스템입니다.  
이미지 속 다양한 음식을 실시간으로 인식할 수 있으며, REST API 형태로 제공되어 다른 서비스와 손쉽게 연동할 수 있습니다.  

---

## 🚀 주요 기능
- YOLO 기반 음식 객체 탐지
- FastAPI 기반 REST API 서버 제공
- Swagger UI(`/docs`)를 통한 직관적인 API 테스트
- Redoc(`/redoc`)으로 API 문서 확인 가능
- 확장성과 재사용성을 고려한 모듈 구조

---

## 📦 설치 방법

### 1. 저장소 클론
git clone https://github.com/username/food-detection.git
cd food-detection

2. Conda 가상 환경 생성 및 패키지 설치
bash
코드 복사
conda create -n fastapi-yolo python=3.10 -y
conda activate fastapi-yolo
pip install -r requirements.txt

▶️ 실행 방법
서버 실행
bash
코드 복사
uvicorn main:app --reload --host 0.0.0.0 --port 8000
디버그 로그 모드 실행
bash
코드 복사
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
API 접속 경로
Swagger UI: http://localhost:8000/docs

📂 프로젝트 구조
food-detection/
│── main.py          # FastAPI 엔트리 포인트
│── models/          # YOLO 모델 가중치 파일
│── utils/           # 전처리 및 보조 함수
│── requirements.txt # 의존성 패키지 목록
└── README.md        # 프로젝트 문서

🖼 사용 예시
이미지를 업로드하면 JSON 형태로 탐지 결과를 반환합니다:

json
{
  "predictions": [
    {"label": "피자", "confidence": 0.92, "bbox": [120, 80, 200, 150]},
    {"label": "햄버거", "confidence": 0.88, "bbox": [250, 100, 350, 200]}
  ]
}

---

anaconda로 가상 환경 세팅함

서버 start
conda activate fastapi-yolo
uvicorn main:app --reload --host 0.0.0.0 --port 8000
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

http://localhost:8000/docs
