anaconda로 가상 환경 세팅함

서버 start
conda activate fastapi-yolo
uvicorn main:app --reload --host 0.0.0.0 --port 8000
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

http://localhost:8000/docs