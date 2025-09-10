# main.py
import os
import sys
import shutil
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# YOLOv5 경로 등록
yolov5_path = os.path.join(os.path.dirname(__file__), "yolov5")
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# FastAPI 앱 생성 및 CORS 허용 설정
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:8080"] 등 필요한 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLOv5 모델 로딩
try:
    model = torch.hub.load(
        'yolov5',
        'custom',
        path='./best.pt',
        source='local',
        force_reload=True
    )
    model.eval()
    model.conf = 0.25
    model.iou = 0.45
    model.max_det = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    names = model.names
    print(f"YOLOv5 model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading YOLOv5 model or modules: {e}")
    sys.exit(1)

# 음식별 100g당 칼로리
calorie_table = {
    "rice": 130, "eels on rice": "---", "pilaf": 125, "chicken-'n'-egg on rice": "---", "pork cutlet on rice": "---",
    "beef curry": 184, "sushi": 143, "chicken rice": "---", "fried rice": 180, "tempura bowl": "---",
    "bibimbap": 79, "toast": 265, "croissant": "---", "roll bread": "---", "raisin bread": 280, "chip butty": "---",
    "hamburger": 250, "pizza": 268, "sandwiches": 220, "udon noodle": 130, "tempura udon": "---",
    "soba noodle": 110, "ramen noodle": 440, "beef noodle": 200, "tensin noodle": "---", "fried noodle": 350,
    "spaghetti": 160, "Japanese-style pancake": 110, "takoyaki": 150, "gratin": 180, "sauteed vegetables": 50,
    "croquette": 200, "grilled eggplant": 60, "sauteed spinach": 40, "vegetable tempura": 180, "miso soup": 50,
    "potage": 60, "sausage": 300, "oden": 70, "omelet": 150, "ganmodoki": 140, "jiaozi": 180, "stew": 120,
    "teriyaki grilled fish": 200, "fried fish": 220, "grilled salmon": 208, "salmon meuniere": 210, "sashimi": 143,
    "grilled pacific saury": 180, "sukiyaki": 130, "sweet and sour pork": 180, "lightly roasted fish": 140,
    "steamed egg hotchpotch": 90, "tempura": 300, "fried chicken": 297, "sirloin cutlet": 250, "nanbanzuke": 180,
    "boiled fish": 120, "seasoned beef with potatoes": 190, "hambarg steak": 280, "beef steak": 250, "dried fish": 350,
    "ginger pork saute": 200, "spicy chili-flavored tofu": 150, "yakitori": 180, "cabbage roll": 120,
    "rolled omelet": 150, "egg sunny‑side up": 155, "fermented soybeans": 200, "cold tofu": 70, "egg roll": 180,
    "chilled noodle": 120, "stir‑fried beef and peppers": 200, "simmered pork": 180, "boiled chicken and vegetables": 150,
    "sashimi bowl": 140, "sushi bowl": 150, "fish‑shaped pancake with bean jam": 220, "shrimp with chill source": 160,
    "roast chicken": 200, "steamed meat dumpling": 180, "omelet with fried rice": 220, "cutlet curry": 230,
    "spaghetti meat sauce": 200, "fried shrimp": 240, "potato salad": 200, "green salad": 40, "macaroni salad": 150,
    "Japanese tofu and vegetable chowder": 100, "pork miso soup": 80, "chinese soup": 90, "beef bowl": 180,
    "kinpira‑style sauteed burdock": 80, "rice ball": 170, "pizza toast": 260, "dipping noodles": 140,
    "hot dog": 280, "french fries": 312, "mixed rice": 160, "goya chanpuru": 200
}

# 음식 이미지 분석 API
@app.post("/detect/")
async def detect_objects(image: UploadFile = File(...)):
    temp_image_path = f"temp_{image.filename}"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        img_pil = Image.open(temp_image_path).convert('RGB')
        img_np = np.array(img_pil)
        results = model(img_np, size=640)
        detections_df = results.pandas().xyxy[0]

        counts = {}
        calories = {}
        total_calories = 0

        if not detections_df.empty:
            for _, row in detections_df.iterrows():
                name = row['name']
                counts[name] = counts.get(name, 0) + 1

            for food, count in counts.items():
                kcal = calorie_table.get(food, 0)
                if isinstance(kcal, int):
                    total = kcal * count
                    calories[food] = total
                    total_calories += total
                else:
                    calories[food] = "unknown"

        return JSONResponse(content={
            "detected_counts": counts,
            "estimated_calories": calories,
            "total_calories": total_calories
        })

    except Exception as e:
        print(f"Error during object detection: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_image_path)
