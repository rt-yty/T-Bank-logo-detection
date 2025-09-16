import io
from typing import List, Dict

from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
model = None
CONFIDENCE_THRESHOLD = 0.2


def load_model():
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Модель YOLOv8 из файла {MODEL_PATH} успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        model = None


def predict(image_bytes: bytes) -> List[Dict[str, int]]:
    if model is None:
        raise RuntimeError("Модель не загружена. Проверьте путь и целостность файла весов.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model.predict(source=image, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()

        output = []
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            output.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            })
        return output
    except Exception as e:
        print(f"Ошибка во время детекции: {e}")
        return []