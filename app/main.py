from fastapi import FastAPI, File, UploadFile, HTTPException, status
import traceback

from .models import DetectionResponse, ErrorResponse, Detection, BoundingBox
from .detection import predict, load_model

app = FastAPI(
    title="T-Bank Logo Detection API",
    description="API для детекции логотипа Т-Банка на изображениях.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    print("Запуск сервера и загрузка модели...")
    load_model()

@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        200: {"description": "Успешная детекция"},
        400: {"model": ErrorResponse, "description": "Неверный формат файла"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"}
    }
)
async def detect_logo(file: UploadFile = File(...)):
    supported_formats = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    if file.content_type not in supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый формат файла: {file.content_type}. Используйте JPEG, PNG, BMP или WEBP."
        )

    try:
        image_bytes = await file.read()
        detection_results = predict(image_bytes)
        detections = [
            Detection(bbox=BoundingBox(**res)) for res in detection_results
        ]
        return DetectionResponse(detections=detections)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при обработке изображения: {str(e)}"
        )