# Шаг 1: Выбор базового образа Python
FROM python:3.9-slim

# Шаг 2: Установка системных зависимостей для OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Шаг 3: Установка рабочей директории
WORKDIR /app

# Шаг 4: Копирование и установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Шаг 5: Копирование кода проекта и весов
COPY ./app /app/app
COPY ./weights /app/weights

# Шаг 6: Указание порта для API
EXPOSE 8000

# Шаг 7: Команда для запуска Uvicorn сервера
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]