import os
import glob
import cv2
import pandas as pd
from tqdm import tqdm

IMAGES_DIR = 'data/validation/images'
LABELS_DIR = 'data/validation/labels'
OUTPUT_CSV_PATH = 'data/validation/labels.csv'

def yolo_to_xyxy(box, w, h):
    xc, yc, bw, bh = map(float, box[1:])
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    return x1, y1, x2, y2


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    label_files = glob.glob(os.path.join(LABELS_DIR, '*.txt'))
    all_annotations = []

    print(f"Найдено {len(label_files)} файлов с разметкой. Начинаем конвертацию...")

    if not label_files:
        print("\nВнимание: Файлы разметки (.txt) не найдены в директории:", LABELS_DIR)
        print("Пожалуйста, убедитесь, что файлы лежат в правильной папке.")

    for lbl_path in tqdm(label_files):
        base_name = os.path.splitext(os.path.basename(lbl_path))[0]

        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            potential_path = os.path.join(IMAGES_DIR, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if not img_path:
            print(f"Внимание: не найдено изображение для разметки {lbl_path}")
            continue

        try:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
        except Exception as e:
            print(f"\nОшибка при чтении файла изображения {img_path}: {e}")
            continue

        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                x1, y1, x2, y2 = yolo_to_xyxy(parts, w, h)

                all_annotations.append({
                    'filename': os.path.basename(img_path),
                    'x_min': x1,
                    'y_min': y1,
                    'x_max': x2,
                    'y_max': y2
                })

    if not all_annotations:
        print("\nНе удалось создать аннотации. Проверьте содержимое файлов разметки.")
        return

    df = pd.DataFrame(all_annotations)

    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nКонвертация завершена. Данные сохранены в {OUTPUT_CSV_PATH}")


if __name__ == '__main__':
    main()