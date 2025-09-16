import os
import cv2
import pandas as pd
from tqdm import tqdm
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.detection import load_model, predict

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def main(args):
    print("Загрузка модели для валидации...")
    load_model()

    if not os.path.exists(args.labels_path):
        print(f"Ошибка: Файл с разметкой не найден по пути: {args.labels_path}")
        print("Пожалуйста, сначала запустите скрипт convert_yolo_to_csv.py")
        return

    validation_df = pd.read_csv(args.labels_path)
    true_positives, false_positives, false_negatives = 0, 0, 0
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print("Начало валидации...")
    for image_name, group in tqdm(validation_df.groupby('filename')):
        image_path = os.path.join(args.images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Внимание: файл {image_path} не найден.")
            continue
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        predicted_boxes = predict(image_bytes)
        true_boxes = [list(row) for row in group[['x_min', 'y_min', 'x_max', 'y_max']].values]
        matched_true_boxes = [False] * len(true_boxes)
        if not predicted_boxes and true_boxes:
            false_negatives += len(true_boxes)
            continue
        if predicted_boxes and not true_boxes:
            false_positives += len(predicted_boxes)
            continue

        for pred_box_dict in predicted_boxes:
            pred_box = [pred_box_dict['x_min'], pred_box_dict['y_min'], pred_box_dict['x_max'], pred_box_dict['y_max']]
            best_iou, best_true_idx = 0, -1
            for i, true_box in enumerate(true_boxes):
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou: best_iou, best_true_idx = iou, i
            if best_iou >= args.iou_threshold and not matched_true_boxes[best_true_idx]:
                true_positives += 1
                matched_true_boxes[best_true_idx] = True
            else:
                false_positives += 1
        false_negatives += matched_true_boxes.count(False)

        if args.draw_results:
            img = cv2.imread(image_path)
            for box in true_boxes: cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            for box_dict in predicted_boxes: cv2.rectangle(img, (box_dict['x_min'], box_dict['y_min']),
                                                           (box_dict['x_max'], box_dict['y_max']), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(output_dir, image_name), img)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("\n--- Результаты Валидации ---")
    print(f"IoU порог: {args.iou_threshold}\nPrecision: {precision:.4f}\nRecall:    {recall:.4f}\nF1-score:  {f1_score:.4f}")
    if args.draw_results: print(f"Изображения с результатами сохранены в: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт валидации модели детекции.")
    parser.add_argument('--images-dir', type=str, default='data/validation/images',
                        help='Путь к папке с изображениями.')
    parser.add_argument('--labels-path', type=str, default='data/validation/labels.csv',
                        help='Путь к CSV файлу с разметкой.')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='Порог IoU для F1-score.')
    parser.add_argument('--draw-results', action='store_true', help='Сохранить изображения с отрисованными рамками.')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                        help='Папка для сохранения результатов.')
    args = parser.parse_args()
    main(args)