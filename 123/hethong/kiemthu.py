import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from ultralytics import YOLO
from processing import PlateProcessor

# ========== CẤU HÌNH ==========
MODEL_CNN_PATH = "model_cnn_v2.keras"
MODEL_YOLO_PATH = "best.pt"
TEST_DATA_DIR = "train/images"
CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9',
               'A','B','C','D','E','F','G','H','K','L',
               'M','N','P','R','S','T','U','V','X','Y','Z']

# ========== KHỞI TẠO ==========
print("Đang tải các mô hình...")
yolo = YOLO(MODEL_YOLO_PATH)
cnn = keras.models.load_model(MODEL_CNN_PATH)
processor = PlateProcessor(yolo, cnn, CLASS_NAMES, debug=True, debug_wait=0, debug_prefix="PROC")

def run_test():
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục {TEST_DATA_DIR}")
        return

    image_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print("Thư mục test trống rỗng!")
        return

    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu nhận diện...\n")

    for img_name in image_files:
        img_path = os.path.join(TEST_DATA_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # Thực hiện nhận diện qua Processor
        plate_text, confidence, error = processor.recognize(img)

        # Hiển thị kết quả ra Console
        if error:
            print(f" Ảnh: {img_name} | Lỗi: {error}")
        else:
            print(f" Ảnh: {img_name} | Biển số: {plate_text} | Độ tin cậy: {confidence:.2f}%")

        # Hiển thị ảnh kết quả (Tùy chọn)
        # Vẽ chữ lên ảnh để xem cho trực quan
        display_img = img.copy()
        cv2.putText(display_img, f"{plate_text} ({confidence:.1f}%)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Test Result - Press any key", display_img)
        if cv2.waitKey(0) & 0xFF == ord('q'): # Nhấn 'q' để thoát sớm
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()