import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
import cv2
from ultralytics import YOLO
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from processing import PlateProcessor

# --- 1. CÁC HÀM XỬ LÝ ẢNH CƠ BẢN  ---
def Crop_img(img, model):
    if img is None: return None
    result = model(img, verbose=False)
    if len(result[0].boxes) == 0:
        return None
    box = result[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    return img[y1:y2, x1:x2]

def warp_perspective(img):
    if img is None or img.size == 0: return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    
    max_cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(max_cnt, True)
    approx = cv2.approxPolyDP(max_cnt, 0.03 * peri, True)
    
    if len(approx) == 4 and cv2.contourArea(max_cnt) > (img.shape[0] * img.shape[1] * 0.3):
        pts = approx.reshape(4, 2).astype("float32")
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (width, height))
    return img

def sort_rects_direct(rects):
    if len(rects) == 0: return []
    rects = sorted(rects, key=lambda r: r[1])
    mean_h = np.mean([r[3] for r in rects])
    lines, curr_line = [], [rects[0]]
    for i in range(1, len(rects)):
        curr, prev = rects[i], rects[i-1]
        if abs(curr[1] - prev[1]) > mean_h * 0.5:
            lines.append(sorted(curr_line, key=lambda r: r[0]))
            curr_line = [curr]
        else:
            curr_line.append(curr)
    lines.append(sorted(curr_line, key=lambda r: r[0]))
    return [r for line in lines for r in line]

# --- 2. HÀM TÁCH KÝ TỰ ---
def find_character(image):
    if image is None: return [], None
    
    # Hàm preprocess trả về (imgGrayscale, imgThresh)
    try:
        gray, binary = PlateProcessor.preprocess(image)
    except Exception as e:
        print(f"Lỗi trong Preprocess.preprocess: {e}")
        return [], None
    # -----------------------------

    # Đảm bảo ảnh binary là nền đen chữ trắng để findContours hoạt động tốt
    if np.sum(binary == 255) > np.sum(binary == 0):
        # Nếu số điểm ảnh trắng nhiều hơn đen -> Có thể là nền trắng chữ đen -> Đảo ngược
        binary = cv2.bitwise_not(binary)

    # Các bước lọc nhiễu bổ sung
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Loại bỏ nhiễu nhỏ
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1) 
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_h, plate_w = binary.shape
    plate_area = plate_h * plate_w
    candidates = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cnt_area = w * h
        ratio = w / float(h) if h > 0 else 0
        area_ratio = cnt_area / plate_area
        height_ratio = h / float(plate_h)
        
        # Bộ lọc kích thước (Heuristic filter)
        check_height = 0.35 < height_ratio < 0.98
        check_ratio = 0.08 < ratio < 1.0
        check_area = 0.005 < area_ratio < 0.2
        
        if check_height and check_ratio and check_area:
            roi = binary[y:y+h, x:x+w]
            white_density = np.sum(roi == 255) / float(roi.size)
            if white_density > 0.15: # Mật độ điểm ảnh chữ phải đủ dày
                candidates.append((x, y, w, h))

    valid_chars = remove_inner_boxes_improved(candidates)
    valid_chars = sort_rects_direct(valid_chars)
    
    # Trả về binary để hàm normalize_char cắt ảnh từ đây
    return valid_chars, binary

def remove_inner_boxes_improved(boxes):
    if not boxes: return []
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j: continue
            (xi, yi, wi, hi) = boxes[i]
            (xj, yj, wj, hj) = boxes[j]
            x_inter = max(0, min(xi + wi, xj + wj) - max(xi, xj))
            y_inter = max(0, min(yi + hi, yj + hj) - max(yi, yj))
            inter_area = x_inter * y_inter
            box_i_area = wi * hi
            if inter_area / float(box_i_area) > 0.9 and box_i_area < (wj * hj):
                keep[i] = False
                break
    return [boxes[i] for i in range(len(boxes)) if keep[i]]

# --- 3. NHẬN DIỆN VÀ CHUẨN HÓA ---
def normalize_char(binary, x, y, w, h, size=28, pad=2):
    roi = binary[y:y+h, x:x+w]
    if roi.size == 0: return None
    scale = (size - 2*pad) / max(roi.shape)
    nw, nh = int(roi.shape[1]*scale), int(roi.shape[0]*scale)
    res = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = res
    return canvas

def predict_single_character(img, model, CLASS_NAMES):
    img_input = np.expand_dims(np.expand_dims(img.astype("float32")/255.0, -1), 0)
    pred = model.predict(img_input, verbose=0)[0]
    return CLASS_NAMES[np.argmax(pred)], np.max(pred) * 100

def train_random_forest(data_dir, save_path="model_rf.pkl"):
    print("[INFO] Bắt đầu chuẩn bị dữ liệu huấn luyện RF...")
    data = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Không tìm thấy thư mục data: {data_dir}")
        return None

    classes = os.listdir(data_dir)
    
    for label in classes:
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path): continue
        
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                img = cv2.resize(img, (28, 28))
                flat_img = img.flatten() 
                data.append(flat_img)
                labels.append(label)

    X = np.array(data) / 255.0 
    y = np.array(labels)
    
    print(f"[INFO] Dữ liệu: {X.shape}, Nhãn: {len(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[INFO] Đang huấn luyện Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    preds = rf_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[INFO] Độ chính xác trên tập test: {acc * 100:.2f}%")
    
    joblib.dump(rf_model, save_path)
    print(f"[INFO] Đã lưu model RF tại: {save_path}")
    return rf_model

def predict_single_char_rf(img, model):
    img_flat = img.flatten().reshape(1, -1) / 255.0
    pred_label = model.predict(img_flat)[0]
    probs = model.predict_proba(img_flat)
    confidence = np.max(probs) * 100
    return pred_label, confidence

def process_batch_rf(input_folder, output_folder, yolo_path, rf_model_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"[INFO] Đang tải model YOLO và Random Forest...")
    try:
        yolo_model = YOLO(yolo_path)
        rf_model = joblib.load(rf_model_path)
    except Exception as e:
        print(f"[ERROR] Lỗi tải model: {e}")
        return

    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in supported_ext]
    
    print(f"[INFO-RF] Tìm thấy {len(files)} ảnh. Đang xử lý bằng Random Forest...")

    for idx, filename in enumerate(files):
        img_path = os.path.join(input_folder, filename)
        try:
            orig = cv2.imread(img_path)
            if orig is None: continue

            # 1. YOLO
            plate_crop = Crop_img(orig, yolo_model)
            if plate_crop is None: continue

            # 2. Warp
            warped = warp_perspective(plate_crop)

            # 3. Segment
            rects, full_bin = find_character(warped)
            if not rects: continue

            # 4. Recognize RF
            res_img = warped.copy()
            plate_text = ""
            
            for x, y, w, h in rects:
                char_img = normalize_char(full_bin, x, y, w, h)
                if char_img is not None:
                    char, conf = predict_single_char_rf(char_img, rf_model)
                    
                    if conf > 40:
                        plate_text += char
                        cv2.rectangle(res_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(res_img, char, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 5. Save
            if len(plate_text) > 2:
                save_path = get_unique_filename(output_folder, plate_text, ".jpg")
                cv2.imwrite(save_path, res_img)
                print(f" [RF] -> {plate_text} | Saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f" -> Lỗi: {e}")

    print("\n[INFO] Hoàn tất xử lý RF.")

# --- 4. HÀM XỬ LÝ FILE VÀ THƯ MỤC ---
def get_unique_filename(directory, base_name, ext=".jpg"):
    filename = f"{base_name}{ext}"
    file_path = os.path.join(directory, filename)
    counter = 0
    while os.path.exists(file_path):
        counter += 1
        suffix = "a" * counter 
        filename = f"{base_name}{suffix}{ext}"
        file_path = os.path.join(directory, filename)
    return file_path

def process_batch(input_folder, output_folder, yolo_path, cnn_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    CLASS_NAMES = list("0123456789ABCDEFGHKLMNOPRSTUVXYZ")
    print(f"[INFO] Đang tải models...")
    try:
        yolo_model = YOLO(yolo_path)
        cnn_model = keras.models.load_model(cnn_path)
    except Exception as e:
        print(f"[ERROR] Không thể tải model: {e}")
        return

    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in supported_ext]
    
    print(f"[INFO] Tìm thấy {len(files)} ảnh trong {input_folder}")

    for idx, filename in enumerate(files):
        img_path = os.path.join(input_folder, filename)
        print(f"\n[{idx+1}/{len(files)}] Đang xử lý: {filename}")
        
        try:
            orig = cv2.imread(img_path)
            if orig is None: continue

            # 1. YOLO
            plate_crop = Crop_img(orig, yolo_model)
            if plate_crop is None:
                print(" -> Không tìm thấy biển số.")
                continue

            # 2. Warp
            warped = warp_perspective(plate_crop)

            # 3. Segment
            rects, full_bin = find_character(warped)
            if not rects:
                print(" -> Không tách được ký tự.")
                continue

            # 4. Recognize CNN
            res_img = warped.copy()
            plate_text = ""
            
            for x, y, w, h in rects:
                char_img = normalize_char(full_bin, x, y, w, h)
                if char_img is not None:
                    char, conf = predict_single_character(char_img, cnn_model, CLASS_NAMES)
                    if conf > 60:
                        plate_text += char
                        cv2.rectangle(res_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(res_img, char, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 5. Save
            if len(plate_text) > 2:
                save_path = get_unique_filename(output_folder, plate_text, ".jpg")
                cv2.imwrite(save_path, res_img)
                print(f" -> Kết quả: {plate_text} | Đã lưu tại: {os.path.basename(save_path)}")
            else:
                print(" -> Ký tự nhận diện quá ít, bỏ qua.")

        except Exception as e:
            print(f" -> Lỗi khi xử lý ảnh này: {e}")

    print("\n[INFO] Hoàn tất xử lý toàn bộ thư mục.")

# --- 5. CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    INPUT_DIR = "D:\\Code\\CDIO3\\test\\images"
    OUTPUT_DIR_CNN = "D:\\Code\\CDIO3\\results"
    OUTPUT_DIR_RF = "D:\\Code\\CDIO3\\result_rf"
    
    YOLO_MODEL = "best.pt"
    CNN_MODEL = "model_cnn_v1.keras"
    RF_MODEL_PATH = "model_rf.pkl"
    DATA_TRAIN_DIR = "dataset" 

    print("1. Chạy nhận diện bằng CNN")
    print("2. Huấn luyện model Random Forest mới")
    print("3. Chạy nhận diện bằng Random Forest (Lưu vào result_rf)")
    choice = input("Nhập lựa chọn (1/2/3): ")

    if choice == '1':
        process_batch(INPUT_DIR, OUTPUT_DIR_CNN, YOLO_MODEL, CNN_MODEL)
        
    elif choice == '2':
        if os.path.exists(DATA_TRAIN_DIR):
            train_random_forest(DATA_TRAIN_DIR, RF_MODEL_PATH)
        else:
            print(f"Lỗi: Không tìm thấy thư mục dữ liệu train tại {DATA_TRAIN_DIR}")
            print("Vui lòng cập nhật biến DATA_TRAIN_DIR trỏ đến thư mục chứa ảnh ký tự.")
            
    elif choice == '3':
        if os.path.exists(RF_MODEL_PATH):
            process_batch_rf(INPUT_DIR, OUTPUT_DIR_RF, YOLO_MODEL, RF_MODEL_PATH)
        else:
            print("Chưa có model RF. Vui lòng chọn mục 2 để huấn luyện trước.")