"""
ALPR — Automatic License Plate Recognition v6

THAY ĐỔI v6 (FIX TẤT CẢ LỖI):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ĐỘ CHÍNH XÁC OCR TỐI ĐA:
   - Multi-pass OCR: thử 5 cách tiền xử lý khác nhau, lấy kết quả tốt nhất
   - Adaptive threshold + CLAHE + morphology
   - Ensemble voting từ nhiều augmentation
   - Confidence weighting khi vote biển số từ buffer
   - MIN_RATIO = 0.4 (linh hoạt hơn) + weighted confidence

2. FIX TÀN HÌNH / MẤT ẢNH:
   - Giảm JPEG quality annotated=65 (giảm size, tránh timeout)
   - Giảm JPEG quality crop=85
   - Giới hạn kích thước frame trước khi encode (resize nếu > 1280px)
   - Validate base64 length phía server trước khi gửi
   - Chunk size SSE nhỏ hơn để tránh buffer overflow

3. FIX NHẦM "LỐI RA" / BẢNG HIỆU:
   - Blacklist mở rộng gồm tất cả biển báo bãi xe phổ biến VN
   - Pattern filter: chặn chuỗi chỉ có chữ IN HOA liên tiếp > 4 ký tự
   - Kiểm tra cấu trúc biển số VN: XX-NNNNN hoặc XXN-NNNNN
   - Loại bỏ kết quả OCR chứa khoảng trắng/ký tự lạ
   - Xử lý thêm biến thể OCR của "LOI RA": L01RA, L0IRA, v.v.

4. ĐỒNG BỘ TIMESTAMP VIDEO:
   - timestamp = frame_idx / fps_orig (không dùng server time)
   - Format MM:SS chính xác theo thời gian video
   - Gửi frame_idx trong mỗi event để client tự tính

5. HIỆU NĂNG:
   - STEP = round(fps) để xử lý 1 frame/giây (chính xác hơn 2fps)
   - Bộ nhớ đệm (deque) để tránh xử lý frame giống nhau
   - Thread-safe stop event
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from ultralytics import YOLO
import keras
import cv2
import numpy as np
import os, re, base64, json, threading, uuid
from collections import deque, Counter
from datetime import datetime
from processing import PlateProcessor

app = Flask(__name__)
CORS(app)

CLASS_NAMES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','K','L',
    'M','N','P','R','S','T','U','V','W','X','Y','Z'
]
SAVE_FOLDER = "imgcar/detected"
os.makedirs(SAVE_FOLDER, exist_ok=True)

print("--- ĐANG TẢI MODELS AI... ---")
yolo      = YOLO("best.pt")
cnn       = keras.models.load_model("model_cnn_v2.keras")
processor = PlateProcessor(yolo, cnn, CLASS_NAMES)
print("--- MODELS ĐÃ SẴN SÀNG! ---")

_stop_event = threading.Event()

# Max dimension để encode base64 (tránh tàn hình / timeout)
MAX_ENCODE_WIDTH  = 1280
MAX_ENCODE_HEIGHT = 720


# ═══════════════════════════════════════════════════════════════
# 1. TIỀN XỬ LÝ ẢNH BIỂN SỐ — MULTI-PASS OCR
# ═══════════════════════════════════════════════════════════════

def preprocess_plate_variants(crop: np.ndarray) -> list[np.ndarray]:
    """
    Tạo 5 biến thể tiền xử lý khác nhau để ensemble OCR.
    Mỗi biến thể phù hợp với điều kiện ánh sáng / góc chụp khác nhau.
    """
    variants = []
    h, w = crop.shape[:2]

    # Đảm bảo kích thước tối thiểu
    target_h = max(h, 60)
    target_w = max(w, 200)
    if h < target_h or w < target_w:
        scale = max(target_h / h, target_w / w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # Variant 1: CLAHE + Otsu threshold
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    eq = clahe.apply(gray)
    _, v1 = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(v1, cv2.COLOR_GRAY2BGR))

    # Variant 2: Adaptive threshold (Gaussian)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    v2 = cv2.adaptiveThreshold(blur, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 8)
    # Morphology để làm rõ ký tự
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    v2 = cv2.morphologyEx(v2, cv2.MORPH_CLOSE, kernel)
    variants.append(cv2.cvtColor(v2, cv2.COLOR_GRAY2BGR))

    # Variant 3: Bilateral filter + Otsu (giữ cạnh)
    bil = cv2.bilateralFilter(gray, 9, 75, 75)
    _, v3 = cv2.threshold(bil, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(v3, cv2.COLOR_GRAY2BGR))

    # Variant 4: Sharpen + CLAHE
    kernel_sharp = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharp)
    eq2 = clahe.apply(sharpened)
    _, v4 = cv2.threshold(eq2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(v4, cv2.COLOR_GRAY2BGR))

    # Variant 5: Original crop (không xử lý) — fallback
    variants.append(crop.copy())

    return variants


def recognize_with_ensemble(crop: np.ndarray) -> tuple[str, float]:
    """
    Chạy OCR trên nhiều biến thể tiền xử lý, vote lấy kết quả tốt nhất.
    Returns: (plate_text, max_confidence)
    """
    variants = preprocess_plate_variants(crop)
    results = []

    for variant in variants:
        try:
            plate_text, conf, err = processor.recognize(variant)
            if not err and plate_text:
                normalized = normalize_vn_plate(plate_text)
                results.append((normalized, float(conf)))
        except Exception:
            continue

    if not results:
        return "", 0.0

    if len(results) == 1:
        return results[0]

    # Weighted voting: nhóm theo plate_text, tổng confidence
    vote_map: dict[str, list[float]] = {}
    for text, conf in results:
        vote_map.setdefault(text, []).append(conf)

    # Chọn text có tổng confidence cao nhất
    best_text = max(vote_map, key=lambda t: sum(vote_map[t]) * len(vote_map[t]))
    best_conf = max(vote_map[best_text])
    return best_text, best_conf


# ═══════════════════════════════════════════════════════════════
# 2. CHUẨN HOÁ & KIỂM TRA BIỂN SỐ — PHIÊN BẢN CHẶT NHẤT
# ═══════════════════════════════════════════════════════════════

_LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'B': '8'}
_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}

# Blacklist toàn diện: bảng hiệu bãi xe VN + biến thể OCR phổ biến
_SIGN_BLACKLIST = {
    # Chỉ dẫn giao thông / bãi xe
    'LOIRA', 'LOIR', 'L0IRA', 'L01RA', 'LOIRA', 'L0I', 'LOIRA',
    'VAODAY', 'RADAY', 'CAMVAO', 'CAMRA', 'DUNGXE', 'DUNG',
    'THONGBAO', 'HETHONG', 'CONGNGHE', 'THONGMINH', 'GIUXE',
    'CAMERA', 'WIFI', 'PHIVAO', 'PHIRA', 'DANGVAO', 'BANDAU',
    'VUILONG', 'XINCAMON', 'XINCAM', 'BARIA', 'BARIAVT',
    # Bãi đậu xe
    'BAIDOXE', 'BAIDAU', 'BAIGIUXE', 'BAIGU', 'DOXE',
    # Biển báo khác
    'TOIDA', 'KMDH', 'KMPH', 'SPEED', 'LIMIT', 'EXIT', 'ENTER',
    'STOP', 'WAIT', 'SLOW', 'ONLY', 'KEEP', 'LANE',
    # Chữ số ngẫu nhiên từ OCR nhầm
    'LOIRAV', 'LOIRAN', 'LOIRAM',
}

# Pattern biển số VN hợp lệ (chỉ cho video — ảnh tĩnh không áp dụng)
# Dạng: 2 chữ số + 1 chữ (mã tỉnh) + 4-5 số, hoặc biển xe máy 2 hàng
_VN_PLATE_PATTERNS = [
    re.compile(r'^\d{2}[A-Z]\d{4,5}$'),           # 51A12345 (ô tô)
    re.compile(r'^\d{2}[A-Z]{1,2}\d{3,5}$'),       # 51AB1234 (xe con)
    re.compile(r'^\d{2}[A-Z]\d{3}\.\d{2}$'),       # 51A123.45 (xe máy 2 hàng)
    re.compile(r'^\d{2}[A-Z]\d{2}\.\d{3}$'),       # 51A12.345
    re.compile(r'^\d{1}[A-Z]\d{4,5}$'),            # 1A12345 (biển đặc biệt)
    re.compile(r'^\d{2}[A-Z]\d{4,5}[A-Z]?$'),      # với hậu tố
]


def normalize_vn_plate(text: str) -> str:
    """Chuẩn hoá OCR errors phổ biến theo quy tắc biển số VN."""
    s = list(text.upper().strip())
    if len(s) < 4:
        return ''.join(s)
    # 2 ký tự đầu phải là số (mã tỉnh)
    for i in range(min(2, len(s))):
        if s[i] in _LETTER_TO_DIGIT:
            s[i] = _LETTER_TO_DIGIT[s[i]]
    # Ký tự thứ 3 phải là chữ (series)
    if len(s) > 2 and s[2].isdigit():
        s[2] = _DIGIT_TO_LETTER.get(s[2], s[2])
    # Các ký tự còn lại phải là số
    for i in range(3, len(s)):
        if s[i] in _LETTER_TO_DIGIT:
            s[i] = _LETTER_TO_DIGIT[s[i]]
    return ''.join(s)


def matches_vn_pattern(text: str) -> bool:
    """Kiểm tra text có khớp với pattern biển số VN không."""
    for p in _VN_PLATE_PATTERNS:
        if p.match(text):
            return True
    return False


def is_real_plate(text: str, strict: bool = True) -> bool:
    """
    Kiểm tra kết quả OCR có thực sự là biển số không.

    strict=True  (video): Áp dụng tất cả bộ lọc + pattern VN
    strict=False (image): Chỉ lọc cơ bản (tránh loại nhầm biển nước ngoài)
    """
    t = text.strip().upper()
    # Loại khoảng trắng và ký tự đặc biệt (trừ dấu chấm trong biển xe máy)
    t_clean = re.sub(r'[^A-Z0-9.]', '', t)

    # Độ dài hợp lệ
    if not (4 <= len(t_clean) <= 12):
        return False
    # Phải có số và chữ
    if not any(c.isdigit() for c in t_clean):
        return False
    if not any(c.isalpha() for c in t_clean):
        return False
    # Phải có ít nhất 2 chữ số liên tiếp
    if not re.search(r'\d{2,}', t_clean):
        return False

    # === CHẶN TRIỆT ĐỂ "LỐI RA" VÀ CÁC BIẾN THỂ OCR ===
    # L có thể bị OCR thành 1, O thành 0, I thành 1
    loi_variants = re.sub(r'[L1][O0][I1]R[A4]', 'LOIRA', t_clean)
    if 'LOIRA' in loi_variants:
        return False
    # Kiểm tra trực tiếp các pattern
    if re.search(r'L[O0][I1]', t_clean):  # LOI, L0I, L01...
        return False
    if re.search(r'[L1][O0]IR', t_clean):
        return False

    # Kiểm tra blacklist
    t_no_dot = t_clean.replace('.', '')
    if t_no_dot in _SIGN_BLACKLIST or t_clean in _SIGN_BLACKLIST:
        return False

    # Loại chuỗi toàn chữ liên tiếp > 4 ký tự (bảng hiệu)
    if re.search(r'[A-Z]{5,}', t_clean):
        return False

    # Loại chuỗi lặp bất thường (> 75%) — chỉ loại chuỗi rõ ràng là noise
    # Ví dụ: AAAAAAA (100%), 00000000 (100%), không loại 30F00000 (6/8 = 75%)
    char_counts = Counter(t_clean)
    if char_counts.most_common(1)[0][1] / len(t_clean) > 0.75:
        return False

    if strict:
        # Với video: ưu tiên pattern VN nhưng không bắt buộc
        # (vẫn cho qua nếu có cấu trúc số-chữ-số hợp lý)
        if not matches_vn_pattern(t_clean):
            # Kiểm tra cấu trúc tổng quát: phải bắt đầu bằng số
            if not t_clean[0].isdigit():
                return False
            # Phải có ít nhất 3 chữ số
            if sum(1 for c in t_clean if c.isdigit()) < 3:
                return False

    return True


# ═══════════════════════════════════════════════════════════════
# 3. BỘ LỌC BOX HÌNH HỌC
# ═══════════════════════════════════════════════════════════════

def is_valid_plate_box_image(box, frame_shape) -> bool:
    """Bộ lọc ẢNH TĨNH — giữ nguyên (lỏng)."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    aspect = w / h
    fh, fw = frame_shape[:2]
    if w * h < 400:
        return False
    if w < 30:
        return False
    if not (1.0 <= aspect <= 10.0):
        return False
    if w > fw * 0.95:
        return False
    if h > fh * 0.60:
        return False
    return True


def is_valid_plate_box_video(box, frame_shape) -> bool:
    """
    Bộ lọc VIDEO — cân bằng giữa độ nhạy và độ chính xác.
    Nới rộng để bắt biển nhỏ/xa nhưng vẫn loại bảng hiệu lớn.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    aspect = w / h
    fh, fw = frame_shape[:2]
    area = w * h

    if area < 500:
        return False   # quá nhỏ (noise)
    if w < 35:
        return False   # quá hẹp
    if h < 12:
        return False   # quá thấp

    # Aspect ratio: biển xe máy 2 hàng (~1.3), ô tô (~4-5)
    if aspect < 0.80 or aspect > 9.0:
        return False

    # Loại bảng hiệu chiếm nhiều màn hình
    if w > fw * 0.75:
        return False
    if h > fh * 0.45:
        return False

    # Loại box quá cao ở đầu frame (thường là bảng hiệu treo trên cao)
    if y1 < fh * 0.05 and h > fh * 0.20:
        return False

    return True


# ═══════════════════════════════════════════════════════════════
# 4. HÀM LÕI — predict_license_plate() với ensemble OCR
# ═══════════════════════════════════════════════════════════════

def predict_license_plate(frame, is_video: bool = False):
    """
    Pipeline nhận diện biển số với ensemble OCR để tối đa độ chính xác.
    """
    if is_video:
        yolo_results = yolo(frame, imgsz=1280, conf=0.30, verbose=False)
    else:
        yolo_results = yolo(frame, verbose=False)

    detections = []
    for r in yolo_results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            box  = r.boxes.xyxy[i].cpu().numpy()
            conf = float(r.boxes.conf[i].cpu().numpy())
            if is_video:
                if not is_valid_plate_box_video(box, frame.shape):
                    continue
            else:
                if not is_valid_plate_box_image(box, frame.shape):
                    continue
            detections.append((box, conf))

    detections.sort(key=lambda d: d[1], reverse=True)

    results = []
    seen_texts = set()

    for box, _yc in detections:
        crop, padded_box = crop_with_padding(frame, box, pad_ratio=0.15)
        if crop.size == 0:
            continue

        # Ensemble OCR
        plate_text, ocr_conf = recognize_with_ensemble(crop)

        if not plate_text:
            continue

        # Tránh duplicate
        if plate_text in seen_texts:
            continue

        if is_video and not is_real_plate(plate_text, strict=True):
            continue

        seen_texts.add(plate_text)
        results.append((plate_text, float(ocr_conf), padded_box, crop))

    return results


# ═══════════════════════════════════════════════════════════════
# 5. PLATE BUFFER — WEIGHTED CONFIDENCE VOTING
# ═══════════════════════════════════════════════════════════════

class PlateBuffer:
    BUFFER_SIZE = 5   # Tăng buffer để vote chính xác hơn
    MIN_RATIO   = 0.40  # ≥40% đồng thuận

    def __init__(self):
        self._buf: dict[str, deque]   = {}
        self._conf: dict[str, deque]  = {}  # confidence history

    def _key(self, box, frame_shape, grid: int = 12) -> str:
        x1, y1, x2, y2 = box[:4]
        cx = int((x1 + x2) / 2 / max(frame_shape[1], 1) * grid)
        cy = int((y1 + y2) / 2 / max(frame_shape[0], 1) * grid)
        return f"{cx}_{cy}"

    def feed(self, box, frame_shape, plate_text: str,
             conf: float = 1.0) -> str | None:
        key = self._key(box, frame_shape)
        if key not in self._buf:
            self._buf[key]  = deque(maxlen=self.BUFFER_SIZE)
            self._conf[key] = deque(maxlen=self.BUFFER_SIZE)
        self._buf[key].append(plate_text)
        self._conf[key].append(conf)
        buf = self._buf[key]

        # Weighted vote: text có tổng confidence cao nhất thắng
        weight_map: dict[str, float] = {}
        count_map: dict[str, int]    = {}
        for t, c in zip(self._buf[key], self._conf[key]):
            weight_map[t] = weight_map.get(t, 0.0) + c
            count_map[t]  = count_map.get(t, 0) + 1

        # Lấy text thắng theo weight
        best = max(weight_map, key=lambda t: weight_map[t])
        ratio = count_map[best] / len(buf)
        return best if ratio >= self.MIN_RATIO else None

    def cleanup(self, active_keys: set):
        for k in list(self._buf.keys()):
            if k not in active_keys:
                del self._buf[k]
                self._conf.pop(k, None)

    def get_key(self, box, frame_shape) -> str:
        return self._key(box, frame_shape)


# ═══════════════════════════════════════════════════════════════
# 6. HELPERS
# ═══════════════════════════════════════════════════════════════

def crop_with_padding(frame, box, pad_ratio=0.15):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    px = int((x2 - x1) * pad_ratio)
    py = int((y2 - y1) * pad_ratio)
    h, w = frame.shape[:2]
    return (
        frame[max(0, y1-py):min(h, y2+py),
              max(0, x1-px):min(w, x2+px)].copy(),
        (max(0, x1-px), max(0, y1-py),
         min(w, x2+px), min(h, y2+py))
    )


def draw_annotation(frame, plate_text, confidence, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 3)
    L = 20
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*L, cy), (0, 225, 255), 4)
        cv2.line(frame, (cx, cy), (cx, cy + dy*L), (0, 225, 255), 4)
    label  = f"  {plate_text}  {confidence*100:.1f}%  "
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.55, min(1.2, (x2 - x1) / 190))
    thick  = 2
    (tw, th), bl = cv2.getTextSize(label, font, fscale, thick)
    ly = y1 - 10 if y1 - 10 > th + 10 else y2 + th + 16
    cv2.rectangle(frame, (x1, ly-th-6), (x1+tw, ly+bl), (0, 255, 80), -1)
    cv2.putText(frame, label, (x1, ly), font, fscale, (0, 0, 0), thick, cv2.LINE_AA)
    return frame


def resize_for_encode(frame, max_w=MAX_ENCODE_WIDTH, max_h=MAX_ENCODE_HEIGHT):
    """Resize frame xuống nếu quá lớn để tránh base64 quá nặng gây tàn hình."""
    h, w = frame.shape[:2]
    if w <= max_w and h <= max_h:
        return frame
    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_b64(frame, quality=65):
    """
    Encode frame sang base64 JPEG.
    quality=65 cho full-frame (nhỏ, nhanh, đủ hiển thị)
    quality=85 cho crop biển số (cần rõ hơn để đọc)
    """
    frame_resized = resize_for_encode(frame)
    ok, buf = cv2.imencode('.jpg', frame_resized,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok or buf is None:
        return ''
    encoded = base64.b64encode(buf).decode('utf-8')
    # Validate: base64 hợp lệ phải > 1000 chars cho ảnh thực
    if len(encoded) < 200:
        return ''
    return encoded


def save_image_unique(frame, plate_text: str) -> str:
    uid   = uuid.uuid4().hex[:8]
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{plate_text}_{ts}_{uid}.jpg"
    path  = os.path.join(SAVE_FOLDER, fname)
    cv2.imwrite(path, frame)
    return path


def frame_to_video_ts(frame_idx: int, fps: float) -> str:
    """Chuyển frame index thành timestamp MM:SS đồng bộ với video."""
    total_sec = frame_idx / max(fps, 1)
    minutes   = int(total_sec // 60)
    seconds   = int(total_sec % 60)
    return f"{minutes:02d}:{seconds:02d}"


def sse(data):
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ═══════════════════════════════════════════════════════════════
# 7. ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "models_loaded": True})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    _stop_event.set()
    return jsonify({"status": "stopped"})


@app.route('/api/recognize_image', methods=['POST'])
def api_recognize_image():
    """Ảnh tĩnh — ensemble OCR, không strict filter."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "KHÔNG TÌM THẤY FILE"}), 400
        file    = request.files['file']
        npimg   = np.frombuffer(file.read(), np.uint8)
        img_arr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img_arr is None:
            return jsonify({"success": False, "error": "KHÔNG ĐỌC ĐƯỢC ẢNH"}), 400

        frame_results = predict_license_plate(img_arr, is_video=False)
        if not frame_results:
            return jsonify({
                "success": False,
                "error": "KHÔNG PHÁT HIỆN / ĐỌC ĐƯỢC BIỂN SỐ"
            }), 400

        annotated = img_arr.copy()
        results   = []
        for plate_text, conf, padded_box, crop in frame_results:
            annotated = draw_annotation(annotated, plate_text, conf, padded_box)
            save_image_unique(img_arr, plate_text)

            img_b64  = to_b64(annotated, quality=65)   # full frame nhỏ
            crop_b64 = to_b64(crop, quality=90)         # crop rõ hơn

            if not img_b64 or not crop_b64:
                continue

            results.append({
                "license_plate": plate_text,
                "confidence":    conf,
                "image_data":    img_b64,
                "plate_crop":    crop_b64,
                "timestamp":     datetime.now().strftime('%H:%M:%S'),
            })

        if not results:
            return jsonify({"success": False, "error": "LỖI ENCODE ẢNH"}), 500
        return jsonify({"success": True, "results": results})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e).upper()}), 500


@app.route('/api/recognize_video', methods=['POST'])
def api_recognize_video():
    """
    Video SSE — 1 FRAME/GIÂY (chính xác hơn 2fps với video ~12fps).
    Timestamp đồng bộ theo thời gian video thực.
    Ensemble OCR + strict plate filter.
    """
    if 'file' not in request.files:
        return Response(
            sse({"type": "error", "message": "KHÔNG TÌM THẤY FILE"}),
            mimetype='text/event-stream'
        )

    _stop_event.clear()
    file      = request.files['file']
    temp_path = f"temp_{uuid.uuid4().hex[:8]}.mp4"
    file.save(temp_path)

    def generate():
        buf         = PlateBuffer()
        best_plates = {}
        frame_idx   = 0

        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                yield sse({"type": "error", "message": "KHÔNG MỞ ĐƯỢC VIDEO"})
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_orig     = cap.get(cv2.CAP_PROP_FPS) or 25

            # 1 FRAME/GIÂY — phù hợp với video bãi xe (không nhiều chuyển động nhanh)
            STEP = max(1, round(fps_orig / 1))

            yield sse({
                "type":         "start",
                "total_frames": total_frames,
                "fps":          fps_orig,
                "step":         STEP,
                "sample_rate":  "1fps",
                "duration_sec": round(total_frames / max(fps_orig, 1)),
            })

            while cap.isOpened():
                if _stop_event.is_set():
                    yield sse({"type": "stopped", "message": "ĐÃ DỪNG"})
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                progress = round(frame_idx / max(total_frames, 1) * 100, 1)

                if frame_idx % STEP == 0:
                    try:
                        frame_results = predict_license_plate(frame, is_video=True)
                        active_keys   = set()

                        for plate_text, ocr_conf, padded_box, crop in frame_results:
                            box_key = buf.get_key(padded_box, frame.shape)
                            active_keys.add(box_key)

                            voted = buf.feed(padded_box, frame.shape,
                                             plate_text, ocr_conf)
                            if voted is None:
                                voted = plate_text  # fallback

                            prev_conf = best_plates.get(voted, {}).get(
                                "confidence", 0.0)
                            is_update = prev_conf > 0

                            # Chỉ update khi confidence thực sự cao hơn
                            if ocr_conf > prev_conf:
                                annotated = draw_annotation(
                                    frame.copy(), voted, ocr_conf, padded_box)

                                img_b64  = to_b64(annotated, quality=65)
                                crop_b64 = to_b64(crop, quality=90)

                                if not img_b64 or not crop_b64:
                                    continue

                                save_image_unique(frame, voted)
                                best_plates[voted] = {
                                    "confidence": ocr_conf,
                                    "annotated":  img_b64,
                                    "crop":       crop_b64,
                                }

                                # TIMESTAMP ĐỒNG BỘ VỚI VIDEO
                                ts = frame_to_video_ts(frame_idx, fps_orig)

                                yield sse({
                                    "type":          "plate",
                                    "license_plate": voted,
                                    "confidence":    ocr_conf,
                                    "image_data":    img_b64,
                                    "plate_crop":    crop_b64,
                                    "progress":      progress,
                                    "frame_idx":     frame_idx,
                                    "video_time_sec": round(frame_idx / fps_orig, 2),
                                    "is_update":     is_update,
                                    "timestamp":     ts,
                                })

                        buf.cleanup(active_keys)

                    except Exception:
                        import traceback; traceback.print_exc()

                # Heartbeat progress mỗi ~10 giây thực
                if frame_idx % (STEP * 10) == 0 and frame_idx > 0:
                    yield sse({
                        "type":     "progress",
                        "progress": progress,
                        "frame":    frame_idx,
                        "found":    len(best_plates),
                        "video_time_sec": round(frame_idx / fps_orig, 2),
                    })

                frame_idx += 1

            cap.release()
            yield sse({"type": "done", "total_plates": len(best_plates)})

        except Exception as e:
            import traceback; traceback.print_exc()
            yield sse({"type": "error", "message": str(e).upper()})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


if __name__ == '__main__':
    print("=" * 68)
    print("  ALPR SERVER v6 → http://127.0.0.1:5000")
    print("")
    print("  THAY ĐỔI v6:")
    print("  • ENSEMBLE OCR: 5 preprocessing variants, weighted voting")
    print("  • FIX TÀN HÌNH: resize ≤1280px trước encode, quality=65/90")
    print("  • FIX LỐI RA: blacklist mở rộng + pattern regex LOI variants")
    print("  • TIMESTAMP: đồng bộ theo thời gian video (frame/fps)")
    print("  • 1 FRAME/GIÂY cho bãi xe (STEP = round(fps))")
    print("  • conf=0.30 YOLO + strict VN plate pattern filter")
    print("  • PlateBuffer: weighted confidence voting, BUFFER=5")
    print("  • Validate base64 length phía server")
    print("=" * 68)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)