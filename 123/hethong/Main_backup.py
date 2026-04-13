"""
ALPR — Automatic License Plate Recognition
Flask + Flask-SocketIO backend

Routes:
  POST /api/recognize_image   → Ảnh tĩnh (HTTP)
  POST /api/stop              → Dừng quét video
  GET  /api/health            → Health check

SocketIO events (client → server):
  'request_scan'       → Quét 1 frame camera (base64)
  'start_video_scan'   → Quét video file (base64)
  'stop_video'         → Dừng quét video

SocketIO events (server → client):
  'scan_result'        → Kết quả 1 frame
  'scan_no_plate'      → Không tìm thấy
  'scan_error'         → Lỗi
  'plate_found'        → Tìm thấy biển số trong video (real-time)
  'video_progress'     → Tiến độ quét video
  'video_done'         → Quét xong
  'video_stopped'      → Đã dừng
  'video_error'        → Lỗi video

Kỹ thuật Đề 2 (CHỈ video/stream):
  - imgsz=1280, conf=0.35 trong YOLO
  - Grayscale → GaussianBlur → adaptiveThreshold trước OCR

Ảnh tĩnh: KHÔNG thay đổi pipeline OCR, chỉ thêm padding bbox 13%.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import keras
import cv2
import numpy as np
import os
import base64
import threading
from datetime import datetime
from processing import PlateProcessor

# ─── APP ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'alpr_2025_key'
CORS(app, origins="*")
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=50 * 1024 * 1024  # 50 MB — đủ cho 1 frame 4K
)

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

# Stop flag riêng theo session
_session_stop: dict = {}


# ─── HELPER: PADDING + CROP ──────────────────────────────────────────────────

def crop_with_padding(frame, box, pad_ratio=0.13):
    """
    Cắt vùng biển số với padding 13% mỗi chiều.

    Khắc phục: khung YOLO không 'ăn trọn' biển số trên xe màu xanh lá/xanh dương
    dẫn đến mất nét ký tự ở rìa.

    Returns:
        crop        : ảnh vùng biển số đã padding
        padded_box  : (x1p, y1p, x2p, y2p) để vẽ annotation
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    h, w = frame.shape[:2]
    x1p = max(0, x1 - px)
    y1p = max(0, y1 - py)
    x2p = min(w, x2 + px)
    y2p = min(h, y2 + py)
    crop = frame[y1p:y2p, x1p:x2p].copy()
    return crop, (x1p, y1p, x2p, y2p)


# ─── HELPER: YOLO DETECT ─────────────────────────────────────────────────────

def detect_plates_image(frame):
    """YOLO cho ảnh tĩnh — tham số gốc, không thay đổi."""
    results = yolo(frame, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            dets.append((
                r.boxes.xyxy[i].cpu().numpy(),
                float(r.boxes.conf[i].cpu().numpy())
            ))
    dets.sort(key=lambda d: d[1], reverse=True)
    return dets


def detect_plates_video(frame):
    """
    YOLO cho Video/Stream — Đề 2:
      imgsz=1280 để bắt biển số nhỏ ở xa
      conf=0.35  để tăng độ nhạy
    """
    results = yolo(frame, imgsz=1280, conf=0.35, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            dets.append((
                r.boxes.xyxy[i].cpu().numpy(),
                float(r.boxes.conf[i].cpu().numpy())
            ))
    dets.sort(key=lambda d: d[1], reverse=True)
    return dets


# ─── HELPER: ĐỀ 2 PREPROCESSING (CHỈ VIDEO/STREAM) ──────────────────────────

def preprocess_de2(crop_bgr):
    """
    Xử lý ảnh biển số theo kỹ thuật Đề 2.
    CHỈ dùng cho Video/Stream, KHÔNG áp dụng cho ảnh tĩnh.

    Pipeline: Grayscale → GaussianBlur → adaptiveThreshold → BGR (cho CNN)
    Mục đích: tăng độ nét chữ mờ khi biển số ở xa hoặc ánh sáng kém.
    """
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 4
    )
    # 3 channel BGR để tương thích với pipeline CNN hiện tại
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ─── HELPER: ANNOTATION ──────────────────────────────────────────────────────

def draw_annotation(frame, plate_text, confidence, padded_box):
    """Vẽ bounding box đã padding và nhãn lên frame."""
    x1, y1, x2, y2 = padded_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 3)
    # Góc nhấn mạnh
    L = 20
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx + dx*L, cy),  (0, 225, 255), 4)
        cv2.line(frame, (cx, cy), (cx, cy + dy*L),  (0, 225, 255), 4)
    # Label
    label  = f"  {plate_text}  {confidence*100:.2f}%  "
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.6, min(1.3, (x2 - x1) / 180))
    thick  = 2
    (tw, th), bl = cv2.getTextSize(label, font, fscale, thick)
    ly = y1 - 10 if y1 - 10 > th + 10 else y2 + th + 16
    cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw, ly + bl), (0, 255, 80), -1)
    cv2.putText(frame, label, (x1, ly), font, fscale, (0, 0, 0), thick, cv2.LINE_AA)
    return frame


def to_b64(frame, quality=88):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


# ─── HTTP: HEALTH ─────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "models_loaded": True})


# ─── HTTP: STOP ───────────────────────────────────────────────────────────────

@app.route('/api/stop', methods=['POST'])
def api_stop():
    for sid in list(_session_stop.keys()):
        _session_stop[sid] = True
    return jsonify({"status": "stopped"})


# ─── HTTP: ẢNH TĨNH ──────────────────────────────────────────────────────────

@app.route('/api/recognize_image', methods=['POST'])
def api_recognize_image():
    """
    Nhận diện biển số từ ảnh tĩnh.

    Thay đổi DUY NHẤT so với phiên bản gốc:
      → Thêm crop_with_padding(pad_ratio=0.13) trước khi đưa vào OCR.
    Pipeline OCR / AI model KHÔNG thay đổi.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "KHÔNG TÌM THẤY FILE"}), 400

        file    = request.files['file']
        npimg   = np.frombuffer(file.read(), np.uint8)
        img_arr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img_arr is None:
            return jsonify({"success": False, "error": "KHÔNG ĐỌC ĐƯỢC ẢNH"}), 400

        # Phát hiện (tham số YOLO giữ nguyên)
        detections = detect_plates_image(img_arr)
        if not detections:
            return jsonify({"success": False, "error": "KHÔNG PHÁT HIỆN BIỂN SỐ"}), 400

        annotated = img_arr.copy()
        results   = []

        for box, _yc in detections:
            # ── PADDING 13% ──────────────────────────────────────────────────
            crop_padded, padded_box = crop_with_padding(img_arr, box, pad_ratio=0.13)
            if crop_padded.size == 0:
                continue

            # ── OCR (pipeline gốc, KHÔNG thay đổi) ──────────────────────────
            plate_text, conf, err = processor.recognize(crop_padded)
            if err or not plate_text:
                continue

            plate_text = plate_text.upper().strip()
            annotated  = draw_annotation(annotated, plate_text, conf, padded_box)

            fname = f"{plate_text}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
            cv2.imwrite(os.path.join(SAVE_FOLDER, fname), img_arr)

            results.append({
                "license_plate": plate_text,
                "confidence":    float(conf),
                "image_data":    to_b64(annotated),         # frame + annotation
                "plate_crop":    to_b64(crop_padded, 95),   # vùng biển số đã padding
                "timestamp":     datetime.now().strftime('%H:%M:%S'),
            })

        if not results:
            return jsonify({"success": False, "error": "OCR KHÔNG ĐỌC ĐƯỢC KÝ TỰ"}), 400

        return jsonify({"success": True, "results": results})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e).upper()}), 500


# ─── SOCKETIO: EVENTS ────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    _session_stop[request.sid] = False
    print(f"[Socket] Kết nối: {request.sid}")
    emit('connected', {'status': 'ok'})


@socketio.on('disconnect')
def on_disconnect():
    _session_stop[request.sid] = True
    print(f"[Socket] Ngắt kết nối: {request.sid}")


# ─── SOCKETIO: SCAN 1 FRAME (CAMERA / STREAM) ────────────────────────────────

@socketio.on('request_scan')
def on_request_scan(data):
    """
    Frontend gửi 1 frame (base64 JPEG/PNG) → server nhận diện → emit kết quả ngay.

    Payload: { frame_b64: "<base64 string>" }

    Áp dụng: Đề 2 preprocessing + padding bbox 13%.
    Fallback: nếu Đề 2 fail → thử lại với ảnh gốc đã padding.
    """
    try:
        frame_b64 = data.get('frame_b64', '')
        if not frame_b64:
            emit('scan_error', {'message': 'KHÔNG CÓ DỮ LIỆU FRAME'})
            return

        img_bytes = base64.b64decode(frame_b64)
        npimg     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            emit('scan_error', {'message': 'KHÔNG GIẢI MÃ ĐƯỢC FRAME'})
            return

        # Đề 2: imgsz=1280, conf=0.35
        detections = detect_plates_video(frame)
        if not detections:
            emit('scan_no_plate', {'message': 'KHÔNG TÌM THẤY BIỂN SỐ'})
            return

        results   = []
        annotated = frame.copy()

        for box, _yc in detections:
            crop_padded, padded_box = crop_with_padding(frame, box, pad_ratio=0.13)
            if crop_padded.size == 0:
                continue

            # Đề 2 preprocessing
            crop_de2 = preprocess_de2(crop_padded)
            p_text, conf, err = processor.recognize(crop_de2)

            # Fallback → ảnh gốc đã padding
            if err or not p_text or conf < 0.45:
                p_text, conf, err = processor.recognize(crop_padded)
                if err or not p_text:
                    continue

            p_text    = p_text.upper().strip()
            annotated = draw_annotation(annotated, p_text, conf, padded_box)

            fname = f"{p_text}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
            cv2.imwrite(os.path.join(SAVE_FOLDER, fname), frame)

            results.append({
                "license_plate": p_text,
                "confidence":    float(conf),
                "image_data":    to_b64(annotated),
                "plate_crop":    to_b64(crop_padded, 95),
                "timestamp":     datetime.now().strftime('%H:%M:%S'),
            })

        if results:
            emit('scan_result', {
                "success": True,
                "total":   len(results),
                "results": results,
            })
        else:
            emit('scan_no_plate', {'message': 'OCR KHÔNG ĐỌC ĐƯỢC KÝ TỰ'})

    except Exception as e:
        import traceback; traceback.print_exc()
        emit('scan_error', {'message': str(e).upper()})


# ─── SOCKETIO: QUÉT VIDEO FILE ───────────────────────────────────────────────

@socketio.on('start_video_scan')
def on_start_video_scan(data):
    """
    Quét video file real-time.

    Payload: { video_b64: "<base64 string>" }

    Emit:
        'video_progress' — tiến độ
        'plate_found'    — tìm thấy biển số (real-time, ngay lập tức)
        'video_done'     — xong
        'video_stopped'  — bị dừng
        'video_error'    — lỗi

    Đề 2: detect_plates_video + preprocess_de2 + padding 13%.
    Lọc trùng: chỉ emit khi confidence mới > cũ.
    """
    sid = request.sid
    _session_stop[sid] = False

    video_b64 = data.get('video_b64', '')
    if not video_b64:
        emit('video_error', {'message': 'KHÔNG CÓ DỮ LIỆU VIDEO'})
        return

    try:
        video_bytes = base64.b64decode(video_b64)
    except Exception:
        emit('video_error', {'message': 'DỮ LIỆU VIDEO BỊ LỖI'})
        return

    temp_path = f"tmp_{sid}_{datetime.now().strftime('%H%M%S%f')}.mp4"
    with open(temp_path, 'wb') as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.remove(temp_path)
        emit('video_error', {'message': 'KHÔNG MỞ ĐƯỢC VIDEO'})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig     = cap.get(cv2.CAP_PROP_FPS) or 25
    STEP         = max(4, int(fps_orig / 6))
    MAX_SEC      = 180
    max_frames   = min(total_frames, int(fps_orig * MAX_SEC))
    best_plates  = {}   # plate_text → {'confidence': float}

    emit('video_progress', {
        'progress': 0,
        'status':   f'BẮT ĐẦU — {total_frames} FRAMES · STEP {STEP}'
    })

    frame_idx = 0
    while cap.isOpened():
        if _session_stop.get(sid, False):
            emit('video_stopped', {'message': 'ĐÃ DỪNG QUÉT'})
            break

        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        if frame_idx % STEP == 0:
            progress = round(frame_idx / max(max_frames, 1) * 100, 1)

            if frame_idx % (STEP * 20) == 0 and frame_idx > 0:
                emit('video_progress', {
                    'progress': progress,
                    'status':   f'QUÉT {progress:.0f}% — {len(best_plates)} BIỂN SỐ'
                })
                socketio.sleep(0)

            try:
                detections = detect_plates_video(frame)
                for box, _yc in detections:
                    crop_padded, padded_box = crop_with_padding(frame, box, pad_ratio=0.13)
                    if crop_padded.size == 0:
                        continue

                    crop_de2 = preprocess_de2(crop_padded)
                    p_text, conf, err = processor.recognize(crop_de2)

                    if err or not p_text or conf < 0.50:
                        p_text, conf, err = processor.recognize(crop_padded)
                        if err or not p_text or conf < 0.50:
                            continue

                    p_text    = p_text.upper().strip()
                    prev_conf = best_plates.get(p_text, {}).get('confidence', 0.0)

                    if conf > prev_conf:
                        is_update = prev_conf > 0
                        annotated = draw_annotation(frame.copy(), p_text, conf, padded_box)
                        best_plates[p_text] = {'confidence': float(conf)}

                        fname = f"{p_text}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
                        cv2.imwrite(os.path.join(SAVE_FOLDER, fname), frame)

                        emit('plate_found', {
                            'license_plate': p_text,
                            'confidence':    float(conf),
                            'image_data':    to_b64(annotated),
                            'plate_crop':    to_b64(crop_padded, 95),
                            'is_update':     is_update,
                            'progress':      progress,
                            'frame_idx':     frame_idx,
                            'timestamp':     datetime.now().strftime('%H:%M:%S'),
                        })
                        socketio.sleep(0)

            except Exception:
                pass

        frame_idx += 1

    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)

    emit('video_done', {
        'total_plates': len(best_plates),
        'progress':     100,
        'status':       'QUÉT HOÀN TẤT',
    })
    print(f"[Socket] Video done — {len(best_plates)} biển số. SID: {sid}")


@socketio.on('stop_video')
def on_stop_video():
    _session_stop[request.sid] = True
    emit('video_stopped', {'message': 'ĐÃ DỪNG QUÉT'})


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 56)
    print("  ALPR SERVER  →  http://127.0.0.1:5000")
    print("  Flask-SocketIO · threading mode")
    print("  pip install flask-socketio eventlet")
    print("=" * 56)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)