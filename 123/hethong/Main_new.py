from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from ultralytics import YOLO
import keras
import cv2
import numpy as np
import os
import base64
import json
import time
import threading
from datetime import datetime
from processing import PlateProcessor

app = Flask(__name__)
CORS(app)

CLASS_NAMES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','K','L',
    'M','N','P','R','S','T','U','V','W','X','Y','Z'
]
ENTRY_FOLDER = "imgcar/entry"
EXIT_FOLDER  = "imgcar/exit"
os.makedirs(ENTRY_FOLDER, exist_ok=True)
os.makedirs(EXIT_FOLDER,  exist_ok=True)

print("--- ĐANG TẢI MODELS AI... ---")
yolo      = YOLO("best.pt")
cnn       = keras.models.load_model("model_cnn_v2.keras")
processor = PlateProcessor(yolo, cnn, CLASS_NAMES)
print("--- MODELS ĐÃ SẴN SÀNG! ---")

# ── Stop flag (thread-safe) ──
_stop_event = threading.Event()


# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────

def detect_and_crop_plate(frame, yolo_model, conf_thresh=0.4, imgsz=1280, pad=12):
    results = yolo_model(frame, conf=conf_thresh, imgsz=imgsz, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            score = float(r.boxes.conf[i].cpu().numpy())
            box   = r.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1p = max(0, x1 - pad);  y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad);  y2p = min(h, y2 + pad)
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            detections.append((crop, (x1, y1, x2, y2), score))
    detections.sort(key=lambda d: d[2], reverse=True)
    return detections


def enhance_crop(crop, zoom_scale=2.5):
    """Phóng to + tăng tương phản nhẹ."""
    cw, ch = crop.shape[1], crop.shape[0]
    zoomed = cv2.resize(crop, (int(cw * zoom_scale), int(ch * zoom_scale)),
                        interpolation=cv2.INTER_CUBIC)
    zoomed = cv2.convertScaleAbs(zoomed, alpha=1.35, beta=15)
    return zoomed


def clahe_enhance(crop):
    """
    Tăng tương phản bằng CLAHE trên kênh L (LAB color space).
    Nhẹ hơn Adaptive Threshold — giữ nguyên cấu trúc ảnh mà CNN đã học.
    Hiệu quả với biển số trên xe màu gây nhiễu (xanh lá, đỏ, vàng...).
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_plate_region(frame, box, pad=6):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1p = max(0, x1 - pad);  y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad);  y2p = min(h, y2 + pad)
    return frame[y1p:y2p, x1p:x2p]


def draw_annotation(frame, plate_text, confidence, box):
    if box is None:
        return frame
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 3)
    L = 18
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx+dx*L, cy), (0, 230, 255), 4)
        cv2.line(frame, (cx, cy), (cx, cy+dy*L), (0, 230, 255), 4)
    label  = f"  {plate_text}  {confidence*100:.2f}%  "
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.6, min(1.3, (x2 - x1) / 180))
    thick  = 2
    (tw, th), bl = cv2.getTextSize(label, font, fscale, thick)
    ly = y1 - 10 if y1 - 10 > th + 10 else y2 + th + 16
    cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw, ly + bl), (0, 255, 80), -1)
    cv2.putText(frame, label, (x1, ly), font, fscale, (0, 0, 0), thick, cv2.LINE_AA)
    return frame


def frame_to_b64(frame, quality=88):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ──────────────────────────────────────────
# STOP
# ──────────────────────────────────────────

@app.route('/api/stop', methods=['POST'])
def api_stop():
    _stop_event.set()
    return jsonify({"status": "stopped"})


# ──────────────────────────────────────────
# ẢNH
# ──────────────────────────────────────────

@app.route('/api/recognize_image', methods=['POST'])
def api_recognize_image():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "KHÔNG TÌM THẤY FILE"}), 400

        file    = request.files['file']
        npimg   = np.frombuffer(file.read(), np.uint8)
        img_arr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img_arr is None:
            return jsonify({"success": False, "error": "KHÔNG ĐỌC ĐƯỢC ẢNH"}), 400

        h_orig, w_orig = img_arr.shape[:2]
        native_imgsz   = max(h_orig, w_orig, 640)

        detections = detect_and_crop_plate(
            img_arr, yolo, conf_thresh=0.35, imgsz=native_imgsz, pad=12
        )

        if not detections:
            return jsonify({"success": False, "error": "KHÔNG PHÁT HIỆN BIỂN SỐ"}), 400

        annotated = img_arr.copy()
        results   = []

        for crop, box, yolo_score in detections:
            enhanced = enhance_crop(crop, zoom_scale=2.5)
            refined  = clahe_enhance(enhanced)

            plate_text, conf, err = processor.recognize(refined)
            if err or not plate_text:
                plate_text, conf, err = processor.recognize(enhanced)
                if err or not plate_text:
                    continue

            plate_text = plate_text.upper().strip()
            annotated  = draw_annotation(annotated, plate_text, conf, box)

            plate_crop     = crop_plate_region(img_arr, box, pad=6)
            plate_crop_b64 = frame_to_b64(plate_crop, quality=95)

            fname = f"{plate_text}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
            cv2.imwrite(os.path.join(ENTRY_FOLDER, fname), img_arr)

            results.append({
                "license_plate": plate_text,
                "confidence":    float(conf),
                "image_data":    frame_to_b64(annotated),
                "plate_crop":    plate_crop_b64,
            })

        if not results:
            return jsonify({"success": False, "error": "OCR KHÔNG ĐỌC ĐƯỢC KÝ TỰ"}), 400

        return jsonify({"success": True, "results": results})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e).upper()}), 500


# ──────────────────────────────────────────
# VIDEO — 1 FRAME/GIÂY + 3 PHÚT QUÉT
# ──────────────────────────────────────────

@app.route('/api/recognize_video', methods=['POST'])
def api_recognize_video():
    if 'file' not in request.files:
        return Response(
            sse({"type": "error", "message": "KHÔNG TÌM THẤY FILE"}),
            mimetype='text/event-stream'
        )

    _stop_event.clear()

    file      = request.files['file']
    temp_path = f"temp_{datetime.now().strftime('%H%M%S%f')}.mp4"
    file.save(temp_path)

    def generate():
        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                yield sse({"type": "error", "message": "KHÔNG MỞ ĐƯỢC VIDEO"})
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_orig     = cap.get(cv2.CAP_PROP_FPS) or 25

            # ────────────────────────────────────────────────────────────────
            # LẤY 1 FRAME/GIÂY
            # Nếu fps=25, STEP=25 → lấy frame 0, 25, 50, 75... (tương ứng 1 frame/giây)
            # ────────────────────────────────────────────────────────────────
            STEP = max(1, int(fps_orig))
            
            # ────────────────────────────────────────────────────────────────
            # QUÉT TỐI ĐA 3 PHÚT (180 giây)
            # Số frame cần xử lý = min(tổng frame, fps * 180)
            # Ví dụ: ở 25fps → min(total_frames, 25*180) = min(total_frames, 4500)
            # ────────────────────────────────────────────────────────────────
            MAX_SCAN_SECONDS = 180
            max_frames_to_scan = int(fps_orig * MAX_SCAN_SECONDS)
            frames_to_process = min(total_frames, max_frames_to_scan)

            MIN_CONF   = 0.50
            ZOOM       = 2.5
            YOLO_CONF  = 0.40
            YOLO_IMGSZ = 1280

            best_plates: dict = {}
            frame_idx = 0
            scanned_count = 0  # số frame đã quét (mỗi STEP frame)

            yield sse({
                "type":              "start",
                "total_frames":      total_frames,
                "fps":               fps_orig,
                "step":              STEP,
                "max_scan_seconds":  MAX_SCAN_SECONDS,
                "frames_to_process": frames_to_process,
            })

            while cap.isOpened() and frame_idx < frames_to_process:
                if _stop_event.is_set():
                    yield sse({"type": "stopped", "message": "ĐÃ DỪNG"})
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                progress = round(frame_idx / max(frames_to_process, 1) * 100, 1)

                # ────────────────────────────────────────────────────────────
                # QUÉT 1 FRAME MỖI STEP (= 1 frame/giây)
                # ────────────────────────────────────────────────────────────
                if frame_idx % STEP == 0:
                    scanned_count += 1
                    try:
                        detections = detect_and_crop_plate(
                            frame, yolo, conf_thresh=YOLO_CONF, imgsz=YOLO_IMGSZ, pad=12
                        )

                        for crop, box, _ in detections:
                            # ────── PREPROCESSING ──────
                            enhanced = enhance_crop(crop, zoom_scale=ZOOM)
                            refined  = clahe_enhance(enhanced)

                            # ────── OCR (không đổi logic) ──────
                            p_text, conf, err = processor.recognize(refined)
                            if err or not p_text or conf < MIN_CONF:
                                p_text, conf, err = processor.recognize(enhanced)
                                if err or not p_text or conf < MIN_CONF:
                                    continue

                            p_text    = p_text.upper().strip()
                            prev_conf = best_plates.get(p_text, {}).get("confidence", 0.0)

                            # ────── CẬP NHẬT BIỂN SỐ NẾU TIN CẬY CAO HƠN ──────
                            if conf > prev_conf:
                                plate_crop     = crop_plate_region(frame, box, pad=6)
                                plate_crop_b64 = frame_to_b64(plate_crop, quality=95)
                                annotated      = draw_annotation(frame.copy(), p_text, conf, box)
                                is_update      = prev_conf > 0

                                best_plates[p_text] = {
                                    "confidence": float(conf),
                                    "frame":      frame.copy(),
                                    "box":        box,
                                    "plate_crop": plate_crop_b64,
                                    "annotated":  frame_to_b64(annotated),
                                }

                                fname = f"{p_text}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
                                cv2.imwrite(os.path.join(ENTRY_FOLDER, fname), frame)

                                # ────── GỬI KẾT QUẢ LÊN CLIENT ──────
                                yield sse({
                                    "type":          "plate",
                                    "license_plate": p_text,
                                    "confidence":    float(conf),
                                    "image_data":    frame_to_b64(annotated),
                                    "plate_crop":    plate_crop_b64,
                                    "progress":      progress,
                                    "frame_idx":     frame_idx,
                                    "scanned_count": scanned_count,
                                    "is_update":     is_update,
                                })

                    except Exception:
                        pass

                # ────── GỬI BẢN CẬP NHẬT TIẾN ĐỘ (mỗi ~5 giây) ──────
                if scanned_count % max(1, 5) == 0 and frame_idx > 0:
                    yield sse({
                        "type":     "progress",
                        "progress": progress,
                        "frame":    frame_idx,
                        "scanned":  scanned_count,
                        "found":    len(best_plates),
                    })

                frame_idx += 1

            cap.release()

            summary = [
                {
                    "license_plate": pt,
                    "confidence":    info["confidence"],
                    "image_data":    info.get("annotated", ""),
                    "plate_crop":    info.get("plate_crop", ""),
                }
                for pt, info in best_plates.items()
            ]
            yield sse({
                "type":         "done",
                "total_plates": len(best_plates),
                "total_scanned": scanned_count,
                "summary":      summary,
            })

        except Exception as e:
            import traceback; traceback.print_exc()
            yield sse({"type": "error", "message": str(e).upper()})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ──────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "models_loaded": True})


if __name__ == '__main__':
    print("SERVER TẠI http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
