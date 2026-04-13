import cv2
import time
import numpy as np
import keras
from ultralytics import YOLO
from processing import PlateProcessor

MODEL_CNN_PATH = "model_cnn_v2.keras"
MODEL_YOLO_PATH = "best.pt"
VIDEO_PATH = "TesterVideo.mp4"

CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9',
               'A','B','C','D','E','F','G','H','K','L',
               'M','N','P','R','S','T','U','V','X','Y','Z']

CONF_THRES = 0.25
IOU_THRES = 0.45

TRACK_EVERY_N_FRAMES = 2     # track mỗi 2 frame
RECOG_EVERY_N_FRAMES = 6     # mỗi track nhận diện lại mỗi 6 frame (tuỳ máy)
MIN_CROP_SIZE = (40, 14)     # (w,h)

def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def draw_label_below(frame, text, x1, y1, x2, y2, color=(0,255,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    tx = x1
    ty = y2 + th + 8
    if ty + baseline > frame.shape[0]:
        ty = y1 - 8

    xA, yA = max(0, tx), max(0, ty - th - 4)
    xB, yB = min(frame.shape[1]-1, tx + tw + 6), min(frame.shape[0]-1, ty + baseline + 4)

    cv2.rectangle(frame, (xA, yA), (xB, yB), (0,0,0), -1)
    cv2.putText(frame, text, (tx + 3, ty), font, scale, color, thickness, cv2.LINE_AA)

def run():
    print("Loading models...")
    yolo = YOLO(MODEL_YOLO_PATH)
    cnn = keras.models.load_model(MODEL_CNN_PATH)
    processor = PlateProcessor(yolo, cnn, CLASS_NAMES)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video:", VIDEO_PATH)
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cache = {}  # tid -> {"text": str, "conf": float, "last_frame": int}
    last_tracks = []  # list of (tid, x1,y1,x2,y2)

    frame_idx = 0
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Track không nhất thiết chạy mọi frame
        if frame_idx % TRACK_EVERY_N_FRAMES == 0:
            results = yolo.track(frame, persist=True, verbose=False, conf=CONF_THRES, iou=IOU_THRES)
            tracks = []

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxys = boxes.xyxy.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                for i, (x1, y1, x2, y2) in enumerate(xyxys):
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
                    tid = int(ids[i]) if ids is not None else -1
                    tracks.append((tid, x1, y1, x2, y2))

            last_tracks = tracks

        # Vẽ + recognize (CNN) dựa trên bbox đã có
        for (tid, x1, y1, x2, y2) in last_tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            crop = frame[y1:y2, x1:x2]
            cw, ch = (x2 - x1), (y2 - y1)

            if crop is None or crop.size == 0 or cw < MIN_CROP_SIZE[0] or ch < MIN_CROP_SIZE[1]:
                draw_label_below(frame, f"ID:{tid} | small", x1,y1,x2,y2)
                continue

            need_recog = True
            if tid != -1 and tid in cache:
                if (frame_idx - cache[tid]["last_frame"]) < RECOG_EVERY_N_FRAMES:
                    need_recog = False

            if need_recog:
                # QUAN TRỌNG: gọi recognize_plate_crop, không gọi recognize() nữa
                plate_text, confidence, error = processor.recognize_plate_crop(crop)

                if error:
                    label = f"ID:{tid} | {error}"
                    conf = 0.0
                else:
                    label = f"ID:{tid} | {plate_text} ({confidence:.1f}%)"
                    conf = float(confidence)

                if tid != -1:
                    cache[tid] = {"text": label, "conf": conf, "last_frame": frame_idx}
            else:
                label = cache[tid]["text"]

            draw_label_below(frame, label, x1,y1,x2,y2)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - t_prev))
        t_prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Tracking smooth", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()