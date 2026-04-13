import numpy as np
import cv2

class PlateProcessor:
    def __init__(self, yolo_model, cnn_model, class_names, debug=False, debug_wait=1, debug_prefix="DBG"):
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.class_names = class_names

        # Debug flags
        self.debug = debug              # bật/tắt debug
        self.debug_wait = debug_wait    # waitKey(ms). 0 = đứng lại từng bước
        self.debug_prefix = debug_prefix

    # ---------------- DEBUG HELPERS ----------------
    def _dbg_show(self, title, img, resize_w=900):
        """Hiển thị ảnh debug (an toàn nếu img None)."""
        if not self.debug:
            return
        if img is None:
            print(f"[{self.debug_prefix}] {title}: None")
            return
        if img.size == 0:
            print(f"[{self.debug_prefix}] {title}: empty")
            return

        show = img
        if len(show.shape) == 2:
            show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)

        h, w = show.shape[:2]
        if resize_w is not None and w > resize_w:
            scale = resize_w / w
            show = cv2.resize(show, (int(w * scale), int(h * scale)))

        cv2.imshow(f"{self.debug_prefix} - {title}", show)
        cv2.waitKey(self.debug_wait)

    def _dbg_draw_char_boxes(self, plate_bgr, rects, color=(0, 255, 255)):
        """Vẽ box ký tự lên ảnh plate để xem lọc rects đúng chưa."""
        if plate_bgr is None or plate_bgr.size == 0:
            return plate_bgr
        out = plate_bgr.copy()
        for (x, y, w, h) in rects:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        return out

    # ---------------- CORE FUNCTIONS ----------------
    def crop_img(self, img):
        if img is None:
            return None
        result = self.yolo_model(img, verbose=False)
        if len(result[0].boxes) == 0:
            return None
        box = result[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
        return img[y1:y2, x1:x2]

    def warp_perspective(self, img):
        if img is None or img.size == 0:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._dbg_show("warp_gray", gray)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        self._dbg_show("warp_blur", blur)

        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._dbg_show("warp_otsu", binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

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

            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (width, height))
            self._dbg_show("warp_result", warped)
            return warped

        return img

    def remove_inner_boxes(self, boxes):
        keep = [True] * len(boxes)
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                (xi, yi, wi, hi), (xj, yj, wj, hj) = boxes[i], boxes[j]
                inter_area = max(0, min(xi+wi, xj+wj) - max(xi, xj)) * max(0, min(yi+hi, yj+hj) - max(yi, yj))
                if (wi * hi) > 0 and inter_area / (wi*hi) > 0.9 and (wi*hi) < (wj*hj):
                    keep[i] = False
                    break
        return [boxes[i] for i in range(len(boxes)) if keep[i]]

    def sort_rects(self, rects):
        if not rects:
            return []
        rects = sorted(rects, key=lambda r: r[1])
        mean_h = np.mean([r[3] for r in rects]) if rects else 0

        lines, curr_line = [], [rects[0]]
        for i in range(1, len(rects)):
            if abs(rects[i][1] - rects[i-1][1]) > mean_h * 0.5:
                lines.append(sorted(curr_line, key=lambda r: r[0]))
                curr_line = [rects[i]]
            else:
                curr_line.append(rects[i])

        lines.append(sorted(curr_line, key=lambda r: r[0]))
        return [r for line in lines for r in line]

    def find_character(self, image):
        if image is None:
            return [], None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        self._dbg_show("find_gray", gray)

        mean_brightness = np.mean(gray)
        if mean_brightness < 90:
            gamma = 1.5
            table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in range(256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
            self._dbg_show("find_gamma", gray)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        self._dbg_show("find_clahe", gray)

        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        self._dbg_show("find_bilateral", gray)

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 5
        )
        self._dbg_show("find_adapt_inv", binary)

        if np.sum(binary == 255) > np.sum(binary == 0):
            binary = cv2.bitwise_not(binary)
            self._dbg_show("find_inverted", binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plate_h, plate_w = binary.shape
        candidates = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if (0.35 < h/plate_h < 0.98) and (0.08 < w/h < 1.0) and (0.005 < (w*h)/(plate_h*plate_w) < 0.2):
                candidates.append((x, y, w, h))

        valid_chars = self.remove_inner_boxes(candidates)
        sorted_chars = self.sort_rects(valid_chars)

        if self.debug:
            print(f"[{self.debug_prefix}] contours={len(contours)} candidates={len(candidates)} valid={len(valid_chars)} sorted={len(sorted_chars)}")

        return sorted_chars, binary

    def normalize_char(self, binary, x, y, w, h, size=28, pad=2):
        roi = binary[y:y+h, x:x+w]
        if roi.size == 0:
            return None

        scale = (size - 2*pad) / max(roi.shape)
        nw, nh = int(roi.shape[1]*scale), int(roi.shape[0]*scale)

        # chọn interpolation hợp lý
        if nw > roi.shape[1] or nh > roi.shape[0]:
            interp = cv2.INTER_LINEAR
        else:
            interp = cv2.INTER_AREA

        res = cv2.resize(roi, (nw, nh), interpolation=interp)
        canvas = np.zeros((size, size), dtype=np.uint8)
        canvas[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = res
        return canvas

    # ---------------- PUBLIC API ----------------
    def recognize(self, img_array):
        """Cho ảnh/frame: tự crop bằng YOLO rồi mới nhận diện."""
        try:
            self._dbg_show("input_frame", img_array)

            plate_crop = self.crop_img(img_array)
            if plate_crop is None:
                return None, 0.0, "No plate detected"

            self._dbg_show("plate_crop", plate_crop)
            return self.recognize_plate_crop(plate_crop)

        except Exception as e:
            return None, 0.0, str(e)

    def recognize_plate_crop(self, plate_crop):
        """Cho video tracking: nhận diện trực tiếp từ crop, KHÔNG chạy YOLO."""
        try:
            self._dbg_show("plate_crop(input)", plate_crop)

            warped = self.warp_perspective(plate_crop)
            self._dbg_show("warped", warped)

            rects, binary_enhanced = self.find_character(warped)
            self._dbg_show("binary_enhanced", binary_enhanced)

            warped_box = self._dbg_draw_char_boxes(warped if warped is not None else plate_crop, rects)
            self._dbg_show(f"char_boxes(count={len(rects)})", warped_box)

            if not rects:
                return None, 0.0, "No characters found"

            plate_text, confidences = "", []
            for idx, (x, y, w, h) in enumerate(rects):
                char_img = self.normalize_char(binary_enhanced, x, y, w, h)
                if char_img is None:
                    continue

                self._dbg_show(f"char_{idx+1}_28x28", char_img, resize_w=None)

                img_input = np.expand_dims(np.expand_dims(char_img.astype("float32")/255.0, -1), 0)
                pred = self.cnn_model.predict(img_input, verbose=0)[0]
                char = self.class_names[int(np.argmax(pred))]
                conf = float(np.max(pred) * 100)

                plate_text += char if conf > 70 else "?"
                confidences.append(conf)

                if self.debug:
                    print(f"[{self.debug_prefix}] char#{idx+1}: {char} conf={conf:.1f}% box=({x},{y},{w},{h})")

            return plate_text, (float(np.mean(confidences)) if confidences else 0.0), None

        except Exception as e:
            return None, 0.0, str(e)