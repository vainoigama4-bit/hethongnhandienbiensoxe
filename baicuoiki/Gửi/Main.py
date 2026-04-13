from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import keras
import cv2
import numpy as np
import base64
import osz
from datetime import datetime
import traceback
from processing import PlateProcessor

app = Flask(__name__)
CORS(app)

# Cấu hình
CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z']
ENTRY_FOLDER = "imgcar/entry"
EXIT_FOLDER = "imgcar/exit"
os.makedirs(ENTRY_FOLDER, exist_ok=True)
os.makedirs(EXIT_FOLDER, exist_ok=True)

# Khởi tạo models và Processor
print("--- Loading Models ---")
yolo = YOLO("best.pt")
cnn = keras.models.load_model("model_cnn_v1.keras")
processor = PlateProcessor(yolo, cnn, CLASS_NAMES)
print("--- Models Ready ---")

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    try:
        gate = request.form.get('gate', 'entry')
        save_folder = ENTRY_FOLDER if gate == 'entry' else EXIT_FOLDER
        img_array = None

        # Xử lý các dạng input
        if 'image' in request.files:
            npimg = np.frombuffer(request.files['image'].read(), np.uint8)
            img_array = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        elif 'image_base64' in request.form:
            img_bytes = base64.b64decode(request.form['image_base64'])
            npimg = np.frombuffer(img_bytes, np.uint8)
            img_array = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img_array is None:
            return jsonify({"success": False, "error": "No image"}), 400

        # Gọi xử lý từ processor
        plate_text, confidence, error = processor.recognize(img_array)

        if error:
            return jsonify({"success": False, "error": error}), 400

        # Lưu ảnh
        fname = f"{plate_text}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(save_folder, fname)
        cv2.imwrite(save_path, img_array)

        return jsonify({
            "success": True,
            "license_plate": plate_text,
            "confidence": float(confidence),
            "image_path": save_path,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)