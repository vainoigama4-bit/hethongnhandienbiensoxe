import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random

class LicensePlateDataGenerator:
    def __init__(self, font_dir='fonts', output_dir='synthetic_dataset'):
        self.font_dir = font_dir
        self.output_dir = output_dir
        self.labels = '0123456789ABCDEFGHKLMNPSTUVXYZ'
        self.size = (32, 64) # Width x Height
        
        self.fonts = self._load_specific_fonts()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for label in self.labels:
            os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    def _load_specific_fonts(self):
        font_files = ['Soxe2banh.ttf']
        loaded_fonts = []
        for f in font_files:
            path = os.path.join(self.font_dir, f)
            if os.path.exists(path):
                for size in [45, 50, 55]:
                    loaded_fonts.append(ImageFont.truetype(path, size))
                print(f"Da load font: {f}")
            else:
                print(f"Canh bao: Khong tim thay {path}")
        return loaded_fonts

    # --- CÁC HÀM MÔ PHỎNG BIẾN DẠNG NÂNG CAO ---

    def simulate_heavy_perspective(self, img):
        """Giả lập ký tự bị méo mó và thu nhỏ mạnh do góc nhìn xa/chéo"""
        h, w = img.shape
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Tạo độ lệch lớn để gây ra hiệu ứng thu nhỏ và méo
        # d_near: lệch ít (phần gần camera), d_far: lệch nhiều (phần xa camera)
        d_far = random.randint(8, 15) 
        d_near = random.randint(0, 5)
        
        # Ngẫu nhiên chọn kiểu méo (chéo trái, chéo phải, hoặc thu nhỏ đỉnh)
        mode = random.choice(['top_small', 'bottom_small', 'left_small', 'right_small'])
        
        if mode == 'top_small':
            pts2 = np.float32([[d_far, d_far], [w-d_far, d_far], [d_near, h-d_near], [w-d_near, h-d_near]])
        elif mode == 'bottom_small':
            pts2 = np.float32([[d_near, d_near], [w-d_near, d_near], [d_far, h-d_far], [w-d_far, h-d_far]])
        elif mode == 'left_small':
            pts2 = np.float32([[d_far, d_near], [w-d_near, d_near], [d_far, h-d_near], [w-d_near, h-d_near]])
        else: # right_small
            pts2 = np.float32([[d_near, d_near], [w-d_far, d_far], [d_near, h-d_near], [w-d_far, h-d_far]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        # borderValue=0 để giữ nền đen
        return cv2.warpPerspective(img, M, (w, h), borderValue=0)

    def simulate_touching(self, img):
        h, w = img.shape
        thickness = random.randint(2, 4)
        side = random.choice(['left', 'right', 'top', 'bottom'])
        if side == 'left': cv2.line(img, (0, 0), (0, h), 255, thickness)
        elif side == 'right': cv2.line(img, (w-1, 0), (w-1, h), 255, thickness)
        elif side == 'top': cv2.line(img, (0, 0), (w, 0), 255, thickness)
        elif side == 'bottom': cv2.line(img, (0, h-1), (w, h-1), 255, thickness)
        return img

    def simulate_broken_char(self, img):
        h, w = img.shape
        num_cuts = random.randint(1, 2)
        for _ in range(num_cuts):
            y_cut = random.randint(15, h-15)
            thickness = random.randint(1, 2)
            cv2.line(img, (0, y_cut), (w, y_cut + random.randint(-3, 3)), 0, thickness)
        return img

    def add_specular_noise(self, img):
        h, w = img.shape
        num_spots = random.randint(15, 30)
        for _ in range(num_spots):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            cv2.circle(img, (x, y), random.randint(1, 2), 255, -1)
        return img

    def random_perspective(self, img):
        h, w = img.shape
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        d = random.randint(2, 6)
        pts2 = np.float32([
            [random.randint(0, d), random.randint(0, d)],
            [w - random.randint(0, d), random.randint(0, d)],
            [random.randint(0, d), h - random.randint(0, d)],
            [w - random.randint(0, d), h - random.randint(0, d)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderValue=0)

    # --- HÀM ADD_NOISE CŨ (TÍCH HỢP BIẾN DẠNG NẶNG) ---

    def add_noise(self, img):
        # 1. Biến dạng hình học: Ưu tiên Perspective méo mó mạnh
        prob_geo = random.random()
        if prob_geo > 0.6:
            img = self.simulate_heavy_perspective(img) # Trường hợp méo mó thu nhỏ
        elif prob_geo > 0.3:
            img = self.random_perspective(img) # Nghiêng nhẹ
        else:
            h, w = img.shape
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h), borderValue=0)

        # 2. Ngẫu nhiên áp dụng các lỗi nhiễu khác
        prob_noise = random.random()
        if prob_noise < 0.2:
            img = self.simulate_touching(img)
        elif prob_noise < 0.4:
            img = self.simulate_broken_char(img)
        elif prob_noise < 0.6:
            img = self.add_specular_noise(img)

        # 3. Làm nhòe và thay đổi độ dày
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        kernel = np.ones((2, 2), np.uint8)
        if random.random() > 0.8:
            img = cv2.dilate(img, kernel, iterations=1) 
        elif random.random() > 0.8:
            img = cv2.erode(img, kernel, iterations=1)

        return img

    def generate(self, samples_per_label=100):
        print(f"Bat dau sinh dataset: {len(self.labels) * samples_per_label} mau...")
        for char in self.labels:
            for i in range(samples_per_label):
                img_pil = Image.new('L', (100, 100), color=0)
                draw = ImageDraw.Draw(img_pil)
                font = random.choice(self.fonts)
                w_txt, h_txt = draw.textbbox((0, 0), char, font=font)[2:]
                draw.text(((100-w_txt)/2, (100-h_txt)/2), char, font=font, fill=255)
                
                img_cv = np.array(img_pil)
                coords = cv2.findNonZero(img_cv)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    img_cv = img_cv[y:y+h, x:x+w]
                
                img_cv = cv2.resize(img_cv, (32, 64), interpolation=cv2.INTER_AREA)
                img_final = self.add_noise(img_cv)
                
                path = os.path.join(self.output_dir, char, f"{char}_{i}.jpg")
                cv2.imwrite(path, img_final)
            print(f"Hoan thanh: {char}")

if __name__ == "__main__":
    gen = LicensePlateDataGenerator(font_dir='fonts', output_dir='synthetic_dataset')
    if len(gen.fonts) > 0:
        gen.generate(samples_per_label=1000)
        print("--- Thanh cong! Dataset da san sang! ---")