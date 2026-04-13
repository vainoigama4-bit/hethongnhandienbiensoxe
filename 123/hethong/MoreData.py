import albumentations as A
import cv2
import os
import glob
from tqdm import tqdm

# 1. Định nghĩa bộ biến đổi 
transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(std_range=(0.02, 0.05), p=0.3), 
])

def augment_everything(root_input, root_output, num_variants=10):
    # Lấy danh sách tất cả folder con
    subfolders = [f.path for f in os.scandir(root_input) if f.is_dir()]
    
    if not subfolders:
        print("Không tìm thấy folder con nào. Kiểm tra lại root_input!")
        return

    print(f"Tìm thấy {len(subfolders)} lớp ký tự. Bắt đầu tạo ảnh...")

    for folder in subfolders:
        class_name = os.path.basename(folder)
        output_class_path = os.path.join(root_output, class_name)
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        # Quét mọi định dạng ảnh
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            image_files.extend(glob.glob(os.path.join(folder, ext)))

        print(f"Đang xử lý Class [{class_name}] - {len(image_files)} ảnh gốc")

        # Dùng tqdm để hiện thanh tiến trình cho mỗi folder
        for img_path in tqdm(image_files, desc=f"Augmenting {class_name}"):
            image = cv2.imread(img_path)
            if image is None: continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            base_name = os.path.basename(img_path).rsplit('.', 1)[0]

            for i in range(num_variants):
                augmented = transform(image=image)["image"]
                aug_img_name = f"{base_name}_aug_{i}.png"
                
                cv2.imwrite(os.path.join(output_class_path, aug_img_name), 
                            cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))


INPUT_PATH = r"dataset" 
OUTPUT_PATH = r"D:\Code\CDIO3\dataset_augmented"

augment_everything(INPUT_PATH, OUTPUT_PATH, num_variants=15)