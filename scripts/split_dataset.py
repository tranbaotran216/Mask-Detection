import os
import shutil
import random

IMG_DIR = "./data/images"
LAB_DIR = "./dataset/yolo_annotations"

OUT_IMG_TRAIN = "./dataset/images/train"
OUT_IMG_VAL = "./dataset/images/val"
OUT_LAB_TRAIN = "./dataset/labels/train"
OUT_LAB_VAL = "./dataset/labels/val"

for d in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LAB_TRAIN, OUT_LAB_VAL]:
    os.makedirs(d, exist_ok=True)

images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(images)

split_idx = int(0.8 * len(images))
train_list = images[:split_idx]
val_list = images[split_idx:]

def move_files(file_list, img_dst, lab_dst):
    for img_file in file_list:
        base = os.path.splitext(img_file)[0]
        txt_file = base + ".txt"

        shutil.copy(os.path.join(IMG_DIR, img_file), os.path.join(img_dst, img_file))
        shutil.copy(os.path.join(LAB_DIR, txt_file), os.path.join(lab_dst, txt_file))

move_files(train_list, OUT_IMG_TRAIN, OUT_LAB_TRAIN)
move_files(val_list, OUT_IMG_VAL, OUT_LAB_VAL)

print("TRAIN:", len(train_list), " | VAL:", len(val_list))
