import os
import random
import shutil

# 설정값
image_dir = 'C:/Users/user/Desktop/image/ignition_dataset/images'
label_dir = 'C:/Users/user/Desktop/image/ignition_dataset/labels'

# train/val 비율 설정
train_ratio = 0.8

# train, val 폴더 생성
for folder in ['train', 'val']:
    os.makedirs(os.path.join(image_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(label_dir, folder), exist_ok=True)

# 이미지 리스트 가져오기
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)

train_count = int(len(images) * train_ratio)
train_images = images[:train_count]
val_images = images[train_count:]

def move_file_safe(src, dst):
    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print(f"[주의] 라벨 파일 없음: {src}")

# 파일 이동
for img_name in train_images:
    shutil.move(os.path.join(image_dir, img_name), os.path.join(image_dir, 'train', img_name))
    label_name = os.path.splitext(img_name)[0] + '.txt'
    move_file_safe(os.path.join(label_dir, label_name), os.path.join(label_dir, 'train', label_name))

for img_name in val_images:
    shutil.move(os.path.join(image_dir, img_name), os.path.join(image_dir, 'val', img_name))
    label_name = os.path.splitext(img_name)[0] + '.txt'
    move_file_safe(os.path.join(label_dir, label_name), os.path.join(label_dir, 'val', label_name))

print('✅ 데이터셋 분할 완료')
