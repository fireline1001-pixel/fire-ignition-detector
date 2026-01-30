from pathlib import Path

import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, cv2, increment_path, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# 모델 경로와 이미지 폴더 경로 설정
weights = "C:/Users/user/yolov5/runs/train/ignition_yolo_final_retrain2/weights/best.pt"
source = "C:/Users/user/Desktop/image/test_images"
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45

# 장치 설정
device = select_device("")
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)

# 이미지 불러오기
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

# 저장 경로 설정
save_dir = increment_path(Path("runs/detect/fire_detect_result"), exist_ok=False)
(save_dir / "labels").mkdir(parents=True, exist_ok=True)

# 추론 시작
for path, img, im0s, vid_cap, s in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    for i, det in enumerate(pred):
        im0 = im0s.copy()

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                cv2.putText(
                    im0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

        save_path = str(save_dir / Path(path).name)
        cv2.imwrite(save_path, im0)

print(f"결과 이미지가 다음 경로에 저장되었습니다: {save_dir}")
