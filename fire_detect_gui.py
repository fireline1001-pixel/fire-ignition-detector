import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import Canvas, Scrollbar, filedialog, messagebox

import torch
from PIL import Image, ImageTk

# 내부 모듈 import를 위한 경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, cv2, increment_path, non_max_suppression, scale_coords
from utils.torch_utils import select_device

DEFAULT_WEIGHT_PATH = resource_path("best.pt")
IMG_DISPLAY_SIZE = (800, 600)
ZOOM_STEPS = [0.5, 1.0, 1.5, 2.0]


# 리소스 경로 설정 (PyInstaller 환경 고려)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller 임시 디렉토리
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ignition Point Detector")
        self.root.configure(bg="white")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)

        self.image_paths = []
        self.image_index = 0
        self.zoom_index = 1

        # 상단 통합 로고 이미지
        top_frame = tk.Frame(root, bg="white")
        top_frame.pack(pady=10)

        try:
            all_logo_path = resource_path("logoall.jpg")
            all_logo_img = Image.open(all_logo_path).resize((500, 80))
            self.all_logo_tk = ImageTk.PhotoImage(all_logo_img)
            logo_label = tk.Label(top_frame, image=self.all_logo_tk, bg="white")
            logo_label.pack(pady=5)
        except Exception as e:
            print("logoall.jpg 로딩 실패:", e)

        # 이미지 출력 캔버스 + 스크롤바
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self.canvas = Canvas(canvas_frame, bg="white", width=850, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_y = Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.scroll_x = Scrollbar(root, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(fill=tk.X)
        self.canvas.configure(xscrollcommand=self.scroll_x.set)

        # 버튼
        btn_frame = tk.Frame(root, bg="white")
        btn_frame.pack(pady=10)

        btn_style = {"bg": "blue", "fg": "white", "font": ("맑은 고딕", 12, "bold")}
        tk.Button(btn_frame, text="이미지 선택 및 분석", command=self.select_and_detect_images, **btn_style).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="← 이전", command=self.prev_image, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="다음 →", command=self.next_image, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="＋ 확대", command=self.zoom_in, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="－ 축소", command=self.zoom_out, **btn_style).pack(side=tk.LEFT, padx=5)

    def select_and_detect_images(self):
        file_paths = filedialog.askopenfilenames(title="이미지 선택", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return

        # 가중치 자동 지정
        weights = resource_path(DEFAULT_WEIGHT_PATH)
        if not os.path.exists(weights):
            messagebox.showerror("오류", f"가중치 파일을 찾을 수 없습니다:\n{weights}")
            return

        device = select_device("")
        model = DetectMultiBackend(weights, device=device)
        stride, names = model.stride, model.names
        imgsz = check_img_size(640, s=stride)
        save_dir = increment_path(Path("runs/fire_detect"), exist_ok=False)
        os.makedirs(save_dir, exist_ok=True)

        self.image_paths.clear()
        self.zoom_index = 1

        for file_path in file_paths:
            dataset = LoadImages(file_path, img_size=imgsz, stride=stride)
            for path, im, im0s, _ in dataset:
                im = torch.from_numpy(im).to(device)
                im = im.float() / 255.0
                if im.ndimension() == 3:
                    im = im.unsqueeze(0)
                pred = model(im)
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

                for i, det in enumerate(pred):
                    im0 = im0s.copy()
                    if len(det):
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label = f"{names[int(cls)]} {conf:.2f}"
                            cv2.rectangle(
                                im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2
                            )
                            cv2.putText(
                                im0,
                                label,
                                (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )

                    save_path = os.path.join(save_dir, os.path.basename(path))
                    cv2.imwrite(save_path, im0)
                    self.image_paths.append(save_path)

        self.image_index = 0
        if self.image_paths:
            self.show_image()

    def show_image(self):
        if not self.image_paths:
            return
        image_path = self.image_paths[self.image_index]
        img = Image.open(image_path)
        zoom = ZOOM_STEPS[self.zoom_index]
        img = img.resize((int(IMG_DISPLAY_SIZE[0] * zoom), int(IMG_DISPLAY_SIZE[1] * zoom)))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def next_image(self):
        if self.image_paths and self.image_index < len(self.image_paths) - 1:
            self.image_index += 1
            self.show_image()

    def prev_image(self):
        if self.image_paths and self.image_index > 0:
            self.image_index -= 1
            self.show_image()

    def zoom_in(self):
        if self.zoom_index < len(ZOOM_STEPS) - 1:
            self.zoom_index += 1
            self.show_image()

    def zoom_out(self):
        if self.zoom_index > 0:
            self.zoom_index -= 1
            self.show_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = FireDetectionApp(root)
    root.mainloop()
