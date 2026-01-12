import os
import tkinter as tk
from pathlib import Path
from tkinter import Canvas, Scrollbar, filedialog, messagebox

import torch
from PIL import Image, ImageTk

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, cv2, increment_path, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# 설정
DEFAULT_WEIGHT_PATH = "runs/train/ignition_yolo_final_retrain2/weights/best.pt"
IMG_DISPLAY_SIZE = (800, 550)
ZOOM_STEPS = [0.5, 1.0, 1.5, 2.0]


class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ignition Point Detector v1.6.0")
        self.root.configure(bg="white")
        self.image_paths = []
        self.image_index = 0
        self.zoom_index = 1

        # 상단 레이아웃 (로고 + 제목 + 심볼)
        top_frame = tk.Frame(root, bg="white")
        top_frame.grid(row=0, column=0, columnspan=6, pady=(10, 5))

        # 로고
        logo_img = Image.open("logo.png").resize((120, 60))
        self.logo_tk = ImageTk.PhotoImage(logo_img)
        tk.Label(top_frame, image=self.logo_tk, bg="white").pack(side="left", padx=5)

        # 제목
        tk.Label(
            top_frame, text="Ignition Point Detector v1.6.0", bg="white", fg="black", font=("맑은 고딕", 20, "bold")
        ).pack(side="left", padx=20)

        # 인천소방 심볼
        simbol_img = Image.open("symbol.png").resize((60, 60))
        self.simbol_tk = ImageTk.PhotoImage(simbol_img)
        tk.Label(top_frame, image=self.simbol_tk, bg="white").pack(side="left", padx=5)

        # 이미지 표시 캔버스 + 스크롤
        self.canvas = Canvas(root, bg="white", width=850, height=550)
        self.canvas.grid(row=1, column=0, columnspan=5, sticky="nsew")

        self.scroll_y = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scroll_y.grid(row=1, column=5, sticky="ns")
        self.scroll_x = Scrollbar(root, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.grid(row=2, column=0, columnspan=5, sticky="ew")

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        # 버튼 레이아웃
        button_frame = tk.Frame(root, bg="white")
        button_frame.grid(row=3, column=0, columnspan=6, pady=(5, 10))

        btn_style = {"bg": "blue", "fg": "white", "font": ("맑은 고딕", 11, "bold"), "width": 15}
        tk.Button(button_frame, text="이미지 선택 및 분석", command=self.select_and_detect_images, **btn_style).pack(
            side="left", padx=5
        )
        tk.Button(button_frame, text="← 이전", command=self.prev_image, **btn_style).pack(side="left", padx=5)
        tk.Button(button_frame, text="다음 →", command=self.next_image, **btn_style).pack(side="left", padx=5)
        tk.Button(button_frame, text="＋ 확대", command=self.zoom_in, **btn_style).pack(side="left", padx=5)
        tk.Button(button_frame, text="－ 축소", command=self.zoom_out, **btn_style).pack(side="left", padx=5)

    def select_and_detect_images(self):
        file_paths = filedialog.askopenfilenames(title="이미지 선택", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return

        weights = (
            DEFAULT_WEIGHT_PATH
            if os.path.exists(DEFAULT_WEIGHT_PATH)
            else filedialog.askopenfilename(title="가중치 파일 선택", filetypes=[("Model weights", "*.pt")])
        )
        if not weights or not os.path.exists(weights):
            messagebox.showerror("오류", "가중치 파일을 찾을 수 없습니다.")
            return

        device = select_device("cpu")
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
