import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Canvas, Frame
from PIL import Image, ImageTk
from pathlib import Path

from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, cv2, increment_path
)
from utils.torch_utils import select_device
from models.common import DetectMultiBackend

DEFAULT_WEIGHT_PATH = "runs/train/ignition_yolo_final_retrain2/weights/best.pt"
IMG_DISPLAY_SIZE = (800, 600)
ZOOM_STEPS = [0.5, 1.0, 1.5, 2.0]

class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ignition Point Detector [인천소방] v1.6.0")
        self.image_paths = []
        self.image_index = 0
        self.zoom_index = 1

        # 상단 프레임 (로고 + 제목)
        top_frame = tk.Frame(root, bg="white")
        top_frame.pack(side="top", pady=(10, 5))

        try:
            logo_img = Image.open("logo.png").resize((200, 100))  # 세로 크기 늘림
            self.logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(top_frame, image=self.logo_tk, bg="white")
            logo_label.pack(side="left", padx=10)
        except Exception as e:
            print("로고 이미지 로딩 실패:", e)

        title_label = tk.Label(top_frame, text="Ignition Point Detector [인천소방] v1.6.0",
                               bg="white", fg="black", font=("맑은 고딕", 18, "bold"))
        title_label.pack(side="left", padx=10)

        # 중앙 프레임 (이미지 + 스크롤)
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = Canvas(canvas_frame, bg="white", width=850, height=550)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scroll_y = Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        self.scroll_x = Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # 하단 버튼 프레임
        btn_frame = tk.Frame(root, bg="white")
        btn_frame.pack(pady=10)

        btn_style = {"bg": "blue", "fg": "white", "font": ("맑은 고딕", 12, "bold")}
        tk.Button(btn_frame, text="이미지 선택 및 분석", command=self.select_and_detect_images, **btn_style).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="← 이전", command=self.prev_image, **btn_style).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="다음 →", command=self.next_image, **btn_style).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="＋ 확대", command=self.zoom_in, **btn_style).grid(row=0, column=3, padx=5)
        tk.Button(btn_frame, text="－ 축소", command=self.zoom_out, **btn_style).grid(row=0, column=4, padx=5)

    def select_and_detect_images(self):
        file_paths = filedialog.askopenfilenames(title="이미지 선택", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return

        weights = DEFAULT_WEIGHT_PATH if os.path.exists(DEFAULT_WEIGHT_PATH) else filedialog.askopenfilename(
            title="가중치 파일 선택", filetypes=[("Model weights", "*.pt")]
        )
        if not weights or not os.path.exists(weights):
            messagebox.showerror("오류", "가중치 파일을 찾을 수 없습니다.")
            return

        device = select_device('')
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
                            cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                            cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
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
        img = img.resize((int(IMG_DISPLAY_SIZE[0]*zoom), int(IMG_DISPLAY_SIZE[1]*zoom)))
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
    root.geometry("900x750")  # 충분한 공간 확보
    app = FireDetectionApp(root)
    root.mainloop()
