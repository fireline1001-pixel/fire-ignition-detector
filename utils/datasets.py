
import os
import glob
import torch
import cv2
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

class LoadImages:
    def __init__(self, path, img_size=640, stride=32):
        path = str(Path(path).resolve())
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]
        self.files = [f for f in files if f.split('.')[-1].lower() in IMG_FORMATS]
        self.nf = len(self.files)
        self.img_size = img_size
        self.stride = stride
        self.mode = 'image'
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, f'Image Not Found {path}'

        img = letterbox(img0, new_shape=self.img_size, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        self.count += 1
        return path, img, img0, None

def letterbox(img, new_shape=640, stride=32, auto=True, scaleFill=False, scaleup=True, color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)
