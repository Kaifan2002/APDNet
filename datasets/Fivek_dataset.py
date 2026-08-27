import os
import cv2
import torch
import math
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class FivekDataset(Dataset):
    def __init__(self, normal_img_dir, low_img_dir, patch_size=128, transform=None, resize=True, mode="train"):
        self.normal_img_dir = normal_img_dir
        self.low_img_dir = low_img_dir
        self.resize = resize
        self.mode = mode

        self.normal_image_files = sorted([f for f in os.listdir(normal_img_dir) 
                                        if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.low_image_files = sorted([f for f in os.listdir(low_img_dir) 
                                      if f.endswith(('.jpg', '.png', '.jpeg'))])

        if len(self.normal_image_files) != len(self.low_image_files):
            raise ValueError("The number of images in the normal and low light directories must be the same.")
        

        self.image_size_w = 600
        self.image_size_h = 400

        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.normal_image_files)

    def __getitem__(self, idx):
        # 获取文件路径
        normal_img_name = os.path.join(self.normal_img_dir, self.normal_image_files[idx])
        if "normal" in self.normal_image_files[idx]:
            low_img_file = 'low' + self.normal_image_files[idx].replace('normal', '')
        else:
            low_img_file = self.normal_image_files[idx]
        low_img_name = os.path.join(self.low_img_dir, low_img_file)
        
        normal_image = cv2.cvtColor(cv2.imread(normal_img_name), cv2.COLOR_BGR2RGB)
        low_image = cv2.cvtColor(cv2.imread(low_img_name), cv2.COLOR_BGR2RGB)
        
        if low_image.shape[0] >= low_image.shape[1]:
            low_image = cv2.transpose(low_image)
            normal_image = cv2.transpose(normal_image)

        if self.resize:
            normal_image = pad_to_divisible_by_8(normal_image)
            low_image = pad_to_divisible_by_8(low_image)

        normal_image = transforms.ToTensor()(normal_image)
        low_image = transforms.ToTensor()(low_image)

        if self.mode == "train":
            _, h, w = normal_image.shape
            ps = self.patch_size
            if h <= ps or w <= ps:
                normal_image = normal_image
                low_image = low_image
            else:
                rr = random.randint(0, h - ps)
                cc = random.randint(0, w - ps)
                normal_image = normal_image[:, rr:rr + ps, cc:cc + ps]
                low_image = low_image[:, rr:rr + ps, cc:cc + ps]
            # ---- 数据增强 (旋转 & 翻转) ----
            aug = random.randint(0, 8)
            if aug == 1:
                normal_image = normal_image.flip(1)  # 垂直翻转
                low_image = low_image.flip(1)
            elif aug == 2:
                normal_image = normal_image.flip(2)  # 水平翻转
                low_image = low_image.flip(2)
            elif aug == 3:
                normal_image = torch.rot90(normal_image, dims=(1, 2))  # 90°
                low_image = torch.rot90(low_image, dims=(1, 2))
            elif aug == 4:
                normal_image = torch.rot90(normal_image, dims=(1, 2), k=2)  # 180°
                low_image = torch.rot90(low_image, dims=(1, 2), k=2)
            elif aug == 5:
                normal_image = torch.rot90(normal_image, dims=(1, 2), k=3)  # 270°
                low_image = torch.rot90(low_image, dims=(1, 2), k=3)
            elif aug == 6:
                normal_image = torch.rot90(normal_image.flip(1), dims=(1, 2))  # 垂直翻转+90°
                low_image = torch.rot90(low_image.flip(1), dims=(1, 2))
            elif aug == 7:
                normal_image = torch.rot90(normal_image.flip(2), dims=(1, 2))  # 水平翻转+90°
                low_image = torch.rot90(low_image.flip(2), dims=(1, 2))
                
        return normal_image, low_image


def pad_to_divisible_by_8(img, mode='reflect'):
    h, w = img.shape[:2]
    new_h = math.ceil(h / 8) * 8
    new_w = math.ceil(w / 8) * 8

    pad_bottom = new_h - h
    pad_right = new_w - w
    pad_top = pad_left = 0
    if pad_bottom == 0 and pad_right == 0:
        return img
    padded_img = np.pad(img,((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=mode)
    return padded_img

class Fivek_val_Dataset(Dataset):
    def __init__(self, normal_img_dir, low_img_dir, patch_size=128, transform=None):
        self.normal_img_dir = normal_img_dir
        self.low_img_dir = low_img_dir
        self.normal_image_files = sorted([f for f in os.listdir(normal_img_dir) 
                                        if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.low_image_files = sorted([f for f in os.listdir(low_img_dir) 
                                      if f.endswith(('.jpg', '.png', '.jpeg'))])

        if len(self.normal_image_files) != len(self.low_image_files):
            raise ValueError("The number of images in the normal and low light directories must be the same.")

        self.patch_size = patch_size

    def __len__(self):
        return len(self.normal_image_files)

    def __getitem__(self, idx):
        # 获取文件路径
        normal_img_name = os.path.join(self.normal_img_dir, self.normal_image_files[idx])
        if "normal" in self.normal_image_files[idx]:
            low_img_file = 'low' + self.normal_image_files[idx].replace('normal', '')
        else:
            low_img_file = self.normal_image_files[idx]
        low_img_name = os.path.join(self.low_img_dir, low_img_file)

        normal_image = cv2.cvtColor(cv2.imread(normal_img_name), cv2.COLOR_BGR2RGB)
        low_image = cv2.cvtColor(cv2.imread(low_img_name), cv2.COLOR_BGR2RGB)
        if low_image.shape[0] >= low_image.shape[1]:
            low_image = cv2.transpose(low_image)
            normal_image = cv2.transpose(normal_image)
        
        normal_image = pad_to_divisible_by_8(normal_image)
        low_image = pad_to_divisible_by_8(low_image)


        normal_image = transforms.ToTensor()(normal_image)
        low_image = transforms.ToTensor()(low_image)



        return normal_image, low_image