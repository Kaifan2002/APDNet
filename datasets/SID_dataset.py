import os
import cv2
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SID_Dataset_Val(Dataset):
    def __init__(self, normal_img_dir, low_img_dir, transform=None):
        self.normal_img_dir = normal_img_dir
        self.low_img_dir = low_img_dir
        self.normal_image_files = sorted([f for f in os.listdir(normal_img_dir) 
                                        if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.low_image_files = sorted([f for f in os.listdir(low_img_dir) 
                                      if f.endswith(('.jpg', '.png', '.jpeg'))])

        if len(self.normal_image_files) != len(self.low_image_files):
            raise ValueError("The number of images in the normal and low light directories must be the same.")

        self.transform = transform if transform else self._default_transform()

    def __len__(self):
        return len(self.normal_image_files)

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        # 获取文件路径
        normal_img_name = os.path.join(self.normal_img_dir, self.normal_image_files[idx])
        if "normal" in self.normal_image_files[idx]:
            low_img_file = 'low' + self.normal_image_files[idx].replace('normal', '')
        else:
            low_img_file = self.normal_image_files[idx]
        low_img_name = os.path.join(self.low_img_dir, low_img_file)

        normal_image = cv2.imread(normal_img_name, cv2.IMREAD_UNCHANGED)
        low_image = cv2.imread(low_img_name, cv2.IMREAD_UNCHANGED)

        normal_image = transforms.ToTensor()(normal_image)
        low_image = transforms.ToTensor()(low_image)

        return normal_image, low_image
    
class SID_Dataset_Train(Dataset):
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
        self.transform = transform if transform else self._default_transform()

    def __len__(self):
        return len(self.normal_image_files)

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        # 获取文件路径
        normal_img_name = os.path.join(self.normal_img_dir, self.normal_image_files[idx])
        if "normal" in self.normal_image_files[idx]:
            low_img_file = 'low' + self.normal_image_files[idx].replace('normal', '')
        else:
            low_img_file = self.normal_image_files[idx]
        low_img_name = os.path.join(self.low_img_dir, low_img_file)

        normal_image = cv2.imread(normal_img_name, cv2.IMREAD_UNCHANGED)
        low_image = cv2.imread(low_img_name, cv2.IMREAD_UNCHANGED)

        normal_image = transforms.ToTensor()(normal_image)
        low_image = transforms.ToTensor()(low_image)

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
    

