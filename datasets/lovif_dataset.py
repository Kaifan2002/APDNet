import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class LOLv2Dataset(Dataset):
    def __init__(self, normal_img_dir, low_img_dir, transform: transforms.Compose = None):
        self.normal_img_dir = normal_img_dir
        self.low_img_dir = low_img_dir
        self.normal_image_files = sorted([f for f in os.listdir(normal_img_dir) 
                                        if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.low_image_files = sorted([f for f in os.listdir(low_img_dir) 
                                      if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transform if transform else self._default_transform()
    def __len__(self):
        if len(self.normal_image_files) != len(self.low_image_files):
            raise ValueError("The number of images in the normal and low light directories must be the same.")
        return len(self.normal_image_files)
    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    def default_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    def __getitem__(self, idx):
        normal_img_name = os.path.join(self.normal_img_dir, self.normal_image_files[idx])
        low_img_file = 'LQ' + self.normal_image_files[idx].strip('GT')
        low_img_name = os.path.join(self.low_img_dir, low_img_file)
        normal_image = Image.open(normal_img_name).convert('RGB')  # 转换为 RGB 格式
        low_image = Image.open(low_img_name).convert('RGB')  # 转换为 RGB 格式
        if self.transform:
            normal_image = self.transform(normal_image)
            low_image = self.transform(low_image)
        return normal_image, low_image
    
class LOLv2Dataset_Train(Dataset):
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

        normal_image = Image.open(normal_img_name).convert('RGB')
        low_image = Image.open(low_img_name).convert('RGB')

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
    
def get_transforms(input_size=None):
    """获取常用的图像变换"""
    transform_list = []
    
    if input_size is not None:
        transform_list.append(transforms.Resize(input_size))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transforms.Compose(transform_list)
if __name__ == '__main__':
    # 示例用法
    #import matplotlib.pyplot as plt
    
    # 创建数据集实例
    dataset = LOLv2Dataset(
        '/disk527/Commondisk/a804_qkf/vscodeproject/data/mixed_luner_lowlight/train/normal',
        '/disk527/Commondisk/a804_qkf/vscodeproject/data/mixed_luner_lowlight/train/low',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    )
    
    # 获取一个样本
    low_light, normal_light = dataset[100]

    print(low_light.shape)