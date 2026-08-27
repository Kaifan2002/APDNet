import yaml
import os
import torch
import cv2
import math
import argparse
import time

import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torch.nn.functional as F
from model.model import light_model
import caculate
from skimage import img_as_ubyte

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
def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
def save_img(filepath, img):
    img = Image.fromarray(img)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    save_kwargs = dict(format="JPEG", quality=96, subsampling=0, optimize=True)
    img.save(filepath, **save_kwargs)
def unpad_image(img, original_shape):
    h, w = original_shape[:2]
    return img[:h, :w]

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = light_model().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with open(args.yml_path, 'r') as f:
        opt = yaml.safe_load(f)
    Train = opt.get('TRAINING', {})
    Model = opt.get('MODEL', {})
    dataset_name = opt.get('Datasets') or Model.get('MODE') or 'Lovif'
    if not isinstance(dataset_name, (str, bytes, os.PathLike)):
        dataset_name = str(dataset_name)
    print("★" * 3,"Loaded training config from:", args.yml_path)
    print("★" * 3,f"Will test the {dataset_name} dataset")
    val_normal_dir = Train.get('VAL_DIR_NORMAL')
    val_low_dir = Train.get('VAL_DIR_LOW')
    
    def preprocess(image):
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(image)
        input_ = img.unsqueeze(0).to(device)
        return input_
    
    def postprocess(tensor):
        tensor = tensor.squeeze(0).clamp(0, 1).cpu()  # [C,H,W]
        tensor = tensor.permute(1, 2, 0).numpy()      # [H,W,C]
        tensor = img_as_ubyte(tensor)
        return tensor

    total_inference_time = 0.0
    if os.path.isfile(val_low_dir):
        file_list = [val_low_dir]
    else:
        file_list = [os.path.join(val_low_dir, f) for f in os.listdir(val_low_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp','.tif'))]
    print("★" * 3, f"发现 {len(file_list)} 张待处理图像")
    output_dir = os.path.join(args.output_dir, dataset_name)
    print("★" * 3,f"输出路径：{output_dir}")
    psnr = []
    ssim = []
    test_loop =tqdm(file_list, desc=f'测试中',colour='green')
    start_time = time.time()
    for batch_idx, file_path in enumerate(test_loop):
        name = file_path.split('/')[-1] 
        if dataset_name =="LOLv2_real":
            normal_img_file = 'normal' + name.strip('low')
        else: normal_img_file = name
        img = load_img(file_path)
        target = load_img(os.path.join(val_normal_dir,normal_img_file))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        with torch.no_grad():
            start_time = time.time()
            original_shape = img.shape
            img = pad_to_divisible_by_8(img)
            input_tensor = preprocess(img)
            
            enhanced_img = model(input_tensor)
            filename = os.path.basename(file_path)
            restored_ = postprocess(enhanced_img)
            restored_ = unpad_image(restored_, original_shape)
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            psnr.append(caculate.calculate_psnr(restored_, target))
            ssim.append(caculate.calculate_ssim(restored_, target))
            save_img(os.path.join(output_dir, filename), restored_)
    psnr = np.mean(np.array(psnr))
    ssim = np.mean(np.array(ssim))
    print("-" * 5, "PSNR: %f " % (psnr))
    print("-" * 5, "SSIM: %f " % (ssim))
    print("-" * 5, f"Total inference time: {total_inference_time:.4f} seconds")
    print("-" * 5, f"Average inference time per image: {total_inference_time/len(file_list):.4f} seconds")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--weights', type=str, default='/home/qiaokaifan/LLIEcode/APDNet-main/snapshots/Lovif/20260618/models/best_psnr_model.pth',help='.pth')
    parser.add_argument('--device', type=str, default='cuda:4', help='device')
    parser.add_argument('--output_dir', type=str, default='./result', help='output_dir')
    parser.add_argument('--yml_path', type=str, default="/home/qiaokaifan/LLIEcode/APDNet-main/configs/Lovif_test.yaml", help='yml_path')
    args = parser.parse_args()
    
    print(args)
    
    main(args)