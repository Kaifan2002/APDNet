import argparse
import logging
import math
import time
import torch
import datetime

from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from caculate import *
from losses import My_loss, Vgg_loss
from datasets.LOLv2_dataset import LOLv2Dataset
from datasets.luner_dataset import LunerLowLightDataset
from model.model import light_model
from losses.loss_factory import build_loss_manager_from_yaml

def main(args):
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    loss_manager = build_loss_manager_from_yaml("/disk527/Commondisk/a804_qkf/vscodeproject/LowLight_code/new_model/losses/config/loss.yaml")
    writer = SummaryWriter(log_dir=args.snapshots_folder)
    # 1.加载数据集
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), transforms.Resize((640,640)),
    #                                 transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
    train_transform = transforms.Compose([
        #transforms.Resize((600, 400)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        #transforms.Resize((600, 400)),
        transforms.ToTensor()
    ])
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), transforms.Resize((600,400))])
    train_set = LunerLowLightDataset(args.train_normal_img_dir, args.train_low_img_dir, transform=train_transform)
    val_set = LunerLowLightDataset(args.val_normal_img_dir, args.val_low_img_dir, transform=val_transform)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 2.初始化模型
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = light_model()
    model.to(device = device)
    if args.load_pretrain and os.path.exists(args.pretrain_dir):
        model.load_state_dict(torch.load(args.pretrain_dir))
        print(f"Loaded pretrained weights from {args.pretrain_dir}")
    L1_loss = My_loss.L1_loss()
    L_vgg = Vgg_loss.VGGLoss()
    L_ssim = My_loss.SSIM_loss()
    L_edge = My_loss.EdgeLoss()

    # 3.定义损失函数和优化器
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, dampening=0.5)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lf)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    best_loss = float('inf')
    best_epoch = 0
    total_start = time.time()
    # 4.训练循环
    for epoch in range(args.epochs):
        loss_manager.reset()
        epoch_loss = 0.0
        model.train()
        start = time.time()
        # 训练阶段
        train_loop = tqdm(train_loader, desc=f'训练中 [{epoch+1}/{args.epochs}]',colour='green')
        for batch_idx, (normal_images, low_images) in enumerate(train_loop):
            #数据准备
            normal_images = normal_images.to(device=device)
            low_images = low_images.to(device=device)
            # 前向传播
            enhanced = model(low_images)
            loss = L1_loss(enhanced, normal_images) + L_ssim(enhanced, normal_images) + 0.001*L_vgg(enhanced, normal_images)
            # 反向传播
            optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer.step()

            epoch_loss += loss.item()
            train_loop.set_postfix(train_loss=loss.item())

        loss_manager.log_epoch_losses(epoch+1)
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Train_Loss', avg_loss, epoch)
        scheduler.step() 

        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'验证中 [{epoch+1}/{args.epochs}]')
            for batch_idx, (normal_images, low_images) in enumerate(val_loop):
                normal_images = normal_images.to(device)
                low_images = low_images.to(device)
                enhanced_image = model(low_images)
                outputs = {'enhanced': enhanced_image, 'LLEM_out': LLEM_out}
                targets = {'normal': normal_images}
                loss = loss_manager.compute(outputs, targets, batch_size = 1, batch_info=None)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Val_loss', avg_val_loss, epoch)
        #scheduler.step(avg_val_loss)  # 调整学习率
        #scheduler.step()
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch
            current_time = datetime.datetime.now().strftime("%Y%m%d")
            model_path = os.path.join(args.pth_dir, current_time)
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, f"best_model.pth"))
            print(f"🎯 当前{epoch+1}轮为最佳模型并已保存，验证损失: {best_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % args.snapshot_iter == 0:
            torch.save(model.state_dict(), os.path.join(model_path, f"epoch_{epoch+1}.pth")
            )
            print(f"💾 检查点已保存: epoch {epoch+1}")
    total_end = time.time()
    print(f"训练完成！最佳模型来自第 {best_epoch+1} 轮，验证损失: {best_loss:.4f},总耗时：{(total_end-total_start):.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

	# Input Parameters
    parser.add_argument('--train_normal_img_dir', type=str, default="/disk527/Commondisk/a804_qkf/vscodeproject/data/LOL-v2/Synthetic/Train/Normal")
    parser.add_argument('--train_low_img_dir', type=str, default="/disk527/Commondisk/a804_qkf/vscodeproject/data/LOL-v2/Synthetic/Train/Low")
    parser.add_argument('--val_normal_img_dir', type=str, default="/disk527/Commondisk/a804_qkf/vscodeproject/data/LOL-v2/Synthetic/Test/Normal")
    parser.add_argument('--val_low_img_dir', type=str, default="/disk527/Commondisk/a804_qkf/vscodeproject/data/LOL-v2/Synthetic/Test/Low")
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")
    parser.add_argument('--pth_dir', type=str, default= "snapshots/LOLV2")
    parser.add_argument('--snapshots_folder', type=str, default="logs/", help='模型保存目录')

    args = parser.parse_args() 
     
    print(args)
    
    main(args)