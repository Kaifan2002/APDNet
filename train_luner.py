import argparse

import os
import random
import time
import datetime
import yaml
import logging
import wandb

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm

from caculate import *
from losses import My_loss, Vgg_loss
from datasets.LOLv2_dataset import LOLv2Dataset, LOLv2Dataset_Train
from datasets.luner_dataset import LunerLowLightDataset
from model.model import light_model
from utils.distributed_utils import init_distributed_mode, is_main_process, save_on_master, setup_for_distributed
from utils.warmup import CosineAnnealingWarmupRestarts

def postprocess(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1).cpu()  # [C,H,W]
    tensor = tensor.permute(1, 2, 0).numpy()      # [H,W,C]
    tensor = (tensor * 255.0).round().astype('uint8')
    return tensor


def setup_logger(save_dir, name="train_log"):
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_filename, mode='w')
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    logger.info(f"📜 Log file created: {log_filename}")
    return logger

def reduce_value(value, average=True):
    """
    对单个指标进行 all_reduce，保证所有 GPU 得到相同的值
    """
    if not torch.distributed.is_initialized():
        return value
    value = value.clone()
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    if average:
        value /= torch.distributed.get_world_size()
    return value




def build_dataloaders(train_normal_dir, train_low_dir, val_normal_dir, val_low_dir, batch_size, num_workers,  patch_size, distributed=False):
    val_transform = transforms.Compose([transforms.ToTensor()])

    train_set = LOLv2Dataset_Train(train_normal_dir, train_low_dir, patch_size)
    val_set = LunerLowLightDataset(val_normal_dir, val_low_dir, transform=val_transform)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        val_sampler = torch.utils.data.SequentialSampler(val_set)

    train_loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_set, sampler=train_sampler, **train_loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=True, **val_loader_args)

    return train_loader, val_loader, train_sampler, val_sampler


def main(args):

    init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    seed = 42 + (args.rank if hasattr(args, 'rank') else 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    ## Load yaml
    with open(args.yml_path, 'r') as f:
        opt = yaml.safe_load(f)
    Train = opt.get('TRAINING', {})


    patch_sizes = Train.get('PATCH_SIZES', [256])
    batch_sizes = Train.get('BATCH_SIZES', [args.batch_size])
    epochs_per_size = Train.get('EPOCHS_PER_SIZE', [args.epochs])
    num_sizes = len(patch_sizes)

    if not (len(batch_sizes) == num_sizes == len(epochs_per_size)):
        raise ValueError("PATCH_SIZES, BATCH_SIZES and EPOCHS_PER_SIZE must have the same length in yaml.")

    train_normal_dir = Train.get('TRAIN_DIR_NORMAL', args.train_normal_img_dir)
    train_low_dir = Train.get('TRAIN_DIR_LOW', args.train_low_img_dir)
    val_normal_dir = Train.get('VAL_DIR_NORMAL', args.val_normal_img_dir)
    val_low_dir = Train.get('VAL_DIR_LOW', args.val_low_img_dir)
    save_dir_root = Train.get('SAVE_DIR', args.pth_dir)


    # 检查路径创建，只读取一次日期
    current_time = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_path = os.path.join(save_dir_root, current_time)
    model_path = os.path.join(checkpoint_path, 'models')
    log_path = os.path.join(checkpoint_path, 'logs')

    logger = setup_logger(log_path) if is_main_process() else None

    if is_main_process():
        logger.info(f"Loaded training config: {args.yml_path}")
        logger.info(str(Train))
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="kaifan2002-beihang-university",
            # Set the wandb project where this run will be logged.
            project="LLEM",
            # Track hyperparameters and run metadata.
            name="Luner",
            config={
                "learning_rate": 3e-4,
                "architecture": "PGD-Net",
                "dataset": "Luner",
                "epochs": 1000,
            },
            )

    L1_loss = My_loss.L1_loss()
    L_vgg = Vgg_loss.VGGLoss()
    L_ssim = My_loss.SSIM_loss()
    

    if is_main_process():
        writer = SummaryWriter(log_dir=log_path)

    initial_batch = batch_sizes[0] if len(batch_sizes) > 0 else args.batch_size
    initial_patch = patch_sizes[0]
    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        train_normal_dir, train_low_dir, val_normal_dir, val_low_dir,
        batch_size=initial_batch, num_workers=args.num_workers, patch_size=initial_patch, distributed=args.distributed
    )

    model = light_model().to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(
                    optimizer,
                    first_cycle_steps=args.epochs,   # 第一个周期的总步数（通常等于总epoch数，也可以小于）
                    cycle_mult=1.0,                  # 每次周期长度是否放大（1.0 表示固定周期）
                    max_lr=args.lr,                  # 余弦曲线的最高学习率
                    min_lr=args.min_lr,              # 最低学习率
                    warmup_steps=3,                  # 前5个epoch线性warmup
                    gamma=0.9                        # 每个cycle后最大学习率衰减因子
                )


    start_epoch = 0
    if args.load_pretrain and os.path.exists(args.pretrain_dir):
        checkpoint = torch.load(args.pretrain_dir, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint.get('model', checkpoint))
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except Exception:
                print("Warning: failed to load scheduler state_dict (incompatible).")
        start_epoch = checkpoint.get('epoch', 0) + 1
        if is_main_process():
            print(f"🚀 Loaded pretrained weights from {args.pretrain_dir}")

    best_ssim, best_ssim_epoch = 0.0, 0
    best_psnr, best_psnr_epoch = 0.0, 0
    total_start = time.time()

    current_size_index = 0
    current_size_epochs = 0
    patch_size = patch_sizes[0]
    batch_size = batch_sizes[0]

    if is_main_process():
        print('==> Multi-scale Training start with patch_sizes: ', patch_sizes, 'Batchsizes: ', batch_sizes)

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            if isinstance(train_sampler, torch.utils.data.distributed.DistributedSampler):
                train_sampler.set_epoch(epoch)

        # 检查是否需要切换到下一个尺度
        if current_size_epochs >= epochs_per_size[current_size_index]:
            current_size_index += 1
            current_size_epochs = 0
            if current_size_index >= num_sizes:
                if is_main_process():
                    print("✅ All scales have been trained, finishing.")
                break

            patch_size = patch_sizes[current_size_index]
            batch_size = batch_sizes[current_size_index]
            train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
                train_normal_dir, train_low_dir, val_normal_dir, val_low_dir,
                batch_size=batch_size, num_workers=args.num_workers, patch_size=patch_size, distributed=args.distributed
            )
            if is_main_process():
                print(f"--> Switched to scale {current_size_index+1}/{num_sizes}: patch={patch_size}, batch={batch_size}")

        # ----------------- 训练 -----------------
        epoch_loss = 0.0
        model.train()
        start = time.time()

        total_l1 = total_FFT = total_L_ssim = total_vgg = total_light = 0.0

        train_loop = tqdm(train_loader, desc=f'Train [{epoch+1}/{args.epochs}]', disable=not is_main_process(), colour='green')
        for batch_idx, (normal_image, low_image) in enumerate(train_loop):
            
            normal_image = normal_image.to(device)
            low_image = low_image.to(device)
            low_images, normal_images = low_image, normal_image
            enhanced_image= model(low_images)
            vgg_loss = L_vgg(enhanced_image, normal_images)
            L1 = L1_loss(enhanced_image, normal_images) if 'L1_loss' in globals() else My_loss.L1_loss()(enhanced_image, normal_images)
            ssim_loss = L_ssim(enhanced_image, normal_images) if 'L_ssim' in globals() else My_loss.SSIM_loss()(enhanced_image, normal_images)
            loss = L1 + ssim_loss + 0.01 * vgg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            reduced_loss = reduce_value(loss.detach(), average=True)

            epoch_loss += reduced_loss.item()
            total_l1 += L1.item()
            total_L_ssim += ssim_loss.item()
            total_vgg += vgg_loss.item()

            if is_main_process():
                train_loop.set_postfix(train_loss=float(loss.item()))

        end = time.time()
        avg_loss = epoch_loss / max(1, len(train_loader))

        # ----------------- 验证 -----------------
        model.eval()
        epoch_ssim = epoch_psnr = 0.0
        with torch.no_grad():
            for batch_idx, (normal_images, low_images) in enumerate(val_loader):
                normal_images = normal_images.to(device)
                low_images = low_images.to(device)
                enhanced_image = model(low_images)
                ssim = calculate_ssim(postprocess(enhanced_image), postprocess(normal_images))
                psnr = calculate_psnr(postprocess(enhanced_image), postprocess(normal_images))
                ssim = torch.tensor(ssim, device=device)
                psnr = torch.tensor(psnr, device=device)
                reduced_ssim = reduce_value(ssim.detach(), average=True)
                reduced_psnr = reduce_value(psnr.detach(), average=True)
                epoch_ssim += reduced_ssim
                epoch_psnr += reduced_psnr

        avg_ssim = epoch_ssim / max(1, len(val_loader))
        avg_psnr = epoch_psnr / max(1, len(val_loader))
        current_lr = optimizer.param_groups[0]["lr"]
        if is_main_process():
            writer.add_scalar('Train_Loss', avg_loss, epoch)
            writer.add_scalar('Val_SSIM', avg_ssim, epoch)
            writer.add_scalar('Val_PSNR', avg_psnr, epoch)
            logger.info(f"[Epoch {epoch+1:04d}] Loss: {avg_loss:.4f} | SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.4f} | LR: {current_lr:.6f} | Times:{(end-start):.2f}s")

            run.log({
                    'Train_Loss': avg_loss,
                    'Val_SSIM': avg_ssim,
                    'Val_PSNR': avg_psnr
                }, step=epoch)


        if avg_ssim > best_ssim and is_main_process():
            best_ssim = float(avg_ssim)
            best_ssim_epoch = epoch
            os.makedirs(model_path, exist_ok=True)
            torch.save(model_without_ddp.state_dict(), os.path.join(model_path, f"best_ssim_model.pth"))
            logger.info(f"🎯 Best SSIM updated to {best_ssim:.4f} at epoch {epoch+1}")
        if avg_psnr > best_psnr and is_main_process():
            best_psnr = float(avg_psnr)
            best_psnr_epoch = epoch
            os.makedirs(model_path, exist_ok=True)
            torch.save(model_without_ddp.state_dict(), os.path.join(model_path, f"best_psnr_model.pth"))
            logger.info(f"🎯 Best PSNR updated to {best_psnr:.4f} at epoch {epoch+1}")

        if is_main_process() and (epoch + 1) % args.snapshot_iter == 0:
            torch.save(model_without_ddp.state_dict(), os.path.join(model_path, f"epoch_{epoch+1}.pth"))
            logger.info(f"💾 Checkpoint saved: epoch {epoch+1}")

        scheduler.step()
        current_size_epochs += 1
        if is_main_process():
            print(f"-------Best ssim epoch: {best_ssim_epoch+1}, SSIM: {best_ssim:.4f}, Best psnr epoch: {best_psnr_epoch+1}, PSNR: {best_psnr:.4f}")

    total_end = time.time()
    if is_main_process():
        logger.info(f"✅ Training done in {(total_end - total_start)/60:.2f} min |  Best ssim epoch: {best_ssim_epoch+1}, Best SSIM: {best_ssim:.4f} | Best psnr epoch: {best_psnr_epoch+1}, PSNR: {best_psnr:.4f}")
        writer.close()
        run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据路径（仍保留命令行覆盖）
    parser.add_argument('--train_normal_img_dir', type=str, default="/disk527/Commondisk/a804_qkf/vscodeproject/data/LOL-v2/Real_captured/Train/Normal")
    parser.add_argument('--train_low_img_dir', type=str, default="/disk527/Commondisk/a804_qkf/vscodeproject/data/LOL-v2/Real_captured/Train/Low")
    parser.add_argument('--val_normal_img_dir', type=str, default="/disk527/Commondisk/a804_qkf/vscodeproject/data/LOL-v2/Real_captured/Test/Normal")
    parser.add_argument('--val_low_img_dir', type=str, default="/disk527/Commondisk/a804_qkf/vscodeproject/data/LOL-v2/Real_captured/Test/Low")

    # 设备与分布式
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    parser.add_argument('--distributed', action='store_true', help='use distributed training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id for local process')
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # 超参
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--snapshot_iter', type=int, default=100)
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/LOLV2_multi/20250901/best_model.pth")
    parser.add_argument('--pth_dir', type=str, default="snapshots/LOLV2")
    parser.add_argument('--snapshots_folder', type=str, default="logs/", help='模型保存目录')
    
    # LRs
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--lrf', type=float, default=0.01)

    # yaml 多尺度配置
    parser.add_argument('--yml_path', type=str, default="./configs/Luner.yaml")

    args = parser.parse_args()
    print(args)
    main(args)
