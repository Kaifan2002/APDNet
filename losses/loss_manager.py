import torch
from collections import defaultdict

class LossManager:
    def __init__(self):
        self.loss_fns = {}       # 存储损失函数
        self.loss_weights = {}   # 每个损失的权重
        self.loss_values = defaultdict(float)  # 累积损失
        self.total_samples = 0
    def add_loss(self, name, fn, weight = 1.0):
        '''损失函数注册'''
        self.loss_fns[name] = fn
        self.loss_weights[name] = weight
    def reset(self):
        '''重置'''
        self.loss_values = defaultdict(float) 
        self.total_samples = 0 
    def compute(self, output, gt, batch_info=None):
        """
        outputs: dict，包括 model 最终输出和中间层
        targets: dict，包括 ground truth
        batch_info: 可选，提供路径、batch_size 等信息
        """
        total_loss = 0.0
        batch_size = 1
        for name, fn in self.loss_fns.items():
            loss_value = fn(output, gt)
            weighted = loss_value*self.loss_weights[name]
            self.loss_values[name] += loss_value.item() 
            total_loss += weighted
        self.loss_values["total"] += total_loss.item()
        self.total_samples += batch_size
        return total_loss
    def get_avg_losses(self):
        """获取平均损失"""
        return {k: v / self.total_samples for k, v in self.loss_values.items()}
    def log_epoch_losses(self, epoch):
        """打印损失"""
        avg_losses = self.get_avg_losses()
        print(f"[Epoch {epoch}] Losses:")
        for name, val in avg_losses.items():
            print(f"  {name}: {val:.6f}")


        