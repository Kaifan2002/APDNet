import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ligh_map_loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss = nn.L1Loss()
        self.device = device
    def forward(self, light_map, normal_img):
        loss = self.loss(light_map, normal_img).to(self.device)

        return loss
    
    
class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val, device):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
        self.device = device
        
    def forward(self, x):
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(self.device),2))
        return d
    
class L1_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, light_map, normal_img):
        loss = self.loss(light_map, normal_img)

        return loss



class ColorLoss(nn.Module):
    def __init__(self, device):
        super(ColorLoss, self).__init__()
        self.device = device

    def angle(self, a, b):
        vector = torch.mul(a, b)
        up     = torch.sum(vector)
        down   = torch.sqrt(torch.sum(torch.square(a))) * torch.sqrt(torch.sum(torch.square(b)))
        theta = torch.acos(torch.clamp(up/down, -1.0 + 1e-6, 1.0 - 1e-6))
        return theta

    def forward(self, out_image, gt_image):
        out_image = out_image.to(self.device)
        gt_image = gt_image.to(self.device)

        loss = torch.mean(self.angle(out_image[:,0,:,:],gt_image[:,0,:,:]) + 
                        self.angle(out_image[:,1,:,:],gt_image[:,1,:,:]) +
                        self.angle(out_image[:,2,:,:],gt_image[:,2,:,:])).to(self.device)
        return loss
    
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k.mean()



class ReflectionSeparationLoss(nn.Module):
    def __init__(self, lambda_smooth=0.1, lambda_sparse=0.01, lambda_g=5.0):
        super(ReflectionSeparationLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.lambda_g = lambda_g

    def forward(self, I, D, S):
        """
        I: 输入图像 (RGB), 形状 (B, 3, H, W)
        D: 漫反射输出 (RGB), 形状 (B, 3, H, W)
        S: 镜面反射输出 (RGB), 形状 (B, 3, H, W)
        """
        recon_loss = F.l1_loss(D + S, I)  # 直接比较RGB三通道
        smooth_loss = self.total_variation_loss(D)  # 输入为RGB三通道
        dx_I, dy_I = self.gradient(I)
        dx_D, dy_D = self.gradient(D)
        structure_weight_x = torch.exp(-self.lambda_g * torch.abs(dx_D))
        structure_weight_y = torch.exp(-self.lambda_g * torch.abs(dy_D))
        loss_structure = torch.mean(torch.abs(dx_I * structure_weight_x)) + \
                        torch.mean(torch.abs(dy_I * structure_weight_y))

        total_loss = recon_loss + self.lambda_smooth * smooth_loss + loss_structure 
        return total_loss
    def gradient(self, x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def total_variation_loss(self, x):
        """
        计算RGB三通道的总变分损失（TV Loss）
        x: 输入张量 (B, 3, H, W)
        """
        # 水平方向差分（RGB三通道分别计算）
        diff_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])  # 形状 (B, 3, H, W-1)
        # 垂直方向差分（RGB三通道分别计算）
        diff_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])  # 形状 (B, 3, H-1, W)
        
        # 对RGB三通道和空间维度取平均
        loss_h = torch.mean(diff_h)
        loss_v = torch.mean(diff_v)
        
        return loss_h + loss_v

class SSIM_loss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, channel=1):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channel = channel
        
        # 创建高斯核
        self.gaussian_kernel = self._create_gaussian_kernel(window_size, sigma)
        
    def _create_gaussian_kernel(self, window_size, sigma):
        """创建高斯核"""
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                            for x in range(window_size)])
        gauss = gauss / gauss.sum()
        
        # 创建2D高斯核
        kernel = torch.outer(gauss, gauss)
        return kernel.view(1, 1, window_size, window_size).repeat(self.channel, 1, 1, 1)
    
    def ssim(self, img1, img2):
        """计算SSIM"""
        C1 = (0.01 * self.data_range)**2
        C2 = (0.03 * self.data_range)**2
        
        kernel = self.gaussian_kernel.to(img1.device)
        
        # 计算局部均值
        mu1 = F.conv2d(img1, kernel, padding=0, groups=self.channel)
        mu2 = F.conv2d(img2, kernel, padding=0, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算局部方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=0, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=0, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=0, groups=self.channel) - mu1_mu2
        
        # SSIM公式
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                  ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, img1, img2):
        """
        计算SSIM损失
        参数:
            img1, img2: 输入图像张量 (B, C, H, W)
        返回:
            loss: 1 - SSIM (作为损失函数，越小越好)
        """
        # 检查输入尺寸
        if img1.size() != img2.size():
            raise ValueError(f"Input images must have the same dimensions. Got {img1.size()} and {img2.size()}")
            
        # 自动确定通道数
        if self.channel != img1.shape[1]:
            self.channel = img1.shape[1]
            self.gaussian_kernel = self._create_gaussian_kernel(self.window_size, self.sigma)
        
        # 计算SSIM
        ssim_val = self.ssim(img1, img2)
        
        # 返回损失 (1 - SSIM)
        return 1 - ssim_val
    
class FrequencyLoss(nn.Module):
    '''
    Calculates the amplitude of frequencies loss.
    '''
    def __init__(self, loss_weight = 0.01, criterion ='l1', reduction = 'mean'):
        super(FrequencyLoss, self).__init__()   
        self.loss_weight = loss_weight
        self.reduction = reduction

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        pred_freq = self.get_fft_amplitude(pred)
        target_freq = self.get_fft_amplitude(target)
        
        return self.loss_weight * self.criterion(pred_freq, target_freq)

    def get_fft_amplitude(self, inp):
        
        inp_freq = torch.fft.rfft2(inp, norm='backward')
        amp = torch.abs(inp_freq)
        return amp
    
    
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel 算子（单通道）
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, x, x_hat):
        """
        输入：
            x (Tensor): 真实图像 (B, C, H, W)
            x_hat (Tensor): 预测图像 (B, C, H, W)
        返回：
            loss (Tensor): 边缘损失（标量）
        """
        # 确保 Sobel 算子在正确的设备上
        self.sobel_kernel_x = self.sobel_kernel_x.to(x.device)
        self.sobel_kernel_y = self.sobel_kernel_y.to(x.device)
        
        # 初始化梯度张量
        grad_x = torch.zeros_like(x)
        grad_x_hat = torch.zeros_like(x_hat)
        
        # 对每个通道分别计算梯度
        for c in range(x.shape[1]):  # 遍历通道维度
            # 计算 x 的梯度
            grad_x_c_x = F.conv2d(x[:, c:c+1, :, :], self.sobel_kernel_x, padding=1)
            grad_x_c_y = F.conv2d(x[:, c:c+1, :, :], self.sobel_kernel_y, padding=1)
            grad_x[:, c:c+1, :, :] = torch.sqrt(grad_x_c_x ** 2 + grad_x_c_y ** 2 + 1e-6)

            # 计算 x_hat 的梯度
            grad_x_hat_c_x = F.conv2d(x_hat[:, c:c+1, :, :], self.sobel_kernel_x, padding=1)
            grad_x_hat_c_y = F.conv2d(x_hat[:, c:c+1, :, :], self.sobel_kernel_y, padding=1)
            grad_x_hat[:, c:c+1, :, :] = torch.sqrt(grad_x_hat_c_x ** 2 + grad_x_hat_c_y ** 2 + 1e-6)

        
        # 计算 L2 损失（MSE）
        loss = F.mse_loss(grad_x, grad_x_hat)
        return loss

class ColorConstancyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(ColorConstancyLoss, self).__init__()
        self.weight = weight

    def forward(self, img):
        mean_rgb = torch.mean(img.view(img.shape[0], 3, -1), dim=2)  # shape: [B, 3]
        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        loss = ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()
        return self.weight * loss

class light_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmod = nn.Sigmoid()
    def forward(self, I_high, I_low):
        loss_high_prior = torch.abs(torch.mean(torch.relu(1 - self.sigmod(I_high))) ) # 鼓励高光图亮度更亮
        loss_low_prior  = torch.mean(self.sigmod(I_low))  # 鼓励低光图覆盖基础亮度
        return loss_high_prior+loss_low_prior

if __name__ == '__main__':
    I = torch.randn(1, 1, 640, 640)
    D = torch.randn(1, 1, 640, 640)
    S = torch.randn(1, 1, 640, 640)
    l_rebuild = ReflectionSeparationLoss()
    loss = l_rebuild(I, D, S)
