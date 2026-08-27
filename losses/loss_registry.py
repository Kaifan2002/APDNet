import torch.nn.functional as F
from losses import My_loss, Vgg_loss

def VGGLoss(outputs, targets):
    loss_vgg = Vgg_loss.VGGLoss()
    return loss_vgg(outputs['enhanced'], targets['normal'])

def EdgeLoss(outputs, targets):
    loss_edge = My_loss.EdgeLoss()
    return  loss_edge(outputs['enhanced'], targets['normal'])

def SSIM_loss(outputs, targets):
    loss_ssim = My_loss.SSIM_loss()
    return  loss_ssim(outputs['enhanced'], targets['normal'])

def L1_loss(outputs, targets):
    loss_L1 = My_loss.L1_loss()
    return  loss_L1(outputs['enhanced'], targets['normal'])

def Light_loss(outputs, targets):
    loss_light = My_loss.L1_loss() 
    return  loss_light(outputs['LLEM_out'], targets['normal'])

# 注册损失函数
LOSS_REGISTRY = {
    'VGGLoss': VGGLoss,
    'EdgeLoss': EdgeLoss,
    'SSIM_loss': SSIM_loss,
    'L1_loss': L1_loss,
    'Light_loss': Light_loss,
}
