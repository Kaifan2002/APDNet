import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from model.arch import *


class light_model(nn.Module):
    def __init__(self):
        super(light_model, self).__init__()
        self.LLEM = LLEM()
        self.DDIIR = DDIIR(inc = 3 , outc = 24, dim = 24, num_blocks = [1,2,2,1])

    def forward(self, x):
        D = self.LLEM(x)
        out = self.DDIIR(x , D)
        return out
    

if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    model = light_model()
    model.to(device)
    input_image = torch.randn(1, 3, 256, 256) 
    input_image = input_image.to(device)
    S_map, D_map, lightmap, out = model(input_image)
    
    flops, params = profile(model, (input_image,))

    print(out.shape)
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
