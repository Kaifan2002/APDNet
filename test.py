import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from fvcore.nn import FlopCountAnalysis

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = light_model()
    model.to(device)
    input_image = torch.randn(1, 3, 256, 256) 
    input_image = input_image.to(device)
    out = model(input_image)
    flops = FlopCountAnalysis(model, input_image)
    macs, params = profile(model, (input_image,), verbose=False)

    print(out.shape)
    print("FLOPs:", flops.total()/ 1e9, "G")
    print("MACs   : {:.3f} G".format(macs / 1e9))
    print('params: %.2f M' % (params / 1000000.0))
