import torch
from model import model
from model.arch import LLEM,DDIIR
class FeatureExtractor:
    def __init__(self, model, target_layer_types = ('LLEM', 'DDIIR'), name_filter =  None):
        '''
        :param model: 被分析的模型
        :param target_layer_types: 要 hook 的模块类型（如 nn.Conv2d, nn.ReLU）
        :param name_filter: 可选字符串，只 hook 名称中包含该字符串的层
        '''
        self.model = model
        self.target_layer_types = target_layer_types
        self.name_filter = name_filter
        self.outputs = {}
        self.handles = []

    def _hook_fn(self, layer_name):
        '''
        hook函数-用于记录输出
        '''
        def hook(module, input, output):
            # self.outputs[layer_name] = [item.detach().cpu() for item in output]
            self.outputs[layer_name] = output.detach().cpu()
        return hook

    def register_hooks(self):
        """
        遍历模型所有子模块，注册 hook
        """
        for name, module in self.model.named_modules():
            # for item in self.target_layer_types:
            #     if isinstance(module, item):
            #         handle = module.register_forward_hook(self._hook_fn(name))
            #         self.handles.append(handle)
            if name in self.target_layer_types:
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)
                

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear_outputs(self):
        self.outputs.clear()
    def get_outputs(self):
        """
        获取保存的中间输出字典
        """
        return self.outputs
