import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.mLSTMblock import mLSTMblock
from xLSTM.sLSTMblock import sLSTMblock

class xLSTM(nn.Module):
    def __init__(self, layers, x_example, depth=4, factor=2):
        super(xLSTM, self).__init__()

        self.layers = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMblock(x_example, depth)
            elif layer_type == 'm':
                layer = mLSTMblock(x_example, factor, depth)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)
    
    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
        
    def forward(self, x):
        x_original = x.clone()
        for l in self.layers:
             x = l(x) + x_original

        return x
