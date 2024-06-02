import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.utils import BlockDiagonal, CausalConv1D

class mLSTMblock(nn.Module):
    def __init__(self, x_example, factor, depth):
        super().__init__()
        self.input_size = x_example.shape[2]
        conv_channels = x_example.shape[1]
        self.hidden_size = int(self.input_size*factor)
        
        self.ln = nn.LayerNorm(self.input_size)
        
        self.left = nn.Linear(self.input_size, self.hidden_size)
        self.right = nn.Linear(self.input_size, self.hidden_size)
        
        self.conv = CausalConv1D(conv_channels, conv_channels, self.hidden_size)
        
        self.lskip = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.wq = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wk = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wv = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        
        self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.GN = nn.LayerNorm(self.hidden_size)
        
        self.proj = nn.Linear(self.hidden_size, self.input_size)
        
        self.init_states(x_example)
    
    def init_states(self, x_example):
        self.ct_1 = torch.zeros([x_example.shape[0], x_example.shape[1], self.hidden_size])
        self.nt_1 = torch.zeros([x_example.shape[0], x_example.shape[1], self.hidden_size])
    
    def forward(self, x):
        assert x.ndim == 3
        
        x = self.ln(x) # layer norm on x
        
        left = self.left(x) # part left 
        right = F.silu(self.right(x)) # part right with just swish (silu) function

        left_left = F.silu(self.conv(left))
        l_skip = self.lskip(left_left)

        # start mLSTM
        q = self.wq(left_left)
        k = self.wk(left_left)
        v = self.wv(left)
        
        i = torch.exp(self.i_gate(left_left))
        f = torch.exp(self.f_gate(left_left))
        o = torch.sigmoid(self.o_gate(left_left))
        
        ct_1 = self.ct_1
        ct = f*ct_1 + i*v*k
        self.ct_1 = ct.detach()
        
        nt_1 = self.nt_1
        nt = f*nt_1 + i*k
        self.nt_1 = nt.detach()
        
        ht = o * ((ct*q) / torch.max(nt*q)) # [batchs_size, ?, hiddden_size]
        # end mLSTM
        
        left = self.GN(ht) + l_skip
        
        out = left * right
        out = self.proj(out) + x
        
        return out
        
        
