import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.utils import BlockDiagonal, CausalConv1D

class mLSTMblock(nn.Module):
    def __init__(self, x_example, factor, depth, dropout=0.2):
        super().__init__()
        self.input_size = x_example.shape[2]
        self.hidden_size = int(self.input_size*factor)
        
        self.ln = nn.LayerNorm(self.input_size)
        
        self.left = nn.Linear(self.input_size, self.hidden_size)
        self.right = nn.Linear(self.input_size, self.hidden_size)
        
        self.conv = CausalConv1D(self.hidden_size, self.hidden_size, self.input_size) 
        self.drop = nn.Dropout(dropout)
        
        self.lskip = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.wq = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wk = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wv = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        
        self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_c = nn.LayerNorm(self.hidden_size)
        self.ln_n = nn.LayerNorm(self.hidden_size)
        
        self.lnf = nn.LayerNorm(self.hidden_size)
        self.lno = nn.LayerNorm(self.hidden_size)
        self.lni = nn.LayerNorm(self.hidden_size)
        
        self.GN = nn.LayerNorm(self.hidden_size)
        self.ln_out = nn.LayerNorm(self.hidden_size)
        
        self.proj = nn.Linear(self.hidden_size, self.input_size)
        self.ln_proj = nn.LayerNorm(self.input_size)
        
        self.init_states(x_example)
    
    def init_states(self, x_example):
        self.ct_1 = torch.zeros([1, 1, self.hidden_size])
        self.nt_1 = torch.zeros([1, 1, self.hidden_size])
    
    def forward(self, x):
        assert x.ndim == 3
        
        x = self.ln(x) # layer norm on x
        
        left = self.left(x) # part left 
        right = F.silu(self.right(x)) # part right with just swish (silu) function

        left_left = left.transpose(1, 2)
        left_left = F.silu( self.drop( self.conv( left_left ).transpose(1, 2) ) )
        l_skip = self.lskip(left_left)

        # start mLSTM
        q = self.wq(left_left)
        k = self.wk(left_left)
        v = self.wv(left)
        
        i = torch.exp(self.lni(self.i_gate(left_left)))
        f = torch.exp(self.lnf(self.f_gate(left_left)))
        o = torch.sigmoid(self.lno(self.o_gate(left_left)))

        ct_1 = self.ct_1
        ct = f*ct_1 + i*v*k
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()
        
        nt_1 = self.nt_1
        nt = f*nt_1 + i*k
        nt =torch.mean( self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()
        
        ht = o * ((ct*q) / torch.max(nt*q)) # [batchs_size, ?, hiddden_size]
        # end mLSTM
        
        left = self.GN(ht + l_skip)
        
        out = self.ln_out(left * right)
        out = self.ln_proj(self.proj(out))
        
        return out
