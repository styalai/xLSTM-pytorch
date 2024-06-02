import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.utils import BlockDiagonal, CausalConv1D

class sLSTMblock(nn.Module):
    def __init__(self, x_example, depth):
        super().__init__()
        self.input_size = x_example.shape[2]
        conv_channels = x_example.shape[1]
        
        self.ln = nn.LayerNorm(self.input_size)
        
        self.conv = CausalConv1D(conv_channels, conv_channels, self.input_size)
        
        self.i_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.f_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.o_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.z_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        
        self.ri_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.rf_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.ro_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.rz_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        
        self.GN = nn.LayerNorm(self.input_size)
        self.ln_c = nn.LayerNorm(self.input_size)
        self.ln_n = nn.LayerNorm(self.input_size)
        self.ln_h = nn.LayerNorm(self.input_size)
        
        self.left_linear = nn.Linear(self.input_size, int(self.input_size*(4/3)))
        self.right_linear = nn.Linear(self.input_size, int(self.input_size*(4/3)))
        
        self.proj = nn.Linear(int(self.input_size*(4/3)), self.input_size)
        
        self.init_states(x_example)
        
    def init_states(self, x):
        self.nt_1 = torch.zeros(x.shape)
        self.ct_1 = torch.zeros(x.shape)
        self.ht_1 = torch.zeros(x.shape)
        
    def forward(self, x):
        x = self.ln(x)
        
        x_conv = F.silu(self.conv(x))
        
        # start sLSTM
        ht_1 = self.ht_1
        
        i = torch.exp(self.i_gate(x_conv) + self.ri_gate(ht_1))
        f = torch.exp(self.f_gate(x_conv) + self.rf_gate(ht_1))
        
        o = torch.sigmoid(self.o_gate(x) + self.ro_gate(ht_1))
        z = torch.tanh(self.z_gate(x) + self.rz_gate(ht_1))
        
        ct_1 = self.ct_1
        ct = f*ct_1 + i*z
        ct = self.ln_c(ct)
        self.ct_1 = ct.detach()
        
        nt_1 = self.nt_1
        nt = f*nt_1 + i
        nt = self.ln_n(nt)
        self.nt_1 = nt.detach()
        
        ht = o*(ct/nt) # torch.Size([4, 8, 16])
        ht = self.ln_h(ht)
        self.ht_1 = ht.detach()
        # end sLSTM
        
        slstm_out = self.GN(ht)
        
        left = self.left_linear(slstm_out)
        right = F.gelu(self.right_linear(slstm_out))
        
        out = left*right
        out = self.proj(out) + x
        return out
  
