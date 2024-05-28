original code from https://github.com/muditbhargava66/PyxLSTM
# xLSTM-pytorch
A easy to use implementation of xLSTM

## Installation
```python
pip install git+https://github.com/styalai/xLSTM-pytorch
```
## Example

```python
from xLSTM.xlstm import *

input_size = 54
hidden_size = 54
num_layers = 3
batch_size = 4
seq_len = 8
config_block = ["mlstm", "slstm"]
num_blocks = len(config_block)

model = xLSTM(input_size, hidden_size, num_layers, num_blocks, config_block)

x = torch.randn(batch_size, seq_len, input_size)
out = model(x)
print(out.shape)
# torch.Size([4, 8, 54])
```
