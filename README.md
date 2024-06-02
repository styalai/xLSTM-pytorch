original code from https://github.com/muditbhargava66/PyxLSTM
# xLSTM-pytorch
A easy to use implementation of xLSTM.
I work on a version more advanced with an entirely xLSTM implementation.

## Installation
```python
pip install git+https://github.com/styalai/xLSTM-pytorch
```
## Example

```python
import torch
import torch.nn as nn
from xLSTM.xLSTM import xLSTM as xlstm

batch_size = 4
seq_lenght = 8
input_size = 32
x_example = torch.zeros(batch_size, seq_lenght, input_size)
factor = 2 # by how much is input_size multiplied to give hidden_size
depth = 4 # number of block for q, k and v
layers = 'ms' # m for mLSTMblock and s for sLSTMblock

model = xlstm(layers, x_example, factor=factor, depth=depth)

x = torch.randn(batch_size, seq_lenght, input_size)
out = model(x)
print(out.shape)
# torch.Size([4, 8, 32])
```
Note : if you load a xlstm after training you must do : model.init_states(x_example)
## Citation

If you use xlstm-pytorch in your research or projects, please cite the original xLSTM paper:

```bibtex
@article{Beck2024xLSTM,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and Pöppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, Günter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```
