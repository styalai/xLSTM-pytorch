original code from https://github.com/muditbhargava66/PyxLSTM
# xLSTM-pytorch
A easy to use implementation of xLSTM
I work on a version more advanced with an entirely xLSTM implementation.

## Installation
```python
pip install git+https://github.com/styalai/xLSTM-pytorch
```
## Example

```python
from xLSTM_simple.xlstm import *

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
