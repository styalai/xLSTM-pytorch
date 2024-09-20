# xLSTM-pytorch
An easy to use and efficient implementation of xLSTM.
Here are a few articles to help you understand :
 - [Understanding xLSTM through code implementation(pytorch)](https://medium.com/@arthur.lagacherie/implement-the-xlstm-paper-from-scratch-with-pytorch-3a2a4ddb4f94)
 - [Implement the xLSTM paper from scratch with Pytorch](https://medium.com/@arthur.lagacherie/implement-the-xlstm-paper-from-scratch-with-pytorch-5157a1b40ec8)

## INFO
I am sorry for the potential mistakes on my docs because I am French and and don't speak english very well. <b>And if you like this repository you can put a star.</b>

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
factor = 2 # how much input_size will be multiply to give hidden_size
depth = 4 # number of blocks for q, k and v
layers = 'ms' # m for mLSTMblock and s for sLSTMblock

model = xlstm(layers, x_example, factor=factor, depth=depth)

x = torch.randn(batch_size, seq_lenght, input_size)
out = model(x)
print(out.shape)
# torch.Size([4, 8, 32])
```
## Result
I test a mLSTM block (18M parameters) on an NLP task (tiny shakespeare dataset of Karpathy).

<img src="/assets/loss3000xlstm.PNG" alt="drawing" width="400"/>
<img src="/assets/loss6000.PNG" alt="drawing" width="400"/>

Code in 'examples/tinyshakespeare'.
We can see that the model overfits a little.

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
## Code from
The original code of 'xLSTM/utils.py' come from https://github.com/akaashdash/xlstm
