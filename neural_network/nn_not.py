'''
a) Lag en modell som predikerer tilsvarende NOT-operatoren.
Visualiser resultatet etter optimalisering av modellen.
'''
import torch
from torch._C import float32


class model:
    def __init__(self):
        self.W = torch.zeros(1, 1, requires_grad=True, dtype=float32)
        self.b = torch.zeros(1, 1, requires_grad=True, dtype=float32)
