'''
b) Lag en modell som predikerer tilsvarende NAND-operatoren.
Visualiser resultatet etter optimalisering av modellen.
'''

import torch
from torch import float32
import torch.nn.functional as func
import plot3d.plotter3d as plot


STEP_SIZE = 0.1
EPOCH_AMOUNT = 50000


device = ("cuda" if torch.cuda.is_available() else "cpu")


class SigmoidModel:
    def __init__(self):
        self.W = torch.zeros(2, 1, requires_grad=True,
                             dtype=float32, device=device)
        self.b = torch.zeros(1, 1, requires_grad=True,
                             dtype=float32, device=device)

    def z(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self.z(x))

    def loss(self, x, y):
        return func.binary_cross_entropy_with_logits(self.z(x), y)


def visualize_nand():

    model = SigmoidModel()

    # input for the NAND operator
    observed_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
    observed_output = [[1], [1], [1], [0]]

    x_train = torch.tensor(observed_input, dtype=float32, device=device)
    y_train = torch.tensor(observed_output, dtype=float32, device=device)

    optimizer = torch.optim.SGD([model.W, model.b], STEP_SIZE)

    frac = 100/EPOCH_AMOUNT
    print_amount = EPOCH_AMOUNT/100
    current = 0

    for epoch in range(EPOCH_AMOUNT):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()
        current += 1
        if (current/print_amount).is_integer():
            print("  ", int(current*frac), "%", end='\r')

    print("W = %s, b = %s, loss = %s" %
          (model.W[0].item(), model.b[0].item(), model.loss(x_train, y_train).item()))

    if device == 'cuda':
        x_train = x_train.to('cpu')
        y_train = y_train.to('cpu')
        model.W = model.W.to('cpu')
        model.b = model.b.to('cpu')

    model.W = model.W.detach()
    model.b = model.b.detach()

    plot.torch_plotplane3d("NAND", x_train[:, 0], x_train[:, 1], model)
