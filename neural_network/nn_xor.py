'''
c) Lag en modell som predikerer tilsvarende XOR-operatoren. Før
du optimaliserer denne modellen må du initialisere
modellvariablene med tilfeldige tall for eksempel mellom -1 og
1. Visualiser både når optimaliseringen konvergerer og ikke
konvergerer mot en riktig modell.
'''
from matplotlib.pyplot import get
import torch
from torch import float32
from torch.cuda import random
import torch.nn.functional as func
import random
import plot3d.plotter3d as plotter


STEP_SIZE = 0.1
EPOCH_AMOUNT = 1000000


device = ("cuda" if torch.cuda.is_available() else "cpu")


def get_rand():
    return random.uniform(-1, 1)


class SigmoidModel:

    def __init__(self):
        # Initialize the model variables with the same
        # floats between 1, -1 generated with random.uniform(-1,1)
        self.W1 = torch.tensor([[get_rand(), get_rand()], [get_rand(), get_rand()]], requires_grad=True,
                               dtype=float32, device=device)
        self.b1 = torch.tensor([get_rand(), get_rand()], requires_grad=True,
                               dtype=float32, device=device)

        self.W2 = torch.tensor([[get_rand()], [get_rand()]], requires_grad=True,
                               dtype=float32, device=device)
        self.b2 = torch.tensor([get_rand()], requires_grad=True,
                               dtype=float32, device=device)

    def f(self, x):
        h = torch.sigmoid(x@self.W1+self.b1)

        return torch.sigmoid(h@self.W2+self.b2)

    def loss(self, x, y):
        return func.binary_cross_entropy_with_logits(self.f(x), y)


def visualize_xor():

    model = SigmoidModel()

    # input for the XOR operator
    observed_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
    observed_output = [[0], [1], [1], [0]]

    x_train = torch.tensor(observed_input, dtype=float32, device=device)
    y_train = torch.tensor(observed_output, dtype=float32, device=device)

    optimizer = torch.optim.SGD(
        [model.W1, model.b1, model.W2, model.b2], STEP_SIZE)

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

    print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" %
          (model.W1, model.b1, model.W2, model.b2, model.loss(x_train, y_train).item()))

    x_train = x_train.to("cpu")
    y_train = y_train.to("cpu")
    model.W1 = model.W1.detach().to("cpu")
    model.b1 = model.b1.detach().to("cpu")
    model.W2 = model.W2.detach().to("cpu")
    model.b2 = model.b2.detach().to("cpu")

    plotter.torch_plotplane3d("xor", x_train[:, 0],
                              x_train[:, 1], model)
