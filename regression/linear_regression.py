import torch
import matplotlib.pyplot as plt
import torch.nn.functional as function
from mpl_toolkits import mplot3d
import numpy as np

'''
Tar imot både 2 og 3D datasett og utfører lineær regresjon på de.
'''

# Variable som kan justeres:
EPOCH_AMOUNT_2D = 100000
STEP_SIZE_2D = 0.00015

EPOCH_AMOUNT_3D = 1500000
STEP_SIZE_3D = 0.00011


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class model_2d:
    def __init__(self):
        # requires_grad enables calculation of gradients
        self.W = torch.zeros(1, 1, dtype=torch.float32,
                             device=device, requires_grad=True)
        self.b = torch.zeros(1, 1, dtype=torch.float32,
                             device=device, requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return function.mse_loss(self.f(x), y)


def linear2d(data_list):

    epoch_amount = EPOCH_AMOUNT_2D
    step_size = STEP_SIZE_2D

    print("performing two-dimensional linear regression using " + device.type)
    print("with " + str(epoch_amount) +
          " epochs, and a step size of: " + str(step_size))
    model = model_2d()
    headers = data_list.pop(0)
    x_data = []
    y_data = []
    for row in data_list:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

    x = torch.tensor(x_data, dtype=torch.float32, device=device).reshape(-1, 1)
    y = torch.tensor(y_data, dtype=torch.float32, device=device).reshape(-1, 1)

    optimizer = torch.optim.SGD([model.W, model.b], step_size)

    frac = 100/epoch_amount
    print_amount = epoch_amount/100
    current = 0

    for epoch in range(epoch_amount):
        model.loss(x, y).backward()
        optimizer.step()
        optimizer.zero_grad()
        current += 1
        if (current/print_amount).is_integer():
            print("  ", int(current*frac), "%", end='\r')

    x = x.to("cpu")
    y = y.to("cpu")
    model.W = model.W.to("cpu")
    model.b = model.b.to("cpu")

    print("W = %s, b = %s, loss = %s" %
          (model.W[0].item(), model.b[0].item(), model.loss(x, y).item()))

    plt.plot(x, y, 'o', label='$(x^{(i)},y^{(i)})$')
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    x1 = torch.tensor([[torch.min(x)], [torch.max(x)]])
    plt.plot(x1, model.f(x1).detach(), label='$\\hat y = f(x) = xW+b$')
    plt.legend()
    plt.show()


class model_3d:
    def __init__(self):
        self.W = torch.zeros(2, 1, dtype=torch.float32,
                             device=device, requires_grad=True)
        self.b = torch.zeros(1, 1, dtype=torch.float32,
                             device=device, requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return function.mse_loss(self.f(x), y)


def linear3d(data_list):

    epoch_amount = EPOCH_AMOUNT_3D
    step_size = STEP_SIZE_3D

    print("performing three-dimensional linear regression using " + device.type)
    print("with " + str(epoch_amount) +
          " epochs, and a step size of: " + str(step_size))
    model = model_3d()
    headers = data_list.pop(0)
    x_data = []
    y_data = []
    for row in data_list:
        x_data.append([float(row[1]), float(row[2])])
        y_data.append([float(row[0])])

    x = torch.tensor(x_data, dtype=torch.float32, device=device).reshape(-1, 2)
    y = torch.tensor(y_data, dtype=torch.float32, device=device).reshape(-1, 1)

    optimizer = torch.optim.SGD([model.W, model.b], step_size)

    frac = 100/epoch_amount
    print_amount = epoch_amount/100
    current = 0

    for epoch in range(epoch_amount):
        model.loss(x, y).backward()
        optimizer.step()
        optimizer.zero_grad()
        current += 1
        if (current/print_amount).is_integer():
            print("  ", int(current*frac), "%", end='\r')

    print("W = %s, b = %s, loss = %s" %
          (model.W[0].item(), model.b[0].item(), model.loss(x, y).item()))

    x = x.to("cpu").numpy()
    y = y.to("cpu").numpy()
    model.W = model.W.to("cpu").detach().numpy()
    model.b = model.b.to("cpu").detach().numpy()

    plot = plt.axes(projection="3d")
    plot.plot3D(x[:, 0], x[:, 1], y[:, 0], 'o')
    plot.set_xlabel(headers[1])
    plot.set_ylabel(headers[2])
    plot.set_zlabel(headers[0])
    plt.show()


def performLinearRegression(data_list):

    if len(data_list[0]) == 2:
        linear2d(data_list)

    if len(data_list[0]) == 3:
        linear3d(data_list)
