import torch
from torch import float32
import numpy as np
import matplotlib.pyplot as plt

'''
Make sure to detach and move any torch tensors to cpu as device before using this method
also make sure that your models function is named f.
example:

    x_train = x_train.to('cpu')
    y_train = y_train.to('cpu')
    model.W = model.W.to('cpu')
    model.b = model.b.to('cpu')
    model.W = model.W.detach()
    model.b = model.b.detach()

    torch_plot3d("NAND", x_train[:, 0], x_train[:, 1], model)
'''


def torch_plotplane3d(title, x_vector, y_vector, model):
    """
    A  method that takes in the vectors from a torch optimization and plots the resulting plane
    in 3d.
    title = String
    x_vector = Torch.Tensor (make sure to detach and that it is located on the cpu)
    y_vector = Torch.Tensor (make sure to detach and that it is located on the cpu)
    model = Class (make sure that the model function method is named f() and takes one argument: x)
    """
    fig = plt.figure(title)
    plot1 = fig.add_subplot(111, projection='3d')

    x = np.linspace(torch.min(x_vector), torch.max(x_vector), 10)
    y = np.linspace(torch.min(y_vector), torch.max(y_vector), 10)

    X, Y = np.meshgrid(x, y)
    Z = np.empty([10, 10])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            Z[i, j] = model.f(torch.tensor(
                [[X[i, j], Y[i, j]]], dtype=float32)).numpy()
    plot1.plot_wireframe(X, Y, Z, color="green")

    plt.show()


def torch_plotplane3d_withpoints(title, x_vector, y_vector, z_vector, model):
    """
    A  method that takes in the vectors from a torch optimization and plots the resulitng plane
    with points in 3D.
    title = String
    x_vector = Torch.Tensor (make sure to detach and that it is located on the cpu)
    y_vector = Torch.Tensor (make sure to detach and that it is located on the cpu)
    z_vector = Torch.Tensor (make sure to detach and that it is located on the cpu)
    model = Class (make sure that the model function method is named f() and takes one argument: x)
    """
    fig = plt.figure(title)

    plot1 = fig.add_subplot(111, projection='3d')
    plot1.plot3D(x_vector.numpy(), y_vector.numpy(),
                 z_vector.numpy(), 'o')

    x = np.linspace(torch.min(x_vector), torch.max(x_vector), 10)
    y = np.linspace(torch.min(y_vector), torch.max(y_vector), 10)

    X, Y = np.meshgrid(x, y)
    Z = np.empty([10, 10])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            Z[i, j] = model.f(torch.tensor(
                [[X[i, j], Y[i, j]]], dtype=float32)).numpy()
    plot1.plot_wireframe(X, Y, Z, color="green")

    plt.show()
