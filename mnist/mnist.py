from neural_network.nn_xor import EPOCH_AMOUNT, STEP_SIZE
import torch
from torch import float32
import torch.nn.functional as func
import torchvision
import matplotlib.pyplot as plt

device = ("cuda" if torch.cuda.is_available() else "cpu")

STEP_SIZE = 0.0002
EPOCH_AMOUNT = 10000


class mnist_model:
    def __init__(self):
        self.W = torch.zeros(784, 10, requires_grad=True,
                             dtype=float32, device=device)
        self.b = torch.zeros(10, requires_grad=True,
                             dtype=float32, device=device)

    def logits(self, x):
        return x@self.W + self.b

    def f(self, x):
        return torch.softmax(self.logits(x), dim=-1, dtype=float32)

    def loss(self, x, y):
        return func.binary_cross_entropy_with_logits(self.f(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


def do_mnist():
    # Load observations from the mnist dataset. The observations are divided into a training set and a test set
    mnist_train = torchvision.datasets.MNIST(
        './data', train=True, download=True)
    # Reshape input
    x_train = mnist_train.data.reshape(-1, 784).float().to(device=device)
    # Create output tensor
    y_train = torch.zeros((mnist_train.targets.shape[0], 10), device=device)
    y_train[torch.arange(mnist_train.targets.shape[0]),
            mnist_train.targets] = 1  # Populate output

    mnist_test = torchvision.datasets.MNIST(
        './data', train=False, download=True)
    # Reshape input
    x_test = mnist_test.data.reshape(-1, 784).float().to(device=device)
    # Create output tensor
    y_test = torch.zeros((mnist_test.targets.shape[0], 10), device=device)
    y_test[torch.arange(mnist_test.targets.shape[0]),
           mnist_test.targets] = 1  # Populate output

    model = mnist_model()

    optimizer = torch.optim.SGD(
        [model.W, model.b], STEP_SIZE)

    frac = 100/EPOCH_AMOUNT
    print_amount = EPOCH_AMOUNT/100
    current = 0

    for epoch in range(EPOCH_AMOUNT):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()
        current += 1
        if (current/print_amount).is_integer():
            print(" ", (int(current*frac)), "%", "loss = %s, accuracy = %s" % (model.loss(
                x_train, y_train).item(), model.accuracy(x_test, y_test).item()), end='\r')

    print("loss = %s, accuracy = %s" % (model.loss(
        x_train, y_train).item(), model.accuracy(x_test, y_test).item()))

    if device == 'cuda':
        x_train = x_train.to('cpu')
        y_train = y_train.to('cpu')
        model.W = model.W.to('cpu')
        model.b = model.b.to('cpu')

    model.W = model.W.detach()
    model.b = model.b.detach()

    fig = plt.figure("W after optimalization")
    rows = 2
    colums = 5
    for i in range(10):
        fig.add_subplot(rows, colums, i+1)
        plt.imshow(model.W[:, i].reshape(28, 28))
        plt.axis('off')
        plt.title(i)
        plt.imsave('W%s.png' % (i), model.W[:, i].reshape(28, 28))

    plt.show()
