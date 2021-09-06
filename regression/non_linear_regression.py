import torch
import matplotlib.pyplot as plt
import torch.nn.functional as func


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variable som kan justeres
EPOCH_AMOUNT = 100000
STEP_SIZE = 0.0000013


class non_linear_model:
    def __init__(self):
        # requires_grad enables calculation of gradients
        self.W = torch.zeros(1, 1, dtype=torch.float32,
                             device=device, requires_grad=True)
        self.b = torch.zeros(1, 1, dtype=torch.float32,
                             device=device, requires_grad=True)

    # Matrix multiplication error
    def f(self, x):
        return 20*torch.sigmoid(x*self.W+self.b)+31

    # must be used with logical regression, not sure if this applies.
    def loss(self, x, y):
        return func.mse_loss(self.f(x), y)


def non_linear2d(data_list):

    epoch_amount = EPOCH_AMOUNT
    step_size = STEP_SIZE

    print("performing two-dimensional non linear regression using " + device.type)
    print("with " + str(epoch_amount) +
          " epochs, and a step size of: " + str(step_size))
    model = non_linear_model()
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
