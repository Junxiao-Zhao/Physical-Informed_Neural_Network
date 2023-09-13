import os
from collections import OrderedDict
from typing import List, Union
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import trange

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DNN(torch.nn.Module):
    """A deep neural network"""

    def __init__(self, layers: List[int], lw_b: Union[torch.Tensor, float],
                 hg_b: Union[torch.Tensor, float]) -> None:
        """Constructor

        :param layers: the number of nodes in each layer
        :param lw_b: the lower bound
        :param hg_b: the upper bound
        """
        super(DNN, self).__init__()

        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        self.lw_b = lw_b
        self.hg_b = hg_b

        # initialize layers
        layer_list = list()
        for i in range(self.depth - 1):
            each_layer = torch.nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(each_layer.weight)
            nn.init.zeros_(each_layer.bias)
            layer_list.append(('layer_%d' % i, each_layer))
            layer_list.append(('activation_%d' % i, self.activation()))

        each_layer = torch.nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(each_layer.weight)
        nn.init.zeros_(each_layer.bias)
        layer_list.append(('layer_%d' % (self.depth - 1), each_layer))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

        # register lambda
        self.lmda = torch.tensor([0.5],
                                 dtype=torch.float32,
                                 device=DEVICE,
                                 requires_grad=True)
        self.lmda = torch.nn.Parameter(self.lmda)
        self.register_parameter('lambda', self.lmda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Propagation

        :param x: a input data
        :return: the output
        """
        x = 2 * (x - self.lw_b) / (self.hg_b - self.lw_b)
        out = self.layers(x)
        return out


class LOSS(nn.Module):
    """The loss function"""

    def __init__(self, network) -> None:
        """Constructor

        :param network: a neural network
        """
        super().__init__()
        self.dnn = network

    def govern_func(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the govern function

        :param x: a input tensor
        :return: the result of the govern function
        """

        y = self.dnn(x)
        y_x = torch.autograd.grad(y,
                                  x,
                                  grad_outputs=torch.ones_like(y),
                                  retain_graph=True,
                                  create_graph=True)[0]

        f = -y + y_x
        # f = y - x + y_x
        return f

    def forward(self, data_x: torch.Tensor, data_y: torch.Tensor,
                govern_x: torch.Tensor) -> torch.Tensor:
        """Forward Propagation

        :param data_x: the x-axis of the data points
        :param data_y: the y-axis of the data points
        :param govern_x: the x-axis of sampled points in the range
        :return: the loss
        """

        u_pred = self.dnn(data_x)  # predict y-axis of data points
        f_pred = self.govern_func(govern_x)  # result of the govern function
        loss = (1 - self.dnn.lmda) * F.mse_loss(
            u_pred, data_y) + self.dnn.lmda * torch.mean(f_pred**2)

        return loss


if __name__ == '__main__':

    # initialize
    dnn = DNN([1, 30, 30, 30, 1], 0.0, 1.0).to(DEVICE)
    optimizer_Adam = optim.Adam(dnn.parameters())
    loss_func = LOSS(dnn)
    writer = SummaryWriter('./tensorboard')

    # input data
    data_x = torch.tensor([[0.0]],
                          dtype=torch.float32,
                          device=DEVICE,
                          requires_grad=True)
    data_y = torch.tensor([[1.0]], dtype=torch.float32, device=DEVICE)

    input_x = np.linspace(0, 1, 200).reshape(200, 1)
    input_y = np.exp(input_x)
    # input_y = 1 - input_x + input_x**2 - input_x**3 / 3 \
    #     + input_x**4 / 12 - input_x**5 / 60
    govern_x = torch.tensor(input_x,
                            dtype=torch.float32,
                            device=DEVICE,
                            requires_grad=True)

    # train
    for epoch in trange(5000, desc='Training', unit='epoch'):
        # calculate loss and record
        loss = loss_func(data_x, data_y, govern_x)
        writer.add_scalar('lambda', dnn.lmda, epoch)
        writer.add_scalar('loss', loss, epoch)

        # backward and optimize
        optimizer_Adam.zero_grad()
        loss.backward()
        optimizer_Adam.step()

        if epoch % 100 == 99:
            logging.getLogger(__name__).info(
                'Epoch: %d, Loss: %.3e, Lambda: %.3f', epoch, loss.item(),
                dnn.lmda.item())

    # predict
    with torch.no_grad():
        predict_y = dnn(govern_x).cpu().detach().numpy()

    # plot
    fig = plt.figure(figsize=(10, 10))
    plt.plot(input_x, input_y, color='red', label='target')
    plt.plot(input_x, predict_y, color='blue', label='predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('PINN 1D')
    plt.legend()
    plt.savefig('./imgs/pinn 1d.png')
    plt.show()
