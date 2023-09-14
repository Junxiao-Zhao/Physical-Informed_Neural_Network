import os
from collections import OrderedDict
from typing import List, Union
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import trange
from pyDOE import lhs
import seaborn

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Propagation

        :param x: a input data
        :return: the output
        """
        x = 2 * (x - self.lw_b) / (self.hg_b - self.lw_b) - torch.tensor(
            [1.0, 1.0], dtype=torch.float32, device=DEVICE)
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

    def govern_func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate the govern function

        :param x: a input tensor (x-axis)
        :param y: a input tensor (y-axis)
        :return: the result of the govern function
        """

        z = self.dnn(torch.hstack((x, y)))
        z_x = torch.autograd.grad(z,
                                  x,
                                  grad_outputs=torch.ones_like(z),
                                  retain_graph=True,
                                  create_graph=True)[0]
        z_y = torch.autograd.grad(z,
                                  y,
                                  grad_outputs=torch.ones_like(z),
                                  retain_graph=True,
                                  create_graph=True)[0]
        z_yy = torch.autograd.grad(z_y,
                                   y,
                                   grad_outputs=torch.ones_like(z_y),
                                   retain_graph=True,
                                   create_graph=True)[0]

        f = z_x + z * z_y - torch.tensor(
            (0.01 / np.pi), dtype=torch.float32, device=DEVICE) * z_yy
        return f

    def forward(self, data_x: torch.Tensor, data_y: torch.Tensor,
                data_z: torch.Tensor, govern_x: torch.Tensor,
                govern_y: torch.Tensor, lmda: float) -> torch.Tensor:
        """Forward Propagation

        :param data_x: the x-axis of the data points
        :param data_y: the y-axis of the data points
        :param data_z: the z-axis of the data points
        :param govern_x: the x-axis of sampled points in the range
        :param govern_y: the y-axis of sampled points in the range
        :param lmda: the lambda
        :return: the loss
        """

        u_pred = self.dnn(torch.hstack(
            (data_x, data_y)))  # predict z-axis of data points
        f_pred = self.govern_func(govern_x,
                                  govern_y)  # result of the govern function
        loss = (1 - lmda) * F.mse_loss(u_pred, data_z) + lmda * torch.mean(
            f_pred**2)

        return loss


if __name__ == '__main__':

    input_x = np.linspace(0, 1, 200).reshape(200, 1)
    input_y = np.linspace(-1, 1, 200).reshape(200, 1)
    point_x, point_y = np.meshgrid(input_x, input_y)

    # boundary conditions
    xy0 = np.hstack((point_x[0:1, :].T, point_y[0:1, :].T))
    z_y0 = np.zeros(input_x.shape)

    xy1 = np.hstack((point_x[-1:, :].T, point_y[-1:, :].T))
    z_y1 = np.zeros(input_x.shape)

    yx0 = np.hstack((point_x[:, 0:1], point_y[:, 0:1]))
    z_x0 = -np.sin(np.pi * point_y[:, 0:1])

    # data points
    xy_train = np.vstack((xy0, xy1, yx0))
    z_train = np.vstack((z_y0, z_y1, z_x0))

    # sample points in the range
    sample_xy = [0.0, -1.0] + [1, 2] * lhs(2, 3200)
    govern_xy = np.vstack((xy_train, sample_xy))

    # create tensors
    tensor_x = torch.tensor(xy_train[:, 0:1],
                            dtype=torch.float32,
                            device=DEVICE)
    tensor_y = torch.tensor(xy_train[:, 1:],
                            dtype=torch.float32,
                            device=DEVICE)
    tensor_z = torch.tensor(z_train, dtype=torch.float32, device=DEVICE)
    tensor_govern_x = torch.tensor(govern_xy[:, 0:1],
                                   dtype=torch.float32,
                                   device=DEVICE,
                                   requires_grad=True)
    tensor_govern_y = torch.tensor(govern_xy[:, 1:],
                                   dtype=torch.float32,
                                   device=DEVICE,
                                   requires_grad=True)

    # initialize
    dnn = DNN([2, 50, 50, 50, 1],
              torch.tensor([0, -1], dtype=torch.float32, device=DEVICE),
              torch.tensor([1, 1], dtype=torch.float32,
                           device=DEVICE)).to(DEVICE)
    optimizer_Adam = optim.Adam(dnn.parameters())
    loss_func = LOSS(dnn)
    writer = SummaryWriter('./tensorboard')

    # train
    for epoch in trange(5000, desc='Training', unit='epoch'):
        # calculate loss and record
        loss = loss_func(tensor_x, tensor_y, tensor_z, tensor_govern_x,
                         tensor_govern_y, 0.15)
        writer.add_scalar('loss', loss, epoch)

        # backward and optimize
        optimizer_Adam.zero_grad()
        loss.backward()
        optimizer_Adam.step()

        if epoch % 100 == 99:
            logging.getLogger(__name__).info('Epoch: %d, Loss: %.3e', epoch,
                                             loss.item())

    # predict
    intp_x = np.linspace(0, 1, 200, endpoint=False)
    intp_y = np.linspace(-1, 1, 200, endpoint=False)
    pred_x, pred_y = np.meshgrid(intp_x.reshape(-1, 1), intp_y.reshape(-1, 1))
    pred_xy = np.dstack((pred_x, pred_y)).reshape(-1, 2)

    with torch.no_grad():
        pred_z = dnn(torch.tensor(
            pred_xy, dtype=torch.float32,
            device=DEVICE)).cpu().detach().numpy().reshape(200, 200)

    # plot
    fig, axs = plt.subplots(3,
                            2,
                            figsize=(30, 30),
                            gridspec_kw={'width_ratios': [20, 10]})
    axs[0, 0].remove()
    axs[1, 0].remove()

    # 3D
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
    axs[0, 0] = fig.add_subplot(gs[0, 0], projection='3d')
    axs[0, 0].plot_surface(pred_x, pred_y, pred_z, cmap='rainbow')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_zlabel('z')
    axs[0, 0].view_init(elev=24, azim=24)

    # heatmap
    seaborn.set()
    seaborn.heatmap(pd.DataFrame(pred_z[::-1], np.around(intp_x[::-1], 2),
                                 np.around(intp_y[::-1], 2)),
                    cmap='rainbow',
                    ax=axs[2, 0])

    # curves
    def plot_sub(ax, x: float) -> None:
        """Plot curves on subplots

        :param ax: axes
        :param x: a specified x
        """
        x_index = np.where(intp_x == x)[0][0]
        ax.plot(intp_y, pred_z[:, x_index:x_index + 1])
        ax.set_title('x=%.2f' % x)
        ax.set_xlabel('y')
        ax.set_ylabel('z')

    for i, x in enumerate([0.25, 0.5, 0.75]):
        plot_sub(axs[:, 1][i], x)

    fig.suptitle('2D Burgers\' Equation')
    fig.savefig('./imgs/2d burgers.png')

    plt.show()
