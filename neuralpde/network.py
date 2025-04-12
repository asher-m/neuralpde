import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from pathlib import Path
from typing import List, Tuple

from torchviz import make_dot



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# dt = 1, right?
dt = 1



def normalize_xy(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Normalize spatial coordinates (from meters to unitless dimension on the interval [-1, 1]).

    Returns a tuple of ((scalex, scaley), (x_normalized, y_normalized)).
    """
    assert len(x.shape) == 1 and len(y.shape) == 1, "I don't know how to handle multi-d arrays!"
    scalex, scaley = np.ptp(x), np.ptp(y)
    return (scalex, scaley), ((x - np.mean(x)) / scalex, (y - np.mean(y)) / scaley)


def normalize_data(u: np.ndarray):
    raise ValueError('Sea ice data already normalized!')


def RK(q: int = 100):
    """
    Get an implicit Runge-Kutta scheme with `q` stages.

    Default of 100 because similar problems seem to use about the same, judging from a
    cursory perusing of github/maziarraissa/PINNs and github/rezaakb/pinns-torch.

    Arguments:
        q:      Integer number of stages.
    """
    d = np.loadtxt(Path(__file__).parent / f'../raissi-2019/Utilities/IRK_weights/Butcher_IRK{q}.txt').astype(np.float32)
    A = np2torch(d[:q**2].reshape((q, q)))
    b = np2torch(d[q**2: q**2 + q])
    c = np2torch(d[q**2 + q:])

    return A, b, c


def np2torch(d):
    """
    Export numpy data to torch in every meaningful way, including sending it to the
    compute accelerator and casting it to the appropriate datatype.
    """
    return torch.from_numpy(d).to(DEVICE, DTYPE)


def torch2np(d):
    """
    Export torch data back to numpy.
    """
    return d.detach().cpu().numpy()


class Network(nn.Module):
    def __init__(self, q: int, shape: Tuple[int], kernel: int = 5) -> None:
        """
        Initialize the PINN.

        Arguments:
            loss:           Loss function over which to train.  Loss function must only use Python
                            primitive operators (+, -, *, /, **, etc...) and Torch built-ins.
            q:              Number of stages q used in RK scheme to compute loss.  See Raissi 2019.
            shape:          Shape of input data like (T, N, M), where T is the number of timesteps,
                            and N and M are spatial dimensions.
            kernel:         Size of kernel over which to convolve.
        """
        super().__init__()

        self.q = q
        self.shape = shape
        self.kernel = kernel

        assert self.kernel % 2 == 1, "Kernel size must be odd!"
        self.padding = self.kernel // 2

        self.rk_A, self.rk_b, self.rk_c = RK(self.q)

        self.channels_in = 2 + self.shape[0]  # 2 channels for spatial coords + shape[0] channels for input timesteps
        self.channels_hidden = 2**(int(np.log2(self.channels_in)) + 1)
        self.channels_out = 4 + q  # 4 channels for kappa, v (2d vector) and f + q channels for intermediate RK stages

        self.layers = nn.Sequential(
            nn.Conv2d(self.channels_in, self.channels_hidden, kernel_size=self.kernel, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(self.channels_hidden, self.channels_hidden, kernel_size=self.kernel, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(self.channels_hidden, self.channels_hidden, kernel_size=self.kernel, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(self.channels_hidden, self.channels_hidden, kernel_size=self.kernel, padding=self.padding),
            nn.ReLU()
        )
        self.out = nn.Conv2d(self.channels_hidden, self.channels_out, kernel_size=self.kernel, padding=self.padding)

    def forward(self, data: torch.tensor):
        """
        Push data through the network for training.

        Arguments:
            data:       Stacked and exported (i.e., moved to GPU if desired) data, where `data[0:2]` are spatial
                        coordinate arrays each of size `shape[1:]`, and `data[2:]` is a collection of stacked solution
                        data.

        Returns a tensor on `DEVICE`.
        """
        outs = self.out(self.layers(data))
        return outs

    def predict(self, x: np.ndarray, y: np.ndarray, u: np.ndarray):
        """
        Push data through the network for evaluation.

        Arguments:
            x:          x-spatial coordinates of each cell in the solution u.  Must be of size `shape[1:])`
                        specified at model initialization.
            y:          y-spatial coordinates of each cell in the solution u.  Must be of size `shape[1:])`
                        specified at model initialization.
            u:          Solution data of each cell.  Must be of size `shape` specified at model initialization.

        Returns a numpy array.
        """
        self.eval()
        x, y, u = np2torch(x).requires_grad_(True), np2torch(y).requires_grad_(True), np2torch(u).requires_grad_(False)
        return torch2np(self.forward(torch.stack((x, y, *u))))

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            u: np.ndarray,
            loss_weights: np.ndarray,
            epochs: int = 1000, lr: float = 1e-3,
            do_graphs: bool = False
        ):
        """
        Train the PINN.

        Arguments:
            x:              x-spatial coordinates of each cell in the solution u.  Must be of size `shape[1:])`
                            specified at model initialization.
            y:              y-spatial coordinates of each cell in the solution u.  Must be of size `shape[1:])`
                            specified at model initialization.
            u:              Solution data of each cell.  Must be of size `shape` specified at model initialization.
            loss_weights:   Weights of each term in the loss.
            epochs:         Number of epochs to run.
            lr:             Learning rate passed to Adam optimizer.
            do_graphs:      Make torchviz graphs of the computational graph.
        """
        x, y, u = np2torch(x).requires_grad_(True), np2torch(y).requires_grad_(True), np2torch(u).requires_grad_(False)
        loss_weights = np2torch(loss_weights)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # get true solution values
        u_i = u[len(u) // 2]
        u_f = u[len(u) // 2 + 1]

        self.train()
        losses = list()
        for e in range(epochs):
            optimizer.zero_grad()

            outs = self.forward(torch.stack((x, y, *u)))

            # break out nn params
            kappa = outs[0]
            v1 = outs[1]
            v2 = outs[2]
            f = outs[3]
            u_rk = outs[4:]

            # gradient insanity
            ones_kappa = torch.zeros_like(outs)
            ones_kappa[0] = 1
            ones_u_rk = list()
            for i in range(self.q):
                ones_u_rk.append(torch.zeros_like(outs))
                ones_u_rk[i][4 + i] = 1
            ones_u_rk.append(torch.ones_like(x).requires_grad_(True))  # for second derivatives

            # compute gradients
            kappa_x = torch.autograd.grad(outs, x, ones_kappa, create_graph=True)[0]
            kappa_y = torch.autograd.grad(outs, y, ones_kappa, create_graph=True)[0]

            u_rk_x = torch.empty_like(outs[4:])
            u_rk_y = torch.empty_like(outs[4:])
            u_rk_xx = torch.empty_like(outs[4:])
            u_rk_yy = torch.empty_like(outs[4:])
            for i in range(self.q):
                u_rk_x[i] = torch.autograd.grad(outs, x, ones_u_rk[i], create_graph=True)[0]
                u_rk_y[i] = torch.autograd.grad(outs, y, ones_u_rk[i], create_graph=True)[0]
                u_rk_xx[i] = torch.autograd.grad(u_rk_x[i], x, ones_u_rk[-1], create_graph=True)[0]
                u_rk_yy[i] = torch.autograd.grad(u_rk_y[i], y, ones_u_rk[-1], create_graph=True)[0]

            # evaluate pde
            pde = kappa[None, ...] * (u_rk_xx + u_rk_yy) + (kappa_x[None, ...] * u_rk_x + kappa_y[None, ...] * u_rk_y) + \
                    (v1[None, ...] * u_rk_x + v2[None, ...] * u_rk_y) + f

            # estimate solution with pde
            uhat_i = u_rk + dt * torch.einsum('ij,jkl->ikl', self.rk_A, pde)  # as in eq. (22) in Raissi 2019
            # u_hat_i = u_rk - dt * torch.einsum('ij,jkl->ikl', self.rk_A, pde)  # as in Raissi's PINN codebase
            uhat_f = u_rk + dt * torch.einsum('ij,jkl->ikl', (self.rk_A - self.rk_b[None, ...]), pde)

            # compute loss with estimate and actual solution
            loss_u_i = torch.sum((uhat_i - u_i[None, ...])**2)
            loss_u_f = torch.sum((uhat_f - u_f[None, ...])**2)

            # compute other loss terms
            loss_forcing = torch.sum(f**2)

            # compute final loss
            loss = torch.stack((loss_u_i, loss_u_f, loss_forcing)) @ loss_weights

            # make graphs, if you want
            if do_graphs:
                make_dot(kappa, params=dict(list(self.named_parameters()))).render('graph_kappa', format='pdf', cleanup=True)
                make_dot(v1, params=dict(list(self.named_parameters()))).render('graph_v1', format='pdf', cleanup=True)
                make_dot(f, params=dict(list(self.named_parameters()))).render('graph_f', format='pdf', cleanup=True)
                make_dot(u_rk, params=dict(list(self.named_parameters()))).render('graph_u_rk', format='pdf', cleanup=True)
                make_dot(kappa_x, params=dict(list(self.named_parameters()))).render('graph_kappa_x', format='pdf', cleanup=True)
                make_dot(u_rk_x, params=dict(list(self.named_parameters()))).render('graph_u_rk_x', format='pdf', cleanup=True)
                make_dot(u_rk_xx, params=dict(list(self.named_parameters()))).render('graph_u_rk_xx', format='pdf', cleanup=True)
                make_dot(uhat_i, params=dict(list(self.named_parameters()))).render('graph_uhat_i', format='pdf', cleanup=True)
                make_dot(uhat_f, params=dict(list(self.named_parameters()))).render('graph_uhat_f', format='pdf', cleanup=True)
                break  # break, because we're not making graphs every time

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if e % 10 == 0:
                print(f'Epoch {e:5d}, loss {losses[-1]:10.2f}')

        return losses
