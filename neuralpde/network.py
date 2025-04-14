import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from pathlib import Path
from typing import List, Tuple

from torchviz import make_dot

from .layer import LocallyConnected2d



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
        self.channels_hidden = 2 * (4 + self.q)
        self.channels_out = 4 + self.q  # 4 channels for kappa, v (2d vector) and f + q channels for intermediate RK stages

        self.layers = nn.Sequential(
            nn.Conv2d(self.channels_in, self.channels_hidden, kernel_size=self.kernel, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(self.channels_hidden, self.channels_hidden, kernel_size=self.kernel, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(self.channels_hidden, self.channels_hidden, kernel_size=self.kernel, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(self.channels_hidden, self.channels_hidden, kernel_size=self.kernel, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(self.channels_hidden, self.channels_out, kernel_size=self.kernel, padding=self.padding),
        )

        # gradient insanity
        ones_dkappa = torch.zeros((self.channels_out,) + self.shape[1:])
        ones_dkappa[0] = 1
        self.register_buffer('ones_dkappa', ones_dkappa)
        ones_dv1 = torch.zeros((self.channels_out,) + self.shape[1:])
        ones_dv1[1] = 1
        self.register_buffer('ones_dv1', ones_dv1)
        ones_dv2 = torch.zeros((self.channels_out,) + self.shape[1:])
        ones_dv2[2] = 1
        self.register_buffer('ones_dv2', ones_dv2)
        ones_du_rk = list()
        for i in range(self.q):
            ones_du_rk.append(torch.zeros((self.channels_out,) + self.shape[1:]))
            ones_du_rk[i][4 + i] = 1
        self.register_buffer('ones_du_rk', torch.stack(ones_du_rk).contiguous())
        ones_ddu_rk = torch.ones(self.shape[1:])
        self.register_buffer('ones_ddu_rk', ones_ddu_rk)

    def forward(self, x: torch.tensor, y: torch.tensor, u: torch.tensor, do_graphs: bool = False):
        """
        Push data through the network for training.

        Arguments:
            x:              x-spatial coordinates of each cell in the solution u.  Must be of size `shape[1:])`
                            specified at model initialization.
            y:              y-spatial coordinates of each cell in the solution u.  Must be of size `shape[1:])`
                            specified at model initialization.
            u:              Solution data of each cell.  Must be of size `shape` specified at model initialization.

        Returns a tuple of tensors on `DEVICE`.
        """
        outs = self.layers(torch.stack((x, y, *u)).contiguous())

        # break out nn params
        kappa = outs[0]
        v1 = outs[1]
        v2 = outs[2]
        f = outs[3]
        u_rk = outs[4:]

        # compute gradients
        kappa_x, kappa_y = torch.autograd.grad(outs, (x, y), self.ones_dkappa, create_graph=True)
        v1_x, v1_y = torch.autograd.grad(outs, (x, y), self.ones_dv1, create_graph=True)
        v2_x, v2_y = torch.autograd.grad(outs, (x, y), self.ones_dv2, create_graph=True)
        u_rk_x = torch.empty_like(outs[4:])  # can I optimize creating all of these arrays?
        u_rk_y = torch.empty_like(outs[4:])
        u_rk_xx = torch.empty_like(outs[4:])
        u_rk_yy = torch.empty_like(outs[4:])
        for i in range(self.q):
            u_rk_x[i] = torch.autograd.grad(outs, x, self.ones_du_rk[i], create_graph=True)[0]
            u_rk_y[i] = torch.autograd.grad(outs, y, self.ones_du_rk[i], create_graph=True)[0]
            u_rk_xx[i] = torch.autograd.grad(u_rk_x[i], x, self.ones_ddu_rk, create_graph=True)[0]
            u_rk_yy[i] = torch.autograd.grad(u_rk_y[i], y, self.ones_ddu_rk, create_graph=True)[0]

        # evaluate pde
        pde = kappa.unsqueeze(0) * (u_rk_xx + u_rk_yy) + (kappa_x.unsqueeze(0) * u_rk_x + kappa_y.unsqueeze(0) * u_rk_y) + \
            (v1.unsqueeze(0) * u_rk_x + v2.unsqueeze(0) * u_rk_y) + f

        # estimate solution with pde
        uhat_i = u_rk + dt * torch.einsum('ij,jkl->ikl', self.rk_A, pde)  # as in eq. (22) in Raissi 2019
        # uhat_i = u_rk - dt * torch.einsum('ij,jkl->ikl', self.rk_A, pde)  # as in Raissi's PINN codebase
        uhat_f = u_rk + dt * torch.einsum('ij,jkl->ikl', (self.rk_A - self.rk_b.unsqueeze(0)), pde)

        # make graphs, if you want
        if do_graphs:
            make_dot(kappa, params=dict(list(self.named_parameters()))).render('graph/kappa', format='pdf', cleanup=True)
            make_dot(v1, params=dict(list(self.named_parameters()))).render('graph/v1', format='pdf', cleanup=True)
            make_dot(f, params=dict(list(self.named_parameters()))).render('graph/f', format='pdf', cleanup=True)
            make_dot(u_rk, params=dict(list(self.named_parameters()))).render('graph/u_rk', format='pdf', cleanup=True)
            make_dot(kappa_x, params=dict(list(self.named_parameters()))).render('graph/kappa_x', format='pdf', cleanup=True)
            make_dot(u_rk_x, params=dict(list(self.named_parameters()))).render('graph/u_rk_x', format='pdf', cleanup=True)
            make_dot(u_rk_xx, params=dict(list(self.named_parameters()))).render('graph/u_rk_xx', format='pdf', cleanup=True)
            make_dot(uhat_i, params=dict(list(self.named_parameters()))).render('graph/uhat_i', format='pdf', cleanup=True)
            make_dot(uhat_f, params=dict(list(self.named_parameters()))).render('graph/uhat_f', format='pdf', cleanup=True)

        return (uhat_i, uhat_f), (kappa, kappa_x, kappa_y), (v1, v1_x, v1_y), (v2, v2_x, v2_y), (f,)

    def predict(
            self,
            x: np.ndarray,
            y: np.ndarray,
            u: np.ndarray
        ):
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
        return torch2np(self.forward(torch.stack((x, y, *u)).contiguous()))

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            u: np.ndarray,
            weights: np.ndarray,
            mask_coast: np.ndarray,
            mask_other: np.ndarray,
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
            weights:        Weights of each term in the loss. These are NOT weights of the model.
            epochs:         Number of epochs to run.
            lr:             Learning rate passed to Adam optimizer.
            do_graphs:      Make torchviz graphs of the computational graph.
        """
        x, y, u = np2torch(x).requires_grad_(True), np2torch(y).requires_grad_(True), np2torch(u).requires_grad_(False)
        weights = np2torch(weights)
        mask_coast, mask_other = np2torch(mask_coast), np2torch(~mask_other)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # get true solution values
        u_i = u[len(u) // 2]
        u_f = u[len(u) // 2 + 1]

        self.train()
        losses = list()
        for e in range(epochs):
            print(f'Starting epoch {e}...', end='\r')

            optimizer.zero_grad()

            (uhat_i, uhat_f), (kappa, kappa_x, kappa_y), (v1, v1_x, v1_y), (v2, v2_x, v2_y), (f,) = \
                self.forward(x, y, u)

            # compute loss with estimate and actual solution
            loss_u_i = torch.sum(mask_other.unsqueeze(0) * (uhat_i - u_i.unsqueeze(0))**2)
            loss_u_f = torch.sum(mask_other.unsqueeze(0) * (uhat_f - u_f.unsqueeze(0))**2)

            # compute other loss terms
            loss_bc = torch.sum(mask_coast.unsqueeze(0) * (v1**2 + v2**2))
            loss_kappa_reg = torch.sum(mask_other.unsqueeze(0) * (kappa_x**2 + kappa_y**2))
            loss_v_reg = torch.sum(mask_other.unsqueeze(0) * (v1_x**2 + v1_y**2 + v2_x**2 + v2_y**2))
            loss_f_min = torch.sum(mask_other.unsqueeze(0) * (f**2))

            # compute final loss
            loss = torch.stack((loss_u_i, loss_u_f, loss_bc, loss_kappa_reg, loss_v_reg, loss_f_min)) @ weights

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if e % 10 == 0:
                print(
                    f'Epoch {e:5d}, loss {losses[-1]:10.2f}' +
                    (f', relative improvement {100 * (1 - losses[-1] / losses[-10]):10.2f}%' if e > 0 else '')
                )
                if Path('stop-training').exists():
                    break

        return losses
