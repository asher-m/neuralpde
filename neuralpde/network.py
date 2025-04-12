import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from pathlib import Path
from typing import List, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

DEVICE = 'cpu'  # TODO: remove this
warnings.warn('Forcing PyTorch on CPU!')


def RK(q: int = 100):
    """
    Get an implicit Runge-Kutta scheme with `q` stages.

    Default of 100 because similar problems seem to use about the same, judging from a
    cursory perusing of github/maziarraissa/PINNs and github/rezaakb/pinns-torch.

    Arguments:
        q:      Integer number of stages.
    """
    d = np.loadtxt(Path(__file__).parent / f'../raissi-2019/Utilities/IRK_weights/Butcher_IRK{q}.txt').astype(np.float32)
    A = d[:q**2].reshape((q, q))
    b = d[q**2: q**2 + q]
    c = d[q**2 + q:]

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
    def __init__(self, loss, q: int, shape: Tuple[int], kernel: int = 5) -> None:
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

        self.loss = loss
        self.q = q
        self.shape = shape
        self.kernel = kernel

        assert self.kernel % 2 == 1, "Kernel size must be odd!"
        self.padding = self.kernel // 2

        # kappa and f are scalar fields, v is a 2d vector field
        self.kappa = nn.Parameter(torch.zeros(self.shape[1:]))
        self.v = nn.Parameter(torch.zeros((2,) + self.shape[1:]))
        self.f = nn.Parameter(torch.zeros(self.shape[1:]))

        # need 2 channels for spatial coordinates; need shape[0] channels for time axis of solution data 
        self.channels_in = 2 + self.shape[0]
        self.channels_hidden = 2**(int(np.log2(self.channels_in)) + 1)
        self.channels_out = q

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
        """
        return self.out(self.layers(data))

    def predict(self, x: np.ndarray, u: np.ndarray):
        """
        Push data through the network for evaluation.

        Arguments:
            x:          Spatial coordinates of each cell in the solution u.  Must be of size `((2,) + shape[1:])`
                        specified at model initialization.
            u:          Solution data of each cell.  Must be of size `shape` specified at model initialization.
        """
        self.eval()
        return torch2np(self.forward(torch.cat(np2torch((x, u)))))

    def train(self, x: np.ndarray, u: np.ndarray, epochs: int = 1000, lr: float = 1e-3):
        """
        Train the PINN.

        Arguments:
            x:          Spatial coordinates of each cell in the solution u.  Must be of size `((2,) + shape[1:])`
                        specified at model initialization.
            u:          Solution data of each cell.  Must be of size `shape` specified at model initialization.
            epochs:     Number of epochs to run.
            lr:         Learning rate passed to Adam optimizer.
        """
        # FIXME: this might all be nonsense and it'll just work without this
        x, u = np2torch(x).requires_grad_(True), np2torch(u).requires_grad_(False)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # get true solution values
        u_i = np2torch(u[len(u) // 2])
        u_f = np2torch(u[len(u) // 2 + 1])

        self.train()
        losses = list()
        for i in range(epochs):
            optimizer.zero_grad()

            uj = self.forward(torch.cat(x, u))
            loss = self.loss(self, x, uj, u_i, u_f)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if i % 10 == 0:
                print(f'Epoch {i:5d}, loss {losses[-1]:10.2f}')

        return losses
