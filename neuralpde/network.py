import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from pathlib import Path
from typing import List, Tuple

from torchviz import make_dot

from . import layer



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# dt = 1, right?
dt = 1



def normalize_xy(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Normalize spatial coordinates (from meters to unitless dimension on the interval [-1, 1]).

    Returns a tuple of ((scalex, scaley), (x_normalized, y_normalized)).
    """
    assert x.ndim == 1 and y.ndim == 1, "I don't know how to handle multi-D arrays!"
    scalex, scaley = np.ptp(x), np.ptp(y)
    return (scalex, scaley), ((x - np.mean(x)) / scalex, (y - np.mean(y)) / scaley)


def normalize_data(u: np.ndarray):
    raise ValueError('Sea ice data already normalized!')


def RK(q: int = 100) -> Tuple[torch.Tensor]:
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


def np2torch(d, dtype=DTYPE) -> torch.Tensor:
    """
    Export numpy data to torch in every meaningful way, including sending it to the
    compute accelerator and casting it to the appropriate datatype.
    """
    return torch.from_numpy(d).to(DEVICE, dtype)


def torch2np(d) -> np.ndarray:
    """
    Export torch data back to numpy.
    """
    return d.detach().cpu().numpy()


class Network(nn.Module):
    def __init__(
            self,
            q: int,
            Nt: int,
            x_range: np.ndarray,
            y_range: np.ndarray,
            kernel_xy: int,
            kernel_stack: int = 10
    ) -> None:
        """
        Initialize the PINN.

        Arguments:
            q:                  Number of stages q used in RK scheme to compute loss.  See Raissi 2019.
            Nt:                 Number of solution maps (i.e., in time) included in buffer.  The network is conditioned on these known solutions.
            x_range:            The x coordinate range as a 1-D array.
            y_range:            The y coordinate range as a 1-D array.
            kernel_xy:          Size of kernel over which to convolve.
            kernel_stack:       Number of neurons in the FC stack per output channel.
        Nt must be the same for every batch, data for which must be loaded through `Network.load_data`.
        """
        super().__init__()

        self.q = q
        self.Nt = Nt
        self.dx = np.diff(x_range)[0]
        assert np.all(np.isclose(np.diff(x_range), self.dx)), 'Received irregularly shaped x_range!'
        self.dy = np.diff(y_range)[0]
        assert np.all(np.isclose(np.diff(y_range), self.dy)), 'Received irregularly shaped y_range!'
        self.kernel = kernel_xy
        assert kernel_xy % 2 == 1, 'Kernel size must be odd!'

        # make kernel offsets
        self.offsets_xy = nn.Buffer(
            np2torch(
                ((np.indices((kernel_xy, kernel_xy)) - kernel_xy // 2) * \
                 np.array((self.dx, self.dy))[:, None, None]).transpose((1, 2, 0))
            )
        )

        self.rk_A, self.rk_b, self.rk_c = map(nn.Buffer, RK(q))

        self.channels = 4 + q  # parameters + rk stages
        self.spatial_correlation = layer.GaussianDistanceWeight(  # recreate ranges to handle padding
            (
                torch.linspace(x_range[0] - kernel_xy//2 * self.dx, x_range[-1] + kernel_xy//2 * self.dx, len(x_range) + kernel_xy - 1),
                torch.linspace(y_range[0] - kernel_xy//2 * self.dy, y_range[-1] + kernel_xy//2 * self.dy, len(y_range) + kernel_xy - 1)
            )
        )
        nn.init.constant_(self.spatial_correlation.width, min(abs(self.dx), abs(self.dy)) / 1.5)
        self.padding = nn.ReflectionPad2d(kernel_xy//2)
        self.layers = nn.Sequential(
            nn.Linear(Nt * kernel_xy**2, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels)
        )


    def data_(
            self,
            u: np.ndarray,
    ) -> None:
        """
        Load sea ice data into the network.

        Arguments:
            u:      Data conditioning the model.
        """
        if u.ndim == 4:
            data = np2torch(u).contiguous()
        elif u.ndim == 3:  # i.e., batch_size = 1
            data = np2torch(u).unsqueeze(0).contiguous()
        else:
            raise ValueError('Receieved data of an incompatible shape.  Check docstring!')
        self.data = nn.Buffer(self.padding(data))


    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            batch_num: int = 0,
    ) -> torch.Tensor:
        """
        Push data through the network for training.

        Arguments:
            x:          1-D tensor of x-values at which to evaluate the solution of length batch size.
            y:          1-D tensor of y-values at which to evaluate the solution of length batch size.
            batch_num:  Index of self.data to use.  Only used when x and y are scalars.

        Returns a tensor on `DEVICE`.
        """
        if x.ndim > 0:  # non-scalar case
            r = self.spatial_correlation(torch.stack((x, y), axis=-1)[:, None, None, :] + self.offsets_xy[None, ...])
            r = r[:, None, ...] * self.data[:, :, None, None, ...]
            r = torch.sum(r, dim=(-1, -2))
            r = r.flatten(1, -1)
            r = self.layers(r)
            return r
        else:
            r = self.spatial_correlation(torch.stack((x, y))[None, None, :] + self.offsets_xy)
            r = r[None, ...] * self.data[batch_num, :, None, None, ...]
            r = torch.sum(r, dim=(-1, -2))
            r = r.flatten(0, -1)
            r = self.layers(r)
            return r


    def predict(
            self,
            x_range: np.ndarray,
            y_range: np.ndarray,
            batch_size: int = 100
    ):
        """
        Push data through the network for evaluation.
        """
        self.eval()
        x, y, u = np2torch(x).requires_grad_(True), np2torch(y).requires_grad_(True), np2torch(u).requires_grad_(False)
        return map(torch2np, self.forward(x, y, u))

        results = {
            'k': list(),
            'v1': list(),
            'v2': list(),
            'f': list(),
            'uhat_i': list(),
            'uhat_f': list(),
        }
        indices = np.random.permutation(np.indices((len(x_range), len(y_range))).reshape((2, -1)).T)
        x_range = np2torch(x_range).requires_grad_(True)
        y_range = np2torch(y_range).requires_grad_(True)
        for b in range(len(indices) // batch_size + 1):
            idx_x = np2torch(indices[b*batch_size:(b+1)*batch_size, 0], dtype=torch.int)
            idx_y = np2torch(indices[b*batch_size:(b+1)*batch_size, 1], dtype=torch.int)
            x_fit = x_range[idx_x]
            y_fit = y_range[idx_y]

            r = torch.func.vmap(self.forward)(x_fit, y_fit)
            r_x = torch.func.vmap(torch.func.jacrev(self.forward, 0))(x_fit, y_fit)
            r_y = torch.func.vmap(torch.func.jacrev(self.forward, 1))(x_fit, y_fit)
            r_xx = torch.func.vmap(torch.func.jacrev(torch.func.jacrev(self.forward, 0), 0))(x_fit, y_fit)
            r_yy = torch.func.vmap(torch.func.jacrev(torch.func.jacrev(self.forward, 1), 1))(x_fit, y_fit)

            k, v1, v2, f, urk = r[..., 0], r[..., 1], r[..., 2], r[..., 3], r[..., 4:]
            k_x, v1_x, v2_x, urk_x = r_x[..., 0], r_x[..., 1], r_x[..., 2], r_x[..., 4:]
            k_y, v1_y, v2_y, urk_y = r_y[..., 0], r_y[..., 1], r_y[..., 2], r_y[..., 4:]
            urk_xx = r_xx[..., 4:]
            urk_yy = r_yy[..., 4:]

            pde = (  # not particularly pythonic, but easier to read
                k.unsqueeze(-1) * (urk_xx + urk_yy) + (k_x.unsqueeze(-1) * urk_x + k_y.unsqueeze(-1) * urk_y)
                - (v1_x.unsqueeze(-1) + v2_y.unsqueeze(-1)) * urk - (v1.unsqueeze(-1) * urk_x + v2.unsqueeze(-1) * urk_y)
                + f
            )

            uhat_i = urk - dt * torch.einsum('ij,bj->bi', self.rk_A, pde)
            uhat_f = urk - dt * torch.einsum('ij,bj->bi', self.rk_A - self.rk_b.unsqueeze(0), pde)

            results['k'].append(k)
            results['v1'].append(v1)
            results['v2'].append(v2)
            results['f'].append(f)
            results['uhat_i'].append(uhat_i)
            results['uhat_f'].append(uhat_f)

        for key in results.keys():
            results[key] = torch2np(torch.cat(results[key])).reshape((len(x_range), len(y_range), -1))

        return results


    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            u: np.ndarray,
            weights: np.ndarray,
            mask_coast: np.ndarray,
            mask_other: np.ndarray,
            epochs: int = 1000,
            lr: float = 1e-3,
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

        # touch training file (if this disappears, we stop)
        _training = Path('training')
        _training.touch()

        self.train()
        losses = list()
        for e in range(epochs):
            print(f'Starting epoch {e}...', end='\r')

            optimizer.zero_grad()

            uhat_i, uhat_f, kappa, kappa_x, kappa_y, v1, v1_x, v1_y, v2, v2_x, v2_y, f = \
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
                if not _training.exists():  # stop if training file disappeared
                    break

        _training.unlink(missing_ok=True)

        return losses
