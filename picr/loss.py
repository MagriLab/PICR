from typing import Protocol

import einops
import opt_einsum as oe
import torch

from .solvers.torch import LinearCDS, NonlinearCDS
from .solvers.proto import Solver
from .utils.checks import ValidateDimension
from .utils.enums import eSolverFunction
from .utils.config import ExperimentConfig


class PILoss(Protocol):

    solver: Solver

    def _residual(self, t_hat: torch.Tensor) -> torch.Tensor:

        """Calculate physics-based residual from the Fourier field.

        Parameters
        ----------
        t_hat: torch.Tensor
            Tensor field to calculate residual of.

        Returns
        -------
        torch.Tensor
            Residual of the input field.
        """

    def calc_residual_loss(self, u: torch.Tensor) -> torch.Tensor:

        """Calculate residual loss for physical field.

        Parameters
        ----------
        u: torch.Tensor
            Field to calculate residual loss for.

        Returns
        -------
        torch.Tensor
            Residual of the input field.
        """


class BaseLoss:

    solver: Solver

    def __init__(self, dt: float, fwt_lb: float) -> None:

        """Base class for the Physics-Informed Loss.

        Parameters
        ----------
        dt: float
            Length of the time-step
        fwt_lb: float
            Fourier weighting term - lower bound.
        """

        self.dt = dt
        self.fwt_lb = fwt_lb

    @property
    def fwt(self) -> torch.Tensor:

        """Fourier weighting term.

        Returns
        -------
        torch.Tensor
            The Fourier weighting term, given the lower-bound.
        """

        return torch.abs((self.fwt_lb ** -(1.0 / self.solver.nk)) ** -torch.sqrt(self.solver.kk / self.solver.ndim))

    @ValidateDimension(ndim=5)
    def _residual(self, t_hat: torch.Tensor) -> torch.Tensor:

        """Calculate the residual of t_hat in the Fourier domain.

        Parameters
        ----------
        t_hat: torch.Tensor
            Tensor, t, in the Fourier domain.

        Returns
        -------
        residual: torch.Tensor
            Residual of tensor, t, in the Fourier domain.
        """

        # analytical derivative
        a_dudt_hat = self.solver.dynamics(t_hat)
        a_dudt_hat = a_dudt_hat[:, :-1, ...]

        # empirical derivative
        e_dudt_hat = (1.0 / self.dt) * (t_hat[:, 1:, ...] - t_hat[:, :-1, ...])

        residual = a_dudt_hat - e_dudt_hat
        residual = oe.contract('ij, btiju -> btiju', self.fwt, residual)

        return residual

    @ValidateDimension(ndim=5)
    def calc_residual_loss(self, u: torch.Tensor) -> torch.Tensor:

        """Calculate the L2 norm of a field, u, in the Fourier domain.

        Parameters
        ----------
        u: torch.Tensor
            Field in the physical domain.

        Returns
        -------
        loss: torch.Tensor
            L2 norm of the field, calculated in the Fourier domain.
        """

        u = einops.rearrange(u, 'b t u i j -> b t i j u')
        u_hat = self.solver.phys_to_fourier(u)

        res_u = self._residual(u_hat)
        loss = oe.contract('... -> ', res_u * torch.conj(res_u)) / res_u.numel()

        return loss


class LinearCDLoss(BaseLoss):

    def __init__(self, nk: int, c: float, re: float, dt: float, fwt_lb: float, device: torch.device) -> None:

        """Linear Convection-Diffusion Loss.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        c: float
            Convection coefficient.
        re: float
            Reynolds number of the flow.
        dt: float
            Length of the time-step.
        fwt_lb: float
            Fourier weighting term - lower bound.
        device: torch.device
            Device on which to initialise the tensors.
        """

        super().__init__(dt=dt, fwt_lb=fwt_lb)
        self.solver = LinearCDS(nk=nk, c=c, re=re, ndim=2, device=device)


class NonlinearCDLoss(BaseLoss):

    def __init__(self, nk: int, re: float, dt: float, fwt_lb: float, device: torch.device) -> None:

        """Non-linear Convection-Diffusion Loss.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        re: float
            Reynolds number of the flow.
        dt: float
            Length of the time-step.
        fwt_lb: float
            Fourier weighting term - lower bound.
        device: torch.device
            Device on which to initialise the tensors.
        """

        super().__init__(dt=dt, fwt_lb=fwt_lb)
        self.solver = NonlinearCDS(nk=nk, re=re, ndim=2, device=device)


def get_loss_fn(config: ExperimentConfig, device: torch.device) -> PILoss:

    """Get the relevant loss function.

    Parameters
    ----------
    config: ExperimentConfig
        Config used to generate loss function.
    device: torch.device
        Device on which to place the loss function.

    Returns
    -------
    PILoss
        Generated loss function.
    """

    if config.SOLVER_FN == eSolverFunction.LINEAR:
        return LinearCDLoss(nk=config.NK, c=config.C, re=config.RE, dt=config.DT, fwt_lb=config.FWT_LB, device=device)

    if config.SOLVER_FN == eSolverFunction.NONLINEAR:
        return NonlinearCDLoss(nk=config.NK, re=config.RE, dt=config.DT, fwt_lb=config.FWT_LB, device=device)

    raise ValueError('Incompatible Loss Type...')
