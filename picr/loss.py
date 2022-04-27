from typing import Union

import opt_einsum as oe
import torch

from .solvers.torch import LinearCDS, NonLinearKFS
from .utils.checks import ValidateDimension


class LinearCDLoss:

    def __init__(self,
                 nk: int,
                 c: float,
                 re: float,
                 dt: float,
                 fwt_lb: float = 1.0,
                 device: Union[torch.device, str] = torch.device('cpu')) -> None:

        """Calculate Residuals for Linear Convection-Diffusion Problem.

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
        device: Union[torch.device, str]
            Device on which to initialise the tensors.
        """

        self.cds = LinearCDS(nk=nk, c=c, re=re, ndim=2, device=device)
        self.dt = dt

        self.fwt = torch.abs((fwt_lb ** -(1.0 / self.cds.nk)) ** -torch.sqrt(self.cds.kk / self.cds.ndim))

    @ValidateDimension(ndim=5)
    def g_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor) -> torch.Tensor:
        return self.cds.g_u_phi(u_hat, phi_hat)

    @ValidateDimension(ndim=5)
    def residual(self, u_hat: torch.Tensor) -> torch.Tensor:

        """Calculates residual of the given field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Field to calculate the residual of in the Fourier domain.

        Returns
        -------
        residual: torch.Tensor
            Residual of the field in the Fourier domain.
        """

        # compute analytical derivatives
        a_dudt_hat = self.cds.dynamics(u_hat).to(u_hat.dtype)
        a_dudt_hat = a_dudt_hat[:, 1:, ...]

        # compute empirical derivatives
        e_dudt_hat = (1.0 / self.dt) * (u_hat[:, 1:, ...] - u_hat[:, :-1, ...])

        # scale derivative fields
        fwt_a_dudt_hat = oe.contract('ij, btiju -> btiju', self.fwt, a_dudt_hat)
        fwt_e_dudt_hat = oe.contract('ij, btiju -> btiju', self.fwt, e_dudt_hat)

        residual = fwt_a_dudt_hat - fwt_e_dudt_hat

        return residual


class NonLinearKFLoss:

    def __init__(self,
                 nk: int,
                 nf: int,
                 re: float,
                 dt: float,
                 fwt_lb: float = 1.0,
                 device: Union[torch.device, str] = torch.device('cpu')) -> None:

        """BasePhysicsInformedLoss Class.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        nf: int
            Prescribed forcing frequency.
        re: float
            Reynolds number of the flow.
        dt: float
            Size of time-step.
        fwt_lb: float
            Fourier weighting term - lower bound.
        device: Union[torch.device, str]
            Device on which to initialise the tensors.
        """

        self.ks = NonLinearKFS(nk=nk, nf=nf, re=re, ndim=2, device=device)
        self.dt = dt

        self.fwt = torch.abs((fwt_lb ** -(1.0 / self.ks.nk)) ** -torch.sqrt(self.ks.kk / self.ks.ndim))

    @ValidateDimension(ndim=5)
    def g_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor) -> torch.Tensor:
        return self.ks.g_u_phi(u_hat, phi_hat)

    @ValidateDimension(ndim=5)
    def residual(self, u_hat: torch.Tensor) -> torch.Tensor:

        """Calculates residual of the given field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Field to calculate the residual of in the Fourier domain.

        Returns
        -------
        residual: torch.Tensor
            Residual of the field in the Fourier domain.
        """

        # compute analytical derivatives
        a_dudt_hat = self.ks.dynamics(u_hat).to(u_hat.dtype)
        a_dudt_hat = a_dudt_hat[:, 1:, ...]

        # compute empirical derivatives
        e_dudt_hat = (1.0 / self.dt) * (u_hat[:, 1:, ...] - u_hat[:, :-1, ...])

        # scale derivative fields
        fwt_a_dudt_hat = oe.contract('ij, btiju -> btiju', self.fwt, a_dudt_hat)
        fwt_e_dudt_hat = oe.contract('ij, btiju -> btiju', self.fwt, e_dudt_hat)

        residual = fwt_a_dudt_hat - fwt_e_dudt_hat

        return residual
