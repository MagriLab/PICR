from typing import Union

import einops
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

        self.solver = LinearCDS(nk=nk, c=c, re=re, ndim=2, device=device)
        self.dt = dt

        self.fwt = torch.abs((fwt_lb ** -(1.0 / self.solver.nk)) ** -torch.sqrt(self.solver.kk / self.solver.ndim))

    @ValidateDimension(ndim=5)
    def calc_g_u_phi(self, u: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:

        nx = u.size(3)

        u = einops.rearrange(u, 'b t u i j -> b t i j u')
        phi = einops.rearrange(phi, 'b t u i j -> b t i j u')

        u_hat = self.solver.phys_to_fourier(u)
        phi_hat = self.solver.phys_to_fourier(phi)

        guphi_hat = self._g_u_phi(u_hat, phi_hat)

        guphi = self.solver.fourier_to_phys(guphi_hat, nx)
        guphi = oe.contract('b t i j u -> ', guphi ** 2) / guphi.numel()

        return guphi

    @ValidateDimension(ndim=5)
    def _g_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor) -> torch.Tensor:
        return self.solver.g_u_phi(u_hat, phi_hat)[:, :-1, ...]

    @ValidateDimension(ndim=5)
    def calc_residual(self, u: torch.Tensor) -> torch.Tensor:

        nx = u.size(3)

        u = einops.rearrange(u, 'b t u i j -> b t i j u')
        u_hat = self.solver.phys_to_fourier(u)

        residual_hat = self._residual(u_hat)

        residual = self.solver.fourier_to_phys(residual_hat, nx)
        residual = oe.contract('b t i j u -> ', residual ** 2) / residual.numel()

        return residual

    @ValidateDimension(ndim=5)
    def _residual(self, u_hat: torch.Tensor) -> torch.Tensor:

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
        a_dudt_hat = self.solver.dynamics(u_hat).to(u_hat.dtype)
        a_dudt_hat = a_dudt_hat[:, :-1, ...]

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

        self.solver = NonLinearKFS(nk=nk, nf=nf, re=re, ndim=2, device=device)
        self.dt = dt

        self.fwt = torch.abs((fwt_lb ** -(1.0 / self.solver.nk)) ** -torch.sqrt(self.solver.kk / self.solver.ndim))

    @ValidateDimension(ndim=5)
    def calc_g_u_phi(self, u: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:

        nx = u.size(3)

        u = einops.rearrange(u, 'b t u i j -> b t i j u')
        phi = einops.rearrange(phi, 'b t u i j -> b t i j u')

        u_hat = self.solver.phys_to_fourier(u)
        phi_hat = self.solver.phys_to_fourier(phi)

        guphi_hat = self._g_u_phi(u_hat, phi_hat)

        guphi = self.solver.fourier_to_phys(guphi_hat, nx)
        guphi = oe.contract('b t i j u -> ', guphi ** 2) / guphi.numel()

        return guphi

    @ValidateDimension(ndim=5)
    def _g_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor) -> torch.Tensor:
        return self.solver.g_u_phi(u_hat, phi_hat)[:, :-1, ...]

    @ValidateDimension(ndim=5)
    def calc_residual(self, u: torch.Tensor) -> torch.Tensor:

        nx = u.size(3)

        u = einops.rearrange(u, 'b t u i j -> b t i j u')
        u_hat = self.solver.phys_to_fourier(u)

        residual_hat = self._residual(u_hat)

        residual = self.solver.fourier_to_phys(residual_hat, nx)
        residual = oe.contract('b t i j u -> ', residual ** 2) / residual.numel()

        return residual

    @ValidateDimension(ndim=5)
    def _residual(self, u_hat: torch.Tensor) -> torch.Tensor:

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
        a_dudt_hat = self.solver.dynamics(u_hat).to(u_hat.dtype)
        a_dudt_hat = a_dudt_hat[:, :-1, ...]

        # compute empirical derivatives
        e_dudt_hat = (1.0 / self.dt) * (u_hat[:, 1:, ...] - u_hat[:, :-1, ...])

        # scale derivative fields
        fwt_a_dudt_hat = oe.contract('ij, btiju -> btiju', self.fwt, a_dudt_hat)
        fwt_e_dudt_hat = oe.contract('ij, btiju -> btiju', self.fwt, e_dudt_hat)

        residual = fwt_a_dudt_hat - fwt_e_dudt_hat

        return residual
