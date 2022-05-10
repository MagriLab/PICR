from typing import Union

import opt_einsum as oe
import torch
from kolsol.torch.solver import KolSol

from ..utils.exceptions import DimensionError


class LinearCDS(KolSol):

    def __init__(self,
                 nk: int,
                 c: float,
                 re: float,
                 ndim: int = 2,
                 device: Union[torch.device, str] = torch.device('cpu')) -> None:

        """Linear Convection-Diffusion Solver.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        c: float
            Convection coefficient.
        re: float
            Reynolds number of the flow.
        ndim: int, default=2
            Number of dimensions to solve for.
        device: Union[torch.device, str]
            Device on which to run solver.
        """

        super().__init__(nk=nk, nf=0.0, re=re, ndim=ndim, device=device)

        self.c = c
        self.nu = 1.0 / self.re

    def dynamics(self, u_hat: torch.Tensor) -> torch.Tensor:

        """Calculate the time-derivative of the velocity field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.

        Returns
        -------
        dudt: torch.Tensor
            Time-derivative of the velocity field in the Fourier domain.
        """

        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for _ in range(self.ndim):
                uj_aapt.append(u_hat[..., u_i])

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        aapt = torch.stack(uij_aapt, dim=0)

        convective_term = self.c * oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        diffusive_term = self.nu * oe.contract('..., ...u -> ...u', self.kk, u_hat)

        dudt = -convective_term - diffusive_term

        return dudt

    @staticmethod
    def g_u_phi(u_hat: torch.Tensor, phi_hat: torch.Tensor) -> torch.Tensor:

        """Calculate g(u, \phi) for the linear convection diffusion equations.

        Parameters
        ----------
        u_hat: torch.Tensor
            Predicted velocity field in the Fourier domain.
        phi_hat: torch.Tensor
            Predicted corruption field in the Fourier domain.

        Returns
        -------
        guphi: torch.Tensor
            g(u, \phi) for the linear convection diffusion equations.
        """

        if not u_hat.shape == phi_hat.shape:
            raise DimensionError(msg=f'u_hat.shape ({u_hat.shape}) should match phi_hat.shape ({phi_hat.shape})')

        guphi = torch.zeros_like(phi_hat)

        return guphi

    def pressure(self, u_hat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('LinearCDS::pressure()')


class NonlinearCDS(KolSol):

    def __init__(self,
                 nk: int,
                 re: float,
                 ndim: int = 2,
                 device: Union[torch.device, str] = torch.device('cpu')) -> None:

        """Non-Linear Convection Diffusion Solver.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        re: float
            Reynolds number of the flow.
        ndim: int, default=2
            Number of dimensions to solve for.
        device: Union[torch.device, str]
            Device on which to run solver.
        """

        super().__init__(nk=nk, nf=0.0, re=re, ndim=ndim, device=device)

        self.nu = 1.0 / self.re

    def dynamics(self, u_hat: torch.Tensor) -> torch.Tensor:

        """Calculate the time-derivative of the velocity field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.

        Returns
        -------
        dudt: torch.Tensor
            Time-derivative of the velocity field in the Fourier domain.
        """

        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(u_hat[..., u_j], u_hat[..., u_i]))

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        aapt = torch.stack(uij_aapt, dim=0)

        u_dot_nabla_u = oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        laplace_u = self.nu * oe.contract('..., ...u -> ...u', self.kk, u_hat)

        dudt = - u_dot_nabla_u - laplace_u

        return dudt

    def g_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor, dt: float) -> torch.Tensor:

        """Calculate g(u, \phi) for the nonlinear convection diffusion equations.

        Parameters
        ----------
        u_hat: torch.Tensor
            Predicted velocity field in the Fourier domain.
        phi_hat: torch.Tensor
            Predicted corruption field in the Fourier domain.
        dt: float
            Time-step of the simulation.

        Returns
        -------
        guphi: torch.Tensor
            g(u, \phi) for the nonlinear convection diffusion equations.
        """

        # calculate u \cdot \nabla \phi, \phi \cdot \nabla u
        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(u_hat[..., u_j], phi_hat[..., u_i]))

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        aapt = torch.stack(uij_aapt, dim=0)

        u_dot_nabla_phi = oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        phi_dot_nabla_u = oe.contract('...t, tu... -> ...u', self.nabla, aapt)

        # calculate u \cdot \nabla \u
        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(u_hat[..., u_j], u_hat[..., u_i]))

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        aapt = torch.stack(uij_aapt, dim=0)

        u_dot_nabla_u = oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        laplacian_u = self.nu * oe.contract('..., ...u -> ...u', self.kk, u_hat)

        # time derivative and slice
        du_dt = (1.0 / dt) * (u_hat[:, 1:, ...] - u_hat[:, :-1, ...])
        u_dot_nabla_u = u_dot_nabla_u[:, :-1, ...]
        u_dot_nabla_phi = u_dot_nabla_phi[:, :-1, ...]
        phi_dot_nabla_u = phi_dot_nabla_u[:, :-1, ...]
        laplacian_u = laplacian_u[:, :-1, ...]

        guphi = du_dt + u_dot_nabla_u + u_dot_nabla_phi + phi_dot_nabla_u - laplacian_u

        return guphi

    def h_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor, dt: float) -> torch.Tensor:

        """Calculate h(u, \phi) for the nonlinear convection diffusion equations.

        Parameters
        ----------
        u_hat: torch.Tensor
            Predicted velocity field in the Fourier domain.
        phi_hat: torch.Tensor
            Predicted corruption field in the Fourier domain.
        dt: float
            Time-step of the simulation.

        Returns
        -------
        huphi: torch.Tensor
            h(u, \phi) for the nonlinear convection diffusion equations.
        """

        # calculate u \cdot \nabla \phi, \phi \cdot \nabla u
        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(u_hat[..., u_j], phi_hat[..., u_i]))

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        aapt = torch.stack(uij_aapt, dim=0)

        u_dot_nabla_phi = oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        phi_dot_nabla_u = oe.contract('...t, tu... -> ...u', self.nabla, aapt)

        # calculate \phi \cdot \nabla \phi
        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(phi_hat[..., u_j], phi_hat[..., u_i]))

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        aapt = torch.stack(uij_aapt, dim=0)

        phi_dot_nabla_phi = oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        laplacian_phi = self.nu * oe.contract('..., ...u -> ...u', self.kk, phi_hat)

        # time derivative and slice
        dphi_dt = (1.0 / dt) * (phi_hat[:, 1:, ...] - phi_hat[:, :-1, ...])
        u_dot_nabla_phi = u_dot_nabla_phi[:, :-1, ...]
        phi_dot_nabla_u = phi_dot_nabla_u[:, :-1, ...]
        phi_dot_nabla_phi = phi_dot_nabla_phi[:, :-1, ...]
        laplacian_phi = laplacian_phi[:, :-1, ...]

        huphi = dphi_dt + u_dot_nabla_phi + phi_dot_nabla_u + phi_dot_nabla_phi - laplacian_phi

        return huphi

    def pressure(self, u_hat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('NonlinearCDS::pressure()')
