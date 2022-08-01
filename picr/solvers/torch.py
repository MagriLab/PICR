from typing import Union

import opt_einsum as oe
import torch
from kolsol.torch.solver import KolSol


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

        self.kk: torch.Tensor = self.kk.to(torch.cfloat)
        self.nabla: torch.Tensor = self.nabla.to(torch.cfloat)

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

        dudt = -1.0 * u_dot_nabla_u - laplace_u

        return dudt
