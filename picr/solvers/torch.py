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
        self.nu = 1.0 / re

        self.kk = self.kk.to(torch.cfloat)
        self.nabla = self.nabla.to(torch.cfloat)

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

        """Calculate g(u, \phi) for the linear case.

        Parameters
        ----------
        u_hat: torch.Tensor
            Predicted velocity field in the Fourier domain.
        phi_hat: torch.Tensor
            Predicted corruption field in the Fourier domain.

        Returns
        -------
        guphi: torch.Tensor
            Array g(u, \phi) -- zeros in the linear case.
        """

        if not u_hat.shape == phi_hat.shape:
            raise DimensionError(msg=f'u_hat.shape ({u_hat.shape}) should match phi_hat.shape ({phi_hat.shape})')

        guphi = torch.zeros_like(phi_hat)

        return guphi

    def pressure(self, u_hat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('LinearCDS::pressure()')


class NonLinearKFS(KolSol):

    def __init__(self,
                 nk: int,
                 nf: int,
                 re: float,
                 ndim: int = 2,
                 device: Union[torch.device, str] = torch.device('cpu')) -> None:

        """Non-Linear Kolmogorov Flow Solver.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        nf: int
            Prescribed forcing frequency.
        re: float
            Reynolds number of the flow.
        ndim: int, default=2
            Number of dimensions to solve for.
        device: Union[torch.device, str]
            Device on which to run solver.
        """

        super().__init__(nk=nk, nf=nf, re=re, ndim=ndim, device=device)

    def g_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor) -> torch.Tensor:

        """Calculate g(u, \phi) for the NS equations.

        Parameters
        ----------
        u_hat: torch.Tensor
            Predicted velocity field in the Fourier domain.
        phi_hat: torch.Tensor
            Predicted corruption field in the Fourier domain.

        Returns
        -------
        guphi: torch.Tensor
            Array g(u, \phi) -- calculated for NS equations.
        """

        # calculate u \cdot \nabla \phi, \phi \cdot \nabla u
        ij_aapt_phi_u = []
        for i in range(self.ndim):

            j_aapt_phi_u = []
            for j in range(self.ndim):
                j_aapt_phi_u.append(self.aap(u_hat[..., i], phi_hat[..., j]))

            ij_aapt_phi_u.append(torch.stack(j_aapt_phi_u, dim=0))

        aapt_phi_u = torch.stack(ij_aapt_phi_u, dim=0)

        phi_dot_nabla_u = oe.contract('...t, ut... -> ...u', self.nabla, aapt_phi_u)
        u_dot_nabla_phi = oe.contract('...t, tu... -> ...u', self.nabla, aapt_phi_u)

        nonlinear_term = u_dot_nabla_phi + phi_dot_nabla_u

        # calculate pressure terms
        p_u = self.p_u(u_hat)
        p_phi = self.p_phi(phi_hat)
        p_u_phi = self.p_u_phi(u_hat, phi_hat)

        pressure_term = oe.contract('iju, ij -> iju', self.nabla, p_u_phi - p_phi - p_u)

        # calculate g(u, \phi)
        guphi = nonlinear_term + pressure_term + self.f

        return guphi

    def p_u(self, u_hat: torch.Tensor) -> torch.Tensor:

        """Calculate pressure for the velocity field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.

        Returns
        -------
        p_hat: torch.Tensor
            Pressure field in the Fourier domain.
        """

        ij_aapt = []
        for i in range(self.ndim):

            j_aapt = []
            for j in range(self.ndim):
                j_aapt.append(self.aap(u_hat[..., j], u_hat[..., i]))

            ij_aapt.append(torch.stack(j_aapt, dim=0))

        aapt = torch.stack(ij_aapt, dim=0)
        f_hat = oe.contract('...t, ut... -> ...u', -self.nabla, aapt)

        p_hat = oe.contract('...u, ...u -> ...', -self.nabla, f_hat + self.f) / self.kk_div
        p_hat[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        return p_hat

    def p_phi(self, phi_hat: torch.Tensor) -> torch.Tensor:

        """Calculate pressure for the corruption field.

        Parameters
        ----------
        phi_hat: torch.Tensor
            Corruption field in the Fourier domain.

        Returns
        -------
        p_hat: torch.Tensor
            Pressure field in the Fourier domain.
        """

        ij_aapt = []
        for i in range(self.ndim):

            j_aapt = []
            for j in range(self.ndim):
                j_aapt.append(self.aap(phi_hat[..., j], phi_hat[..., i]))

            ij_aapt.append(torch.stack(j_aapt, dim=0))

        aapt = torch.stack(ij_aapt, dim=0)

        f_hat = oe.contract('...t, ut... -> ...u', -self.nabla, aapt)

        # TODO :: Do we need to include time derivative term here?
        brackets = (1.0 / self.re) * oe.contract('ij, iju -> iju', self.kk, phi_hat) - f_hat - self.f

        p_hat = oe.contract('...u, ...u -> ...', self.nabla, brackets) / self.kk_div
        p_hat[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        return p_hat

    def p_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor) -> torch.Tensor:

        """Calculate pressure for corrupted velocity field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.
        phi_hat: torch.Tensor
            Corruption field in the Fourier domain.

        Returns
        -------
        p_hat: torch.Tensor
            Pressure field in the Fourier domain.
        """

        zeta_hat = u_hat + phi_hat

        ij_aapt = []
        for i in range(self.ndim):

            j_aapt = []
            for j in range(self.ndim):
                j_aapt.append(self.aap(zeta_hat[..., j], zeta_hat[..., i]))

            ij_aapt.append(torch.stack(j_aapt, dim=0))

        aapt = torch.stack(ij_aapt, dim=0)
        f_hat = oe.contract('...t, ut... -> ...u', -self.nabla, aapt)

        # TODO :: Do we need to include time derivative term here?
        brackets = (1.0 / self.re) * oe.contract('ij, iju -> iju', self.kk, phi_hat) - f_hat - self.f

        p_hat = oe.contract('...u, ...u -> ...', self.nabla, brackets) / self.kk_div
        p_hat[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        return p_hat
