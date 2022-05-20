from typing import Union

import einops
import opt_einsum as oe
import torch

from .solvers.torch import LinearCDS, NonlinearCDS
from .utils.checks import ValidateDimension


class LinearCDLoss:

    def __init__(self,
                 nk: int,
                 c: float,
                 re: float,
                 dt: float,
                 fwt_lb: float = 1.0,
                 device: Union[torch.device, str] = torch.device('cpu')) -> None:

        """Calculate Losses for Linear Convection-Diffusion Problem.

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
    def _linear_residual(self, t_hat: torch.Tensor) -> torch.Tensor:

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

        res_u = self._linear_residual(u_hat)
        loss = oe.contract('... -> ', res_u * torch.conj(res_u)) / res_u.numel()

        return loss

    @ValidateDimension(ndim=5)
    def calc_g_loss(self, u: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:

        """Calculates the g-loss based on matching residuals.

        Note: in the linear case, this equates to finding the residual of the
              velocity field -- there is no interaction of non-linearities.

        Parameters
        ----------
        u: torch.Tensor
            Velocity field in the physical domain.
        phi: torch.Tensor
            Corruption field in the physical domain.
            Note: this is not used in the linear case.

        Returns
        -------
        torch.Tensor
            g-loss in the Fourier domain.
        """

        return self.calc_residual_loss(u)


class NonlinearCDLoss:

    def __init__(self,
                 nk: int,
                 re: float,
                 dt: float,
                 fwt_lb: float = 1.0,
                 device: Union[torch.device, str] = torch.device('cpu')) -> None:

        """Calculate Losses for Nonlinear Convection-Diffusion Problem.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        re: float
            Reynolds number of the flow.
        dt: float
            Size of time-step.
        fwt_lb: float
            Fourier weighting term - lower bound.
        device: Union[torch.device, str]
            Device on which to initialise the tensors.
        """

        self.solver = NonlinearCDS(nk=nk, re=re, ndim=2, device=device)
        self.dt = dt

        self.fwt = torch.abs((fwt_lb ** -(1.0 / self.solver.nk)) ** -torch.sqrt(self.solver.kk / self.solver.ndim))

    @ValidateDimension(ndim=5)
    def _x_dot_nabla_y(self, x_hat: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:

        """Computes [x \cdot \nabla] y in the Fourier domain.

        Parameters
        ----------
        x_hat: torch.Tensor
            Tensor, in the Fourier domain, to take the scalar product with nabla.
        y_hat: torch.Tensor
            Free-standing tensor in the Fourier domain.

        Returns
        -------
        xdny: torch.Tensor
            Computed [x \cdot \nabla] y in the Fourier domain.
        """

        # TODO :: Add option to return y_dot_nabla_x as well

        uij_aapt = []
        for u_i in range(self.solver.ndim):

            uj_aapt = []
            for u_j in range(self.solver.ndim):
                uj_aapt.append(self.solver.aap(x_hat[..., u_i], y_hat[..., u_j]))

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        aapt = torch.stack(uij_aapt, dim=0)

        xdny = oe.contract('...t, tu... -> ...u', self.solver.nabla, aapt)

        return xdny

    @ValidateDimension(ndim=5)
    def _linear_residual(self, t_hat: torch.Tensor) -> torch.Tensor:

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

        return residual

    @ValidateDimension(ndim=5)
    def _nonlinear_residual(self, t_hat: torch.Tensor, p_hat: torch.Tensor) -> torch.Tensor:

        """Computes a residual of the form: R(t + p) in the Fourier domain.

        Parameters
        ----------
        t_hat: torch.Tensor
            First tensor in non-linearity - must be in the Fourier domain.
        p_hat: torch.Tensor
            Second tensor in the non-linearity - must be in the Fourier domain.

        Returns
        -------
        residual: torch.Tensor
            Non-linear residual in the Fourier domain.
        """

        res_t = self._linear_residual(t_hat)
        res_p = self._linear_residual(p_hat)

        t_dot_nabla_p = self._x_dot_nabla_y(t_hat, p_hat)
        p_dot_nabla_t = self._x_dot_nabla_y(p_hat, t_hat)

        residual = res_t + res_p + t_dot_nabla_p + p_dot_nabla_t

        return residual

    @ValidateDimension(ndim=5)
    def _g_u_phi(self, u_hat: torch.Tensor, phi_hat: torch.Tensor) -> torch.Tensor:

        """Computes g(u, \phi) for the given problem in the Fourier domain.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.
        phi_hat: torch.Tensor
            Corruption field in the Fourier domain.

        Returns
        -------
        guphi: torch.Tensor
            Calculated g(u, \phi) in the Fourier domain.
        """

        res_u = self._linear_residual(u_hat)

        u_dot_nabla_phi = self._x_dot_nabla_y(u_hat, phi_hat)
        phi_dot_nabla_u = self._x_dot_nabla_y(phi_hat, u_hat)

        guphi = res_u + u_dot_nabla_phi + phi_dot_nabla_u

        return guphi

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

        res_u = self._linear_residual(u_hat)
        loss = oe.contract('... -> ', res_u * torch.conj(res_u)) / res_u.numel()

        return loss

    @ValidateDimension(ndim=5)
    def calc_g_loss(self, u: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:

        """Calculates the g-loss based on matching residuals.

        Parameters
        ----------
        u: torch.Tensor
            Velocity field in the physical domain.
        phi: torch.Tensor
            Corruption field in the physical domain.

        Returns
        -------
        loss: torch.Tensor
            g-loss in the Fourier domain.
        """

        u = einops.rearrange(u, 'b t u i j -> b t i j u')
        phi = einops.rearrange(phi, 'b t u i j -> b t i j u')

        u_hat = self.solver.phys_to_fourier(u)
        phi_hat = self.solver.phys_to_fourier(phi)

        res_zeta = self._nonlinear_residual(u_hat, phi_hat)
        res_phi = self._linear_residual(phi_hat)
        res_guphi = self._g_u_phi(u_hat, phi_hat)

        r_g = res_zeta - res_phi - res_guphi
        loss = oe.contract('... -> ', r_g * torch.conj(r_g)) / r_g.numel()

        return loss
