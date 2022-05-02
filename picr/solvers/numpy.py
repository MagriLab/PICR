import numpy as np
import opt_einsum as oe

from kolsol.numpy.solver import KolSol

from ..utils.exceptions import DimensionError


class LinearCDS(KolSol):

    def __init__(self, nk: int, c: float, re: float, ndim: int = 2) -> None:

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
        """

        super().__init__(nk=nk, nf=0.0, re=re, ndim=ndim)

        self.c = c
        self.nu = 1.0 / re

    def dynamics(self, u_hat: np.ndarray) -> np.ndarray:

        """Calculate the time-derivative of the velocity field.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.

        Returns
        -------
        dudt: np.ndarray
            Time-derivative of the velocity field in the Fourier domain.
        """

        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for _ in range(self.ndim):
                uj_aapt.append(u_hat[..., u_i])

            uij_aapt.append(np.stack(uj_aapt, axis=0))

        aapt = np.stack(uij_aapt, axis=0)

        convective_term = self.c * oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        diffusive_term = self.nu * oe.contract('..., ...u -> ...u', self.kk, u_hat)

        dudt = -convective_term - diffusive_term

        return dudt

    @staticmethod
    def g_u_phi(u_hat: np.ndarray, phi_hat: np.ndarray) -> np.ndarray:

        """Calculate g(u, \phi) for the linear convection diffusion equations.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.
        phi_hat: np.ndarray
            Corruption field in the Fourier domain.

        Returns
        -------
        guphi: np.ndarray
            g(u, \phi) for the linear convection diffusion equations.
        """

        if not u_hat.shape == phi_hat.shape:
            raise DimensionError(msg=f'u_hat.shape ({u_hat.shape}) should match phi_hat.shape ({phi_hat.shape})')

        guphi = np.zeros_like(phi_hat)

        return guphi

    def pressure(self, u_hat: np.ndarray) -> np.ndarray:
        raise NotImplementedError('LinearCDS::pressure()')


class NonlinearCDS(KolSol):

    def __init__(self,
                 nk: int,
                 re: float,
                 ndim: int = 2) -> None:

        """Non-Linear Convection Diffusion Solver.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        re: float
            Reynolds number of the flow.
        ndim: int, default=2
            Number of dimensions to solve for.
        """

        super().__init__(nk=nk, nf=0.0, re=re, ndim=ndim)

        self.nu = 1.0 / self.re

    def dynamics(self, u_hat: np.ndarray) -> np.ndarray:

        """Calculate the time-derivative of the velocity field.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.

        Returns
        -------
        dudt: np.ndarray
            Time-derivative of the velocity field in the Fourier domain.
        """

        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(u_hat[..., u_j], u_hat[..., u_i]))

            uij_aapt.append(np.stack(uj_aapt, axis=0))

        aapt = np.stack(uij_aapt, axis=0)

        u_dot_nabla_u = oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        laplace_u = self.nu * oe.contract('..., ...u -> ...u', self.kk, u_hat)

        dudt = - u_dot_nabla_u - laplace_u

        return dudt

    def g_u_phi(self, u_hat: np.ndarray, phi_hat: np.ndarray) -> np.ndarray:

        """Calculate g(u, \phi) for the nonlinear convection diffusion equations.

        Parameters
        ----------
        u_hat: np.ndarray
            Predicted velocity field in the Fourier domain.
        phi_hat: np.ndarray
            Predicted corruption field in the Fourier domain.

        Returns
        -------
        guphi: np.ndarray
            g(u, \phi) calculated for the nonlinear convection diffusion equations.
        """

        # calculate u \cdot \nabla \phi, \phi \cdot \nabla u
        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(u_hat[..., u_j], phi_hat[..., u_i]))

            uij_aapt.append(np.stack(uj_aapt, axis=0))

        aapt = np.stack(uij_aapt, axis=0)

        u_dot_nabla_phi = oe.contract('...t, ut... -> ...u', self.nabla, aapt)
        phi_dot_nabla_u = oe.contract('...t, tu... -> ...u', self.nabla, aapt)

        guphi = u_dot_nabla_phi + phi_dot_nabla_u

        return guphi

    def pressure(self, u_hat: np.ndarray) -> np.ndarray:
        raise NotImplementedError('NonlinearCDS::pressure()')
