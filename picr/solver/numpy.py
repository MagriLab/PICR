import numpy as np
import opt_einsum as oe

from kolsol.numpy.solver import KolSol


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

        super().__init__(nk, 0.0, re, ndim)

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

    def pressure(self, u_hat: np.ndarray) -> np.ndarray:
        raise NotImplementedError('LinearCDS::pressure()')
