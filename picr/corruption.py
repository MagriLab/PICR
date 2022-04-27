from typing import TypeVar

import numpy as np
import opt_einsum as oe
import torch

from .utils.checks import ValidateDimension

T = TypeVar('T', np.ndarray, torch.Tensor)


@ValidateDimension(ndim=3)
def ackley(x: T, freq: float) -> T:

    """Generic implementation of Ackley function -- parameterised by frequency.

    Parameters
    ----------
    x: T
        Spatial grid on which to compute the Ackley function.
    freq: float
        Parameterised frequency of the Ackley function.

    Returns
    -------
    fx: T
        Ackley function on given grid.
    """

    t1 = -0.2 * np.sqrt(oe.contract('iju -> ij', (x - np.pi) ** 2) / 2)
    t2 = oe.contract('iju -> ij', np.cos(freq * (x - np.pi))) / 2

    fx = -20 * np.exp(t1) - np.exp(t2) + 20 + np.exp(1)

    return fx


@ValidateDimension(ndim=3)
def rastrigin(x: T, freq: float) -> T:

    """Generic implementation of Rastrigin function -- parameterised by frequency.

    Parameters
    ----------
    x: T
        Spatial grid on which to compute the Rastrigin function.
    freq: float
        Parameterised frequency of the Rastrigin function.

    Returns
    -------
    T
        Rastrigin function on given grid.
    """

    return 10.0 * 2 + oe.contract('iju -> ij', x ** 2 - 10.0 * np.cos(freq * (x - np.pi)))
