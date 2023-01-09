from types import ModuleType
from typing import Callable

import numpy as np
import opt_einsum as oe
import torch

from .utils.checks import ValidateDimension
from .utils.enums import eCorruption
from .utils.types import T


def get_corruption_fn(e_corruption: eCorruption) -> Callable[[T, float, float], T]:

    """Corruption Function Factory.

    Parameters
    ----------
    e_corruption: eCorruption
        Type of corruption function to return.

    Returns
    -------
    Callable[[T, float, float], T]
        Corruption function.
    """

    match e_corruption:

        case eCorruption.ackley:
            return ackley

        case eCorruption.rastrigin:
            return rastrigin

        case _:
            raise ValueError('Incompatible corruption function.')


@ValidateDimension(ndim=3)
def ackley(x: T, freq: float, limit: float = 1.0) -> T:

    """Generic implementation of Ackley function -- parameterised by frequency.

    Parameters
    ----------
    x: T
        Spatial grid on which to compute the Ackley function.
    freq: float
        Parameterised frequency of the Ackley function.
    limit: float
        Largest value in the corruption field.

    Returns
    -------
    fx: T
        Ackley function on given grid.
    """

    lib: ModuleType
    if isinstance(x, np.ndarray):
        lib = np
    elif isinstance(x, torch.Tensor):
        lib = torch
    else:
        raise ValueError('Unsupported data structure.')

    t1 = -0.2 * lib.sqrt(oe.contract('iju -> ij', (x - np.pi) ** 2) / 2)
    t2 = oe.contract('iju -> ij', lib.cos(freq * (x - np.pi))) / 2

    fx = -20 * lib.exp(t1) - lib.exp(t2) + 20 + lib.exp(1)

    fx = limit * (fx - lib.min(fx)) / (lib.max(fx) - lib.min(fx))

    return fx


@ValidateDimension(ndim=3)
def rastrigin(x: T, freq: float, limit: float = 1.0) -> T:

    """Generic implementation of Rastrigin function -- parameterised by frequency.

    Parameters
    ----------
    x: T
        Spatial grid on which to compute the Rastrigin function.
    freq: float
        Parameterised frequency of the Rastrigin function.
    limit: float
        Largest value in the corruption field.

    Returns
    -------
    T
        Rastrigin function on given grid.
    """

    lib: ModuleType
    if isinstance(x, np.ndarray):
        lib = np
    elif isinstance(x, torch.Tensor):
        lib = torch
    else:
        raise ValueError('Unsupported data structure.')

    val = 10.0 * 2 + oe.contract('iju -> ij', (x - np.pi) ** 2 - 10.0 * lib.cos(freq * (x - np.pi)))
    val = limit * (val - lib.min(val)) / (lib.max(val) - lib.min(val))

    return val