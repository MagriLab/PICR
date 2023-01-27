from typing import Callable

import numpy as np
import opt_einsum as oe
import torch

from .utils.checks import ValidateDimension
from .utils.enums import eCorruption, eCorruptionOperation


def get_corruption_fn(e_corruption: eCorruption) -> Callable[[torch.Tensor, float, float], torch.Tensor]:

    """Corruption Function Factory.

    Parameters
    ----------
    e_corruption: eCorruption
        Type of corruption function to return.

    Returns
    -------
    Callable[[torch.Tensor, float, float], torch.Tensor]
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
def ackley(x: torch.Tensor, freq: float, limit: float = 1.0) -> torch.Tensor:

    """Generic implementation of Ackley function -- parameterised by frequency.

    Parameters
    ----------
    x: torch.Tensor
        Spatial grid on which to compute the Ackley function.
    freq: float
        Parameterised frequency of the Ackley function.
    limit: float
        Largest value in the corruption field.

    Returns
    -------
    fx: torch.Tensor
        Ackley function on given grid.
    """

    t1 = -0.2 * torch.sqrt(oe.contract('iju -> ij', (x - np.pi) ** 2) / 2)
    t2 = oe.contract('iju -> ij', torch.cos(freq * (x - np.pi))) / 2

    fx = -20 * torch.exp(t1) - torch.exp(t2) + 20 + torch.exp(torch.tensor(1.0))

    fx = limit * (fx - torch.min(fx)) / (torch.max(fx) - torch.min(fx))

    return fx


@ValidateDimension(ndim=3)
def rastrigin(x: torch.Tensor, freq: float, limit: float = 1.0) -> torch.Tensor:

    """Generic implementation of Rastrigin function -- parameterised by frequency.

    Parameters
    ----------
    x: torch.Tensor
        Spatial grid on which to compute the Rastrigin function.
    freq: float
        Parameterised frequency of the Rastrigin function.
    limit: float
        Largest value in the corruption field.

    Returns
    -------
    torch.Tensor
        Rastrigin function on given grid.
    """

    val = 10.0 * 2 + oe.contract('iju -> ij', (x - np.pi) ** 2 - 10.0 * torch.cos(freq * (x - np.pi)))
    val = limit * (val - torch.min(val)) / (torch.max(val) - torch.min(val))

    return val


def corruption_operation(data: torch.Tensor, phi: torch.Tensor, e_operation: eCorruptionOperation) -> torch.Tensor:

    """Corrupt data with respect to relevant corruption operation.

    Parameters
    ----------
    data: torch.Tensor
        Field to corrupt.
    phi: torch.Tensor
        Corruption field to incorporate into observations.
    e_operation: eCorruptionOperation
        Operation to use to corrupt the data.

    Returns
    -------
    torch.Tensor
        Corrupted field.
    """

    match e_operation:

        case eCorruptionOperation.additive:
            return data + phi

        case eCorruptionOperation.multiplicative:
            return data * (1 + phi)

        case _:
            raise ValueError('Incompatible corruption operation.')


def phi_from_zeta_u(zeta: torch.Tensor, data: torch.Tensor, e_operation: eCorruptionOperation) -> torch.Tensor:

    """Recover corruption field from corrupted observations and underlying data.

    Parameters
    ----------
    zeta: torch.Tensor
        Corrupted observations.
    data: torch.Tensor
        Underlying solution.
    e_operation: eCorruptionOperation
        Operation used to corrupt the observations.

    Returns
    -------
    torch.Tensor
        Recovered corruption field.
    """

    match e_operation:

        case eCorruptionOperation.additive:
            return zeta - data

        case eCorruptionOperation.multiplicative:
            return (zeta / data) - 1.0

        case _:
            raise ValueError('Incompatible corruption operation.')
