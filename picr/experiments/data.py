import operator
from pathlib import Path
from typing import Any, Dict, Tuple

import einops
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.config import ExperimentConfig
from ..utils.dataset import UnlabeledTensorDataset
from ..utils.enums import eSolverFunction


def load_data(h5_file: Path, config: ExperimentConfig, t_stack: int) -> torch.Tensor:

    """Loads simulation data as torch.Tensor.

    Parameters
    ----------
    h5_file: Path
        Path to the .h5 file containing simulation data.
    config: ExperimentConfig
        Configuration object holding key information about simulation.
    t_stack: int
        Number of consecutive timesteps to stack.

    Returns
    -------
    u_all: torch.Tensor
        Loaded simulation data.
    """

    with h5py.File(h5_file, 'r') as hf:

        # check configuration matches given simulation file
        config_mismatch = []
        for x, config_x in zip(['re', 'nk', 'dt', 'resolution', 'ndim'], [config.RE, config.NK, config.DT, config.NX, config.NU]):
            if not np.array(hf.get(x)) == np.array(config_x):
                config_mismatch.append(x)

        if config.SOLVER_FN == eSolverFunction.LINEAR:
            if not np.array(hf.get('c')) == np.array(config.C):
                config_mismatch.append('c')

        if config.SOLVER_FN == eSolverFunction.NONLINEAR:
            if not np.array(hf.get('nf')) == np.array(config.NF):
                config_mismatch.append('nf')

        if config_mismatch:
            raise ValueError(f'Configuration does not match simulation: {config_mismatch}')

        # load data from h5 file
        u_all = np.array(hf.get('velocity_field'))

    # stack N consecutive time-steps in a new dimension
    list_u = []
    for i, j in zip(range(t_stack), map(operator.neg, reversed(range(t_stack)))):
        sl = slice(i, j) if j < 0 else slice(i, None)
        list_u.append(u_all[sl])

    u_all = np.stack(list_u, axis=1)

    u_all = einops.rearrange(u_all, 'b t i j u -> b t u i j')
    u_all = torch.from_numpy(u_all).to(torch.float)

    return u_all


def generate_random_idx(data: torch.Tensor, n_points: int, step: int) -> np.ndarray:

    """Generate random indices without replacement - ensuring no duplicate fields.

    Parameters
    ----------
    data: torch.Tensor
        Dataset to choose random indices from.
    n_points: int
        Number of indices to choose.
    step: int
        Minimum step between consecutive indices.

    Returns
    -------
    idx: np.ndarray
        Randomly generated indices.
    """

    if not step >= data.shape[1]:
        msg = f'Train / validation indices may contain duplicates. Step should be minimum: {data.shape[1]}'
        raise Warning(msg)

    idx_to_choose_from = np.arange(0, data.shape[0], step=step)
    if idx_to_choose_from.shape[0] < n_points:
        raise ValueError('Not enough points in the dataset.')

    idx = np.random.choice(idx_to_choose_from, size=n_points, replace=False)
    return idx


def train_validation_split(data: torch.Tensor, n_train: int, n_validation: int, step: int) -> Tuple[torch.Tensor, torch.Tensor]:

    """Create train / validation split from given data.

    Parameters
    ----------
    data: torch.Tensor
        Data to generate train / validation split from.
    n_train: int
        Number of data points in the training set.
    n_validation: int
        Number of data points in the validation set.
    step: int
        Step between subsequent samples.

    Returns
    -------
    d_train: torch.Tensor
        Training dataset.
    d_validation: torch.Tensor
        Validation dataset.
    """

    # choose n_train + n_validation random indices from the dataset without replacement
    n_points = n_train + n_validation
    idx = generate_random_idx(data, n_points, step=step)

    # split data into train / validation
    d_train, d_validation = torch.split(data[idx], [n_train, n_validation])

    return d_train, d_validation


def generate_dataloader(data: torch.Tensor, batch_size: int, device_kwargs: Dict[str, Any]) -> DataLoader:

    """Generate DataLoader from given data.

    Parameters
    ----------
    data: torch.Tensor
        Data to generate DataLoader from.
    batch_size: int
        Number of items per batch.
    device_kwargs: Dict[str, Any]
        Kwargs for the device.

    Returns
    -------
    dataloader: DataLoader
        Produced DataLoader for the given data.
    """

    dataset = UnlabeledTensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, **device_kwargs)

    return dataloader
