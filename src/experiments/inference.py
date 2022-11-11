
import argparse
import csv
import functools as ft
import operator
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import einops
import h5py
import numpy as np
import opt_einsum as oe
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader


sys.path.append('../..')
import warnings

from picr.corruption import get_corruption_fn
from picr.experiments.data import generate_dataloader
from picr.model import Autoencoder
from picr.utils.config import ExperimentConfig
from picr.utils.enums import eCorruption, eSolverFunction


warnings.filterwarnings("ignore", category=UserWarning)


# machine constants
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == 'cuda' else {}


def load_data(h5_file: Path, config: ExperimentConfig) -> torch.Tensor:

    """Loads simulation data as torch.Tensor.

    Parameters
    ----------
    h5_file: Path
        Path to the .h5 file containing simulation data.
    config: ExperimentConfig
        Configuration object holding key information about simulation.

    Returns
    -------
    u_all: torch.Tensor
        Loaded simulation data.
    """

    with h5py.File(h5_file, 'r') as hf:

        # check configuration matches given simulation file
        config_mismatch = []
        for x, config_x in zip(['re', 'nk', 'dt', 'resolution', 'ndim'], [config.RE, config.NK, config.DT, config.NX, config.NU]):
            if np.array(hf.get(x)) != np.array(config_x):
                config_mismatch.append(x)

        if config.SOLVER_FN == eSolverFunction.LINEAR:
            if np.array(hf.get('c')) != np.array(config.C):
                config_mismatch.append('c')

        if config_mismatch:
            raise ValueError(f'Configuration does not match simulation: {config_mismatch}')

        # load data from h5 file
        u_all = np.array(hf.get('velocity_field'))

    u_all = einops.rearrange(u_all, 'b i j u -> b 1 u i j')
    u_tensor_all = torch.from_numpy(u_all).to(torch.float)

    return u_tensor_all


def set_corruption_fn(e_phi_fn: eCorruption,
                      resolution: int,
                      frequency: float,
                      magnitude: float) -> ft.partial:

    """Generate partial corruption function with given parameters.

    Parameters
    ----------
    e_phi_fn: eCorruption
        Type of corruption function to generate.
    resulution: int
        Resolution of the field to generate from the corruption function.
    frequency: float
        Parameterised frequency for the corruption function.
    magnitude: float
        Parameterised magnitude for the corruption function.

    Returns
    -------
    phi_fn: ft.partial
        Generated partial corruption function.
    """

    x = torch.linspace(0.0, 2.0 * np.pi, resolution)
    xx = torch.stack(torch.meshgrid(x, x), dim=-1).to(DEVICE)

    # get corruption function
    _phi_fn = get_corruption_fn(e_phi_fn)
    phi_fn = ft.partial(_phi_fn, x=xx, freq=frequency, limit=magnitude)                                   # type: ignore

    return phi_fn


def initialise_model(config: ExperimentConfig, model_path: Path) -> nn.Module:

    """Iniitalise CNN Model for inference.

    Parameters
    ----------
    config: ExperimentConfig
        Parameters to use for inference.
    model_path: Path
        Model to load.

    Returns
    -------
    model: nn.Module
        Initialised model.
    """

    # get activation function
    activation_fn = getattr(nn, config.ACTIVATION)()

    # initialise model
    model = Autoencoder(
        nx=config.NX,
        nc=config.NU,
        layers=config.LAYERS,
        latent_dim=config.LATENT_DIM,
        decoder=config.DECODER,
        activation=activation_fn,
        dropout=config.DROPOUT,
        batch_norm=config.BATCH_NORM
    )

    # load model from file if applicable.
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    return model


def write_h5(path: Path, data: Dict[str, Any]) -> None:

    """Writes results dictionary to .h5 file.

    Parameters
    ----------
    path: Path
        Corresponding .h5 file to write results to.
    data: Dict[str, Any]
        Data to write to file.
    """

    hf = h5py.File(path, 'w')

    for k, v in data.items():
        hf.create_dataset(k, data=v)

    hf.close()


def inference(model: nn.Module,
              data: torch.Tensor,
              batch_size: int,
              phi_fn: ft.partial) -> Dict[str, Any]:

    """Run inference on the given data.

    Parameters:
    ===========
    model: nn.Module
        Model to use for inference.
    data: torch.Tensor
        Data to conduct inference on.
    batch_size: int
        Number of items to analyse per batch.
    phi_fn: ft.parial
        Function used to generate the required corruption field.

    Returns:
    ========
    return_dict: Dict[str, Any]
        Dictionary containing information from inference.
    """

    # initialise variables to store data
    corrupted_u = torch.zeros_like(data, device=torch.device('cpu'))

    u_predictions = torch.zeros_like(data, device=torch.device('cpu'))
    phi_predictions = torch.zeros_like(data, device=torch.device('cpu'))

    dataloader = generate_dataloader(data, batch_size, DEVICE_KWARGS)

    model.train(mode=False)

    msg = ' 01 :: Conducting inference on batches.'
    for idx, batch in enumerate(tqdm.tqdm(dataloader, total=len(dataloader), desc=msg)):

        # conduct inference on the batch
        batch = batch.to(DEVICE)

        # create corrupted batch
        phi = phi_fn()
        phi = einops.repeat(phi, 'i j -> b t u i j', b=batch.size(0), t=batch.size(1), u=batch.size(2))
        zeta = batch + phi

        # predict u, phi
        u_pred = model(zeta)
        phi_pred = zeta - u_pred

        # move predictions
        batch_slice = slice(idx * batch_size, idx * batch_size + u_pred.size(0))

        corrupted_u[batch_slice, ...] = zeta.detach().cpu()
        u_predictions[batch_slice, ...] = u_pred.detach().cpu()
        phi_predictions[batch_slice, ...] = phi_pred.detach().cpu()

    return_dict: Dict[str, Any] = {
        'phi': phi[0, 0, 0, ...].detach().cpu(),
        'corrupted_u': corrupted_u,
        'u_predictions': u_predictions,
        'phi_predictions': phi_predictions,
    }

    return return_dict


def main(args: argparse.Namespace) -> None:

    """Run inference on given dataset with provided model.

    Parameters
    ----------
    args: argparse.Namespace
        Command-line arguments to dictate inference run.
    """

    print('00 :: Running inference on given data.')

    if args.run_gpu is not None and args.run_gpu >= 0 and args.run_gpu < torch.cuda.device_count():

        global DEVICE
        global DEVICE_KWARGS

        if not torch.cuda.is_available():
            raise ValueError('Specified CUDA device unavailable.')

        DEVICE = torch.device(f'cuda:{args.run_gpu}')
        DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True}

    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, DEVICE)

    # load yaml configuration file
    config = ExperimentConfig()
    config.load_config(args.config_path)

    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    u_all: torch.Tensor = load_data(h5_file=args.data_path, config=config)
    u_max: float = torch.max(u_all).item()

    phi_limit = config.PHI_LIMIT * u_max
    phi_fn: ft.partial = set_corruption_fn(config.PHI_FN, config.NX, config.PHI_FREQ, phi_limit)

    # initialise model / optimizer
    model = initialise_model(config, args.model_path)
    model.to(torch.float)

    # run inference
    inference_dict = inference(model, u_all, config.BATCH_SIZE, phi_fn)

    # create dictionary and log to file
    u_original = u_all.detach().numpy()

    u_predictions = inference_dict['u_predictions'].numpy()
    phi_predictions = inference_dict['phi_predictions'].numpy()

    phi = inference_dict['phi'].numpy()
    corrupted_u = inference_dict['corrupted_u'].numpy()

    h5_dict = {
        'phi': phi,
        'u_original': u_original,
        'corrupted_u': corrupted_u,
        'u_predictions': u_predictions,
        'phi_predictions': phi_predictions
    }

    print('02 :: Writing results to file.')
    write_h5(args.save_path, h5_dict)

    print('03 :: Inference complete.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference on Dataset.')

    parser.add_argument('-dp', '--data-path', type=Path, required=True)
    parser.add_argument('-cp', '--config-path', type=Path, required=True)
    parser.add_argument('-mp', '--model-path', type=Path, required=True)
    parser.add_argument('-sp', '--save-path', type=Path, required=True)

    parser.add_argument('-bs', '--batch-size', type=int, required=False)
    parser.add_argument('-gpu', '--run-gpu', type=int, required=False)
    parser.add_argument('-mf', '--memory-fraction', type=float, required=False)

    parsed_args = parser.parse_args()

    main(parsed_args)
