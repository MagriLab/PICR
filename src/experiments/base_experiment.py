import argparse
import csv
import functools as ft

import sys
from pathlib import Path
from shutil import copyfile
from typing import Callable, Dict, NamedTuple, Optional, Union

import einops
import numpy as np
import opt_einsum as oe
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

sys.path.append('../..')
from picr.model import Autoencoder
from picr.loss import get_loss_fn, PILoss

from picr.corruption import get_corruption_fn
from picr.experiments.data import load_data, train_validation_split, generate_dataloader

from picr.utils.config import ExperimentConfig
from picr.utils.enums import eCorruption
from picr.utils.loss_tracker import LossTracker

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# machine constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == 'cuda' else {}


class WandbConfig(NamedTuple):

    entity: Optional[str]
    project: Optional[str]
    group: Optional[str]


def initialise_csv(csv_path: Path) -> None:

    """Initialise the results .csv file.

    Parameters
    ----------
    csv_path: Path
        Path for the .csv file to create and write header to.
    """

    lt = LossTracker()
    with open(csv_path, 'w+', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(['epoch', *lt.get_fields(training=True), *lt.get_fields(training=False)])


def initialise_wandb(wandb_config: WandbConfig,
                     config: ExperimentConfig,
                     experiment_path: Path,
                     log_code: bool) -> Union[Run, RunDisabled, None]:

    """Initialise the Weights and Biases API.

    Parameters
    ----------
    wandb_config: WandbConfig
        Arguments for Weights and Biases.
    config: ExperimentConfig
        Configuration used to run the experiment.
    experiment_path: Path
        Path for the experiment -- location of code to copy
    log_code: bool
        Whether to save a copy of the code to Weights and Biases.
    """

    wandb_run = None
    if wandb_config.entity:

        # initialise W&B API
        wandb_run = wandb.init(
            config=config.config,
            entity=wandb_config.entity,
            project=wandb_config.project,
            group=wandb_config.group,
            name=str(experiment_path)
        )

    # log current code state to W&B
    if log_code and isinstance(wandb_run, Run):
        wandb_run.log_code(str(Path.cwd()))

    return wandb_run


def initialise_model(config: ExperimentConfig, model_path: Optional[Path] = None) -> nn.Module:

    """Iniitalise CNN Model for experiment.

    Parameters
    ----------
    config: ExperimentConfig
        Parameters to use for the experiment.
    model_path: Optional[Path]
        Optional model to load.

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
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.to(torch.double)
    model.to(DEVICE)

    return model


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


def get_boundaries(arr: torch.Tensor) -> torch.Tensor:

    """Retrieve boundaries of the given tensor.

    Note :: This assumes an input of a two-dimensional field.

    Parameters
    ----------
    arr: torch.Tensor
        Tensor to extract the boundaries from.
    """

    _boundary_list = []
    for b in range(-1, 0 + 1):
        _idx1 = tuple([..., slice(None), b])
        _idx2 = tuple([..., b, slice(None)])

        _boundary_list.extend([arr[_idx1], arr[_idx2]])

    boundaries = torch.cat(_boundary_list, dim=-1)

    return boundaries


def train_loop(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: PILoss,
               simulation_dt: float,
               phi_fn: Callable[[], torch.Tensor],
               optimizer: Optional[torch.optim.Optimizer] = None,
               s_lambda: float = 1e3,
               set_train: bool = False) -> LossTracker:

    """Run a single training / evaluation loop.

    Parameters
    ----------
    model: nn.Module
        Model to use for evaluation.
    dataloader: DataLoader
        Generator to retrieve data for evaluation from.
    loss_fn: PILoss
        Loss function for calculating physics-informed aspect of the loss.
    simulation_dt: float
        Size of the time-step used in the simulation data.
    phi_fn: ft.partial
        Function used to generate the required corruption field.
    optimizer: Optional[torch.optim.Optimizer]
        Optimiser used to update the weights of the model, when applicable.
    s_lambda: float
        Scaling parameter for the loss.
    set_train: bool
        Determine whether run in training / evaluation mode.

    Returns
    -------
    LossTracker
        Loss tracking object to hold information about the training progress.    
    """

    # reset losses to zero
    batched_residual_loss = (0.0 + 0.0j)
    batched_boundary_loss = (0.0 + 0.0j)

    batched_phi_dot_loss = (0.0 + 0.0j)
    batched_phi_mean_loss = (0.0 + 0.0j)

    batched_total_loss = (0.0 + 0.0j)

    batched_clean_u_loss = (0.0 + 0.0j)
    batched_clean_phi_loss = (0.0 + 0.0j)

    model.train(mode=set_train)
    for data in dataloader:

        # conduct inference on the batch
        data = data.to(DEVICE)

        # create corrupted data
        phi = phi_fn()
        phi = einops.repeat(phi, 'i j -> b t u i j', b=data.size(0), t=data.size(1), u=data.size(2))
        zeta = data + phi

        # predict u, phi
        u_prediction = model(zeta)
        phi_prediction = zeta - u_prediction

        # LOSS :: 01 :: Clean Velocity Field :: || R(\hat{u}) ||
        r_u_loss = loss_fn.calc_residual_loss(u_prediction)

        # LOSS :: 02 :: Constraint Loss :: || C(\hat{u}) ||
        c_u_loss = torch.zeros_like(r_u_loss)
        if loss_fn.constraints:
            c_u_loss = loss_fn.calc_constraint_loss(u_prediction)

        # LOSS :: 03 :: Boundary Loss :: || \hat{u_b} - u_b ||
        u_boundaries = get_boundaries(data)
        u_prediction_boundaries = get_boundaries(u_prediction)

        boundary_loss = oe.contract('... -> ', (u_boundaries - u_prediction_boundaries) ** 2) / u_boundaries.numel()
        boundary_loss *= s_lambda

        # LOSS :: 04 :: Stationary Corruption :: || \partial_t \hat{\phi} ||
        dphi_dt = (1.0 / simulation_dt) * (phi_prediction[:, 1:, ...] - phi_prediction[:, :-1, ...])
        dphi_dt = oe.contract('... -> ', dphi_dt ** 2) / dphi_dt.numel()
        dphi_dt *= s_lambda

        # LOSS :: 05 :: Mean Corruption :: || \hat{\phi} - <\hat{\phi}> ||
        r_phi = phi_prediction - einops.reduce(phi_prediction, 'b t u i j -> i j', torch.mean)
        mean_phi_loss = oe.contract('... -> ', r_phi ** 2) / r_phi.numel()

        mean_phi_loss *= s_lambda

        # LOSS :: 06 :: Total Loss
        total_loss = r_u_loss + boundary_loss + dphi_dt + mean_phi_loss + loss_fn.constraints * c_u_loss

        # LOSS :: 07 :: u, \phi -- Clean
        clean_u_loss = torch.sqrt(torch.sum((data - u_prediction) ** 2) / torch.sum(data ** 2))
        clean_phi_loss = torch.sqrt(torch.sum((phi - phi_prediction) ** 2) / torch.sum(phi ** 2))

        # update batch losses
        batched_residual_loss += r_u_loss.item() * data.size(0)
        batched_boundary_loss += boundary_loss.item() * data.size(0)
        batched_phi_dot_loss += dphi_dt.item() * data.size(0)
        batched_phi_mean_loss += dphi_dt.item() * data.size(0)
        batched_total_loss += total_loss.item() * data.size(0)
        batched_clean_u_loss += clean_u_loss.item() * data.size(0)
        batched_clean_phi_loss += clean_phi_loss.item() * data.size(0)

        # update gradients
        if set_train and optimizer:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # normalise and find absolute value
    batched_residual_loss = float(abs(batched_residual_loss)) / len(dataloader.dataset)                   # type: ignore
    batched_boundary_loss = float(abs(batched_boundary_loss)) / len(dataloader.dataset)                   # type: ignore
    batched_phi_dot_loss = float(abs(batched_phi_dot_loss)) / len(dataloader.dataset)                     # type: ignore
    batched_phi_mean_loss = float(abs(batched_phi_mean_loss)) / len(dataloader.dataset)                   # type: ignore
    batched_total_loss = float(abs(batched_total_loss)) / len(dataloader.dataset)                         # type: ignore
    batched_clean_u_loss = float(abs(batched_clean_u_loss)) / len(dataloader.dataset)                     # type: ignore
    batched_clean_phi_loss = float(abs(batched_clean_phi_loss)) / len(dataloader.dataset)                 # type: ignore

    loss_dict: Dict[str, float] = {
        'residual_loss': batched_residual_loss,
        'boundary_loss': batched_boundary_loss,
        'phi_dot_loss': batched_phi_dot_loss,
        'phi_mean_loss': batched_phi_mean_loss,
        'total_loss': batched_total_loss,
        'clean_u_loss': batched_clean_u_loss,
        'clean_phi_loss': batched_clean_phi_loss
    }

    return LossTracker(**loss_dict)


def main(args: argparse.Namespace) -> None:

    """Run the Experiment.

    Parameters
    ----------
    args: argparse.Namespace
        Command-line arguments to dictate experiment run.
    """

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

    # initialise weights and biases
    wandb_config = WandbConfig(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group)
    wandb_run = initialise_wandb(wandb_config, config, args.experiment_path, log_code=True)

    # setup the experiment path and copy config file
    args.experiment_path.mkdir(parents=True, exist_ok=True)
    copyfile(args.config_path, args.experiment_path / 'config.yml')

    # initialise csv
    csv_path = args.experiment_path / 'results.csv'
    initialise_csv(csv_path)

    # load data
    u_all: torch.Tensor = load_data(h5_file=args.data_path, config=config).to(torch.float)
    train_u, validation_u = train_validation_split(u_all, config.NTRAIN, config.NVALIDATION, step=config.TIME_STACK)

    train_loader: DataLoader = generate_dataloader(train_u, config.BATCH_SIZE, DEVICE_KWARGS)
    validation_loader: DataLoader = generate_dataloader(validation_u, config.BATCH_SIZE, DEVICE_KWARGS)

    # get corruption function and loss function
    u_max: float = torch.max(u_all).item()
    phi_limit = config.PHI_LIMIT * u_max

    phi_fn: ft.partial = set_corruption_fn(config.PHI_FN, config.NX, config.PHI_FREQ, phi_limit)
    loss_fn: PILoss = get_loss_fn(config, DEVICE)

    # We can optionally disable the constraints here for testing
    # loss_fn.constraints = False

    # initialise model / optimizer
    model = initialise_model(config, args.model_path)
    model.to(torch.float)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.L2)

    # generate training functions
    _loop_params = dict(model=model, loss_fn=loss_fn, simulation_dt=config.DT, phi_fn=phi_fn)

    train_fn = ft.partial(train_loop, **_loop_params, dataloader=train_loader, optimizer=optimizer, set_train=True)
    validation_fn = ft.partial(train_loop, **_loop_params, dataloader=validation_loader, set_train=False)

    # main training loop
    min_validation_loss = np.Inf
    for epoch in range(config.N_EPOCHS):

        lt_training: LossTracker = train_fn()
        lt_validation: LossTracker = validation_fn()

        # update global validation loss if model improves
        if lt_validation.total_loss < min_validation_loss:
            min_validation_loss = lt_validation.total_loss
            torch.save(model.state_dict(), args.experiment_path / 'autoencoder.pt')

        # log results to weights and biases
        if isinstance(wandb_run, Run):
            wandb_log = {**lt_training.get_dict(training=True), **lt_validation.get_dict(training=False)}
            wandb_run.log(data=wandb_log)

        # print update to stdout
        msg = f'Epoch: {epoch:05}'
        for k, v in lt_training.get_dict(training=True).items():
            msg += f' | {k}: {v:08.5e}'

        if epoch % args.log_freq == 0:
            print(msg)

        # write new results to .csv file
        _results = [epoch, *lt_training.get_loss_keys, *lt_validation.get_loss_keys]
        with open(csv_path, 'a', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)

    # upload results to weights and biases
    if isinstance(wandb_run, Run):
        artifact = wandb.Artifact(name=str(args.experiment_path).replace('/', '.'), type='dataset')
        artifact.add_dir(local_path=str(args.experiment_path))

        wandb_run.log_artifact(artifact_or_path=artifact)


if __name__ == '__main__':

    # read arguments from command line
    parser = argparse.ArgumentParser(description='PICR :: Physics-Informed Corruption Removal Experiment.')

    # arguments to define paths for experiment run
    parser.add_argument('-ep', '--experiment-path', type=Path, required=True)
    parser.add_argument('-dp', '--data-path', type=Path, required=True)
    parser.add_argument('-cp', '--config-path', type=Path, required=True)

    # argument to define optional path to load pre-trained model
    parser.add_argument('-mp', '--model-path', type=Path, required=False)

    parser.add_argument('-gpu', '--run-gpu', type=int, required=False)
    parser.add_argument('-mf', '--memory-fraction', type=float, required=False)

    parser.add_argument('-lf', '--log-freq', type=int, required=False, default=5)

    # arguments to define wandb parameters
    parser.add_argument('--wandb-entity', default=None, type=str)
    parser.add_argument('--wandb-project', default=None, type=str)
    parser.add_argument('--wandb-group', default=None, type=str)

    parsed_args = parser.parse_args()

    main(parsed_args)
