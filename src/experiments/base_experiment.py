import argparse
import csv
import functools as ft
import sys
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, Optional, Union

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
from picr.loss import LinearCDLoss, NonlinearCDLoss, get_loss_fn

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


def initialise_csv(csv_path: Path) -> None:

    lt = LossTracker()
    with open(csv_path, 'w+', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(['epoch', *lt.get_fields(training=True), *lt.get_fields(training=False)])


def initialise_wandb(args: argparse.Namespace,
                     config: Dict[str, Any],
                     log_code: bool = True) -> Union[Run, RunDisabled, None]:

    wandb_run = None
    if args.wandb_entity:

        # initialise W&B API
        wandb_run = wandb.init(
            config=config,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=str(args.experiment_path)
        )

    # log current code state to W&B
    if log_code and isinstance(wandb_run, Run):
        wandb_run.log_code(str(Path.cwd()))

    return wandb_run


def initialise_model(config: ExperimentConfig, model_path: Optional[Path] = None) -> nn.Module:

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
                      magnitude: float) -> Callable[[], torch.Tensor]:

    x = torch.linspace(0.0, 2.0 * np.pi, resolution)
    xx = torch.stack(torch.meshgrid(x, x), dim=-1).to(DEVICE)

    # get corruption function
    _phi_fn = get_corruption_fn(e_phi_fn)
    phi_fn = ft.partial(_phi_fn, x=xx, freq=frequency, limit=magnitude)

    return phi_fn


def get_boundaries(arr: torch.Tensor) -> torch.Tensor:

    _boundary_list = []
    for b in range(-1, 0 + 1):
        _idx1 = tuple([..., slice(None), b])
        _idx2 = tuple([..., b, slice(None)])

        _boundary_list.extend([arr[_idx1], arr[_idx2]])

    boundaries = torch.cat(_boundary_list, dim=-1)

    return boundaries


def train_loop(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: Union[LinearCDLoss, NonlinearCDLoss],
               simulation_dt: float,
               phi_fn: Callable[[], torch.Tensor],
               optimizer: Optional[torch.optim.Optimizer] = None,
               s_lambda: float = 1e3,
               set_train: bool = False) -> LossTracker:

    # reset losses to zero
    batched_residual_loss = (0.0 + 0.0j)
    batched_boundary_loss = (0.0 + 0.0j)

    batched_phi_dot_loss = (0.0 + 0.0j)
    batched_phi_mean_loss = (0.0 + 0.0j)

    batched_total_loss = (0.0 + 0.0j)

    batched_clean_u_loss = (0.0 + 0.0j)
    batched_clean_phi_loss = (0.0 + 0.0j)

    model.train(mode=set_train)
    for batch_idx, data in enumerate(dataloader):

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

        # LOSS :: 02 :: Boundary Loss :: || \hat{u_b} - u_b ||
        u_boundaries = get_boundaries(data)
        u_prediction_boundaries = get_boundaries(u_prediction)

        boundary_loss = oe.contract('... -> ', (u_boundaries - u_prediction_boundaries) ** 2) / u_boundaries.numel()
        boundary_loss *= s_lambda

        # LOSS :: 03 :: Stationary Corruption :: || \partial_t \hat{\phi} ||
        dphi_dt = (1.0 / simulation_dt) * (phi_prediction[:, 1:, ...] - phi_prediction[:, :-1, ...])
        dphi_dt = oe.contract('... -> ', dphi_dt ** 2) / dphi_dt.numel()
        dphi_dt *= s_lambda

        # LOSS :: 04 :: Mean Corruption :: || \hat{\phi} - <\hat{\phi}> ||
        r_phi = phi_prediction - einops.reduce(phi_prediction, 'b t u i j -> i j', torch.mean)
        mean_phi_loss = oe.contract('... -> ', r_phi ** 2) / r_phi.numel()

        mean_phi_loss *= s_lambda

        # LOSS :: 05 :: Total Loss
        total_loss = r_u_loss + boundary_loss + dphi_dt + mean_phi_loss

        # LOSS :: 06 :: u, \phi -- Clean
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
    batched_residual_loss = float(abs(batched_residual_loss)) / len(dataloader.dataset)
    batched_boundary_loss = float(abs(batched_boundary_loss)) / len(dataloader.dataset)
    batched_phi_dot_loss = float(abs(batched_phi_dot_loss)) / len(dataloader.dataset)
    batched_phi_mean_loss = float(abs(batched_phi_mean_loss)) / len(dataloader.dataset)
    batched_total_loss = float(abs(batched_total_loss)) / len(dataloader.dataset)
    batched_clean_u_loss = float(abs(batched_clean_u_loss)) / len(dataloader.dataset)
    batched_clean_phi_loss = float(abs(batched_clean_phi_loss)) / len(dataloader.dataset)

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

    # load yaml configuration file
    config = ExperimentConfig()
    config.load_config(args.config_path)

    # initialise weights and biases
    wandb_run = initialise_wandb(args, config.config, log_code=True)

    # setup the experiment path and copy config file
    args.experiment_path.mkdir(parents=True, exist_ok=False)
    copyfile(args.config_path, args.experiment_path / 'config.yml')

    # initialise csv
    csv_path = args.experiment_path / 'results.csv'
    initialise_csv(csv_path)

    # load data
    u_all = load_data(h5_file=args.data_path, config=config)
    train_u, validation_u = train_validation_split(u_all, config.NTRAIN, config.NVALIDATION, step=config.TIME_STACK)

    train_loader = generate_dataloader(train_u, config.BATCH_SIZE, DEVICE_KWARGS)
    validation_loader = generate_dataloader(validation_u, config.BATCH_SIZE, DEVICE_KWARGS)

    # get corruption function and loss function
    phi_fn = set_corruption_fn(config.PHI_FN, config.NX, config.PHI_FREQ, config.PHI_LIMIT)
    loss_fn = get_loss_fn(config, DEVICE)

    # initialise model / optimizer
    model = initialise_model(config, args.model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.L2)

    # generate training functions
    _loop_params = dict(model=model, loss_fn=loss_fn, simulation_dt=config.DT, phi_fn=phi_fn)

    train_fn = ft.partial(train_loop, **_loop_params, dataloader=train_loader, optimizer=optimizer, set_train=True)
    validation_fn = ft.partial(train_loop, **_loop_params, dataloader=validation_loader, set_train=False)

    # training loop
    min_validation_loss = np.Inf
    for epoch in range(config.N_EPOCHS):

        lt_training = train_fn()
        lt_validation = validation_fn()

        if lt_validation.total_loss < min_validation_loss:
            min_validation_loss = lt_validation.total_loss
            torch.save(model.state_dict(), args.experiment_path / 'autoencoder.pt')

        # log results to weights and biases
        if isinstance(wandb_run, Run):
            wandb_log = {**lt_training.get_dict(training=True), **lt_validation.get_dict(training=False)}
            wandb_run.log(data=wandb_log)

        msg = f'Epoch: {epoch:05}'
        for k, v in lt_training.get_dict(training=True).items():
            msg += f' | {k}: {v:08.5e}'

        if epoch % 5 == 0:
            print(msg)

        _results = [epoch, *lt_training.get_values(), *lt_validation.get_values()]
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

    # arguments to define wandb parameters
    parser.add_argument('--wandb-entity', default=None, type=str)
    parser.add_argument('--wandb-project', default=None, type=str)
    parser.add_argument('--wandb-group', default=None, type=str)

    parsed_args = parser.parse_args()

    main(parsed_args)
