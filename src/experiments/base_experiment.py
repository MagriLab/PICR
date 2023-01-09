import csv
import functools as ft
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import einops
import numpy as np
import opt_einsum as oe
import torch
import torch.backends.cudnn
from absl import app, flags
from ml_collections import config_flags
from torch import nn
from torch.utils.data import DataLoader
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

import wandb

from ..picr.corruption import get_corruption_fn
from ..picr.data import generate_dataloader, load_data, train_validation_split
from ..picr.experimental import define_path as picr_flags
from ..picr.loss import get_loss_fn, PILoss
from ..picr.model import BottleneckCNN
from ..picr.utils.enums import eCorruption
from ..picr.utils.loss_tracker import LossTracker
from ..picr.utils.types import ExperimentConfig
from .configs.wandb import WANDB_CONFIG


warnings.filterwarnings("ignore", category=UserWarning)


FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config')
_WANBD_CONFIG = config_flags.DEFINE_config_dict('wandb', WANDB_CONFIG)

_EXPERIMENT_PATH = picr_flags.DEFINE_path(
    'experiment_path',
    None,
    'Directory to store experiment results'
)

_DATA_PATH = picr_flags.DEFINE_path(
    'data_path',
    None,
    'Path to .h5 file storing the data.'
)

_GPU = flags.DEFINE_integer(
    'run_gpu',
    0,
    'Which GPU to run on.'
)

_MEMORY_FRACTION = flags.DEFINE_float(
    'memory_fraction',
    None,
    'Memory fraction of GPU to use.'
)

_LOG_FREQUENCY = flags.DEFINE_integer(
    'log_frequency',
    1,
    'Frequency at which to log results.'
)

_CUDNN_BENCHMARKS = flags.DEFINE_boolean(
    'cudnn_benchmarks',
    True,
    'Whether to use CUDNN benchmarks or not.'
)

flags.mark_flags_as_required(['config', 'experiment_path', 'data_path'])


# machine constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == 'cuda' else {}


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


def initialise_wandb() -> Optional[Run | RunDisabled]:

    """Initialise the Weights and Biases API."""

    wandb_run = None

    wandb_config = FLAGS.wandb
    experiment_name = wandb_config.name or str(FLAGS.experiment_path)

    tags = wandb_config.tags
    if tags:
        tags = tags.split(':')

    # provide a better check for wandb_run
    if wandb_config.entity and wandb_config.project:

        # initialise W&B API
        wandb_run = wandb.init(
            config=FLAGS.config.to_dict(),
            entity=wandb_config.entity,
            project=wandb_config.project,
            name=experiment_name,
            group=wandb_config.group,
            tags=tags,
            job_type=wandb_config.job_type,
            notes=wandb_config.notes
        )

    # log current code state to W&B
    if wandb_run:
        wandb_run.log_code('./src')

    return wandb_run


def initialise_model() -> nn.Module:

    config = FLAGS.config

    # initialise model
    model = BottleneckCNN(
        nx=config.data.resolution,
        nc=2,
        layers=config.training.layers,
        latent_dim=config.training.latent_dim,
        decoder=config.training.decoder,
    )

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

    batched_phi_dt_loss = (0.0 + 0.0j)
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

        # LOSS :: 02 :: Boundary Loss :: || \hat{u_b} - u_b ||
        u_boundaries = get_boundaries(data)
        u_prediction_boundaries = get_boundaries(u_prediction)

        boundary_loss = torch.mean((u_boundaries - u_prediction_boundaries) ** 2)

        # LOSS :: 03 :: Stationary Corruption :: || \partial_t \hat{\phi} ||
        dphi_dt = (1.0 / FLAGS.config.simulation.dt) * (phi_prediction[:, 1:, ...] - phi_prediction[:, :-1, ...])
        dphi_dt_loss = torch.mean(dphi_dt ** 2)

        # LOSS :: 04 :: Mean Corruption :: || \hat{\phi} - <\hat{\phi}> ||
        mean_phi_loss = torch.mean((phi_prediction - einops.reduce(phi_prediction, 'b t u i j -> i j', torch.mean)) ** 2)

        # LOSS :: 05 :: Total Loss
        total_loss = r_u_loss + s_lambda * (boundary_loss + dphi_dt_loss + mean_phi_loss)

        # LOSS :: 06 :: u, \phi -- Clean
        clean_u_loss = torch.sqrt(torch.sum((data - u_prediction) ** 2) / torch.sum(data ** 2))
        clean_phi_loss = torch.sqrt(torch.sum((phi - phi_prediction) ** 2) / torch.sum(phi ** 2))

        # update batch losses
        batched_residual_loss += r_u_loss.item() * data.size(0)
        batched_boundary_loss += boundary_loss.item() * data.size(0)
        batched_phi_dt_loss += dphi_dt_loss.item() * data.size(0)
        batched_phi_mean_loss += mean_phi_loss.item() * data.size(0)
        batched_total_loss += total_loss.item() * data.size(0)
        batched_clean_u_loss += clean_u_loss.item() * data.size(0)
        batched_clean_phi_loss += clean_phi_loss.item() * data.size(0)

        # update gradients
        if set_train and optimizer:

            optimizer.zero_grad(set_to_none=True)

            total_loss.backward()
            optimizer.step()

    # normalise and find absolute value
    batched_residual_loss = float(abs(batched_residual_loss)) / len(dataloader.dataset)                   # type: ignore
    batched_boundary_loss = float(abs(batched_boundary_loss)) / len(dataloader.dataset)                   # type: ignore
    batched_phi_dt_loss = float(abs(batched_phi_dt_loss)) / len(dataloader.dataset)                     # type: ignore
    batched_phi_mean_loss = float(abs(batched_phi_mean_loss)) / len(dataloader.dataset)                   # type: ignore
    batched_total_loss = float(abs(batched_total_loss)) / len(dataloader.dataset)                         # type: ignore
    batched_clean_u_loss = float(abs(batched_clean_u_loss)) / len(dataloader.dataset)                     # type: ignore
    batched_clean_phi_loss = float(abs(batched_clean_phi_loss)) / len(dataloader.dataset)                 # type: ignore

    loss_dict: Dict[str, float] = {
        'residual_loss': batched_residual_loss,
        'boundary_loss': batched_boundary_loss,
        'phi_dot_loss': batched_phi_dt_loss,
        'phi_mean_loss': batched_phi_mean_loss,
        'total_loss': batched_total_loss,
        'clean_u_loss': batched_clean_u_loss,
        'clean_phi_loss': batched_clean_phi_loss
    }

    return LossTracker(**loss_dict)


def main(_) -> None:

    if FLAGS.run_gpu is not None and FLAGS.run_gpu >= 0 and FLAGS.run_gpu < torch.cuda.device_count():

        global DEVICE
        global DEVICE_KWARGS

        if not torch.cuda.is_available():
            raise ValueError('Specified CUDA device unavailable.')

        DEVICE = torch.device(f'cuda:{FLAGS.run_gpu}')
        DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True}

        torch.backends.cudnn.benchmark = FLAGS.cudnn_benchmarks

    if FLAGS.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(FLAGS.memory_fraction, DEVICE)

    # easy access to config
    config: ExperimentConfig = FLAGS.config

    # initialise weights and biases
    wandb_run = initialise_wandb()

    # setup the experiment path
    FLAGS.experiment_path.mkdir(parents=True, exist_ok=True)

    # save config to yaml
    with open(FLAGS.experiment_path / 'config.yml', 'w') as f:
        config.to_yaml(stream=f)

    # initialise csv
    csv_path = FLAGS.experiment_path / 'results.csv'
    initialise_csv(csv_path)

    # load data
    u_all = load_data(h5_file=FLAGS.data_path, config=config)
    train_u, validation_u = train_validation_split(u_all, config.data.ntrain, config.data.nvalidation, step=config.data.tau)

    # set `drop_last = True` if using: `torch.backends.cudnn.benchmark = True`
    dataloader_kwargs = dict(shuffle=True, drop_last=FLAGS.cudnn_benchmarks)

    train_loader = generate_dataloader(train_u, config.training.batch_size, dataloader_kwargs, DEVICE_KWARGS)
    validation_loader = generate_dataloader(validation_u, config.training.batch_size, dataloader_kwargs, DEVICE_KWARGS)

    # get corruption function and loss function
    u_max = torch.max(u_all).item()
    phi_limit = config.corruption.phi_magnitude * u_max

    phi_fn = set_corruption_fn(
        config.corruption.phi_fn,
        config.data.resolution,
        config.corruption.phi_frequency,
        phi_limit
    )

    loss_fn = get_loss_fn(config, DEVICE)

    # initialise model / optimizer
    model = initialise_model().to(torch.float)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.l2)

    # generate training functions
    _loop_params = dict(model=model, loss_fn=loss_fn, phi_fn=phi_fn)

    train_fn = ft.partial(train_loop, **_loop_params, dataloader=train_loader, optimizer=optimizer, set_train=True)
    validation_fn = ft.partial(train_loop, **_loop_params, dataloader=validation_loader, set_train=False)

    # main training loop
    min_validation_loss = np.Inf
    for epoch in range(config.training.epochs):

        lt_training: LossTracker = train_fn()
        lt_validation: LossTracker = validation_fn()

        # update global validation loss if model improves
        if lt_validation.total_loss < min_validation_loss:
            min_validation_loss = lt_validation.total_loss
            torch.save(model.state_dict(), FLAGS.experiment_path / 'autoencoder.pt')

        # log results to weights and biases
        if isinstance(wandb_run, Run):
            wandb_log = {**lt_training.get_dict(training=True), **lt_validation.get_dict(training=False)}
            wandb_run.log(data=wandb_log)

        # print update to stdout
        msg = f'Epoch: {epoch:05}'
        for k, v in lt_training.get_dict(training=True).items():
            msg += f' | {k}: {v:08.5e}'

        if epoch % FLAGS.log_frequency == 0:
            print(msg)

        # write new results to .csv file
        _results = [epoch, *lt_training.get_loss_keys, *lt_validation.get_loss_keys]
        with open(csv_path, 'a', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)

    # upload results to weights and biases
    if isinstance(wandb_run, Run):
        artifact = wandb.Artifact(name=str(FLAGS.experiment_path).replace('/', '.'), type='dataset')
        artifact.add_dir(local_path=str(FLAGS.experiment_path))

        wandb_run.log_artifact(artifact_or_path=artifact)


if __name__ == '__main__':
    app.run(main)
