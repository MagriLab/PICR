import warnings
from typing import Any

import h5py
import numpy as np
import torch
import tqdm
import yaml
from absl import app, flags
from ml_collections import config_dict, config_flags
from torch import nn

from ..picr.corruption import corruption_operation, get_corruption_fn
from ..picr.data import generate_dataloader, load_data
from ..picr.experimental import define_path as picr_flags
from ..picr.model import BottleneckCNN
from ..picr.utils.enums import eCorruption


warnings.filterwarnings("ignore", category=UserWarning)


FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_dict(
    'config',
    config_dict.ConfigDict(),
    'Global config to populate'
)

_EXPERIMENT_PATH = picr_flags.DEFINE_path(
    'experiment_path',
    None,
    'Directory from which to read experimental results'
)

_DATA_PATH = picr_flags.DEFINE_path(
    'data_path',
    None,
    'Path to .h5 file storing the data.'
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    32,
    'Batch size to use for inference.'
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

_CUDNN_BENCHMARKS = flags.DEFINE_boolean(
    'cudnn_benchmarks',
    True,
    'Whether to use CUDNN benchmarks or not.'
)

flags.mark_flags_as_required(['config', 'experiment_path', 'data_path'])


# machine constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == 'cuda' else {}


def update_config() -> None:

    def _update_config_dict(a: config_dict.ConfigDict, b: config_dict.ConfigDict) -> config_dict.ConfigDict:

        da = a.to_dict(preserve_field_references=True)
        db = b.to_dict(preserve_field_references=True)

        return config_dict.ConfigDict(da | db)

    with open(FLAGS.experiment_path / 'config.yml', 'r') as f:
        config_from_yaml = yaml.load(f, Loader=yaml.UnsafeLoader)

    FLAGS.config = _update_config_dict(FLAGS.config, config_from_yaml)


def get_corruption(e_phi_fn: eCorruption,
                   resolution: int,
                   frequency: float,
                   magnitude: float) -> torch.Tensor:

    """Generate base corruption field.

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
    phi: torch.Tensor
        Base corruption field to add.
    """

    x = torch.linspace(0.0, 2.0 * np.pi, resolution)
    xx = torch.stack(torch.meshgrid(x, x), dim=-1)

    # get corruption function
    _phi_fn = get_corruption_fn(e_phi_fn)
    phi = _phi_fn(xx, frequency, magnitude)

    return phi


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

    model.load_state_dict(torch.load(FLAGS.experiment_path / 'autoencoder.pt', map_location=DEVICE))
    model.to(DEVICE)

    return model


def write_h5(data: dict[str, Any]) -> None:

    """Writes results dictionary to .h5 file.

    Parameters
    ----------
    data: dict[str, Any]
        Data to write to file.
    """

    with h5py.File(FLAGS.experiment_path / 'inference.h5', 'w') as hf:

        for k, v in data.items():
            hf.create_dataset(k, data=v)


def inference(model: nn.Module,
              data: torch.Tensor,
              phi: torch.Tensor) -> dict[str, Any]:

    """Run inference on the given data.

    Parameters:
    ----------
    model: nn.Module
        Model to use for inference.
    data: torch.Tensor
        Data to conduct inference on.
    phi: torch.Tensor
        Base corruption to apply for observations.

    Returns:
    --------
    return_dict: dict[str, Any]
        Dictionary containing information from inference.
    """

    # initialise variables to store data
    u_corrupted = torch.zeros_like(data, device=torch.device('cpu'))
    u_predicted = torch.zeros_like(data, device=torch.device('cpu'))

    # set `drop_last = True` if using: `torch.backends.cudnn.benchmark = True`
    dataloader_kwargs = dict(drop_last=FLAGS.cudnn_benchmarks)
    dataloader = generate_dataloader(data, FLAGS.batch_size, dataloader_kwargs, DEVICE_KWARGS)

    model.train(mode=False)

    msg = ' 01 :: Conducting inference on batches.'
    for idx, batch_data in enumerate(tqdm.tqdm(dataloader, total=len(dataloader), desc=msg)):

        # conduct inference on the batch
        batch_data = batch_data.to(DEVICE)

        # add noise if applicable
        if FLAGS.config.corruption.noise_std > 0.0:

            mean = torch.zeros(*batch_data.shape)
            std = FLAGS.config.corruption.noise_std * torch.ones(*batch_data.shape)

            maybe_noisy_phi = torch.normal(mean=mean, std=std).to(DEVICE)

        else:
            maybe_noisy_phi = phi

        # corrupt the data
        zeta = corruption_operation(batch_data, maybe_noisy_phi, FLAGS.config.phi_operation)

        # predict u, phi
        batched_u_predictions = model(zeta)

        # move predictions
        batch_slice = slice(idx * FLAGS.batch_size, idx * FLAGS.batch_size + batched_u_predictions.size(0))

        u_corrupted[batch_slice, ...] = zeta.detach().cpu()
        u_predicted[batch_slice, ...] = batched_u_predictions.detach().cpu()

    return_dict: dict[str, Any] = {
        'u_corrupted': u_corrupted,
        'u_predicted': u_predicted,
    }

    return return_dict


def main(_) -> None:

    print('00 :: Running inference on given data.')

    if FLAGS.run_gpu is not None and FLAGS.run_gpu >= 0 and FLAGS.run_gpu < torch.cuda.device_count():

        global DEVICE
        global DEVICE_KWARGS

        if not torch.cuda.is_available():
            raise ValueError('Specified CUDA device unavailable.')

        DEVICE = torch.device(f'cuda:{FLAGS.run_gpu}')
        DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True}

    if FLAGS.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(FLAGS.memory_fraction, DEVICE)

    # load yaml configuration file
    update_config()
    config = FLAGS.config

    # load data from file
    u_all = load_data(h5_file=FLAGS.data_path, config=config)

    # get corruption function and loss function
    u_max = torch.max(u_all).item()
    phi_limit = config.corruption.phi_magnitude * u_max

    phi = get_corruption(
        config.corruption.phi_fn,
        config.data.resolution,
        config.corruption.phi_frequency,
        phi_limit
    )

    # pre-allocate corruption to GPU
    phi = phi.to(DEVICE)

    # initialise model / optimizer
    model = initialise_model().to(torch.float)
    model.to(DEVICE)

    # run inference
    inference_dict = inference(model, u_all, phi)
    inference_dict |= {'phi': phi, 'u_original': u_all}

    # move all tensors to cpu and convert to numpy
    fn_to_numpy = lambda x: x.detach().cpu().numpy()
    inference_dict = dict(zip(inference_dict, map(fn_to_numpy, inference_dict.values())))

    print('02 :: Writing results to file.')
    write_h5(inference_dict)

    print('03 :: Inference complete.')


if __name__ == '__main__':
    app.run(main)
