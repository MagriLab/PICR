import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import tqdm


sys.path.append('../..')
from picr.solvers.numpy import LinearCDS


def setup_directory(path: Path) -> None:

    """Sets up the relevant simulation directory.

    Parameters
    ----------
    path: Path
        Path to .h5 file to write results to.
    """

    if not path.suffix == '.h5':
        raise ValueError('setup_directory() :: Must pass .h5 path.')

    if path.exists():
        raise FileExistsError(f'setup_directory() :: {path} already exists.')

    path.parent.mkdir(parents=True, exist_ok=True)


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


def main(args: argparse.Namespace) -> None:

    """Generate Linear Convection-Diffusion Data.

    Parameters
    ----------
    args: argparse.Namespace
        Command line arguments.
    """

    print('00 :: Initialising Linear Convection-Diffusion Solver.')

    setup_directory(args.data_path)

    cds = LinearCDS(nk=args.nk, c=args.c, re=args.re, ndim=args.ndim)
    field_hat = cds.random_field(magnitude=10.0, sigma=1.2)

    # define time-arrays for simulation run
    t_arange = np.arange(0.0, args.time_simulation, args.dt)

    nt = t_arange.shape[0]

    # setup recording arrays
    velocity_arr = np.zeros(shape=(nt, args.resolution, args.resolution, args.ndim))
    velocity_hat_arr = np.zeros(shape=(nt, cds.nk_grid, cds.nk_grid, args.ndim), dtype=np.complex128)

    dissipation_arr = np.zeros(shape=(nt, 1))

    # integrate over simulation domain
    msg = '01 :: Integrating over simulation domain'
    for t in tqdm.trange(nt, desc=msg):

        # time integrate
        field_hat += args.dt * cds.dynamics(field_hat)

        # record metrics
        velocity_hat_arr[t, ...] = field_hat
        velocity_arr[t, ...] = cds.fourier_to_phys(field_hat, nref=args.resolution)

        dissipation_arr[t, ...] = cds.dissip(field_hat)

    data_dict = {
        'c': args.c,
        're': args.re,
        'dt': args.dt,
        'nk': args.nk,
        'ndim': args.ndim,
        'time': t_arange,
        'resolution': args.resolution,
        'velocity_field': velocity_arr,
        'velocity_field_hat': velocity_hat_arr,
        'dissipation': dissipation_arr
    }

    print('02 :: Writing results to file.')
    write_h5(args.data_path, data_dict)

    print('03 :: Simulation Done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Linear Convection-Diffusion Data.')

    # arguments to define output
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=64)

    # arguments to define simulation
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--re', type=float, default=500.0)

    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--time-simulation', type=float, required=True)

    # arguments for LinearCDS
    parser.add_argument('--nk', type=int, default=30)
    parser.add_argument('--ndim', type=int, default=2)

    parsed_args = parser.parse_args()

    main(parsed_args)
