import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import einops
import h5py
import numpy as np
import tqdm

sys.path.append('../..')
from picr.solvers.numpy import NonLinearKFS


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

    """Run Kolmogorov flow solvers.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from the command line.
    """

    print('00 :: Initialising Kolmogorov Flow Solver.')

    setup_directory(args.data_path)

    ks = NonLinearKFS(nk=args.nk, nf=args.nf, re=args.re, ndim=args.ndim)
    field_hat = ks.random_field(magnitude=10.0, sigma=2.0)

    # define time-arrays for simulation run
    t_arange = np.arange(0.0, args.time_simulation, args.dt)
    transients_arange = np.arange(0.0, args.time_transient, args.dt)

    nt = t_arange.shape[0]
    nt_transients = transients_arange.shape[0]

    # setup recording arrays
    velocity_arr = np.zeros(shape=(nt, args.resolution, args.resolution, args.ndim))
    velocity_hat_arr = np.zeros(shape=(nt, ks.nk_grid, ks.nk_grid, args.ndim), dtype=np.complex128)

    pressure_arr = np.zeros(shape=(nt, args.resolution, args.resolution, 1))
    pressure_arr_hat = np.zeros(shape=(nt, ks.nk_grid, ks.nk_grid, 1), dtype=np.complex128)

    dissipation_arr = np.zeros(shape=(nt, 1))

    # integrate over transients
    msg = '01 :: Integrating over transients.'
    for _ in tqdm.trange(nt_transients, desc=msg):
        field_hat += args.dt * ks.dynamics(field_hat)

    # integrate over simulation domain
    msg = '02 :: Integrating over simulation domain.'
    for t in tqdm.trange(nt, desc=msg):

        # time integrate
        field_hat += args.dt * ks.dynamics(field_hat)
        pressure_hat = einops.rearrange(ks.pressure(field_hat), 'i j -> i j 1')

        # record metrics
        velocity_hat_arr[t, ...] = field_hat
        velocity_arr[t, ...] = ks.fourier_to_phys(field_hat, nref=args.resolution)

        pressure_arr_hat[t, ...] = pressure_hat
        pressure_arr[t, ...] = ks.fourier_to_phys(pressure_hat, nref=args.resolution)

        dissipation_arr[t, ...] = ks.dissip(field_hat)

    data_dict = {
        're': args.re,
        'dt': args.dt,
        'nk': args.nk,
        'nf': args.nf,
        'ndim': args.ndim,
        'time': t_arange,
        'resolution': args.resolution,
        'velocity_field': velocity_arr,
        'velocity_field_hat': velocity_hat_arr,
        'pressure_field': pressure_arr,
        'pressure_field_hat': pressure_arr_hat,
        'dissipation': dissipation_arr
    }

    print('03 :: Writing results to file.')
    write_h5(args.data_path, data_dict)

    print('04 :: Simulation Done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Kolmogorov Flow Data')

    # arguments to define output
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--resolution', type=int, required=True)

    # arguments to define simulation
    parser.add_argument('--re', type=float, required=True)

    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--time-simulation', type=float, required=True)
    parser.add_argument('--time-transient', type=float, default=180.0)

    # arguments for KolSol
    parser.add_argument('--nk', type=int, default=8)
    parser.add_argument('--nf', type=int, default=4)
    parser.add_argument('--ndim', type=int, default=2)

    parsed_args = parser.parse_args()

    main(parsed_args)
