from typing import Any, Dict

import h5py
import numpy as np
import tqdm
from absl import app, flags
from kolsol.numpy.solver import KolSol

from src.picr.solvers.proto import Solver

from ..picr.experimental import define_path as pisr_flags
from ..picr.solvers.numpy import LinearCDS, NonlinearCDS
from ..picr.utils.enums import eSystem


NDIM: int = 2

FLAGS = flags.FLAGS

_DATA_PATH = pisr_flags.DEFINE_path(
    'data_path',
    None,
    'Path to save data to.'
)

_SYSTEM = flags.DEFINE_enum_class(
    'system',
    None,
    eSystem,
    'Which system to simulate',
)

_RE = flags.DEFINE_float(
    're',
    42.0,
    'Reynolds number of the flow.'
)

_C_CONVECTIVE = flags.DEFINE_float(
    'c_convective',
    None,
    'Convective coefficient to use -- only for Linear case.'
)

_NF = flags.DEFINE_float(
    'nf',
    4.0,
    'Forcing frequency for Kolmogorov flow'
)

_NK = flags.DEFINE_integer(
    'nk',
    30,
    'Number of symmetric wavenumbers to solve with.'
)

_DT = flags.DEFINE_float(
    'dt',
    5e-3,
    'Time-step for the simulation,'
)

_TIME_SIMULATION = flags.DEFINE_float(
    'time_simulation',
    120.0,
    'Number of seconds to run simulation for.'
)

_TIME_TRANSIENT = flags.DEFINE_float(
    'time_transient',
    180.0,
    'Number of seconds to run transient simulation for.'
)



flags.mark_flags_as_required(['data_path', 'system'])


def setup_directory() -> None:

    """Sets up the relevant simulation directory."""

    if not FLAGS.data_path.suffix == '.h5':
        raise ValueError('setup_directory() :: Must pass .h5 data_path')

    if FLAGS.data_path.exists():
        raise FileExistsError(f'setup_directory() :: {FLAGS.data_path} already exists.')

    FLAGS.data_path.parent.mkdir(parents=True, exist_ok=True)


def write_h5(data: Dict[str, Any]) -> None:

    """Writes results dictionary to .h5 file.

    Parameters
    ----------
    data: Dict[str, Any]
        Data to write to file.
    """

    with h5py.File(FLAGS.data_path, 'w') as hf:

        for k, v in data.items():
            hf.create_dataset(k, data=v)


def get_solver() -> Solver:

    solver: Solver
    match FLAGS.system:

        case eSystem.linear:

            if not FLAGS.re and not FLAGS.c_convective:
                raise ValueError('Need to pass both re and c_convective')

            solver = LinearCDS(nk=FLAGS.nk, c=FLAGS.c_convective, re=FLAGS.re, ndim=NDIM)

        case eSystem.nonlinear:

            if FLAGS.c_convective:
                raise ValueError('Do not pass c_convective if using nonlinear case')

            solver = NonlinearCDS(nk=FLAGS.nk, re=FLAGS.re, ndim=NDIM)

        case eSystem.kolmogorov:

            if FLAGS.c_convective:
                raise ValueError('Do not pass c_convective if using kolmogorov case')

            solver = KolSol(nk=FLAGS.nk, nf=FLAGS.nf, re=FLAGS.re, ndim=NDIM)

        case _:
            raise ValueError('Invalid system given.')

    return solver


def main(_) -> None:

    """Generate System Flow Data."""

    print('Initialising Flow Solver.')

    setup_directory()

    solver = get_solver()
    field_hat = solver.random_field(magnitude=10.0, sigma=1.2, k_offset=None)

    # define time-arrays for simulation run
    t_arange = np.arange(0.0, FLAGS.time_simulation, FLAGS.dt)
    transients_arange = np.arange(0.0, FLAGS.time_transient, FLAGS.dt)

    nt = t_arange.shape[0]
    nt_transients = transients_arange.shape[0]

    # setup recording arrays - only need to record fourier field
    velocity_hat_arr = np.zeros(shape=(nt, solver.nk_grid, solver.nk_grid, NDIM), dtype=np.complex128)

    # integrate over transients if running Kolmogorov case
    if FLAGS.system == eSystem.kolmogorov:

        msg = 'Integrating over transients'
        for _ in tqdm.trange(nt_transients, desc=msg):
            field_hat += FLAGS.dt * solver.dynamics(field_hat)

    # integrate over simulation domain
    msg = 'Integrating over simulation domain'
    for t in tqdm.trange(nt, desc=msg):

        # time integrate
        field_hat += FLAGS.dt * solver.dynamics(field_hat)

        # record metrics
        velocity_hat_arr[t, ...] = field_hat

    # hdf5 cannot store null values
    if not FLAGS.c_convective:
        FLAGS.c_convective = -1.0

    data_dict = {
        'system': FLAGS.system.value,
        're': FLAGS.re,
        'c_convective': FLAGS.c_convective,
        'nf': FLAGS.nf,
        'nk': FLAGS.nk,
        'dt': FLAGS.dt,
        'time': t_arange,
        'velocity_field_hat': velocity_hat_arr,
    }

    print('02 :: Writing results to file.')
    write_h5(data_dict)

    print('03 :: Simulation Done.')


if __name__ == '__main__':
    app.run(main)
