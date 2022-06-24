import argparse
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import time
import subprocess

from multiprocessing import Pool, Queue

import yaml
import torch


queue: Queue = Queue()

FREQUENCIES = [1.0, 3.0, 5.0, 7.0, 9.0]

NUM_GPUS = torch.cuda.device_count()
PROC_PER_GPU = 15
MEMORY_FRACTION = 0.065


class WandbConfig(NamedTuple):

    entity: Optional[str]
    project: Optional[str]
    group: Optional[str]


class Job:

    def __init__(self, config_path: Path, data_path: Path, experiment_path: Path, wandb_config: WandbConfig) -> None:

        """Object to hold job-specific information.

        Parameters
        ----------
        config_path: Path
            Path to the config file used to run the experiment.
        data_path: Path
            Path to the location of the simulation data.
        experiment_path: Path
            Path to save experiment run to.
        wandb_config: WandbConfig
            Named tuple containing relevant information for the W&B interface.
        """

        self.config_path = config_path
        self.data_path = data_path

        self.experiment_path = experiment_path
        self.wandb_config = wandb_config

    def __str__(self) -> str:
        msg = f'Job({self.experiment_path})'
        return msg


def generate_config(config_path: Path, freq: float) -> None:

    """Generate config file for experiment.

    Parameters
    ----------
    config_path: Path
        Path to save the config file to.
    freq: float
        Frequency used to generate the config file.
    """

    # make config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Dict[str, Any]] = {}

    # config :: data_constraints
    config['DATA_PARAMETERS'] = {}
    config['DATA_PARAMETERS']['NTRAIN'] = 1000
    config['DATA_PARAMETERS']['NVALIDATION'] = 500
    config['DATA_PARAMETERS']['TIME_STACK'] = 2

    # config :: ml_constraints
    config['ML_PARAMETERS'] = {}

    config['ML_PARAMETERS']['N_EPOCHS'] = 50
    config['ML_PARAMETERS']['BATCH_SIZE'] = 64
    config['ML_PARAMETERS']['LAYERS'] = [32, 64, 128]
    config['ML_PARAMETERS']['LATENT_DIM'] = 0
    config['ML_PARAMETERS']['ACTIVATION'] = 'Tanh'
    config['ML_PARAMETERS']['LR'] = 0.0003
    config['ML_PARAMETERS']['L2'] = 0.0
    config['ML_PARAMETERS']['DROPOUT'] = 0.0
    config['ML_PARAMETERS']['BATCH_NORM'] = False
    config['ML_PARAMETERS']['DECODER'] = 'UPSAMPLING'

    # config :: simulation_constants
    config['SIMULATION_PARAMETERS'] = {}
    config['SIMULATION_PARAMETERS']['SOLVER_FN'] = 'NONLINEAR'
    config['SIMULATION_PARAMETERS']['NK'] = 8
    config['SIMULATION_PARAMETERS']['DT'] = 0.001
    config['SIMULATION_PARAMETERS']['C'] = 0.0
    config['SIMULATION_PARAMETERS']['RE'] = 500.0
    config['SIMULATION_PARAMETERS']['NX'] = 64
    config['SIMULATION_PARAMETERS']['NU'] = 2

    # config :: corruption_parameters
    config['CORRUPTION_PARAMETERS'] = {}
    config['CORRUPTION_PARAMETERS']['PHI_FN'] = 'RASTRIGIN'
    config['CORRUPTION_PARAMETERS']['PHI_FREQ'] = freq
    config['CORRUPTION_PARAMETERS']['PHI_LIMIT'] = 0.1

    # config :: corruption_parameters
    config['LOSS_PARAMETERS'] = {}
    config['LOSS_PARAMETERS']['FWT_LB'] = 1.0
    config['LOSS_PARAMETERS']['LOSS_SCALING'] = 10000

    with open(config_path, 'w+') as f:
        yaml.dump(config, f)


def run_job(job: Job) -> None:

    """Runs a single job on the next available GPU.

    Parameters
    ----------
    job: Job
        Job object specifying parameters of the experiment to run.
    """

    gpu_id = queue.get()

    try:

        # run processing on GPU <gpu_id>
        print(f'Running {job} on GPU {gpu_id}')

        subprocess_args = [
           'python',
            'base_experiment.py',
            '--experiment-path', job.experiment_path,
            '--data-path', job.data_path,
            '--config-path', job.config_path,
            '--run-gpu', gpu_id,
            '--memory-fraction', MEMORY_FRACTION
        ]

        # ensure all arguments are strings
        subprocess_args = list(map(str, subprocess_args))

        # add wandb options to args
        for k in filter(lambda _k: wandb_dict[_k], wandb_dict := job.wandb_config._asdict()):                # type: str
            subprocess_args.extend([f'--wandb-{k}', wandb_dict[k]])

        stdout_path = job.experiment_path / 'stdout.log'
        stderr_path = job.experiment_path / 'stderr.log'

        job.experiment_path.mkdir(parents=True, exist_ok=False)

        with open(stdout_path, 'w+') as out, open(stderr_path, 'w+') as err:
            _ = subprocess.run(subprocess_args, stdout=out, stderr=err)

        print(f'{job} finished')

    finally:
        queue.put(gpu_id)


def main(args: argparse.Namespace) -> None:

    """Run the Frequency Experiment.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments passed from the command line.
    """

    ts = time.time()

    base_config_path = args.base_experiment_path / 'CONFIGS'
    wandb_config = WandbConfig(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group)

    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    job_list = []
    for freq in FREQUENCIES:

        config_path = base_config_path / f'config_f{int(freq)}.yml'
        generate_config(config_path, freq)

        for idx_run in range(args.n_samples):
            experiment_path = args.base_experiment_path / f'FREQ{int(freq):02}' / f'{idx_run:03}'
            job_list.append(Job(config_path, args.data_path, experiment_path, wandb_config))


    for job in job_list:
        print(job)

    pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)
    pool.map(run_job, job_list)

    pool.close()
    pool.join()

    run_time = time.time() - ts
    print(f'Simulations took {run_time}s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FREQ EXPERIMENT')

    # arguments to define experiment
    parser.add_argument('-ns', '--n-samples', type=int, required=True)

    # arguments to define paths
    parser.add_argument('-bep', '--base-experiment-path', type=Path, required=True)
    parser.add_argument('-dp', '--data-path', type=Path, required=True)

    # arguments for W&B API
    parser.add_argument('--wandb-entity', default=None, type=str)
    parser.add_argument('--wandb-project', default=None, type=str)
    parser.add_argument('--wandb-group', default=None, type=str)

    parsed_args = parser.parse_args()

    main(parsed_args)
