import argparse
from pathlib import Path
from typing import List, NamedTuple, Optional

import time
import subprocess

from multiprocessing import Pool, Queue

import torch
import yaml


queue: Queue = Queue()

NUM_GPUS = torch.cuda.device_count()
PROC_PER_GPU = 10
MEMORY_FRACTION = 0.095


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


def generate_config(experiment_config_path: Path, derived_config_path: Path, mag: float) -> None:

    """Generate config file for experiment.

    Parameters
    ----------
    experiment_config_path: Path
        Path pointing to existing config file to change.
    derived_config_path: Path
        Path to save the config file to.
    mag: float
        Magnitude used to generate the config file.
    """

    # make config directory if it doesn't exist
    derived_config_path.parent.mkdir(parents=True, exist_ok=True)

    # open experiment config
    with open(experiment_config_path, 'r', encoding='utf8') as f:
        config = yaml.load(stream=f, Loader=yaml.CLoader)

    # override magnitude and write derived config
    config['CORRUPTION_PARAMETERS']['PHI_LIMIT'] = mag

    with open(derived_config_path, 'w+') as f:
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
        for k in filter(lambda _k: wandb_dict[_k], wandb_dict := job.wandb_config._asdict()):             # type: ignore
            subprocess_args.extend([f'--wandb-{k}', wandb_dict[k]])

        stdout_path = job.experiment_path / 'stdout.log'
        stderr_path = job.experiment_path / 'stderr.log'

        job.experiment_path.mkdir(parents=True, exist_ok=False)

        with open(stdout_path, 'w+') as out, open(stderr_path, 'w+') as err:
            _ = subprocess.run(subprocess_args, stdout=out, stderr=err, check=False)

        print(f'{job} finished')

    finally:
        queue.put(gpu_id)


def get_magnitudes(experiment_config_path: Path) -> List[float]:

    """Extract magnitudes from experiment config.

    Parameters
    ----------
    experiment_config_path: Path
        Path to config file to dictate magnitude experiments.

    Returns
    -------
    magnitudes: List[float]
        List of all the magnitudes to run for the experiment.
    """

    with open(experiment_config_path, 'r', encoding='utf8') as f:
        experiment_config = yaml.load(stream=f, Loader=yaml.CLoader)

    magnitudes = experiment_config['CORRUPTION_PARAMETERS']['PHI_LIMIT']

    if not isinstance(magnitudes, list):
        raise ValueError('Must provide a list of PHI_LIMIT.')

    return magnitudes


def main(args: argparse.Namespace) -> None:

    """Run the Magnitude Experiment.

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

    # get magnitudes to test
    magnitudes = get_magnitudes(args.experiment_config_path)

    job_list = []
    for idx_mag, mag in enumerate(magnitudes):

        config_path = base_config_path / f'config_f{idx_mag}.yml'
        generate_config(experiment_config_path=args.experiment_config_path, derived_config_path=config_path, mag=mag)

        for idx_run in range(args.n_samples):
            experiment_path = args.base_experiment_path / f'MAG{idx_mag:02}' / f'{idx_run:03}'
            job_list.append(Job(config_path, args.data_path, experiment_path, wandb_config))

    with Pool(processes=PROC_PER_GPU * NUM_GPUS) as pool:
        pool.map(run_job, job_list)

        pool.close()
        pool.join()

    run_time = time.time() - ts
    print(f'Simulations took {run_time}s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MAG EXPERIMENT')

    # arguments to define experiment
    parser.add_argument('-ns', '--n-samples', type=int, required=True)

    # arguments to define paths
    parser.add_argument('-ecp', '--experiment-config-path', type=Path, required=True)
    parser.add_argument('-bep', '--base-experiment-path', type=Path, required=True)
    parser.add_argument('-dp', '--data-path', type=Path, required=True)

    # arguments for W&B API
    parser.add_argument('--wandb-entity', default=None, type=str)
    parser.add_argument('--wandb-project', default=None, type=str)
    parser.add_argument('--wandb-group', default=None, type=str)

    parsed_args = parser.parse_args()

    main(parsed_args)
