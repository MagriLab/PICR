import argparse
import csv
import sys
from pathlib import Path
from shutil import copyfile
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

sys.path.append('../..')
from picr.model import Autoencoder
from picr.loss import LinearCDLoss, NonLinearKFLoss

from picr.corruption import ackley, rastrigin
from picr.experiments.data import load_data, train_validation_split, generate_dataloader

from picr.utils.config import ExperimentConfig
from picr.utils.enums import eCorruption, eSolverFunction

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# machine constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == 'cuda' else {}


def train(train_loader: DataLoader,
          validation_loader: DataLoader,
          config: ExperimentConfig,
          args: argparse.Namespace,
          wandb_run: Union[Run, RunDisabled, None]) -> None:

    # get phi function
    if config.PHI_FN == eCorruption.ACKLEY:
        phi_fn = ackley
    elif config.PHI_FN == eCorruption.RASTRIGIN:
        phi_fn = rastrigin
    else:
        raise ValueError('Incompatible corruption function...')

    # define spatial grid for phi
    x = torch.linspace(0.0, 2.0 * np.pi, config.NX)
    xx = torch.stack(torch.meshgrid(x, x), dim=-1).to(DEVICE)

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

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    model.to(DEVICE)

    # define optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.L2)

    # define loss functions
    mse_loss_fn = nn.MSELoss()
    piml_loss_fn: Union[LinearCDLoss, NonLinearKFLoss]

    if config.SOLVER_FN == eSolverFunction.LINEAR:
        piml_loss_fn = LinearCDLoss(nk=config.NK, c=config.C, re=config.RE, dt=config.DT, fwt_lb=config.FWT_LB, device=DEVICE)
    elif config.SOLVER_FN == eSolverFunction.NONLINEAR:
        piml_loss_fn = NonLinearKFLoss(nk=config.NK, nf=config.NF, re=config.RE, dt=config.DT, fwt_lb=config.FWT_LB, device=DEVICE)
    else:
        raise ValueError('Incompatible Loss Type...')

    # monitor gradients of the model
    if isinstance(wandb_run, Run):
        wandb_run.watch(models=model, criterion=mse_loss_fn)

    # prepare results file
    with open(args.experiment_path / 'results.csv', 'w+', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow([
            'epoch',
            'train_Ru_loss',
            'validation_Ru_loss',
            'train_Rg_loss',
            'validation_Rg_loss',
            'train_total_loss',
            'validation_total_loss',
            'train_clean_loss',
            'validation_clean_loss'
        ])

    # initialise minimum validation loss
    min_validation_loss = np.Inf

    # iterate over N_EPOCHS
    for epoch in range(config.N_EPOCHS):

        # reset losses to zero
        train_Ru_loss = 0.0
        train_Rg_loss = 0.0
        train_total_loss = 0.0
        train_clean_loss = 0.0

        validation_Ru_loss = 0.0
        validation_Rg_loss = 0.0
        validation_total_loss = 0.0
        validation_clean_loss = 0.0

        model.train()
        for batch_idx, data in enumerate(train_loader):

            # conduct inference on the batch
            data = data.to(DEVICE)

            # create corrupted data
            phi = phi_fn(xx, freq=config.PHI_FREQ, limit=config.PHI_LIMIT)
            zeta = data + phi

            # predict phi
            u_prediction = model(zeta)
            phi_prediction = zeta - u_prediction

            # 01 :: Clean Velocity Field :: || R(\hat{u}) || = 0
            r_zeta_phi_loss = piml_loss_fn.calc_residual(u_prediction)

            # 02 :: Residual Matching :: || R(u + \phi) - R(\hat{phi}) - g(\hat{u}, \hat{\phi}) || = 0
            r_zeta = piml_loss_fn.calc_residual(zeta)
            r_phi = piml_loss_fn.calc_residual(phi_prediction)
            g_u_phi = piml_loss_fn.calc_g_u_phi(u_prediction, phi_prediction)

            r_g_loss = torch.abs(r_zeta - r_phi - g_u_phi)

            # 03 :: Total Loss
            total_loss = r_zeta_phi_loss + r_g_loss
            total_loss *= config.LOSS_SCALING

            # 04 :: Phi Loss -- Clean
            clean_loss = mse_loss_fn(phi, phi_prediction)

            # update batch losses
            train_Ru_loss += r_zeta_phi_loss.item() * data.size(0)
            train_Rg_loss += r_g_loss.item() * data.size(0)
            train_total_loss += total_loss.item() * data.size(0)
            train_clean_loss += clean_loss.item() * data.size(0)

            # update network
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()
        for batch_idx, data in enumerate(validation_loader):

            # conduct inference on the batch
            data = data.to(DEVICE)

            # create corrupted data
            phi = phi_fn(xx, freq=config.PHI_FREQ, limit=config.PHI_LIMIT)
            zeta = data + phi

            # predict phi
            u_prediction = model(zeta)
            phi_prediction = zeta - u_prediction

            # 01 :: Clean Velocity Field :: || R(\hat{u}) || = 0
            r_zeta_phi_loss = piml_loss_fn.calc_residual(u_prediction)

            # 02 :: Residual Matching :: || R(u + \phi) - R(\hat{phi}) - g(\hat{\u}, \hat{\phi}) || = 0
            r_zeta = piml_loss_fn.calc_residual(zeta)
            r_phi = piml_loss_fn.calc_residual(phi_prediction)
            g_u_phi = piml_loss_fn.calc_g_u_phi(u_prediction, phi_prediction)

            r_g_loss = torch.abs(r_zeta - r_phi - g_u_phi)

            # 03 :: Total Loss
            total_loss = r_zeta_phi_loss + r_g_loss
            total_loss *= config.LOSS_SCALING

            # 04 :: Phi Loss -- Clean
            clean_loss = mse_loss_fn(phi, phi_prediction)

            # update batch losses
            validation_Ru_loss += r_zeta_phi_loss.item() * data.size(0)
            validation_Rg_loss += r_g_loss.item() * data.size(0)
            validation_total_loss += total_loss.item() * data.size(0)
            validation_clean_loss += clean_loss.item() * data.size(0)

        # normalising batch losses
        train_Ru_loss /= len(train_loader.dataset)
        train_Rg_loss /= len(train_loader.dataset)
        train_total_loss /= len(train_loader.dataset)
        train_clean_loss /= len(train_loader.dataset)

        validation_Ru_loss /= len(validation_loader.dataset)
        validation_Rg_loss /= len(validation_loader.dataset)
        validation_total_loss /= len(validation_loader.dataset)
        validation_clean_loss /= len(validation_loader.dataset)

        # checkpointing the model
        if validation_total_loss < min_validation_loss:
            min_validation_loss = validation_total_loss
            torch.save(model.state_dict(), args.experiment_path / 'autoencoder.pt')

        # logging results
        log_dict = {
            'train_Ru_loss': train_Ru_loss,
            'validation_Ru_loss': validation_Ru_loss,
            'train_Rg_loss': train_Rg_loss,
            'validation_Rg_loss': validation_Rg_loss,
            'train_total_loss': train_total_loss,
            'validation_total_loss': validation_total_loss,
            'train_clean_loss': train_clean_loss,
            'validation_clean_loss': validation_clean_loss
        }

        # log results to W&B
        if isinstance(wandb_run, Run):
            wandb_run.log(data=log_dict)

        # log results to sys.stdout
        msg = f'Epoch: {epoch:05}'
        for k, v in log_dict.items():
            msg += f' | {k}: {v:011.5f}'
        print(msg)

        # log results to file
        _results = [
            epoch,
            train_Ru_loss,
            validation_Ru_loss,
            train_Rg_loss,
            validation_Rg_loss,
            train_total_loss,
            validation_total_loss,
            train_clean_loss,
            validation_clean_loss
        ]

        with open(args.experiment_path / 'results.csv', 'a', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)

        # upload results to W&B
    if isinstance(wandb_run, Run):
        artifact = wandb.Artifact(name=str(args.experiment_path).replace('/', '.'), type='dataset')
        artifact.add_dir(local_path=str(args.experiment_path))

        wandb_run.log_artifact(artifact_or_path=artifact)


def main(args: argparse.Namespace) -> None:

    # load yaml configuration file for experiment
    config = ExperimentConfig()
    config.load_config(args.config_path)

    # setup wandb api
    wandb_run: Union[Run, RunDisabled, None] = None
    if args.wandb_entity:

        # initialise W&B API
        wandb_run = wandb.init(
            config=config.config,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=str(args.experiment_path)
        )

    # log current code state to W&B
    if isinstance(wandb_run, Run):
        wandb_run.log_code(str(Path.cwd()))

    # setup the experiment path and copy config file
    args.experiment_path.mkdir(parents=True, exist_ok=False)
    copyfile(args.config_path, args.experiment_path / 'config.yml')

    # load data
    u_all = load_data(h5_file=args.data_path, config=config)
    train_u, validation_u = train_validation_split(u_all, config.NTRAIN, config.NVALIDATION, step=config.TIME_STACK)

    train_loader = generate_dataloader(train_u, config.BATCH_SIZE, DEVICE_KWARGS)
    validation_loader = generate_dataloader(validation_u, config.BATCH_SIZE, DEVICE_KWARGS)

    # run the training process
    train(train_loader, validation_loader, config, args, wandb_run)

    print('DONE.')


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
