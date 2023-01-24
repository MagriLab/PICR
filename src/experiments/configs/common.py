import ml_collections

from src.picr.utils.enums import eCorruption, eCorruptionOperation, eDecoder


def get_config() -> ml_collections.ConfigDict:

    config = ml_collections.ConfigDict()

    # data configuration
    config.data = ml_collections.ConfigDict()

    config.data.ntrain = 1024
    config.data.nvalidation = 256

    config.data.resolution = 64
    config.data.tau = 2

    # simulation configuration -- will be set from ./simulation.py
    config.simulation = ml_collections.ConfigDict()

    # training configuration
    config.training = ml_collections.ConfigDict()

    config.training.layers = [32, 64, 128]
    config.training.latent_dim = 0
    config.training.decoder = eDecoder.upsampling

    config.training.epochs = 1000
    config.training.batch_size = 32
    config.training.learning_rate = 3e-4
    config.training.l2 = 0.0
    config.training.fwt_lb = 1.0
    config.training.lambda_weight = 1e3

    # corruption configuration
    config.corruption = ml_collections.ConfigDict()

    config.corruption.phi_fn = eCorruption.rastrigin
    config.corruption.phi_frequency = 3.0
    config.corruption.phi_magnitude = 0.5
    config.corruption.phi_operation = eCorruptionOperation.additive

    config.corruption.noise_std = 0.0

    return config
