import ml_collections

from src.picr.utils.enums import eSystem


def get_linear_config() -> ml_collections.ConfigDict:

    simulation_config = ml_collections.ConfigDict()

    simulation_config.system = eSystem.linear

    simulation_config.nk = 30
    simulation_config.dt = 5e-3

    simulation_config.re = None
    simulation_config.c_convective = 1.0

    return simulation_config


def get_nonlinear_config() -> ml_collections.ConfigDict:

    simulation_config = ml_collections.ConfigDict()

    simulation_config.system = eSystem.nonlinear

    simulation_config.nk = 30
    simulation_config.dt = 5e-3

    simulation_config.re = 500.0
    simulation_config.c_convective = None

    return simulation_config


def get_kolmogorov_config() -> ml_collections.ConfigDict:

    simulation_config = ml_collections.ConfigDict()

    simulation_config.system = eSystem.kolmogorov

    simulation_config.nk = 30
    simulation_config.dt = 5e-3

    simulation_config.re = 42.0
    simulation_config.c_convective = None

    return simulation_config
