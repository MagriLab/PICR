import ml_collections

from src.experiments.configs import common, simulation


def get_config(system: str) -> ml_collections.ConfigDict:

    config = common.get_config()

    get_simulation_config = getattr(simulation, f'get_{system}_config')
    simulation_config = get_simulation_config()

    config.simulation = simulation_config

    return config
