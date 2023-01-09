import ml_collections

from src.experiments.configs import common, experiment_config, simulation


def get_config(system_exp_case: str) -> ml_collections.ConfigDict:

    system, exp, case = system_exp_case.split('@')

    config = common.get_config()

    # use the relevant simulation configuration
    get_simulation_config = getattr(simulation, f'get_{system}_config')
    simulation_config = get_simulation_config()

    config.simulation = simulation_config

    # update with the relevant experiment configuration
    get_experiment_config_diff = getattr(experiment_config, f'get_{exp}_config')
    config_diff = get_experiment_config_diff(case)

    config.update_from_flattened_dict(config_diff)

    return config
