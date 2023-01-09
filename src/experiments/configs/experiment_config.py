from typing import Any, TypeAlias


FlattenedDict: TypeAlias = dict[str, Any]


def get_frequency_config(case: str) -> FlattenedDict:

    """Frequency Experiment -- f in {1, 3, 5, 7, 9}"""

    magnitude = 0.5

    frequency = {
        'C1': 1,
        'C2': 3,
        'C3': 5,
        'C4': 7,
        'C5': 9,
    }[case]

    config_dict = {
        'corruption.phi_frequency': frequency,
        'corruption.phi_magnitude': magnitude
    }

    return config_dict


def get_magnitude_config(case: str) -> FlattenedDict:

    """Magnitude Experiment -- M in {0.01, 0.1, 0.25, 0.5, 1.0}"""

    frequency = 3.0

    magnitude = {
        'C1': 0.01,
        'C2': 0.1,
        'C3': 0.25,
        'C4': 0.5,
        'C5': 1.0,
    }[case]

    config_dict = {
        'corruption.phi_frequency': frequency,
        'corruption.phi_magnitude': magnitude
    }

    return config_dict
