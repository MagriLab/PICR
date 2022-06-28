from pathlib import Path
from typing import Any, Dict, List, Union

from dataclasses import dataclass, field, fields
import yaml

from .enums import eCorruption, eDecoder, eSolverFunction
from .exceptions import SolverConsistencyError, SolverConsistencyWarning


@dataclass
class ExperimentConfig:

    # _config: Dict[str, Any] = {}
    _config: Dict[str, Any] = field(default_factory=dict, repr=False)

    # data parameters
    NTRAIN: int = field(init=False)
    NVALIDATION: int = field(init=False)
    TIME_STACK: int = field(init=False)

    # model parameters
    N_EPOCHS: int = field(init=False)
    BATCH_SIZE: int = field(init=False)
    LAYERS: list = field(init=False)
    LATENT_DIM: int = field(init=False)
    ACTIVATION: str = field(init=False)
    LR: float = field(init=False)
    L2: int = field(init=False)
    DROPOUT: float = field(init=False)
    BATCH_NORM: bool = field(init=False)
    DECODER: eDecoder = field(init=False)

    # solvers parameters
    SOLVER_FN: eSolverFunction = field(init=False)

    NK: int = field(init=False)
    DT: float = field(init=False)

    C: float = field(init=False)
    RE: float = field(init=False)

    # resolution parameters
    NX: int = field(init=False)
    NU: int = field(init=False)

    # corruption parameters
    PHI_FN: eCorruption = field(init=False)
    PHI_FREQ: Union[float, List[float]] = field(init=False)
    PHI_LIMIT: Union[float, List[float]] = field(init=False)

    # loss parameters
    FWT_LB: float = field(init=False)

    def load_config(self, config_path: Path) -> None:

        """Load values from .yml file into class attributes."""

        # load yaml file to dictionary
        with open(config_path, 'r', encoding='utf8') as f:
            tmp_config = yaml.load(stream=f, Loader=yaml.CLoader)

        # assuming values are dictionaries
        for v in tmp_config.values():
            self._config.update(v)

        # generate a set of existing field names
        field_names = set(map(lambda x: x.name, fields(self)))

        # remove any private fields
        _private_fields = [f for f in field_names if f.startswith('_')]
        field_names.difference_update(_private_fields)

        # update class attributes
        for k, v in self._config.items():

            if k not in field_names:
                raise ValueError(f'Invalid Field: {k} with value {v}')

            k_field = next(filter(lambda x, key=k: x.name == key, fields(self)))                          # type: ignore
            setattr(self, k, k_field.type(v))

            field_names.remove(k)

        if len(field_names) > 0:

            msg = 'Missing values in config file:\n'
            for fname in field_names:
                _field = next(filter(lambda x, key=fname: x.name == key, fields(self)))                   # type: ignore
                msg += f'{_field.name}: {_field.type.__name__}\n'

            raise ValueError(msg)

        self._check_consistency()

    def _check_consistency(self) -> None:

        if self.SOLVER_FN == eSolverFunction.LINEAR and self.C == 0:
            raise SolverConsistencyWarning(solver_type=self.SOLVER_FN)

        if self.SOLVER_FN == eSolverFunction.NONLINEAR and self.C > 0:
            raise SolverConsistencyError(solver_type=self.SOLVER_FN)

    @property
    def config(self) -> Dict[str, Any]:

        """Returns the config dictionary.

        Returns
        -------
        Dict[str, Any]
            Config dictionary.
        """

        return self._config
