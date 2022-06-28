import enum


class eDecoder(enum.Enum):
    UPSAMPLING = 'UPSAMPLING'
    TRANSPOSE = 'TRANSPOSE'


class eSolverFunction(enum.Enum):
    LINEAR = 'LINEAR'
    NONLINEAR = 'NONLINEAR'
    KOLMOGOROV = 'KOLMOGOROV'


class eCorruption(enum.Enum):
    ACKLEY = 'ACKLEY'
    RASTRIGIN = 'RASTRIGIN'
    POWER = 'POWER'
