import enum


class eDecoder(enum.Enum):
    upsampling = 'upsampling'
    transpose = 'transpose'


class eSystem(enum.Enum):
    linear = 'linear'
    nonlinear = 'nonlinear'
    kolmogorov = 'kolmogorov'


class eCorruption(enum.Enum):
    ackley = 'ackley'
    rastrigin = 'rastrigin'
