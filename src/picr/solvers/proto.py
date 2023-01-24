from typing import Optional, Protocol, TypeVar


T = TypeVar('T')

class Solver(Protocol[T]):

    nk: int
    nk_grid: int

    ndim: int

    kk: T

    def dynamics(self, u_hat: T) -> T:
        ...

    # TODO >> Should not be part of Protocol
    def phys_to_fourier(self, t: T) -> T:
        ...

    # TODO >> Should not be part of Protocol
    def random_field(self, magnitude: float, sigma: float, k_offset: Optional[list[int]]) -> T:
        ...
