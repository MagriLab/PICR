from typing import Any, TypeVar

import numpy as np
import torch


TypeTensor = np.ndarray | torch.Tensor
T = TypeVar('T', np.ndarray, torch.Tensor)

ExperimentConfig = Any
