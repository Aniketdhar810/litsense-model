"""Global seeding (CLAUDE.md s2.2 #8): one seed from config for every RNG a stage may touch."""

import importlib.util
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed ``random``, numpy's global RNG, torch (if installed) and ``PYTHONHASHSEED``.

    ``PYTHONHASHSEED`` only affects interpreters started after this call (for example spaCy
    worker processes); the current process fixed its string hashing at startup.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)  # seeds the CPU and every CUDA device
