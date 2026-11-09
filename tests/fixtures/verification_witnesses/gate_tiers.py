import numpy as np
import pytest

from merlin.runtime.backends.zephyr_model import _gate


def _ramp(n=512, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n).astype(np.float32)


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "_gate",
    "_ramp",
    "np",
    "pytest",
]
