from dataclasses import replace

import pytest

from merlin.experiments.phase2 import protected_final_evaluation as F
from merlin.experiments.phase2.contracts import StageGateError, sha256_file
from verification_witnesses.numerical_readback import native_case as _native_case
from verification_witnesses.protected_final_evaluation import _request

native_case = _native_case

# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "F",
    "StageGateError",
    "_native_case",
    "_request",
    "native_case",
    "pytest",
    "replace",
    "sha256_file",
]
