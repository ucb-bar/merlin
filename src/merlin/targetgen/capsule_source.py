"""Historical class and pickle lookups for capsule sources.

Current callers import shared capture definitions from ``merlin.capture`` and
corpus construction from ``merlin.experiments.corpus`` at their defining owners.
This module retains only the established class and exception identities.
"""

from merlin.capture.artifacts import CapsuleArtifacts as CapsuleArtifacts
from merlin.capture.artifacts import M2MUnavailable as M2MUnavailable
from merlin.capture.demands import ModelDemandIncomplete as ModelDemandIncomplete
from merlin.capture.pytorch import PytorchRefSource as PytorchRefSource
from merlin.capture.spec import SpecArtifacts as SpecArtifacts
from merlin.capture.spec import SpecProgramUnavailable as SpecProgramUnavailable
from merlin.capture.spec import SpecRefSource as SpecRefSource
