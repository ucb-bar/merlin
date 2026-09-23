"""Experiment definitions over Merlin's existing phase engines.

The execution record tracks process attempts only. Native engine artifacts and AET
remain authoritative for grades, measurements, accounting, and agent checkpoints.
"""

from .spec import ExperimentSpec, SpecError, load_spec

__all__ = ["ExperimentSpec", "SpecError", "load_spec"]
