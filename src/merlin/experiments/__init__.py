"""Experiment definitions for test generation and backend generation.

The execution record tracks process attempts only. Native engine artifacts and AET
remain authoritative for grades, measurements, accounting, and agent checkpoints.
"""

from merlin.experiments.definition import ExperimentSpec, SpecError, load_spec

__all__ = ["ExperimentSpec", "SpecError", "load_spec"]
