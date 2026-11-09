"""Host-owned experiment declarations; engines retain their existing lifecycle owners."""

from .definition import ExperimentSpec, SpecError, load_spec

__all__ = ["ExperimentSpec", "SpecError", "load_spec"]
