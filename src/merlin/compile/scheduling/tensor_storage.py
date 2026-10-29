"""Scheduling-namespace name for dense factored-axis storage permutations.

The declaration and its bijection checker live in ``merlin.perf.factored_storage``
beside the grouped-axes storage contract, so interface and storage-encoding
checks need no scheduling-package imports. This module only re-exports it.
"""

from merlin.perf.factored_storage import FactoredTensorStorage, column_panel_storage

__all__ = ["FactoredTensorStorage", "column_panel_storage"]
