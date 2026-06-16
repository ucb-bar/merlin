"""DSE guidance: turn a flat workload capture into grounded DSE-axis guidance.

Merlin is not the DSE optimizer. This package is a DSE *instrument*: it reconstructs the
multi-rate structure a flat capture hides, measures (or trace-derives) where the time goes,
and ranks accelerator DSE axes by how much of the *measured* target gap each axis can close.

    Merlin does not perform DSE. Merlin prevents DSE from optimizing the wrong abstraction.

Every emitted number carries an evidence tag (see :mod:`merlin.dse_guidance.evidence`):
``measured | trace_derived | calibrated | structural_bound | analytical | assumed``. No
important number is reported without a source. Measured/trace evidence is ingested from the
``aet`` harness (see :mod:`merlin.dse_guidance.aet_ingest`), never hand-coded as constants.
"""
from __future__ import annotations
