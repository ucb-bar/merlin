"""Dependence primitives: what an instruction defines and uses, what is live, and what that costs.

Split out from :mod:`merlin.perf.depgraph` so the def-use layer can be read and tested without the
graph, and so the graph's file stays about edges and separations rather than about register files.

The analysis functions are deliberately NOT re-exported by their bare names here: one of them is
called ``liveness``, which is also the module's name, and binding the function over the module makes
``from merlin.perf.deps import liveness`` mean two different things depending on import order. Import
them from :mod:`merlin.perf.deps.liveness` directly.
"""
from .liveness import Access, Effects, Instruction, LivenessReport, Pressure, ValueRange

__all__ = ["Access", "Effects", "Instruction", "LivenessReport", "Pressure", "ValueRange"]
