"""RVV target-package machinery: fork an iteration of the RVV codegen (a transform-dialect
SCHEDULE + cflags, captured as data), build it in isolation, measure it on coupled targets
(spike correctness + K1 cycles, FireSim later), and compare to a frozen baseline package —
WITHOUT perturbing the global compiler flow.

This mirrors the gemmini ``targetgen`` isolation/certification PATTERN (per-run package dirs
under ``generated_targets/<target>/<run_id>/``, a provenance manifest, a K-ladder runner that
never raises and records ``not_run`` rather than a false pass) but treats RVV as a
schedule-package, not a resident-accelerator dialect — so there is no ``dialect.py`` /
``SPEC_OPS`` / command-buffer here. The plug-back-in seam is the existing
``lower_to_llvm_ir(transform_schedule=...)`` parameter, threaded through ``build_app``.
"""
from __future__ import annotations

from .registry import RvvPackage, load_rvv_package, default_run

__all__ = ["RvvPackage", "load_rvv_package", "default_run"]
