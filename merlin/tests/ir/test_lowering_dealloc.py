"""Every lowering pipeline must free the buffers it allocates, and must do it in the right place.

Bufferization emits a `memref.alloc` per intermediate tensor and leaves deallocation to a separate
pass. That pass was missing, so a lowered model's heap demand was the sum of every temporary it ever
materialized rather than the peak live at once -- whisper_tiny allocated 678 buffers, freed none, and
died on a board when a 54 MB `malloc` returned NULL and generated code stored through it.

Two properties are worth pinning down, because both were arrived at by being wrong first:

1. the deallocation passes are in ALL the pipelines (it is a whole-module correctness property, not
   an RVV one);
2. in the vectorized pipeline they run AFTER `lower-vector-mask`. `ownership-based-buffer-deallocation`
   materializes an i1 ownership constant beside each buffer use and walks into `vector.mask` regions,
   which accept exactly one masked operation -- so running it earlier makes the region fail its own
   verifier. deepjscc happens not to trip that; whisper does. An ordering that only some models
   exercise is exactly the kind that regresses silently.
"""
from __future__ import annotations

from merlin.llvmlower import pipeline as P

DEALLOC = "ownership-based-buffer-deallocation"


def _stages(text: str) -> list[str]:
    return [s for s in text.split(",") if s]


def test_every_pipeline_deallocates():
    for name, text in (("upstream", P._upstream_pipeline()),
                       ("parallel", P._parallel_pipeline()),
                       ("rvv", P.build_rvv_pipeline("", hoist_static_allocs=True,
                                                    features=frozenset()))):
        assert DEALLOC in _stages(text), f"{name} pipeline bufferizes but never deallocates"


def test_dealloc_runs_after_vector_masks_are_lowered():
    stages = _stages(P.build_rvv_pipeline("", hoist_static_allocs=True, features=frozenset()))
    mask = max(i for i, s in enumerate(stages) if "lower-vector-mask" in s)
    assert stages.index(DEALLOC) > mask


def test_dealloc_runs_before_control_flow_is_flattened():
    """Ownership analysis wants structured control flow; once scf is cf there is nothing to own."""
    stages = _stages(P.build_rvv_pipeline("", hoist_static_allocs=True, features=frozenset()))
    assert stages.index(DEALLOC) < stages.index("convert-scf-to-cf")


def test_dealloc_is_defeatable_for_an_ab(monkeypatch):
    """The A/B escape hatch is the only honest way to defend "numerics unchanged"."""
    monkeypatch.setenv("MERLIN_NO_DEALLOC", "1")
    assert DEALLOC not in _stages(P._upstream_pipeline())
