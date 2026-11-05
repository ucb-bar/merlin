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

import pytest

from merlin.llvmlower import pipeline as P

DEALLOC = "ownership-based-buffer-deallocation"


def _stages(text: str) -> list[str]:
    return [s for s in text.split(",") if s]


def test_every_pipeline_deallocates():
    for name, text in (
        ("upstream", P._upstream_pipeline()),
        ("parallel", P._parallel_pipeline()),
        ("rvv", P.build_rvv_pipeline("", hoist_static_allocs=True, features=frozenset())),
    ):
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


def test_allocation_moves_precede_ownership_and_serial_output_forwarding():
    for pipeline in (P._upstream_pipeline(), P.build_rvv_pipeline("", features=frozenset())):
        stages = _stages(pipeline)
        assert max(
            i for i, s in enumerate(stages) if "buffer-hoisting" in s or "buffer-loop-hoisting" in s
        ) < stages.index(DEALLOC)
    stages = _stages(P._upstream_pipeline())
    bufferized = next(i for i, s in enumerate(stages) if "one-shot-bufferize" in s)
    forwarded = next(i for i, s in enumerate(stages) if "buffer-results-to-out-params" in s)
    assert "canonicalize" in stages[bufferized + 1 : forwarded]


def test_explicit_hoist_after_ownership_is_rejected():
    with pytest.raises(ValueError, match="invalidates ownership"):
        P._splice(["__DEALLOC__", "func.func(buffer-hoisting,buffer-loop-hoisting)"])


@pytest.mark.parametrize("hoist_before", [True, False])
@pytest.mark.parametrize("sliced", [True, False])
def test_split_runners_preserve_callback_stage_anchors(hoist_before, sliced):
    calls = []
    namespace = {
        "_run_stages": lambda *args: calls.append(args),
        "_split_fresh_local_deallocs": lambda *args: None,
        "_slice_unused_elementwise_outputs": lambda *args: None,
        "_dealloc_placement_violations": lambda *args: ([], 0, 0),
    }
    runner = P.SLICED_OUTPUT_RUNNER if sliced else P.SPLIT_FRESH_DEALLOCS_RUNNER
    marker = (
        P.SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER if sliced else "__merlin_split_fresh_deallocs__"
    )
    exec(runner, namespace)
    hoist = "func.func(buffer-hoisting,buffer-loop-hoisting)"
    passes = ["linalg-generalize-named-ops"]
    if hoist_before:
        passes.append(hoist)
    passes.extend([DEALLOC, "optimize-allocation-liveness", marker])
    if not hoist_before:
        passes.append(hoist)
    passes.append("convert-scf-to-cf")
    namespace["_run_stages"](None, None, ",".join(passes), 3, ("mid",), ("late",), ("post",), ("pre",))
    assert len(calls) == 2
    head, tail = calls
    assert head[7] == ("pre",) and tail[7] == ()
    assert head[5:7] == ((), ()) and tail[5:7] == (("late",), ("post",))
    chosen, other = (head, tail) if hoist_before else (tail, head)
    assert "buffer-loop-hoisting" in chosen[2]
    assert chosen[3:5] == (3, ("mid",)) and other[3:5] == (0, ())
