"""WS-C C1: the LoweringTrace (graph -> transform steps -> asm), deterministic pipeline assembly."""
from __future__ import annotations

from merlin.kernels import trace as T


def test_pipeline_steps_spans_all_three_planes():
    steps = T.pipeline_steps("rvv")
    planes = {s.plane for s in steps}
    assert planes == {"dialect", "transform_schedule", "llvm"}
    names = {s.name for s in steps}
    # dialect-plane authored passes, the transform-schedule tiling/vectorize, and lowering are all present
    assert "merlin-outline-dispatches" in names
    assert {"tile_using_for", "vectorize", "lower_contraction"} <= names
    # every step is region-tagged (the coarse C3 taxonomy)
    assert all(s.region for s in steps)


def test_pipeline_steps_ordered_dialect_then_schedule_then_llvm():
    steps = T.pipeline_steps("rvv")
    planes = [s.plane for s in steps]
    # dialect passes come first, llvm descent last (the schedule sits between)
    assert planes[0] == "dialect"
    assert planes[-1] == "llvm"
    assert planes.index("transform_schedule") < planes.index("llvm")


def test_split_pass_list_respects_brace_options():
    # a pass with {space-separated options} must not be split on any comma inside the braces
    s = "canonicalize,one-shot-bufferize{a=1 b=2},cse"
    assert T._split_pass_list(s) == ["canonicalize", "one-shot-bufferize{a=1 b=2}", "cse"]


def test_non_rvv_target_raises():
    import pytest
    with pytest.raises(ValueError):
        T.pipeline_steps("gemmini")


def test_lowering_trace_roundtrips_and_renders():
    steps = T.pipeline_steps("rvv")[:3]
    lt = T.LoweringTrace(
        kernel="matmul_f32_64", target="rvv", source="ours",
        graph=T.GraphRegion(region_id="r0", op="matmul", family="contraction",
                            shape={"M": 64, "N": 64, "K": 64}),
        steps=steps,
        asm=T.AsmRegion(label="k-loop", span=(100, 180),
                        facts={"contraction": "fused_fma", "vfmacc": 64}))
    d = lt.to_dict()
    assert d["kernel"] == "matmul_f32_64" and d["source"] == "ours"
    assert len(d["steps"]) == 3 and d["graph"]["op"] == "matmul"
    md = lt.to_markdown()
    assert "LoweringTrace: matmul_f32_64" in md
    assert "Transformation steps" in md and "Asm region" in md
    # the edit-point (entry) surfaces so an LLM/engineer sees WHERE to change each step
    assert "merlin.llvmlower.passes_xdsl" in md
