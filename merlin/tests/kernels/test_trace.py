"""WS-C C1: the LoweringTrace (graph -> transform steps -> asm), deterministic pipeline assembly."""
from __future__ import annotations

from pathlib import Path

from merlin.common.paths import merlin_dir
from merlin.kernels import trace as T
from merlin.kernels.decode import objdump, rvv

_ASM_DIR = merlin_dir() / "tests" / "data" / "cca_asm"


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


def test_asm_region_from_expert_fixture(monkeypatch):
    # the asm end of the trace, read structurally from a real expert disassembly (reuses cca.lift_asm)
    monkeypatch.setattr(objdump, "disassemble_text",
                        lambda *a, **k: (_ASM_DIR / "openblas_sgemm_rvv.objdump").read_text())
    ar = T.asm_region_from_stream(rvv.decode("openblas.o"), op="matmul", source="openblas", label="k-loop")
    assert ar.label == "k-loop"
    assert ar.facts["contraction_form"] == "fused_fma"
    assert ar.facts["accumulator_resident"] is True
    assert ar.span is not None and ar.span[0] <= ar.span[1]


def test_graph_region_from_record():
    from merlin.frontends.linalg_mlir import MatmulRecord
    rec = MatmulRecord(kind="linalg.matmul", m=64, k=64, n=64, lhs_shape=(64, 64), rhs_shape=(64, 64),
                       dtype="f32", weight_arg_index=1, weight_name="w",
                       prov={"prov.op": "matmul", "prov.family": "contraction",
                             "prov.region_id": "r7", "prov.module": "model.layers.0"})
    gr = T.graph_region_from_record(rec)
    assert gr.region_id == "r7" and gr.op == "matmul" and gr.family == "contraction"
    assert gr.shape == {"M": 64, "N": 64, "K": 64}
    assert gr.module == "model.layers.0"


def _fix(monkeypatch, name):
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: (_ASM_DIR / name).read_text())
    return rvv.decode(f"{name}.o")


def test_expert_steps_from_contract():
    # the expert's DECLARED transformation sequence, in lowering order, from the framework contract
    for fw in ("openblas", "xnnpack"):
        steps = T.expert_steps_from_contract(fw)
        assert [s.name for s in steps] == ["operand-layout", "operand-prepack",
                                           "accumulator-block", "epilogue"]
        assert all(s.plane == "framework" for s in steps)
        assert any(s.region == "register-residency" for s in steps)   # the accumulator block


def test_build_expert_trace_graph_none_by_design(monkeypatch):
    tr = T.build_expert_trace("xnnpack", _fix(monkeypatch, "xnnpack_f32_gemm_rvv.objdump"),
                              op="matmul", kernel_id="xnnpack_f32_gemm_rvv")
    assert tr.source == "xnnpack" and tr.graph is None        # expert graph deferred by design
    assert len(tr.steps) == 4
    assert tr.asm.facts["contraction_form"] == "fused_fma"


def test_build_our_trace_threads_graph_pipeline_asm(monkeypatch):
    from merlin.frontends.linalg_mlir import MatmulRecord
    rec = MatmulRecord(kind="linalg.matmul", m=64, k=64, n=64, lhs_shape=(64, 64), rhs_shape=(64, 64),
                       dtype="f32", weight_arg_index=1, weight_name="w", prov={"prov.op": "matmul"})
    tr = T.build_our_trace(_fix(monkeypatch, "ours_baseline_matmul.objdump"), op="matmul", record=rec)
    assert tr.source == "ours" and tr.graph is not None       # full thread: graph present
    assert {s.plane for s in tr.steps} == {"dialect", "transform_schedule", "llvm"}
    assert tr.asm is not None


def test_emit_trace_versioned(tmp_path, monkeypatch):
    # redirect the out/ root to tmp so the test doesn't pollute the real out/artifacts
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    tr = T.build_expert_trace("openblas", _fix(monkeypatch, "openblas_sgemm_rvv.objdump"),
                              op="matmul", kernel_id="openblas_sgemm_rvv")
    out = T.emit_trace(tr, version=1)
    assert (out / "manifest.yaml").is_file()
    assert any(p.suffix == ".yaml" and p.name.startswith("trace_") for p in out.iterdir())
    assert any(p.suffix == ".md" for p in out.iterdir())
    # convention-compliant location: out/artifacts/lowering-trace/rvv/v1/...
    assert "lowering-trace" in str(out) and "/rvv/v1/" in str(out).replace("\\", "/")
