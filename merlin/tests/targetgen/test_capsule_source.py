"""The grounded capsule sources: a capsule defined in PyTorch must lower to 0-opaque linalg-on-tensors
via model2MLIR and carry a host torch-eager reference golden. The op->loader vocabulary must render for
every supported op without torch (that half runs in the merlin venv); the actual capture is skipped when
the m2m venv (torch) is absent, so the suite stays green on a bare checkout.

Target-agnostic: nothing here names a target; precision is a parameter (the token a target's
``compute_units`` declares)."""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_source as CSrc


def test_supported_ops_all_render():
    """Every op template renders to valid python source (no torch needed) with a Model + loader."""
    for op in CSrc.supported_ops():
        spec = {"op": op, "M": 16, "K": 16, "N": 16, "Dv": 16, "dtype": "fp32", "seed": 0}
        src = CSrc.build_loader_src(spec)
        assert "def get_model_and_inputs()" in src
        assert "class Model" in src or "def get_model_and_inputs" in src
        compile(src, f"<loader:{op}>", "exec")   # parseable python


def test_unknown_op_fails_closed():
    with pytest.raises(KeyError):
        CSrc.build_loader_src({"op": "not_a_real_op", "M": 16, "K": 16})


_M2M = CSrc.PytorchRefSource()
_needs_m2m = pytest.mark.skipif(not _M2M.available(),
                                reason="m2m venv (torch) unavailable; set MERLIN_M2M_PYTHON/MERLIN_M2M_DIR")


@_needs_m2m
@pytest.mark.parametrize("spec", [
    {"op": "matmul", "M": 16, "K": 16, "N": 16, "dtype": "fp32", "seed": 1},
    {"op": "rmsnorm", "M": 16, "K": 16, "dtype": "bf16", "seed": 2, "eps": 1e-5},
    {"op": "attention_full", "M": 16, "K": 64, "N": 16, "Dv": 64, "dtype": "fp16", "seed": 3, "causal": True},
])
def test_pytorch_capture_is_clean_with_golden(spec):
    """m2m produces a 0-opaque linalg program + a host-eager golden of the right shape, per dtype."""
    art = _M2M.capture(spec)
    assert art.meta["ok"] and art.meta["opaque"] == 0, art.meta
    assert "linalg." in art.linalg_mlir
    assert art.op == spec["op"] and art.dtype == spec["dtype"]
    # golden is a 2-D reference tensor matching the op's declared output rows
    assert isinstance(art.golden, list) and isinstance(art.golden[0], list)
    assert len(art.golden) == spec["M"]
    # the pytorch source is agent-visible context and must round-trip to the same op
    assert "get_model_and_inputs" in art.pytorch_src


def _float_binding(operand_dtype="f32"):
    from merlin.targetgen import corpus_spec as CS
    return CS.CorpusBinding(
        target="t", tile_dim=16, operand_dtype=operand_dtype, accum_dtype="f32", integer=False,
        tiers=["L0", "L1", "L3"], compare="tolerance_float", atol=0.03125, rtol=0.015625,
        classes_for=lambda **_: [])


@_needs_m2m
@pytest.mark.parametrize("entry,dtype", [
    ({"name": "PM0_matmul", "kind": "model_slice", "cat": "model_slices",
      "source_role": "handauthored_compiler_test", "source_reference": "pytorch matmul",
      "op": "matmul", "M": 16, "K": 16, "N": 16}, "f32"),
    ({"name": "PR0_rmsnorm", "kind": "model_slice", "cat": "model_slices",
      "source_role": "handauthored_compiler_test", "source_reference": "pytorch rmsnorm",
      "op": "rmsnorm", "M": 16, "K": 16}, "bf16"),
])
def test_write_pytorch_capsule_is_schema_valid(entry, dtype, tmp_path):
    """A PyTorch op is materialized into a complete, schema-valid capsule dir: the merlin_iface interface
    (what the agent compiles) plus the pytorch loader + linalg (visible grounding) plus a host-eager
    golden whose recorded input shapes match the interface, plus expected coverage."""
    import yaml
    from merlin.targetgen import capsule_common as CC

    d = CSrc.write_pytorch_capsule(entry, _float_binding(dtype), tmp_path)
    cap = CC.load_capsule(d)                       # raises on schema violation
    assert cap["source_role"] == "pytorch_model_slice"
    for f in ("capsule.interface.mlir", "capsule.pytorch.py", "capsule.linalg.mlir", "golden.yaml"):
        assert (d / f).exists(), f
    g = yaml.safe_load((d / "golden.yaml").read_text())
    assert g["golden_source"] == "host_torch_eager"
    prov = g["oracle_provenance"]["inputs"]
    # every declared capsule input has a recorded golden operand of the same shape
    for inp in cap["inputs"]:
        assert inp["name"] in prov and prov[inp["name"]]["shape"] == inp["shape"]
    out = g["outputs"][entry.get("out", "Y0")]
    assert isinstance(out, list) and isinstance(out[0], list)


def test_write_pytorch_capsule_rejects_unmapped_op(tmp_path):
    """An op with no merlin_iface builder must fail closed here (it belongs to the direct-MLIR path)."""
    entry = {"name": "PX", "kind": "model_slice", "cat": "model_slices",
             "source_role": "handauthored_compiler_test", "source_reference": "x", "op": "softmax",
             "M": 16, "K": 16}
    with pytest.raises(ValueError):
        CSrc.write_pytorch_capsule(entry, _float_binding(), tmp_path)


def _load_generate_corpus():
    import importlib.util
    from merlin.common.paths import repo_root
    p = repo_root() / "merlin" / "contract" / "capsules" / "generate_corpus.py"
    spec = importlib.util.spec_from_file_location("generate_corpus_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_corpus_pytorch_route_fails_closed_on_int(tmp_path):
    """The generator's ``source: pytorch`` branch is float-only; an int dtype must fail closed (no m2m
    needed — it raises before any capture)."""
    import dataclasses
    GC = _load_generate_corpus()
    ib = dataclasses.replace(_float_binding(), operand_dtype="int8", accum_dtype="i32",
                             integer=True, compare="exact_int")
    entry = {"name": "WBAD", "kind": "isa", "cat": "isa", "source": "pytorch",
             "source_role": "handauthored_compiler_test", "source_reference": "x", "op": "matmul",
             "M": 16, "K": 16, "N": 16, "modes": {}}
    with pytest.raises(ValueError):
        GC._write_capsule(entry, ib, tmp_path)


def test_generate_corpus_pytorch_skips_when_m2m_absent(tmp_path, monkeypatch):
    """A pytorch entry is additive: when the m2m venv is absent it is skipped (returns None), not fatal —
    so a checkout without torch still regenerates the direct-MLIR corpus."""
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(tmp_path / "no_such_python"))
    GC = _load_generate_corpus()
    entry = {"name": "WSKIP", "kind": "model_slice", "cat": "model_slices", "source": "pytorch",
             "source_role": "pytorch_model_slice", "source_reference": "x", "op": "matmul",
             "M": 16, "K": 16, "N": 16}
    assert GC._write_capsule(entry, _float_binding(), tmp_path) is None


@_needs_m2m
def test_generate_corpus_routes_pytorch_source(tmp_path):
    """A profile entry with ``source: pytorch`` on a float dtype routes to the host-eager PyTorch source."""
    import yaml
    GC = _load_generate_corpus()
    entry = {"name": "WM0_matmul_f32", "kind": "model_slice", "cat": "model_slices", "source": "pytorch",
             "source_role": "handauthored_compiler_test", "source_reference": "pytorch matmul",
             "op": "matmul", "M": 16, "K": 16, "N": 16}
    d = GC._write_capsule(entry, _float_binding(), tmp_path)
    g = yaml.safe_load((d / "golden.yaml").read_text())
    assert g["golden_source"] == "host_torch_eager"
    assert (d / "capsule.linalg.mlir").exists()
