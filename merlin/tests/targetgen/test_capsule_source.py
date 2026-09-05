"""The grounded capsule sources: a capsule defined in PyTorch must lower to 0-opaque linalg-on-tensors
via model2MLIR and carry a host torch-eager reference golden. The op->loader vocabulary must render for
every supported op without torch (that half runs in the merlin venv); the actual capture is skipped when
the m2m venv (torch) is absent, so the suite stays green on a bare checkout.

Target-agnostic: nothing here names a target; precision is a parameter (the token a target's
``compute_units`` declares)."""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_source as CSrc


def test_explicit_model_loader_is_repository_relative(monkeypatch, tmp_path):
    repo = tmp_path / "merlin-repo"
    loader = repo / "merlin/contract/capsules/model/M2/capsule.pytorch.py"
    loader.parent.mkdir(parents=True)
    loader.write_text("def get_model_and_inputs(): pass\n", encoding="utf-8")
    monkeypatch.setattr(CSrc, "repo_root", lambda: repo)

    resolved = CSrc.resolve_model_loader(
        {"loader": "merlin/contract/capsules/model/M2/capsule.pytorch.py"},
        m2m_dir=tmp_path / "model2MLIR",
    )

    assert resolved == loader


def test_implicit_model_loader_remains_model2mlir_relative(tmp_path):
    assert CSrc.resolve_model_loader(
        {"model": "small_llama"}, m2m_dir=tmp_path / "model2MLIR"
    ) == tmp_path / "model2MLIR/workloads/small_llama/loader.py"


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


# --- Phase 2: derive-and-verify the mapped-op interface FROM the captured linalg (structural, no regex) ---
_MATMUL_ENTRY = {"name": "X_matmul", "kind": "isa", "cat": "isa", "op": "matmul",
                 "M": 16, "K": 16, "N": 16, "lhs": "A0", "weight": "W", "out": "Y0",
                 "source_role": "handauthored_compiler_test", "source_reference": "t"}
_MATMUL_LINALG = (
    "builtin.module {\n"
    "  func.func @forward(%0: tensor<16x16xf32>, %1: tensor<16x16xf32>) -> tensor<16x16xf32> {\n"
    '    %2 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
    "ins(%0, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>\n"
    "    return %2 : tensor<16x16xf32>\n  }\n}\n")


def test_linalg_summary_is_structural():
    """The structural reader recovers prov tags + @forward operand/result tensor types (no regex)."""
    s = CSrc.linalg_summary(_MATMUL_LINALG)
    assert s["prov_ops"] == ["matmul"] and s["prov_families"] == ["contraction"]
    assert s["inputs"] == [([16, 16], "f32"), ([16, 16], "f32")]
    assert s["output"] == ([16, 16], "f32")


def test_linalg_to_iface_matches_handwritten_builder():
    """A matching linalg yields the SAME interface the merlin_iface builder emits (byte-identical to the
    handwritten form) — the interface is verified against the lowering, not merely assumed."""
    from merlin.targetgen import corpus_spec as CS
    b = _float_binding()
    _, want = CS.build(_MATMUL_ENTRY, b)
    _, got = CSrc.linalg_to_iface(_MATMUL_LINALG, _MATMUL_ENTRY, b)
    assert got == want


def test_linalg_to_iface_fails_closed_on_missing_op():
    """A lowering that does NOT contain the claimed op family is rejected (never emit an unsupported iface)."""
    bad = _MATMUL_LINALG.replace('prov.family = "contraction"', 'prov.family = "normalization"') \
                        .replace('prov.op = "matmul"', 'prov.op = "rmsnorm"')
    with pytest.raises(CSrc.M2MUnavailable):
        CSrc.linalg_to_iface(bad, _MATMUL_ENTRY, _float_binding())


def test_linalg_to_iface_fails_closed_on_shape_mismatch():
    """Right op family, wrong operand shapes -> fail closed (interface/lowering shape mismatch)."""
    bad = _MATMUL_LINALG.replace("tensor<16x16xf32>", "tensor<8x8xf32>")
    with pytest.raises(CSrc.M2MUnavailable):
        CSrc.linalg_to_iface(bad, _MATMUL_ENTRY, _float_binding())


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


def test_write_pytorch_capsule_rejects_unknown_op(tmp_path):
    """An op with neither a merlin_iface builder nor a fused template must fail closed."""
    entry = {"name": "PX", "kind": "model_slice", "cat": "model_slices",
             "source_role": "handauthored_compiler_test", "source_reference": "x", "op": "not_an_op",
             "M": 16, "K": 16}
    with pytest.raises(ValueError):
        CSrc.write_pytorch_capsule(entry, _float_binding(), tmp_path)


@_needs_m2m
@pytest.mark.parametrize("entry", [
    {"name": "FS0_softmax", "kind": "model_slice", "cat": "model_slices",
     "source_role": "pytorch_model_slice", "source_reference": "softmax", "op": "softmax", "M": 16, "K": 16},
    {"name": "FL0_layernorm", "kind": "model_slice", "cat": "model_slices",
     "source_role": "pytorch_model_slice", "source_reference": "layernorm", "op": "layernorm",
     "M": 16, "K": 16},
    {"name": "FA0_attn_full", "kind": "model_slice", "cat": "model_slices",
     "source_role": "pytorch_model_slice", "source_reference": "attn", "op": "attention_full",
     "M": 16, "K": 64, "N": 16, "Dv": 64, "causal": True},
])
def test_write_fused_pytorch_capsule_linalg_interface(entry, tmp_path):
    """A FUSED op (softmax/layernorm/geglu/rope/attention_full) ships the linalg module AS the interface
    (positional), schema-valid, with a host-eager golden and an arg_order the harness can feed."""
    import yaml
    from merlin.targetgen import capsule_common as CC
    d = CSrc.write_pytorch_capsule(entry, _float_binding(), tmp_path)
    cap = CC.load_capsule(d)
    assert cap["operation"]["op"] == entry["op"]
    iface = (d / "capsule.interface.mlir").read_text()
    assert "func.func" in iface and "linalg." in iface       # the interface IS standard-dialect linalg
    g = yaml.safe_load((d / "golden.yaml").read_text())
    assert g["oracle_provenance"]["interface"] == "linalg_positional"
    order = g["oracle_provenance"]["arg_order"]
    assert order[-1] == entry.get("out", "Y0") and len(order) >= 2


def test_args_from_cb_linalg_positional():
    """The positional-input harness path feeds a linalg-interface capsule's inputs in arg_order (incl.
    a rank-1 operand), producing the output last — no merlin_iface commands needed."""
    from merlin.runtime.backends.base import get_backend
    MH = get_backend("muon").muon_harness
    cb = {
        "target": "t", "interface": "linalg_positional", "arg_order": ["X", "W", "B", "Y0"],
        "tensors": {"X": {"role": "input", "shape": [4, 4]}, "W": {"role": "input", "shape": [4]},
                    "B": {"role": "input", "shape": [4]}, "Y0": {"role": "output", "shape": [4, 4]}},
        "canonical_inputs": {"X": {"values": [float(i) for i in range(16)]},
                             "W": {"values": [1.0, 1.0, 1.0, 1.0]}, "B": {"values": [0.0, 0.0, 0.0, 0.0]}},
    }
    in_args, out_args = MH.args_from_cb(cb)
    assert [a.name for a in in_args] == ["X", "W", "B"]
    assert (in_args[1].rows, in_args[1].cols) == (1, 4)      # rank-1 flattened
    assert len(in_args[0].values) == 16
    assert out_args[0].name == "Y0" and (out_args[0].rows, out_args[0].cols) == (4, 4)


def test_model_gate_satisfied():
    """The whole-model capsule's schedule gate: unlocked at/above the pass fraction, locked below, and a
    capsule with no gate is always schedulable."""
    cap = {"gate": {"after_op_pass_fraction": 0.8}}
    assert CSrc.model_gate_satisfied(cap, 0.8) and CSrc.model_gate_satisfied(cap, 0.95)
    assert not CSrc.model_gate_satisfied(cap, 0.5)
    assert CSrc.model_gate_satisfied({}, 0.0)


@_needs_m2m
def test_write_model_capsule_small_llama(tmp_path):
    """A whole-model capsule lowers small_llama end-to-end (0-opaque), externalizes weights alongside the
    linalg interface, records a host-eager golden, and carries the schedule gate. Schema-valid kind=model."""
    import yaml
    from merlin.targetgen import capsule_common as CC
    entry = {"name": "M0_small_llama_fp32", "cat": "model", "model": "small_llama", "operand_dtype": "f32",
             "source_reference": "tiny full LLaMA", "label": "public", "gate": {"after_op_pass_fraction": 0.8}}
    d = CSrc.write_model_capsule(entry, _float_binding(), tmp_path)
    cap = CC.load_capsule(d)
    assert cap["kind"] == "model" and cap["operation"]["op"] == "model"
    assert cap["gate"]["after_op_pass_fraction"] == 0.8
    assert (d / "capsule.weights.safetensors").exists()
    iface = (d / "capsule.interface.mlir").read_text()
    assert "linalg." in iface and "capsule.weights.safetensors" in iface   # weights path made relative
    g = yaml.safe_load((d / "golden.yaml").read_text())
    out = list(g["outputs"].values())[0]
    assert isinstance(out, list) and isinstance(out[0], list) and isinstance(out[0][0], list)  # rank-3 logits


@_needs_m2m
@pytest.mark.parametrize("entry,dtype,op", [
    ({"name": "XM_matmul", "kind": "isa", "cat": "isa", "source_role": "pytorch_model_slice",
      "source_reference": "m", "op": "matmul", "M": 16, "K": 16, "N": 16}, "f32", "matmul"),
    ({"name": "XA_attn", "kind": "model_slice", "cat": "model_slices", "source_role": "pytorch_model_slice",
      "source_reference": "a", "op": "attention_qk", "M": 16, "K": 16, "N": 16}, "f32", "attention_qk"),
    ({"name": "XR_rms", "kind": "model_slice", "cat": "model_slices", "source_role": "pytorch_model_slice",
      "source_reference": "r", "op": "rmsnorm", "M": 16, "K": 16, "eps": 1e-5}, "f32", "rmsnorm"),
])
def test_pytorch_golden_matches_reference_math(entry, dtype, op, tmp_path):
    """Regression vs the trusted reference math the original hand-written MLIR capsules encode: the host
    torch-eager golden must equal a plain numpy reference recomputed on the capsule's OWN recorded inputs.
    Proves the PyTorch source did not change the operation's semantics."""
    import numpy as np
    import yaml
    d = CSrc.write_pytorch_capsule(entry, _float_binding(dtype), tmp_path)
    g = yaml.safe_load((d / "golden.yaml").read_text())
    prov = g["oracle_provenance"]["inputs"]

    def arr(name):
        return np.array(prov[name]["decoded"], dtype=np.float64).reshape(prov[name]["shape"])

    got = np.array(list(g["outputs"].values())[0], dtype=np.float64)
    if op == "matmul":
        ref = arr("A0").astype(np.float32) @ arr("W").astype(np.float32)
    elif op == "attention_qk":
        ref = arr("Q").astype(np.float32) @ arr("K").astype(np.float32).T
    else:  # rmsnorm
        x, gm = arr("X").astype(np.float32), arr("G").astype(np.float32).reshape(-1)
        ms = (x * x).mean(-1, keepdims=True)
        ref = x * (1.0 / np.sqrt(ms + np.float32(entry["eps"]))) * gm
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


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

def test_extent_scan_matches_the_op_name_exactly():
    """`linalg.matmul_transpose_b` also starts with "linalg.matmul". Accepting it recorded the operand
    types of a TRANSPOSED pair as a plain matmul's (M, K, N) -- and because model_op_demands joins
    extents positionally, advancing only on prov.op == "matmul", one spurious entry mis-shaped every
    later layer of the model."""
    from merlin.targetgen.capsule_source import _matmul_extents

    plain = ("linalg.matmul ins(%a, %b : tensor<8x16xf32>, tensor<16x32xf32>) "
             "outs(%c : tensor<8x32xf32>)")
    tb = ("linalg.matmul_transpose_b ins(%a, %b : tensor<8x16xf32>, tensor<32x16xf32>) "
          "outs(%c : tensor<8x32xf32>)")
    assert _matmul_extents(plain) == [(8, 16, 32)]
    assert _matmul_extents(tb) == [], "a transpose-b matmul must not masquerade as a plain one"
    assert _matmul_extents(plain + "\n" + tb) == [(8, 16, 32)]
