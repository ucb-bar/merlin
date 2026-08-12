"""Regression tests for the pre-spend validation de-overfit fixes.

Each test pins a target-agnosticism / robustness fix found by replicating the agent grade loop offline
before spending on a real run. They are pure-Python (no sim, no LLM) so they run in any checkout.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root


# --- capsule_dram: MX dtype byte widths are a single source (no drift), never a KeyError ---------------
def test_capsule_dram_knows_mx_dtypes():
    from merlin.targetgen import capsule_dram as CD
    # mxfp6/mxfp4 were absent and crashed the runner on the MX-tile capsules; now 1 byte/element like mxfp8.
    assert CD.dtype_bytes("mxfp8") == 1
    assert CD.dtype_bytes("mxfp6") == 1
    assert CD.dtype_bytes("mxfp4") == 1
    # the MLIR spellings resolve too
    assert CD.dtype_bytes("f6E3M2FN") == 1 and CD.dtype_bytes("f4E2M1FN") == 1


def test_capsule_dram_delegates_to_corpus_spec_on_miss():
    """A dtype the operand-gen authority (corpus_spec._DTYPE) knows is never a DRAM KeyError even if the
    local table lacks it — one source of dtype widths, not two that can drift."""
    from merlin.targetgen import capsule_dram as CD
    from merlin.targetgen.corpus_spec import dtype_info
    assert CD.dtype_bytes("fp6_e3m2") == dtype_info("fp6_e3m2")[2]


# --- interface_emit: MX float tensor types parse structurally (no narrow-regex silent drop) ------------
@pytest.mark.parametrize("ttype,dims,dt", [
    ("tensor<16x32xf32>", [16, 32], "f32"),
    ("tensor<16x32xbf16>", [16, 32], "bf16"),
    ("tensor<32x16xf8E4M3FN>", [32, 16], "f8E4M3FN"),
    ("tensor<32x32xf6E3M2FN>", [32, 32], "f6E3M2FN"),
    ("tensor<32x32xf4E2M1FN>", [32, 32], "f4E2M1FN"),
    ("tensor<8x8xi8>", [8, 8], "i8"),
])
def test_interface_emit_parses_mx_tensor_types(ttype, dims, dt):
    from merlin.targetgen.contract.interface_emit import _shape_dtype
    assert _shape_dtype(ttype) == (dims, dt)


# --- gen_numeric_facts: FAIL CLOSED (no baked gemmini i8/i32/32) + no regex in generated code ----------
def test_gen_numeric_facts_fails_closed_without_datapaths():
    from merlin.targetgen.rtl import gen_numeric_facts as G
    code = G.generate({"facts": {"datapaths": [], "memories": []}})
    assert "INPUT_DTYPE = None" in code and "ACC_DTYPE = None" in code and "ACC_WIDTH_BITS = None" in code
    header = code.split("def check_numeric_shapes")[0]
    assert "'i8'" not in header and "'i32'" not in header, "baked gemmini datapath default leaked"
    assert "import re" not in code and "re.search" not in code, "regex leaked into generated code"
    ns: dict = {}
    exec(code, ns)  # generated module must be valid and skip the width check when the width is unknown
    assert ns["check_numeric_shapes"](
        {"tensors": {"acc": {"dtype": "i8"}}, "commands": [{"opcode": "MATMUL", "operands": {"dst": "acc"}}]}
    ) == []


def test_arm4_generators_have_no_baked_target_name():
    """The arm-4 agent-facing generators emit into ANY target's module — they must not bake one target's
    name into their operative output (a second RoCC target was mislabeled 'Gemmini')."""
    import inspect
    from merlin.targetgen.rtl import gen_isa_module, gen_rtl_digest
    assert "Gemmini" not in gen_isa_module._HEADER and "gemmini" not in gen_isa_module._HEADER
    assert "Gemmini accelerator" not in inspect.getsource(gen_rtl_digest.generate)


def test_gen_numeric_facts_derives_from_facts():
    from merlin.targetgen.rtl import gen_numeric_facts as G
    code = G.generate({"facts": {"datapaths": [{"name": "input", "dtype": "i8"},
                                               {"name": "accumulator", "dtype": "i32"}],
                                 "memories": [{"name": "accumulator", "lane_bits": 32}]}})
    assert "ACC_WIDTH_BITS = 32" in code
    ns: dict = {}
    exec(code, ns)
    # a narrow i8 accumulator vs the derived 32b width is flagged; _bits is structural (no regex)
    assert len(ns["check_numeric_shapes"](
        {"tensors": {"acc": {"dtype": "i8"}}, "commands": [{"opcode": "MATMUL", "operands": {"dst": "acc"}}]})) == 1
    assert ns["_bits"]("bf16") == 16 and ns["_bits"]("f8E4M3FN") == 8


# --- capsule_golden: nested (specir 2D) decoded operands flatten to the documented flat list ----------
def test_capsule_golden_flatten_row_major():
    from merlin.targetgen.capsule_golden import _flatten_row_major
    assert _flatten_row_major([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert _flatten_row_major([1, 2, 3]) == [1, 2, 3]      # already-flat unchanged
    assert _flatten_row_major(5) == [5]                     # scalar -> singleton


# --- capsule_runner: force_match_policy merges with (never discards) the capsule's declared tolerance --
def test_merge_match_policy_takes_looser_tolerance():
    from merlin.targetgen.capsule_runner import _merge_match_policy
    force = {"compare": "float", "atol": 0.001}
    cap = {"compare": "float", "atol": 0.03125, "rtol": 0.015625}
    merged = _merge_match_policy(force, cap)
    # compare mode from the force policy; atol/rtol are the LOOSER (max) so a bf16 capsule's tolerance
    # (RP3) is never discarded by the tight global default
    assert merged["compare"] == "float"
    assert merged["atol"] == 0.03125 and merged["rtol"] == 0.015625
    # a capsule with no policy keeps the force policy; a force-less target keeps the capsule policy
    assert _merge_match_policy(force, None) == force
    assert _merge_match_policy(None, cap) == cap


# --- answer_surfaces: EVERY hidden-capsule dir is masked, not just the target's own -------------------
def test_answer_surfaces_masks_all_hidden_dirs():
    """The shared merlin/contract/capsules/hidden and other targets' <t>/hidden are answer surfaces too
    (the bundle grants merlin/contract broadly). A run must mask all of them, not only te.hidden_corpus()."""
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces
    desc = repo_root() / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"
    if not desc.is_file():
        pytest.skip("radiance descriptor not present in this checkout")
    te = load_target_experiment(desc)
    masked = {str(s.path) for s in answer_surfaces(te) if s.origin == "hidden"}
    caps = repo_root() / "merlin/contract/capsules"
    for hd in caps.rglob("hidden"):
        if hd.is_dir():
            assert str(hd) in masked, f"hidden dir not masked: {hd}"
