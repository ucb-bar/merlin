"""Phase 2b — the shared task prompt template. render_prompt composes ONE template + the derived slots,
so a target's prompt is generated, never hand-authored. The invariant: for a fixed (experiment, arm),
two targets' prompts differ ONLY in the derived slots — the shared skeleton is byte-identical.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.target_experiment import load_target_experiment, load_capability_manifest
from merlin.targetgen.generate_prompt import render_prompt, prompt_slots

_GEM = "merlin/experiments/gemmini_capsule_bench_v0/target_experiment.yaml"
_RAD = "merlin/experiments/radiance_capsule_bench_v0/target_experiment.yaml"

_SHARED_BLOCKS = [
    "non-exempt out-of-tree MLIR target backend",
    "never author a compute kernel",                       # compiler-not-kernel
    "Compute must be compiler-GENERATED, never an authored/library kernel",
    "integrity_exempt: false",
    "qa/verdict.json",
    "Final status line",
    "parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact",
]


def _render(target, desc, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("MERLIN_TARGET_PATH", "out/artifacts/targets")
    te = load_target_experiment(desc)
    m = load_capability_manifest(target)
    return render_prompt(te, m, "full", "raw_baseline"), prompt_slots(te, m)


def test_gemmini_prompt_has_shared_blocks_and_its_slots():
    p, s = _render("gemmini", _GEM)
    for b in _SHARED_BLOCKS:
        assert b in p, f"missing shared block: {b}"
    assert "gemmini-opt" in p and "`gemmini_kernel`" in p and "--convert-iface-to-gemmini" in p
    assert "Target ISA facts: gemmini" in p


def test_radiance_prompt_has_radiance_slots_and_no_gemmini_leakage(monkeypatch):
    try:
        p, s = _render("radiance", _RAD, monkeypatch)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"radiance not resolvable: {e}")
    for b in _SHARED_BLOCKS:
        assert b in p
    assert "radiance-opt" in p and "`radiance_kernel`" in p
    assert "gemmini" not in p.lower() and "rocc" not in p.lower() and "0x7b" not in p


_ATLAS = "merlin/experiments/atlas_capsule_bench_v0/target_experiment.yaml"


def test_grading_model_is_derived_from_the_corpus_not_hardcoded_integer():
    """The certification-model sentence must follow the corpus goldens: gemmini's integer corpus grades
    exact-integer 3-way; atlas's independent-float (fp8/bf16) corpus grades within a tolerance against the
    program-oracle and marks the integer self-consistency cross-checks not_applicable. Telling a float-MXU
    agent the grading is 'exact-integer, no tolerance' would make it build the wrong backend."""
    _, gs = _render("gemmini", _GEM)
    assert "exact-integer" in gs["grading_model"] and "no tolerance" in gs["grading_model"]
    try:
        _, as_ = _render("atlas", _ATLAS)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"atlas not resolvable: {e}")
    assert "tolerance" in as_["grading_model"] and "not_applicable" in as_["grading_model"]
    assert "exact-integer" not in as_["grading_model"]


def _canonicalize(p: str, s: dict) -> str:
    # replace target-specific slot VALUES with fixed tokens, and drop the derived per-target blocks
    # (the ISA-facts brief + the corpus-family bullets) — what remains is the shared skeleton.
    for val, tok in ((s["kernel_symbol"], "KSYM"), (s["tool_stem"], "TOOL"), (s["target"], "T")):
        p = p.replace(val, tok)
    p = p.replace(s["endpoint_desc"], "ENDPOINT")
    p = p.replace(s["grading_model"], "GRADING")   # float-vs-integer grading model is a derived slot
    out, in_facts = [], False
    for ln in p.splitlines():
        if ln.startswith("## Target ISA facts"):
            in_facts = True
        if ln.startswith("## Final status line"):
            in_facts = False
        if in_facts or ln.startswith("- `"):
            continue
        out.append(ln)
    return "\n".join(out)


def test_cross_target_sameness_shared_skeleton_is_identical(monkeypatch):
    gp, gs = _render("gemmini", _GEM)
    try:
        rp, rs = _render("radiance", _RAD, monkeypatch)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"radiance not resolvable: {e}")
    # for a fixed (experiment, arm), the prompts differ ONLY in the derived slots
    assert _canonicalize(gp, gs) == _canonicalize(rp, rs)
