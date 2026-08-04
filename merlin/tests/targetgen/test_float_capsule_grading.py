"""Datatype-general capsule grading: the FLOAT (fp8/bf16) atlas path grades end-to-end via an
INDEPENDENT golden while the INTEGER (gemmini) path stays byte-identical.

The golden is the independent oracle's answer. For an integer capsule (gemmini / exact_int /
golden_source merlin_tensor_int) it is RECOMPUTED on the integer Tensor engine. For a float capsule
that ships its own golden.yaml (atlas fp8-e4m3 -> bf16, golden_source specir_refmodel_fp8_bf16) the
integer engine cannot reproduce the float datapath, so the golden is READ from golden.yaml and the
integer L0-reference / L1-simulate tiers are skipped as inapplicable — the grade rides the RTL
program-oracle output vs the independent golden (tolerance_float). ``not_run_is_not_pass`` is preserved:
a required RTL oracle that is unavailable still makes the capsule ``incomplete``.
"""
from __future__ import annotations

import copy

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.contract import schemas
from merlin.targetgen.runner_config import RunnerConfig

CAPS = repo_root() / "merlin/contract/capsules"
ATLAS_AT2 = CAPS / "atlas/isa/AT2_single_tile_matmul"          # fp8-e4m3 in / bf16 out, independent golden
GEMMINI_A2 = CAPS / "isa/A2_single_tile_matmul"                # i8 x i8 -> i32, recomputed golden


def _atlas_config() -> RunnerConfig:
    """An atlas-shaped grading config: the external_backend program oracle is the RTL tier (L3), no RoCC
    trace gate, output equality is tolerance_float. Mirrors what the atlas harness threads into run_capsule."""
    return RunnerConfig(
        target="atlas", suite="atlas-capsule-bench", dtype="fp8_e4m3",
        fourth_output_name="kernel.S", tier_sim={"L3": "atlas-arc"},
        rtl_tiers=frozenset({"L3"}), oracle_tiers=("L3",), perf_fields=(), trace_gate=None)


# --------------------------------------------------------------------------------------------
# golden source resolution
# --------------------------------------------------------------------------------------------
def test_integer_capsule_uses_recompute_path():
    cap = load_capsule(GEMMINI_A2, contract="merlin/contract")
    assert CG.is_independent_float_golden(cap) is False
    assert CG.golden_source(cap) == "merlin_tensor_int"
    # golden() dispatches to the integer Tensor engine — byte-identical to a direct recompute.
    assert CG.golden(cap) == CG._recompute_golden(cap)
    # integer values (i32 accumulation), not floats.
    y = next(iter(CG.golden(cap).values()))
    flat = y[0] if isinstance(y[0], list) else y
    assert all(isinstance(v, int) for v in flat)


def test_float_capsule_reads_independent_golden():
    cap = load_capsule(ATLAS_AT2, contract="merlin/contract")
    assert CG.is_independent_float_golden(cap) is True
    assert CG.golden_source(cap) == "specir_refmodel_fp8_bf16"
    g = CG.golden(cap)
    y = g["Y0"]
    flat = [v for row in y for v in row]
    # The bf16 golden is READ as floats (an integer engine would yield whole numbers) and is a NON-degenerate
    # reference (many distinct, genuinely-fractional values). We assert those invariants — not exact values:
    # golden.yaml is an untracked, regenerable answer key, so pinning literals goes stale on every recapture.
    assert isinstance(y[0][0], float)
    assert any(v != int(v) for v in flat), "bf16 refmodel golden should be genuinely fractional, not int-valued"
    assert len(set(flat)) > 1, "golden must be non-degenerate (not a constant), else addressing bugs are invisible"


# --------------------------------------------------------------------------------------------
# tolerance_float comparator + honest golden_source
# --------------------------------------------------------------------------------------------
def test_compare_tolerance_float_and_source_reported():
    cap = load_capsule(ATLAS_AT2, contract="merlin/contract")
    pol = cap["numeric_policy"]                                 # tolerance_float, atol 0.25, rtol 0.02
    g = CG.golden(cap)
    src = CG.golden_source(cap)

    ok = CG.compare(g, copy.deepcopy(g), pol, golden_source=src)
    assert ok["status"] == "pass" and ok["golden_source"] == "specir_refmodel_fp8_bf16"

    within = copy.deepcopy(g)
    within["Y0"][0][0] += 0.2                                   # inside atol 0.25 -> still pass
    assert CG.compare(g, within, pol, golden_source=src)["status"] == "pass"

    bad = copy.deepcopy(g)
    bad["Y0"][0][0] += 100.0                                    # far outside tolerance -> fail
    rep = CG.compare(g, bad, pol, golden_source=src)
    assert rep["status"] == "fail" and rep["max_abs_error"] == 100.0


# --------------------------------------------------------------------------------------------
# run_capsule end-to-end (front half + integer engines stubbed; oracle injected via the sanctioned seam)
# --------------------------------------------------------------------------------------------
def _stub_front_half(monkeypatch):
    """Bypass the package build + 4 ABI entrypoints — return a minimal (pkg, cb, kernel_text)."""
    cb = {"tensors": {"Y0": {"role": "output", "base": 0, "shape": [32, 32], "dtype": "bf16"}}}
    monkeypatch.setattr(CR, "run_entrypoints",
                        lambda *a, **k: (object(), cb, "# kernel.S (stub)\n"))
    return cb


def _oracle_returning(outputs):
    def run(cb, llvm_text, workdir, timeout):
        return {"outputs": copy.deepcopy(outputs), "cycles": 123, "oracle": "atlas-arc-test"}
    return {"L3": run}


def test_float_run_capsule_grades_pass(tmp_path, monkeypatch):
    _stub_front_half(monkeypatch)
    cap = load_capsule(ATLAS_AT2, contract="merlin/contract")
    gold = CG.golden(cap)

    res = CR.run_capsule(cap, "unused-package", runs_root=tmp_path, run_id="AT2_pass",
                         config=_atlas_config(), oracle_adapters=_oracle_returning(gold))

    assert res["status"] == "pass", res.get("failure")
    # integer floor is honestly skipped (N/A for float), not failed.
    assert res["tiers"]["L0"]["status"] == "skipped"
    assert res["tiers"]["L0"]["not_applicable"] is True
    assert res["tiers"]["L1"]["status"] == "skipped"
    # the grade rode the RTL oracle vs the independent golden, tolerance_float.
    assert res["tiers"]["L3"]["status"] == "pass"
    assert res["numeric"]["status"] == "pass"
    assert res["numeric"]["policy"] == "tolerance_float"
    assert res["numeric"]["golden_source"] == "specir_refmodel_fp8_bf16"
    schemas.validate(res, "capsule_result", contract="merlin/contract")


def test_float_run_capsule_grades_fail_on_mismatch(tmp_path, monkeypatch):
    _stub_front_half(monkeypatch)
    cap = load_capsule(ATLAS_AT2, contract="merlin/contract")
    bad = CG.golden(cap)
    bad = copy.deepcopy(bad)
    bad["Y0"][0][0] += 100.0                                    # outside tolerance

    res = CR.run_capsule(cap, "unused-package", runs_root=tmp_path, run_id="AT2_fail",
                         config=_atlas_config(), oracle_adapters=_oracle_returning(bad))

    assert res["status"] == "fail"
    assert res["tiers"]["L3"]["status"] == "fail"
    assert res["numeric"]["status"] == "fail"
    assert res["numeric"]["golden_source"] == "specir_refmodel_fp8_bf16"


def test_float_run_capsule_not_run_is_not_pass(tmp_path, monkeypatch):
    """A required RTL oracle that is absent -> incomplete, never pass — even though the integer L0/L1
    floor is legitimately skipped for the float datapath."""
    _stub_front_half(monkeypatch)
    cap = load_capsule(ATLAS_AT2, contract="merlin/contract")

    res = CR.run_capsule(cap, "unused-package", runs_root=tmp_path, run_id="AT2_incomplete",
                         config=_atlas_config(), oracle_adapters={})     # no L3 adapter

    assert res["status"] == "incomplete"
    assert res["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert res["tiers"]["L3"]["status"] == "unavailable"
    # the N/A integer skips did NOT trip not_run_is_not_pass; the missing RTL oracle did.
    assert res["tiers"]["L0"]["not_applicable"] is True


def test_no_oracle_smoke_is_not_gradeable_never_pass(tmp_path, monkeypatch):
    """--no-oracle STRUCTURE-ONLY smoke: the SAME missing numeric tier that makes a GRADED run
    `incomplete`/`oracle_unavailable` instead reads back as the DISTINCT `not_gradeable_no_oracle`
    (a withheld numeric verdict) — never a numeric pass. This is the honest no-oracle plane; the
    not_run_is_not_pass gate stays fully in force for graded runs (asserted separately)."""
    _stub_front_half(monkeypatch)
    cap = load_capsule(ATLAS_AT2, contract="merlin/contract")

    res = CR.run_capsule(cap, "unused-package", runs_root=tmp_path, run_id="AT2_no_oracle",
                         config=_atlas_config(), oracle_adapters={}, no_oracle=True)

    assert res["status"] == "not_gradeable_no_oracle"
    assert res["status"] != "pass"                              # HARD INVARIANT: never a numeric pass
    assert res["failure"]["plane"] == "not_gradeable_no_oracle"
    assert res["failure"]["category"] == "NOT_GRADEABLE_NO_ORACLE"
    assert res["tiers"]["L3"]["status"] == "unavailable"
    schemas.validate(res, "capsule_result", contract="merlin/contract")


def test_atlas_oracle_routes_to_program_oracle():
    ad = CR.oracle_adapters("atlas")
    assert {"L2", "L3"} <= set(ad)                              # fast functional loop + cycle-exact cosim
    assert ad["L2"].__module__ == "merlin.targetgen.program_oracle"
    assert ad["L3"].__module__ == "merlin.targetgen.program_oracle"
    if "L4" in ad:                                              # additive RTL-certified verilator tier
        assert ad["L4"].__module__ == "merlin.targetgen.program_oracle"
    assert "program_functional_adapter" in ad["L2"].__qualname__   # fast per-round tier
    assert "program_oracle_adapter" in ad["L3"].__qualname__       # gold checkpoint tier


def test_external_backend_requires_model_ext_no_target_default(monkeypatch):
    """oracle_adapters no longer defaults an external_backend's model_ext to a target literal
    ('npu_model'): a contract that declares none FAILS CLOSED with an actionable error, rather than
    silently binding one target's model project. The cosim backend itself is DERIVED by target from mlc
    (mlc.discover.cosim_backend), so no cosim-module literal lives in merlin either."""
    monkeypatch.setattr(CR, "_endpoint_of", lambda t: ("external_backend", None))
    with pytest.raises(ValueError, match="model_ext"):
        CR.oracle_adapters("fake_external_target")
