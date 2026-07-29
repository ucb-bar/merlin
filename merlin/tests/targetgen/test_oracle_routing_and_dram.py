"""WS-A: the capsule grader routes to the TARGET'S OWN oracle (never a silent gemmini default), and the
external_backend DRAM address contract is harness-owned + deterministic.

These lock the two harness bugs that made an atlas run score 0/11 without ever grading the agent:
  * an unrouted grade (oracle_adapters=None) fell back to the gemmini spike/verilator MLIR-lowering
    oracle and crashed on the atlas dialect (run_lowering.py / ir.Module.parse);
  * the emitted command buffer's output tensor carried no DRAM base, so the program oracle crashed
    ('L3 crash: base') before comparing numerics.
"""
from __future__ import annotations

import yaml
import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import capsule_dram as D
from merlin.common.paths import merlin_dir


# ---- FAULT 1: oracle routing is contract-derived, never a silent gemmini fallback ----------------

def test_bare_oracle_adapters_self_routes_from_the_contract():
    # atlas (external_backend) -> the program oracle at L3 (NOT verilator); gemmini -> spike/verilator.
    atlas = CR.oracle_adapters("atlas")
    assert set(atlas) == {"L3"}
    assert "program_oracle" in getattr(atlas["L3"], "__qualname__", "")
    gem = CR.oracle_adapters("gemmini")
    assert set(gem) == {"L2", "L3"}                      # chipyard spike/verilator, recovered from tier_sim


def test_explicit_sim_via_is_honored_and_arc_only_not_reresolved():
    # an explicit "" (arc-only, e.g. atlas descriptor) must NOT be re-resolved to chipyard
    assert set(CR.oracle_adapters("atlas", "")) == {"L3"}
    assert CR._bespoke_sim_via("gemmini") == "chipyard"
    assert CR._bespoke_sim_via("atlas") == ""


def test_run_capsule_default_is_contract_routed_not_gemmini_default():
    # the run_capsule fallback (oracle_adapters=None) resolves to the target's endpoint oracle, so an
    # external_backend target is NEVER graded by the gemmini default_adapters (the 0/11 mis-route).
    a3 = CR._resolve_oracle_adapters("atlas")
    ref = CR.oracle_adapters("atlas")
    assert set(a3) == set(ref)                           # same tier routing (closures differ by identity)
    assert {k: v.__qualname__ for k, v in a3.items()} == {k: v.__qualname__ for k, v in ref.items()}
    assert "program_oracle" in getattr(a3["L3"], "__qualname__", "")
    # default_adapters is retained for the explicitly-gemmini perf-bench, but is NOT the atlas route
    assert "program_oracle" not in getattr(CR.default_adapters()["L3"], "__qualname__", "")


# ---- FAULT 2/3: the DRAM address map is harness-owned, deterministic, agent-respecting -----------

def _atlas_matmul_capsule() -> dict:
    p = merlin_dir() / "contract/capsules/atlas/isa/AT2_single_tile_matmul/capsule.yaml"
    if not p.is_file():
        pytest.skip("atlas AT2 capsule absent")
    return yaml.safe_load(p.read_text())


def test_layout_is_deterministic_and_non_overlapping():
    cap = _atlas_matmul_capsule()
    lay1, lay2 = D.layout(cap), D.layout(cap)
    assert lay1 == lay2, "layout must be a pure function of the capsule (same map every process)"
    # inputs + the output are all placed, aligned, and non-overlapping
    assert set(lay1) == {"A0", "W", "Y0"}
    spans = []
    for t in cap["inputs"]:
        spans.append((lay1[t["name"]], D.tensor_nbytes(t["shape"], t["dtype"])))
    for a, n in spans:
        assert a % D.DEFAULT_ALIGN == 0
    # each input's [base, base+nbytes) is disjoint from the next input's base
    a0, w = lay1["A0"], lay1["W"]
    assert w >= a0 + D.tensor_nbytes([32, 32], "fp8_e4m3")


def test_inject_fills_missing_base_but_never_clobbers_agent_declared():
    cap = _atlas_matmul_capsule()
    cb = {"tensors": {
        "A0": {"shape": [32, 32], "dtype": "fp8_e4m3", "role": "input"},           # no base -> filled
        "Y0": {"shape": [32, 32], "dtype": "bf16", "role": "output", "base": 0xBEEF},  # declared -> kept
    }}
    D.inject_bases(cb, cap)
    assert cb["tensors"]["A0"]["base"] == D.layout(cap)["A0"]     # harness default supplied
    assert cb["tensors"]["Y0"]["base"] == 0xBEEF                  # agent's choice preserved


def test_output_tensor_resolves_for_matmul_even_when_not_in_inputs():
    cap = _atlas_matmul_capsule()
    ot = D.output_tensor(cap)
    assert ot is not None and ot["name"] == "Y0"
    assert ot["shape"] == [32, 32] and ot["dtype"] == "bf16"


def test_command_buffer_schema_accepts_fp8_and_base():
    from merlin.targetgen.contract import schemas
    cb = {"abi_version": "0.1", "target": "atlas", "commands": [{"opcode": "COMMIT"}],
          "tensors": {"A0": {"shape": [32, 32], "dtype": "fp8_e4m3", "role": "input", "base": 4096},
                      "Y0": {"shape": [32, 32], "dtype": "bf16", "role": "output", "base": 6144}}}
    schemas.validate_command_buffer(cb, contract=str(merlin_dir() / "contract"))  # must not raise


def test_external_backend_prompt_declares_the_dram_contract():
    from merlin.targetgen.generate_prompt import render_prompt
    from merlin.targetgen.target_experiment import load_target_experiment, load_capability_manifest
    desc = merlin_dir() / "experiments/capsule_bench/targets/atlas/target_experiment.yaml"
    if not desc.is_file():
        pytest.skip("atlas descriptor absent")
    te = load_target_experiment(desc)
    m = load_capability_manifest("atlas")
    p = render_prompt(te, m, experiment="full", arm="merlin_rtlchecks")
    assert "DRAM address map" in p and "base" in p
    # a non-external_backend target must NOT get the block
    gdesc = merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    if gdesc.is_file():
        gp = render_prompt(load_target_experiment(gdesc), load_capability_manifest("gemmini"),
                           experiment="full", arm="merlin_rtlchecks")
        assert "DRAM address map" not in gp
