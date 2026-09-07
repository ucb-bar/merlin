"""Whole-program kernels are graded by executing the complete submitted ELF.

The command buffer remains an accelerator-work projection for structural checks and work accounting;
it is not a semantic interpreter for scalar host islands.  Treating it as one would either reject every
honest mixed program or encourage authors to smuggle host computation through fake accelerator commands.
"""
from __future__ import annotations

import copy

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.runner_config import RunnerConfig


CAPSULE = repo_root() / "merlin/contract/capsules/isa/A2_single_tile_matmul"


def _config() -> RunnerConfig:
    return RunnerConfig(
        target="gemmini", suite="whole-program-test", dtype="i8xi8_i32",
        fourth_output_name="lowered.llvm.mlir",
        tier_sim={"L2": "spike", "L3": "gsim"},
        rtl_tiers=frozenset({"L3"}), oracle_tiers=("L2", "L3"),
        perf_fields=(), trace_gate=None)


def _whole_program_cb(cap: dict) -> dict:
    inputs = list(cap["inputs"])
    out = cap["operation"]["attributes"]["out"]
    tensors = {spec["name"]: dict(spec) for spec in inputs}
    tensors[out] = {"shape": [16, 16], "dtype": "i32", "role": "output"}
    return {
        "abi_version": "0.1", "target": "gemmini", "tensors": tensors,
        # Deliberately not a semantic rendering of the complete program.  A real mixed program's
        # commands describe only its accelerator regions; the host regions live in the submitted ELF.
        "commands": [],
        "kernel_abi": {
            "kind": "whole_program",
            "args": ([{"tensor": spec["name"], "access": "read"} for spec in inputs]
                     + [{"tensor": out, "access": "write"}]),
            "outputs": [out],
        },
    }


def _oracles(outputs: dict, seen: list[dict] | None = None) -> dict:
    def run(_cb, _llvm, _workdir, _timeout):
        if seen is not None:
            seen.append(copy.deepcopy(_cb))
        return {"outputs": copy.deepcopy(outputs), "cycles": 123, "oracle": "test-engine"}
    return {"L2": run, "L3": run}


def test_whole_program_grades_the_complete_elf_against_the_capsule_golden(
        tmp_path, monkeypatch):
    cap = load_capsule(CAPSULE, contract="merlin/contract")
    cb = _whole_program_cb(cap)
    monkeypatch.setattr(CR, "run_entrypoints",
                        lambda *args, **kwargs: (object(), cb, "module {}"))
    monkeypatch.setattr("merlin.runtime.reference.reference_outputs", lambda _cb: (_ for _ in ()).throw(
        AssertionError("an accelerator command projection is not the mixed program's semantic oracle")))
    monkeypatch.setattr("merlin.runtime.simulator.simulate", lambda _cb: (_ for _ in ()).throw(
        AssertionError("an accelerator command projection is not the mixed program's semantic oracle")))
    seen: list[dict] = []

    result = CR.run_capsule(
        cap, "unused-package", runs_root=tmp_path, run_id="whole_program_pass",
        config=_config(), oracle_adapters=_oracles(CG.golden(cap), seen))

    assert result["status"] == "pass", result.get("failure")
    assert result["tiers"]["L0"]["status"] == "skipped"
    assert result["tiers"]["L0"]["not_applicable"] is True
    assert result["tiers"]["L1"]["status"] == "skipped"
    assert result["tiers"]["L1"]["not_applicable"] is True
    assert result["tiers"]["L2"]["status"] == "pass"
    assert result["tiers"]["L3"]["status"] == "pass"
    assert result["numeric"]["status"] == "pass"
    assert result["numeric"]["golden_source"] == "merlin_tensor_int"
    assert len(seen) == 2
    assert all(one["canonical_inputs"] == CG.materialized_input_values(cap) for one in seen)


def test_whole_program_fails_when_the_complete_elf_disagrees_with_the_golden(
        tmp_path, monkeypatch):
    cap = load_capsule(CAPSULE, contract="merlin/contract")
    cb = _whole_program_cb(cap)
    monkeypatch.setattr(CR, "run_entrypoints",
                        lambda *args, **kwargs: (object(), cb, "module {}"))
    bad = CG.golden(cap)
    bad = copy.deepcopy(bad)
    bad["Y0"][0][0] += 1

    result = CR.run_capsule(
        cap, "unused-package", runs_root=tmp_path, run_id="whole_program_fail",
        config=_config(), oracle_adapters=_oracles(bad))

    assert result["status"] == "fail"
    assert result["tiers"]["L2"]["status"] == "fail"
    assert result["tiers"]["L3"]["status"] == "fail"
    assert result["numeric"]["status"] == "fail"


def test_recomputed_integer_whole_program_ignores_stale_recorded_operands(
        tmp_path, monkeypatch):
    """Integer whole-program stimulus comes from the recomputed golden, even if a historical
    golden.yaml happens to carry a different decoded operand payload."""
    cap = load_capsule(CAPSULE, contract="merlin/contract")
    cb = _whole_program_cb(cap)
    monkeypatch.setattr(CR, "run_entrypoints",
                        lambda *args, **kwargs: (object(), cb, "module {}"))
    stale = {
        spec["name"]: {"shape": list(spec["shape"]),
                       "values": [0] * __import__("math").prod(spec["shape"])}
        for spec in cap["inputs"]
    }
    monkeypatch.setattr(CG, "canonical_input_values", lambda *_args, **_kwargs: stale)
    seen: list[dict] = []

    result = CR.run_capsule(
        cap, "unused-package", runs_root=tmp_path, run_id="whole_program_stale_inputs",
        config=_config(), oracle_adapters=_oracles(CG.golden(cap), seen))

    expected = CG.materialized_input_values(cap)
    assert result["status"] == "pass", result.get("failure")
    assert all(one["canonical_inputs"] == expected for one in seen)
