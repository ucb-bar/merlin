"""The same semantic output contract grades pointer and whole-program kernels.

Adapters and entrypoints are stubs: these tests run the actual ladder without compiling or
executing a kernel. The golden and public logical readback roster remain their real owners.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.runner_config import RunnerConfig


@pytest.fixture
def ladder(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_GRADE_CPU_PROCESSES", "0")
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", str(tmp_path / "cert-cache"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    monkeypatch.setattr(CR, "_order_after_certificate", lambda *a, **k: None)
    # The entrypoint fixture has no real source/ELF to bind to scratch custody.
    monkeypatch.setattr(CR, "oracle_source_scope", lambda *a, **k: nullcontext())
    cap = load_capsule(
        repo_root() / "merlin/contract/capsules/isa/A2_single_tile_matmul", contract=repo_root() / "merlin/contract"
    )
    cap["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
    cap["numeric_policy"] = {"compare": "bounded_int", "atol": 0, "rtol": 0}
    gold = CG.golden(cap)
    out = next(iter(gold))
    cb = {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            **{spec["name"]: dict(spec) for spec in cap["inputs"]},
            out: {"shape": [16, 16], "dtype": "i32", "role": "output"},
            "Yc": {"shape": [16, 16], "dtype": "i32", "role": "intermediate"},
            "unpublished": {"shape": [1, 1], "dtype": "i32", "role": "intermediate"},
        },
        # A published intermediate is legal readback; a mere tensor-table entry is not.
        "commands": [{"opcode": "COMMIT", "operands": {"dst": "Yc"}}],
    }
    reference = {**copy.deepcopy(gold), "Yc": [[4]]}
    simulation = {**copy.deepcopy(gold), "Yc": [[5]]}
    monkeypatch.setattr("merlin.runtime.reference.reference_outputs", lambda _cb: reference)
    monkeypatch.setattr("merlin.runtime.simulator.simulate", lambda _cb: {"outputs": simulation})
    config = RunnerConfig(
        target="gemmini",
        suite="output-contract-test",
        dtype="i8xi8_i32",
        fourth_output_name="lowered.llvm.mlir",
        tier_sim={"L2": "test", "L3": "test"},
        rtl_tiers=frozenset({"L3"}),
        oracle_tiers=("L2", "L3"),
        perf_fields=(),
        trace_gate=None,
    )

    def grade(whole_program, defect=None, multi_output=False):
        selected_cap = copy.deepcopy(cap)
        selected_gold = copy.deepcopy(gold)
        command_buffer = copy.deepcopy(cb)
        readouts = [out, "Yc"]
        if multi_output:
            selected_gold["F"] = reference["F"] = simulation["F"] = [[1.25, 2.5]]
            selected_cap["numeric_policy"] = {
                "schema": "merlin.numeric_policy.per_output.v1",
                "compare": "per_output",
                "outputs": {out: {"compare": "exact_int"}, "F": {"compare": "tolerance_float", "atol": 0.1}},
            }
            command_buffer["tensors"]["F"] = {"shape": [1, 2], "dtype": "f32", "role": "output"}
            readouts.append("F")
            monkeypatch.setattr(CG, "golden", lambda *a, **k: copy.deepcopy(selected_gold))
        if whole_program:
            command_buffer["kernel_abi"] = {
                "kind": "whole_program",
                "outputs": readouts,
                "args": (
                    [{"tensor": spec["name"], "access": "read"} for spec in cap["inputs"]]
                    + [{"tensor": name, "access": "write"} for name in readouts]
                ),
            }
        outputs = {**copy.deepcopy(selected_gold), "Yc": [[999]]}
        if defect == "missing":
            del outputs[out]
        elif defect == "empty":
            outputs[out] = []
        elif defect == "size":
            outputs[out] = [[1]]
        elif defect == "value":
            outputs[out][0][0] += 1
        elif defect == "nonfinite":
            outputs[out][0][0] = float("nan")
        elif defect == "undeclared":
            outputs["unpublished"] = [[1]]
        elif defect == "restricted":
            command_buffer["outputs"] = [out]
            reference.pop("Yc")
            simulation.pop("Yc")
            if whole_program:
                command_buffer["kernel_abi"]["outputs"] = [out]
        monkeypatch.setattr(CR, "run_entrypoints", lambda *a, **k: (object(), command_buffer, "module {}"))

        def oracle(*_args):
            return {"outputs": copy.deepcopy(outputs), "cycles": 123, "oracle": "stub"}

        return CR.run_capsule(
            selected_cap,
            "unused-package",
            runs_root=tmp_path,
            run_id=f"{'whole' if whole_program else 'pointer'}-{defect or 'good'}",
            config=config,
            oracle_adapters={"L2": oracle, "L3": oracle},
        )

    return grade


@pytest.mark.parametrize("whole_program", [False, True])
@pytest.mark.parametrize("multi_output", [False, True])
def test_declared_intermediate_readback_passes_both_real_ladder_routes(ladder, whole_program, multi_output):
    result = ladder(whole_program, multi_output=multi_output)
    assert result["status"] == "pass", result.get("failure")
    assert result["tiers"]["L2"]["status"] == result["tiers"]["L3"]["status"] == "pass"
    assert result["tiers"]["L0"]["status"] == ("skipped" if whole_program else "pass")
    assert result["tiers"]["L1"]["status"] == ("skipped" if whole_program else "pass")


@pytest.mark.parametrize("whole_program", [False, True])
@pytest.mark.parametrize("defect", ["missing", "empty", "size", "value", "nonfinite", "undeclared", "restricted"])
def test_bad_semantic_or_unknown_outputs_fail_both_ladder_routes(ladder, whole_program, defect):
    result = ladder(whole_program, defect)
    assert result["status"] == "fail", result
    assert result["tiers"]["L2"]["status"] == result["tiers"]["L3"]["status"] == "fail"


@pytest.mark.parametrize("whole_program", [False, True])
def test_numeric_success_still_reaches_failing_completion_order_gate(ladder, monkeypatch, whole_program):
    seen = []

    def stepper(*_args):
        def step(tier, tiers, generated, *_rest):
            seen.append(tier)
            return {"verdict": "hazard"}, CR.CertFailure(
                "completion_order", "protocol_violation", "test ordering hazard"
            )

        return step

    monkeypatch.setattr(CR._CO_GATE, "stepper", stepper)
    result = ladder(whole_program)
    assert seen == ["L2"]
    assert result["status"] == "fail"
    assert result["failure"]["plane"] == "completion_order"


PER_OUTPUT = {
    "schema": "merlin.numeric_policy.per_output.v1",
    "compare": "per_output",
    "outputs": {"I": {"compare": "exact_int"}, "F": {"compare": "tolerance_float", "atol": 0.1}},
}


def test_multileaf_per_output_policy_keeps_complete_semantic_roster_and_tolerances():
    gold = {"I": [[1, 2]], "F": [[1.25, 2.5]]}
    observed = {"I": [[1, 2]], "F": [[1.3, 2.5]], "published": [[999]]}
    report = CG.compare_declared_outputs(gold, observed, PER_OUTPUT, readback_outputs=["published"])
    assert report["status"] == "pass"
    assert set(report["per_output"]) == {"I", "F"}
    # Reference maps can include intermediate results without expanding required leaf policies.
    assert (
        CG.compare_declared_outputs(
            {**gold, "published": [[0]]},
            observed,
            PER_OUTPUT,
            readback_outputs=["published"],
            required_outputs=tuple(gold),
        )["status"]
        == "pass"
    )
    observed["I"][0][1] = 2.5
    assert CG.compare_declared_outputs(gold, observed, PER_OUTPUT, readback_outputs=["published"])["status"] == "fail"
    observed["I"][0][1] = 2
    observed["F"][0][0] = 1.5
    assert CG.compare_declared_outputs(gold, observed, PER_OUTPUT, readback_outputs=["published"])["status"] == "fail"
    del observed["F"]
    assert CG.compare_declared_outputs(gold, observed, PER_OUTPUT, readback_outputs=["published"])["status"] == "fail"


@pytest.mark.parametrize("gold", [{}, {"Y": []}])
def test_empty_expected_contract_cannot_pass_vacuously(gold):
    assert CG.compare_declared_outputs(gold, copy.deepcopy(gold), {"compare": "exact_int"})["status"] == "fail"


def test_legacy_compare_behavior_is_unchanged():
    gold, observed = {"Y": [[1]]}, {"Y": [[1]], "extra": [[2]]}
    assert CG.compare(gold, observed, {"compare": "exact_int"})["status"] == "pass"
    assert CG.compare_declared_outputs(gold, observed, {"compare": "exact_int"})["status"] == "fail"


def test_memory_order_recheck_uses_the_same_readback_boundary(tmp_path, monkeypatch):
    from merlin.targetgen import load_order

    observed = {"Y": [[1]], "published": [[99]]}
    monkeypatch.setattr(
        CR._backends_mod, "get_backend", lambda _target: SimpleNamespace(parse_output=lambda _text: (observed, {}))
    )
    monkeypatch.setattr(load_order, "after_certificate", lambda **kwargs: kwargs["judge"](0, "stub console"))

    def check():
        return CR._order_after_certificate(
            "stub",
            SimpleNamespace(rtl_tiers={"L3"}),
            {},
            None,
            tmp_path,
            gold={"Y": [[1]]},
            policy={"compare": "exact_int"},
            gsource="test",
            readback_outputs=["published"],
        )

    assert check() == "pass"
    observed["unknown"] = [[0]]
    assert check() == "fail"
    del observed["unknown"]
    observed["Y"] = [[2]]
    assert check() == "fail"
