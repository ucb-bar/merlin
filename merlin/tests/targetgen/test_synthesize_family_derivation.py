"""OV4 regression: the onboarding synthesizers derive their plans from the compute-unit FAMILY /
contract features, never from a hardware target name.

The former ``if name=="toy_npu"/"saturn"`` branches + ``CURATED_TARGETS={"saturn"}`` + the per-name
``_DEFAULTS`` table are gone: the LLVM-fork posture rides ``families.contract_endpoint_kind``, and the
runtime-adapter / zephyr concrete plans ride the contract's command-buffer tensor-resident features.
The neutral ``toy_npu`` example (``families.DEFAULT_EXAMPLE_TARGET``) is the FAMILY DEFAULT, not a
special case.
"""
from __future__ import annotations

from merlin.common import schemas
from merlin.targetgen import families
from merlin.targetgen.evidence.store import Evidence
from merlin.targetgen.synthesize.llvm_extension_plan import synthesize_llvm_extension_plan
from merlin.targetgen.synthesize.runtime_adapter_plan import synthesize_runtime_adapter_plan
from merlin.targetgen.synthesize.target_contract import synthesize_target_contract
from merlin.targetgen.synthesize.zephyr_plan import synthesize_zephyr_plan


def _unit_contract(name: str, kind: str) -> dict:
    return {"name": name, "compute_units": [
        {"name": "u", "kind": kind, "ops": ["matmul"], "dtypes": ["int8"]}]}


def _ev(name: str) -> Evidence:
    return Evidence(target=name, sources={})


def test_llvm_fork_posture_is_keyed_on_family_endpoint_not_name():
    # vector/scalar (upstream LLVM RISC-V/RVV path) -> a fork is MAYBE needed; systolic/simt (.insn on
    # stock LLVM) + spatial (command buffer) -> no fork. The name is irrelevant — only the family kind.
    for kind in ("vector", "scalar"):
        p = synthesize_llvm_extension_plan(_ev("acme"), _unit_contract("acme", kind))
        assert p["requires_llvm_fork"] == "maybe"
        assert schemas.validate(p, "llvm_extension_plan") == []
    for kind in ("systolic", "simt", "spatial"):
        p = synthesize_llvm_extension_plan(_ev("acme"), _unit_contract("acme", kind))
        assert p["requires_llvm_fork"] is False

    # Two DIFFERENTLY-named contracts of the SAME family derive the identical posture.
    a = synthesize_llvm_extension_plan(_ev("alpha"), _unit_contract("alpha", "vector"))
    b = synthesize_llvm_extension_plan(_ev("omega"), _unit_contract("omega", "vector"))
    assert (a["requires_llvm_fork"], a["initial_strategy"]) == (b["requires_llvm_fork"], b["initial_strategy"])

    # A contract that resolves no family (no compute_units) falls back to the family-default seed.
    d = synthesize_llvm_extension_plan(_ev("bare"), {"name": "bare"})
    assert d["requires_llvm_fork"] is False and d["confidence"] == "high"


def test_runtime_adapter_is_concrete_for_any_command_buffer_resident_contract():
    # An arbitrarily-named contract that declares the command-buffer tensor-resident family gets the
    # concrete command-stream adapter (parameterized), flagged for review since it is not the example.
    tc = {"name": "acme_npu",
          "features": ["command_buffer", "accumulator_commit", "resident_packed_tensor", "metrics"]}
    p = synthesize_runtime_adapter_plan(_ev("acme_npu"), tc)
    assert p["command_encoding"]["format"] == "acmenpu_command_stream"
    assert p["requires_human_review"] is True
    assert schemas.validate(p, "runtime_adapter_plan") == []
    # A non-command-buffer contract gets the review-flagged skeleton (no name check).
    skel = synthesize_runtime_adapter_plan(_ev("plain"), {"name": "plain", "features": []})
    assert skel["requires_human_review"] is True and skel["command_encoding"]["format"] == "TODO_human_review"


def test_neutral_example_is_the_family_default_across_synthesizers():
    # toy_npu is selected via the DEFAULT_EXAMPLE_TARGET constant (the one neutral example), not a
    # hardware-name branch: it yields the concrete, review-free seed plans.
    name = families.DEFAULT_EXAMPLE_TARGET
    tc = synthesize_target_contract(_ev(name), name)
    assert tc["name"] == name and tc["requires_human_review"] is False
    rap = synthesize_runtime_adapter_plan(_ev(name), tc)
    assert rap["requires_human_review"] is False
    assert rap["command_encoding"]["format"] == "toynpu_command_stream"
    zp = synthesize_zephyr_plan(_ev(name), tc)
    assert zp["requires_human_review"] is False and zp["samples"] == ["toynpu_repeated_rhs_matmul"]
