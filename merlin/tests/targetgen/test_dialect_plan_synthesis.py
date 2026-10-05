"""TargetGen synthesizes a USABLE dialect_plan from a tensor-resident contract (WS-D).

Previously every non-toy_npu target fell to an empty `requires_human_review` stub. Now a contract
that advertises the Merlin tensor-resident interface is generated into a real plan (reproducing the
committed reference plans, and buildable by the dialect factory).
"""

from __future__ import annotations

import yaml

from merlin.common import schemas
from merlin.common.paths import merlin_dir
from merlin.targetgen.evidence.store import Evidence
from merlin.targetgen.synthesize.dialect_plan import (
    _generate, _generate_from_operation_capabilities, _is_tensor_resident,
    synthesize_dialect_plan,
)


def _contract(target: str) -> dict:
    return yaml.safe_load((merlin_dir() / "targets" / target / "contracts" / "target_contract.yaml").read_text())


def test_generator_reproduces_committed_reference_plans():
    """_generate(contract) reproduces the committed reference dialect_name + lowering.

    gemmini no longer ships a committed dialect_plan (the target dialect is an agent-owned design
    decision, verified against the CIRCT oracle), so only saturn is checked here.
    """
    for t in ("saturn",):
        committed = yaml.safe_load((merlin_dir() / "targets" / t / "contracts" / "dialect_plan.yaml").read_text())
        g = _generate(_contract(t))
        assert g["dialect_name"] == committed["dialect_name"]
        assert {r["from"]: r["to"] for r in g["lowering"]} == {r["from"]: r["to"] for r in committed["lowering"]}
        assert schemas.validate(g, "dialect_plan") == []


def test_detects_tensor_resident_from_contract():
    assert _is_tensor_resident(_contract("saturn")) is True
    assert _is_tensor_resident({"features": [], "ops": []}) is False


def test_synthesize_generates_usable_plan_for_new_target():
    """A new tensor-resident contract synthesizes a usable plan (not the review-flagged stub)."""
    newc = {
        "name": "demo_npu",
        "features": ["resident_packed_tensor", "accumulator_commit", "command_buffer"],
        "ops": ["pack", "matmul", "commit", "evict"],
        "types": ["resident_tensor", "accumulator"],
        "capabilities": {"ops": ["matmul"]},
    }
    plan = synthesize_dialect_plan(Evidence(target="demo_npu", sources={}), newc)
    assert plan.get("generated_from_contract") is True
    assert plan["requires_human_review"] is False
    assert schemas.validate(plan, "dialect_plan") == []
    assert [o["name"] for o in plan["ops"]] == ["pack", "matmul", "commit", "evict"]


def test_generated_plan_builds_a_dialect():
    """The generated plan is buildable by the parametric factory (contract -> plan -> dialect)."""
    import pytest

    from merlin.xdsl_dialects import _common

    if not _common.HAS_XDSL:
        pytest.skip("xDSL not installed")
    from merlin.xdsl_dialects.targets.factory import build_dialect

    newc = {
        "name": "demo_npu",
        "features": ["accumulator_commit", "command_buffer"],
        "ops": ["pack", "matmul", "commit", "evict"],
        "capabilities": {"ops": ["matmul"]},
    }
    plan = _generate(newc)
    built = build_dialect("demo_npu", plan=plan)
    assert built.dialect.name == "demonpu"
    assert {o.name for o in built.dialect.operations} == {
        "demonpu.pack",
        "demonpu.matmul",
        "demonpu.commit",
        "demonpu.evict",
    }


def _typed_contract() -> dict:
    return {
        "name": "synthetic", "dialect_name": "synthetic", "types": [{"name": "state"}],
        "operation_capabilities": {"version": 1, "operations": [{
            "domain": "dialect", "dialect": "synthetic", "operation": "step",
            "status": "unknown", "effects": ["movement"],
            "semantics": {"kind": "machine", "mlir_signature": {
                "operands": [{"name": "state", "type": "!synthetic.state"}],
                "results": [{"name": "next", "type": "!synthetic.state"}],
                "attributes": [{
                    "name": "mode", "type": "string", "role": "mode", "choices": ["read", "write"],
                }],
                "effects": ["read", "write"],
            }},
        }]},
    }


def test_typed_plan_is_generated_from_existing_operation_contract():
    from merlin.targetgen.generate import mlir_scaffold

    contract = _typed_contract()
    plan = _generate_from_operation_capabilities(contract)
    assert plan is not None
    assert synthesize_dialect_plan(Evidence(target="synthetic", sources={}), contract) == plan
    assert plan["generated_from_contract"] is True
    assert plan["requires_human_review"] is True
    assert plan["source_operation_ids"] == [
        {"domain": "dialect", "dialect": "synthetic", "operation": "step"}
    ]
    assert schemas.validate(plan, "dialect_plan") == []
    artifacts = {item.relpath: item.content for item in mlir_scaffold.generate(plan)}
    ops = artifacts["include/MerlinTargetSynthetic/Dialect/Synthetic/IR/SyntheticOps.td"]
    assert 'Synthetic_State:$state' in ops
    assert 'StrAttr:$mode' in ops
    assert 'Synthetic_StepOp' in ops
    contract["operation_capabilities"]["operations"][0]["semantics"]["mlir_signature"]["attributes"][0]["choices"] = ["read"]
    changed = _generate_from_operation_capabilities(contract)
    assert changed["ops"][0]["signature"]["attributes"][0]["choices"] == ["read"]


def test_incomplete_typed_contract_refuses_name_only_fallback():
    import pytest

    contract = _typed_contract()
    contract["operation_capabilities"]["operations"].append({
        "domain": "dialect", "dialect": "synthetic", "operation": "other",
        "semantics": {"kind": "machine"},
    })
    with pytest.raises(ValueError, match="every selected dialect operation"):
        _generate_from_operation_capabilities(contract)
