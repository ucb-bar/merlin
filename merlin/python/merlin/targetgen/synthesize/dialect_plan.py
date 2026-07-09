"""Synthesize a dialect_plan (validates against dialect_plan.schema.yaml).

Required fields: target, dialect_name, ops, types, lowering, tests.

toy_npu -> the concrete ToyNPU dialect (res_pack/matmul/commit/evict). Real targets -> a
skeleton with no asserted ops (ops are a human-review decision), flagged for review.
"""
from __future__ import annotations

from typing import Any

from ..evidence.store import Evidence


def _toy_npu() -> dict[str, Any]:
    return {
        "target": "toy_npu",
        "dialect_name": "toynpu",
        "ops": [
            {"name": "res_pack", "summary": "pack + make RHS resident",
             "source_interface": "interface.resident_pack"},
            {"name": "matmul", "summary": "matmul vs resident tensor -> accumulator",
             "source_interface": "interface.matmul"},
            {"name": "commit", "summary": "apply epilogue + commit accumulator",
             "source_interface": "interface.commit"},
            {"name": "evict", "summary": "free resident storage",
             "source_interface": "interface.resident_evict"},
        ],
        "types": [
            {"name": "resident_tensor"},
            {"name": "accumulator"},
        ],
        "lowering": [
            {"from": "interface.resident_pack", "to": "toynpu.res_pack"},
            {"from": "interface.matmul", "to": "toynpu.matmul"},
            {"from": "interface.commit", "to": "toynpu.commit"},
            {"from": "interface.resident_evict", "to": "toynpu.evict"},
        ],
        "tests": [
            {"lit": "res_pack_roundtrip"},
            {"lit": "matmul_commit_epilogue"},
            {"lit": "evict_after_use"},
        ],
        "confidence": "high",
        "requires_human_review": False,
    }


def _conservative(evidence: Evidence) -> dict[str, Any]:
    concepts = sorted(evidence.concept_names())
    return {
        "target": evidence.target,
        "dialect_name": evidence.target,
        "ops": [],
        "types": [],
        "lowering": [],
        "tests": [],
        "detected_concepts": concepts,
        "notes": "Ops/types/lowerings are a human-review decision; do not auto-generate "
                 "dialect ops directly from instruction names.",
        "confidence": "low",
        "requires_human_review": True,
    }


def _curated(target_name: str) -> dict[str, Any] | None:
    """Load a curated in-tree dialect plan (merlin/targets/<t>/contracts/) if any."""
    import yaml

    from ...common.paths import targets_dir

    path = targets_dir() / target_name / "contracts" / "dialect_plan.yaml"
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# The Merlin tensor-resident interface a target dialect lowers. role -> (canonical op name, matcher,
# summary). A contract that advertises this shape gets a GENERATED, usable dialect_plan (not a stub).
_ROLES: list[tuple[str, str, Any, str]] = [
    ("resident_pack", "pack", lambda o: "pack" in o, "pack + make RHS resident"),
    ("matmul", "matmul", lambda o: "matmul" in o, "matmul vs resident tensor -> accumulator"),
    ("commit", "commit", lambda o: "commit" in o, "apply epilogue + commit accumulator"),
    ("resident_evict", "evict", lambda o: "evict" in o or "release" in o, "free resident storage"),
]


def _is_tensor_resident(tc: dict[str, Any]) -> bool:
    """A target that implements the Merlin tensor-resident interface (packs a resident weight,
    accumulates, commits via a command buffer). Detected from the contract's own declarations."""
    feats = set(tc.get("features") or [])
    ops = set(tc.get("ops") or []) | set((tc.get("capabilities") or {}).get("ops") or [])
    return ("command_buffer" in feats
            and ("resident_packed_tensor" in feats or "accumulator_commit" in feats)
            and "matmul" in ops)


def _generate(target_contract: dict[str, Any]) -> dict[str, Any]:
    """Generate a usable dialect_plan from a tensor-resident contract (its op/type names drive the
    dialect; the interface->target lowering is the canonical mapping). Feeds the dialect factory."""
    name = target_contract["name"]
    dname = name.replace("_", "")
    decl_ops = list(target_contract.get("ops") or [])
    role_op = {role: (next((o for o in decl_ops if match(o)), canon))
               for role, canon, match, _ in _ROLES}
    types = target_contract.get("types") or ["resident_tensor", "accumulator"]
    plan = {
        "target": name,
        "dialect_name": dname,
        "ops": [{"name": role_op[r], "summary": summ, "source_interface": f"interface.{r}"}
                for r, _c, _m, summ in _ROLES],
        "types": [{"name": t} for t in types],
        "lowering": [{"from": f"interface.{r}", "to": f"{dname}.{role_op[r]}"}
                     for r, _c, _m, _s in _ROLES],
        "tests": [{"lit": "pack_roundtrip"}, {"lit": "matmul_commit_epilogue"},
                  {"lit": "evict_after_use"}],
        "confidence": "medium",
        "requires_human_review": False,
        "generated_from_contract": True,
    }
    from ...common.schemas import validate_or_raise
    validate_or_raise(plan, "dialect_plan")
    return plan


def synthesize_dialect_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return a dialect_plan dict for the contract's target: a committed curated plan wins; else a
    tensor-resident contract is GENERATED into a usable plan; else a review-flagged skeleton."""
    name = target_contract.get("name")
    if name == "toy_npu":
        return _toy_npu()
    curated = _curated(name)                 # committed reference plan (saturn, gemmini, ...) wins
    if curated is not None:
        return curated
    if _is_tensor_resident(target_contract):
        return _generate(target_contract)    # usable generated plan (was: empty _conservative stub)
    return _conservative(evidence)
