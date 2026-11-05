"""Select proof semantics only from exact original software and raw hardware facts.

Inputs are host selections, never a candidate registry, filename-derived target
or serialized resolved model. Reconstruction rereads snapshot bytes and reruns
the fixed resolver. Hardware correspondence remains independent of this model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

SCHEMA = "merlin.frozen_isa_semantics.v1"
ADMISSION_SCHEMA = "merlin.formal_semantics_admission.v1"
_PREMISES = frozenset(
    {"standard_memory_sequential", "standard_allocations_successful", "dynamic_memref_accesses_defined"}
)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file(path: Path | None, resolver) -> tuple[dict | None, bytes | None]:
    if path is None:
        return None, None
    original = Path(path)
    selected = Path(resolver(original))
    if (
        not original.is_absolute()
        or ".." in original.parts
        or selected.is_symlink()
        or selected.resolve() != selected
        or not selected.is_file()
    ):
        raise ValueError("formal semantic input must be an explicitly selected ordinary file")
    payload = selected.read_bytes()
    return {"path": str(original), "sha256": hashlib.sha256(payload).hexdigest()}, payload


def _admission(document, model):
    policy = document.get("formal_semantics")
    if policy is None:
        return None
    fields = {"schema", "admitted_instructions", "atomic_instructions", "disjoint_resources", "premises"}
    if not isinstance(policy, dict) or set(policy) != fields or policy.get("schema") != ADMISSION_SCHEMA:
        raise ValueError("formal semantics admission requires the original explicit reviewed schema")
    if document.get("status") != "reviewed":
        raise ValueError("formal semantics admission must originate in reviewed software semantics")
    declared = {f"{model.namespace}.{instruction.name}" for instruction in model.instructions} if model else set()
    for key in ("admitted_instructions", "atomic_instructions"):
        values = policy[key]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("formal ISA admission requires unique explicit instruction identities")
        if set(values) - declared:
            raise ValueError("formal ISA admission refers to undeclared instructions")
    if set(policy["admitted_instructions"]) != set(policy["atomic_instructions"]):
        raise ValueError("every admitted ISA instruction needs its original atomic completion admission")
    resources = {f"{model.namespace}:{resource.name}" for resource in model.resources} if model else set()
    pairs = policy["disjoint_resources"]
    if not isinstance(pairs, list):
        raise ValueError("formal resource disjointness must be an explicit reviewed list")
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or pair[0] == pair[1]
            or any(not isinstance(value, str) or value not in resources for value in pair)
        ):
            raise ValueError("formal resource separation refers to undeclared or identical resources")
    premises = policy["premises"]
    if (
        not isinstance(premises, dict)
        or set(premises) - _PREMISES
        or any(type(value) is not bool for value in premises.values())
    ):
        raise ValueError("formal memory premises must be original explicit supported booleans")
    return json.loads(json.dumps(policy))


@dataclass(frozen=True)
class FrozenISASemantics:
    target: str | None
    software_spec: dict | None
    rtl_facts: dict | None
    models: tuple
    registry: dict
    effect_policy: dict | None

    def record(self) -> dict:
        return {
            "schema": SCHEMA,
            "selection": "software_isa" if self.models else "builtins_only",
            "target": self.target,
            "software_spec": self.software_spec,
            "rtl_facts": self.rtl_facts,
            "model_sha256": _digest([model.record() for model in self.models]),
            "registry_sha256": _digest(self.registry),
            "effect_policy_sha256": _digest(self.effect_policy),
        }

    def effect_contract(self, *, index_bits: int):
        """Original host defined-behavior model, with no automatic OOT admission."""
        from .lean_state_effects import EffectScheduleContract, effect_contract_from_isa

        premises = {
            "standard_memory_sequential": True,
            "standard_allocations_successful": True,
            "dynamic_memref_accesses_defined": False,
            "index_bits": index_bits,
        }
        if self.effect_policy is None:
            return EffectScheduleContract(**premises)
        premises.update(self.effect_policy["premises"])
        return effect_contract_from_isa(
            self.models,
            admitted_instructions=tuple(self.effect_policy["admitted_instructions"]),
            atomic_instructions=tuple(self.effect_policy["atomic_instructions"]),
            disjoint_resources=tuple(tuple(pair) for pair in self.effect_policy["disjoint_resources"]),
            **premises,
        )


def _select(*, software_spec_path, facts_path, target, resolver):
    from merlin.targetgen.isa_spec import resolve_isa

    software, software_bytes = _file(software_spec_path, resolver)
    facts, facts_bytes = _file(facts_path, resolver)
    spec = yaml.safe_load(software_bytes) if software_bytes is not None else None
    if spec is not None:
        if not isinstance(spec, dict) or spec.get("schema") != "merlin.software_spec.v1":
            raise ValueError("formal semantic input is not an original software specification")
        name = spec.get("target")
        if not isinstance(name, str) or not name.strip() or (target is not None and target != name):
            raise ValueError("formal software semantics disagree with the explicit target selection")
        target = name
    raw_facts = json.loads(facts_bytes) if facts_bytes is not None else None
    if raw_facts is not None:
        body = raw_facts.get("facts") if isinstance(raw_facts, dict) else None
        name = body.get("target") if isinstance(body, dict) else None
        if not isinstance(name, str) or not name.strip() or (target is not None and target != name):
            raise ValueError("formal raw facts disagree with the explicit target selection")
        target = name
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("formal semantics require an explicit target identity or no target")
    models = ()
    registry = {}
    if spec is not None and "isa" in spec:
        if spec.get("status") != "reviewed" or raw_facts is None:
            raise ValueError("formal ISA semantics require reviewed software and exact original raw facts")
        model = resolve_isa(spec["isa"], raw_facts, target=target, facts_sha256=facts["sha256"])
        models = (model,)
        for instruction in model.instructions:
            registry[f"{model.namespace}.{instruction.name}"] = {
                "effects": json.loads(instruction.effects_json),
                "semantics": json.loads(instruction.semantics_json),
                "operand_names": [arg.name for arg in instruction.operands],
                "operand_types": [
                    {"kind": arg.type.kind, "bits": arg.type.bits, "signed": arg.type.signed}
                    for arg in instruction.operands
                ],
                "result_types": [
                    {"kind": arg.type.kind, "bits": arg.type.bits, "signed": arg.type.signed}
                    for arg in instruction.results
                ],
                "immediates": [arg.name for arg in instruction.immediates],
            }
    policy = _admission(spec, models[0] if models else None) if spec is not None else None
    return FrozenISASemantics(target, software, facts, models, registry, policy)


def select_isa_semantics(
    *, software_spec_path: Path | None = None, facts_path: Path | None = None, target: str | None = None
) -> FrozenISASemantics:
    return _select(
        software_spec_path=software_spec_path, facts_path=facts_path, target=target, resolver=lambda path: path
    )


def validate_semantic_inputs(record: dict) -> None:
    fields = {
        "schema",
        "selection",
        "target",
        "software_spec",
        "rtl_facts",
        "model_sha256",
        "registry_sha256",
        "effect_policy_sha256",
    }
    if not isinstance(record, dict) or set(record) != fields or record.get("schema") != SCHEMA:
        raise ValueError("unknown frozen formal semantic input schema")
    if record["selection"] not in {"builtins_only", "software_isa"}:
        raise ValueError("frozen formal semantic selection is unknown")
    if record["target"] is not None and (not isinstance(record["target"], str) or not record["target"].strip()):
        raise ValueError("frozen formal semantics have an invalid explicit target identity")
    for key in ("software_spec", "rtl_facts"):
        row = record[key]
        if row is None:
            continue
        if not isinstance(row, dict) or set(row) != {"path", "sha256"} or not isinstance(row["path"], str):
            raise ValueError("frozen formal semantic input member is malformed")
        path = Path(row["path"])
        if not path.is_absolute() or ".." in path.parts or not _sha(row["sha256"]):
            raise ValueError("frozen formal semantic input identity is malformed")
    if not all(_sha(record[key]) for key in ("model_sha256", "registry_sha256", "effect_policy_sha256")):
        raise ValueError("frozen formal semantic model identity is malformed")
    if record["selection"] == "software_isa" and (record["software_spec"] is None or record["rtl_facts"] is None):
        raise ValueError("frozen ISA semantics omit their original software or raw hardware facts")


def reconstruct_isa_semantics(
    record: dict, *, resolve_path: Callable[[Path], Path] = lambda path: path
) -> FrozenISASemantics:
    """Resolve original recorded inputs through a host-owned snapshot mapping."""
    validate_semantic_inputs(record)
    result = _select(
        software_spec_path=Path(record["software_spec"]["path"]) if record["software_spec"] else None,
        facts_path=Path(record["rtl_facts"]["path"]) if record["rtl_facts"] else None,
        target=record["target"],
        resolver=resolve_path,
    )
    if result.record() != record:
        raise ValueError("frozen formal semantic inputs or resolved model differ from original admission")
    return result
