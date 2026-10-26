"""Owner-selected ordering evidence supplementing immutable RTL facts, without replacing them.

Only dispatch port/configuration names may supplement the original selected contract. Queue releases,
completion order and spans are still derived by the existing RTL producer from the generic HW already
pinned by the base facts. The selection is not a timing-model or author-supplied effect grant.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.rtl_command_ordering_selection.v1"
ENV = "MERLIN_COMPLETION_ORDER_SELECTION"


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha(path)}


def _reopen(pin):
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"} or not Path(pin["path"]).is_absolute():
        raise ValueError("ordering selection requires an absolute path and SHA256 pin")
    path = Path(pin["path"])
    if _sha(path) != pin["sha256"]:
        raise ValueError(f"ordering selection source changed: {path}")
    return path


def _contract(path):
    import yaml

    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError("ordering selection contract must be a mapping")
    return doc


def _effective_contract(original, supplement):
    result = copy.deepcopy(original)
    base = result["rtl_extraction"]["timing_structure"]["dispatch"]
    extra = supplement["rtl_extraction"]["timing_structure"]["dispatch"]
    if {k: v for k, v in base.items() if k != "ordering"} != {k: v for k, v in extra.items() if k != "ordering"}:
        raise ValueError("ordering declarations change the selected dispatch module or classes")
    ordering = extra.get("ordering")
    if not isinstance(ordering, dict) or ("ordering" in base and base["ordering"] != ordering):
        raise ValueError("ordering declarations missing or conflict with the selected contract")
    base["ordering"] = copy.deepcopy(ordering)
    return result


def _generic(document):
    inputs = document.get("inputs") or {}
    if (document.get("source_consistency") or {}).get("status") != "verified":
        raise ValueError("ordering selection requires verified original RTL source consistency")
    return _reopen({"path": inputs.get("generic_hw_path", ""), "sha256": inputs.get("generic_hw_sha256")})


def _projection(target, contract, original):
    """Detached probe inputs from the already selected source-verified normalized observer.

    DEFINES/CONSUMES are the existing named local-memory ABI, also used by the ordering producer;
    no target names, operand bit positions, access dimensions or configuration values are invented.
    """
    if "rocc_operand_selection" not in contract:
        return contract, original["facts"], ()
    from merlin.perf.deps.rocc import CONSUMES, DEFINES
    from merlin.targetgen.rocc import semantics
    from merlin.targetgen.rocc.selected_operand_protocol import selected_inputs

    from .firrtl_operand_slices import apply_selected_layouts
    from .source_relocations import relocate_operand_sources

    owner = semantics.bind(target=target, contract=contract, facts=original)
    if owner.operand_observer is None:
        raise ValueError("ordering probes require the original selected operand observer")
    selection = contract["rocc_operand_selection"]
    files = tuple(_pin(p) for p in selected_inputs(selection))
    selected = json.loads(_reopen(selection["sidecar"]).read_bytes())
    protocol = selected["protocol"]
    observed = apply_selected_layouts(
        relocate_operand_sources(original, selected.get("source_relocations")), selected["layouts"]
    )
    projected = copy.deepcopy(contract)
    projected["encoding"] = {**contract["encoding"], **protocol["encoding"]}
    roles = copy.deepcopy(protocol["roles"])
    local, class_roles, sentinel = [], {}, None
    for cls, spec in roles["classes"].items():
        for field in spec["fields"]:
            name = field.get("name")
            if name in DEFINES or name in CONSUMES:
                if not all(field.get(k) for k in ("bundle", "field", "operand")):
                    raise ValueError("ordering address must bind an original source bundle field")
                local.append({"role": cls, "address": name})
                class_roles[cls] = [cls]
            if name == "garbage" and field.get("as") == "all_ones":
                source = {"value": "all_ones", "bundle": field["bundle"], "field": field["field"]}
                if sentinel is not None and sentinel != source:
                    raise ValueError("ordering source declares ambiguous retained-operand sentinel")
                sentinel = source
    if not local:
        raise ValueError("selected operand observer carries no named local-memory accesses")
    if sentinel is not None:
        roles["retain_sentinel"] = sentinel
    projected["rocc_operand_roles"] = roles
    projected["rtl_checks"] = {"class_roles": class_roles, "local_addresses": local}
    # Actual address width and selector must agree with the source-checked observer.
    isa = owner.operand_observer.isa_constants(target)
    if {int(k): v for k, v in protocol["encoding"]["semantic_class"].items()} != isa["FUNCT_CLASS"]:
        raise ValueError("ordering probe classes differ from the original selected observer")
    if files != tuple(_pin(p) for p in selected_inputs(selection)):
        raise ValueError("operand sources changed while preparing ordering probes")
    return projected, observed["facts"], files


def _readers():
    from merlin.perf.deps import rocc
    from merlin.targetgen import rtl_checks_generic
    from merlin.targetgen.rocc import normalized_operands, selected_operand_protocol, semantics

    from . import command_ordering, firrtl_operand_slices, hw_cycle_eval, hw_design, source_relocations

    modules = (
        command_ordering,
        hw_cycle_eval,
        hw_design,
        firrtl_operand_slices,
        source_relocations,
        rocc,
        rtl_checks_generic,
        normalized_operands,
        selected_operand_protocol,
        semantics,
    )
    return {m.__name__: _sha(m.__file__) for m in modules} | {__name__: _sha(__file__)}


def prepare(*, target: str, facts: Path, contract: Path, declarations: Path, output: Path):
    """Derive one explicit supplement; write only NEW output, never the original facts/contract."""
    from .command_ordering import derive_from_paths

    output = Path(output)
    if output.exists():
        raise ValueError("ordering selection output already exists")
    facts_pin, contract_pin, declarations_pin = (_pin(p) for p in (facts, contract, declarations))
    original = json.loads(Path(facts).read_bytes())
    if (original.get("inputs") or {}).get("target") != target:
        raise ValueError("ordering selection facts target mismatch")
    generic = _generic(original)
    effective = _effective_contract(_contract(contract), _contract(declarations))
    readers = _readers()
    projected, observed, operand_inputs = _projection(target, effective, original)
    record = derive_from_paths(target, generic, observed, contract=projected)
    if record.get("status") not in ("derived", "partial"):
        raise ValueError(f"ordering derivation unavailable: {record.get('reason')}")
    if readers != _readers():
        raise ValueError("ordering readers changed during derivation")
    doc = {
        "schema": SCHEMA,
        "target": target,
        "base_facts": facts_pin,
        "base_contract": contract_pin,
        "declarations": declarations_pin,
        "generic_hw": _pin(generic),
        "readers": readers,
        "declarations_sha256": _digest(effective["rtl_extraction"]["timing_structure"]["dispatch"]),
        "operand_inputs": operand_inputs,
        "probe_contract_sha256": _digest(projected),
        "probe_facts_sha256": _digest(observed),
        "ordering_sha256": _digest(record),
        "ordering": record,
    }
    # Reopen all original inputs again before publishing any selection.
    _validate(doc, target=target, document=original, selected_contract=Path(contract))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(doc, handle, indent=2)
        handle.write("\n")
    return _pin(output)


def _validate(doc, *, target, document, selected_contract):
    if doc.get("schema") != SCHEMA or doc.get("target") != target:
        raise ValueError("ordering selection schema or target mismatch")
    base = json.loads(_reopen(doc["base_facts"]).read_bytes())
    # load_facts may add a declared alias stamp outside the unchanged facts/inputs.
    selected = {k: v for k, v in document.items() if k != "served_for"}
    if base != selected:
        raise ValueError("ordering selection does not bind the complete selected base facts")
    contract = _reopen(doc["base_contract"])
    if _sha(selected_contract) != doc["base_contract"]["sha256"]:
        raise ValueError("ordering selection does not bind the selected target contract")
    effective = _effective_contract(_contract(contract), _contract(_reopen(doc["declarations"])))
    if _digest(effective["rtl_extraction"]["timing_structure"]["dispatch"]) != doc["declarations_sha256"]:
        raise ValueError("ordering declaration identity mismatch")
    generic = _reopen(doc["generic_hw"])
    if _sha(_generic(base)) != _sha(generic):
        raise ValueError("ordering generic HW differs from the original facts source")
    for pin in doc.get("operand_inputs", ()):
        _reopen(pin)
    record = doc["ordering"]
    provenance = record.get("provenance") or {}
    if (
        record.get("status") not in ("derived", "partial")
        or _digest(record) != doc["ordering_sha256"]
        or provenance.get("declarations_sha256") != doc["declarations_sha256"]
        or (provenance.get("core_hw_generic") or {}).get("sha256") != doc["generic_hw"]["sha256"]
        or doc["readers"] != _readers()
    ):
        raise ValueError("ordering record/source/reader identity mismatch")
    return record


def read(pin, *, target, document):
    """Verify an owner-selected absolute JSON pin against the actual selected facts and contract."""
    from .facts import find_facts, target_contract_path

    path = _reopen(pin)
    doc = json.loads(path.read_bytes())
    selected_facts = find_facts(target)
    if selected_facts is None or _sha(selected_facts) != doc["base_facts"]["sha256"]:
        raise ValueError("ordering selection base facts are not the selected artifact")
    record = _validate(doc, target=target, document=document, selected_contract=target_contract_path(target))
    return record, {"selection": dict(pin), "ordering_sha256": doc["ordering_sha256"]}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("target", "facts", "contract", "declarations", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    pin = prepare(
        target=args.target,
        facts=Path(args.facts),
        contract=Path(args.contract),
        declarations=Path(args.declarations),
        output=Path(args.output),
    )
    print(json.dumps(pin))


if __name__ == "__main__":
    main()
