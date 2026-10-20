"""Compare one accelerator unit's elaborated structure across two SoC configurations.

Measurements taken on one configuration (a simulation-friendly SoC) transfer to another (the
chosen target SoC) only when the accelerator unit is the same hardware in both. This reader
compares the unit's module closure three ways, most authoritative first:

1. the elaborated FIRRTL module bodies (exact bytes, including source locators);
2. the CIRCT HW-dialect modules (exact canonical operation text per module);
3. a structural HW form that ignores only ``sv.namehint`` -- a naming hint for emitted Verilog
   that carries no hardware meaning.

Top-level HW operations outside any module (non-local annotation anchors, macro declarations)
are reported separately with the module symbols they reference: an SoC-level anchor into a module
outside the unit is visible, never mistaken for unit structure, and never silently ignored. Any
other difference -- a module present on one side only, or a structural operation difference --
makes the result ``different``. The comparison names no target, module or configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "merlin.rtl_unit_correspondence.v1"
#: Attributes that only name emitted Verilog wires; everything else is structure.
NAME_HINT_ATTRIBUTES = frozenset({"sv.namehint"})
_FIRRTL_DECLARATIONS = ("module", "extmodule", "intmodule", "public module")


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def firrtl_module_bodies(firrtl: Path, names: set[str]) -> dict[str, str]:
    """Exact FIRRTL text of each selected module declaration (header line through its body)."""
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    with Path(firrtl).open(encoding="utf-8") as source:
        for line in source:
            top = line.startswith("  ") and not line.startswith("   ")
            if top:
                stripped = line.strip()
                keyword = next((kw for kw in _FIRRTL_DECLARATIONS if stripped.startswith(kw + " ")), None)
                if keyword is not None:
                    name = stripped[len(keyword) :].partition(":")[0].split()
                    current = name[0] if len(name) == 1 and name[0] in names else None
                    if current is not None:
                        if current in bodies:
                            raise ValueError(f"FIRRTL declares module {current!r} twice")
                        bodies[current] = []
                elif not line.startswith("    "):
                    current = None
            if current is not None:
                bodies[current].append(line)
    return {name: "".join(lines) for name, lines in bodies.items()}


def _op_name(op) -> str:
    return op.attributes["op_name__"].data if op.name == "builtin.unregistered" else op.name


def _is_location(value) -> bool:
    """A source location, or a nonempty array of them (``result_locs``/``port_locs``).

    Locations record where the PRINTED file put an operation (path and line), which differs between
    two bundles of identical hardware; they are provenance, never structure.
    """
    from typing import get_args

    from xdsl.dialects.builtin import ArrayAttr, LocationAttr

    if isinstance(value, get_args(LocationAttr)):
        return True
    return isinstance(value, ArrayAttr) and bool(value.data) and all(_is_location(item) for item in value.data)


def canonical_lines(op, *, ignore: frozenset[str] = frozenset()) -> list[str]:
    """One line per nested operation: name, operand/result ids, types, attributes and properties.

    SSA values are numbered by definition order over the whole operation (graph regions may use a
    value before its definition), so value spelling never matters and value identity always does.
    Location-valued attributes are dropped (:func:`_is_location`); ``ignore`` drops named ones.
    """
    ids: dict[Any, int] = {}
    nested = list(op.walk())
    for inner in nested:
        for region in inner.regions:
            for block in region.blocks:
                for argument in block.args:
                    ids[argument] = len(ids)
        for result in inner.results:
            ids[result] = len(ids)
    lines = []
    for inner in nested:
        attributes = sorted(
            (key, str(value))
            for key, value in inner.attributes.items()
            if key != "op_name__" and key not in ignore and not _is_location(value)
        )
        properties = sorted(
            (key, str(value))
            for key, value in inner.properties.items()
            if key not in ignore and not _is_location(value)
        )
        operands = [ids.get(value, -1) for value in inner.operands]
        if -1 in operands:
            raise ValueError(f"{_op_name(inner)} uses a value defined outside the compared operation")
        lines.append(
            json.dumps(
                [
                    _op_name(inner),
                    operands,
                    [str(result.type) for result in inner.results],
                    [
                        [str(argument.type) for argument in block.args]
                        for region in inner.regions
                        for block in region.blocks
                    ],
                    attributes,
                    properties,
                ],
                separators=(",", ":"),
            )
        )
    return lines


def _symbols(text: str) -> list[str]:
    from .extract_module import _read_symbol

    found, index = [], text.find("@")
    while index >= 0:
        read = _read_symbol(text, index)
        if read is not None and read[0] not in found:
            found.append(read[0])
        index = text.find("@", index + 1)
    return found


def hw_unit_view(generic_text: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """``({module: {exact, structural, lines}}, [file-level op records])`` of a generic HW file."""
    from .hw_graph import parse_generic_hw
    from .hw_observations import _module_name

    parsed = parse_generic_hw(generic_text)
    modules: dict[str, dict[str, Any]] = {}
    file_level: list[dict[str, Any]] = []
    for op in parsed.body.block.ops:
        name = _op_name(op)
        if name in {"hw.module", "hw.module.extern"}:
            symbol = _module_name(op)
            if symbol in modules:
                raise ValueError(f"HW source defines module {symbol!r} twice")
            exact = canonical_lines(op)
            structural = canonical_lines(op, ignore=NAME_HINT_ATTRIBUTES)
            modules[symbol] = {
                "exact": _sha("\n".join(exact)),
                "structural": _sha("\n".join(structural)),
                "lines": exact,
                "structural_lines": structural,
            }
        else:
            text = "\n".join(canonical_lines(op))
            file_level.append({"operation": name, "sha256": _sha(text), "symbols": _symbols(text)})
    return modules, file_level


def _side(facts: dict, label: str) -> dict[str, Any]:
    """The comparable identity of one facts document's selected sources."""
    from .source_selection import digest, load_selection

    inputs, consistency = facts.get("inputs") or {}, facts.get("source_consistency") or {}
    if consistency.get("status") != "verified":
        raise ValueError(f"{label} facts do not carry verified source consistency")
    bundle = inputs.get("source_bundle_path")
    if not isinstance(bundle, str):
        raise ValueError(f"{label} facts were not extracted from an explicit source bundle")
    selection = load_selection(bundle)
    if digest(bundle) != inputs.get("source_bundle_sha256"):
        raise ValueError(f"{label} source bundle bytes differ from those its facts record")
    generic = (consistency.get("genericization") or {}).get("output") or {}
    if not generic.get("path") or digest(generic["path"]) != generic.get("sha256"):
        raise ValueError(f"{label} generic HW bytes differ from their genericization receipt")
    production = selection.get("production") or {}
    return {
        "config": selection.get("config"),
        "generator": selection.get("generator"),
        "core_root": production.get("core_root"),
        "selection": {"path": selection["selection_path"], "sha256": selection["selection_sha256"]},
        "firrtl": selection["sources"]["firrtl"],
        "hierarchy": selection["sources"]["hierarchy"],
        "soc_hw": selection["sources"]["soc_hw"],
        "core_hw": selection["sources"]["core_hw"],
        "core_hw_generic": {"path": generic["path"], "sha256": generic["sha256"]},
        "firtool": production.get("tool"),
        "included_modules": production.get("included_modules") or [],
    }


def compare_unit(target_facts: dict, simulation_facts: dict) -> dict[str, Any]:
    """Compare the accelerator unit closure of a target and a simulation configuration."""
    target, simulation = _side(target_facts, "target"), _side(simulation_facts, "simulation")
    if target["core_root"] != simulation["core_root"] or not target["core_root"]:
        raise ValueError("compared configurations must select the same declared core root")
    left_names, right_names = set(target["included_modules"]), set(simulation["included_modules"])
    names = left_names & right_names
    module_set = {
        "target": len(left_names),
        "simulation": len(right_names),
        "only_target": sorted(left_names - right_names),
        "only_simulation": sorted(right_names - left_names),
        "identical": left_names == right_names,
    }
    left_fir = firrtl_module_bodies(Path(target["firrtl"]["path"]), names)
    right_fir = firrtl_module_bodies(Path(simulation["firrtl"]["path"]), names)
    fir_missing = sorted(names - (set(left_fir) & set(right_fir)))
    fir_differ = sorted(name for name in set(left_fir) & set(right_fir) if left_fir[name] != right_fir[name])
    left_hw, left_file = hw_unit_view(Path(target["core_hw_generic"]["path"]).read_text())
    right_hw, right_file = hw_unit_view(Path(simulation["core_hw_generic"]["path"]).read_text())
    hw_missing = sorted(names - (set(left_hw) & set(right_hw)))
    module_rows, structural_differ = [], []
    for name in sorted(names - set(hw_missing)):
        a, b = left_hw[name], right_hw[name]
        if a["exact"] == b["exact"]:
            continue
        if a["structural"] != b["structural"]:
            structural_differ.append(name)
            module_rows.append({"module": name, "kind": "structural", "evidence": "operation roster differs"})
            continue
        changed = [{"target": x, "simulation": y} for x, y in zip(a["lines"], b["lines"], strict=True) if x != y]
        module_rows.append(
            {
                "module": name,
                "kind": "name_hints_only",
                "operations": len(a["lines"]),
                "differing_operations": len(changed),
                "evidence": f"{len(changed)} operation(s) differ only in {sorted(NAME_HINT_ATTRIBUTES)}",
                "examples": changed[:3],
            }
        )
    left_keys = {row["sha256"]: row for row in left_file}
    right_keys = {row["sha256"]: row for row in right_file}

    def file_rows(keys, other):
        rows = []
        for key in sorted(set(keys) - set(other)):
            row = keys[key]
            inside = sorted(set(row["symbols"]) & names)
            rows.append({**row, "unit_modules_referenced": inside})
        return rows

    only_target, only_simulation = file_rows(left_keys, right_keys), file_rows(right_keys, left_keys)
    unit_anchored = [row for row in only_target + only_simulation if row["unit_modules_referenced"]]
    exact = (
        module_set["identical"]
        and not fir_missing
        and not fir_differ
        and not hw_missing
        and not module_rows
        and not only_target
        and not only_simulation
    )
    equivalent = (
        module_set["identical"]
        and not fir_missing
        and not fir_differ
        and not hw_missing
        and not structural_differ
        and not unit_anchored
    )
    return {
        "schema": SCHEMA,
        "status": "identical" if exact else ("structurally_equivalent" if equivalent else "different"),
        "unit_root": target["core_root"],
        "configs": {"target": _public(target), "simulation": _public(simulation)},
        "module_set": module_set,
        "firrtl_module_bodies": {
            "compared": len(names) - len(fir_missing),
            "missing": fir_missing,
            "differ": fir_differ,
            "basis": "exact elaborated FIRRTL text of each unit module declaration",
        },
        "hw_modules": {
            "compared": len(names) - len(hw_missing),
            "missing": hw_missing,
            "exact_identical": len(names) - len(hw_missing) - len(module_rows),
            "structurally_identical": len(names) - len(hw_missing) - len(structural_differ),
            "differences": module_rows,
            "ignored_attributes": sorted(NAME_HINT_ATTRIBUTES),
            "basis": "canonical generic HW operation text per module; SSA spelling-independent",
        },
        "hw_file_level_operations": {
            "only_target": only_target,
            "only_simulation": only_simulation,
            "anchored_in_unit": unit_anchored,
            "basis": "top-level non-module operations; those referencing no unit module are SoC-level",
        },
        "interpretation": (
            "identical/structurally_equivalent: the unit elaborates to the same hardware under both "
            "configurations, so unit-internal structure and fixed latencies transfer; SoC-level timing "
            "(memory system, clocks, bridges) is not compared here and must be qualified separately"
        ),
    }


#: Correspondence results under which one configuration's unit stands for the other's.
EQUIVALENT = frozenset({"identical", "structurally_equivalent"})


class ExecutionBindingError(ValueError):
    """Facts that cannot bind an execution of the requested configuration."""


def _hex(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def facts_config(document: dict) -> str:
    """The one configuration a facts document describes; every place it is declared must agree."""
    body = document.get("facts") if isinstance(document, dict) else None
    source = body.get("source") if isinstance(body, dict) else None
    consistency = document.get("source_consistency") if isinstance(document, dict) else None
    if not isinstance(source, dict) or not isinstance(consistency, dict):
        raise ExecutionBindingError("facts omit their source configuration binding")
    configs = [source.get("config"), consistency.get("config")]
    if "source" in document:
        if not isinstance(document["source"], dict):
            raise ExecutionBindingError("facts have an ambiguous source configuration")
        configs.append(document["source"].get("config"))
    if any(not isinstance(config, str) or not config for config in configs) or len(set(configs)) != 1:
        raise ExecutionBindingError("facts declare no single source configuration")
    return configs[0]


def execution_binding(document: dict, *, config: str) -> dict[str, Any]:
    """How facts of one configuration may bind an execution of ``config``.

    Same configuration: the facts' own FIRRTL/hierarchy/core identity. Another configuration only
    through the facts' own ``simulation_correspondence``: its status must be equivalent, its target
    side must be exactly these facts' source (configuration, FIRRTL, hierarchy, core HW and source
    bundle), and its simulation side must be ``config`` with complete hashes. The returned
    simulation identity is what an executed engine or bundle must then match; everything else
    refuses. The record's equivalence result is data checked for binding, not recomputed here.
    """
    if not isinstance(config, str) or not config:
        raise ExecutionBindingError("execution binding requires an explicit configuration")
    own = facts_config(document)
    inputs = document.get("inputs") if isinstance(document.get("inputs"), dict) else {}
    source = document["facts"]["source"]
    if own == config:
        return {
            "basis": "same_configuration",
            "config": config,
            "facts_config": own,
            "firrtl": {"path": inputs.get("fir_path"), "sha256": inputs.get("fir_sha256")},
            "hierarchy": {"path": source.get("hierarchy_path"), "sha256": source.get("hierarchy_sha256")},
            "core_hw": {"path": inputs.get("core_hw_path"), "sha256": inputs.get("core_hw_sha256")},
        }
    record = document["facts"].get("simulation_correspondence")
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ExecutionBindingError(f"facts of {own} carry no simulation correspondence for {config}")
    if record.get("status") not in EQUIVALENT:
        raise ExecutionBindingError(f"facts of {own} do not correspond to {config}: {record.get('status')}")
    sides = record.get("configs") if isinstance(record.get("configs"), dict) else {}
    target, simulation = sides.get("target"), sides.get("simulation")
    if not isinstance(target, dict) or not isinstance(simulation, dict):
        raise ExecutionBindingError("simulation correspondence omits a configuration side")

    def sha(side: dict, role: str) -> str | None:
        member = side.get(role)
        return member.get("sha256") if isinstance(member, dict) else None

    own_identity = {
        "firrtl": inputs.get("fir_sha256"),
        "hierarchy": source.get("hierarchy_sha256"),
        "core_hw": inputs.get("core_hw_sha256"),
        "selection": inputs.get("source_bundle_sha256"),
    }
    if (
        target.get("config") != own
        or source.get("fir_sha256") != own_identity["firrtl"]
        or any(not _hex(value) or sha(target, role) != value for role, value in own_identity.items())
    ):
        raise ExecutionBindingError("simulation correspondence target side is not these facts' own source")
    if simulation.get("config") != config:
        raise ExecutionBindingError(f"simulation correspondence binds {simulation.get('config')}, not {config}")
    if target.get("core_root") != record.get("unit_root") or simulation.get("core_root") != record.get("unit_root"):
        raise ExecutionBindingError("simulation correspondence compares different unit roots")
    if any(not _hex(sha(simulation, role)) for role in ("firrtl", "hierarchy", "core_hw", "selection")):
        raise ExecutionBindingError("simulation correspondence has an incomplete simulation identity")
    return {
        "basis": "simulation_correspondence",
        "status": record["status"],
        "config": config,
        "facts_config": own,
        "unit_root": record["unit_root"],
        **{role: dict(simulation[role]) for role in ("firrtl", "hierarchy", "core_hw", "selection")},
    }


def bound_firrtl_digests(document: dict) -> set[str]:
    """FIRRTL digests an engine built for these facts may carry: their own, or a verified simulation's.

    For readers that see an engine build receipt but no execution configuration. A simulation digest
    is admitted only through :func:`execution_binding` of the correspondence's own simulation side.
    """
    inputs = document.get("inputs") if isinstance(document, dict) else None
    own = inputs.get("fir_sha256") if isinstance(inputs, dict) else None
    digests = {own} if _hex(own) else set()
    record = ((document.get("facts") or {}) if isinstance(document, dict) else {}).get("simulation_correspondence")
    simulation = ((record or {}).get("configs") or {}).get("simulation") if isinstance(record, dict) else None
    if isinstance(simulation, dict) and isinstance(simulation.get("config"), str):
        try:
            digests.add(execution_binding(document, config=simulation["config"])["firrtl"]["sha256"])
        except ExecutionBindingError:
            pass
    return digests


def _public(side: dict) -> dict:
    return {key: value for key, value in side.items() if key != "included_modules"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-facts", required=True, help="facts extracted from the chosen target bundle")
    parser.add_argument("--simulation-facts", required=True, help="facts extracted from the simulation bundle")
    parser.add_argument("--out", required=True, help="fresh output path for the correspondence record")
    parser.add_argument("--facts-out", help="optional fresh copy of --target-facts with the record attached")
    args = parser.parse_args(argv)
    target = json.loads(Path(args.target_facts).read_text())
    simulation = json.loads(Path(args.simulation_facts).read_text())
    record = compare_unit(target, simulation)
    for path in (args.out, args.facts_out):
        if path is not None and Path(path).exists():
            parser.error(f"{path} exists; outputs are never overwritten")
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if args.facts_out:
        from .facts import write_facts_guarded

        target["facts"]["simulation_correspondence"] = record
        write_facts_guarded(Path(args.facts_out), target)
    print(record["status"], json.dumps(record["hw_modules"]["differences"])[:400])
    return 0 if record["status"] != "different" else 1


if __name__ == "__main__":
    raise SystemExit(main())
