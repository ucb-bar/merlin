"""Retain actual ordinary stage observations without issuing semantic roles.

The selected source producer owns its finite effect judgments. This consumer
only joins its actually invoked products to the context's original execution.
Reported dispositions are diagnostics, never stage witnesses or runtime grants.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record as I
from merlin_experiments.phase1.component_witness import REQUIRED_EXECUTION_EFFECTS

from .component_decode_products import join_component_decode_products
from .contracts import StageGateError, exact_tree_record, mapping_file, sha256_file

EMISSION_FACETS = ("prepared_ir", "partition", "emitted_host", "emitted_device")
UNKNOWN = (
    "complete_independent_execution_effects",
    "original_fourteen_runtime_controls",
    "observer_integrity",
    "resource_effects",
    "physical_execution_domain",
    "physical_timing",
)


def member(path, *, owner=None):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
        or owner is not None
        and not path.is_relative_to(owner)
    ):
        raise StageGateError("stage diagnostic requires canonical admitted membership before reading")
    return {"path": str(path), "sha256": sha256_file(path)}


def reopen(pin, *, owner=None):
    if type(pin) is not dict or set(pin) != {"path", "sha256"} or type(pin["path"]) is not str:
        raise StageGateError("stage diagnostic has no exact original file membership")
    if member(pin["path"], owner=owner) != pin:
        raise StageGateError("stage diagnostic product changed")
    return Path(pin["path"])


def collect(*, ordinary_result, capsule, candidate, target_descriptor, coherent, grade_owner):
    ordinary_result, capsule, candidate = Path(ordinary_result), Path(capsule), Path(candidate)
    root = ordinary_result.parent
    grade_owner = Path(grade_owner)
    if (
        not grade_owner.is_absolute()
        or grade_owner.resolve() != grade_owner
        or any(part.is_symlink() for part in (grade_owner, *grade_owner.parents))
        or root.parent != grade_owner
    ):
        raise StageGateError("ordinary products require the actual canonical grade owner")
    result_pin = member(ordinary_result, owner=root)
    source_pin = member(capsule / "source.mlir", owner=capsule)
    result = mapping_file(ordinary_result)
    if type(result.get("numeric_report")) is not dict:
        raise StageGateError("ordinary stage observation omitted its original numerical report")
    records = []
    for path in root.rglob("invocation.json"):
        pin = member(path, owner=root)
        record = I.verify(path)
        records.append((pin, record))
    sources = [row for _, row in records if row["stage"] == "primitive_source_verification"]
    if len(sources) != 1 or source_pin not in sources[0]["inputs"]:
        raise StageGateError("stage observation omitted the actual original source verification")
    decoded = join_component_decode_products(result_path=ordinary_result, execution_root=root) if coherent else None
    return {
        "ordinary_result": result_pin,
        "grade_owner": str(grade_owner),
        "original_source": source_pin,
        "original_candidate_root": str(candidate),
        "original_candidate": exact_tree_record(candidate),
        "target_descriptor": member(target_descriptor),
        "invocations": [pin for pin, _ in records],
        "coherent_products": decoded.record() if decoded is not None else None,
        "required_emission_facets": list(EMISSION_FACETS),
        "required_execution_effects": list(REQUIRED_EXECUTION_EFFECTS),
        "unknown": list(UNKNOWN),
        "scope": "actual context-owned source/execution products only; no stage or runtime authority",
    }


def verify_collected(observation):
    if (
        observation.get("required_emission_facets") != list(EMISSION_FACETS)
        or observation.get("required_execution_effects") != list(REQUIRED_EXECUTION_EFFECTS)
        or observation.get("unknown") != list(UNKNOWN)
    ):
        raise StageGateError("stage diagnostic original denominator or unknown scope changed")
    root = Path(observation["ordinary_result"]["path"]).parent
    grade_owner = Path(observation["grade_owner"])
    if (
        root.parent != grade_owner
        or grade_owner.resolve() != grade_owner
        or any(part.is_symlink() for part in (grade_owner, *grade_owner.parents))
    ):
        raise StageGateError("ordinary product grade ownership changed")
    for name in ("ordinary_result", "original_source", "target_descriptor"):
        reopen(observation[name])
    candidate = observation["original_candidate"]
    if exact_tree_record(Path(observation["original_candidate_root"])) != candidate:
        raise StageGateError("stage observation original compiler changed")
    for pin in observation["invocations"]:
        I.verify(reopen(pin))
    coherent = observation["coherent_products"]
    if coherent is not None:
        actual = join_component_decode_products(
            result_path=Path(observation["ordinary_result"]["path"]),
            execution_root=Path(coherent["execution_root"]),
        ).record()
        if coherent != actual:
            raise StageGateError("stage observation coherent execution join changed")


def _facets(rows, names):
    if type(rows) is not list or len(rows) != len(names):
        raise StageGateError("finite observation omitted the original facet denominator")
    index = {}
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != {"name", "outcome", "observation"}
            or type(row["name"]) is not str
            or type(row["outcome"]) is not str
            or row["name"] not in names
            or row["name"] in index
            or row["outcome"] not in {"OBSERVED", "REFUTED", "PARTIAL", "UNKNOWN"}
            or type(row["observation"]) is not str
        ):
            raise StageGateError("finite observation has unsupported or duplicate diagnostic facets")
        index[row["name"]] = row
    return [index[name] for name in names]


def attach(*, collected, product_path, producer_record, context_sources):
    """Consume selected produced data only, without executing supplied code."""
    verify_collected(collected)
    root = Path(collected["ordinary_result"]["path"]).parent
    product_pin = member(product_path, owner=root)
    record_pin = member(producer_record, owner=Path(collected["grade_owner"]))
    produced = I.verify(Path(producer_record))
    allowed_sources = {str(path): digest for path, digest in context_sources}
    if (
        produced["kind"] != "python_call"
        or product_pin not in produced["outputs"]
        or produced["executable"]["path"] not in allowed_sources
        or any(allowed_sources.get(pin["path"]) != pin["sha256"] for pin in produced["dependencies"])
        or allowed_sources[produced["executable"]["path"]] != produced["executable"]["sha256"]
    ):
        raise StageGateError("finite observation lacks an actual selected source-owned producer")
    coherent = collected["coherent_products"]
    if coherent is None:
        raise StageGateError("finite effect attachment requires actual complete coherent products")
    required = (collected["original_source"], coherent["elf"], coherent["decoder_product"], coherent["payload"])
    if any(pin not in produced["inputs"] for pin in required):
        raise StageGateError("finite producer did not consume the exact original source/ELF/decoder/packet")
    data = mapping_file(Path(product_path))
    unresolved = data.get("unresolved_prerequisites")
    if type(unresolved) is not list or any(type(value) is not str for value in unresolved):
        raise StageGateError("finite observation omitted explicit unresolved diagnostic prerequisites")
    return {
        "product": product_pin,
        "producer": record_pin,
        "emission_facets": _facets(data.get("emission_facets"), EMISSION_FACETS),
        "execution_effects": _facets(data.get("execution_effects"), REQUIRED_EXECUTION_EFFECTS),
        "producer_unresolved": unresolved,
        "unknown": list(UNKNOWN),
        "scope": "selected producer's finite reported diagnostics; no semantic witness or qualification",
    }


def write_refusal_report(*, diagnostics, products, finite, required_controls):
    """Keep the complete original refusal denominator in an exclusive product."""
    verify_collected(products)
    with Path(diagnostics).open("x", encoding="utf-8") as output:
        json.dump(
            {
                "products": products,
                "finite_observations": finite,
                "unknown": list(UNKNOWN),
                "required_controls": list(required_controls),
                "scope": "retained actual ordinary products and finite reports; no stage/runtime authority",
            },
            output,
            sort_keys=True,
            indent=2,
        )
        output.write("\n")
