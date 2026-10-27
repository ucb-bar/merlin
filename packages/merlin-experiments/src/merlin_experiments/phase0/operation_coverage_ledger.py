"""Trace each source operation and composition to its declaration, reference, test and evidence.

One row per ATen overload that a captured workload contains or that the spec-validation record
exercised, and one row per declared composition chain. Each row follows the Phase 0 coverage table:
source requirement -> declaration -> reference -> test identity -> required engine -> evidence state,
with a recorded justification. Every capsule records which rows it covers and what only it adds.

Omissions are listed, not dropped: an accelerator-admitted operation with neither a capsule nor
validation evidence, a declared composition with no capsule, an accelerator declaration nothing
exercises, and every unmodeled operation. The ledger is diagnostic data. It issues no release,
qualification or Phase 1 authority, and it never sets ``rtl_qualified``: only Phase 1 grading or an
operator FireSim run can establish that state.

The spec-validation record (``merlin.phase0.spec_validation.v1``) is read by schema name and digest
only, through :func:`read_spec_validation`. The ledger links to a ``source_requirement_ledger``
record by digest and leaves that ledger's historical schemas unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .component_generation import digest
from .composition_generation import coverage

SCHEMA = "merlin.phase0.operation_coverage_ledger.v1"
SPEC_VALIDATION_SCHEMA = "merlin.phase0.spec_validation.v1"
SOURCE_REQUIREMENT_LEDGER_PREFIX = "merlin.phase0.source_requirement_ledger."

PASSING_VERDICTS = frozenset({"agree_device", "agree_host", "no_work"})
FAILING_VERDICTS = frozenset(
    {"MISMATCH_admitted_not_device", "MISMATCH_device_not_admitted", "numeric_fail", "capture_unavailable"}
)
#: Evidence states, kept distinct. ``rtl_qualified`` is reserved; this ledger never produces it.
STATES = (
    "mismatch",
    "unmodeled",
    "unreviewed_declaration",
    "executed_spike",
    "executed_host",
    "generated",
    "not_evidenced",
    "out_of_scope",
    "rtl_qualified",
)
_ACCELERATOR = frozenset({"accelerator", "fused_accelerator"})
_COVERAGE_WORKLOAD_PREFIX = "core_aten_coverage"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overload_name(name):
    """``aten.add.Tensor`` and ``add.Tensor`` name the same overload."""
    name = str(name)
    return name[len("aten.") :] if name.startswith("aten.") else name


def _base(overload):
    return overload.split(".", 1)[0]


def read_spec_validation(path, *, sha256=None):
    """Read a spec-validation record by schema name and digest only.

    The digest is the sha256 of the canonical JSON of every key except ``sha256``. A caller that
    pinned the record (for example from ``requirements.yaml``) passes that digest as ``sha256``.
    """
    record = json.loads(Path(path).read_bytes())
    if not isinstance(record, dict) or record.get("schema") != SPEC_VALIDATION_SCHEMA:
        raise ValueError(f"spec-validation evidence must declare schema {SPEC_VALIDATION_SCHEMA}")
    actual = digest({key: value for key, value in record.items() if key != "sha256"})
    if record.get("sha256") != actual:
        raise ValueError("spec-validation record bytes differ from its recorded digest")
    if sha256 is not None and sha256 != actual:
        raise ValueError("spec-validation record differs from the pinned digest")
    cases = record.get("cases")
    if not isinstance(cases, list) or any(
        not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case.get("overload")
        for case in cases
    ):
        raise ValueError("spec-validation record needs case rows with case_id and overload")
    if len({(case.get("source"), case["case_id"]) for case in cases}) != len(cases):
        raise ValueError("spec-validation record repeats a (source, case_id)")
    return record


def _placement(row):
    return row.get("placement") or ("accelerator" if "hardware" in row else "unknown")


def _reviewed(spec, row):
    from merlin.targetgen.operator_review import review_accepted

    return review_accepted(spec, default="unreviewed") and review_accepted(row)


def _candidates(spec, overload):
    """Declaration rows that could admit ``overload`` without an observed signature.

    Mirrors ``software_spec.admit_operation``'s selection (an explicit ``ops`` roster first, then
    family rows), but cannot screen the signature, so it reports candidates, not admission.
    """
    from merlin.targetgen.semantic_families import FAMILIES, from_op

    rows = spec.get("operations") or []
    base = _base(overload)
    direct = [row for row in rows if {overload, base} & set(row.get("ops") or [])]
    family = from_op(base)
    by_family = [
        row
        for row in rows
        if not row.get("ops")
        and family is not None
        and (
            family in (row.get("families") or [])
            or family == row.get("family")
            or (row.get("id") in FAMILIES and family == row["id"])
        )
    ]
    return family, direct or by_family


def _validation_rows(record):
    by_overload = {}
    for case in record["cases"]:
        by_overload.setdefault(overload_name(case["overload"]), []).append(case)
    return by_overload


def _unmodeled(record):
    return {
        overload_name(item.get("overload") if isinstance(item, dict) else item)
        for item in record.get("unmodeled") or []
    }


def _declarations_from_cases(cases):
    """Admitting rows the record cites, with their authored source lines."""
    found = {}
    for case in cases:
        for step in (case.get("spec_expectation") or {}).get("steps") or []:
            identity = step.get("declaration_id")
            if not identity:
                continue
            row = found.setdefault(identity, {"id": identity, "families": set(), "sources": set(), "statuses": set()})
            if step.get("family"):
                row["families"].add(step["family"])
            source = step.get("source") or {}
            if source.get("path"):
                row["sources"].add(f"{source['path']}:{source.get('line')}")
            if step.get("status"):
                row["statuses"].add(str(step["status"]))
    return {
        key: {name: sorted(value) if isinstance(value, set) else value for name, value in row.items()}
        for key, row in sorted(found.items())
    }


def _capsule_family(entry):
    from merlin.targetgen.semantic_families import from_op

    return entry.get("family") or from_op(entry.get("op"))


def _capsule_engine(entry):
    lanes = entry.get("lanes") or {}
    if "on_mesh" in (lanes.get("forbid") or ()):
        return "host"
    return "accelerator" if "on_mesh" in (lanes.get("require") or ()) else "unspecified"


def _capsule_coverage(entries, *, traces, chains):
    """What each capsule exercises: overloads (whole programs), families (cells) or a composition."""
    covered = {}
    for entry in entries:
        name = str(entry.get("name") or "")
        if not name:
            raise ValueError("coverage ledger needs named capsule entries")
        if name in covered:
            raise ValueError(f"coverage ledger capsule {name!r} is repeated")
        keys = set()
        model = entry.get("model")
        if entry.get("op") == "model" and model in traces:
            keys |= {f"overload:{op}" for op in traces[model]}
        elif entry.get("op") == "composition_program":
            semantic = entry.get("generalization") or {}
            validation = semantic.get("fusion_validation")
            if validation is not None:
                # Campaign identities are hashes; declared chains are retained as display aliases.
                aliases = validation.get("display_aliases") or []
                keys |= {f"composition:{chain}" for chain in chains if chain in aliases}
            else:
                keys |= {f"composition:{chain}" for chain in chains if name.startswith(f"SY_composition_{chain}_")}
        else:
            family = _capsule_family(entry)
            if family:
                keys.add(f"family:{family}")
        covered[name] = keys
    return covered


def _capsule_rows(entries, covered):
    holders = Counter(key for keys in covered.values() for key in keys)
    rows = []
    for entry in sorted(entries, key=lambda item: str(item["name"])):
        name = str(entry["name"])
        keys = sorted(covered[name])
        unique = sorted(key for key in keys if holders[key] == 1)
        if not keys:
            justification = None
        else:
            what = ", ".join(unique) if unique else "nothing no other capsule covers"
            justification = f"{entry.get('source_role') or 'unspecified role'}: covers {len(keys)} item(s); adds {what}"
            if entry.get("source_reference"):
                justification += f" ({entry['source_reference']})"
        rows.append(
            {
                "name": name,
                "source_role": entry.get("source_role"),
                "op": entry.get("op"),
                "operand_dtype": entry.get("operand_dtype"),
                "required_engine": _capsule_engine(entry),
                "covers": keys,
                "adds": unique,
                "justification": justification,
            }
        )
    return rows


def _state(*, cases, unmodeled, admitting_rows, spec, direct, family_capsules, engine):
    verdicts = Counter(str(case.get("verdict")) for case in cases)
    if any(verdict not in PASSING_VERDICTS for verdict in verdicts):
        return "mismatch"
    if unmodeled or (not cases and not admitting_rows):
        return "unmodeled"
    if engine == "accelerator" and admitting_rows and not any(_reviewed(spec, row) for row in admitting_rows):
        return "unreviewed_declaration"
    if verdicts["agree_device"]:
        return "executed_spike"
    if cases:
        return "executed_host"
    if direct or family_capsules:
        return "generated"
    return "not_evidenced"


def _justify(state, *, overload, family, engine, declarations, direct, family_capsules, cases, demanded):
    ids = ", ".join(declarations) or "no row"
    if state == "mismatch":
        return f"mismatch: spec validation reports failing cases for {overload} against {ids}"
    if state == "unmodeled":
        return f"unmodeled: no declaration admits {overload}; listed, not dropped"
    if state == "unreviewed_declaration":
        return f"unreviewed: {ids} admit {overload} for the accelerator but carry no operator review"
    if engine == "host":
        return f"host: no admitting accelerator row for {overload} ({ids}); host fallback evidence only"
    if direct:
        return f"admitted family {family}; Phase 1 capsule(s) {', '.join(direct)} exercise {overload}"
    if family_capsules and demanded:
        return f"admitted family {family}; family cell capsule(s) {', '.join(family_capsules)} exercise its row"
    if cases:
        reason = "absent from captured workloads" if not demanded else "no capsule contains it"
        return f"evidence-only: {reason}; {len(cases)} spec-validation case(s) cover {overload}"
    if family_capsules:
        return f"admitted family {family}; only family cell capsule(s) {', '.join(family_capsules)}; no case"
    return f"omission: {ids} admit {overload} but no capsule and no spec-validation evidence cover it"


@dataclass(frozen=True)
class OperationCoverageLedger:
    """Immutable diagnostic data; no issuer or conversion to release authority."""

    document_json: str

    def record(self):
        return json.loads(self.document_json)


def prepare(
    *,
    spec_validation,
    software_spec,
    capsules,
    captures=None,
    spec_validation_sha256=None,
    source_requirement_ledger=None,
    target=None,
    descriptor=None,
    workload_spec=None,
):
    """Join validation evidence, the selected spec, derived capsules and captures into one ledger.

    ``spec_validation`` and ``software_spec`` are paths; ``capsules`` are the selected manifest
    entries; ``captures`` maps an application label to its captured ``model.mlir``.
    ``descriptor`` is a mapping or YAML path; an explicit ``workload_spec`` takes precedence.
    Composition synthesis defaults to enabled when the workload spec omits the switch.
    """
    from merlin.targetgen.aten_coverage import aten_ops_in
    from merlin.targetgen.composition_corpus import synthesize_compositions_enabled
    from merlin.targetgen.software_spec import load_software_spec, operation_domains

    if workload_spec is None and descriptor is not None:
        if isinstance(descriptor, (str, Path)):
            import yaml

            descriptor = yaml.safe_load(Path(descriptor).read_bytes()) or {}
        workload_spec = descriptor.get("workload_spec")

    record = read_spec_validation(spec_validation, sha256=spec_validation_sha256)
    spec_sha = _sha256(software_spec)
    if record.get("software_spec_sha256") not in (None, spec_sha):
        raise ValueError("spec-validation record was produced for a different software spec")
    spec = load_software_spec(software_spec, target=target)
    domains = operation_domains(spec)
    rows_by_id = {row["id"]: row for row in spec["operations"]}
    chains = [chain["id"] for chain in (spec.get("compositions") or {}).get("chains") or []]

    captures = dict(captures or {})
    traces, capture_pins = {}, {}
    for label in sorted(captures):
        path = Path(captures[label])
        traces[label] = sorted({overload_name(op) for op in aten_ops_in(path)})
        capture_pins[label] = {"path": str(path), "sha256": _sha256(path)}
    entries = [copy.deepcopy(entry) for entry in capsules]
    composition_coverage = coverage(
        entries, excluded_spec=spec if not synthesize_compositions_enabled(workload_spec) else None
    )
    covered = _capsule_coverage(entries, traces=traces, chains=chains)
    capsule_rows = _capsule_rows(entries, covered)
    by_key = {}
    for name, keys in covered.items():
        for key in keys:
            by_key.setdefault(key, []).append(name)

    validation = _validation_rows(record)
    unmodeled = _unmodeled(record)
    demand = {}
    for label, ops in traces.items():
        for op in ops:
            demand.setdefault(op, []).append(label)
    overloads = sorted(set(validation) | set(demand) | unmodeled)

    operation_rows, exercised = [], set()
    for overload in overloads:
        cases = validation.get(overload, [])
        cited = _declarations_from_cases(cases)
        family, candidates = _candidates(spec, overload)
        if cited:
            declaration_status = "screened"
            admitting = [rows_by_id[key] for key in cited if key in rows_by_id]
            engines = {(case.get("spec_expectation") or {}).get("placement") for case in cases}
            engine = "accelerator" if engines & _ACCELERATOR else "host"
            family = next((name for row in cited.values() for name in row["families"]), family)
        elif cases:
            declaration_status, admitting, engine = "unsupported", [], "host"
        else:
            admitting = candidates
            declaration_status = "candidate_only" if candidates else "unsupported"
            engine = "accelerator" if any(_placement(row) in _ACCELERATOR for row in candidates) else "host"
        admitting_ids = [row["id"] for row in admitting]
        if engine == "accelerator":
            exercised.update(admitting_ids)
        labels = sorted(demand.get(overload, []))
        direct = sorted(by_key.get(f"overload:{overload}", []))
        family_capsules = sorted(by_key.get(f"family:{family}", [])) if family and engine == "accelerator" else []
        state = _state(
            cases=cases,
            unmodeled=overload in unmodeled,
            admitting_rows=admitting,
            spec=spec,
            direct=direct,
            family_capsules=family_capsules,
            engine=engine,
        )
        verdicts = dict(sorted(Counter(str(case.get("verdict")) for case in cases).items()))
        operation_rows.append(
            {
                "kind": "operation",
                "overload": overload,
                "source_requirement": {
                    "demanded_by": [
                        ("coverage_workload:" if label.startswith(_COVERAGE_WORKLOAD_PREFIX) else "workload:") + label
                        for label in labels
                    ]
                    or ["core_aten_only"],
                },
                "declaration": {
                    "status": declaration_status,
                    "family": family,
                    "rows": [
                        {
                            "id": row["id"],
                            "placement": _placement(row),
                            "reviewed": _reviewed(spec, row),
                            "sources": cited.get(row["id"], {}).get("sources", []),
                            "operation_domain": domains.get(row["id"]),
                        }
                        for row in admitting
                    ],
                    "cited_unknown_rows": sorted(set(cited) - set(rows_by_id)),
                },
                "reference": {
                    "spec_validation": {"record_sha256": record["sha256"], "cases": len(cases)} if cases else None,
                    "capsules": "capture golden of each listed capsule" if direct or family_capsules else None,
                },
                "test_identity": {
                    "validation_cases": sorted(case["case_id"] for case in cases),
                    "validation_verdicts": verdicts,
                    "capsules": direct,
                    "family_capsules": family_capsules,
                },
                "required_engine": engine,
                "evidence_state": state,
                "justification": _justify(
                    state,
                    overload=overload,
                    family=family,
                    engine=engine,
                    declarations=admitting_ids,
                    direct=direct,
                    family_capsules=family_capsules,
                    cases=cases,
                    demanded=bool(labels),
                ),
            }
        )

    composition_rows = []
    for chain in chains:
        names = sorted(by_key.get(f"composition:{chain}", []))
        chain_coverage = [row for row in composition_coverage if row["composition"] == chain]
        excluded = next((row for row in chain_coverage if row.get("status") == "out_of_scope"), None)
        if excluded is not None:
            state = excluded["status"]
            justification = excluded["reason"]
        elif names or any(row["count"] for row in chain_coverage):
            state = "generated"
            justification = f"declared chain {chain}; capsule(s) {', '.join(names)} exercise it"
        else:
            state = "not_evidenced"
            justification = f"omission: declared composition {chain} has no capsule"
        composition_rows.append(
            {
                "kind": "composition",
                "composition": chain,
                "source_requirement": {"demanded_by": ["declared_composition_axis"]},
                "declaration": {"status": "declared", "rows": [{"id": chain}]},
                "reference": {"capsules": "capture golden of each listed capsule" if names else None},
                "test_identity": {"capsules": names},
                "required_engine": "accelerator",
                "evidence_state": state,
                "justification": justification,
            }
        )

    linked = None
    if source_requirement_ledger is not None:
        linked = source_requirement_ledger
        if hasattr(linked, "record"):
            linked = linked.record()
        if not str(linked.get("schema", "")).startswith(SOURCE_REQUIREMENT_LEDGER_PREFIX) or digest(
            {key: value for key, value in linked.items() if key != "sha256"}
        ) != linked.get("sha256"):
            raise ValueError("linked source requirement ledger is not an intact ledger record")
        linked = {"schema": linked["schema"], "sha256": linked["sha256"]}

    omissions = {
        "admitted_without_evidence": [
            row["overload"]
            for row in operation_rows
            if row["required_engine"] == "accelerator" and row["evidence_state"] == "not_evidenced"
        ],
        "compositions_without_capsule": [
            row["composition"] for row in composition_rows if row["evidence_state"] == "not_evidenced"
        ],
        "accelerator_declarations_unexercised": sorted(
            row["id"]
            for row in spec["operations"]
            if _placement(row) in _ACCELERATOR
            and row["id"] not in exercised
            and not any(f"family:{name}" in by_key for name in (row.get("families") or [row.get("family")]))
        ),
        "unmodeled": [row["overload"] for row in operation_rows if row["evidence_state"] == "unmodeled"],
        "mismatches": [row["overload"] for row in operation_rows if row["evidence_state"] == "mismatch"],
        "unjustified_capsules": [row["name"] for row in capsule_rows if row["justification"] is None],
    }
    blocking = ("admitted_without_evidence", "compositions_without_capsule", "mismatches", "unjustified_capsules")
    document = {
        "schema": SCHEMA,
        "spec_validation": {
            "schema": SPEC_VALIDATION_SCHEMA,
            "sha256": record["sha256"],
            "verdict": record.get("verdict"),
        },
        "software_spec": {"path": str(software_spec), "sha256": spec_sha, "target": spec["target"]},
        "captures": capture_pins,
        "capsule_manifest_sha256": digest(entries),
        "source_requirement_ledger": linked,
        "states": list(STATES),
        "operations": operation_rows,
        "compositions": composition_rows,
        "capsules": capsule_rows,
        "state_counts": dict(sorted(Counter(row["evidence_state"] for row in operation_rows).items())),
        "omissions": omissions,
        "status": "diagnostic_incomplete" if any(omissions[key] for key in blocking) else "diagnostic_traced",
        "release_authority": "not_issued",
        "scope": "operation/composition coverage trace; never RTL or Phase 1 qualification",
    }
    document["sha256"] = digest(document)
    return OperationCoverageLedger(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False))


def verify(ledger, **inputs):
    """Recompute from the same inputs; a stored ledger cannot stand in for its evidence."""
    if type(ledger) is not OperationCoverageLedger:
        raise ValueError("operation coverage comparison needs the exact ledger data type")
    actual = prepare(**inputs)
    if actual.record() != ledger.record():
        raise ValueError("operation coverage inputs, evidence or denominator changed")
    return actual.record()
