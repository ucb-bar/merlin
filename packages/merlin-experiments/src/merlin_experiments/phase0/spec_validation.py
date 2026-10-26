"""Phase 0 spec validation: Core ATen cases as sealed evidence, never capsules.

The canonical 193 and bounded 10,160 Core ATen cases validate the selected
software spec against the reference backend. Each case contributes one row:

* the spec expectation, screened through the shared source-operation admission
  against the fact-resolved selected software spec (``admit_operation``, with
  the authored ``path:line`` of the deciding row);
* the observed lane, retired device instructions and device receipt;
* the functional result, including the native host floor; and
* a verdict that fails a mismatch in either direction.

The record is evidence for derivation (bound by digest into ``requirements.yaml``)
and the operation coverage ledger. It is never a corpus, contributes no capsule and
is never Phase 1 grading load. The functional-truth table is an optional audit
column that never gates.

The record consumes the per-case shard results that
``build_tools/scripts/run_bounded_core_aten_suite.py`` writes under
``<output>/<label>/shards/<capture name>/`` (``result.json`` plus the
``spec-int8-admission.json`` preparation plan), and the runner's
``provenance.json`` stamp. Expectations and execution use the component host
capabilities; derivation binds the descriptor host capabilities. Both document
paths and byte digests are sealed, and verification reopens the component document.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "merlin.phase0.spec_validation.v1"
SELECTION_SCHEMA = "merlin.phase0.spec_validation_selection.v1"
NUMERIC_POLICY = "spec_int8"
DEFAULT_QUANTIZATION_SCHEME = "q_pow2_tensor"

AGREE_DEVICE = "agree_device"
AGREE_HOST = "agree_host"
NO_WORK = "no_work"
ADMITTED_NOT_DEVICE = "MISMATCH_admitted_not_device"
DEVICE_NOT_ADMITTED = "MISMATCH_device_not_admitted"
NUMERIC_FAIL = "numeric_fail"
CAPTURE_UNAVAILABLE = "capture_unavailable"
NOT_RUN = "not_run"

PASSING = frozenset({AGREE_DEVICE, AGREE_HOST, NO_WORK})
FAILING = frozenset({ADMITTED_NOT_DEVICE, DEVICE_NOT_ADMITTED, NUMERIC_FAIL, CAPTURE_UNAVAILABLE})
VERDICTS = PASSING | FAILING | {NOT_RUN}
RECEIPT_POLICIES = ("execution_evidence", "spec_qualified")
#: The capture's typed refusal for a floating reference that is itself NaN or infinite.
NONFINITE_REFERENCE = "nonfinite_reference"
_FULL_SCOPES = frozenset({"full", "values+mutation+observed-metadata"})


def canonical_json(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def digest(record: Mapping[str, Any]) -> str:
    """sha256 over the sorted-key compact JSON of every key except ``sha256``."""
    return hashlib.sha256(canonical_json({k: v for k, v in record.items() if k != "sha256"})).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path)}


def case_key(case: Mapping[str, Any]) -> str:
    return str(case.get("case_id") or case["overload"])


def case_sha256(case: Mapping[str, Any]) -> str:
    """The runner's case identity (``core_aten_bounded_runner.case_digest``)."""
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


def selection_identity(*, resolved_spec, rtl_facts, capability_contracts, host_capabilities) -> dict:
    """Bind semantic inputs independently of their filesystem locations."""
    from .provenance import _document_digest

    def documents(values):
        return sorted({_document_digest(value) for value in values})

    return {
        "resolved_software_spec_sha256": _document_digest(resolved_spec),
        "rtl_facts_sha256": _document_digest(rtl_facts),
        "capability_contracts_sha256": _document_digest(documents(capability_contracts)),
        "host_capabilities_sha256": _document_digest(documents(host_capabilities)),
    }


def evidence_identity(selected) -> dict:
    """Use the actual fact-resolved derivation/verification selection."""
    if selected.raw_facts is None:
        raise ValueError("spec-validation requires current RTL facts")
    hosts = selected.host_capabilities
    if not hosts or any(not isinstance(row.get("capability_spec"), dict) for row in hosts.values()):
        raise ValueError("spec-validation requires current host capability declarations")
    import yaml

    contracts = [selected.contract]
    contracts.extend(
        yaml.safe_load(source.content)
        for source in selected.source_snapshots
        if source.role in {"target-contract", "support-contract"}
    )
    return selection_identity(
        resolved_spec=selected.software_spec,
        rtl_facts=json.loads(selected.raw_facts),
        capability_contracts=contracts,
        host_capabilities=[row["capability_spec"] for row in hosts.values()],
    )


def _verify_input_identity(record, current) -> None:
    from merlin.common.digest import is_sha256

    keys = {
        "resolved_software_spec_sha256",
        "rtl_facts_sha256",
        "capability_contracts_sha256",
        "host_capabilities_sha256",
    }
    recorded = record.get("input_identity")
    recorded = recorded if isinstance(recorded, Mapping) else {}
    current = current if isinstance(current, Mapping) else {}
    changed = sorted(key for key in keys if not is_sha256(current.get(key)) or recorded.get(key) != current[key])
    if changed:
        raise ValueError(
            f"spec-validation inputs differ from the current selection ({', '.join(changed)}); "
            "rerun spec validation and rederive"
        )


# --------------------------------------------------------------------------------------------
# Selection


def _selected_suite(source: str, pin: Mapping[str, Any], path: Path) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin.get("file_sha256"):
        raise ValueError(f"{source}: suite file_sha256 mismatch")
    suite = json.loads(raw)
    rows = suite.get({"corpus_193": "cases", "bounded": "selected_cases"}[source])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}: suite needs a nonempty case list")
    keyed = {case_key(case): case for case in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"{source}: duplicate case identities in suite")
    wanted = pin.get("case_ids")
    if (
        not isinstance(wanted, list)
        or not wanted
        or any(not isinstance(key, str) or not key for key in wanted)
        or wanted != sorted(set(wanted))
        or set(wanted) - set(keyed)
        or pin.get("case_count") != len(wanted)
    ):
        raise ValueError(f"{source}: case_ids must be sorted, unique known case identities matching case_count")
    for key in ("denominator_sha256", "suite_sha256"):
        if pin.get(key) != suite.get(key):
            raise ValueError(f"{source}: {key} mismatch")
    return [{"source": source, "case": keyed[key]} for key in wanted]


def load_selection(path: str | Path, suites: Mapping[str, str | Path]) -> dict:
    """Resolve a v1 selection against explicitly supplied, byte-pinned suite files."""
    document = json.loads(Path(path).read_bytes())
    if not isinstance(document, dict) or document.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"{path}: not a {SELECTION_SCHEMA} selection")
    payload = {key: value for key, value in document.items() if key != "selection_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if document.get("selection_sha256") != hashlib.sha256(encoded.encode()).hexdigest():
        raise ValueError(f"{path}: selection_sha256 mismatch")
    selected = document.get("suites")
    if not isinstance(selected, dict) or not selected or set(selected) - {"corpus_193", "bounded"}:
        raise ValueError(f"{path}: selection needs corpus_193 and/or bounded suites")
    cases, sources = [], []
    for source, pin in sorted(selected.items()):
        if source not in suites:
            raise ValueError(f"{source}: suite file is required")
        suite_path = Path(suites[source])
        if not isinstance(pin, dict):
            raise ValueError(f"{source}: suite selection must be an object")
        rows = _selected_suite(source, pin, suite_path)
        cases.extend(rows)
        sources.append(
            {"source": source, "path": str(suite_path.resolve()), "sha256": pin["file_sha256"], "case_count": len(rows)}
        )
    if document.get("case_count") != len(cases):
        raise ValueError(f"{path}: selection case_count mismatch")
    return {"cases": cases, "sources": sources, "selection_sha256": document["selection_sha256"]}


def runner_case(source: str, case: Mapping[str, Any]) -> dict:
    """Namespace runner IDs so suite-local identities cannot overwrite shards."""
    return {**case, "case_id": f"{source}:{case_key(case)}"}


def runner_suite(selection: Mapping[str, Any]) -> dict:
    """The case list ``run_bounded_core_aten_suite.py`` executes, in selection order."""
    return {"selected_cases": [runner_case(row["source"], row["case"]) for row in selection["cases"]]}


# --------------------------------------------------------------------------------------------
# Spec expectation


def _location(decision: Mapping[str, Any]) -> dict | None:
    source = decision.get("source")
    if not isinstance(source, dict) or not source.get("path"):
        return None
    line = source.get("restriction_line") or source.get("line")
    return {"path": source["path"], "line": line}


def spec_expectation(directory: Path, case: Mapping[str, Any], admission: Mapping[str, Any]) -> dict:
    """Screen the source construction against the fact-resolved selected software spec."""
    from merlin.targetgen.core_aten_spec_int8 import declaration_location
    from merlin.targetgen.functional_truth import derive
    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    source = Case(case, "phase0")
    # The functional rules name the construction; only shared spec admission decides placement.
    decision = admit(source, case, derive(source).family, admission)
    if decision.get("declaration"):
        decision["source"] = declaration_location(admission, "software", decision["declaration"])
    path = directory / "spec-int8-admission.json"
    plan = json.loads(path.read_bytes()) if path.is_file() else {}
    native = plan.get("native_admission") or {}
    identities = (plan.get("selection") or {}).get("identities") or {}
    return {
        "placement": "accelerator" if decision["status"] == "admitted" else "host",
        "plan_status": plan.get("status"),
        "backend_reason": native.get("reason_code") or plan.get("refusal_code") or plan.get("host_reason"),
        "software_spec_sha256": (identities.get("software_spec") or {}).get("sha256"),
        "host_capabilities_sha256": (identities.get("host_capabilities") or {}).get("sha256"),
        "admission_sha256": _sha(path) if path.is_file() else None,
        "steps": [
            {
                "step": None,
                "authority": "software",
                "placement": "accelerator",
                "family": (decision.get("observed_signature") or {}).get("family"),
                "declaration_id": decision.get("declaration"),
                "source": _location(decision),
                "status": decision["status"],
                "reason": decision.get("reason"),
                "candidate": "selected_source",
            }
        ],
    }


# --------------------------------------------------------------------------------------------
# Observation and verdict


def _tensors(node: Any) -> Iterable[dict]:
    if isinstance(node, dict):
        if node.get("kind") == "tensor":
            yield node
            return
        for value in node.values():
            yield from _tensors(value)
    elif isinstance(node, list):
        for value in node:
            yield from _tensors(value)


def structural_no_work(case: Mapping[str, Any]) -> bool:
    """Metadata-only comparison, or every result tensor is empty; decided from the case alone."""
    if case.get("comparison") == "metadata":
        return True
    results = list(_tensors(case.get("expected")))
    return bool(results) and all(0 in (tensor.get("shape") or [1]) for tensor in results)


def _nonfinite_leaf(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("kind") == "float":
            return not math.isfinite(float(value["value"]))
        return any(_nonfinite_leaf(child) for child in value.values())
    if isinstance(value, list):
        return any(_nonfinite_leaf(child) for child in value)
    return isinstance(value, float) and not math.isfinite(value)


def nonfinite_reference(case: Mapping[str, Any]) -> bool:
    """Some recorded floating-point result is NaN or infinite; decided from the case alone."""
    return any(_nonfinite_leaf(tensor.get("values")) for tensor in _tensors(case.get("expected")))


def _case_dtype(case: Mapping[str, Any]) -> str | None:
    assigned = (case.get("partition_assignment") or {}).get("dtype")
    if assigned:
        return assigned
    first = next(iter(_tensors(case.get("arguments"))), None)
    return first.get("dtype") if first else None


def observed(result: Mapping[str, Any]) -> dict:
    observation = result.get("observation") or {}
    evidence = observation.get("evidence") or {}
    semantic = evidence.get("semantic_verdict") or {}
    routing = semantic.get("routing") or evidence.get("routing") or {}
    execution = routing.get("execution_evidence") if isinstance(routing.get("execution_evidence"), dict) else None
    count = execution.get("executed_instructions") if execution else 0
    floor = semantic.get("host_floor")
    receipt = evidence.get("spec_qualified_device")
    identity = evidence.get("full_call_execution_identity")
    return {
        "lane": evidence.get("lane", "unassigned"),
        "status": observation.get("status", "not_run"),
        "functional_status": semantic.get("status"),
        "semantic_scope": semantic.get("semantic_scope"),
        "numeric_status": semantic.get("numeric_status") or (evidence.get("numeric_verdict") or {}).get("status"),
        "numeric_execution_policy": semantic.get("numeric_execution_policy"),
        "executed_instructions": count if type(count) is int else 0,
        "execution_evidence": {
            key: execution.get(key)
            for key in ("target", "opcode", "executed_instructions", "trace_sha256", "elf_sha256", "facts_sha256")
        }
        if execution
        else None,
        "spec_qualified_device": receipt if isinstance(receipt, dict) else None,
        "full_call_execution_identity": identity if isinstance(identity, dict) else None,
        "host_floor": {"status": floor.get("status"), "comparison": floor.get("comparison")}
        if isinstance(floor, dict)
        else None,
        "reason": observation.get("reason"),
    }


def _functional_pass(row: Mapping[str, Any]) -> bool:
    if row["functional_status"] != "pass" or row["semantic_scope"] not in _FULL_SCOPES:
        return False
    floor = row["host_floor"]
    return floor is None or floor["status"] == "pass"


def classify(
    case: Mapping[str, Any],
    expectation: Mapping[str, Any],
    seen: Mapping[str, Any],
    *,
    device_verified: bool,
    receipt_policy: str = "execution_evidence",
) -> tuple[str, str]:
    """``(verdict, reason)`` for one case; a mismatch in either direction fails."""
    if seen["status"] == "not_run":
        return NOT_RUN, seen.get("reason") or "no completed result"
    if seen["status"] == "capture_unavailable":
        return CAPTURE_UNAVAILABLE, seen.get("reason") or "capture unavailable"
    admitted = expectation["placement"] == "accelerator"
    on_device = seen["lane"] == "device" and seen["executed_instructions"] > 0 and device_verified
    passed = _functional_pass(seen)
    if not admitted and (seen["lane"] != "host" or seen["executed_instructions"] != 0):
        return DEVICE_NOT_ADMITTED, "spec declines; agreement requires the host lane and zero device instructions"
    if (
        not on_device
        and passed
        and seen["lane"] == "host"
        and seen["executed_instructions"] == 0
        and structural_no_work(case)
    ):
        return NO_WORK, "empty or metadata-only result on the host lane"
    if (
        not on_device
        and passed
        and seen["lane"] == "host"
        and seen["executed_instructions"] == 0
        and expectation.get("backend_reason") == NONFINITE_REFERENCE
        and nonfinite_reference(case)
    ):
        # No int8 model agrees with a NaN/inf reference: the case and the capture's typed
        # refusal both name it, so the host lane is the expected outcome.
        return AGREE_HOST, "nonfinite floating reference; typed capture refusal on the host lane"
    if admitted and not on_device:
        if seen["lane"] == "device" and seen["executed_instructions"] > 0:
            return ADMITTED_NOT_DEVICE, "device execution evidence did not re-verify"
        return ADMITTED_NOT_DEVICE, f"spec admits but lane {seen['lane']} retired {seen['executed_instructions']}"
    if not passed:
        return NUMERIC_FAIL, seen.get("reason") or f"functional status {seen['functional_status']}"
    if admitted:
        if receipt_policy == "spec_qualified":
            from merlin.targetgen.capsule_common import spec_qualified_device_entry

            if not spec_qualified_device_entry(
                {
                    "status": "pass",
                    "lane": "device",
                    "executed_instructions": seen["executed_instructions"],
                    "spec_qualified_device": seen.get("spec_qualified_device"),
                    "full_call_execution_identity": seen.get("full_call_execution_identity"),
                }
            ):
                return ADMITTED_NOT_DEVICE, "no spec-qualified device receipt"
        return AGREE_DEVICE, "spec admits; device instructions retired; functional pass"
    return AGREE_HOST, "spec does not admit; host lane; functional pass"


def truth_audit(case: Mapping[str, Any], placement: str) -> dict:
    """Optional functional-truth lane beside the spec expectation. Never gates."""
    try:
        from merlin.targetgen.functional_truth import expectation

        truth = expectation(dict(case))
    except Exception as exc:  # noqa: BLE001 -- an audit row records why the table has no answer
        return {"lane": None, "status": "no_rule", "reason": f"{type(exc).__name__}: {exc}"}
    disagrees = (truth["lane"] == "device") != (placement == "accelerator")
    return {"lane": truth["lane"], "rule_family": truth.get("family"), "agrees": not disagrees}


# --------------------------------------------------------------------------------------------
# Record


def _review_statuses(software_spec: Path | None, declarations: set[str]) -> dict:
    if software_spec is None:
        return {}
    import yaml

    from merlin.targetgen.operator_review import review_status

    operations = (yaml.safe_load(Path(software_spec).read_bytes()) or {}).get("operations") or {}
    if isinstance(operations, list):
        operations = {row.get("id"): row for row in operations if isinstance(row, dict)}
    statuses = {}
    for declaration in sorted(declarations):
        row = operations.get(declaration)
        statuses[declaration] = review_status(row, default="unreviewed") if isinstance(row, dict) else "unknown"
    return statuses


def _scheme(software_spec: Path | None, scheme: str) -> dict:
    """Name an existing quantization scheme; validation never adds a scheme (each one is a composition axis)."""
    if software_spec is None:
        return {"id": scheme, "status": "unchecked"}
    import yaml

    document = yaml.safe_load(Path(software_spec).read_bytes()) or {}
    declared = (((document.get("quantization") or {}).get("schemes") or {}).get("schemes")) or []
    ids = {row.get("id") for row in declared if isinstance(row, dict)}
    if scheme not in ids:
        raise ValueError(f"quantization scheme {scheme!r} is not declared by the selected software spec")
    return {"id": scheme, "status": "declared"}


def _row_label(step: Mapping[str, Any]) -> str:
    source = step.get("source") or {}
    where = f"{source.get('path', '?')}:{source.get('line', '?')}" if source else "unsourced"
    return f"{step.get('declaration_id') or 'unsupported'} @ {where}"


def prepare(
    selection: Mapping[str, Any],
    run_root: str | Path,
    *,
    label: str,
    component_host_capabilities: str | Path,
    software_spec: str | Path | None = None,
    host_capabilities: str | Path | None = None,
    rtl_facts: str | Path | None = None,
    target: str | None = None,
    case_keys: Iterable[tuple[str, str]] | None = None,
    receipt_policy: str = "execution_evidence",
    reverify_evidence: bool = True,
    with_truth_audit: bool = False,
    quantization_scheme: str = DEFAULT_QUANTIZATION_SCHEME,
    operator_review: Mapping[str, Any] | None = None,
    verify_evidence: Callable[[dict], bool] | None = None,
) -> dict:
    """Build the sealed spec-validation record from completed runner shards.

    ``host_capabilities`` is the descriptor lane bound for derivation, while
    ``component_host_capabilities`` is the suite execution and expectation lane.
    Both document identities (path and sha256) are recorded.

    ``case_keys`` names ``(source, case_id)`` pairs and validates a subset (for example
    the 193); such a record is at most ``incomplete`` and never satisfies the derivation gate.
    """
    from merlin.common.yaml import load_yaml
    from merlin.targetgen.core_aten_capture import case_capture_name
    from merlin.targetgen.functional_truth_admission import select_admission
    from merlin.targetgen.host_capabilities import validate_host_capabilities
    from merlin.targetgen.operator_review import acceptance

    if receipt_policy not in RECEIPT_POLICIES:
        raise ValueError(f"receipt policy must be one of {RECEIPT_POLICIES}")
    root = Path(run_root) / label
    stamp_path = root / "provenance.json"
    if not stamp_path.is_file():
        raise ValueError(f"{stamp_path}: runner provenance is missing; run the suite first")
    stamp = json.loads(stamp_path.read_bytes())
    software_spec = Path(software_spec) if software_spec is not None else None
    spec_sha = _sha(software_spec) if software_spec is not None else stamp.get("software_spec_sha256")
    component_host = _file_identity(component_host_capabilities)
    if not spec_sha:
        raise ValueError("spec validation requires the selected software spec identity")
    if stamp.get("software_spec_sha256") not in (None, spec_sha):
        raise ValueError("runner executed a different software spec than the one being validated")
    if stamp.get("host_capabilities_sha256") != component_host["sha256"]:
        raise ValueError("runner executed different component host capabilities than the ones being validated")
    target = target or stamp.get("target")
    if software_spec is None or host_capabilities is None or rtl_facts is None or target is None:
        raise ValueError("spec validation requires the selected software spec, host capabilities, RTL facts and target")
    if stamp.get("target") not in (None, target):
        raise ValueError("runner executed a different target than the one being validated")
    descriptor_host = _file_identity(host_capabilities)
    admission = select_admission(
        target, rtl_facts=Path(rtl_facts), software_spec=software_spec, host_capabilities=Path(component_host["path"])
    )
    input_identity = selection_identity(
        resolved_spec=admission["spec"],
        rtl_facts=admission["rtl_facts"],
        capability_contracts=[admission["capability_contract"]],
        host_capabilities=[validate_host_capabilities(load_yaml(Path(descriptor_host["path"])))],
    )
    if verify_evidence is None and reverify_evidence:
        from merlin.targetgen.core_aten_device import verify_execution_evidence as verify_evidence
    rows_by_key = {(row["source"], case_key(row["case"])): row for row in selection["cases"]}
    keys = list(rows_by_key) if case_keys is None else list(dict.fromkeys(case_keys))
    unknown = sorted(set(keys) - set(rows_by_key))
    if unknown:
        raise ValueError(f"case subset names unknown cases: {unknown[:5]}")
    rows = []
    declarations: set[str] = set()
    for key in keys:
        source, case = rows_by_key[key]["source"], rows_by_key[key]["case"]
        executed_case = runner_case(source, case)
        directory = root / "shards" / case_capture_name(executed_case)
        result_path = directory / "result.json"
        result = json.loads(result_path.read_bytes()) if result_path.is_file() else {}
        stale = result and (
            result.get("case_sha256") != case_sha256(executed_case)
            or result.get("numeric_policy") != NUMERIC_POLICY
            or result.get("provenance") != stamp
        )
        if not result or stale:
            result = {"observation": {"status": "not_run", "reason": "stale result" if stale else "no result"}}
        expectation = spec_expectation(directory, case, admission)
        if expectation.get("software_spec_sha256") not in (None, spec_sha):
            raise ValueError(f"{key}: admission plan selected a different software spec")
        if expectation.get("host_capabilities_sha256") not in (None, component_host["sha256"]):
            raise ValueError(f"{key}: admission plan selected different component host capabilities")
        seen = observed(result)
        evidence = (
            (((result.get("observation") or {}).get("evidence") or {}).get("semantic_verdict") or {})
            .get("routing", {})
            .get("execution_evidence")
        )
        device_verified = seen["executed_instructions"] > 0 and (
            not reverify_evidence or (isinstance(evidence, dict) and bool(verify_evidence(evidence)))
        )
        verdict, reason = classify(
            case, expectation, seen, device_verified=device_verified, receipt_policy=receipt_policy
        )
        declarations.update(
            step["declaration_id"]
            for step in expectation["steps"]
            if step["authority"] == "software" and step["declaration_id"]
        )
        row = {
            "case_id": case_key(case),
            "source": source,
            "overload": case["overload"],
            "dtype": _case_dtype(case),
            "partition_assignment": case.get("partition_assignment"),
            "case_sha256": case_sha256(case),
            "result_sha256": _sha(result_path) if result_path.is_file() and not stale else None,
            "spec_expectation": expectation,
            "observed": {**seen, "device_evidence_verified": device_verified},
            "verdict": verdict,
            "reason": reason,
            "truth_audit": truth_audit(case, expectation["placement"]) if with_truth_audit else None,
        }
        rows.append(row)
    return _assemble(
        rows,
        selection=selection,
        complete_scope=case_keys is None,
        spec_sha=spec_sha,
        descriptor_host=descriptor_host,
        component_host=component_host,
        input_identity=input_identity,
        stamp=stamp,
        receipt_policy=receipt_policy,
        reverify_evidence=reverify_evidence,
        scheme=_scheme(software_spec, quantization_scheme),
        review={
            "acceptance": dict(operator_review) if operator_review is not None else acceptance(),
            "declarations": _review_statuses(software_spec, declarations),
        },
    )


def _assemble(
    rows,
    *,
    selection,
    complete_scope,
    spec_sha,
    descriptor_host,
    component_host,
    input_identity,
    stamp,
    receipt_policy,
    reverify_evidence,
    scheme,
    review,
):
    by_overload: dict[str, Counter] = defaultdict(Counter)
    by_row: dict[str, Counter] = defaultdict(Counter)
    mismatches, admitting = [], defaultdict(bool)
    for row in rows:
        by_overload[row["overload"]][row["verdict"]] += 1
        software = [s for s in row["spec_expectation"]["steps"] if s["authority"] == "software"]
        for step in software:
            if step["status"] == "admitted":
                admitting[row["overload"]] = True
        if not software:
            by_row["unsupported @ no admission plan"][row["verdict"]] += 1
        for label in sorted({_row_label(step) for step in software}):
            by_row[label][row["verdict"]] += 1
        if row["verdict"] in FAILING:
            wanted = "admitted" if row["verdict"] == ADMITTED_NOT_DEVICE else None
            implicated = sorted({_row_label(s) for s in software if wanted is None or s["status"] == wanted}) or [
                "unsupported @ no admitting row"
            ]
            mismatches.append(
                {
                    "case_id": row["case_id"],
                    "source": row["source"],
                    "overload": row["overload"],
                    "verdict": row["verdict"],
                    "reason": row["reason"],
                    "spec_rows": implicated,
                }
            )
        admitting.setdefault(row["overload"], False)
    counts = Counter(row["verdict"] for row in rows)
    truth = [r for r in rows if r["truth_audit"] and r["truth_audit"].get("agrees") is False]
    complete = complete_scope and counts[NOT_RUN] == 0 and len(rows) == len(selection["cases"])
    verdict = "failed" if any(counts[v] for v in FAILING) else "validated" if complete else "incomplete"
    record = {
        "schema": SCHEMA,
        "verdict": verdict,
        "input_identity": input_identity,
        "software_spec_sha256": spec_sha,
        "descriptor_host_capabilities": descriptor_host,
        "component_host_capabilities": component_host,
        "numeric_policy": NUMERIC_POLICY,
        "quantization_scheme": scheme,
        "operator_review": review,
        "receipt_policy": receipt_policy,
        "device_evidence": "reverified" if reverify_evidence else "recorded_at_execution",
        "selection": {
            "sha256": selection["selection_sha256"],
            "sources": selection["sources"],
            "case_count": len(selection["cases"]),
        },
        "scope": {"validated_cases": len(rows), "complete": complete, "subset": not complete_scope},
        "runner_provenance_sha256": hashlib.sha256(canonical_json(stamp)).hexdigest(),
        "verdict_counts": dict(sorted(counts.items())),
        "by_overload": {k: dict(sorted(v.items())) for k, v in sorted(by_overload.items())},
        "by_spec_row": {k: dict(sorted(v.items())) for k, v in sorted(by_row.items())},
        "mismatches": mismatches,
        "unmodeled": sorted(op for op, admitted in admitting.items() if not admitted),
        "truth_disagreements": [
            {
                "case_id": r["case_id"],
                "source": r["source"],
                "overload": r["overload"],
                "truth_lane": r["truth_audit"]["lane"],
                "spec_placement": r["spec_expectation"]["placement"],
            }
            for r in truth
        ],
        "role": "phase0_evidence; never capsules; never phase1 grading load",
        "cases": rows,
    }
    record["sha256"] = digest(record)
    return record


def write(record: Mapping[str, Any], path: str | Path) -> Path:
    """Write the canonical sealed bytes; the content digest is inside the record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(canonical_json(record) + b"\n")
    temporary.replace(destination)
    return destination


def verify(path: str | Path) -> dict:
    """Refuse changed record/component bytes, a wrong schema or an inconsistent verdict."""
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} record")
    if record.get("sha256") != digest(record) or raw != canonical_json(record) + b"\n":
        raise ValueError(f"{path}: spec-validation record bytes differ from its sealed digest")
    component = record.get("component_host_capabilities")
    if (
        not isinstance(component, dict)
        or not component.get("path")
        or _sha(Path(component["path"])) != component.get("sha256")
    ):
        raise ValueError(f"{path}: component host capabilities differ from the sealed record; rerun spec validation")
    counts = Counter(row["verdict"] for row in record["cases"])
    if set(counts) - VERDICTS or dict(sorted(counts.items())) != record["verdict_counts"]:
        raise ValueError(f"{path}: spec-validation case verdicts differ from the record summary")
    failed = any(counts[v] for v in FAILING)
    complete = record["scope"]["complete"]
    expected = "failed" if failed else "validated" if complete else "incomplete"
    if record["verdict"] != expected:
        raise ValueError(f"{path}: spec-validation verdict {record['verdict']!r} contradicts its cases")
    return record


# --------------------------------------------------------------------------------------------
# Derivation gate

GATE_KEY = "phase0_spec_validation"
_GATE_VALUES = ("off", "required")


def gate_required(workload_spec: Mapping[str, Any] | None) -> bool:
    """The descriptor's explicit switch; off by default so spec edits need no full rerun."""
    value = (workload_spec or {}).get(GATE_KEY, "off")
    if value not in _GATE_VALUES:
        raise ValueError(f"workload_spec.{GATE_KEY} must be one of {_GATE_VALUES}, not {value!r}")
    return value == "required"


def _load_bound(path: str | Path) -> dict:
    try:
        return verify(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"spec-validation record {path} is unreadable: {exc}") from exc


def bind(
    path: str | Path | None,
    *,
    software_spec: str | Path,
    workload_spec: Mapping[str, Any] | None,
    input_identity: Mapping[str, str] | None = None,
) -> dict:
    """Bind the validation record to the authored spec and all resolved selection inputs."""
    required = gate_required(workload_spec)
    gate = "required" if required else "off"
    spec_sha = _sha(Path(software_spec))
    if path is None:
        if required:
            raise ValueError(
                f"workload_spec.{GATE_KEY}: required, but no spec-validation record was supplied; "
                "run build_tools/scripts/run_spec_validation.py for this software spec and pass --spec-validation"
            )
        return {"gate": gate, "status": "not_bound", "software_spec_sha256": spec_sha}
    record = _load_bound(path)
    if record["software_spec_sha256"] != spec_sha:
        raise ValueError(
            f"spec-validation record {path} validated software spec {record['software_spec_sha256']}, "
            f"not the selected {spec_sha}; rerun spec validation for this spec"
        )
    _verify_input_identity(record, input_identity)
    if required and record["verdict"] != "validated":
        raise ValueError(
            f"spec-validation record {path} is {record['verdict']} "
            f"({record['verdict_counts']}); a seal requires a complete validated record"
        )
    return {
        "gate": gate,
        "status": "bound",
        "path": str(Path(path).absolute()),
        "sha256": record["sha256"],
        "software_spec_sha256": spec_sha,
        "selection_sha256": record["selection"]["sha256"],
        "input_identity": dict(input_identity),
        "verdict": record["verdict"],
    }


def verify_binding(
    binding: Mapping[str, Any] | None,
    *,
    software_spec_sha256: str | None,
    workload_spec: Mapping[str, Any] | None,
    input_identity: Mapping[str, str] | None = None,
) -> None:
    """Refuse, under the required gate, a missing, stale, changed or failing bound record."""
    if not gate_required(workload_spec):
        return
    if not isinstance(binding, dict) or binding.get("status") != "bound":
        raise ValueError(
            f"workload_spec.{GATE_KEY}: required, but the selected requirement binds no spec-validation "
            "record; rederive with --spec-validation"
        )
    record = _load_bound(binding["path"])
    if (
        record["sha256"] != binding.get("sha256")
        or software_spec_sha256 is None
        or record["software_spec_sha256"] != software_spec_sha256
        or binding.get("software_spec_sha256") != software_spec_sha256
    ):
        raise ValueError(
            "bound spec-validation record differs from the selected software spec or its sealed digest; "
            "rerun spec validation and rederive"
        )
    _verify_input_identity(record, input_identity)
    _verify_input_identity(binding, input_identity)
    if record["verdict"] != "validated":
        raise ValueError(f"bound spec-validation record is {record['verdict']}, not validated")
