"""Pinned structural intake of previously recorded compiler experiments.

Bindings select JSON fields, not workload names or prose. A historical result is
prior evidence: this module neither executes its compiler nor qualifies a new
candidate. Missing comparison identities and gates remain explicit unknowns.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from merlin.kernels.reference_inventory import digest, read_pinned

BINDING_SCHEMA = "merlin.reconstruction_history_binding.v1"
RECORD_SCHEMA = "merlin.reconstruction_history.v1"
_SECTIONS = {"before", "after", "measurement", "context", "metadata"}
_AXES = ("source", "input", "shape", "roi", "environment")
_GATES = ("numeric", "placement", "attribution")


def _identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _pointer(document, pointer):
    if type(pointer) is not str or (pointer and not pointer.startswith("/")):
        raise ValueError("history selector requires a JSON Pointer")
    value = document
    for raw in pointer.split("/")[1:] if pointer else ():
        # Reject malformed escapes instead of accepting a different field.
        if any(not part or part[0] not in "01" for part in raw.split("~")[1:]):
            raise ValueError("history selector has an invalid JSON Pointer escape")
        part = raw.replace("~1", "/").replace("~0", "~")
        if type(value) is dict:
            value = value[part]
        elif type(value) is list and part.isdecimal() and (part == "0" or not part.startswith("0")):
            value = value[int(part)]
        else:
            raise KeyError(pointer)
    return deepcopy(value)


@dataclass(frozen=True)
class HistoryRecord:
    """Serializable recorded evidence, with byte pins that can be checked again."""

    record: dict

    def to_dict(self):
        return deepcopy(self.record)

    def revalidate(self):
        for pin in self.record["artifact_pins"]:
            read_pinned(pin["path"], pin["sha256"], max_bytes=pin["bytes"])


def read_history(binding, *, root=".", max_artifact_bytes=32 << 20, max_total_bytes=64 << 20):
    """Read one caller-selected schema binding without importing historical code.

    ``artifacts`` maps names to path/SHA256 pins. ``fields`` maps output sections
    and field names to ``{artifact, pointer, optional?}`` selectors. ``identities``
    maps comparison axes to either ``{shared: selector}`` (an original declared
    common identity) or ``{before: selector, after: selector}``. Gate bindings use
    ``{selector, pass_values?, fail_values?, unresolved_values?}``; exact JSON
    values are compared. An unresolved attribution gate requests evidence repair
    without inventing a failed numerical execution.
    ``measurement_kind`` is the caller-selected schema's measured/provisional/
    unknown classification and is retained as a declaration, never inferred from
    an elapsed-looking number. A numeric pass alone never grants placement.
    """
    if type(binding) is not dict or binding.get("schema") != BINDING_SCHEMA:
        raise ValueError("unsupported reconstruction history binding")
    if set(binding) - {
        "schema",
        "id",
        "artifacts",
        "fields",
        "identities",
        "comparison_axes",
        "measurement_kind",
        "gates",
    }:
        raise ValueError("history binding has unsupported fields")
    if any(type(limit) is not int or limit <= 0 for limit in (max_artifact_bytes, max_total_bytes)):
        raise ValueError("history intake requires positive byte budgets")
    artifacts = binding.get("artifacts", {})
    if type(artifacts) is not dict or not 1 <= len(artifacts) <= 64:
        raise ValueError("history requires a bounded nonempty artifact selection")
    documents, pins, consumed = {}, [], 0
    for name, pin in artifacts.items():
        if type(name) is not str or not name or type(pin) is not dict or set(pin) != {"path", "sha256"}:
            raise ValueError("history artifacts require named path and SHA256 pins")
        path = (Path(root) / pin["path"]).resolve()
        data = read_pinned(path, pin["sha256"], max_bytes=min(max_artifact_bytes, max_total_bytes - consumed))
        consumed += len(data)
        documents[name] = json.loads(data)
        pins.append(dict(name=name, path=str(path), sha256=pin["sha256"], bytes=len(data)))
    missing, selectors = [], {}

    def select(spec, destination):
        if (
            type(spec) is not dict
            or set(spec) - {"artifact", "pointer", "optional"}
            or spec.get("artifact") not in documents
            or type(spec.get("optional", False)) is not bool
        ):
            raise ValueError("history requires an exact selector of a selected artifact")
        if "pointer" not in spec:
            raise ValueError("history selector is missing its JSON Pointer")
        selectors[destination] = deepcopy(spec)
        try:
            return _pointer(documents[spec["artifact"]], spec["pointer"])
        except (KeyError, IndexError):
            if not spec.get("optional", False):
                raise ValueError("required history field is missing: " + destination) from None
            missing.append(destination)
            return None

    fields = binding.get("fields", {})
    if type(fields) is not dict or set(fields) - _SECTIONS:
        raise ValueError("history fields require supported output sections")
    record = {name: {} for name in _SECTIONS}
    for section, members in fields.items():
        if type(members) is not dict or len(members) > 128:
            raise ValueError("history fields require bounded named selectors")
        for name, spec in members.items():
            if type(name) is not str or not name:
                raise ValueError("history field names must be nonempty")
            record[section][name] = select(spec, section + "." + name)
    for endpoint in ("before", "after"):
        for name in ("compiler_sha256", "program_sha256"):
            value = record[endpoint].setdefault(name, None)
            if value is not None and not _sha(value):
                raise ValueError("history compiler/program identities require exact SHA256")
        settings = record[endpoint].setdefault("settings", None)
        if settings is not None and type(settings) is not dict:
            raise ValueError("history settings must be an original JSON object")

    axes = binding.get("comparison_axes", list(_AXES))
    if (
        type(axes) is not list
        or not axes
        or len(axes) > 32
        or any(type(axis) is not str or not axis for axis in axes)
        or len(set(axes)) != len(axes)
        or not set(_AXES).issubset(axes)
    ):
        raise ValueError("history comparison axes must be bounded distinct names")
    witnesses = {}
    identities = binding.get("identities", {})
    if type(identities) is not dict or len(identities) > 32:
        raise ValueError("history identities must be bounded named field bindings")
    for axis, specs in identities.items():
        if (
            type(axis) is not str
            or not axis
            or type(specs) is not dict
            or set(specs) not in ({"shared"}, {"before", "after"})
        ):
            raise ValueError("history identity requires an original common identity or an exact pair")
        if "shared" in specs:
            before = after = select(specs["shared"], "identities." + axis + ".shared")
            custody = "declared_common"
        else:
            before = select(specs["before"], "identities." + axis + ".before")
            after = select(specs["after"], "identities." + axis + ".after")
            custody = "compared_fields"
        known = before is not None and after is not None
        witnesses[axis] = dict(
            before=before,
            after=after,
            status=("same" if _identity(before) == _identity(after) else "different") if known else "unknown",
            custody=custody,
        )
    statuses = [witnesses.get(axis, {}).get("status", "unknown") for axis in axes]
    same_domain = False if "different" in statuses else None if "unknown" in statuses else True
    kind = binding.get("measurement_kind", "unknown")
    if kind not in ("measured", "provisional", "unknown"):
        raise ValueError("history measurement kind must be explicitly declared")
    measurement = record["measurement"]
    for name in ("before_cycles", "after_cycles"):
        value = measurement.setdefault(name, None)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("history cycles must be finite nonnegative numbers or unknown")
    before, after = (measurement[name] for name in ("before_cycles", "after_cycles"))
    direction = (
        "unknown"
        if before is None or after is None
        else ("regression" if after > before else "improvement" if after < before else "unchanged")
    )
    measurement.update(
        kind=kind,
        kind_custody="selected_schema_binding",
        same_domain=same_domain,
        comparison_axes=axes,
        identity_witnesses=witnesses,
        domain_sha256=_identity({axis: witnesses[axis]["before"] for axis in axes}) if same_domain else None,
        delta_percent=100 * (after - before) / before if before not in (None, 0) and after is not None else None,
        direction=direction,
    )
    gate_bindings = binding.get("gates", {})
    if type(gate_bindings) is not dict or set(gate_bindings) - set(_GATES):
        raise ValueError("history has an unsupported qualification gate")
    gates, gate_evidence = dict.fromkeys(_GATES, "unknown"), {}
    for gate, spec in gate_bindings.items():
        if type(spec) is not dict or set(spec) - {"selector", "pass_values", "fail_values", "unresolved_values"}:
            raise ValueError("history gate requires an exact field/status binding")
        value = select(spec.get("selector"), "gates." + gate)
        passed, failed = spec.get("pass_values", ["pass"]), spec.get("fail_values", ["fail"])
        unresolved = spec.get("unresolved_values", [])
        if any(type(values) is not list or None in values for values in (passed, failed, unresolved)):
            raise ValueError("history gate values must be JSON lists")

        # JSON true and integer 1 are different declarations.
        def matches(values):
            return any(type(value) is type(item) and value == item for item in values)

        if sum(matches(values) for values in (passed, failed, unresolved)) > 1:
            raise ValueError("history gate has ambiguous status declarations")
        gates[gate] = "pass" if matches(passed) else "fail" if matches(failed) else "unknown"
        gate_evidence[gate] = dict(value=value, binding=deepcopy(spec), unresolved=matches(unresolved))
    settings_before, settings_after = (record[side]["settings"] for side in ("before", "after"))
    changed = (
        None
        if settings_before is None or settings_after is None
        else [
            dict(
                name=name,
                before=settings_before.get(name),
                after=settings_after.get(name),
                before_present=name in settings_before,
                after_present=name in settings_after,
            )
            for name in sorted(set(settings_before) | set(settings_after))
            if (name in settings_before) != (name in settings_after)
            or _identity(settings_before.get(name)) != _identity(settings_after.get(name))
        ]
    )
    rejected = "fail" in gates.values()
    outcome = (
        "rejected_gate"
        if rejected
        else (
            kind + "_" + direction
            if kind != "unknown" and direction in ("regression", "improvement", "unchanged")
            else "unresolved"
        )
    )
    action = (
        "attribution_evidence_repair"
        if gates["attribution"] == "fail" or gate_evidence.get("attribution", {}).get("unresolved")
        else "numeric_repair"
        if gates["numeric"] == "fail"
        else "placement_repair"
        if gates["placement"] == "fail"
        else "controlled_factor_comparison"
        if direction == "regression"
        else "qualification_needed"
    )
    complete = same_domain is True and all(value == "pass" for value in gates.values())
    trial = dict(
        before=record["before"],
        after=record["after"],
        domain_sha256=measurement["domain_sha256"],
        context=record["context"],
    )
    record.update(
        schema=RECORD_SCHEMA,
        authority="recorded_prior_only",
        history_id=binding.get("id"),
        artifact_pins=pins,
        selectors=selectors,
        gates=gates,
        gate_evidence=gate_evidence,
        changed_factors=changed,
        settings_custody={
            side: "selected_json_artifact" if record[side]["settings"] is not None else "unknown"
            for side in ("before", "after")
        },
        outcome=outcome,
        next_action=action,
        ranking_eligible=complete and kind != "unknown",
        unknowns=sorted(
            set(
                missing
                + ["identities." + axis for axis, status in zip(axes, statuses, strict=True) if status == "unknown"]
                + ["gates." + gate for gate in _GATES if gates[gate] == "unknown"]
            )
        ),
        trial_identity=_identity(trial)
        if complete
        and settings_before is not None
        and settings_after is not None
        and all(record[side]["compiler_sha256"] is not None for side in ("before", "after"))
        else None,
    )
    record["record_identity"] = _identity({key: value for key, value in record.items() if key != "history_id"})
    return HistoryRecord(record)


def read_histories(bindings, *, root=".", **budgets):
    """Deduplicate identical pinned records, without suppressing future trials."""
    if type(bindings) not in (tuple, list) or len(bindings) > 1024:
        raise ValueError("history selection requires a bounded list of bindings")
    records, duplicates, seen = [], [], set()
    for binding in bindings:
        row = read_history(binding, root=root, **budgets).to_dict()
        identity = row["record_identity"]
        if identity in seen:
            duplicates.append(dict(history_id=row["history_id"], record_identity=identity))
        else:
            records.append(row)
            seen.add(identity)
    return dict(schema="merlin.reconstruction_history_collection.v1", records=records, duplicate_records=duplicates)


def load_history(selection, *, root=".", **budgets):
    """CLI/packet entry: return normalized records from a parsed pinned selection."""
    if (
        type(selection) is not dict
        or selection.get("schema") != "merlin.reconstruction_history_selection.v1"
        or set(selection) != {"schema", "bindings"}
    ):
        raise ValueError("unsupported reconstruction history selection")
    return tuple(read_histories(selection["bindings"], root=root, **budgets)["records"])
