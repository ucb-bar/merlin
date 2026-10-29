"""Bounded descriptive source/label joins from original public archive intake.

Historical producer measurements stay historical. This supplies neither a new
measurement, numerical verdict, native timing qualification nor promotion.
"""

from __future__ import annotations

import hashlib
import tarfile

from merlin.perf.candidate_source import MAX_TEXT, source_observation

from . import _archive_source_domain as DOMAIN
from . import archive_spike_runtime as RT
from .contracts import StageGateError

SCHEMA = "merlin.phase2.retained_archive_feedback.v1"


def _source_artifacts(archive, wanted):
    """Reopen exact indexed bytes without extraction or another source parser."""
    from merlin_timing import archive as A

    found, total = {}, 0
    with tarfile.open(archive, mode="r|", tarinfo=A._BoundedTarInfo) as stream:
        for count, member in enumerate(stream):
            if count >= A.MAX_MEMBERS:
                raise StageGateError("archive feedback member admission exceeded")
            if member.name not in wanted:
                continue
            pin = wanted[member.name]
            if not member.isreg() or member.size != pin["bytes"] or not 0 < member.size <= MAX_TEXT:
                raise StageGateError("original archive source exceeds its indexed byte scope")
            total += member.size
            if total > 128 << 20 or member.name in found:
                raise StageGateError("archive feedback source bytes repeat or exceed admission")
            raw = stream.extractfile(member).read(MAX_TEXT + 1)
            if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
                raise StageGateError("original archive source changed during reopening")
            found[member.name] = raw
    if set(found) != set(wanted):
        raise StageGateError("original indexed source missing from archive feedback")
    return found


def prepare(*, archive, archive_sha256, binding_path, dataset):
    """Reconstruct original intake; missing optional source work stays UNKNOWN."""
    from merlin_timing import archive as A

    inventory = A.inspect_archive(archive, archive_sha256)
    binding, _pin = RT.metadata(binding_path)
    plan = A.prepare(inventory, binding)
    if plan["dataset_template"]["cases"] != dataset["cases"]:
        raise StageGateError("retained feedback changed original case/source/split membership")
    wanted, rows = {}, {row["record_id"]: row for row in inventory["records"]}
    for item in inventory["records"]:
        row = item["row"]
        if row.get("visibility") != "public":
            raise StageGateError("retained archive feedback cannot consume hidden source evidence")
        pin = row["artifacts"]["input_interface"]
        name = "run/" + A._path(pin["path"])
        if name in wanted and wanted[name] != pin:
            raise StageGateError("original source archive pins disagree")
        wanted[name] = pin
    original_sources = _source_artifacts(archive, wanted)
    sources = {}
    for name, raw in original_sources.items():
        digest = wanted[name]["sha256"]
        if digest in sources:
            continue
        try:
            source = source_observation(raw.decode())
            sources[digest] = {
                key: source.get(key) for key in ("status", "grammar", "macs", "work_basis", "work_refusals")
            }
            try:
                sources[digest]["semantic_profile"] = DOMAIN.interface_profile(raw.decode())
            except (ValueError, KeyError, TypeError):
                sources[digest]["semantic_profile"] = None
        except (ValueError, TypeError, KeyError, UnicodeError, NotImplementedError) as error:
            sources[digest] = {"status": "UNKNOWN", "macs": None, "reason": type(error).__name__}
    cases = {case["case_id"]: {"case": case, "records": []} for case in dataset["cases"]}
    available = {row["record_id"] for row in dataset["records"]}
    for observation in plan["observations"]:
        item = rows[observation["record_id"]]
        row = item["row"]
        case = cases[observation["case_id"]]["case"]
        interface = row["artifacts"]["input_interface"]
        if case["source"]["source_sha256"] != interface["sha256"]:
            raise StageGateError("retained source group is not its original interface bytes")
        label = observation["label"]
        cases[case["case_id"]]["records"].append(
            {
                "record_id": observation["record_id"],
                "original_result_sha256": item["result_sha256"],
                "label_receipt_sha256": label["receipt_sha256"],
                "originally_measured": label["measured_now"] is True,
                "original_numerical_status": label["numeric"],
                "cycles": label["cycles"],
                "engine": label["engine"],
                "engine_sha256": label["engine_sha256"],
                "scope_binding": observation["scope_binding"],
                "native_observation_available": observation["record_id"] in available,
                "caller_sha256": row["harness_sha256"],
                "shape": observation["shape"],
            }
        )
    if A.file_identity(archive) != archive_sha256:
        raise StageGateError("original archive changed during source/measurement feedback preparation")
    return {
        "schema": SCHEMA,
        "archive_sha256": archive_sha256,
        "domain": dataset["domain"],
        "sources": sources,
        "cases": list(cases.values()),
        "measurement_origin": "retained_original_public_gsim_execution",
        "measured_now": False,
        "promotion_authority": "NONE",
        "readout_service": "UNKNOWN; logical shapes and requested counts do not prove physical readout ownership",
    }


def source_identity(compiled):
    """Raw compiled interface identity, distinct from a frozen capsule closure."""
    artifacts = (compiled.get("candidate_diagnostics") or {}).get("artifacts") or {}
    observed = (artifacts.get("source") or {}).get("sha256")
    joined = (compiled.get("artifact_join") or {}).get("source_sha256")
    if observed and joined and observed != joined:
        return None
    return observed or joined


def candidates(catalog, *, compiled, report):
    """Only matching original bytes/domain proceed to the caller-context check."""
    source = source_identity(compiled)
    if compiled.get("status") != "observed" or not source:
        return []
    return [
        entry
        for entry in catalog.get("cases", [])
        if entry["case"]["source"]["source_sha256"] == source
        and entry["case"]["elf_sha256"] == compiled.get("elf_sha256") == report.get("elf_sha256")
        and all(entry["case"][name] == report.get(name) for name in ("scope_sha256", "device_sha256"))
    ]


def match(catalog, *, compiled, report, callers):
    """An explicit caller bridge preserves both original input identity schemes.

    The original queue owner reopens actual consumed caller source and producers
    before passing ``callers``. Neither shape equality nor an ELF alone suffices.
    All matching repeats remain; no fastest repeat or mean is selected.
    """
    from merlin_timing import contracts as C

    unknown = {"status": "UNKNOWN", "measurement": None, "measured_now": False, "promotion_authority": "NONE"}
    entries = candidates(catalog, compiled=compiled, report=report)
    if not entries:
        return {**unknown, "reason": "no exact original source/ELF/scope/device archived execution"}
    matches = []
    scope = report.get("scope_binding") or {}
    for entry in entries:
        case = entry["case"]
        if any(
            row["scope_binding"] is None
            or row["originally_measured"] is not True
            or row["original_numerical_status"] != "pass"
            or not row["native_observation_available"]
            or type(row["cycles"]) is not int
            or not 0 < row["cycles"] < 2**64
            for row in entry["records"]
        ):
            continue  # Missing/failed original repeats cannot be dropped to improve this execution.
        matched_entry = []
        for row in entry["records"]:
            original = row["scope_binding"]
            if (
                row["caller_sha256"] not in callers
                or original is None
                or row["originally_measured"] is not True
                or row["original_numerical_status"] != "pass"
                or not row["native_observation_available"]
                or C.digest({"schema": "merlin.embedded_input_context.v1", "harness_sha256": row["caller_sha256"]})
                != case["input_sha256"]
                or any(original[name] != scope.get(name) for name in ("begin_pc", "end_pc", "occurrence"))
            ):
                break
            matched_entry.append(
                {
                    **row,
                    "case_id": case["case_id"],
                    "original_input_sha256": case["input_sha256"],
                    **({"archive_sha256": entry["archive_sha256"]} if "archive_sha256" in entry else {}),
                }
            )
        if len(matched_entry) == len(entry["records"]):
            matches.extend(matched_entry)
    if not matches:
        return {**unknown, "reason": "original caller/input/counter-window bridge unavailable or mismatched"}
    archives = {row.get("archive_sha256", catalog["archive_sha256"]) for row in matches}
    if len(archives) != 1:
        return {**unknown, "reason": "original repeats span different archives; no measurement is selected"}
    cycles = [row["cycles"] for row in matches]
    # The worst repeat is the frozen diagnostic representative, never a score.
    return {
        "status": "retained_historical_measurement",
        "measured_now": False,
        "measurement": {
            "kind": "retained_measured_target_cycles",
            "measured_now": False,
            "functional_status": "PASS",
            "complete_source_scope": True,
            **{name: report[name] for name in ("elf_sha256", "scope_sha256", "device_sha256")},
            "cycles": max(cycles),
            "archive_sha256": next(iter(archives)),
            "record_ids": [row["record_id"] for row in matches],
            "receipt_sha256s": [row["label_receipt_sha256"] for row in matches],
        },
        "input_identity_bridge": {
            "current_frozen_context_sha256": report["input_sha256"],
            "original_embedded_context_sha256s": sorted({row["original_input_sha256"] for row in matches}),
        },
        "retained_repeats": matches,
        "worst_repeat_cycles": max(cycles),
        "promotion_authority": "NONE",
        "fresh_candidate_measurement": False,
    }


def queued(catalog, *, compiled, report, directory, value):
    """The normal owner supplies its actual queued arm, never a candidate receipt."""
    from . import _archive_feedback_context as CONTEXT

    entries = candidates(catalog, compiled=compiled, report=report)
    if not entries:
        return match(catalog, compiled=compiled, report=report, callers=())
    callers = {row["caller_sha256"] for entry in entries for row in entry["records"]}
    original = CONTEXT.reopen(directory, value=value, compiled=compiled, report=report, expected_callers=callers)
    result = match(catalog, compiled=compiled, report=report, callers=original["caller_sha256s"])
    result["original_current_context"] = original
    return result
