"""Audit and archive host-only Phase 1 capsule simulator evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.phase1.capsule_timing_archive.v3"
TRAINING_SCOPE = "training_eligible_public"
#: Where a run keeps raw capsule executions, and who could see the capsule. Formal grading writes the
#: public and hidden suites into their own directories; authoring-time grades, agent self-checks and
#: simulator jobs only ever run the public loop suite. Anything else is not indexed.
RESULT_AREAS = (
    ("_qa_work/**/runs/*/*/capsule_result.json", "authoring", "public"),
    ("grading_public/runs/*/*/capsule_result.json", "grading_public", "public"),
    ("grading_hidden/runs/*/*/capsule_result.json", "grading_hidden", "hidden"),
)
_REQUIRED_ARTIFACTS = ("harness", "input_interface", "command_buffer", "static_instruction_trace", "numeric_report")
_MAX_INVOCATIONS = 4096
_MAX_INVOCATION_BYTES = 1 << 20
_MAX_OBSERVER_DOCUMENT_BYTES = 16 << 20


class TimingArchiveError(ValueError):
    """The selected run has no usable gSIM capsule timing evidence."""


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _kernel_scopes(text):
    matches = []
    for tail in text.split("/* merlin.kernel_call_timing.v1 ")[1:]:
        comment, closed, _ = tail.partition(" */")
        fields = comment.split()
        if (
            closed
            and len(fields) == 3
            and fields[0].startswith("selection_sha256=")
            and fields[1] == "boundary=entry-through-completion"
            and fields[2].startswith("record_b64=")
        ):
            digest = fields[0].partition("=")[2]
            if _sha256(digest):
                matches.append((digest, fields[2].partition("=")[2]))
    return matches


def _cycle_value(line):
    fields = line.strip().split()
    if (
        len(fields) == 3
        and fields[:2] == ["METRIC", "cycles"]
        and fields[2]
        and fields[2][0] != "0"
        and all(c in "0123456789" for c in fields[2])
    ):
        return int(fields[2])
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_name(value, default: str) -> str:
    """A single path component named by a build receipt, or the receipt's own default."""
    if value is None:
        return default
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise TimingArchiveError(f"build receipt names an unsafe member: {value!r}")
    return value


def _compiler_candidate(document, generated, artifacts):
    """Accept original child custody or the ordinary producer's stable snapshot."""
    source = document.get("compiler_source")
    if source is None:
        return (document.get("original_compiler_child") or {}).get("candidate_sha256")
    if source.get("schema") not in {"merlin.capsule_compiler_source.v1", "merlin.capsule_compiler_source.v2"} or source.get("status") != "stable":
        return None
    candidate = source.get("candidate_sha256")
    if (
        not isinstance(candidate, str)
        or not _sha256(candidate)
    ):
        return None
    if source.get("schema") == "merlin.capsule_compiler_source.v1":
        if source.get("snapshot") != "compiler-source/" + candidate:
            return None
        snapshot = generated / source["snapshot"]
        limits = dict(max_bytes=32 << 20, max_files=8192)
    else:
        snapshot = Path(source.get("snapshot", ""))
        if (not snapshot.is_absolute() or snapshot.resolve() != snapshot or snapshot.is_symlink()
                or snapshot.parts[-2:] != ("compiler-source", candidate)):
            return None
        limits = dict(max_bytes=128 << 20, max_files=16384)
    from merlin.common.tree_hash import hash_tree
    from merlin_experiments.phase2.retained_revisions import source_inventory

    try:
        source_inventory(snapshot, **limits, exclude_generated=True)
        if hash_tree(snapshot)["sha256"] != candidate:
            return None
    except (OSError, ValueError, RuntimeError):
        return None
    for name, key in (
        ("package_kernel.elf", "elf"),
        ("harness.c", "harness"),
        ("input.interface.mlir", "input_interface"),
        ("command_buffer.json", "command_buffer"),
    ):
        actual = artifacts.get(key)
        if actual is None or source.get("artifacts", {}).get(name) != actual["sha256"]:
            return None
    return candidate


def _result_paths(root: Path) -> list[tuple[Path, str, str]]:
    found = {}
    for pattern, area, visibility in RESULT_AREAS:
        for path in sorted(root.glob(pattern)):
            found.setdefault(path, (area, visibility))
    return [(path, *found[path]) for path in sorted(found)]


def _invocations(generated: Path) -> list[dict]:
    """The retained process records of one capsule build/run, oldest first; unreadable ones are skipped."""
    rows = []
    for path in sorted((generated / "invocations").glob("*/invocation.json"))[:_MAX_INVOCATIONS]:
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_INVOCATION_BYTES:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and document.get("schema") == "merlin.invocation_record.v1":
            rows.append(document)
    return sorted(rows, key=lambda row: row.get("finished_ns") if type(row.get("finished_ns")) is int else 0)


def _members(entries, generated: Path) -> dict[str, str]:
    """``{name: sha256}`` for recorded files inside this capsule's generated directory.

    Matched on the trailing ``<capsule>/<generated>/<name>`` components rather than the absolute path, so
    a run directory that was moved after execution still joins its own records."""
    members = {}
    for entry in entries or ():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not _sha256(entry.get("sha256")):
            continue
        parts = PurePosixPath(entry["path"]).parts
        if len(parts) >= 3 and parts[-3:-1] == (generated.parent.name, generated.name):
            members[parts[-1]] = entry["sha256"]
    return members


def _relative_parts(value) -> tuple[str, ...]:
    """A recorded path inside the capsule's build directory, as safe components."""
    parts = PurePosixPath(value).parts if isinstance(value, str) and value else ()
    if not parts or PurePosixPath(value).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise TimingArchiveError(f"executed program path escapes its capsule: {value!r}")
    return parts


def _retained_execution(records, engine_sha256):
    """The capsule result's own record of what this engine binary executed (the latest such run)."""
    if not _sha256(engine_sha256) or not isinstance(records, list):
        return None
    runs = [
        row
        for row in records
        if isinstance(row, dict) and row.get("engine_sha256") == engine_sha256 and row.get("status") == "retained"
    ]
    if not runs:
        return None
    row = runs[-1]
    return {
        "source": "capsule_result",
        "engine_sha256": engine_sha256,
        "runs": len(runs),
        "elf_sha256": (row.get("elf") or {}).get("sha256"),
        "kernel_object_sha256": (row.get("kernel_object") or {}).get("sha256"),
        "harness_sha256": (row.get("harness") or {}).get("sha256"),
        "retained": {role: row[role] for role in ("elf", "harness") if isinstance(row.get(role), dict)},
    }


def _recovered_execution(records: list[dict], generated: Path, engine_sha256, elf_name: str, kernel_name: str):
    """What one engine binary executed in this capsule, from the build/run process records.

    A record counts only when its executable digest is the tier's recorded simulator digest and one of its
    inputs is this capsule's linked program. The latest such run is the one the tier result describes."""
    if not _sha256(engine_sha256):
        return None
    runs = [
        row
        for row in records
        if (row.get("executable") or {}).get("sha256") == engine_sha256
        and elf_name in _members(row.get("inputs"), generated)
    ]
    if not runs:
        return None
    elf_sha = _members(runs[-1]["inputs"], generated)[elf_name]
    links = [row for row in records if _members(row.get("outputs"), generated).get(elf_name) == elf_sha]
    link_inputs = _members(links[-1].get("inputs"), generated) if links else {}
    harness_object = link_inputs.get("harness.o")
    harness_builds = [
        row
        for row in records
        if harness_object is not None and _members(row.get("outputs"), generated).get("harness.o") == harness_object
    ]
    return {
        "source": "invocation_records",
        "engine_sha256": engine_sha256,
        "runs": len(runs),
        "elf_sha256": elf_sha,
        "kernel_object_sha256": link_inputs.get(kernel_name),
        "harness_sha256": _members(harness_builds[-1].get("inputs"), generated).get("harness.c")
        if harness_builds
        else None,
    }


def _training_exclusions(row: dict) -> list[str]:
    """Why one gSIM execution cannot label the timing model; empty means it can.

    This judges the MEASUREMENT, independently of the capsule's overall verdict: a capsule may fail a
    host-side or structural rule while its gSIM tier measured and numerically verified the same bytes."""
    reasons = []
    if row["engine"] != "gsim" or row["status"] != "pass":
        reasons.append("no_passing_gsim_tier")
    if not row["measured_now"]:
        reasons.append("carried_not_measured")
    if not row["cycles_measured"]:
        reasons.append("cycles_unmeasured")
    if row["derived_from_rtl"] is not True or row["cycle_accurate"] is not True:
        reasons.append("not_cycle_accurate_rtl")
    if row["cycle_scope"] != "entry-through-completion" or not row["timing_selection_sha256"]:
        reasons.append("kernel_window_unselected")
    if not row["console_cycle_verified"]:
        reasons.append("console_unverified")
    for name in ("elf_sha256", "harness_sha256", "candidate_sha256", "command_buffer_sha256"):
        if not _sha256(row.get(name)):
            reasons.append("identity_missing:" + name.removesuffix("_sha256"))
    if not _sha256((row["sim_provenance"] or {}).get("sha256")):
        reasons.append("identity_missing:simulator")
    for name in (*_REQUIRED_ARTIFACTS, "rtl_console"):
        if not row["artifacts"].get(name):
            reasons.append("artifact_missing:" + name)
    executed = (row["executions"] or {}).get("L3")
    if executed is not None and (
        executed.get("elf_sha256") != row["elf_sha256"]
        or executed.get("harness_sha256") not in (None, row["harness_sha256"])
    ):
        reasons.append("current_program_differs_from_gsim_execution")
    return reasons


def _row(root: Path, path: Path, area: str, visibility: str) -> dict:
    def artifact(location: Path) -> dict | None:
        if location.is_symlink():
            raise TimingArchiveError(f"capsule artifact is a symlink: {location}")
        if not location.is_file():
            return None
        return {
            "path": location.relative_to(root).as_posix(),
            "sha256": _file_sha256(location),
            "bytes": location.stat().st_size,
        }

    if path.is_symlink() or not path.is_file():
        raise TimingArchiveError(f"capsule result is not a regular file: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TimingArchiveError(f"invalid capsule result: {path}")
    tiers = document.get("tiers") or {}
    tier = tiers.get("L3") or {}
    if not isinstance(tier, dict):
        raise TimingArchiveError(f"invalid L3 tier: {path}")
    cycles = tier.get("cycles")
    measured = type(cycles) is int and cycles > 0
    command_buffer = (document.get("command_buffer_artifact") or {}).get("command_buffer") or {}
    tensors = command_buffer.get("tensors") or {}
    generated = path.parent / "generated"
    receipt = {}
    try:
        receipt = json.loads((generated / "readback_build.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    kernel_name = _plain_name(receipt.get("kernel_object_name") if isinstance(receipt, dict) else None, "kernel.o")
    artifacts = {
        name: artifact(location)
        for name, location in {
            "elf": generated / "package_kernel.elf",
            "harness": generated / "harness.c",
            "kernel_object": generated / kernel_name,
            "input_interface": generated / "input.interface.mlir",
            "command_buffer": generated / "command_buffer.json",
            "static_instruction_trace": generated / "instruction_trace.json",
            "numeric_report": generated / "numeric_report.yaml",
            "rtl_console": path.parent / "artifacts" / "rtl_console.log",
            "functional_console": path.parent / "artifacts" / "functional_console.log",
        }.items()
    }
    records = _invocations(generated) if (generated / "invocations").is_dir() else []
    retained = ((document.get("compiler_source") or {}).get("executed_programs")) or []
    executions = {}
    for name in ("L2", "L3"):
        if isinstance(tiers.get(name), dict) and tiers[name].get("measured_now") is not False:
            engine = (tiers[name].get("sim_provenance") or {}).get("sha256")
            executed = _retained_execution(retained, engine) or _recovered_execution(
                records, generated, engine, "package_kernel.elf", kernel_name
            )
            if executed is not None:
                executions[name] = executed
                for role, entry in (executed.get("retained") or {}).items():
                    # The tier's own copy, evidence only while it still holds the recorded bytes.
                    observed = artifact(generated.joinpath(*_relative_parts(entry.get("path"))))
                    artifacts[f"{name}_executed_{role}"] = (
                        observed if observed and observed["sha256"] == entry.get("sha256") else None
                    )
    # Evidence-only annotated-Spike observations of the gSIM-measured program, when a pass wrote them
    # (merlin_experiments.phase1.annotated_spike_evidence). Indexed only while the report names the
    # exact ELF still in place, so a stale observation never rides beside a different program.
    from .annotated_spike_evidence import EVIDENCE_DIR, FUNCTIONAL_PROGRAM_ROLE, PROGRAM_ROLE

    for role, prefix, elf_sha in (
        (PROGRAM_ROLE, "annotated_spike", (artifacts["elf"] or {}).get("sha256")),
        (FUNCTIONAL_PROGRAM_ROLE, "annotated_spike_functional", (executions.get("L2") or {}).get("elf_sha256")),
    ):
        observation = path.parent / EVIDENCE_DIR / role
        for suffix, filename in (
            ("report", "report.json"),
            ("counter_review", "counter_window_review.json"),
            ("scope_binding", "scope_binding.json"),
        ):
            location = observation / filename
            observed = artifact(location)
            try:
                named = json.loads(location.read_text(encoding="utf-8")).get("elf_sha256") if observed else None
            except (OSError, ValueError, AttributeError):
                named = None
            artifacts[prefix + "_" + suffix] = observed if named is not None and named == elf_sha else None
        bound = all(artifacts[prefix + "_" + suffix] for suffix in ("report", "counter_review", "scope_binding"))
        for console in ("functional", "annotated"):
            artifacts[prefix + "_" + console + "_console"] = (
                artifact(observation / (console + "_console.log")) if bound else None
            )
    elf_ref, harness_ref = artifacts["elf"], artifacts["harness"]
    harness = generated / "harness.c"
    scope_matches = _kernel_scopes(harness.read_text(encoding="utf-8")) if harness_ref else []
    selection_sha256 = None
    timing_selection = None
    if len(scope_matches) == 1:
        selected_sha, encoded = scope_matches[0]
        try:
            selected_record = json.loads(base64.b64decode(encoded, validate=True))
        except (ValueError, UnicodeDecodeError):
            selected_record = None
        if (
            type(selected_record) is dict
            and selected_record.get("schema") == "merlin.kernel_call_timing.v1"
            and selected_record.get("selection_sha256") == selected_sha
            and selected_record.get("boundary")
            == "immediately before kernel entry through explicitly selected completion"
            and type(selected_record.get("counter_semantics")) is dict
            and selected_record["counter_semantics"].get("unit") == "cycles"
            and selected_record["counter_semantics"].get("width_bits") == 64
            and selected_record["counter_semantics"].get("wrap") == "modulo"
            and selected_record["counter_semantics"].get("source_pins")
        ):
            selection_sha256 = selected_sha
            timing_selection = selected_record
    rtl_console = path.parent / "artifacts" / "rtl_console.log"
    console_cycles = (
        [
            value
            for line in rtl_console.read_text(encoding="utf-8").splitlines()
            if (value := _cycle_value(line)) is not None
        ]
        if artifacts["rtl_console"]
        else []
    )
    capsule = document.get("capsule")
    interface = artifacts["input_interface"]
    spike, gsim = executions.get("L2") or {}, executions.get("L3") or {}

    def same(field):
        left, right = spike.get(field), gsim.get(field)
        return None if not (_sha256(left) and _sha256(right)) else left == right

    row = {
        "capsule": capsule,
        "result": path.relative_to(root).as_posix(),
        "area": area,
        "visibility": visibility,
        "shape": {
            name: {"dtype": spec.get("dtype"), "shape": spec.get("shape")} for name, spec in sorted(tensors.items())
        },
        "command_buffer_sha256": (document.get("command_buffer_artifact") or {}).get("artifact_sha256"),
        "candidate_sha256": _compiler_candidate(document, generated, artifacts),
        "measured_now": not bool(tier.get("carried")),
        "carried": tier.get("carried"),
        "artifacts": artifacts,
        "elf": elf_ref["path"] if elf_ref else None,
        "elf_sha256": elf_ref["sha256"] if elf_ref else None,
        "harness_sha256": harness_ref["sha256"] if harness_ref else None,
        "timing_selection_sha256": selection_sha256,
        "timing_selection": timing_selection,
        "cycle_scope": "entry-through-completion" if selection_sha256 else None,
        "status": tier.get("status"),
        "result_status": document.get("status"),
        "result_failure": (document.get("failure") or {}).get("category")
        if isinstance(document.get("failure"), dict)
        else None,
        "engine": tier.get("engine"),
        "derived_from_rtl": tier.get("derived_from_rtl"),
        "cycle_accurate": tier.get("cycle_accurate"),
        "cycles": cycles,
        "cycles_measured": measured,
        "console_cycle_verified": len(console_cycles) == 1 and console_cycles[0] == cycles,
        "timing": tier.get("timing"),
        "sim_provenance": tier.get("sim_provenance"),
        "executions": executions,
        # Join keys for pairing this label with a Spike observation. The source group is a proposed
        # leakage group (the capsule and its frozen input interface, shared by every compiler variant);
        # admitting it as a split group remains an operator review.
        "join": {
            "gsim_elf_sha256": gsim.get("elf_sha256"),
            "spike_elf_sha256": spike.get("elf_sha256"),
            "kernel_object_sha256": gsim.get("kernel_object_sha256")
            or (artifacts["kernel_object"] or {}).get("sha256"),
            "spike_kernel_object_sha256": spike.get("kernel_object_sha256"),
            "same_elf": same("elf_sha256"),
            "same_kernel_object": same("kernel_object_sha256"),
            "timing_selection_sha256": selection_sha256,
            "source_group_key": _digest(
                {
                    "schema": "merlin.capsule_source_group.v1",
                    "capsule": capsule,
                    "input_interface_sha256": interface["sha256"],
                }
            )
            if interface and isinstance(capsule, str)
            else None,
            "visibility": visibility,
        },
    }
    exclusions = _training_exclusions(row)
    row["training_eligible"] = not exclusions
    row["training_exclusions"] = exclusions
    return row


def _summary(rows: list[dict]) -> dict:
    gsim_pass = [r for r in rows if r["engine"] == "gsim" and r["status"] == "pass" and r["measured_now"]]
    complete = [
        r
        for r in gsim_pass
        if r["cycles_measured"]
        and all(
            isinstance(r.get(name), str) and _sha256(r[name])
            for name in ("elf_sha256", "harness_sha256", "candidate_sha256", "command_buffer_sha256")
        )
        and r["timing_selection_sha256"]
        and r["console_cycle_verified"]
        and all(r["artifacts"][name] for name in _REQUIRED_ARTIFACTS)
        and isinstance((r["sim_provenance"] or {}).get("sha256"), str)
        and _sha256(r["sim_provenance"]["sha256"])
        and r["result_status"] == "pass"
        and r["derived_from_rtl"] is True
        and r["cycle_accurate"] is True
    ]
    eligible = [r for r in rows if r["training_eligible"]]
    public = [r for r in eligible if r["visibility"] == "public"]
    exclusions: dict[str, int] = {}
    for r in gsim_pass:
        for reason in r["training_exclusions"]:
            exclusions[reason] = exclusions.get(reason, 0) + 1
    return {
        "capsule_records": len(rows),
        "records_by_area": {area: sum(r["area"] == area for r in rows) for _, area, _ in RESULT_AREAS},
        "gsim_pass_records": len(gsim_pass),
        "gsim_reused_records": sum(
            r["engine"] == "gsim" and r["status"] == "pass" and not r["measured_now"] for r in rows
        ),
        "gsim_pass_with_cycles": sum(r["cycles_measured"] for r in gsim_pass),
        "gsim_pass_complete": len(complete),
        "timing_ready": bool(gsim_pass) and len(complete) == len(gsim_pass),
        "training_eligible_records": len(eligible),
        "training_eligible_public_records": len(public),
        "training_eligible_hidden_records": len(eligible) - len(public),
        "training_exclusions": dict(sorted(exclusions.items())),
        "training_ready": bool(public),
        "paired_spike_execution_records": sum(bool(r["join"]["spike_elf_sha256"]) for r in public),
        "same_elf_spike_gsim_records": sum(r["join"]["same_elf"] is True for r in public),
    }


def audit(run_dir: Path) -> dict:
    """Count actual capsule results; a pass with no positive cycle marker is unmeasured."""
    root = run_dir.resolve()
    rows = [_row(root, path, area, visibility) for path, area, visibility in _result_paths(root)]
    return {
        "schema": SCHEMA,
        "run_dir": str(root),
        **_summary(rows),
        "timing_target_scope": "selected_gsim_elaboration_only",
        "cycle_scope": "kernel entry through selected completion; excludes output publication",
        "training_scope": (
            "per-record: a fresh, console-verified, kernel-window gSIM measurement whose gSIM tier passed "
            "numerics with every identity present, independently of the capsule's overall verdict"
        ),
        "visibility_policy": "grading_hidden results are hidden and never enter public calibration",
        "observer_artifacts": _observer_artifacts(root, rows),
        "records": rows,
    }


def _observer_artifacts(root: Path, rows: list[dict]) -> dict:
    """Only the observer's fixed public control documents, never binaries or arbitrary referenced paths."""
    from .annotated_spike_evidence import EVIDENCE_SCHEMA, RETAINED_INPUT_FILES

    summary_path = root / "annotated_spike_evidence.json"
    if not summary_path.exists():
        return {}

    def reference(path, expected=None):
        if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
            raise TimingArchiveError(f"observer artifact is a symlink: {path}")
        if not path.is_file() or path.stat().st_size > _MAX_OBSERVER_DOCUMENT_BYTES:
            raise TimingArchiveError(f"observer control document is missing or oversized: {path}")
        sha = _file_sha256(path)
        if expected is not None and sha != expected:
            raise TimingArchiveError(f"observer control document differs from its recorded identity: {path}")
        return {"path": path.relative_to(root).as_posix(), "sha256": sha, "bytes": path.stat().st_size}

    artifacts = {"summary": reference(summary_path)}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    public_results = {row["result"] for row in rows if row["visibility"] == "public"}
    if (
        not isinstance(summary, dict)
        or summary.get("schema") != EVIDENCE_SCHEMA
        or not isinstance(summary.get("observations"), list)
        or any(not isinstance(row, dict) or row.get("result") not in public_results for row in summary["observations"])
    ):
        raise TimingArchiveError("observer summary does not bind exclusively to this run's public results")
    retained = summary.get("retained_inputs")
    if retained is not None:
        selected = summary.get("selected_inputs")
        if (
            not isinstance(retained, dict)
            or set(retained) != set(RETAINED_INPUT_FILES)
            or not isinstance(selected, dict)
            or any(
                not isinstance(pin, dict)
                or not isinstance(selected.get(name), dict)
                or pin.get("sha256") != selected[name].get("sha256")
                for name, pin in retained.items()
            )
        ):
            raise TimingArchiveError("retained observer controls differ from the selected input identities")
    for name, filename, pin in (
        ("model", "count_only_model.json", summary.get("model")),
        *(
            (name, "observer_inputs/" + filename, pin)
            for name, filename in RETAINED_INPUT_FILES.items()
            if (pin := (summary.get("retained_inputs") or {}).get(name)) is not None
        ),
    ):
        if not isinstance(pin, dict) or not _sha256(pin.get("sha256")):
            raise TimingArchiveError(f"observer control document has no recorded identity: {name}")
        artifacts[name] = reference(root / filename, pin["sha256"])
    return artifacts


def _write_tar(destination: Path, index: dict, members: list[tuple[str, Path, dict | None]], root: Path) -> None:
    """Write ``index.json`` and members atomically; an indexed member must still hold its indexed bytes."""
    if destination.exists():
        raise TimingArchiveError(f"archive already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="capsule-timing-", suffix=".tar", dir=destination.parent, delete=False
    ) as temporary:
        partial = Path(temporary.name)
    try:
        with tarfile.open(partial, "w") as tar:
            payload = (json.dumps(index, sort_keys=True, indent=2) + "\n").encode("utf-8")
            member = tarfile.TarInfo("index.json")
            member.size = len(payload)
            member.mode = 0o644
            tar.addfile(member, io.BytesIO(payload))
            for arcname, path, expected in members:
                data = path.read_bytes()
                if expected is not None and (
                    len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]
                ):
                    raise TimingArchiveError(f"capsule artifact changed after its audit: {path}")
                member = tar.gettarinfo(path, arcname=arcname)
                member.size = len(data)
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ""
                tar.addfile(member, io.BytesIO(data))
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def archive(run_dir: Path, destination: Path, *, require_ready: bool = True) -> dict:
    """Package complete capsule result directories after timing readiness succeeds."""
    index = audit(run_dir)
    if require_ready and not index["timing_ready"]:
        raise TimingArchiveError(
            f"gSIM capsule timing incomplete: {index['gsim_pass_with_cycles']}/"
            f"{index['gsim_pass_records']} passing records have positive cycle markers; "
            f"{index['gsim_pass_complete']}/{index['gsim_pass_records']} have ELF, compiler, command-buffer, "
            "simulator and kernel-scope identities, input/numeric artifacts and one matching console marker"
        )
    if not require_ready:
        index = {**index, "export_scope": "diagnostic_all_results; timing readiness is not implied"}
    if destination.exists():
        raise TimingArchiveError(f"archive already exists: {destination}")
    root = Path(index["run_dir"])
    capsule_dirs = sorted({(root / row["result"]).parent for row in index["records"]})
    files = []
    for directory in capsule_dirs:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise TimingArchiveError(f"capsule artifact is a symlink: {path}")
            if path.is_file():
                files.append(path)
            elif not path.is_dir():
                raise TimingArchiveError(f"capsule artifact is not a regular file: {path}")
    members = [("run/" + p.relative_to(root).as_posix(), p, None) for p in files]
    members.extend(("run/" + ref["path"], root / ref["path"], ref) for ref in index["observer_artifacts"].values())
    _write_tar(destination, index, members, root)
    return {
        **index,
        "archive": str(destination),
        "archive_sha256": _file_sha256(destination),
        "artifact_files": len(members),
    }


def training_index(index: dict) -> dict:
    """The public training-eligible subset of an audit, with every executed repeat of the same programs.

    A repeat whose gSIM tier executed the selected ELF stays in the subset whatever its verdict, so a
    consumer can refuse to drop a slower or failed execution of the same bytes. A result whose gSIM tier
    never executed is not a repeat; carried (copied) verdicts are not observations; and hidden capsules
    never enter a public calibration subset."""
    selected = [r for r in index["records"] if r["training_eligible"] and r["visibility"] == "public"]
    programs = {r["elf_sha256"] for r in selected}
    rows = [
        r
        for r in index["records"]
        if r["visibility"] == "public"
        and r["measured_now"]
        and r["status"] in ("pass", "fail")
        and r["elf_sha256"] in programs
    ]
    return {
        **{key: value for key, value in index.items() if key != "records"},
        **_summary(rows),
        "export_scope": TRAINING_SCOPE,
        "source_audit": {
            "capsule_records": index["capsule_records"],
            "training_eligible_records": index["training_eligible_records"],
            "training_eligible_hidden_records": index["training_eligible_hidden_records"],
            "sha256": _digest(index),
        },
        "records": rows,
    }


def training_archive(run_dir: Path, destination: Path, *, index: dict | None = None) -> dict:
    """Package only the indexed evidence of public training-eligible records (and their repeats)."""
    subset = training_index(audit(run_dir) if index is None else index)
    if not subset["training_ready"]:
        raise TimingArchiveError("no public training-eligible gSIM record to archive")
    root = Path(subset["run_dir"])
    members, seen = [], set()
    for row in subset["records"]:
        for name, reference in [("result", {"path": row["result"]}), *row["artifacts"].items()]:
            if reference is None or reference["path"] in seen:
                continue
            seen.add(reference["path"])
            path = root / reference["path"]
            if path.is_symlink() or not path.is_file():
                raise TimingArchiveError(f"indexed capsule artifact is not a regular file: {path}")
            members.append(("run/" + reference["path"], path, reference if name != "result" else None))
    members.extend(("run/" + ref["path"], root / ref["path"], ref) for ref in subset["observer_artifacts"].values())
    _write_tar(destination, subset, members, root)
    return {
        **subset,
        "archive": str(destination),
        "archive_sha256": _file_sha256(destination),
        "artifact_files": len(members),
    }


def finalize_run(run_dir: Path) -> dict:
    """Persist the audit, its training subset and, only for a complete measured run, the full timing tar.

    Safe to call again on the same run (e.g. by an operator after stopping a supervisor): archives are
    content-named and an existing archive is never rewritten."""
    run_dir = Path(run_dir)
    status_path = run_dir / "capsule_timing_archive_status.json"
    try:
        index = audit(run_dir)
        (run_dir / "capsule_timing_audit.json").write_text(json.dumps(index, sort_keys=True, indent=2) + "\n")
        if index["timing_ready"]:
            destination = run_dir / "capsule_timing.tar"
            if destination.exists():
                with tarfile.open(destination) as existing:
                    member = existing.extractfile("index.json")
                    previous = json.load(member) if member is not None else None
                if previous != index:
                    digest = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()[:12]
                    destination = run_dir / f"capsule_timing_{digest}.tar"
            if not destination.exists():
                archive(run_dir, destination)
            status = {
                "timing_ready": True,
                "archive": str(destination),
                "archive_sha256": _file_sha256(destination),
                "error": None,
            }
        else:
            status = {
                "timing_ready": False,
                "archive": None,
                "archive_sha256": None,
                "error": (
                    f"{index['gsim_pass_complete']}/{index['gsim_pass_records']} passing gSIM records "
                    "have measured cycles and required identities"
                ),
            }
        if index["timing_ready"]:
            status["evidence_archive"] = status["archive"]
            status["evidence_archive_sha256"] = status["archive_sha256"]
        else:
            # Preserve paid native evidence even if a compiler version failed,
            # an optional promotion omitted another tier, or custody is missing.
            # The distinct diagnostic tar never becomes a training-ready claim.
            digest = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()[:12]
            evidence = run_dir / f"capsule_evidence_{digest}.tar"
            if not evidence.exists():
                archive(run_dir, evidence, require_ready=False)
            status["evidence_archive"] = str(evidence)
            status["evidence_archive_sha256"] = _file_sha256(evidence)
        status.update(
            training_ready=index["training_ready"],
            training_eligible_public_records=index["training_eligible_public_records"],
            training_eligible_hidden_records=index["training_eligible_hidden_records"],
            training_archive=None,
            training_archive_sha256=None,
        )
        if index["training_ready"]:
            # A failure here (e.g. evidence rewritten after the audit) must not hide the archives above.
            try:
                subset = training_index(index)
                training = run_dir / f"capsule_timing_training_{_digest(subset)[:12]}.tar"
                if not training.exists():
                    training_archive(run_dir, training, index=index)
                status["training_archive"] = str(training)
                status["training_archive_sha256"] = _file_sha256(training)
            except (OSError, ValueError, tarfile.TarError) as exc:
                status["training_error"] = f"{type(exc).__name__}: {exc}"
    except (OSError, ValueError, tarfile.TarError) as exc:
        status = {
            "timing_ready": False,
            "archive": None,
            "archive_sha256": None,
            "training_ready": False,
            "training_archive": None,
            "training_archive_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    status_path.write_text(json.dumps(status, sort_keys=True, indent=2) + "\n")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--out", type=Path, help="write a host-only .tar after positive gSIM cycle validation")
    action.add_argument("--training-out", type=Path, help="write the public training-eligible subset .tar")
    action.add_argument(
        "--finalize",
        action="store_true",
        help="write the run's audit, status and archives (for a run whose supervisor stopped early)",
    )
    args = parser.parse_args(argv)
    try:
        if args.finalize:
            status = finalize_run(args.run_dir)
            print(json.dumps(status, sort_keys=True, indent=2))
            return 0 if status.get("timing_ready") or status.get("training_archive") else 1
        if args.training_out:
            result = training_archive(args.run_dir, args.training_out)
        else:
            result = archive(args.run_dir, args.out) if args.out else audit(args.run_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"capsule timing archive: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["timing_ready"] or (args.training_out and result["training_ready"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
