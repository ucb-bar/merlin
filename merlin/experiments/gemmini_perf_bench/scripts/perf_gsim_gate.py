"""Fail-closed GSIM qualification for the agentic performance experiment.

GSIM and Verilator are two implementations of the same elaborated RTL.  GSIM is therefore allowed to
provide final cycle evidence, but only after an engine-equivalence certificate proves byte-identical
results on the same ELF and pins every artifact which gives that claim meaning.  This module is the
orchestrator-facing boundary for that rule; it does not launch either simulator.

The certificate envelope is deliberately an exact set of canonical workload descriptors.  It never
turns six observed shapes into an inferred numeric range.  Work outside the set is routed to Verilator
with a recorded reason, while an eligible development evaluation is required to use GSIM and may not
silently fall back.  A deterministic, predeclared Verilator subset corroborates GSIM in the final run.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "merlin.gsim-equivalence.v1"
FIDELITY = "elaborated_rtl_cycle_accurate"
GSIM_ENGINE = "gsim"
REFERENCE_ENGINE = "verilator"
STRONG_EVIDENCE = "output_bytes"
OUTPUT_ENCODING = "command_buffer_declared_tensor_little_endian.v1"
BUILD_RECEIPT_SCHEMA = "merlin.gsim-model-build.v2"
CERTIFICATE_NAMES = ("gsim_equivalence_certificate.json",)
RAW_REPORT_NAMES = ("xval_bytes.json", "xval_gm.json", "xval_gm_bytes.json")
PHASES = frozenset(("development_correctness", "final_correctness", "final_performance"))
REQUIRED_PINS = frozenset((
    "gsim_firrtl", "verilator_firrtl", "gsim_model", "gsim_binary", "verilator_binary"))


class GsimGateError(RuntimeError):
    """Certificate or execution evidence cannot support the requested claim."""


def canonical_json(value: Any) -> str:
    """Stable JSON used for every commitment made by this module."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _json_value(value: Any, *, where: str) -> Any:
    """Return a canonical JSON value, rejecting lossy or ambiguous Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GsimGateError(f"{where} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, where=f"{where}[]") for item in value]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value, key=str):
            if not isinstance(key, str) or not key:
                raise GsimGateError(f"{where} has a non-string or empty key")
            out[key] = _json_value(value[key], where=f"{where}.{key}")
        return out
    raise GsimGateError(f"{where} contains unsupported {type(value).__name__}")


def canonical_workload(workload: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical exact operation/shape/semantic identity used by the envelope."""
    if not isinstance(workload, Mapping):
        raise GsimGateError("workload must be a mapping")
    operation = workload.get("operation")
    shape = workload.get("shape")
    semantics = workload.get("semantics")
    if not isinstance(operation, str) or not operation.strip():
        raise GsimGateError("workload.operation must be a non-empty string")
    if not isinstance(shape, Mapping) or not shape:
        raise GsimGateError("workload.shape must be a non-empty mapping")
    if not isinstance(semantics, Mapping) or not semantics:
        raise GsimGateError("workload.semantics must be a non-empty mapping")
    return {
        "operation": operation.strip(),
        "semantics": _json_value(semantics, where="workload.semantics"),
        "shape": _json_value(shape, where="workload.shape"),
    }


def workload_sha256(workload: Mapping[str, Any]) -> str:
    return _digest_bytes(canonical_json(canonical_workload(workload)).encode("utf-8"))


def _resolve_pin_path(certificate_path: Path, declared: Any) -> Path:
    if not isinstance(declared, str) or not declared.strip():
        raise GsimGateError("artifact pin path must be a non-empty string")
    path = Path(declared)
    return path if path.is_absolute() else certificate_path.parent / path


@dataclass(frozen=True)
class CertificateRecord:
    """Validated certificate plus the digest of its exact bytes."""

    path: Path
    sha256: str
    target: str
    pins: Mapping[str, Mapping[str, str]]
    members: Mapping[str, Mapping[str, Any]]
    unresolved: Mapping[str, str]
    document: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "target": self.target,
            "certified_workloads": len(self.members),
            "unresolved_workloads": len(self.unresolved),
            "fidelity": FIDELITY,
        }


def _validate_pin(name: str, raw: Any, *, certificate_path: Path,
                  artifact_paths: Mapping[str, str | Path] | None) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise GsimGateError(f"pins.{name} must be a mapping")
    claimed = raw.get("sha256")
    if not _is_sha256(claimed):
        raise GsimGateError(f"pins.{name}.sha256 is not a lowercase SHA-256")
    if artifact_paths is not None and name in artifact_paths:
        path = Path(artifact_paths[name])
    else:
        path = _resolve_pin_path(certificate_path, raw.get("path"))
    if not path.is_file():
        raise GsimGateError(f"pinned {name} artifact is absent: {path}")
    actual = _digest_file(path)
    if actual != claimed:
        raise GsimGateError(
            f"pinned {name} digest mismatch: certificate={claimed}, actual={actual}, path={path}")
    return {"path": str(path.resolve()), "sha256": actual}


def _validate_member(raw: Any, *, pins: Mapping[str, Mapping[str, str]], index: int) -> tuple[str, dict]:
    where = f"members[{index}]"
    if not isinstance(raw, Mapping):
        raise GsimGateError(f"{where} must be a mapping")
    workload = canonical_workload(raw.get("workload"))
    identity = workload_sha256(workload)
    if raw.get("workload_sha256") != identity:
        raise GsimGateError(f"{where}.workload_sha256 does not commit its workload descriptor")
    elf = raw.get("elf_sha256")
    if not _is_sha256(elf):
        raise GsimGateError(f"{where}.elf_sha256 is not a lowercase SHA-256")
    if raw.get("agreement") != "AGREE" or raw.get("evidence") != STRONG_EVIDENCE:
        raise GsimGateError(f"{where} lacks byte-level AGREE evidence")
    if raw.get("bytes_match") is not True:
        raise GsimGateError(f"{where}.bytes_match must be true")

    expected = (("reference", REFERENCE_ENGINE, "verilator_binary"),
                ("candidate", GSIM_ENGINE, "gsim_binary"))
    for side, engine, binary_pin in expected:
        run = raw.get(side)
        if not isinstance(run, Mapping):
            raise GsimGateError(f"{where}.{side} must be a mapping")
        if run.get("engine") != engine or run.get("ran") is not True or run.get("verdict") != "pass":
            raise GsimGateError(f"{where}.{side} is not a passing {engine} run")
        if run.get("elf_sha256") != elf:
            raise GsimGateError(
                f"{where} did not run the same ELF on {REFERENCE_ENGINE} and {GSIM_ENGINE}")
        if run.get("binary_sha256") != pins[binary_pin]["sha256"]:
            raise GsimGateError(f"{where}.{side} does not name the pinned {binary_pin}")
        firrtl_pin = "gsim_firrtl" if engine == GSIM_ENGINE else "verilator_firrtl"
        if run.get("firrtl_sha256") != pins[firrtl_pin]["sha256"]:
            raise GsimGateError(f"{where}.{side} does not name the pinned {firrtl_pin}")
        if engine == GSIM_ENGINE and run.get("model_sha256") != pins["gsim_model"]["sha256"]:
            raise GsimGateError(f"{where}.{side} does not name the pinned GSIM model")
        if run.get("derived_from_rtl") is not True or run.get("cycle_accurate") is not True:
            raise GsimGateError(f"{where}.{side} is not cycle-accurate elaborated RTL")
        if not _is_sha256(run.get("output_sha256")):
            raise GsimGateError(f"{where}.{side} has no output-byte SHA-256")
        if run.get("output_encoding") != OUTPUT_ENCODING or not isinstance(
                run.get("output_tensors"), list) or not run["output_tensors"]:
            raise GsimGateError(f"{where}.{side} has no declared-tensor byte encoding")
    if raw["reference"]["output_sha256"] != raw["candidate"]["output_sha256"]:
        raise GsimGateError(f"{where} output-byte SHA-256 differs between engines")
    if raw["reference"]["output_tensors"] != raw["candidate"]["output_tensors"]:
        raise GsimGateError(f"{where} declared output tensor byte records differ between engines")
    return identity, {**dict(raw), "workload": workload}


def _validate_build_binding(raw: Any, *, certificate_path: Path,
                            pins: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise GsimGateError("certificate lacks the GSIM build binding")
    path = _resolve_pin_path(certificate_path, raw.get("path"))
    claimed = raw.get("sha256")
    if not _is_sha256(claimed) or not path.is_file() or _digest_file(path) != claimed:
        raise GsimGateError("GSIM build receipt is absent or does not match its certificate pin")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GsimGateError(f"cannot read pinned GSIM build receipt: {exc}") from exc
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != BUILD_RECEIPT_SCHEMA \
            or receipt.get("status") != "complete":
        raise GsimGateError("pinned GSIM build receipt is incomplete or has the wrong schema")
    expected = {
        "firrtl_sha256": pins["gsim_firrtl"]["sha256"],
        "model_manifest_sha256": pins["gsim_model"]["sha256"],
        "binary_sha256": pins["gsim_binary"]["sha256"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise GsimGateError(f"pinned GSIM build receipt does not bind {key}")
    artifacts, tools, inputs = receipt.get("artifacts"), receipt.get("tools"), receipt.get("inputs")
    if not isinstance(artifacts, Mapping) or not isinstance(tools, Mapping) \
            or set(tools) != {"gsim_emitter", "cxx_wrapper", "cxx_compiler"} \
            or not isinstance(inputs, list) or not inputs:
        raise GsimGateError("pinned GSIM build receipt lacks complete artifact/tool/input pins")
    input_digest = _digest_bytes(canonical_json(inputs).encode("utf-8"))
    if receipt.get("inputs_sha256") != input_digest:
        raise GsimGateError("pinned GSIM build input commitment is invalid")
    for rows in (artifacts.values(), tools.values(), inputs):
        for item in rows:
            if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
                raise GsimGateError("pinned GSIM build receipt contains a malformed artifact pin")
            item_path = Path(str(item.get("path") or ""))
            if item_path.is_symlink() or not item_path.is_file() \
                    or _digest_file(item_path) != item["sha256"]:
                raise GsimGateError(f"pinned GSIM build input is absent or changed: {item_path}")
    commands = receipt.get("commands")
    if not isinstance(commands, list) or not commands:
        raise GsimGateError("pinned GSIM build receipt contains no ordered command transcript")
    stages = []
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            raise GsimGateError(f"pinned GSIM build command {index} is malformed")
        stage, cwd, argv = command.get("stage"), command.get("cwd"), command.get("argv")
        if not isinstance(stage, str) or not isinstance(cwd, str) or not Path(cwd).is_absolute() \
                or not isinstance(argv, list) or not argv \
                or not all(isinstance(arg, str) for arg in argv):
            raise GsimGateError(f"pinned GSIM build command {index} lacks stage/cwd/exact argv")
        stages.append(stage)
    if "elaborate" not in stages or "emit" not in stages or "compile" not in stages \
            or stages[-1] != "link":
        raise GsimGateError("pinned GSIM build command transcript is incomplete or unordered")
    command_digest = _digest_bytes(canonical_json(commands).encode("utf-8"))
    if receipt.get("commands_sha256") != command_digest or raw.get("commands_sha256") != command_digest:
        raise GsimGateError("pinned GSIM build command commitment is invalid")
    return {"path": str(path.resolve()), "sha256": claimed,
            "commands_sha256": command_digest}


def load_certificate(path: str | Path, *,
                     artifact_paths: Mapping[str, str | Path] | None = None,
                     expected_sha256: str | None = None) -> CertificateRecord:
    """Load, content-address, and validate one certificate and all pinned artifacts.

    ``artifact_paths`` lets a sealed experiment resolve logical pin names without trusting paths embedded
    in a copied certificate.  When omitted, relative paths are resolved beside the certificate.
    """
    certificate_path = Path(path)
    try:
        raw_bytes = certificate_path.read_bytes()
        doc = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise GsimGateError(f"cannot read GSIM certificate {certificate_path}: {exc}") from exc
    digest = _digest_bytes(raw_bytes)
    if expected_sha256 is not None and digest != expected_sha256:
        raise GsimGateError(
            f"GSIM certificate digest mismatch: expected={expected_sha256}, actual={digest}")
    if not isinstance(doc, Mapping):
        raise GsimGateError("GSIM certificate root must be a mapping")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise GsimGateError(
            f"unsupported GSIM certificate schema {doc.get('schema_version')!r}; expected {SCHEMA_VERSION}")
    if doc.get("status") != "certified":
        raise GsimGateError(f"GSIM certificate status is {doc.get('status')!r}, not 'certified'")
    target = doc.get("target")
    if not isinstance(target, str) or not target:
        raise GsimGateError("GSIM certificate target is absent")
    if doc.get("fidelity") != FIDELITY:
        raise GsimGateError(f"GSIM certificate fidelity must be {FIDELITY!r}")
    if doc.get("primary_engine") != GSIM_ENGINE or doc.get("reference_engine") != REFERENCE_ENGINE:
        raise GsimGateError("certificate must compare GSIM against Verilator")

    raw_pins = doc.get("pins")
    if not isinstance(raw_pins, Mapping) or set(raw_pins) != REQUIRED_PINS:
        raise GsimGateError(f"certificate pins must be exactly {sorted(REQUIRED_PINS)}")
    pins = {name: _validate_pin(name, raw_pins[name], certificate_path=certificate_path,
                                artifact_paths=artifact_paths)
            for name in sorted(REQUIRED_PINS)}
    _validate_build_binding(doc.get("build_binding"), certificate_path=certificate_path, pins=pins)

    raw_members = doc.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise GsimGateError("certificate must contain at least one agreeing workload member")
    members: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_members):
        identity, member = _validate_member(raw, pins=pins, index=index)
        if identity in members:
            raise GsimGateError(f"duplicate certified workload {identity}")
        members[identity] = member

    unresolved: dict[str, str] = {}
    raw_unresolved = doc.get("unresolved", [])
    if not isinstance(raw_unresolved, list):
        raise GsimGateError("certificate.unresolved must be a list")
    for index, item in enumerate(raw_unresolved):
        if not isinstance(item, Mapping):
            raise GsimGateError(f"unresolved[{index}] must be a mapping")
        identity = workload_sha256(canonical_workload(item.get("workload")))
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise GsimGateError(f"unresolved[{index}].reason must be non-empty")
        if identity in members or identity in unresolved:
            raise GsimGateError(f"workload {identity} is duplicated across the certificate envelope")
        unresolved[identity] = reason.strip()

    return CertificateRecord(certificate_path.resolve(), digest, target, pins, members, unresolved, doc)


def discover_certificate(roots: Iterable[str | Path], *, target: str | None = None,
                         artifact_paths: Mapping[str, str | Path] | None = None,
                         expected_sha256: str | None = None) -> CertificateRecord:
    """Discover a unique valid certificate under declared roots.

    Distinct certificates are never ordered by timestamp: choosing the newest after seeing results is
    post-hoc selection.  Supply ``expected_sha256`` to bind an experiment to one certificate.
    Byte-identical copies collapse to one candidate.
    """
    paths: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file() and root.name in CERTIFICATE_NAMES:
            paths.append(root)
        elif root.is_dir():
            for name in CERTIFICATE_NAMES:
                paths.extend(root.rglob(name))
    paths = sorted(set(path.resolve() for path in paths), key=str)
    if not paths:
        raise GsimGateError(
            f"no GSIM certificate named one of {list(CERTIFICATE_NAMES)} under declared roots")

    by_digest: dict[str, CertificateRecord] = {}
    failures: list[str] = []
    for path in paths:
        try:
            record = load_certificate(path, artifact_paths=artifact_paths,
                                      expected_sha256=expected_sha256)
        except GsimGateError as exc:
            failures.append(f"{path}: {exc}")
            continue
        if target is not None and record.target != target:
            failures.append(f"{path}: target {record.target!r} does not match {target!r}")
            continue
        by_digest.setdefault(record.sha256, record)
    if not by_digest:
        raise GsimGateError("no discovered GSIM certificate validated: " + "; ".join(failures))
    if len(by_digest) != 1:
        raise GsimGateError(
            "multiple distinct valid GSIM certificates discovered; precommit one SHA-256: "
            + ", ".join(sorted(by_digest)))
    return next(iter(by_digest.values()))


@dataclass(frozen=True)
class CrossValidationInventory:
    """Content-addressed inventory of an older cross-engine report.

    These reports are valuable byte-agreement evidence, but they predate artifact/ELF pins.  Inventorying
    one must therefore never turn it into a certificate implicitly.
    """

    path: Path
    sha256: str
    target: str
    agreeing_capsules: tuple[str, ...]
    unresolved_capsules: tuple[str, ...]
    qualifying: bool
    missing_for_qualification: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "target": self.target,
            "agreeing_capsules": list(self.agreeing_capsules),
            "unresolved_capsules": list(self.unresolved_capsules),
            "qualifying": self.qualifying,
            "missing_for_qualification": list(self.missing_for_qualification),
        }


def inspect_cross_validation_report(path: str | Path) -> CrossValidationInventory:
    """Validate and inventory the current cross-validator format without overclaiming it.

    A report can establish byte agreement even when its overall exit is incomplete because its original
    plan included unrelated capsules with no staged artifact.  Only explicit ``AGREE`` rows enter this
    inventory.  They still cannot qualify GSIM until a v1 certificate binds exact workload descriptors,
    the shared ELF, both FIRRTLs, generated model, and binaries.
    """
    report_path = Path(path)
    try:
        raw = report_path.read_bytes()
        doc = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise GsimGateError(f"cannot read cross-validation report {report_path}: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise GsimGateError("cross-validation report root must be a mapping")
    if doc.get("reference_engine") != REFERENCE_ENGINE or doc.get("candidate_engine") != GSIM_ENGINE:
        raise GsimGateError("cross-validation report is not GSIM against Verilator")
    target = doc.get("target")
    if not isinstance(target, str) or not target:
        raise GsimGateError("cross-validation report target is absent")
    rows = doc.get("capsules")
    if not isinstance(rows, list) or not rows:
        raise GsimGateError("cross-validation report contains no capsule rows")
    agreeing: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GsimGateError(f"cross-validation capsule row {index} is not a mapping")
        capsule = row.get("capsule")
        if not isinstance(capsule, str) or not capsule or capsule in seen:
            raise GsimGateError(f"cross-validation capsule row {index} has an absent/duplicate identity")
        seen.add(capsule)
        if row.get("agreement") == "AGREE":
            if row.get("evidence") != STRONG_EVIDENCE or row.get("bytes_match") is not True:
                raise GsimGateError(f"{capsule} claims AGREE without exact output-byte evidence")
            for side, engine in (("reference", REFERENCE_ENGINE), ("candidate", GSIM_ENGINE)):
                run = row.get(side)
                if (not isinstance(run, Mapping) or run.get("engine") != engine
                        or run.get("ran") is not True or run.get("verdict") != "pass"):
                    raise GsimGateError(f"{capsule} has no passing {engine} run")
            agreeing.append(capsule)
        else:
            unresolved.append(capsule)
    missing_items = [
        "canonical operation/shape/semantic descriptors",
        "shared ELF SHA-256 per agreeing capsule",
        "GSIM FIRRTL SHA-256",
        "Verilator FIRRTL SHA-256",
        "GSIM generated-model SHA-256",
        "GSIM and Verilator binary SHA-256 pins",
    ]
    if not agreeing:
        missing_items.insert(0, "no byte-level agreeing capsule")
    return CrossValidationInventory(report_path.resolve(), _digest_bytes(raw), target,
                                    tuple(sorted(agreeing)), tuple(sorted(unresolved)), False,
                                    tuple(missing_items))


def discover_cross_validation_reports(roots: Iterable[str | Path], *,
                                      target: str | None = None) -> tuple[CrossValidationInventory, ...]:
    """Discover and content-address all current-format byte-agreement reports."""
    paths: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file() and root.name in RAW_REPORT_NAMES:
            paths.add(root.resolve())
        elif root.is_dir():
            for name in RAW_REPORT_NAMES:
                paths.update(path.resolve() for path in root.rglob(name))
    inventories = []
    for path in sorted(paths, key=str):
        inventory = inspect_cross_validation_report(path)
        if target is None or inventory.target == target:
            inventories.append(inventory)
    return tuple(inventories)


@dataclass(frozen=True)
class EvaluationDecision:
    """Auditable engine decision for one exact workload and phase."""

    workload: Mapping[str, Any]
    workload_sha256: str
    phase: str
    eligible: bool
    admitted: bool
    selected_engine: str | None
    use_gsim: bool
    fallback_reason: str | None
    refusal_reason: str | None
    certificate_sha256: str
    final_cycle_authority: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": dict(self.workload),
            "workload_sha256": self.workload_sha256,
            "phase": self.phase,
            "eligible": self.eligible,
            "admitted": self.admitted,
            "selected_engine": self.selected_engine,
            "use_gsim": self.use_gsim,
            "fallback_reason": self.fallback_reason,
            "refusal_reason": self.refusal_reason,
            "certificate_sha256": self.certificate_sha256,
            "fidelity": FIDELITY,
            "final_cycle_authority": self.final_cycle_authority,
        }


def plan_evaluation(certificate: CertificateRecord, workload: Mapping[str, Any], *, phase: str,
                    gsim_available: bool, fallback_engine: str = REFERENCE_ENGINE) -> EvaluationDecision:
    """Require GSIM for final timing; permit Verilator fallback only during development.

    Verilator remains useful as an independent correctness oracle, but an out-of-envelope workload is
    not a qualified performance measurement.  This distinction lives in the shared policy layer so a
    caller cannot accidentally promote a Verilator cycle count to the final timing authority.
    """
    if phase not in PHASES:
        raise GsimGateError(f"unknown evaluation phase {phase!r}")
    canonical = canonical_workload(workload)
    identity = workload_sha256(canonical)
    eligible = identity in certificate.members
    if eligible and not gsim_available:
        reason = ("workload is inside the certified GSIM envelope, but the pinned GSIM engine is "
                  "unavailable; fallback is forbidden for eligible work")
        return EvaluationDecision(canonical, identity, phase, True, False, None, False, None, reason,
                                  certificate.sha256, False)
    if eligible:
        return EvaluationDecision(canonical, identity, phase, True, True, GSIM_ENGINE, True, None, None,
                                  certificate.sha256, phase == "final_performance")
    if fallback_engine != REFERENCE_ENGINE:
        raise GsimGateError("the only qualified out-of-envelope fallback is Verilator")
    if identity in certificate.unresolved:
        reason = "certificate explicitly leaves this workload unresolved: " + certificate.unresolved[identity]
    else:
        reason = "workload is outside the certificate's exact operation/shape/semantic envelope"
    if phase == "final_performance":
        refusal = reason + "; final timing requires a GSIM certificate covering this exact workload"
        return EvaluationDecision(canonical, identity, phase, False, False, None, False, None,
                                  refusal, certificate.sha256, False)
    return EvaluationDecision(canonical, identity, phase, False, True, fallback_engine, False, reason,
                              None, certificate.sha256, False)


def validate_execution(certificate: CertificateRecord, decision: EvaluationDecision,
                       execution: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one result against its predeclared engine decision and return an audit record."""
    if decision.certificate_sha256 != certificate.sha256:
        raise GsimGateError("evaluation decision names a different GSIM certificate")
    if not decision.admitted or decision.selected_engine is None:
        raise GsimGateError(decision.refusal_reason or "evaluation was refused")
    engine = execution.get("engine")
    if engine != decision.selected_engine:
        if decision.eligible:
            raise GsimGateError(
                f"eligible {decision.phase} evaluation must use GSIM, not {engine!r}")
        raise GsimGateError(f"out-of-envelope evaluation must use the recorded Verilator fallback")
    if execution.get("status") != "pass":
        raise GsimGateError(f"{engine} execution status is not pass")
    if execution.get("derived_from_rtl") is not True or execution.get("cycle_accurate") is not True:
        raise GsimGateError(f"{engine} result is not cycle-accurate elaborated-RTL evidence")
    expected_pin = "gsim_binary" if engine == GSIM_ENGINE else "verilator_binary"
    if execution.get("binary_sha256") != certificate.pins[expected_pin]["sha256"]:
        raise GsimGateError(f"execution does not name the pinned {expected_pin}")
    firrtl_pin = "gsim_firrtl" if engine == GSIM_ENGINE else "verilator_firrtl"
    if execution.get("firrtl_sha256") != certificate.pins[firrtl_pin]["sha256"]:
        raise GsimGateError(f"execution does not name the pinned {firrtl_pin}")
    if engine == GSIM_ENGINE and execution.get("model_sha256") != certificate.pins["gsim_model"]["sha256"]:
        raise GsimGateError("execution does not name the pinned GSIM model")
    elf = execution.get("elf_sha256")
    if not _is_sha256(elf):
        raise GsimGateError("execution lacks an exact ELF SHA-256")
    cycles = execution.get("cycles")
    if decision.phase == "final_performance":
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise GsimGateError("final performance execution lacks a positive integer cycle count")
    cycle_claim_authority = (
        FIDELITY if decision.phase == "final_performance" and engine == GSIM_ENGINE else None)
    return {
        "decision": decision.to_dict(),
        "execution": dict(execution),
        "admitted": True,
        "cycle_claim_authority": cycle_claim_authority,
    }


def validate_corroboration(certificate: CertificateRecord, predeclared: Mapping[str, Any],
                           primary: Mapping[str, Any],
                           corroborating: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one predeclared GSIM/Verilator same-ELF output-byte corroboration.

    Wall time and cycle counts may differ between simulator implementations.  Correctness bytes may not.
    This is independent corroboration of the engine, never a second performance sample to select from.
    """
    identity = predeclared.get("workload_sha256")
    if identity not in certificate.members:
        raise GsimGateError("corroboration workload is not in the certified GSIM envelope")
    if (predeclared.get("primary_engine") != GSIM_ENGINE
            or predeclared.get("corroborating_engine") != REFERENCE_ENGINE
            or predeclared.get("require_same_elf") is not True
            or predeclared.get("require_output_bytes_match") is not True):
        raise GsimGateError("corroboration was not predeclared with the required strong checks")

    for label, run, engine, pin in (
            ("primary", primary, GSIM_ENGINE, "gsim_binary"),
            ("corroborating", corroborating, REFERENCE_ENGINE, "verilator_binary")):
        if run.get("engine") != engine or run.get("status") != "pass":
            raise GsimGateError(f"{label} corroboration is not a passing {engine} run")
        if run.get("derived_from_rtl") is not True or run.get("cycle_accurate") is not True:
            raise GsimGateError(f"{label} corroboration is not cycle-accurate elaborated RTL")
        if run.get("binary_sha256") != certificate.pins[pin]["sha256"]:
            raise GsimGateError(f"{label} corroboration does not name the pinned {pin}")
        firrtl_pin = "gsim_firrtl" if engine == GSIM_ENGINE else "verilator_firrtl"
        if run.get("firrtl_sha256") != certificate.pins[firrtl_pin]["sha256"]:
            raise GsimGateError(f"{label} corroboration does not name the pinned {firrtl_pin}")
        if engine == GSIM_ENGINE and run.get("model_sha256") != certificate.pins["gsim_model"]["sha256"]:
            raise GsimGateError("primary corroboration does not name the pinned GSIM model")
    elf = primary.get("elf_sha256")
    if not _is_sha256(elf) or corroborating.get("elf_sha256") != elf:
        raise GsimGateError("corroboration did not run the same exact ELF on both engines")
    output = primary.get("output_sha256")
    if not _is_sha256(output) or corroborating.get("output_sha256") != output:
        raise GsimGateError("corroboration output bytes do not match exactly")
    return {
        "workload_sha256": identity,
        "certificate_sha256": certificate.sha256,
        "elf_sha256": elf,
        "output_sha256": output,
        "agreement": "AGREE",
        "evidence": STRONG_EVIDENCE,
        "performance_selection": "none; corroboration is correctness evidence only",
    }


def predeclare_campaign(certificate: CertificateRecord,
                        workloads: Sequence[Mapping[str, Any]], *,
                        gsim_available: bool,
                        corroboration_count: int) -> dict[str, Any]:
    """Predeclare GSIM use and a deterministic Verilator corroboration subset.

    The subset is selected by hashes of the already-sealed certificate and workload identities, not by
    observed cycles.  It therefore remains reproducible without exposing a shape heuristic to the agent.
    """
    if corroboration_count < 0:
        raise GsimGateError("corroboration_count must be non-negative")
    canonical_by_id: dict[str, Mapping[str, Any]] = {}
    for workload in workloads:
        canonical = canonical_workload(workload)
        identity = workload_sha256(canonical)
        if identity in canonical_by_id:
            raise GsimGateError(f"duplicate campaign workload {identity}")
        canonical_by_id[identity] = canonical
    if not canonical_by_id:
        raise GsimGateError("campaign must contain at least one workload")

    development = [plan_evaluation(certificate, canonical_by_id[identity],
                                   phase="development_correctness",
                                   gsim_available=gsim_available).to_dict()
                   for identity in sorted(canonical_by_id)]
    final = [plan_evaluation(certificate, canonical_by_id[identity],
                             phase="final_performance",
                             gsim_available=gsim_available).to_dict()
             for identity in sorted(canonical_by_id)]
    eligible = [identity for identity in sorted(canonical_by_id)
                if identity in certificate.members]
    if eligible and corroboration_count == 0:
        raise GsimGateError("at least one Verilator corroboration point is required when GSIM is used")
    if corroboration_count > len(eligible):
        raise GsimGateError(
            f"corroboration_count {corroboration_count} exceeds {len(eligible)} GSIM-eligible workloads")
    ranked = sorted(eligible, key=lambda identity: _digest_bytes(
        f"{certificate.sha256}:{identity}".encode("ascii")))
    corroboration = [
        {
            "workload_sha256": identity,
            "primary_engine": GSIM_ENGINE,
            "corroborating_engine": REFERENCE_ENGINE,
            "require_same_elf": True,
            "require_output_bytes_match": True,
        }
        for identity in ranked[:corroboration_count]
    ]
    body = {
        "schema_version": "merlin.gsim-campaign-predeclaration.v1",
        "certificate_sha256": certificate.sha256,
        "development": development,
        "final": final,
        "verilator_corroboration": corroboration,
        "selection": "sha256(certificate_sha256:workload_sha256), ascending, before measurement",
    }
    return {**body, "predeclaration_sha256": _digest_bytes(canonical_json(body).encode("utf-8"))}
