"""Execute independent runtime controls before issuing functional authority.

The fixed support context prepares actual private source/runtime controls. This
owner invokes its ordinary grader and semantic verifier itself, reopens every
observed boundary, and retains positive and negative outcomes. Saved pass labels
or source hashes alone never qualify a runtime. Timing roles require separate
held measurement qualification and are not issued by these correctness controls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.common import invocation_record
from merlin_experiments.phase1.component_witness import (
    REQUIRED_EXECUTION_EFFECTS,
    ComponentStageWitness,
    verify_component_stage_witness,
)

from .component_runtime_authority import IndependentRuntimeServices, _callback_identity
from .contracts import StageGateError, document_sha256, mapping_file, sha256_file, write_json

_ISSUED = WeakKeyDictionary()
CONTROL_MECHANISMS = (
    "source_correspondence", "original_output_roster", "original_numeric_gate", "instruction_audit",
    "ownership_lifetime", "host_device_synchronization", "hardware_runtime_binding",
)
CONTROL_CASES = tuple(f"{mechanism}.{direction}" for mechanism in CONTROL_MECHANISMS
                      for direction in ("positive", "negative"))


@dataclass(frozen=True)
class RuntimeControlFixture:
    case_id: str
    grade_arguments: dict
    witness_arguments: dict
    member: dict
    frontend: str
    candidate_sha256: str
    target_descriptor_sha256: str
    capsule_root: Path
    evidence_root: Path
    required_effects: tuple[str, ...]
    required_invocation_stages: tuple[str, ...]


class RuntimeControlRefusal(StageGateError):
    """An evaluated normal gate rejects its actual predeclared defective control."""

    def __init__(self, *, case_id, mechanism, evidence_files):
        super().__init__("independent runtime control refused: " + str(mechanism))
        self.case_id = case_id
        self.mechanism = mechanism
        self.evidence_files = evidence_files


def _argument_binding(value):
    from merlin_experiments.phase1.component_source_applicability import ComponentSourceApplicability

    if type(value) is ComponentSourceApplicability:
        value.verify()
        return {"source_applicability": value.record()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if type(value) is Path or isinstance(value, Path):
        if not value.is_absolute() or value.resolve() != value or value.is_symlink():
            raise StageGateError("runtime control arguments require canonical source/evidence paths")
        return {"path": str(value)}
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _argument_binding(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_argument_binding(item) for item in value]
    if type(value) in (set, frozenset):
        rows = [_argument_binding(item) for item in value]
        return sorted(rows, key=document_sha256)
    raise StageGateError("runtime control arguments contain unbound executable/context objects")


def _fixture_binding(fixture):
    return {
        "case_id": fixture.case_id, "member": fixture.member, "frontend": fixture.frontend,
        "candidate_sha256": fixture.candidate_sha256,
        "target_descriptor_sha256": fixture.target_descriptor_sha256,
        "capsule_root": str(fixture.capsule_root), "evidence_root": str(fixture.evidence_root),
        "required_effects": fixture.required_effects,
        "required_invocation_stages": fixture.required_invocation_stages,
        "grade_arguments": _argument_binding(fixture.grade_arguments),
        "witness_arguments": _argument_binding(fixture.witness_arguments),
    }


def _evidence_file(path, digest, owner):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file()
        or not path.is_relative_to(owner) or sha256_file(path) != digest):
        raise StageGateError("runtime control evidence differs from its actual private product")
    return path


def _invocations(fixture):
    paths = tuple(sorted(fixture.evidence_root.rglob("invocation.json")))
    if not paths:
        raise StageGateError("independent runtime control did not retain actual invoked boundaries")
    records = []
    stages = set()
    for path in paths:
        _evidence_file(path, sha256_file(path), fixture.evidence_root)
        document = invocation_record.verify(path)
        stages.add(document["stage"])
        records.append((path, sha256_file(path)))
    if not set(fixture.required_invocation_stages) <= stages:
        raise StageGateError("independent runtime control omitted normal invoked stages")
    if not any(invocation_record.verify(path)["kind"] == "subprocess" for path, _ in records):
        raise StageGateError("independent runtime controls require actual processes, not callback pass labels")
    return tuple(records)


def _passed_grade(score, fixture):
    if not isinstance(score, dict) or score.get("integrity_status") != "clean":
        raise StageGateError("independent runtime positive control failed ordinary integrity")
    rows = score.get("per_capsule")
    if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
        or rows[0].get("capsule") != fixture.member["name"]
        or rows[0].get("status") != "pass" or rows[0].get("numeric") != "pass" or rows[0].get("cert_verdict")):
        raise StageGateError("independent runtime positive control failed complete ordinary numerical/executable gates")
    result = fixture.witness_arguments.get("result_path")
    if result is None:
        raise StageGateError("independent runtime positive control has no ordinary produced result")
    result = Path(result)
    _evidence_file(result, sha256_file(result), fixture.evidence_root)
    actual = mapping_file(result)
    if (actual.get("capsule") != fixture.member["name"] or actual.get("status") != "pass"
        or actual.get("numeric") != "pass" or actual.get("failure") or actual.get("cert_verdict")):
        raise StageGateError("independent runtime summary disagrees with its actual produced result")
    return result, sha256_file(result)


def _bounded_arguments(arguments, timeout_key, *, started, timeout_s):
    arguments = dict(arguments)
    declared = arguments.get(timeout_key)
    remaining = int(timeout_s - (time.monotonic() - started))
    if type(declared) is not int or not 0 < declared <= 600 or remaining <= 0:
        raise StageGateError("runtime control invocation lacks an explicit bounded remaining budget")
    arguments[timeout_key] = min(declared, remaining)
    return arguments


@dataclass(frozen=True, eq=False)
class IndependentRuntimeQualification:
    context: object
    services: IndependentRuntimeServices
    hardware_intake: object
    target_descriptor: Path
    source_pins: tuple[tuple[Path, str], ...]
    qualified_roles: tuple[str, ...]
    receipt: Path
    receipt_sha256: str
    context_sha256: str
    fixtures: tuple[RuntimeControlFixture, ...]
    fixture_sha256s: tuple[str, ...]
    evidence_files: tuple[tuple[Path, str], ...]
    callbacks: tuple[tuple[str, tuple], ...]

    def verify(self):
        issued = _ISSUED.get(self)
        if issued is None or issued != self.receipt_sha256:
            raise StageGateError("runtime qualification requires live executed controls, not a saved report")
        self.context.verify()
        self.hardware_intake.verify()
        if (self.context.services is not self.services or self.context.hardware_intake is not self.hardware_intake
            or self.context.target_descriptor != self.target_descriptor
            or tuple(self.context.source_pins) != self.source_pins or self.context.sha256 != self.context_sha256):
            raise StageGateError("independent runtime prepared source/context selection changed")
        for path, digest in (*self.source_pins, *self.evidence_files, (self.receipt, self.receipt_sha256)):
            if path.is_symlink() or path.resolve() != path or not path.is_file() or sha256_file(path) != digest:
                raise StageGateError("independent runtime qualification source or control evidence changed")
        for role, identity in self.callbacks:
            if _callback_identity(getattr(self.services, role)) != identity:
                raise StageGateError("independent runtime qualifier callback binding changed")
        for fixture, digest in zip(self.fixtures, self.fixture_sha256s, strict=True):
            self.context.verify_control(fixture)
            if document_sha256(_fixture_binding(fixture)) != digest:
                raise StageGateError("independent runtime prepared control binding changed")
            _invocations(fixture)
        document = mapping_file(self.receipt)
        if (document.get("status") != "qualified" or document.get("control_cases") != list(CONTROL_CASES)
            or document.get("qualified_roles") != ["grade", "stage_verifier"]
            or len(document.get("controls", [])) != len(CONTROL_CASES)
            or any(row.get("status") != "pass" for row in document["controls"])):
            raise StageGateError("independent runtime actual control roster is incomplete")
        return self.receipt_sha256


def qualify_independent_component_runtime(*, context, evidence_root: Path, timeout_s: int = 600):
    """Execute the fixed correctness roster through actual ordinary services.

    The independently prepared context owns public RTL/SDK/tool derivation and
    private control preparation. Each negative must be genuinely defective and
    rejected by the ordinary gate with its actual evidence. Exceptions, crashes
    and successful negative controls leave qualification refused. No decoder,
    cycle or performance-role authority is transferred from metadata.
    """
    try:
        from merlin_experiments.phase0.rtl_intake import IndependentHardwareIntake

        from .component_runtime_support import PreparedIndependentRuntimeContext
    except ImportError as error:
        raise StageGateError("independent runtime preparation/RTL authority is unavailable") from error
    if type(context) is not PreparedIndependentRuntimeContext:
        raise StageGateError("runtime controls require the fixed independently prepared context")
    context.verify()
    if type(context.hardware_intake) is not IndependentHardwareIntake:
        raise StageGateError("runtime controls require independently issued hardware intake")
    context.hardware_intake.verify()
    if type(context.services) is not IndependentRuntimeServices:
        raise StageGateError("runtime controls require the closed ordinary service contract")
    if type(timeout_s) is not int or not 0 < timeout_s <= 600:
        raise StageGateError("independent runtime controls require a bounded execution budget")
    root = Path(evidence_root)
    if root.exists() or root.is_symlink() or root.resolve() != root:
        raise StageGateError("independent runtime control output must be a fresh canonical private directory")
    root.mkdir(parents=True, mode=0o700)
    pins = tuple((Path(path), digest) for path, digest in context.source_pins)
    if not pins or len(pins) != len(dict(pins)):
        raise StageGateError("independent runtime controls lack complete source/tool identity")
    callbacks = tuple((role, _callback_identity(getattr(context.services, role)))
                      for role in ("grade", "stage_verifier"))
    for _role, identity in callbacks:
        if (identity[2], identity[3]) not in pins:
            raise StageGateError("runtime grader/witness source is outside independently prepared membership")
    context_sha = context.sha256
    started, fixtures, bindings, evidence, rows = time.monotonic(), [], [], {}, []
    for name in CONTROL_CASES:
        if time.monotonic() - started >= timeout_s:
            rows.append({"case_id": name, "status": "unavailable", "reason": "control budget exhausted"})
            continue
        mechanism, _, direction = name.partition(".")
        try:
            fixture = context.prepare_control(name, root / name)
            if (type(fixture) is not RuntimeControlFixture or fixture.case_id != name
                or not fixture.evidence_root.is_relative_to(root)
                or fixture.evidence_root.resolve() != fixture.evidence_root
                or not fixture.required_invocation_stages
                or not set(REQUIRED_EXECUTION_EFFECTS) <= set(fixture.required_effects)):
                raise StageGateError("runtime prepared control omits its normal source/output/effect binding")
            context.verify_control(fixture)
            binding = document_sha256(_fixture_binding(fixture))
            result = None
            try:
                score = context.services.grade(**_bounded_arguments(
                    fixture.grade_arguments, "timeout", started=started, timeout_s=timeout_s,
                ))
                if direction == "positive":
                    result = _passed_grade(score, fixture)
                witness = context.services.stage_verifier(**_bounded_arguments(
                    fixture.witness_arguments, "timeout_s", started=started, timeout_s=timeout_s,
                ))
                if direction == "negative":
                    raise StageGateError("independent runtime accepted a predeclared defective control")
                if type(witness) is not ComponentStageWitness:
                    raise StageGateError("runtime positive control lacks actual semantic stage witness")
                verify_component_stage_witness(
                    witness, member=fixture.member, candidate_sha256=fixture.candidate_sha256,
                    target_descriptor_sha256=fixture.target_descriptor_sha256, frontend=fixture.frontend,
                    capsule_root=fixture.capsule_root, evidence_root=fixture.evidence_root,
                    required_effects=fixture.required_effects,
                )
                for stage in witness.stages:
                    evidence[stage.path] = stage.sha256
            except RuntimeControlRefusal as refusal:
                if (direction != "negative" or refusal.case_id != name or refusal.mechanism != mechanism
                    or not isinstance(refusal.evidence_files, tuple) or not refusal.evidence_files):
                    raise StageGateError(
                        "runtime control refusal differs from the actual predeclared defect"
                    ) from refusal
                for path, digest in refusal.evidence_files:
                    evidence[_evidence_file(path, digest, fixture.evidence_root)] = digest
            context.verify_control(fixture)
            for path, digest in _invocations(fixture):
                evidence[path] = digest
            if result is not None:
                evidence[result[0]] = result[1]
            if document_sha256(_fixture_binding(fixture)) != binding:
                raise StageGateError("runtime control binding changed during actual evaluation")
            fixtures.append(fixture)
            bindings.append(binding)
            rows.append({"case_id": name, "status": "pass", "fixture_sha256": binding})
        except Exception as error:  # noqa: BLE001 - incomplete controls never issue authority
            rows.append({"case_id": name, "status": "refused", "reason": type(error).__name__,
                         "detail": str(error)[:512]})
    context.verify()
    if context.sha256 != context_sha:
        raise StageGateError("independent runtime source/context changed during control execution")
    complete = len(fixtures) == len(CONTROL_CASES) and all(row["status"] == "pass" for row in rows)
    status = "qualified" if complete else "refused"
    receipt = root / "qualification.json"
    write_json(receipt, {
        "schema": "merlin.independent_runtime_qualification.v1", "status": status,
        "control_cases": list(CONTROL_CASES), "controls": rows,
        "qualified_roles": ["grade", "stage_verifier"] if status == "qualified" else [],
        "context_sha256": context_sha, "elapsed_seconds": time.monotonic() - started,
        "source_pins": [(str(path), digest) for path, digest in pins],
        "scope": "functional correctness controls only; timing/CCA/RTL feedback remains unqualified",
    })
    receipt.chmod(0o400)
    qualification = IndependentRuntimeQualification(
        context, context.services, context.hardware_intake, context.target_descriptor, pins,
        ("grade", "stage_verifier") if status == "qualified" else (), receipt, sha256_file(receipt),
        context_sha, tuple(fixtures), tuple(bindings),
        tuple(sorted(evidence.items(), key=lambda row: str(row[0]))), callbacks,
    )
    if status == "qualified":
        _ISSUED[qualification] = qualification.receipt_sha256
        qualification.verify()
    return qualification
