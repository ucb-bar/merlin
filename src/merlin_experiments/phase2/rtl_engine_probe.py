"""Execute independent RTL support controls before expensive candidate runs.

This interface has no engine/target discovery, text rewriting or serialized PASS
loader. Its live diagnostic admission checks only the original selected controls;
physical correctness, ELF loading, target equivalence and timers are separate.
"""
from __future__ import annotations

import json
import subprocess
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record

from .contracts import StageGateError, document_sha256, sha256_file
from .rtl_engine_protocol import (
    RTL_CONTRACTS,
    RtlEngineSelection,
    RtlProbeControl,
    canonical_file,
    check_original_outputs,
    relative_member,
)

_ISSUED = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class RtlProbeOutcome:
    control_id: str
    contract: str
    status: str
    failed_step: str | None
    reason: str | None
    invocations: tuple[Path, ...]
    original_rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, eq=False)
class RtlEngineProbeAdmission:
    selection: RtlEngineSelection
    control_bindings: tuple[tuple[str, str], ...]
    outcomes: tuple[RtlProbeOutcome, ...]
    evidence_root: Path
    pins: tuple[tuple[Path, str], ...]
    receipt: Path
    receipt_sha256: str

    def verify(self):
        if _ISSUED.get(self) != self.sha256:
            raise StageGateError("RTL probe admission must be issued by actual independent execution")
        self.selection.verify()
        if sha256_file(canonical_file(self.receipt)) != self.receipt_sha256:
            raise StageGateError("RTL probe private receipt changed")
        for path, digest in self.pins:
            if sha256_file(canonical_file(path)) != digest:
                raise StageGateError("RTL probe original source, invocation or observation changed")
        members = tuple(self.evidence_root.rglob("*"))
        if (any(path.is_symlink() or path.resolve() != path for path in members)
            or {path for path in members if path.is_file()} != {self.receipt, *(path for path, _ in self.pins)}):
            raise StageGateError("RTL probe private evidence membership changed")
        for outcome in self.outcomes:
            for path in outcome.invocations:
                document = json.loads(path.read_text())
                if document.get("status") == "completed":
                    invocation_record.verify(path)
                elif outcome.status == "PASS":
                    raise StageGateError("RTL probe accepted an unsuccessful invocation")
        return self.sha256

    @property
    def sha256(self):
        return document_sha256({
            "schema": "merlin.independent_rtl_engine_probe.v1",
            "selection": self.selection.sha256, "controls": self.control_bindings,
            "outcomes": [_outcome_document(value) for value in self.outcomes],
            "evidence_root": str(self.evidence_root),
            "pins": [(str(path), digest) for path, digest in self.pins],
            "receipt": (str(self.receipt), self.receipt_sha256),
        })

    def readiness(self):
        self.verify()
        contracts = {}
        for contract in RTL_CONTRACTS:
            rows = [outcome for outcome in self.outcomes if outcome.contract == contract]
            contracts[contract] = {
                "status": "UNKNOWN" if not rows else "PASS" if all(row.status == "PASS" for row in rows) else "REFUSED",
                "controls": [{"id": row.control_id, "status": row.status,
                              "failed_step": row.failed_step, "reason": row.reason} for row in rows],
            }
        return {
            "schema": "merlin.rtl_engine_control_readiness.v1", "probe_sha256": self.sha256,
            "contracts": contracts, "scope": "ORIGINAL_PRIVATE_CONTROLS_ONLY",
            "target_runtime": "UNKNOWN", "hardware_equivalence": "UNKNOWN", "target_timer": "UNKNOWN",
        }

    def require_controls(self, control_ids):
        """A pre-run readiness gate, with no target/runtime or performance grant."""
        self.verify()
        if not isinstance(control_ids, tuple) or not control_ids or len(set(control_ids)) != len(control_ids):
            raise StageGateError("RTL readiness needs exact required control membership")
        rows = {row.control_id: row for row in self.outcomes}
        for name in control_ids:
            row = rows.get(name)
            if row is None or row.status != "PASS":
                contract = row.contract if row is not None else "undeclared"
                raise StageGateError(f"RTL engine readiness refuses {contract}/{name}")
        return self.sha256


def _outcome_document(value):
    return {"control_id": value.control_id, "contract": value.contract, "status": value.status,
            "failed_step": value.failed_step, "reason": value.reason,
            "invocations": [str(path) for path in value.invocations], "original_rows": value.original_rows}


def _members(workspace, names):
    result = []
    for name in names:
        path = workspace / relative_member(name)
        if path.is_symlink() or path.resolve() != path:
            raise StageGateError("RTL probe artifact has path indirection")
        if path.is_dir():
            paths = sorted(path.rglob("*"))
            if any(item.is_symlink() or item.resolve() != item for item in paths):
                raise StageGateError("RTL probe artifact directory has path indirection")
            result.extend(item for item in paths if item.is_file())
        elif path.is_file():
            result.append(path)
        else:
            raise StageGateError("RTL probe declared artifact is unavailable")
    return tuple(sorted(set(result)))


def _native_machine(path):
    data = path.read_bytes()[:24]
    if len(data) < 20 or data[:4] != b"\x7fELF" or data[5] not in (1, 2):
        raise StageGateError("RTL probe generated executable needs inspectable native ELF identity")
    return int.from_bytes(data[18:20], "little" if data[5] == 1 else "big")


def _run_control(selection, control, owner, environment):
    work = owner / control.id
    work.mkdir(mode=0o700)
    for name, payload in control.files:
        path = work / relative_member(name)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(payload)
    sources = tuple(work / name for name, _ in control.files)
    dependency_paths = tuple(path for path, _ in selection.pins)
    records, produced = [], set()
    rows, failed_step, reason = (), None, None
    for step in control.steps:
        selection.verify()
        command = tuple(token.replace("@WORK@", str(work)) for token in step.argv)
        executable = Path(command[0])
        if executable not in selection.tools:
            if executable not in produced or executable not in _members(work, step.inputs):
                raise StageGateError("RTL probe executable was not selected or actually produced")
            if _native_machine(executable) != _native_machine(Path(sys.executable).resolve()):
                raise StageGateError("RTL probe never directly executes a target ELF")
        inputs = _members(work, step.inputs)
        if any(path.exists() for path in (work / name for name in step.outputs)):
            raise StageGateError("RTL probe cannot overwrite an existing source or product")
        for name in step.outputs:
            (work / name).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in step.output_directories:
            (work / name).mkdir(mode=0o700)
        with invocation_record.observe(
            work, stage=step.id, argv=command, cwd=work, env=environment,
            inputs=inputs, dependencies=(*dependency_paths, *sources, Path(__file__),
                                       Path(__file__).with_name("rtl_engine_protocol.py")),
        ) as invocation:
            records.append(invocation.path)
            try:
                result = subprocess.run(command, cwd=work, env=environment, capture_output=True,
                                        timeout=step.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                invocation.failed(error)
                for stream in ("stdout", "stderr"):
                    payload = getattr(error, stream, None) or b""
                    (invocation.directory / ("partial_" + stream + ".bin")).write_bytes(payload)
                failed_step, reason = step.id, "bounded_process_timeout"
                break
            if result.returncode == 0:
                try:
                    invocation.outputs = _members(work, step.outputs)
                    if step.outputs and not invocation.outputs:
                        raise StageGateError("RTL probe generated an empty artifact roster")
                except StageGateError:
                    failed_step, reason = step.id, "declared_product_missing"
            invocation.complete(result)
        if failed_step is not None:
            break
        if result.returncode != 0:
            failed_step, reason = step.id, "declared_process_rejected_control"
            break
        invocation_record.verify(invocation.path)
        produced.update(invocation.outputs)
        if step is control.steps[-1] and control.expected_rows:
            try:
                rows = check_original_outputs(result.stdout, control)
            except StageGateError:
                failed_step, reason = step.id, "complete_original_outputs_mismatch"
    if any((work / name).read_bytes() != payload for name, payload in control.files):
        raise StageGateError("RTL probe original control source changed during execution")
    return RtlProbeOutcome(control.id, control.contract, "PASS" if failed_step is None else "REFUSED",
                           failed_step, reason, tuple(records), rows)


def probe_rtl_engine(*, selection, controls, evidence_root, environment):
    """Run evaluator-owned declarations; importing never executes a tool.

    No source normalization, target loading or candidate execution is selected
    here. A supplied command may implement an independently pinned tool route;
    successful controls certify only their exact original source and output scope.
    """
    if type(selection) is not RtlEngineSelection or not isinstance(controls, tuple) or not controls:
        raise StageGateError("RTL engine probing needs typed support and exact controls")
    selection.verify()
    if any(type(control) is not RtlProbeControl for control in controls):
        raise StageGateError("RTL engine control must be an original typed declaration")
    bindings = tuple((control.id, control.verify()) for control in controls)
    if len({name for name, _ in bindings}) != len(bindings):
        raise StageGateError("RTL engine control membership is duplicated")
    owner = Path(evidence_root)
    if not owner.is_absolute() or owner.is_symlink() or owner.resolve() != owner or owner.exists():
        raise StageGateError("RTL probe evidence requires a fresh canonical private owner")
    if any(path.is_relative_to(owner) for path, _ in selection.pins):
        raise StageGateError("RTL probe evidence overlaps immutable tool or support selections")
    if type(environment) is not dict or any(type(key) is not str or type(value) is not str
                                           for key, value in environment.items()):
        raise StageGateError("RTL probe needs an explicit string process environment")
    owner.mkdir(parents=True, mode=0o700)
    outcomes = tuple(_run_control(selection, control, owner, environment) for control in controls)
    selection.verify()
    receipt = owner / "rtl_engine_probe.json"
    receipt.write_text(json.dumps({
        "schema": "merlin.independent_rtl_engine_probe.v1", "selection_sha256": selection.sha256,
        "control_bindings": bindings, "outcomes": [_outcome_document(row) for row in outcomes],
        "environment_sha256": document_sha256(environment),
        "scope": "ORIGINAL_PRIVATE_CONTROLS_ONLY", "target_runtime": "UNKNOWN",
        "hardware_equivalence": "UNKNOWN", "target_timer": "UNKNOWN",
    }, sort_keys=True, indent=2) + "\n")
    files = tuple(path for path in sorted(owner.rglob("*")) if path.is_file() and path != receipt)
    if any(path.is_symlink() or path.resolve() != path for path in owner.rglob("*")):
        raise StageGateError("RTL probe evidence has path indirection")
    admission = RtlEngineProbeAdmission(selection, bindings, outcomes, owner,
                                       tuple((path, sha256_file(path)) for path in files),
                                       receipt, sha256_file(receipt))
    _ISSUED[admission] = admission.sha256
    admission.verify()
    return admission
