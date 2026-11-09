"""Issue observation-only verifier capabilities from frozen control execution.

Receipts document execution; they cannot be reopened into admission authority.
The selected target owner supplies concrete source/hardware controls. Synthetic
controls exercise these joins only and retain that explicit scope in every receipt.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.common.digest import sha256_bytes
from merlin.targetgen.sandbox import bwrap as BW

from .contracts import StageGateError, canonical_json, sha256_file, write_json
from .numerical_readback import ReadbackFiles
from .protected_final_evaluation import (
    REQUIRED_WITNESSES,
    FinalExecutionBinding,
    ProtectedExecutionVerifier,
    _mapping,
    _pins,
    _plain_file,
    _protected_files,
    observe_arm,
)

CONTROL_SCHEMA = "merlin.protected_final_controls.v1"
QUALIFICATION_SCHEMA = "merlin.protected_final_verifier.v3"
_ISSUER = object()


def _issuer_sources() -> tuple[tuple[Path, str], ...]:
    """Bind current evaluator membership as well as the selected target closure."""
    _plain_file(Path(__file__))
    paths = set(Path(__file__).parent.rglob("*.py")) | {Path(BW.__file__)}
    for path in paths:
        _plain_file(path)
    return tuple((path, sha256_file(path)) for path in sorted(paths))


class ProtectedArmRefusal(StageGateError):
    """An actual selected verifier's diagnosed negative-control refusal.

    A crash, unreadable dependency or generic exception never passes a control.
    The private verifier must name the defective authority and its observed bytes.
    """

    def __init__(self, *, witness_kind: str, binding_sha256: str, arm: str,
                 evidence: tuple[tuple[Path, str], ...], reason: str):
        super().__init__(reason)
        self.witness_kind = witness_kind
        self.binding_sha256 = binding_sha256
        self.arm = arm
        self.evidence = evidence


@dataclass(frozen=True)
class ProtectedVerifierControl:
    """A host-selected case whose bytes were frozen before qualification."""

    name: str
    witness_kind: str
    expected: str
    binding: FinalExecutionBinding
    original_binding: Path
    arm: str
    files: ReadbackFiles
    original_reference: ReadbackFiles
    artifacts: tuple[tuple[Path, str], ...]

    def record(self) -> dict:
        if (not isinstance(self.name, str) or not self.name
            or self.witness_kind not in REQUIRED_WITNESSES or self.expected not in {"observe", "refuse"}
            or type(self.binding) is not FinalExecutionBinding or self.arm not in {"reference", "candidate"}
            or type(self.files) is not ReadbackFiles or type(self.original_reference) is not ReadbackFiles):
            raise StageGateError("protected verifier control has an invalid typed domain")
        _plain_file(self.original_binding)
        if _mapping(self.original_binding) != vars(self.binding):
            raise StageGateError("protected verifier control binding differs from its original bytes")
        _pins(self.artifacts)
        paths = tuple(dict.fromkeys((self.original_binding, *self.files.paths(), *self.original_reference.paths(),
                                    *(path for path, _sha in self.artifacts))))
        for path in paths:
            _plain_file(path)
        return {
            "name": self.name, "witness_kind": self.witness_kind, "expected": self.expected,
            "binding_sha256": self.binding.identity(), "original_binding": str(self.original_binding),
            "arm": self.arm, "files": [str(path) for path in self.files.paths()],
            "original_reference": [str(path) for path in self.original_reference.paths()],
            "inputs": [[str(path), sha256_file(path)] for path in paths],
            "artifacts": [[str(path), digest] for path, digest in self.artifacts],
        }


@dataclass(frozen=True)
class ProtectedVerifierQualification:
    """Executed callback observation controls; no physical execution authority."""

    receipt: Path
    receipt_sha256: str
    code_sha256: str
    dependencies: tuple[tuple[Path, str], ...]
    issuer_sources: tuple[tuple[Path, str], ...]
    inputs: tuple[tuple[Path, str], ...]
    original_plan: Path
    plan_sha256: str
    run_dir: Path
    environment_json: str
    scope: str
    execution_domain: tuple[str, str, str, str]
    _issuer: object = None

    def verify(self, selection: ProtectedExecutionVerifier) -> None:
        if self._issuer is not _ISSUER:
            raise StageGateError("protected verifier qualification was not independently issued")
        selection.validate_selection()
        if (selection.code_sha256 != self.code_sha256 or selection.dependencies != self.dependencies
            or selection.qualification_sha256 != self.receipt_sha256):
            raise StageGateError("protected verifier qualification selects different code or dependencies")
        _pins(((self.receipt, self.receipt_sha256),))
        if self.receipt.stat().st_mode & 0o222:
            raise StageGateError("protected verifier issued receipt is no longer immutable")
        if _issuer_sources() != self.issuer_sources:
            raise StageGateError("protected verifier evaluator source membership changed")
        _pins(self.inputs)
        if sha256_file(self.original_plan) != self.plan_sha256:
            raise StageGateError("protected verifier original control plan changed")
        provenance = json.loads(self.environment_json)
        _protected_files(self.run_dir, provenance, tuple(path for path, _sha in self.inputs))
        record = _mapping(self.receipt)
        if (record.get("schema") != QUALIFICATION_SCHEMA or record.get("status") != "observation_qualified"
            or record.get("qualified_roles") != ["private_witness_observation"]
            or record.get("code_sha256") != self.code_sha256
            or record.get("control_plan_sha256") != self.plan_sha256
            or record.get("scope") != self.scope):
            raise StageGateError("protected verifier issued receipt differs from its original evaluation")

    def verify_binding(self, binding: FinalExecutionBinding) -> None:
        if (binding.runtime_sha256, binding.toolchain_sha256, binding.hardware_sha256,
            binding.timer_scope_sha256) != self.execution_domain:
            raise StageGateError(
                "protected final binding differs from the qualified runtime/toolchain/hardware/timer domain"
            )


def qualify_protected_execution_verifier(
    *, selection: ProtectedExecutionVerifier, original_plan: Path,
    controls: tuple[ProtectedVerifierControl, ...], run_dir: Path,
    environment: dict, receipt: Path,
) -> ProtectedExecutionVerifier:
    """Evaluate all positive and negative authorities using the normal verifier.

    Selection validation grants no admission, so this path has no circular need
    for a completed capability. Every control's complete input/closure is checked
    against the original private V4 snapshot before and after each actual call.
    """
    if type(selection) is not ProtectedExecutionVerifier:
        raise StageGateError("protected verifier qualification requires an explicit selected owner")
    selection.validate_selection()
    issuer_sources = _issuer_sources()
    if selection.qualification is not None:
        raise StageGateError("protected verifier qualification requires a fresh unadmitted selection")
    if type(controls) is not tuple or any(type(row) is not ProtectedVerifierControl for row in controls):
        raise StageGateError("protected verifier qualification requires a typed frozen control roster")
    records = [row.record() for row in controls]
    expected = {(kind, expectation) for kind in REQUIRED_WITNESSES for expectation in ("observe", "refuse")}
    roster = {(row.witness_kind, row.expected) for row in controls}
    if (len(controls) != len(expected) or roster != expected
        or len({row.name for row in controls}) != len(controls)):
        raise StageGateError(
            "protected verifier control roster requires independent positive and negative cases for all witnesses"
        )
    domains = {(row.binding.runtime_sha256, row.binding.toolchain_sha256, row.binding.hardware_sha256,
                row.binding.timer_scope_sha256) for row in controls}
    if len(domains) != 1:
        raise StageGateError("protected verifier controls do not bind one exact selected execution domain")
    domain = next(iter(domains))
    plan = _mapping(original_plan)
    if (set(plan) != {"schema", "scope", "dependencies", "controls"}
        or plan["schema"] != CONTROL_SCHEMA or not isinstance(plan["scope"], str) or not plan["scope"].strip()
        or plan["controls"] != records
        or plan["dependencies"] != [[str(path), digest] for path, digest in selection.dependencies]):
        raise StageGateError("protected verifier controls differ from the original independent plan")
    sources = tuple(dict.fromkeys((original_plan, *(path for path, _sha in selection.dependencies),
                                  *(Path(path) for record in records for path, _sha in record["inputs"]))))
    provenance = copy.deepcopy(environment)
    frozen = _protected_files(run_dir, provenance, sources)
    inputs = tuple((path, sha256_file(path)) for path in sources)
    # The issuer's new receipt is private output, never a new control/input seal.
    if (not isinstance(receipt, Path) or not receipt.is_absolute() or ".." in receipt.parts
        or receipt.exists() or receipt.resolve() != receipt
        or any(part.is_symlink() for part in receipt.parents)
        or any(receipt == path or path in receipt.parents for path in sources)
        or receipt.is_relative_to(Path(provenance["workspace_path"]))):
        raise StageGateError("protected verifier issuance requires fresh private receipt output")

    def recheck():
        selection.validate_selection()
        if _issuer_sources() != issuer_sources:
            raise StageGateError("protected verifier evaluator sources changed during qualification")
        _pins(inputs)
        if _protected_files(run_dir, provenance, sources) != frozen:
            raise StageGateError("protected verifier frozen controls changed during qualification")

    observations = []
    for control in controls:
        recheck()
        try:
            result = observe_arm(selection, control.binding, control.arm, control.files, control.original_reference)
        except ProtectedArmRefusal as refusal:
            recheck()
            if (control.expected != "refuse" or refusal.witness_kind != control.witness_kind
                or refusal.binding_sha256 != control.binding.identity() or refusal.arm != control.arm
                or not str(refusal).strip()):
                raise StageGateError("protected verifier negative control diagnosed a different authority") from refusal
            _pins(refusal.evidence)
            if not set(refusal.evidence).issubset(set(control.artifacts)):
                raise StageGateError("protected verifier refusal lacks its pre-frozen control evidence") from refusal
            observations.append({"name": control.name, "status": "refused", "witness_kind": refusal.witness_kind,
                                 "reason": str(refusal), "evidence": [[str(p), s] for p, s in refusal.evidence]})
        except Exception:
            recheck()
            raise
        else:
            recheck()
            if control.expected != "observe":
                raise StageGateError("protected verifier accepted an independently declared negative control")
            if not set(result.evidence).issubset(set(control.artifacts)):
                raise StageGateError("protected verifier observation lacks its pre-frozen control evidence")
            observations.append({"name": control.name, "status": "observed", "witness_kind": control.witness_kind,
                                 "observation_sha256": sha256_bytes(canonical_json(vars(result) | {
                                     "evidence": [[str(p), s] for p, s in result.evidence]}))})
    recheck()
    record = {"schema": QUALIFICATION_SCHEMA, "status": "observation_qualified", "scope": plan["scope"],
              "qualified_roles": ["private_witness_observation"],
              "code_sha256": selection.code_sha256, "control_plan_sha256": sha256_file(original_plan),
              "dependencies": plan["dependencies"], "controls": observations,
              "issuer_sources": [[str(p), s] for p, s in issuer_sources],
              "execution_domain": list(domain),
              "inputs": [[str(p), s] for p, s in inputs],
              "authority": "private callback observations only; no physical hardware/runtime/timer authority"}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt, record)
    receipt.chmod(0o400)
    capability = ProtectedVerifierQualification(
        receipt, sha256_file(receipt), selection.code_sha256, selection.dependencies, issuer_sources, inputs,
        original_plan, sha256_file(original_plan), run_dir, json.dumps(provenance, sort_keys=True),
        plan["scope"], domain, _ISSUER,
    )
    admitted = replace(selection, qualification=capability, qualification_sha256=capability.receipt_sha256)
    admitted.validate()
    return admitted
