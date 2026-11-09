"""Join private verifier observations to complete original numerical readbacks.

The selected host verifier owns actual build, source, toolchain, hardware,
timer and instruction authentication. Observation controls do not establish
those physical roles; production admission requires a separate live issuer.
This lifecycle does not infer physical authority from
a candidate report, a process exit or a copied log. Missing witnesses refuse
comparison construction. Selecting/pinning a callback alone does not qualify
that verifier; its admission must come from the evaluator's protected lifecycle.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import CodeType

from merlin.common.digest import is_sha256, sha256_bytes
from merlin.perf.phase2_portfolio import QualityBudget
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.sandbox import bwrap as BW

from .campaign import verify_private_input_snapshot
from .component_final_policy import strict_final_component_campaign_gate
from .contracts import StageGateError, canonical_json, sha256_file
from .numerical_readback import ReadbackFiles, admit_protected_numerical_readback
from .physical_final_admission import admit_protected_final_comparison
from .protected_final_observation import ProtectedFinalObservation

REQUIRED_WITNESSES = (
    "compiler_invocation",
    "source_correspondence",
    "original_reference_generation",
    "dependency_closure",
    "hardware_execution",
    "timer_scope",
    "final_executable_instruction_audit",
)


def callable_code_sha256(function: Callable) -> str:
    """Hash immutable code fields; marshal string-intern flags are stateful."""
    def value(item):
        if isinstance(item, CodeType):
            return {name: value(getattr(item, name)) for name in (
                "co_code", "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars",
                "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize",
                "co_flags", "co_filename", "co_name", "co_qualname", "co_firstlineno",
                "co_linetable", "co_exceptiontable",
            )}
        if isinstance(item, bytes):
            return {"bytes": item.hex()}
        if isinstance(item, tuple):
            return {"tuple": [value(part) for part in item]}
        if isinstance(item, frozenset):
            return {"frozenset": sorted((value(part) for part in item), key=lambda part: canonical_json(part))}
        if type(item) is float:
            return {"float": item.hex()}
        if type(item) is complex:
            return {"complex": [item.real.hex(), item.imag.hex()]}
        if item is Ellipsis:
            return {"ellipsis": True}
        if item is None or type(item) in {str, int, bool}:
            return item
        raise StageGateError("protected verifier has an unsupported code constant")

    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        raise StageGateError("protected verifier requires an inspectable callable code identity")
    return sha256_bytes(canonical_json(value(code)))


@dataclass(frozen=True)
class FinalExecutionBinding:
    """Original private freeze identities, never selected by candidate metadata."""

    member: str
    reference_compiler_sha256: str
    candidate_compiler_sha256: str
    program_sha256: str
    inputs_sha256: str
    quality_budget_sha256: str
    original_reference_console_sha256: str
    runtime_sha256: str
    toolchain_sha256: str
    hardware_sha256: str
    timer_scope_sha256: str

    def identity(self) -> str:
        if not isinstance(self.member, str) or not self.member:
            raise StageGateError("protected final binding requires a member identity")
        values = vars(self)
        if not all(is_sha256(value) for name, value in values.items() if name != "member"):
            raise StageGateError("protected final binding requires every original freeze identity")
        return sha256_bytes(canonical_json(values))


@dataclass(frozen=True)
class FinalExecutionWitness:
    """Executed callback arm data; independent physical correspondence is separate."""

    binding_sha256: str
    arm: str
    compiler_sha256: str
    executable_sha256: str
    staged_executable_sha256: str
    console_sha256: str
    hardware_sha256: str
    inputs_sha256: str
    timer_scope_sha256: str
    cycles: int
    original_reference_console_sha256: str
    witnesses: tuple[tuple[str, str], ...]
    evidence: tuple[tuple[Path, str], ...]


@dataclass(frozen=True)
class ProtectedExecutionVerifier:
    """Explicit evaluator-selected callable and frozen implementation selection.

    The evaluator owns review and qualification of this capability. These pins
    reject source/code drift and selection substitution, not forged host trust.
    Dependencies must contain the implementation and actual qualified closure.
    """

    verify: Callable
    implementation: Path
    dependencies: tuple[tuple[Path, str], ...]
    qualification: object | None
    qualification_sha256: str | None
    code_sha256: str

    def validate_selection(self) -> None:
        """Pin a selection without granting it execution admission."""
        source = inspect.getsourcefile(self.verify)
        code = getattr(self.verify, "__code__", None)
        _plain_file(self.implementation)
        if source is None or code is None or Path(source) != self.implementation:
            raise StageGateError("protected final verifier does not execute the selected implementation")
        if not is_sha256(self.code_sha256) or callable_code_sha256(self.verify) != self.code_sha256:
            raise StageGateError("protected final verifier code identity changed")
        paths = {path for path, _digest in self.dependencies}
        if not self.dependencies or len(paths) != len(self.dependencies) or self.implementation not in paths:
            raise StageGateError("protected final verifier lacks its exact selected dependency closure")
        _pins(self.dependencies)

    def validate(self) -> None:
        from .protected_verifier_qualification import ProtectedVerifierQualification

        self.validate_selection()
        if type(self.qualification) is not ProtectedVerifierQualification:
            raise StageGateError("protected final verifier requires an independently issued qualification capability")
        self.qualification.verify(self)


def _plain_file(path: Path) -> None:
    """Require the selected path itself, including every ancestor, to be direct."""
    if (not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts
        or path.resolve() != path or not path.is_file()
        or any(part.is_symlink() for part in (path, *path.parents))):
        raise StageGateError("protected final file is absent or indirect")


def _mapping(path: Path) -> dict:
    _plain_file(path)
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise StageGateError("duplicate protected final field")
            result[key] = value
        return result

    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=unique)
    except (OSError, ValueError, UnicodeError) as exc:
        raise StageGateError("protected final record is unreadable") from exc
    if type(result) is not dict:
        raise StageGateError("protected final record must be a mapping")
    return result


def _protected_files(run_dir: Path, environment: dict, files: tuple[Path, ...]) -> tuple[tuple[Path, str], ...]:
    private = verify_private_input_snapshot(run_dir, environment)
    hosts = tuple(Path(row["destination"]) for row in private["marker"]["host_records"])
    for path in files:
        _plain_file(path)
        if not any(path == host or host in path.parents for host in hosts):
            raise StageGateError("protected final selection is not a private host input")
    frozen = BW.snapshot_input_paths(private["workspace"], private["bundle"], list(files), repo=private["repo"])
    result = []
    for source, path in zip(files, frozen, strict=True):
        _plain_file(path)
        digest = sha256_file(path)
        if path.stat().st_mode & 0o222 or sha256_file(source) != digest:
            raise StageGateError("protected final selected source differs from its immutable original snapshot")
        result.append((path, digest))
    return tuple(result)


def _pins(evidence: tuple[tuple[Path, str], ...]) -> None:
    if type(evidence) is not tuple or not evidence or len({path for path, _sha in evidence}) != len(evidence):
        raise StageGateError("protected final evidence must have unique immutable file identities")
    for path, digest in evidence:
        _plain_file(path)
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or ".." in path.parts
            or not is_sha256(digest)
            or sha256_file(path) != digest
        ):
            raise StageGateError("protected final evidence is absent, indirect or changed")


def observe_arm(
    verifier: ProtectedExecutionVerifier,
    binding: FinalExecutionBinding,
    arm: str,
    files: ReadbackFiles,
    original_reference: ReadbackFiles,
) -> FinalExecutionWitness:
    """Observe and structurally validate an arm without authorizing its verifier.

    The qualification owner uses this same normal verifier path on independently
    frozen controls. Admission additionally requires the issued capability.
    """
    verifier.validate_selection()
    if type(binding) is not FinalExecutionBinding or arm not in {"reference", "candidate"}:
        raise StageGateError("protected final observation requires a typed binding and measured arm")
    observed_paths = tuple(dict.fromkeys((*files.paths(), *original_reference.paths())))
    for path in observed_paths:
        _plain_file(path)
    _pins(tuple((path, sha256_file(path)) for path in observed_paths))
    before = (sha256_file(files.executable), sha256_file(files.console))
    result = verifier.verify(binding=binding, arm=arm, files=files, original_reference=original_reference)
    verifier.validate_selection()
    if type(result) is not FinalExecutionWitness:
        raise StageGateError("protected final verifier did not return an authenticated typed arm")
    compiler = binding.reference_compiler_sha256 if arm == "reference" else binding.candidate_compiler_sha256
    expected = (
        binding.identity(),
        arm,
        compiler,
        *before,
        binding.hardware_sha256,
        binding.inputs_sha256,
        binding.timer_scope_sha256,
    )
    actual = (
        result.binding_sha256,
        result.arm,
        result.compiler_sha256,
        result.executable_sha256,
        result.console_sha256,
        result.hardware_sha256,
        result.inputs_sha256,
        result.timer_scope_sha256,
    )
    if (
        expected != actual
        or result.staged_executable_sha256 != result.executable_sha256
        or result.original_reference_console_sha256 != binding.original_reference_console_sha256
        or sha256_file(original_reference.console) != binding.original_reference_console_sha256
    ):
        raise StageGateError("protected final arm differs from original build/input/hardware/timer binding")
    if type(result.cycles) is not int or result.cycles <= 0:
        raise StageGateError("protected final arm lacks positive reported cycle data")
    if (
        type(result.witnesses) is not tuple
        or len(result.witnesses) != len(REQUIRED_WITNESSES)
        or {name for name, _digest in result.witnesses} != set(REQUIRED_WITNESSES)
        or any(not is_sha256(digest) for _name, digest in result.witnesses)
    ):
        raise StageGateError("protected final arm lacks a complete authentication witness roster")
    _pins(result.evidence)
    # Witnesses name the actual evidence records reopened above, not arbitrary
    # strings unconnected to the verifier's files.
    if not {digest for _name, digest in result.witnesses}.issubset({digest for _path, digest in result.evidence}):
        raise StageGateError("protected final witness has no actual bound evidence record")
    if before != (sha256_file(files.executable), sha256_file(files.console)):
        raise StageGateError("protected final execution bytes changed during verification")
    return result


def _arm(verifier, binding, arm, files, original_reference) -> FinalExecutionWitness:
    verifier.validate()
    verifier.qualification.verify_binding(binding)
    result = observe_arm(verifier, binding, arm, files, original_reference)
    verifier.validate()
    return result


def observe_protected_final_comparison(
    *,
    binding: FinalExecutionBinding,
    original_binding: Path,
    verifier: ProtectedExecutionVerifier,
    run_dir: Path,
    environment: dict,
    original_budget: Path,
    budget: QualityBudget,
    reference: ReadbackFiles,
    candidate: ReadbackFiles,
    reference_build: BuildOnlyService,
    candidate_build: BuildOnlyService,
    original_reference: ReadbackFiles,
    original_reference_build: BuildOnlyService,
    physical_execution_domain=None,
) -> ProtectedFinalObservation:
    """Recompute complete numerical and witness joins without physical admission."""
    if type(binding) is not FinalExecutionBinding or type(verifier) is not ProtectedExecutionVerifier:
        raise StageGateError("protected final admission requires original evaluator capabilities")
    if physical_execution_domain is not None:
        raise StageGateError("observation-only final lifecycle cannot consume physical authority declarations")
    binding.identity()
    verifier.validate()
    provenance = copy.deepcopy(environment)
    selection = tuple(
        dict.fromkeys(
            (
                original_binding,
                original_budget,
                *(path for path, _sha in verifier.dependencies),
                *(path for path, _sha in verifier.qualification.inputs),
                *reference.paths(),
                *candidate.paths(),
                *original_reference.paths(),
            )
        )
    )
    frozen_selection = _protected_files(run_dir, provenance, selection)
    if _mapping(frozen_selection[0][0]) != vars(binding):
        raise StageGateError("protected final binding differs from the original evaluator freeze")
    if sha256_file(original_budget) != binding.quality_budget_sha256:
        raise StageGateError("protected final original numerical budget changed")
    if sha256_file(original_reference.console) != binding.original_reference_console_sha256:
        raise StageGateError("protected final original numerical reference changed")
    a = _arm(verifier, binding, "reference", reference, original_reference)
    b = _arm(verifier, binding, "candidate", candidate, original_reference)
    # Performance parity uses the handwritten compiler. Numerical acceptance
    # uses the original oracle for BOTH arms: two approximate outputs agreeing
    # with each other must not acquire twice the original error budget.
    observations = tuple(
        admit_protected_numerical_readback(
            run_dir=run_dir,
            environment=environment,
            original_budget=original_budget,
            budget=budget,
            reference=original_reference,
            candidate=files,
            reference_build=original_reference_build,
            candidate_build=service,
        )
        for files, service in ((reference, reference_build), (candidate, candidate_build))
    )
    # The observer reconstructs exact or elementwise violations over every word.
    # It does not accept task proxies or partial output hashes.
    metric = "bitwise_mismatch_count" if budget.profile == "exact" else "elementwise_violation_count"
    for numerical in observations:
        quality = numerical.quality
        if not quality.complete or dict(quality.values).keys() != {metric} or quality.parameters != budget.parameters:
            raise StageGateError("protected final numerical observation has a different scope or budget")
    if (observations[0].element_count, observations[0].outputs) != (
        observations[1].element_count,
        observations[1].outputs,
    ):
        raise StageGateError("protected final numerical output rosters differ")
    _pins(a.evidence)
    _pins(b.evidence)
    for witness in (a, b):
        _protected_files(run_dir, provenance, tuple(path for path, _sha in witness.evidence))
    if _protected_files(run_dir, provenance, selection) != frozen_selection:
        raise StageGateError("protected final evaluator selection changed during admission")
    verifier.validate()
    if sha256_file(original_reference.console) != binding.original_reference_console_sha256:
        raise StageGateError("protected final original reference changed during admission")
    for files, observed in ((reference, a), (candidate, b)):
        if (sha256_file(files.executable), sha256_file(files.console)) != (
            observed.executable_sha256,
            observed.console_sha256,
        ):
            raise StageGateError("protected final execution changed during numerical admission")
    identity = {
        "schema": "merlin.protected_final_comparison.v1",
        "binding": binding.identity(),
        "verifier_qualification": verifier.qualification_sha256,
        "verifier_scope": verifier.qualification.scope,
        "original_selection": [(str(path), digest) for path, digest in frozen_selection],
        "reference_witnesses": a.witnesses,
        "candidate_witnesses": b.witnesses,
        "numerical_evidence": [row.evidence for row in observations],
        "elements": observations[0].element_count,
        "outputs": observations[0].outputs,
    }
    return ProtectedFinalObservation(
        binding.member,
        a.cycles,
        b.cycles,
        sha256_bytes(canonical_json(vars(a) | {"evidence": [(str(p), s) for p, s in a.evidence]})),
        sha256_bytes(canonical_json(vars(b) | {"evidence": [(str(p), s) for p, s in b.evidence]})),
        sha256_bytes(canonical_json(identity)),
        all(dict(row.quality.values)[metric] == 0 for row in observations),
        observations[0].element_count,
        observations[0].outputs,
        verifier.qualification.scope,
    )


@dataclass(frozen=True)
class ProtectedFinalMember:
    """Host-selected original inputs for one final lifecycle execution.

    This record is private evaluator state, never an authoring tool argument.
    Source/build/hardware qualification remains the selected verifier's duty.
    """

    binding: FinalExecutionBinding
    original_binding: Path
    verifier: ProtectedExecutionVerifier
    run_dir: Path
    environment: dict
    original_budget: Path
    budget: QualityBudget
    reference: ReadbackFiles
    candidate: ReadbackFiles
    reference_build: BuildOnlyService
    candidate_build: BuildOnlyService
    original_reference: ReadbackFiles
    original_reference_build: BuildOnlyService
    physical_execution_domain: object = None


def evaluate_protected_final_campaign(
    members: tuple[ProtectedFinalMember, ...],
    *,
    original_campaign: Path,
    expected_members: tuple[str, ...],
    phase12_wall_s: float | None = None,
    handwritten_wall_s: float | None = None,
) -> dict:
    """Admit every actual member before arithmetic; never expose a tuning tool."""
    if type(members) is not tuple or any(type(row) is not ProtectedFinalMember for row in members):
        raise StageGateError("protected final campaign requires evaluator-owned lifecycle inputs")
    names = tuple(row.binding.member for row in members)
    if len(names) != len(expected_members) or len(set(names)) != len(names) or set(names) != set(expected_members):
        raise StageGateError("protected final campaign member roster differs from the original freeze")
    original = _mapping(original_campaign)
    expected = {
        "schema": "merlin.protected_final_campaign.v1",
        "members": [
            {
                "member": name,
                "binding_sha256": next(row.binding.identity() for row in members if row.binding.member == name),
            }
            for name in expected_members
        ],
    }
    if original != expected:
        raise StageGateError("protected final campaign differs from the original member/binding freeze")
    for row in members:
        _protected_files(row.run_dir, row.environment, (original_campaign,))
    campaign_sha = sha256_file(original_campaign)
    comparisons = tuple(admit_protected_final_comparison(**vars(row)) for row in members)
    for row in members:
        _protected_files(row.run_dir, row.environment, (original_campaign,))
    if sha256_file(original_campaign) != campaign_sha:
        raise StageGateError("protected final campaign changed during evaluation")
    return strict_final_component_campaign_gate(
        comparisons,
        expected_members=expected_members,
        phase12_wall_s=phase12_wall_s,
        handwritten_wall_s=handwritten_wall_s,
    ) | {"campaign_identity_sha256": campaign_sha}
