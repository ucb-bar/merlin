"""Explicit prepare, inspect and operator-reviewed seal for phase-0 corpus releases.

This host-only workflow binds scientific inputs; it never approves them on behalf
of an operator or calls an agent/oracle. Native admission and grading still own
their verdicts. Review is an attributed local acknowledgement, not a signature.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from ..spec import SpecError, read_yaml
from .preparation import (
    admission,
    assemble,
    copy_input,
    generation_lineage,
    ordinary_tree,
    private_json,
    scaffold,
    source_run,
    stage_phase1_policy_descriptor,
)

_INSTRUCTION_MODEL = "instruction-semantics.json"
_PHASE1_POLICY = "phase1-policy-descriptor.yaml"


@dataclass(frozen=True)
class VerifiedCorpusSnapshot:
    """Separate the public review identity from an optional host-only input."""

    review: dict
    private_instruction_model: Path | None


@dataclass(frozen=True)
class ExactOffloadReleaseBinding:
    """Reopen the reviewed source lineage whenever exact offload is used.

    The seal alone approves a corpus, not an arbitrary model. This adapter also
    requires the model and both authored contracts to be exact selected Phase 0
    evidence sources. A transformed model needs its own selected capture receipt.
    """

    seal_path: Path
    descriptor: Path
    application: str
    review_digest: str

    def verify(self, selection) -> None:
        from merlin.common.digest import sha256_bytes

        from ..phase0.evidence import load_exported_evidence
        from ..runner import fingerprint

        review = verify(self.seal_path, self.descriptor)
        if review["review_digest"] != self.review_digest:
            raise SpecError("exact offload release review identity changed")
        root = Path(review["release"])
        prepared = _read(root / "private" / "preparation.json")
        lineage = prepared.get("generation_lineage")
        if not isinstance(lineage, dict):
            raise SpecError("exact offload requires a lineage-bound Phase 0 release")
        plan, attempt, generated = source_run(Path(prepared["source_run"]))
        phase = plan["phases"]["0"]
        selected = phase["inputs"]
        if (
            plan.get("target") != selection.target
            or prepared.get("target") != selection.target
            or prepared.get("source_plan_sha256") != fingerprint(Path(prepared["source_run"]) / "resolved-plan.json")
            or prepared.get("source_output_sha256") != attempt["output_sha256"]
            or prepared.get("source_descriptor_sha256") != fingerprint(Path(selected["descriptor"]))
            or generation_lineage(plan, generated) != lineage
        ):
            raise SpecError("exact offload differs from the sealed Phase 0 source run")
        bundle_path = plan.get("phase0_evidence_bundle")
        if not isinstance(bundle_path, str) or not bundle_path:
            raise SpecError("exact offload release has no selected Phase 0 evidence bundle")
        bundle = Path(bundle_path)
        if fingerprint(bundle / "evidence-manifest.json") != lineage.get("evidence_manifest_sha256"):
            raise SpecError("exact offload Phase 0 evidence differs from sealed generation lineage")
        evidence = load_exported_evidence(bundle)
        if evidence.target != selection.target or evidence.status != "verified":
            raise SpecError("exact offload requires verified same-target Phase 0 evidence")
        inventory_identity = evidence.application_inventory_identity
        inventory = evidence.application_inventory
        if (
            not isinstance(inventory_identity, dict)
            or inventory_identity.get("status") != "digest_bound"
            or not isinstance(inventory, dict)
            or not isinstance(inventory.get("applications"), dict)
        ):
            raise SpecError("exact offload requires a digest-bound application inventory")
        application = inventory["applications"].get(self.application)
        if not isinstance(application, dict) or application.get("capture_sha256") != selection.model_sha256:
            raise SpecError("exact offload model is not the selected Phase 0 application capture")
        inventory_sources = [source for source in evidence.source_snapshots if source.role == "application-inventory"]
        if len(inventory_sources) != 1:
            raise SpecError("exact offload application inventory has no unique selected source")
        capture_source = application.get("capture_source_path")
        if not isinstance(capture_source, str) or not capture_source:
            raise SpecError("exact offload application capture has no selected source path")
        capture_path = Path(capture_source)
        if not capture_path.is_absolute():
            capture_path = inventory_sources[0].path.parent / capture_path
        for role, selected_path, expected in (
            ("software-spec", selected.get("software_spec"), selection.software_spec_sha256),
            ("target-contract", selected.get("capability_contract"), selection.capability_contract_sha256),
            (f"application-capture:{self.application}", str(capture_path), selection.model_sha256),
        ):
            sources = [source for source in evidence.source_snapshots if source.role == role]
            if (
                len(sources) != 1
                or not isinstance(selected_path, str)
                or sources[0].path.resolve() != Path(selected_path).resolve()
                or sha256_bytes(sources[0].content) != expected
            ):
                raise SpecError(f"exact offload {role} bytes are not selected by the reviewed Phase 0 release")
        if (
            verify(self.seal_path, self.descriptor) != review
            or fingerprint(bundle / "evidence-manifest.json") != lineage["evidence_manifest_sha256"]
        ):
            raise SpecError("exact offload release or selected evidence changed during verification")


def verify_exact_offload_binding(binding, selection) -> str:
    """Installed host verifier for core's data-only exact release binding."""
    from merlin.llvmlower.exact_offload import ReleaseBinding

    if type(binding) is not ReleaseBinding:
        raise SpecError("exact offload release binding must be core's data-only record")
    ExactOffloadReleaseBinding(
        binding.seal_path, binding.descriptor, binding.application, binding.review_digest
    ).verify(selection)
    return binding.review_digest


def bind_exact_offload(selection, *, seal_path: Path, descriptor: Path, application: str):
    """Attach the host-owned reviewed release verifier to a core exact selection.

    This is a read-only trusted-host gate, not an in-process sandbox. Core has no
    dependency on experiment orchestration; its numerical oracle and rewrite
    resolve the installed verifier again at use time.
    """
    if not isinstance(application, str) or not application:
        raise SpecError("exact offload requires an explicit application label")
    from merlin.llvmlower.exact_offload import ReleaseBinding

    review = verify(seal_path, descriptor)
    binding = ReleaseBinding(
        seal_path.expanduser().absolute(),
        descriptor.expanduser().absolute(),
        application,
        review["review_digest"],
    )
    verify_exact_offload_binding(binding, selection)
    return replace(selection, release_binding=binding)


def _read(path: Path) -> dict:
    from ..runner import _read_json

    ordinary_tree(path)
    if path.stat().st_mode & 0o077:
        raise SpecError("corpus release metadata must be owner-only")
    document = _read_json(path)
    if document.get("schema_version") != 1:
        raise SpecError("unsupported corpus release record version")
    return document


def _digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _private_root(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise SpecError("corpus release root may not be a symlink")
    root = lexical.resolve(strict=True)
    if (root / "private").is_symlink() or not (root / "private").is_dir():
        raise SpecError("corpus release private metadata directory is missing or symlinked")
    if root.stat().st_mode & 0o077 or (root / "private").stat().st_mode & 0o077:
        raise SpecError("corpus release and private metadata directories must be owner-only")
    return root


@contextmanager
def _review_lock(root: Path):
    descriptor = os.open(root / "private" / ".review.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpecError("another process is reviewing this corpus release") from exc
        yield


def _content(root: Path, prepared: dict) -> dict:
    from ..runner import fingerprint

    ordinary_tree(root / "payload")
    observed = fingerprint(root / "payload")
    if prepared.get("payload_sha256") != observed:
        raise SpecError("prepared corpus payload changed; prepare a new release")
    model = prepared.get("instruction_semantics")
    member = root / "private" / _INSTRUCTION_MODEL
    if model is None:
        if member.exists() or member.is_symlink():
            raise SpecError("unrecorded private instruction model in corpus release")
    else:
        if not isinstance(model, dict) or not isinstance(model.get("sha256"), str):
            raise SpecError("invalid private instruction model commitment")
        ordinary_tree(member)
        if not member.is_file() or member.stat().st_mode & 0o077:
            raise SpecError("private instruction model must be an owner-only ordinary file")
        if fingerprint(member) != model["sha256"]:
            raise SpecError("private instruction model changed; prepare a new release")
    policy = prepared.get("phase1_policy")
    retained = root / "private" / _PHASE1_POLICY
    if policy is None:
        if retained.exists() or retained.is_symlink():
            raise SpecError("unrecorded Phase 1 policy descriptor in corpus release")
    else:
        if not isinstance(policy, dict) or not isinstance(policy.get("sha256"), str):
            raise SpecError("invalid Phase 1 policy descriptor commitment")
        ordinary_tree(retained)
        if not retained.is_file() or retained.stat().st_mode & 0o077:
            raise SpecError("retained Phase 1 policy descriptor must be owner-only")
        if fingerprint(retained) != policy["sha256"]:
            raise SpecError("retained Phase 1 policy descriptor changed; prepare a new release")
        selected = read_yaml(retained)
        promoted = read_yaml(root / "payload" / "experiment" / "target_experiment.yaml")
        if (
            not isinstance(selected, dict)
            or not isinstance(promoted, dict)
            or selected.get("phase1_gates") != promoted.get("phase1_gates")
        ):
            raise SpecError("promoted Phase 1 gates differ from retained policy descriptor")
    return {"preparation_sha256": _digest(prepared), "payload_sha256": observed}


def _stage_instruction_model(plan: dict, private: Path) -> dict | None:
    """Copy only a selected Phase 0 consumer model into host-private release input."""
    from ..runner import fingerprint

    location = plan.get("phase0_evidence_bundle")
    if location is None:
        return None
    if not isinstance(location, str) or not location:
        raise SpecError("selected Phase 0 evidence bundle path is invalid")
    from ..phase0.evidence import load_exported_evidence

    bundle = Path(location).expanduser().absolute()
    selected = load_exported_evidence(bundle)
    if selected.target != plan["target"]:
        raise SpecError("selected instruction model target differs from corpus release")
    member = bundle / "software" / "instruction-semantics.json"
    if member.is_symlink() or not member.is_file():
        raise SpecError("selected Phase 0 instruction model is missing or symlinked")
    destination = private / _INSTRUCTION_MODEL
    digest = copy_input(member, destination, private=True)
    archived = dict(selected.archived_artifacts)
    selected_bytes = archived.get("software/instruction-semantics.json")
    if selected_bytes is None or digest != hashlib.sha256(selected_bytes).hexdigest():
        raise SpecError("selected Phase 0 instruction model changed after evidence verification")
    manifest_bytes = archived.get("evidence-manifest.json")
    manifest_sha = fingerprint(bundle / "evidence-manifest.json")
    if manifest_bytes is None or manifest_sha != hashlib.sha256(manifest_bytes).hexdigest():
        raise SpecError("selected Phase 0 evidence manifest changed after verification")
    destination.chmod(0o400)
    return {"sha256": digest, "source_evidence_sha256": manifest_sha}


def _selected_rtl_facts(plan: dict, generated: Path) -> Path | None:
    """Use the exact facts exported by the selected Phase 0 evidence bundle."""
    location = plan.get("phase0_evidence_bundle")
    if location is None:
        # Historical diagnostic runs could emit facts directly in their
        # generated corpus. Never use this path when evidence was selected.
        legacy = generated / "hardware/effective-views/loaded-facts.json"
        return legacy if legacy.is_file() and not legacy.is_symlink() else None
    from ..phase0.evidence import load_exported_evidence

    bundle = Path(location).expanduser().absolute()
    evidence = load_exported_evidence(bundle)
    relative = Path("hardware/effective-views/loaded-facts.json")
    selected = dict(evidence.archived_artifacts).get(relative.as_posix())
    path = bundle / relative
    if selected is None or path.is_symlink() or not path.is_file() or path.read_bytes() != selected:
        raise SpecError("selected Phase 0 RTL facts are absent or differ from verified evidence")
    try:
        document = json.loads(selected)
    except (TypeError, ValueError) as error:
        raise SpecError("selected Phase 0 RTL facts are not JSON") from error
    facts = document.get("facts") if isinstance(document, dict) else None
    if not isinstance(facts, dict) or facts.get("target") != plan["target"]:
        raise SpecError("selected Phase 0 RTL facts name a different target")
    return path


def _selected_capability_contract(plan: dict, generated: Path) -> tuple[Path, str] | None:
    """Carry the effective Phase 0 view, never re-resolve a live provider contract."""
    location = plan.get("phase0_evidence_bundle")
    relative = Path("software/contract.json")
    if location is None:
        legacy = generated / relative
        if not legacy.exists():
            if plan["phases"]["0"]["inputs"].get("capability_contract"):
                raise SpecError("selected Phase 0 capability contract has no exported view; freeze a new run")
            return None
        path = legacy
        ordinary_tree(path)
        raw = path.read_bytes()
    else:
        from ..phase0.evidence import load_exported_evidence

        bundle = Path(location).expanduser().absolute()
        evidence = load_exported_evidence(bundle)
        raw = dict(evidence.archived_artifacts).get(relative.as_posix())
        path = bundle / relative
        if raw is None or path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise SpecError("selected Phase 0 capability contract is absent or differs from verified evidence")
    try:
        document = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise SpecError("selected Phase 0 capability contract is not JSON") from error
    if not isinstance(document, dict) or document.get("name") != plan["target"]:
        raise SpecError("selected Phase 0 capability contract names a different target")
    return path, hashlib.sha256(raw).hexdigest()


def _freeze_payload(root: Path) -> None:
    from merlin.common.content_store import is_shared

    payload = root / "payload"
    ordinary_tree(payload)
    for path in sorted(payload.rglob("*"), key=lambda member: len(member.parts), reverse=True):
        if path.is_file() and is_shared(path):
            if path.stat().st_mode & 0o222:
                raise SpecError("shared corpus payload is unexpectedly writable")
            continue
        mode = 0o500 if path.is_dir() or path.stat().st_mode & 0o111 else 0o400
        path.chmod(mode)
    payload.chmod(0o500)


def inspect_release(path: Path) -> dict:
    """Return only safe counts/commitments, never member identities or review text."""
    root = _private_root(path)
    prepared = _read(root / "private" / "preparation.json")
    identity = _content(root, prepared)
    sealed = (root / "private" / "seal.json").is_file()
    if sealed:
        verify(root / "private" / "seal.json", root / "payload" / "experiment" / "target_experiment.yaml")
    return {
        "schema_version": 1,
        "release": str(root),
        "target": prepared["target"],
        "state": "sealed" if sealed else "awaiting_operator_review",
        "review_digest": _digest(identity),
        "payload_sha256": identity["payload_sha256"],
        "generation_lineage_sha256": (
            _digest(prepared["generation_lineage"]) if prepared.get("generation_lineage") is not None else None
        ),
        "descriptor": str(root / "payload" / "experiment" / "target_experiment.yaml"),
        "counts": prepared["admission"],
        "engine_readiness": "not_executed",
    }


#: The storage contract's ``product_roots`` entry for sealed phase-0 releases.
RELEASES_HOME = "phase0-releases"


def default_release_root(target: str, *, timestamp: str | None = None, sha: str | None = None) -> Path:
    """``out/artifacts/protocols/<target>/phase0-<TS>-<sha7>/``: the target axis of a new release."""
    from merlin.common.artifacts import declared_home, git_sha7, utc_stamp
    from merlin.common.paths import repo_root

    if not target or Path(target).name != target or target in (".", ".."):
        raise SpecError(f"corpus release target must be one path component: {target!r}")
    stamp = timestamp or utc_stamp()
    return declared_home(RELEASES_HOME) / target / f"phase0-{stamp}-{sha or git_sha7(repo_root())}"


def prepare(
    run_dir: Path,
    output: Path | None = None,
    *,
    private_baseline: Path | None = None,
    retirements: Path | None = None,
    generated_only: bool = False,
    phase1_policy_descriptor: Path | None = None,
    loader_profile_sources: tuple[Path, ...] | None = None,
    optional_levels: tuple[str, ...] = (),
    selected_levels: tuple[str, ...] | None = None,
    selected_variants: tuple[str, ...] | None = None,
    lean_toolchain_manifest: Path | None = None,
    host_index_bits: int | None = None,
) -> dict:
    """Assemble a fresh complete source pool; neither canonical inputs nor approval change."""
    from merlin.common.paths import out_dir
    from merlin.common.storage_lifecycle import lease
    from merlin.targetgen.target_experiment import load_target_experiment

    from ..runner import fingerprint

    if not isinstance(optional_levels, tuple) or len(set(optional_levels)) != len(optional_levels) or any(
        level not in {"EL5", "EL6"} for level in optional_levels
    ):
        raise SpecError("optional corpus levels must be a unique tuple containing only EL5 and EL6")

    source = run_dir.expanduser().resolve(strict=True)
    if output is None:
        selected, _, _ = source_run(source)
        output = default_release_root(load_target_experiment(selected["phases"]["0"]["inputs"]["descriptor"]).target)
    root = output.expanduser().absolute()
    if root.is_symlink() or root.exists():
        raise SpecError("corpus release output already exists; choose a fresh destination")
    root = root.resolve()
    generated_root = out_dir().resolve()
    if not root.is_relative_to(generated_root / "artifacts"):
        raise SpecError("corpus releases must live below the configured artifact root")
    plan, attempt, generated = source_run(source)
    lineage = generation_lineage(plan, generated)
    te = load_target_experiment(plan["phases"]["0"]["inputs"]["descriptor"])
    from .generated_source import bind_generated_corpus

    te = bind_generated_corpus(te, generated, generated_only=generated_only)
    from merlin.common.artifacts import declared_home

    try:
        releases = declared_home(RELEASES_HOME).resolve()
    except ValueError:
        releases = None  # a checkout without the storage contract declares no release home to police
    if releases is not None and root.is_relative_to(releases) and root.relative_to(releases).parts[:1] != (te.target,):
        raise SpecError(f"a corpus release under {releases.name}/ must sit below its target {te.target!r}")
    if private_baseline is not None:
        private_baseline = private_baseline.expanduser().absolute()
        if private_baseline.is_symlink() or not private_baseline.is_dir():
            raise SpecError("private baseline must be an ordinary existing directory")
    if retirements is not None:
        retirements = retirements.expanduser().absolute()
        if not retirements.is_file() or retirements.is_symlink():
            raise SpecError("retirements must be an ordinary existing file")
        retirements = retirements.resolve(strict=True)
    for existing in (source, te.capsule_corpus.parent, *([private_baseline] if private_baseline else [])):
        if root == existing or root.is_relative_to(existing) or existing.is_relative_to(root):
            raise SpecError("corpus release output overlaps an immutable source")
    with lease(root, owner="corpus-release-preparation"):
        root.mkdir(parents=True, mode=0o700)
        (root / "private").mkdir(mode=0o700)
        (root / "payload").mkdir(mode=0o700)
        payload = root / "payload"
        try:
            phase1_policy = None
            phase1_gates = None
            if phase1_policy_descriptor is not None:
                phase1_policy, phase1_gates = stage_phase1_policy_descriptor(
                    te.path, phase1_policy_descriptor, root / "private"
                )
            assembly = assemble(
                te,
                generated,
                payload / "corpus",
                private_baseline=private_baseline,
                retirements=retirements,
                generated_only=generated_only,
                **({"loader_profile_sources": loader_profile_sources} if loader_profile_sources is not None else {}),
            )
            selected_facts = _selected_rtl_facts(plan, generated)
            selected_contract = _selected_capability_contract(plan, generated)
            scaffolding = scaffold(
                te,
                payload / "corpus",
                payload / "experiment",
                private=root / "private",
                selected_facts=selected_facts,
                selected_contract=selected_contract,
                phase1_gates=phase1_gates,
                optional_levels=optional_levels,
                **({"selected_levels": selected_levels} if selected_levels is not None else {}),
                **({"selected_variants": selected_variants} if selected_variants is not None else {}),
                lean_toolchain_manifest=lean_toolchain_manifest,
                host_index_bits=host_index_bits,
            )
            descriptor = payload / "experiment" / "target_experiment.yaml"
            checked = admission(descriptor, coverage_output=root / "private" / "workload-coverage.json")
            instruction_semantics = _stage_instruction_model(plan, root / "private")
            checked_plan, _, checked_generated = source_run(source)
            if generation_lineage(checked_plan, checked_generated) != lineage:
                raise SpecError("Phase-0 generation lineage changed during corpus preparation")
            private_json(
                root / "private" / "preparation.json",
                {
                    "schema_version": 1,
                    "target": te.target,
                    "source_run": str(source),
                    "source_plan_sha256": fingerprint(source / "resolved-plan.json"),
                    "source_output_sha256": attempt["output_sha256"],
                    "source_descriptor_sha256": fingerprint(te.path),
                    "phase1_policy": phase1_policy,
                    "generation_lineage": lineage,
                    "assembly": assembly,
                    "scaffolding": scaffolding,
                    "admission": checked,
                    "instruction_semantics": instruction_semantics,
                    "payload_sha256": fingerprint(payload),
                    "prepared_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as exc:
            private_json(
                root / "private" / "failure.json",
                {
                    "schema_version": 1,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise SpecError(
                "corpus preparation failed; host-only diagnostics are in the release private directory"
            ) from None
    return inspect_release(root)


def seal(path: Path, *, expected_digest: str, reviewed_by: str, review_note: str) -> dict:
    """Record an explicit operator review of these exact bytes; never called by prepare/run."""
    from merlin.common.storage_lifecycle import lease, pin

    if not reviewed_by.strip() or not review_note.strip():
        raise SpecError("sealing requires an explicit reviewer and review note")
    root = _private_root(path)
    with _review_lock(root), lease(root, owner="corpus-release-sealing"):
        if (root / "private" / "seal.json").exists():
            raise SpecError("corpus release is already sealed; review records are not overwritten")
        report = inspect_release(root)
        if report["review_digest"] != expected_digest:
            raise SpecError("review digest does not match the prepared release")
        prepared = _read(root / "private" / "preparation.json")
        _verify_workload_coverage(root, prepared)
        # Preparation remains available for historical/diagnostic runs, but an
        # operator review must not upgrade them to verified corpus provenance.
        # source_run rechecks the frozen plan, source/output receipts and every
        # input fingerprint before these selected bytes are interpreted.
        from ..runner import _phase0_synthesis_status

        plan, _, generated = source_run(Path(prepared["source_run"]))
        if generation_lineage(plan, generated) != prepared.get("generation_lineage"):
            raise SpecError("Phase-0 generation lineage differs from the operator-reviewed preparation")
        phase = plan["phases"]["0"]
        if phase.get("phase0_evidence"):
            from ..phase0.evidence import load_exported_evidence

            observed = load_exported_evidence(plan["phase0_evidence_bundle"])
            if observed.status != "verified":
                raise SpecError("diagnostic Phase 0 evidence cannot be promoted to a verified corpus release")
        if phase.get("module") or phase["inputs"].get("synth_profile"):
            required = {f"phase0:operator:{name}" for name in ("recipe", "conformance_spec", "synth_profile")}
            frozen = plan.get("phase0_operator_inputs") or {}
            if not required <= set(frozen) or any(
                frozen[name] is None or not frozen[name]["present"] for name in required
            ):
                raise SpecError(
                    "corpus seal requires a newly frozen explicit Phase-0 run with recipe, "
                    "conformance spec, and digest-bound selected synthesis"
                )
            synthesis = _phase0_synthesis_status(plan).get("0", {})
            if synthesis.get("status") != "verified":
                raise SpecError(
                    "corpus seal requires verified selected synthesis; regenerate and review a "
                    "digest-bound profile, then freeze a new Phase-0 run"
                )
        try:
            current = admission(Path(report["descriptor"]))
        except Exception:
            raise SpecError("native corpus admission no longer validates the reviewed release") from None
        if current != prepared["admission"] or inspect_release(root)["review_digest"] != expected_digest:
            raise SpecError("reviewed inputs or native admission changed during sealing")
        _freeze_payload(root)
        token = pin(root, reason="explicitly reviewed corpus release")
        private_json(
            root / "private" / "seal.json",
            {
                "schema_version": 1,
                "review_digest": expected_digest,
                "identity": _content(root, prepared),
                "review": {"reviewed_by": reviewed_by, "note": review_note, "at": datetime.now(UTC).isoformat()},
                "retention_pin": token,
            },
        )
    result = inspect_release(root)
    result["seal"] = str(root / "private" / "seal.json")
    return result


def verify(seal_path: Path, descriptor: Path) -> dict:
    """Validate a sealed release selected explicitly by a functional experiment."""
    path = seal_path.expanduser().absolute()
    if path.name != "seal.json" or path.parent.name != "private":
        raise SpecError("corpus seal must name the release's private seal.json record")
    root = _private_root(path.parent.parent)
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise SpecError("corpus seal must be an ordinary owner-only file")
    expected_descriptor = root / "payload" / "experiment" / "target_experiment.yaml"
    if descriptor.resolve() != expected_descriptor:
        raise SpecError("functional descriptor is not the one bound by the corpus seal")
    prepared = _read(root / "private" / "preparation.json")
    workload = _verify_workload_coverage(root, prepared)
    sealed = _read(path)
    identity = _content(root, prepared)
    if any(member.stat().st_mode & 0o222 for member in [root / "payload", *(root / "payload").rglob("*")]):
        raise SpecError("sealed corpus payload became writable")
    if sealed.get("identity") != identity or sealed.get("review_digest") != _digest(identity):
        raise SpecError("corpus seal does not bind this prepared payload")
    review = sealed.get("review")
    if not isinstance(review, dict) or not all(
        isinstance(review.get(k), str) and review[k].strip() for k in ("reviewed_by", "note", "at")
    ):
        raise SpecError("corpus seal has no explicit operator review")
    return {
        "release": str(root),
        "review_digest": sealed["review_digest"],
        "payload_sha256": identity["payload_sha256"],
        "phase0_readiness": (prepared.get("admission") or {}).get("phase0_readiness"),
        "whole_workload_phase1": workload,
    }


def _verify_workload_coverage(root: Path, prepared: dict) -> dict:
    """Require the deterministic handoff, without pre-certifying Phase 1."""
    from merlin.targetgen.target_experiment import load_target_experiment

    from ..phase0.coverage_commitment import (
        build_phase0_readiness,
        phase0_readiness_identity,
        read_inputs,
        require_phase0_ready,
        requires_workload_coverage,
    )
    from ..runner import _read_json

    descriptor = root / "payload" / "experiment" / "target_experiment.yaml"
    te = load_target_experiment(descriptor)
    selected = read_inputs(te.capsule_corpus.parent)
    required = requires_workload_coverage(te, selected)
    summary = (prepared.get("admission") or {}).get("whole_workload_phase1")
    if not isinstance(summary, dict):
        if required:
            raise SpecError("historical corpus release has no verified whole-workload completeness commitment")
        return {"required": False, "status": "not_established"}
    path = root / "private" / "workload-coverage.json"
    ordinary_tree(path)
    if path.stat().st_mode & 0o077:
        raise SpecError("workload coverage metadata must be owner-only")
    report = _read_json(path)
    if _digest(report) != summary.get("report_sha256") or summary.get("required") != required:
        raise SpecError("workload coverage report differs from prepared corpus admission")
    if required:
        readiness = build_phase0_readiness(report)
        selected_readiness = (prepared.get("admission") or {}).get("phase0_readiness")
        if phase0_readiness_identity(readiness, required=True) != selected_readiness:
            raise SpecError("Phase 0 readiness differs from prepared corpus admission")
        try:
            require_phase0_ready(readiness)
        except ValueError as exc:
            raise SpecError(str(exc)) from exc
    return summary


def _verify_snapshot(
    seal_path: Path, descriptor: Path, ws: Path, bundle: dict, *, repo: Path | None = None
) -> VerifiedCorpusSnapshot:
    """Bind actual private snapshot bytes to the operator-reviewed source bytes."""
    from merlin.targetgen.sandbox.bwrap import snapshot_input_paths
    from merlin.targetgen.target_experiment import load_target_experiment

    from ..runner import fingerprint

    identity = verify(seal_path, descriptor)
    te = load_target_experiment(descriptor)
    sources = [*te.graded_roots(), *te.hidden_roots()]
    if te.declared_contract:
        contract = te.declared_contract_path()
        if contract is None:
            raise SpecError("reviewed capability contract is absent; freeze and prepare a new release")
        sources.append(contract)
    sources += [te.resource_path("task"), te.resource_path("scripts/agent_selfcheck.py")]
    if (te.capsule_corpus.parent / "_phase0").is_dir():
        sources += [te.capsule_corpus.parent / "_phase0"]
    sources += [Path(identity["release"]) / "private"]
    snapshots = snapshot_input_paths(ws, bundle, sources, repo=repo)
    for source, snapshot in zip(sources, snapshots, strict=True):
        if fingerprint(source) != fingerprint(snapshot):
            raise SpecError("native corpus snapshot differs from the operator-reviewed release")
    # Review metadata is a host-only snapshot input too. Read its frozen copy,
    # never an absent-copy fallback to mutable release metadata on resume.
    from ..runner import _read_json

    frozen_prepared = _read_json(snapshots[-1] / "preparation.json")
    frozen_sealed = _read_json(snapshots[-1] / "seal.json")
    frozen_identity = {
        "preparation_sha256": _digest(frozen_prepared),
        "payload_sha256": identity["payload_sha256"],
    }
    if (
        frozen_sealed.get("identity") != frozen_identity
        or frozen_sealed.get("review_digest") != identity["review_digest"]
    ):
        raise SpecError("native review snapshot does not bind the operator-reviewed release")
    # A change while the native snapshot was checked cannot borrow its seal.
    if verify(seal_path, descriptor) != identity:
        raise SpecError("corpus release changed while verifying its native snapshot")
    model = frozen_prepared.get("instruction_semantics")
    private_model = None
    if model is not None:
        member = snapshots[-1] / _INSTRUCTION_MODEL
        if (
            not isinstance(model, dict)
            or member.is_symlink()
            or not member.is_file()
            or member.stat().st_mode & 0o077
            or fingerprint(member) != model.get("sha256")
        ):
            raise SpecError("frozen private instruction model differs from reviewed release")
        private_model = member
    return VerifiedCorpusSnapshot(dict(identity), private_model)


def verify_snapshot(seal_path: Path, descriptor: Path, ws: Path, bundle: dict, *, repo: Path | None = None) -> dict:
    """Native pre-agent gate returning only the reviewed corpus identity."""
    return _verify_snapshot(seal_path, descriptor, ws, bundle, repo=repo).review


def verify_snapshot_for_phase1(
    seal_path: Path, descriptor: Path, ws: Path, bundle: dict, *, repo: Path | None = None
) -> VerifiedCorpusSnapshot:
    """Trusted Phase 1 host gate with an explicit, separately typed private input."""
    return _verify_snapshot(seal_path, descriptor, ws, bundle, repo=repo)
