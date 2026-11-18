"""Private corpus review records and source-bound exact-offload verification.

The operator owns review and sealing. These helpers reopen original sources and
private commitments; they do not supply numerical or execution qualification.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.experiments.corpus.source_staging import (
    copy_input,
    generation_lineage,
    ordinary_tree,
    source_run,
)
from merlin.experiments.definition import SpecError, read_yaml

# ---- release -----------------------------------------------------------------------------------

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
        from merlin.experiments.corpus.release import verify

        from merlin.common.digest import sha256_bytes
        from merlin.development.testgen.evidence.evidence import load_exported_evidence
        from merlin.experiments.runs.plan_sources import fingerprint

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
    from merlin.compiler.lower.device.head import ReleaseBinding

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
    from merlin.experiments.corpus.release import verify

    if not isinstance(application, str) or not application:
        raise SpecError("exact offload requires an explicit application label")
    from merlin.compiler.lower.device.head import ReleaseBinding

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
    from merlin.experiments.runs.plan_sources import _read_json

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
    from merlin.experiments.runs.plan_sources import fingerprint

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
    selected_host = (prepared.get("scaffolding") or {}).get("host_capability_selection")
    if selected_host is not None:
        from merlin.experiments.corpus.scaffold import verify_host_capability

        descriptor = root / "payload/experiment/target_experiment.yaml"
        verify_host_capability(read_yaml(descriptor), selected_host, descriptor=descriptor)
    _verify_public_programs(root, prepared)
    return {"preparation_sha256": _digest(prepared), "payload_sha256": observed}


def _verify_public_programs(root: Path, prepared: dict) -> None:
    """Reopen private program custody and its answer-free payload before review."""
    from merlin.experiments.corpus import public_program_suite as programs
    from merlin.target.registry.descriptor_loading import load_target_experiment

    te = load_target_experiment(root / "payload/experiment/target_experiment.yaml")
    binding = (te.workload_spec or {}).get(programs.KEY)
    expected = (prepared.get("assembly") or {}).get("public_programs")
    if binding is None and expected is None:
        return
    if binding != expected:
        raise SpecError("public program descriptor differs from its prepared source selection")
    selected = programs.selected(te)
    if selected is None:
        raise SpecError("prepared public program selection is absent")
    programs.public_view(selected[0], root / "payload/experiment/public-program-sources", target=te.target, verify=True)


def _stage_instruction_model(plan: dict, private: Path) -> dict | None:
    """Copy only a selected Phase 0 consumer model into host-private release input."""
    from merlin.experiments.runs.plan_sources import fingerprint

    location = plan.get("phase0_evidence_bundle")
    if location is None:
        return None
    if not isinstance(location, str) or not location:
        raise SpecError("selected Phase 0 evidence bundle path is invalid")
    from merlin.development.testgen.evidence.evidence import load_exported_evidence

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
    from merlin.development.testgen.evidence.evidence import load_exported_evidence

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
        from merlin.development.testgen.evidence.evidence import load_exported_evidence

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
    from merlin.common.storage.content_store import is_shared

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


def _member_digests(payload: Path) -> dict[str, str]:
    """Every payload file's digest, so a later verification can name what changed or vanished."""
    out = {}
    for member in sorted(payload.rglob("*")):
        if member.is_file() and not member.is_symlink():
            out[member.relative_to(payload).as_posix()] = hashlib.sha256(member.read_bytes()).hexdigest()
    return out


def _payload_drift(root: Path, sealed: dict) -> str | None:
    """Which sealed payload members are missing, tampered or new; host-private names are withheld."""
    recorded = sealed.get("members")
    if not isinstance(recorded, dict):
        return None
    payload = root / "payload"
    observed = _member_digests(payload)
    try:
        from merlin.target.registry.descriptor_loading import load_target_experiment

        private = [
            Path(p) for p in load_target_experiment(payload / "experiment/target_experiment.yaml").hidden_roots()
        ]
    except Exception:  # noqa: BLE001 -- with no readable descriptor every name is withheld
        private = None

    def shown(member: str) -> str:
        path = payload / member
        if private is None or any(path.is_relative_to(r) for r in private):
            return "a host-private member"
        return member

    rows = {
        "missing": sorted(m for m in recorded if m not in observed),
        "tampered": sorted(m for m in recorded if m in observed and observed[m] != recorded[m]),
        "unsealed": sorted(m for m in observed if m not in recorded),
    }
    if not any(rows.values()):
        return None
    parts = []
    for kind, members in rows.items():
        if members:
            names = sorted({shown(m) for m in members})
            parts.append(f"{kind} {len(members)}: " + ", ".join(names[:20]) + (" ..." if len(names) > 20 else ""))
    return "; ".join(parts)


RELEASES_HOME = "phase0-releases"


def default_release_root(target: str, *, timestamp: str | None = None, sha: str | None = None) -> Path:
    """``out/artifacts/protocols/<target>/phase0-<TS>-<sha7>/``: the target axis of a new release."""
    from merlin.common.artifacts import declared_home, git_sha7, utc_stamp
    from merlin.common.paths import repo_root

    if not target or Path(target).name != target or target in (".", ".."):
        raise SpecError(f"corpus release target must be one path component: {target!r}")
    stamp = timestamp or utc_stamp()
    return declared_home(RELEASES_HOME) / target / f"phase0-{stamp}-{sha or git_sha7(repo_root())}"
