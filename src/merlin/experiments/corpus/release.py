"""Explicit prepare, inspect and operator-reviewed seal for phase-0 corpus releases.

This host-only workflow binds scientific inputs; it never approves them on behalf
of an operator or calls an agent/oracle. Native admission and grading still own
their verdicts. Review is an attributed local acknowledgement, not a signature.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from merlin.experiments.corpus.scaffold import scaffold, compiler_source_selection
from merlin.experiments.corpus.admission import admission
from merlin.experiments.corpus.preparation import assemble
from merlin.experiments.corpus.release_binding import VerifiedCorpusSnapshot
from merlin.experiments.corpus.source_staging import (
    generation_lineage,
    ordinary_tree,
    private_json,
    source_run,
    stage_phase1_policy_descriptor,
)
from merlin.experiments.definition import SpecError

# ---- release -----------------------------------------------------------------------------------


def prepare(
    run_dir: Path,
    output: Path | None = None,
    *,
    private_baseline: Path | None = None,
    retirements: Path | None = None,
    generated_only: bool = True,
    capsule_scope: str = "all",
    phase1_policy_descriptor: Path | None = None,
    loader_profile_sources: tuple[Path, ...] | None = None,
    optional_levels: tuple[str, ...] = (),
    selected_levels: tuple[str, ...] | None = None,
    selected_variants: tuple[str, ...] | None = None,
    lean_toolchain_manifest: Path | None = None,
    common_substrate: str | None = None,
    python_source_root: Path | None = None,
    host_index_bits: int | None = None,
    native_timeout_s: int | None = None,
    model_spike_timeout_s: int | None = None,
    gsim_readback_transport: str | None = None,
    operand_protocol_selection: tuple[Path, str] | None = None,
    operand_protocol_revision: tuple[Path, str] | None = None,
    kernel_timing_selection: tuple[Path, str] | None = None,
    logical_storage_budget_bytes: int | None = None,
    public_supplement: Path | None = None,
    public_scalability: Path | None = None,
    public_program_suite: Path | None = None,
    host_capability_selection: tuple[Path, str] | None = None,
    public_reference_manifest: Path | None = None,
    author_profile=None,
    author_profile_sources=None,
    author_profile_common_inputs=None,
    cca_bank: tuple[Path, str] | None = None,
) -> dict:
    """Assemble a fresh complete source pool; neither canonical inputs nor approval change."""
    from merlin.common.paths import out_dir
    from merlin.common.storage.retention import lease
    from merlin.experiments.corpus.coverage import validate_scope
    from merlin.experiments.corpus.release_binding import (
        RELEASES_HOME,
        _selected_capability_contract,
        _selected_rtl_facts,
        _stage_instruction_model,
        default_release_root,
    )
    from merlin.experiments.runs.plan_sources import fingerprint
    from merlin.target.registry.descriptor_loading import load_target_experiment

    validate_scope(capsule_scope, generated_only=generated_only)

    if (
        not isinstance(optional_levels, tuple)
        or len(set(optional_levels)) != len(optional_levels)
        or any(level not in {"EL5", "EL6"} for level in optional_levels)
    ):
        raise SpecError("optional corpus levels must be a unique tuple containing only EL5 and EL6")

    compiler_source_selection(common_substrate, python_source_root)
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
    immutable = (
        source,
        te.capsule_corpus.parent,
        *([private_baseline] if private_baseline else []),
        *([Path(python_source_root)] if python_source_root is not None else []),
        *([public_supplement.expanduser().resolve().parent] if public_supplement is not None else []),
        *([public_scalability.expanduser().resolve().parent] if public_scalability is not None else []),
        *([public_program_suite.expanduser().resolve().parent] if public_program_suite is not None else []),
        *([public_reference_manifest.expanduser().resolve().parent] if public_reference_manifest is not None else []),
        *([Path(cca_bank[0]).expanduser().resolve()] if cca_bank is not None else []),
    )
    for existing in immutable:
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
                **({"capsule_scope": capsule_scope} if capsule_scope != "all" else {}),
                **({"loader_profile_sources": loader_profile_sources} if loader_profile_sources is not None else {}),
            )
            supplement = None
            if public_supplement is not None:
                from merlin.experiments.corpus.public_supplement import stage

                private_sources = [*te.hidden_roots(), generated / "hidden", payload / "corpus/hidden"]
                if private_baseline is not None:
                    private_sources.append(private_baseline)
                supplement = stage(
                    public_supplement,
                    payload / "corpus",
                    te=te,
                    private_roots=[path for path in private_sources if path.exists()],
                )
                assembly["public_supplement"] = supplement
            selected_facts = _selected_rtl_facts(plan, generated)
            scalability = None
            if public_scalability is not None:
                from merlin.experiments.corpus.public_scalability import stage as stage_scalability

                private_sources = [*te.hidden_roots(), generated / "hidden", payload / "corpus/hidden"]
                if private_baseline is not None:
                    private_sources.append(private_baseline)
                scalability = stage_scalability(
                    public_scalability,
                    payload / "corpus/_public_scalability",
                    te=te,
                    private_roots=[p for p in private_sources if p.exists()],
                )
                assembly["public_scalability"] = scalability
            programs = None
            if public_program_suite is not None:
                from merlin.experiments.corpus.public_program_suite import stage as stage_programs

                programs = stage_programs(public_program_suite, root / "private/public_programs", target=te.target)
                assembly["public_programs"] = programs
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
                **({"common_substrate": common_substrate} if common_substrate is not None else {}),
                **({"python_source_root": python_source_root} if python_source_root is not None else {}),
                host_index_bits=host_index_bits,
                native_timeout_s=native_timeout_s,
                model_spike_timeout_s=model_spike_timeout_s,
                gsim_readback_transport=gsim_readback_transport,
                operand_protocol_selection=operand_protocol_selection,
                operand_protocol_revision=operand_protocol_revision,
                kernel_timing_selection=kernel_timing_selection,
                logical_storage_budget_bytes=logical_storage_budget_bytes,
                **({"operator_public_members": tuple(supplement["members"])} if supplement is not None else {}),
                **({"public_scalability": scalability} if scalability is not None else {}),
                **({"public_program_suite": programs} if programs is not None else {}),
                **(
                    {"host_capability_selection": host_capability_selection}
                    if host_capability_selection is not None
                    else {}
                ),
                **(
                    {"public_reference_manifest": public_reference_manifest}
                    if public_reference_manifest is not None
                    else {}
                ),
                **(
                    dict(
                        author_profile=author_profile,
                        author_profile_sources=author_profile_sources,
                        author_profile_common_inputs=author_profile_common_inputs,
                    )
                    if author_profile
                    else {}
                ),
                **({"cca_bank": cca_bank} if cca_bank is not None else {}),
            )
            if public_reference_manifest is not None:
                from merlin.experiments.corpus.performance_baselines import verify_corpus

                # Staged reference-compiler cycles must name public cases of THIS corpus, same interface.
                baselines = verify_corpus(payload / "experiment" / "reference-kernels", payload / "corpus")
                if baselines is not None:
                    scaffolding["performance_baselines"] = baselines
            descriptor = payload / "experiment" / "target_experiment.yaml"
            checked = admission(descriptor, coverage_output=root / "private" / "workload-coverage.json")
            instruction_semantics = _stage_instruction_model(plan, root / "private")
            checked_plan, _, checked_generated = source_run(source)
            if generation_lineage(checked_plan, checked_generated) != lineage:
                raise SpecError("Phase-0 generation lineage changed during corpus preparation")
            if supplement is not None:
                from merlin.experiments.corpus.public_supplement import verify as verify_supplement

                verify_supplement(supplement, payload / "corpus")
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
    from merlin.common.storage.retention import lease, pin
    from merlin.experiments.corpus.release_binding import (
        _content,
        _freeze_payload,
        _member_digests,
        _private_root,
        _read,
        _review_lock,
    )

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
        from merlin.experiments.runs.plan_sources import _phase0_synthesis_status

        plan, _, generated = source_run(Path(prepared["source_run"]))
        if generation_lineage(plan, generated) != prepared.get("generation_lineage"):
            raise SpecError("Phase-0 generation lineage differs from the operator-reviewed preparation")
        supplement = (prepared.get("assembly") or {}).get("public_supplement")
        if supplement is not None:
            from merlin.experiments.corpus.public_supplement import verify as verify_supplement

            verify_supplement(supplement, root / "payload/corpus")
        phase = plan["phases"]["0"]
        if phase.get("phase0_evidence"):
            from merlin.development.testgen.evidence.evidence import load_exported_evidence

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
        # Read-only is what verification requires; the immutable attribute is what stops a cleanup
        # that restores write bits first. Whether the host granted it is recorded, never assumed.
        from merlin.experiments.corpus.evidence_custody import seal_tree

        immutability = seal_tree([root / "payload"])
        private_json(
            root / "private" / "seal.json",
            {
                "schema_version": 1,
                "review_digest": expected_digest,
                "identity": _content(root, prepared),
                "review": {"reviewed_by": reviewed_by, "note": review_note, "at": datetime.now(UTC).isoformat()},
                "retention_pin": token,
                "members": _member_digests(root / "payload"),
                "immutability": immutability,
            },
        )
        records = seal_tree([root / "private" / "seal.json", root / "private" / "preparation.json"])
    result = inspect_release(root)
    result["seal"] = str(root / "private" / "seal.json")
    result["immutability"] = {"payload": immutability, "records": records}
    return result


def verify(seal_path: Path, descriptor: Path) -> dict:
    """Validate a sealed release selected explicitly by a functional experiment."""
    from merlin.experiments.corpus.release_binding import _content, _digest, _payload_drift, _private_root, _read

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
    sealed = _read(path)
    try:
        identity = _content(root, prepared)
    except SpecError:
        # Name what changed, from the per-member digests the seal recorded (older seals have none).
        drift = _payload_drift(root, sealed)
        if drift is None:
            raise
        raise SpecError(f"sealed corpus payload is tampered or incomplete: {drift}") from None
    workload = _verify_workload_coverage(root, prepared)
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
    from merlin.development.testgen.evidence.coverage_commitment import (
        build_phase0_readiness,
        phase0_readiness_identity,
        require_phase0_ready,
    )
    from merlin.development.testgen.evidence.coverage_obligations_coverage_commitment import (
        read_inputs,
        requires_workload_coverage,
    )
    from merlin.experiments.corpus.release_binding import _digest
    from merlin.experiments.runs.plan_sources import _read_json
    from merlin.target.registry.descriptor_loading import load_target_experiment

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
    from merlin.development.targetgen.sandbox.grants import snapshot_input_paths
    from merlin.experiments.corpus.release_binding import _INSTRUCTION_MODEL, VerifiedCorpusSnapshot, _digest
    from merlin.experiments.runs.plan_sources import fingerprint
    from merlin.target.registry.descriptor_loading import load_target_experiment

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
    from merlin.experiments.runs.plan_sources import _read_json

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


# ---- generated_source --------------------------------------------------------------------------
# Release-local corpus selection for a run with no historical corpus input.


def bind_generated_corpus(te, generated: Path, *, generated_only: bool):
    """Choose a public category without discovering any historical source pool.

    The primary category is an anchor: native discovery also includes every
    public sibling category. Its spelling therefore does not select a subset
    of the generated public cohort. Private and feedback categories stay out.
    """
    if te.capsule_corpus is not None:
        return te
    if not generated_only:
        raise SpecError("a release without a source corpus requires generated-only preparation")
    categories = [
        directory
        for directory in sorted(generated.iterdir())
        if directory.is_dir()
        and not directory.is_symlink()
        and directory.name != "hidden"
        and not directory.name.startswith(("_", "."))
        and next(directory.glob("*/capsule.yaml"), None) is not None
    ]
    if not categories:
        raise SpecError("generated-only release has no public capsule category")
    return replace(te, capsule_corpus=categories[0])


def inspect_release(path: Path) -> dict:
    """Return only safe counts/commitments, never member identities or review text."""
    from merlin.experiments.corpus.release_binding import _private_root, _read, _content, _digest

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
