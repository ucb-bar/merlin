"""Host corpus workflows whose tier policy depends on constructed evaluator adapters.

Core owns conformance derivation, copying, admission validation, coverage and
leases. This module owns asking real evaluator factories what they constructed;
advertised tier metadata is never a substitute for that observation. Construction
does not itself prove that an oracle can execute or certify a submitted program.
"""

from __future__ import annotations

from pathlib import Path

from merlin.development.targetgen.conformance import axes as _merlin_axes_1
from merlin.development.targetgen.corpus import materialization as materialize
from merlin.development.targetgen.corpus.materialization import validate_materialized_cohort
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.evaluation.capsule.evidence import certificates as capsule_availability

# ---- admission ---------------------------------------------------------------------------------


def constructed_oracle_tiers(target: str) -> list:
    """Actual constructed tier keys, sorted; unresolved or empty remains unknown.

    Use the standard descriptor for this target, not the single-descriptor override.
    The original best-effort conformance query intentionally turns construction
    failures into an empty observation, never an invented cheaper tier.
    """
    try:
        from merlin.target.registry import descriptor_loading as corpora
        from merlin.target.registry.descriptor_loading import load_target_experiment

        desc = corpora.standard_descriptor_path(str(target))
        if not desc.is_file():
            return []
        te = load_target_experiment(desc)
        return sorted(capsule_adapters.oracle_adapters(target, te.sim_via) or {})
    except Exception:  # noqa: BLE001 — unresolvable: report nothing
        return []


def conformance_spec(target: str, captures: dict[str, str | Path], **kwargs) -> dict:
    """Derive the existing requirement with a fresh constructed-tier observation."""
    return _merlin_axes_1.derive_spec(target, captures, oracle_tiers=lambda: constructed_oracle_tiers(target), **kwargs)


def public_capsules_for(
    te, *, tier_ceiling: str | None = None, corpus_roots: list[Path] | None = None, destination: Path | None = None
) -> Path:
    """Publish the public cohort without substituting undeclared oracle tiers.

    An explicit ceiling bypasses all evaluator imports and construction. Otherwise
    retain the native QA-loop query followed by a second full construction: overrides,
    constructor failures and the empty-declaration legacy behavior remain authoritative.
    """
    roots = materialize.descriptor_corpus_roots(te) if corpus_roots is None else corpus_roots
    if tier_ceiling is None:
        declared = materialize.declared_oracle_tiers(*roots)
        loop = capsule_availability.qa_loop_adapters(te.target, te.sim_via, declared_tiers=declared)
        if not loop:
            reach = sorted(capsule_adapters.oracle_adapters(te.target, te.sim_via))
            if reach:
                raise ValueError(
                    f"target {te.target!r}: its capsule corpus declares required oracle tiers "
                    f"{sorted(declared)} but the endpoint reaches {reach} — no declared tier is "
                    f"reachable, so this phase cannot grade these capsules. Refusing to substitute a "
                    f"tier the capsules never declared; make the declared tier reachable or fix the "
                    f"corpus."
                )
            # No adapter was constructed: retain the legacy floor. The unchanged
            # materializer still records unreachable declared tiers for the grader.
            tier_ceiling = materialize._DEFAULT_CEILING
        else:
            # Preserve the highest declared reachable tier, not just the cheap
            # loop tier; otherwise materialization silently removes cert barriers.
            reach = set(capsule_adapters.oracle_adapters(te.target, te.sim_via))
            usable = (declared & reach) or set(loop)
            tier_ceiling = max(
                usable, key=lambda tier: materialize._TIER_ORDER.index(tier) if tier in materialize._TIER_ORDER else -1
            )
    options = {"destination": destination} if destination is not None else {}
    return materialize.materialize_public_cohort(te, tier_ceiling=tier_ceiling, corpus_roots=roots, **options)


__all__ = [
    "admission",
    "derive_resource_bound",
    "derive_release_admission",
    "conformance_spec",
    "constructed_oracle_tiers",
    "public_capsules_for",
    "validate_materialized_cohort",
]


DERIVED_RESOURCE_BOUND = "phase0_qualified_models_v1"


def derive_resource_bound(corpus: Path) -> dict:
    """The model resource policy the staged corpus itself states, never a hand-maintained name list.

    Every public whole-model capsule Phase 0 GENERATED and ADMITTED is a required capstone: an
    integer model is admitted only with a qualified ``model_qualification`` record, and a model whose
    arithmetic needs no such bound was admitted by being written. Any other public model (a retained,
    hand-authored or unqualified one) is excluded by name. Both lists name only capsules present.
    """
    from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
    from merlin.experiments.definition import SpecError, read_yaml

    manifest = read_yaml(corpus / "MANIFEST.yaml")
    generated = {str(member) for member in manifest.get("generated") or ()}
    required, excluded = [], []
    for cap in capsule_capsule_common.discover_capsules(corpus, labels={"public", "dev"}):
        if cap.get("kind") != "model":
            continue
        name = str(cap["name"])
        member = Path(str(cap.get("__dir__") or "")).resolve()
        relative = f"{member.parent.name}/{member.name}"
        if member.parent.name.startswith("_"):
            continue
        qualification = cap.get("model_qualification") or {}
        needs_bound = "integer_partial_sum_bound" in cap or qualification
        admitted = relative in generated and (not needs_bound or qualification.get("status") == "qualified")
        (required if admitted else excluded).append(name)
    if not required:
        raise SpecError("derived resource policy found no admitted whole-model capstone in the staged corpus")
    return {
        "derive": DERIVED_RESOURCE_BOUND,
        "required_admitted_models": sorted(required),
        "exclude_capsules": sorted(excluded),
    }


def derive_release_admission(staged) -> dict:
    """Seal a staged corpus's capability and cardinality decisions for review.

    Model resource decisions are authored policy until comparable cost evidence
    exists. Require one explicit decision per model so corpus growth cannot
    silently expand the expensive formal denominator.
    """
    from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
    from merlin.evaluation.capsule import eligibility as capsule_eligibility
    from merlin.experiments.definition import SpecError

    from merlin.target.capabilities import eligibility

    if not eligibility.capability_map_for_target(staged.target):
        raise SpecError("release admission requires a resolvable nonempty hardware capability contract")
    public = capsule_capsule_common.discover_capsules(staged.graded_roots(), labels={"public", "dev"})
    public_names = [str(cap.get("name")) for cap in public]
    if len(public_names) != len(set(public_names)):
        raise SpecError("release public corpus has duplicate capsule names")
    models = {str(cap["name"]) for cap in public if cap.get("kind") == "model"}
    resource = set(staged.graded_resource_exclude)
    required = set(staged.graded_required_models)
    policy = resource | required
    issues = []
    if overlap := sorted(resource & required):
        issues.append("models both excluded and required admitted: " + ", ".join(overlap))
    if unclassified := sorted(models - policy):
        issues.append("unclassified public models: " + ", ".join(unclassified))
    if stale := sorted(policy - models):
        issues.append("policy names absent from staged public models: " + ", ".join(stale))
    operations = [cap for cap in public if cap.get("kind") != "model"]
    _, withheld = capsule_eligibility._split_ineligible(operations, staged.target)
    capability = sorted({str(row["capsule"]) for row in withheld})
    if set(capability) & models:
        raise SpecError("model capability admission cannot be inferred from operation admission")
    hidden = capsule_capsule_common.discover_capsules(staged.hidden_roots(), labels={"hidden"})
    hidden_ops = [cap for cap in hidden if cap.get("kind") != "model"]
    _, hidden_withheld = capsule_eligibility._split_ineligible(hidden_ops, staged.target)
    hidden_excluded = {str(row["capsule"]) for row in hidden_withheld}
    if not hidden:
        issues.append("hidden grading cohort is empty; supply an operator-owned private baseline")
    elif len(hidden) == len(hidden_excluded):
        issues.append("hidden grading cohort has no capability-admitted capsules")
    if issues:
        raise SpecError("release admission blocked: " + "; ".join(issues))
    return {
        "capability_exclude_capsules": capability,
        "expected_cohort": {
            "source_capsules": len(public),
            "admitted_capsules": len(public) - len(capability) - len(resource),
        },
        "hidden_capability_admission": {
            "source_capsules": len(hidden),
            "admitted_capsules": len(hidden) - len(hidden_excluded),
        },
        "resource_decisions": len(resource) + len(required),
    }


def admission(descriptor: Path, *, coverage_output: Path | None = None) -> dict:
    """Reuse native discovery/admission without launching an oracle or changing policy."""
    from merlin.target.registry.descriptor_loading import load_target_experiment, observed_experiment_contract

    te = load_target_experiment(descriptor)
    with observed_experiment_contract(te):
        return _admission(te, coverage_output=coverage_output)


def _admission(te, *, coverage_output: Path | None) -> dict:
    import hashlib
    import json
    from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
    from merlin.evaluation.capsule import eligibility as capsule_eligibility
    from merlin.experiments.corpus.source_staging import private_json
    from merlin.experiments.definition import SpecError

    from merlin.experiments.corpus.public_scalability import selected as selected_scalability

    scalability = selected_scalability(te)  # Separate from the core RTL denominator.
    from merlin.experiments.corpus.public_program_suite import selected as selected_programs

    programs = selected_programs(te)  # Complete programs add obligations, never core RTL credit.
    from merlin.development.targetgen.corpus.materialization import (
        _TIER_ORDER,
        materialize_public_cohort,
        pin_cohort_builds,
        resolve_published_cohort,
        validate_materialized_cohort,
    )
    from merlin.development.testgen.requirement_inputs import require_direct_tier, selected_floor
    from merlin.experiments.corpus.publication import require_accelerator_placement

    # Re-checked here, not only at assembly: sealing re-runs this admission over the reviewed payload.
    require_accelerator_placement(te.graded_roots())
    source_public = capsule_capsule_common.discover_capsules(te.graded_roots(), labels={"public", "dev"})
    source_hidden = None
    if (te.workload_spec or {}).get("certification_floor") is not None:
        source_hidden = capsule_capsule_common.discover_capsules(te.hidden_roots(), labels={"hidden"})
        declared_tiers = sorted(
            {tier for cap in [*source_public, *source_hidden] for tier in (cap.get("required_oracle_tiers") or [])}
        )
        try:
            floor = selected_floor(te.workload_spec, declared_tiers)
            require_direct_tier([*source_public, *source_hidden], floor=floor)
        except ValueError as exc:
            raise SpecError(f"release certification-floor preflight failed: {exc}") from exc
    # Read the immutable build behind the published per-target link: coverage observation refuses
    # to traverse links, and the link is this module's own swap point, not a corpus input.
    materialized = resolve_published_cohort(materialize_public_cohort(te, tier_ceiling=_TIER_ORDER[-1]))
    pin_cohort_builds(materialized)
    public = validate_materialized_cohort(materialized, te)
    from merlin.development.testgen.evidence.coverage_commitment import (
        admitted_source_roots,
        build_phase0_readiness,
        observe_cohort,
        phase0_readiness_identity,
    )
    from merlin.development.testgen.evidence.coverage_obligations_coverage_commitment import (
        read_inputs,
        requires_workload_coverage,
    )

    coverage_inputs = read_inputs(te.capsule_corpus.parent)
    from merlin.experiments.corpus.coverage import has_accelerator_work, selected_scope

    completeness = observe_cohort(
        coverage_inputs,
        admitted_source_roots(te.graded_roots(), materialized),
        target=te.target,
    )
    readiness = build_phase0_readiness(completeness)
    workload_required = requires_workload_coverage(te, coverage_inputs)
    if coverage_output is not None:
        private_json(coverage_output, completeness)
    hidden = (
        source_hidden
        if source_hidden is not None
        else capsule_capsule_common.discover_capsules(te.hidden_roots(), labels={"hidden"})
    )
    if selected_scope(coverage_inputs) is not None:
        for capsule in [*source_public, *hidden]:
            if not has_accelerator_work(capsule):
                raise SpecError("accelerator-only release contains a host-only capsule")
    _, withheld = capsule_eligibility._split_ineligible(
        [cap for cap in hidden if cap.get("kind") != "model"], te.target
    )
    excluded = {row["capsule"] for row in withheld}
    count = len(hidden)
    admitted = sum(cap["name"] not in excluded for cap in hidden)
    if public["n_admitted_capsules"] <= 0 or count <= 0 or admitted <= 0:
        raise SpecError("reviewable corpus must have nonempty public and hidden grading cohorts")
    if (te.hidden_expected_source_capsules is not None and count != te.hidden_expected_source_capsules) or (
        te.hidden_expected_admitted_capsules is not None and admitted != te.hidden_expected_admitted_capsules
    ):
        raise SpecError("hidden corpus differs from descriptor-frozen admission counts")
    return {
        **(
            {
                "public_scalability": {
                    "required": True,
                    "source_cases": len(scalability[1]["cases"]),
                    "suite_sha256": (te.workload_spec or {})["public_scalability_suite"]["sha256"],
                    "engine": "spike",
                    "tiers": ["L0", "L1", "L2"],
                    "status": "admitted_not_executed",
                }
            }
            if scalability is not None
            else {}
        ),
        **(
            {
                "public_programs": {
                    "required": True,
                    "source_cases": len(programs[2]),
                    "suite_sha256": (te.workload_spec or {})["public_program_suite"]["sha256"],
                    "engine": "spike",
                    "tiers": ["L0", "L1", "L2"],
                    "status": "admitted_not_executed",
                }
            }
            if programs is not None
            else {}
        ),
        "public_source": public["n_source_capsules"],
        "public_admitted": public["n_admitted_capsules"],
        "hidden_source": count,
        "hidden_admitted": admitted,
        "public_commitment": public["admitted_name_set_sha256"],
        "scope": "native cohort admission only; numerical and hardware readiness not executed",
        "phase0_readiness": phase0_readiness_identity(readiness, required=workload_required),
        "whole_workload_phase1": {
            "status": completeness["status"],
            "cohort_sha256": completeness["cohort"]["sha256"],
            "inputs_sha256": completeness["inputs_sha256"],
            "n_blockers": len(completeness["blockers"]),
            "required": workload_required,
            "report_sha256": hashlib.sha256(
                json.dumps(completeness, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    }
