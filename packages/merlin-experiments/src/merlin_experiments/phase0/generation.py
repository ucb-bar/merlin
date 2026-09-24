"""Host-owned Phase 0 generation implementation."""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

import yaml

from merlin.targetgen import corpus_spec as CS  # noqa: E402
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

from .claim_boundary import assert_no_claim_capsules
from .profiles import load_profile, validate_profile_inputs
from .provenance import _capture_failure_reason, _scrub_capsule_dir, update_provenance_manifest
from .sweeps import _performance_facts, _resolve_flat_extents, expand_sweeps
from .writer import SYNTH_ROLE, UnprovableForbid, _write_capsule


def _descriptor_for(target: str) -> Path:
    from merlin.common.paths import repo_root

    return repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target / "target_experiment.yaml"


def _ensure_contract_on_path(descriptor: Path) -> None:
    """If the descriptor names an out-of-tree ``target_contract`` (e.g. radiance's contract lives under
    the ``radiance`` target package), prepend its package root to ``MERLIN_TARGET_PATH`` so the registry
    resolves the manifest. Read from the descriptor, so it stays target-agnostic."""
    from merlin.common.paths import repo_root

    raw = yaml.safe_load(descriptor.read_text())
    tc = (raw.get("hardware_spec") or {}).get("target_contract")
    if not tc:
        return
    pkg = (repo_root() / tc).resolve().parent.parent  # .../contracts/target_contract.yaml -> package root
    cur = os.environ.get("MERLIN_TARGET_PATH", "")
    if str(pkg) not in cur.split(os.pathsep):
        os.environ["MERLIN_TARGET_PATH"] = os.pathsep.join([str(pkg), cur]) if cur else str(pkg)


def generate_target(
    target: str,
    *,
    descriptor: str | Path | None = None,
    output_root: str | Path | None = None,
    profiles_root: str | Path | None = None,
    recipe: str | Path | None = None,
    performance_template: str | Path | None = None,
    conformance_spec: str | Path | None = None,
    synth_profile: str | Path | None = None,
    smt_profile: str | Path | None = None,
    hidden_profile: str | Path | None = None,
) -> list[Path]:
    from merlin.common.paths import checkout_root

    if output_root is None:
        raise ValueError("Phase 0 requires an explicit output_root; generated capsules must not default to source data")
    profile_inputs = dict(
        profiles_root=profiles_root,
        recipe=recipe,
        performance_template=performance_template,
        conformance_spec=conformance_spec,
        synth_profile=synth_profile,
        smt_profile=smt_profile,
        hidden_profile=hidden_profile,
    )
    validate_profile_inputs(**profile_inputs)
    if profiles_root is None and recipe is None:
        raise ValueError("Phase 0 requires explicit recipe inputs or profiles_root before descriptor setup")
    if checkout_root() is None and (descriptor is None or (profiles_root is None and recipe is None)):
        raise ValueError("installed Phase 0 requires explicit profiles_root or recipe, descriptor and output_root")
    # The profile selector and the hardware descriptor are separate identities:
    # a profile can be reused with an explicitly supplied out-of-tree target.
    explicit_descriptor = descriptor is not None
    descriptor = Path(descriptor).expanduser().resolve() if descriptor is not None else _descriptor_for(target)
    _ensure_contract_on_path(descriptor)
    te = load_target_experiment(descriptor)
    hardware_target = te.target if explicit_descriptor else target
    profile = load_profile(
        target, descriptor=descriptor, **{key: value for key, value in profile_inputs.items() if value is not None}
    )
    declared_claims = [str(model) for model in (getattr(te, "workload_spec", None) or {}).get("models") or ()]
    claim_plan = profile.get("_claim_model_evaluation")
    if profile.get("_synth_verification", {}).get("status") == "verified":
        if (
            not isinstance(claim_plan, dict)
            or claim_plan.get("schema") != "claim_model_evaluation_v1"
            or claim_plan.get("model_count") != len(declared_claims)
            or claim_plan.get("visibility") != "owner_only_after_phase1_freeze"
            or claim_plan.get("public_capsules_emitted") != 0
        ):
            raise ValueError("verified synthesis lacks an owner-only claim-model evaluation obligation")
    if profile.get("_synth_verification", {"status": "absent"})["status"] == "unverified_legacy":
        raise ValueError(
            "selected synthesized profile has no digest-bound conformance/recipe/workload inputs; "
            "regenerate, review, and select a new sidecar before verified Phase 0 execution"
        )
    binding = CS.derive_binding(te, profile.get("datapath", {}))
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # `sweeps:` (if any) expand into the same flat entries `capsules:` holds, so
    # everything downstream — builders, goldens, coverage — is unchanged.
    facts = _performance_facts(hardware_target)
    _sweep_skips: list = []
    _runtime_blocked: list = []
    _performance_errors: list = []
    entries = expand_sweeps(
        profile,
        binding,
        trait_facts=facts,
        skipped=_sweep_skips,
        blocked_unimplemented=_runtime_blocked,
        errors=_performance_errors,
    )
    assert_no_claim_capsules(entries, declared_claims)
    entries = [_resolve_flat_extents(e, binding) for e in entries]
    for _s in _sweep_skips:
        _why = _s.get("reason") or f"gate {(_s.get('gate') or {}).get('outcome')}"
        print(f"  [skip] performance family {_s['family']}: {_why}")
    template = copy.deepcopy(profile.get("_performance_template") or {})
    declared_families = [dict(row) for row in (template.get("families") or [])]
    family_counts = {row["family"]: {"admitted_members": 0, "written_members": 0} for row in declared_families}
    for entry in entries:
        family = (entry.get("performance") or {}).get("family")
        if family:
            family_counts.setdefault(family, {"admitted_members": 0, "written_members": 0})
            family_counts[family]["admitted_members"] += 1
    # SCRUB EACH CAPSULE AS IT IS WRITTEN, not after the whole corpus succeeds. Scrubbing at the end
    # means one unrelated failure -- a capture that needs an external exporter, say -- aborts the run
    # with every capsule written so far still carrying its absolute `prov.weights_file` path. Measured:
    # a run that died on the last entry left `/scratch/.../capsule_m2m_<rand>/weights.safetensors` in
    # tracked MLIR across six capsules, in a repo that is published. Hygiene that only holds on the
    # happy path is not hygiene.
    #
    # ONE FAILING CAPSULE MUST NOT DESTROY THE WHOLE CORPUS. Letting the exception propagate meant a
    # single entry that needs an external exporter took every entry after it down with it: measured, a
    # capture that torch.export refuses (an LSTM whose `_flat_weights` are assigned rather than
    # registered) aborted the run before any of the tail-path sweep capsules were written, so a coverage
    # gap stayed open for a reason that had nothing to do with it. Failures are COLLECTED, reported by
    # name, and re-raised at the end -- the run still fails, it just fails after doing the work it could.
    written, failures, unbuilt_roster, unprovable_forbids = [], [], [], []
    for e in entries:
        family = (e.get("performance") or {}).get("family")
        try:
            w = _write_capsule(e, binding, out_root, facts.get("sha256", ""))
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed
            detail = f"{type(exc).__name__}: {str(exc)[:300]}"
            if isinstance(exc, UnprovableForbid) and str(e.get("source_role") or "") == SYNTH_ROLE:
                # See `UnprovableForbid`. The capsule is removed rather than left on disk: a directory
                # the corpus does not list is exactly the kind of half-written state the seal cannot
                # see, and a stale one would be picked up by the next glob as though it had been built.
                shutil.rmtree(out_root / str(e.get("cat") or "") / str(e.get("name") or ""), ignore_errors=True)
                unprovable_forbids.append(
                    {"capsule": e.get("name", "?"), "family": e.get("op"), "reason": str(exc)[:400]}
                )
                print(f"  [lane] {e.get('name')}: NOT BUILT — its forbid is not provable on this target")
                continue
            if _is_roster_capsule(e):
                # A ROSTER MODEL IS A DECLARED INPUT, NOT A COMPILER RESULT. The roster axis emits one
                # whole-model capsule per model the target's `workload_spec` names, at the format the
                # target admits -- and whether a given network can be CAPTURED at that format depends on
                # things outside this repo: whether the loader's dataset is present, whether the m2m venv
                # has the model's package, whether torchAO's scheme runs on this host, whether
                # torch.export accepts the module under quantization. Measured on the four declared
                # models: one captures, one needs an ImageNet npz, one needs a package the venv lacks,
                # one is refused by torch.export under W8A8 while its weight-only capture succeeds.
                #
                # Aborting the corpus for those would make a fact about the ROSTER read as a broken
                # generator, and would take every other capsule down with it. Recording it keeps the rule
                # that matters -- a requirement that produced no capsule is never indistinguishable from
                # one that is met -- because the manifest names the model and the reason, and no capsule
                # exists to be graded, so nothing can pass in its place.
                why = _capture_failure_reason(exc)
                unbuilt_roster.append(
                    {
                        "capsule": e.get("name", "?"),
                        "model": e.get("model"),
                        "operand_dtype": e.get("operand_dtype"),
                        "quant_scheme": e.get("quant_scheme"),
                        "status": "not_built",
                        "reason": why,
                    }
                )
                print(f"  [roster] {e.get('name')}: NOT BUILT — {why}")
                continue
            failures.append((e.get("name", "?"), detail))
            if family:
                _performance_errors.append(
                    {
                        "family": family,
                        "member": e.get("name", "?"),
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "detail": str(exc)[:500],
                    }
                )
            continue
        if w:
            _scrub_capsule_dir(w)
            written.append(w)
            if family:
                family_counts[family]["written_members"] += 1
        elif family:
            _performance_errors.append(
                {
                    "family": family,
                    "member": e.get("name", "?"),
                    "status": "error",
                    "error_type": "NoOutput",
                    "detail": "capsule writer returned no output",
                }
            )
    if failures:
        print(f"  [FAIL] {len(failures)} capsule(s) could not be written:")
        for name, why in failures:
            print(f"    - {name}: {why}")
    # Record provenance for what we just emitted. The MANIFEST header has always CLAIMED the generator
    # rewrites it, but no writer existed, so it drifted silently as soon as the corpus grew.
    declared_blocked = [
        {
            "family": row["family"],
            "status": "blocked_unimplemented",
            "reason": row["reason"],
            "emitter": copy.deepcopy(row["performance"]["emitter"]),
            "fit_axes": list(row.get("fit_axes") or []),
            "comparison_roles": list(row.get("comparison_roles") or []),
        }
        for row in (template.get("blocked_unimplemented") or [])
    ]
    generated_members = sum(row["written_members"] for row in family_counts.values())
    performance_record = {
        "shared_template": {"path": template.get("path"), "sha256": template.get("sha256")},
        "facts": {"target": hardware_target, "sha256": facts["sha256"]},
        "phase": {
            "category": "_perf",
            "label": "dev",
            "included_in_functional_grade": False,
            "exclusion": "TargetExperiment.corpus_siblings excludes underscore-prefixed categories",
        },
        "families": declared_families,
        "counts": {
            "declared_families": len(declared_families),
            "generated_families": sum(1 for row in family_counts.values() if row["written_members"] > 0),
            "generated_members": generated_members,
            "by_family": family_counts,
        },
        "skipped_inapplicable": _sweep_skips,
        "blocked_unimplemented": declared_blocked + _runtime_blocked,
        "errors": _performance_errors,
    }
    if unbuilt_roster:
        print(
            f"  [roster] {len(unbuilt_roster)} declared roster model(s) could not be captured at this "
            f"target's derived format; they are recorded as not_built, never as covered"
        )
    if unprovable_forbids:
        print(
            f"  [lane] {len(unprovable_forbids)} synthesized host-lane capsule(s) were dropped: their "
            f"program contains regions this target admits, so the forbid they assert is not provable"
        )
    superseded = _prune_superseded_synth(entries, written, target=target)
    if superseded:
        print(
            f"  [prune] {len(superseded)} synthesized capsule(s) whose requirement cell no longer "
            f"exists were removed: {', '.join(superseded)}"
        )
    update_provenance_manifest(
        written,
        cap_root=out_root,
        target=hardware_target,
        performance_record=performance_record,
        unbuilt_roster=unbuilt_roster,
        claim_model_evaluation=claim_plan,
        unprovable_forbids=unprovable_forbids,
        superseded=superseded,
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} capsule(s) failed to generate: {', '.join(n for n, _ in failures)}; the "
            f"rest of the corpus was written, so re-running after fixing them is cheap"
        )
    return written


def _prune_superseded_synth(entries, written, *, target: str) -> list:
    """Remove synthesized capsule directories the profile no longer asks for, and name them.

    THE PROFILE IS THE WHOLE AUTHORITY for `derived_sweep` capsules -- it is regenerated from the
    requirement, so a directory of that role which is not in it is evidence for a cell that is no
    longer required. Nothing removed one, and they do not merely sit there: a superseded
    `SY_kdepth_certified` inflated a sealed cohort count by one and broke the MANIFEST, and twelve
    superseded MX capsules survived the alignment classes their own datapath cannot express, each one
    still offering itself to the cover as evidence for a cell the requirement had dropped.

    Deliberately narrow: only directories whose capsule declares this generator's synthesized role, and
    only under the roots this target just wrote to. A hand-authored capsule is never touched, and a
    target whose profile carries no synthesized entries at all prunes nothing (a profile that failed to
    synthesize must not be read as "every cell was dropped").
    """
    keep = {str(e.get("name")) for e in entries if str(e.get("source_role") or "") == SYNTH_ROLE}
    if not keep:
        return []
    roots = {d.parent for d in written if isinstance(d, Path)}
    removed = []
    for root in sorted(roots):
        for cy in sorted(root.glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if str(cap.get("source_role") or "") != SYNTH_ROLE or name in keep:
                continue
            shutil.rmtree(cy.parent, ignore_errors=True)
            removed.append(name)
    return sorted(removed)


def _is_roster_capsule(entry: dict) -> bool:
    """A whole-model capsule the ROSTER axis emitted, as opposed to any other model capsule.

    Read off the entry's own generalization axis rather than its name, so a renamed prefix does not
    silently reclassify a capsule into the lane that is allowed not to build.
    """
    return (
        str(entry.get("kind")) == "model"
        and str((entry.get("generalization") or {}).get("generalization_axis") or "") == "roster"
    )
