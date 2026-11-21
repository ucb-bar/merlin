"""Keep generated corpus destinations separate from retained source inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from contextlib import closing
from pathlib import Path

import yaml

from merlin.development.targetgen.corpus import binding as CS  # noqa: E402
from merlin.development.testgen.evidence.program_admission import (
    accelerator_placement_refusal,
    entry_refusal_is_final,
    screen_written,
)
from merlin.development.testgen.generation.capsule_writer import (
    SYNTH_ROLE,
    UnprovableForbid,
    _finish_capsule,
    _write_capsule,
    _write_capsule_source,
)
from merlin.development.testgen.generation.claim_boundary import assert_no_claim_capsules, held_out_models
from merlin.development.testgen.generation.provenance import (
    _capture_failure_reason,
    _scrub_capsule_dir,
    update_provenance_manifest,
)
from merlin.development.testgen.generation.staging import PreparedWrite, prepare_staged, write_staged
from merlin.development.testgen.performance.scope import enforcement_problems, validate_roles
from merlin.development.testgen.profiles.loading import load_profile, validate_profile_inputs
from merlin.development.testgen.profiles.screens import diagnostic_entry, screen_entry
from merlin.development.testgen.profiles.sweep_axes import _performance_facts, _resolve_flat_extents
from merlin.development.testgen.profiles.sweep_expansion import expand_sweeps
from merlin.development.testgen.sealed_capture.generation import verified_capture_failure
from merlin.development.testgen.sealed_capture.selection import (
    _prepare_model_capture_entry,
    _with_selected_model_recipe,
)
from merlin.target.registry.descriptor_loading import load_target_experiment  # noqa: E402

# ---- generation_paths --------------------------------------------------------------------------


def _require_distinct_corpus_destinations(te, *, output_root: str | Path, evidence_root: str | Path | None) -> None:
    """Keep generated Phase 0 bytes out of retained and descriptor-selected input corpora.

    The legacy corpus is still addressable by frozen benchmark grants. Rewriting it
    under a new derivation would silently change those experiments, even when the
    caller explicitly supplied ``--output-root``. Resolve paths before comparing
    them so an alias cannot bypass this source-ownership check.
    """
    from merlin.common.paths import checkout_root
    from merlin.target.registry.descriptor_loading import capsule_corpus_roots

    # In source mode MERLIN_REPO_ROOT may select an external experiment
    # workspace that has no copy of this checkout's historical registry. The
    # implementation checkout still owns those retained paths; an installed
    # wheel instead reads its bundled registry. Both modes keep malformed or
    # missing selected metadata fail-closed.
    sources = [*capsule_corpus_roots(owner_root=checkout_root())]
    selected = getattr(te, "capsule_corpus", None)
    if selected:
        sources.append(Path(selected))
        # Minimal independent hardware descriptors select no input corpus.
        # Their legacy sibling methods require a primary corpus and must not
        # discover or dereference a nonexistent historical selection.
        for method_name in ("graded_roots", "perf_roots", "hidden_roots"):
            method = getattr(te, method_name, None)
            if callable(method):
                sources.extend(method())
    sources = sorted({Path(source).expanduser().resolve() for source in sources})
    # A frozen run executes from its own sealed source snapshot below the evidence root, and that
    # snapshot deliberately EXCLUDES the capsule corpus. A corpus path resolved inside it names
    # nothing: it is the run's own read-only copy of the checkout, not a retained input to protect.
    frozen = Path(evidence_root).expanduser().resolve() / "private" / "source" if evidence_root is not None else None
    if frozen is not None:
        sources = [
            source for source in sources if not ((source == frozen or frozen in source.parents) and not source.exists())
        ]
    for field, raw in (("output_root", output_root), ("evidence_root", evidence_root)):
        if raw is None:
            continue
        destination = Path(raw).expanduser().resolve()
        for source in sources:
            if destination == source or destination in source.parents or source in destination.parents:
                raise ValueError(
                    f"Phase 0 {field} {destination} overlaps source capsule corpus {source}; "
                    "select a separate run artifact destination"
                )


# ---- generation --------------------------------------------------------------------------------
# Host-owned Phase 0 generation implementation.


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


def _form_perf_coverage(profile: dict, requirement: dict | None, out_root: Path, members: list) -> dict:
    """The perf-coverage section of the Phase 2 report: each derived form class above the shared
    template's declared share needs a form-perf member and a vendor bar in the emitted cohort."""
    from merlin.development.testgen.performance.form_cohort import form_perf_coverage

    declared = [
        sweep for sweep in profile.get("sweeps") or [] if isinstance(sweep, dict) and sweep.get("requires_form_scope")
    ]
    threshold = (declared[0]["requires_form_scope"] or {}).get("min_predicted_cycle_share") if declared else None
    capsules = []
    for member in members:
        path = out_root / member / "capsule.yaml"
        if path.is_file():
            capsules.append(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return form_perf_coverage(requirement, capsules, threshold=threshold)


def require_enforceable_policy(evidence_mode: str | None, roles: list[str], policy: dict) -> None:
    """Verified Phase 0 refuses to seal a declared prohibition it cannot enforce: an underivable
    taxonomy, or a role that matches none of the target's instructions (a rule that forbids nothing)."""
    if evidence_mode != "verified" or not roles:
        return
    refused = enforcement_problems(policy, roles)
    if refused:
        raise ValueError(
            "verified Phase 0 cannot resolve the declared prohibited instruction roles against this "
            f"target's instruction taxonomy: {'; '.join(refused)}"
        )


def _instruction_policy(target: str, roles: list[str], evidence, rtl_facts) -> dict:
    """The declared prohibition resolved against the target's own derived instruction taxonomy."""
    from merlin.development.testgen.performance.scope import derive_role_taxonomy, resolve_policy

    if not roles:
        return resolve_policy([], {"status": "not_derived"})
    if evidence is None:
        return resolve_policy(roles, derive_role_taxonomy(target))
    from merlin.target.registry.identity import observed_contract
    from merlin.target.rtl.facts import observed_facts

    with (
        observed_contract(target, evidence.contract),
        observed_facts(target, evidence.refreshed_facts, Path(rtl_facts) if rtl_facts is not None else None),
    ):
        taxonomy = derive_role_taxonomy(target)
    policy = resolve_policy(roles, taxonomy)
    policy["taxonomy"] = taxonomy
    return policy


def _with_candidate_policy(entry: dict, policy: dict) -> dict:
    """A form-perf member's candidate arm carries the declared roles by value; the vendor arm is exempt."""
    from merlin.development.testgen.performance.scope import candidate_arm_policy

    arms = (entry.get("performance") or {}).get("arms")
    if not isinstance(arms, dict) or "candidate" not in arms:
        return entry
    out = copy.deepcopy(entry)
    out["performance"]["arms"]["candidate"]["instruction_policy"] = candidate_arm_policy(policy)
    return out


def _with_reference_gate(entry: dict, te, descriptor: Path) -> dict:
    """Attach a model's declared whole-model reference gate result (``workload_spec.reference_gates``)."""
    if entry.get("kind") != "model" and entry.get("op") != "model":
        return entry
    declared = ((getattr(te, "workload_spec", None) or {}).get("reference_gates")) or {}
    if not isinstance(declared, dict):
        raise ValueError("workload_spec.reference_gates must map a model to its gate result")
    path = declared.get(str(entry.get("model") or "")) or declared.get(str(entry.get("name") or ""))
    if path is None:
        return entry
    resolved = Path(str(path))
    if not resolved.is_absolute():
        resolved = Path(descriptor).parent / resolved
    return {**entry, "reference_gate": str(resolved)}


def withhold_unaccelerated(name: str, observed: dict, directory: Path) -> dict | None:
    """Remove a written member whose admitted program places nothing on the accelerator.

    Returns the record that replaces it in the manifest, or ``None`` when the member stays. Only an
    admitted screen is judged here: a refused or unknown one already leaves the graded cohort through
    the diagnostic path, and a member already in that cohort is not public.
    """
    if observed.get("status") != "admitted" or directory.parent.name == "_diagnostic":
        return None
    reason = accelerator_placement_refusal(observed)
    if reason is None:
        return None
    shutil.rmtree(directory)
    return {"capsule": name, "placements": copy.deepcopy(observed.get("placements") or {}), "reason": reason}


def _member_path(entry: dict) -> str | None:
    """``<category>/<name>`` of the member an entry writes, or None when it does not say."""
    category, name = entry.get("cat"), entry.get("name")
    return f"{category}/{name}" if category and name else None


def _qualifying_group_capsules(written: Path, out_root: Path) -> list[Path]:
    """The per-group capsule directories a qualified model capsule names as its evidence."""
    capsule_file = written / "capsule.yaml"
    if not capsule_file.is_file():
        return []
    capsule = yaml.safe_load(capsule_file.read_text(encoding="utf-8")) or {}
    rows = (capsule.get("model_qualification") or {}).get("group_capsules") or []
    out = []
    for row in rows:
        member = out_root / str(row.get("capsule") or "")
        if not (member / "capsule.yaml").is_file():
            raise ValueError(
                f"model {capsule.get('name')!r} names qualifying group capsule {row.get('capsule')!r} that is absent"
            )
        out.append(member)
    return out
