"""Load a declarative per-target experiment descriptor — the target-parameterized replacement for the
former single-target hardcoded experiment setup.

Per the derive-first rule, the hardware FACTS (ISA/opcode set, memory map, mesh DIM, arc model) are
DERIVED from the RTL by mlc (``rtl_backend.target_profile`` / ``mlc_bridge``), never hand-written. What a
run genuinely cannot derive — which RTL repo, which hardware-spec files every arm gets, which capsule
corpus to grade on, how the simulator runs — is the irreducible SETUP, declared in a small YAML
descriptor (``examples/<target>/target/descriptor.yaml`` in the catalog). A new accelerator supplies its own descriptor
and registers its RTL with mlc; no per-target code.
"""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import checkout_root, data_path, repo_root, resolve_grant
from merlin.target.capabilities.manifest import (
    CapabilityManifest,
    _derived_dtype_token,
    _primary_kind,
    load_capability_manifest,
)

# ---- target_experiment -------------------------------------------------------------------------


def _declared_target(descriptor: Path) -> str | None:
    """The ``target`` a descriptor declares, or None when it is unreadable/not a descriptor.

    Deliberately a cheap YAML read rather than :func:`load_target_experiment`: this is used to SCAN, and
    a validation error in one target's descriptor must not hide another target's descriptor.
    """
    try:
        doc = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    name = doc.get("target") if isinstance(doc, dict) else None
    return str(name) if name else None


def descriptor_for(target: str) -> Path | None:
    """The experiment descriptor that DECLARES ``target``, or None when the target ships none.

    Resolution: ``$MERLIN_TARGET_EXPERIMENT`` when it names THIS target (so a run pointed at one
    descriptor is never served another), then authored ``examples/*/target/descriptor.yaml`` inputs,
    then the legacy ``targets/<target>/`` convention and a scan of every discoverable legacy descriptor
    for one whose ``target:`` matches — because A DIRECTORY NAME IS NOT ALWAYS
    THE TARGET NAME (see :func:`~merlin.target.registry.identity.declared_target_for`).

    Returns None rather than raising: a caller that cannot find a descriptor must degrade to the naming
    convention, not fail.
    """
    from merlin.common.paths import repo_root
    from merlin.target.registry.experiment_descriptor import _DESCRIPTOR_FILE, _DESCRIPTOR_GLOBS

    env = os.environ.get("MERLIN_TARGET_EXPERIMENT")
    if env:
        p = Path(env)
        # `is_file()` is the load-bearing half: an agent sandbox inherits this variable from the launcher
        # while the path it names is masked, and a descriptor that cannot be READ must fall through to the
        # conventions below rather than be treated as found.
        if p.is_file() and _declared_target(p) == target:
            return p
    for candidate in sorted((repo_root() / "examples").glob("*/*/target/descriptor.yaml")):
        if candidate.is_file() and _declared_target(candidate) == target:
            return candidate
    root = repo_root()
    for pattern in _DESCRIPTOR_GLOBS:
        for cand in sorted(root.glob(pattern.format(t=target, f=_DESCRIPTOR_FILE))):
            if cand.is_file() and _declared_target(cand) == target:
                return cand
    for pattern in _DESCRIPTOR_GLOBS:
        for cand in sorted(root.glob(pattern.format(t="*", f=_DESCRIPTOR_FILE))):
            if cand.is_file() and _declared_target(cand) == target:
                return cand
    return None


def load_target_experiment(descriptor: str | Path, *, source_root: Path | None = None) -> TargetExperiment:
    """Load + validate a target descriptor. Shared-spec paths are kept as the bundle-
    convention STRINGS (so the governance check compares like-for-like); the capsule corpus is resolved.

    Explicit source_root owns repository-relative descriptor/resource paths and is retained by
    path-resolving methods. Omission preserves legacy checkout discovery and descriptor spelling.
    """
    from merlin.target.registry.experiment_descriptor import (
        HostLaneMatrix,
        PreflightCapabilityProbe,
        TargetExperiment,
        _safe_relative,
        declared_contracts_root,
        declared_resources_root,
        declared_task_root,
    )

    if source_root is not None:
        source_root = Path(source_root).absolute()
    p = Path(descriptor)
    if source_root is not None and not p.is_absolute():
        p = source_root / p
    descriptor_bytes = p.read_bytes()
    doc = yaml.safe_load(descriptor_bytes.decode("utf-8"))
    if not isinstance(doc, dict) or not doc.get("target"):
        raise ValueError(f"{p}: not a target-experiment descriptor (missing 'target')")
    numeric_profile = doc.get("numeric_profile")
    if numeric_profile is not None and (
        not isinstance(numeric_profile, str)
        or not numeric_profile.strip()
        or not Path(numeric_profile).name
        or ".." in Path(numeric_profile).parts
    ):
        raise ValueError(f"{p}: numeric_profile must name a file without parent traversal")
    root = source_root if source_root is not None else repo_root()
    hw = doc.get("hardware_spec") or {}
    grading = doc.get("grading") or {}
    if not isinstance(grading, dict):
        raise ValueError(f"{p}: grading must be a mapping")
    resource_bound = grading.get("resource_bound") or {}
    release_admission = grading.get("release_admission")
    if release_admission not in (None, "derive_from_corpus_v1"):
        raise ValueError(f"{p}: grading.release_admission must be derive_from_corpus_v1")
    phase_bound = grading.get("phase_bound") or {}
    expected_cohort = grading.get("expected_cohort") or {}
    hidden_admission = grading.get("hidden_capability_admission") or {}
    search_cohort = grading.get("search_cohort") or {}
    raw_evaluation_cohorts = grading.get("evaluation_cohorts") or {}
    for field, value in (
        ("grading.resource_bound", resource_bound),
        ("grading.phase_bound", phase_bound),
        ("grading.expected_cohort", expected_cohort),
        ("grading.hidden_capability_admission", hidden_admission),
        ("grading.search_cohort", search_cohort),
        ("grading.evaluation_cohorts", raw_evaluation_cohorts),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{p}: {field} must be a mapping")

    def names(value, *, field: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"{p}: {field} must be a list of non-empty capsule names")
        out = tuple(value)
        if len(set(out)) != len(out):
            raise ValueError(f"{p}: {field} contains duplicate capsule names")
        return out

    def count(mapping: dict, key: str, *, field: str) -> int | None:
        if key not in mapping:
            return None
        value = mapping[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{p}: {field}.{key} must be a non-negative integer")
        return value

    evaluation_cohorts: dict[str, dict[str, Any]] = {}
    for stage, raw_stage in raw_evaluation_cohorts.items():
        field = f"grading.evaluation_cohorts.{stage}"
        if not isinstance(stage, str) or not stage or Path(stage).name != stage:
            raise ValueError(f"{p}: evaluation-cohort name {stage!r} must be one safe name")
        if not isinstance(raw_stage, dict):
            raise ValueError(f"{p}: {field} must be a mapping")
        include = names(raw_stage.get("include_capsules"), field=f"{field}.include_capsules")
        if not include:
            raise ValueError(f"{p}: {field}.include_capsules must not be empty")
        policy = raw_stage.get("policy")
        predecessor = raw_stage.get("after")
        tier = raw_stage.get("oracle_tier")
        engine = raw_stage.get("oracle_engine")
        for key, value in (("policy", policy), ("after", predecessor), ("oracle_engine", engine)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{p}: {field}.{key} must be a non-empty string")
        if tier not in {"L0", "L1", "L2", "L3", "L4", "L5"}:
            raise ValueError(f"{p}: {field}.oracle_tier must be one of L0..L5")
        roles = names(raw_stage.get("require_source_roles"), field=f"{field}.require_source_roles")
        evaluation_cohorts[stage] = {
            **copy.deepcopy(raw_stage),
            "policy": policy.strip(),
            "after": predecessor.strip(),
            "oracle_tier": tier,
            "oracle_engine": engine.strip(),
            "include_capsules": include,
            "require_source_roles": roles,
        }

    raw_information_sets = doc.get("information_sets") or {}
    if not isinstance(raw_information_sets, dict):
        raise ValueError(f"{p}: information_sets must be a mapping keyed by bundle variant")
    information_sets: dict[str, dict[str, Any]] = {}
    for variant, body in raw_information_sets.items():
        field = f"information_sets.{variant}"
        if (
            not isinstance(variant, str)
            or Path(variant).name != variant
            or not (variant.startswith("hwbringup_") or variant in {"public_v0", "realistic_v0"})
        ):
            raise ValueError(
                f"{p}: information-set key {variant!r} must be a supported public, realistic, or hwbringup_* variant"
            )
        if not isinstance(body, dict):
            raise ValueError(f"{p}: {field} must be a mapping")
        condition = body.get("condition")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"{p}: {field}.condition must be a non-empty string")
        normalized: dict[str, Any] = {"condition": condition.strip()}
        for key in ("allowed", "denied"):
            rows = body.get(key) or []
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError(f"{p}: {field}.{key} must be a list of mappings")
            clean = []
            for index, row in enumerate(rows):
                value = row.get("path")
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{p}: {field}.{key}[{index}].path must be non-empty")
                _safe_relative(value, field=f"{field}.{key}[{index}].path")
                clean.append(copy.deepcopy(row))
            normalized[key] = clean
        pins = body.get("source_pins") or []
        if (
            not isinstance(pins, list)
            or any(not isinstance(pin, str) or not pin for pin in pins)
            or len(set(pins)) != len(pins)
        ):
            raise ValueError(f"{p}: {field}.source_pins must be a list of unique pin names")
        normalized["source_pins"] = list(pins)
        information_sets[variant] = normalized

    performance = doc.get("performance") or {}
    if not isinstance(performance, dict):
        raise ValueError(f"{p}: performance must be a mapping")
    raw_global_objective = performance.get("global_objective_capsule")
    if raw_global_objective is not None:
        if (
            not isinstance(raw_global_objective, str)
            or not raw_global_objective
            or Path(raw_global_objective).name != raw_global_objective
            or raw_global_objective in (".", "..")
        ):
            raise ValueError(f"{p}: performance.global_objective_capsule must be one capsule directory name")
        performance_global_objective = raw_global_objective
    else:
        performance_global_objective = None

    preflight = doc.get("preflight") or {}
    if not isinstance(preflight, dict):
        raise ValueError(f"{p}: preflight must be a mapping")
    codegen_backend = preflight.get("codegen_backend")
    if "codegen_backend" in preflight and (not isinstance(codegen_backend, str) or not codegen_backend.strip()):
        raise ValueError(f"{p}: preflight.codegen_backend must be a non-empty string")
    raw_capability_probes = preflight.get("capability_probes") or []
    if not isinstance(raw_capability_probes, list):
        raise ValueError(f"{p}: preflight.capability_probes must be a list")
    capability_probes = tuple(
        PreflightCapabilityProbe.from_mapping(value, descriptor=p, index=index)
        for index, value in enumerate(raw_capability_probes)
    )
    capability_names = [probe.capability for probe in capability_probes]
    if len(set(capability_names)) != len(capability_names):
        raise ValueError(f"{p}: preflight.capability_probes contains duplicate capability names")

    legacy_exclude = names(grading.get("exclude_capsules"), field="grading.exclude_capsules")
    search_include = names(search_cohort.get("include_capsules"), field="grading.search_cohort.include_capsules")
    search_policy = search_cohort.get("policy")
    if search_include and (not isinstance(search_policy, str) or not search_policy.strip()):
        raise ValueError(f"{p}: grading.search_cohort requires a non-empty policy")
    capability_exclude = names(grading.get("capability_exclude_capsules"), field="grading.capability_exclude_capsules")
    resource_exclude = names(resource_bound.get("exclude_capsules"), field="grading.resource_bound.exclude_capsules")
    required_models = names(
        resource_bound.get("required_admitted_models"), field="grading.resource_bound.required_admitted_models"
    )
    phase_exclude = names(phase_bound.get("exclude_capsules"), field="grading.phase_bound.exclude_capsules")
    phase2_only = names(phase_bound.get("phase2_only_capsules"), field="grading.phase_bound.phase2_only_capsules")
    outside = sorted(set(phase_exclude) - set(phase2_only))
    if outside:
        raise ValueError(
            f"{p}: grading.phase_bound.exclude_capsules names {outside}, which the recorded phase-2-only "
            "set does not contain; a row may not be dropped for failing a verdict it did not fail"
        )
    split_exclude = capability_exclude + resource_exclude + phase_exclude
    if search_include and (legacy_exclude or split_exclude):
        raise ValueError(f"{p}: grading.search_cohort cannot be combined with exclusion policies")
    if legacy_exclude and split_exclude:
        raise ValueError(
            f"{p}: grading may use legacy exclude_capsules or the explicit capability/resource split, not both"
        )
    # Each row leaves the denominator for exactly ONE reason. An overlap is not a harmless duplicate: it
    # would let a row be reported under whichever heading reads best, and the arithmetic below (source ==
    # admitted + the three lists) would double-count it.
    for a_name, a, b_name, b in (
        ("capability", capability_exclude, "resource", resource_exclude),
        ("capability", capability_exclude, "phase", phase_exclude),
        ("resource", resource_exclude, "phase", phase_exclude),
    ):
        overlap = sorted(set(a) & set(b))
        if overlap:
            raise ValueError(f"{p}: {a_name} and {b_name} exclusions overlap: {overlap}")
    if resource_exclude and (not resource_bound.get("policy") or not required_models):
        raise ValueError(f"{p}: resource exclusions require a named policy and admitted model capstones")
    phase_number = phase_bound.get("phase")
    phase_budget = phase_bound.get("budget_s")
    if phase_bound:
        if not isinstance(phase_number, int) or isinstance(phase_number, bool) or phase_number < 1:
            raise ValueError(f"{p}: grading.phase_bound.phase must name the phase this run serves")
        if not phase_bound.get("policy"):
            raise ValueError(f"{p}: phase exclusions require a named policy")
        if not isinstance(phase_budget, (int, float)) or isinstance(phase_budget, bool) or phase_budget <= 0:
            raise ValueError(
                f"{p}: grading.phase_bound.budget_s must state the certification budget the phase "
                "verdict was derived at -- the verdict is meaningless without it, because a member "
                "priced out at one budget is admitted at another"
            )
    if set(required_models) & set(split_exclude):
        raise ValueError(f"{p}: a required admitted model is also excluded")

    raw_oracle_resources = doc.get("oracle_resources") or {}
    if not isinstance(raw_oracle_resources, dict):
        raise ValueError(f"{p}: oracle_resources must be a mapping keyed by oracle tier")
    oracle_resources: dict[str, dict[str, Any]] = {}
    for tier, body in raw_oracle_resources.items():
        field = f"oracle_resources.{tier}"
        if tier not in {"L0", "L1", "L2", "L3", "L4", "L5"}:
            raise ValueError(f"{p}: {field} must name an oracle tier L0..L5")
        if not isinstance(body, dict):
            raise ValueError(f"{p}: {field} must be a mapping")
        resource = body.get("exclusive_resource")
        if not isinstance(resource, str) or not resource.strip():
            raise ValueError(f"{p}: {field}.exclusive_resource must name the contended resource")
        if not isinstance(body.get("batched"), bool):
            raise ValueError(
                f"{p}: {field}.batched must say explicitly whether this tier is measured N "
                "windows at a time; a tier whose shape is unstated cannot be scheduled"
            )
        limit = body.get("queue_wall_limit_s")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit <= 0):
            raise ValueError(f"{p}: {field}.queue_wall_limit_s must be a positive number of seconds")
        oracle_resources[tier] = copy.deepcopy(body)

    expected_source = count(expected_cohort, "source_capsules", field="grading.expected_cohort")
    expected_admitted = count(expected_cohort, "admitted_capsules", field="grading.expected_cohort")
    if (expected_source is None) != (expected_admitted is None):
        raise ValueError(f"{p}: grading.expected_cohort must declare both source and admitted counts")
    if expected_source is not None and search_include and expected_admitted != len(search_include):
        raise ValueError(f"{p}: grading.expected_cohort admitted count must equal search_cohort size")
    if expected_source is not None and not search_include and expected_source != expected_admitted + len(split_exclude):
        raise ValueError(f"{p}: grading.expected_cohort arithmetic does not match the declared exclusions")
    hidden_source = count(hidden_admission, "source_capsules", field="grading.hidden_capability_admission")
    hidden_admitted = count(hidden_admission, "admitted_capsules", field="grading.hidden_capability_admission")
    if (hidden_source is None) != (hidden_admitted is None):
        raise ValueError(f"{p}: grading.hidden_capability_admission must declare both source and admitted counts")
    if hidden_source is not None and hidden_admitted > hidden_source:
        raise ValueError(f"{p}: hidden admitted count exceeds its sealed source count")
    if release_admission:
        if expected_cohort or hidden_admission or capability_exclude or legacy_exclude or search_include or phase_bound:
            raise ValueError(
                f"{p}: release admission derives cohort counts and capability exclusions from the staged corpus; "
                "phase partitions require a separate reviewed derivation"
            )
        if not resource_bound.get("policy"):
            raise ValueError(f"{p}: release admission requires a named resource policy")
    return TargetExperiment(
        target=str(doc["target"]),
        isa_headers=tuple(hw.get("isa_headers") or []),
        hwbringup_set=hw.get("hwbringup_set"),
        curated_harness=hw.get("curated_harness"),
        capsule_corpus=root / doc["capsule_corpus"] if doc.get("capsule_corpus") else None,
        sim_via=str((doc.get("toolchain") or {}).get("sim_via", "")),
        rtl_via=str((doc.get("rtl") or {}).get("via", "mlc")),
        rtl_repo=(lambda r: str(r) if r else None)((doc.get("rtl") or {}).get("repo")),
        operand_store=(lambda v: str(v) if v else None)((doc.get("rtl") or {}).get("operand_store")),
        prior_backends=tuple((doc.get("answer_surfaces") or {}).get("prior_backends") or ()),
        path=p,
        descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        source_root=source_root,
        preflight_smoke_program=(lambda s: str(s) if s else None)(preflight.get("smoke_program")),
        preflight_codegen_backend=codegen_backend,
        preflight_capability_probes=capability_probes,
        declared_contract=(lambda s: str(s) if s else None)(hw.get("target_contract")),
        numeric_profile=numeric_profile,
        resources_root=declared_resources_root(doc, root=root),
        task_root=declared_task_root(doc, root=root),
        contracts_root=declared_contracts_root(doc, root=root),
        information_sets=information_sets,
        backend_package_dir=(lambda s: str(s) if s else None)(doc.get("backend_package_dir")),
        # Cohort admission (which capsules are graded, and why one is not) alongside the host-lane
        # MATRIX. `host_lane` is no longer a constructor field: a target declares a lane per dtype and
        # the singular `.host_lane` property returns the default, so every existing caller still reads
        # one lane while a capsule compiled at another dtype resolves its own.
        graded_exclude=legacy_exclude or split_exclude,
        graded_include=search_include,
        graded_cohort_policy=(search_policy.strip() if isinstance(search_policy, str) else None),
        evaluation_cohorts=evaluation_cohorts,
        graded_capability_exclude=capability_exclude,
        graded_resource_exclude=resource_exclude,
        graded_resource_policy=(lambda s: str(s) if s else None)(resource_bound.get("policy")),
        graded_required_models=required_models,
        graded_release_admission=bool(release_admission),
        performance_global_objective=performance_global_objective,
        graded_phase2_only=phase2_only,
        graded_phase_exclude=phase_exclude,
        graded_phase=(
            int(phase_number) if isinstance(phase_number, int) and not isinstance(phase_number, bool) else None
        ),
        graded_phase_budget_s=(
            float(phase_budget)
            if isinstance(phase_budget, (int, float)) and not isinstance(phase_budget, bool)
            else None
        ),
        graded_phase_policy=(lambda s: str(s) if s else None)(phase_bound.get("policy")),
        graded_expected_source_capsules=expected_source,
        graded_expected_admitted_capsules=expected_admitted,
        hidden_expected_source_capsules=hidden_source,
        hidden_expected_admitted_capsules=hidden_admitted,
        host_lanes=HostLaneMatrix.from_mapping(doc.get("host_lane"), descriptor=p),
        host_board=(lambda v: str(v) if v else None)((doc.get("host") or {}).get("board")),
        host_board_catalog=(lambda v: str(v) if v else None)((doc.get("host") or {}).get("catalog")),
        workload_spec=(lambda v: dict(v) if isinstance(v, dict) else None)(doc.get("workload_spec")),
        oracle_resources=oracle_resources or None,
    )


def batched_oracle_tiers(target: str | None) -> frozenset[str]:
    """The tiers ``target``'s own descriptor declares as batched, or ``frozenset()``.

    Deliberately a cheap YAML read of the one key rather than :func:`load_target_experiment`: this
    is consulted on the grading path, where a validation error somewhere else in another target's
    descriptor must not decide whether THIS capsule's tier is batched. A target with no descriptor,
    an unreadable one, or no declaration yields the empty set and every caller behaves exactly as
    it did before the declaration existed.
    """
    if not target:
        return frozenset()
    descriptor = descriptor_for(target)
    if descriptor is None:
        return frozenset()
    try:
        doc = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return frozenset()
    resources = doc.get("oracle_resources") if isinstance(doc, dict) else None
    if not isinstance(resources, dict):
        return frozenset()
    return frozenset(
        str(tier) for tier, body in resources.items() if isinstance(body, dict) and body.get("batched") is True
    )


def selected_experiment_contract(te: TargetExperiment, *, environment: Mapping[str, str]) -> Path | None:
    """Resolve descriptor-owned capability selection, without changing support ownership.

    Explicit conflicting process selections fail rather than silently changing
    the scientific input. Release preparation supplies an empty environment to
    select its newly staged descriptor independently of the caller's live view.
    """
    from merlin.target.registry.experiment_descriptor import TargetExperiment

    declared = te.declared_contract_path()
    if te.declared_contract and declared is None:
        raise ValueError("declared capability contract is absent; prepare a new release")
    explicit = environment.get("MERLIN_TARGET_CONTRACT")
    if declared is not None and explicit and Path(explicit).expanduser().resolve() != declared.resolve():
        raise ValueError("explicit capability contract differs from the experiment's declared selection")
    selected = declared or (Path(explicit).expanduser() if explicit else None)
    if selected is None:
        return None
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("selected capability contract is absent or symlinked")
    document = yaml.safe_load(selected.read_bytes())
    if not isinstance(document, dict) or document.get("name") != te.target:
        raise ValueError("selected capability contract is not a mapping for this target")
    return selected.resolve()


@contextmanager
def observed_experiment_contract(te: TargetExperiment):
    """Bind release-local derivation to its declared bytes, without mutating process env."""
    from merlin.target.registry.experiment_descriptor import TargetExperiment
    from merlin.target.registry.identity import observed_contract

    selected = selected_experiment_contract(te, environment={})
    if selected is None:
        yield
    else:
        with observed_contract(te.target, yaml.safe_load(selected.read_bytes()), source_path=selected):
            yield


def declared_vs_resolved_contract(te: TargetExperiment) -> tuple[Path | None, Path | None, str]:
    """``(declared, resolved, verdict)`` for this target's capability contract.

    Verdicts: ``"agree"`` (same file, or nothing declared and one resolves), ``"declared_only"`` (the
    registry resolves nothing — the declaration is the only contract there is), ``"stale_declaration"``
    (the declared path does not exist but the registry resolves one), ``"mismatch"`` (both exist and are
    DIFFERENT files), or ``"none"`` (no contract at all).

    A mismatch is reported, never resolved here. Picking a winner would silently change what an agent is
    told the hardware is — the two saturn_opu contracts disagree on whether its fp8 datapaths are named
    (``fp8_e4m3``/``fp8_e5m2``) or honestly unnamed (``float8`` + ``unnamed_float_datapaths``), and that is
    the difference between a target that declares two formats and one that admits its RTL does not name
    them. Which is authoritative is a call for whoever owns the contract, so this surfaces it and the
    readiness gate fails on it.
    """
    from merlin.target.registry import identity as target_registry
    from merlin.target.registry.experiment_descriptor import TargetExperiment

    declared = te.declared_contract_path()
    try:
        resolved = target_registry.resolve(te.target).capability_contract_path
        resolved = resolved if resolved and Path(resolved).is_file() else None
    except Exception:  # noqa: BLE001 — an unresolvable target is one of the answers
        resolved = None
    if declared and resolved:
        same = Path(declared).resolve() == Path(resolved).resolve()
        return declared, resolved, "agree" if same else "mismatch"
    if declared:
        return declared, None, "declared_only"
    if resolved:
        return None, resolved, "stale_declaration" if te.declared_contract else "agree"
    return None, None, "none"


def shared_spec_paths(te: TargetExperiment, variant: str | None = None) -> set[str]:
    """The shared hardware-spec path strings the descriptor makes authoritative — the ISA headers + the
    hwbringup set EVERY arm's bundle must grant (a constant input, not assistance)."""
    from merlin.target.registry.experiment_descriptor import TargetExperiment

    paths = set(te.isa_headers)
    if te.declared_contract:
        paths.add(te.declared_contract)
    if te.hwbringup_set:
        paths.add(te.hwbringup_set)
    if variant:
        paths.update(
            entry["path"]
            for entry in te.information_set(variant).get("allowed", ())
            if isinstance(entry, dict) and entry.get("path")
        )
    return paths


def bundles_match_descriptor(te: TargetExperiment, manifest_paths) -> list[str]:
    """Governance: the descriptor is the single source of truth for the shared hardware spec. Return the
    drift — for each bundle manifest, the shared-spec paths it fails to grant in ``allowed``. Empty list
    means every arm's bundle is consistent with the descriptor (so a run for this target is honest)."""
    from merlin.target.registry.experiment_descriptor import TargetExperiment

    drift: list[str] = []
    for mp in manifest_paths:
        doc = yaml.safe_load(Path(mp).read_text())
        variant = str(doc.get("variant") or "")
        info = te.information_set(variant) if variant else {}
        required = shared_spec_paths(te, variant or None)
        allowed = {e.get("path") for e in (doc.get("allowed") or []) if isinstance(e, dict)}
        denied = {e.get("path") for e in (doc.get("denied") or []) if isinstance(e, dict)}
        missing = required - allowed
        if missing:
            drift.append(f"{Path(mp).parent.name}: missing shared-spec {sorted(missing)}")
        expected_denied = {
            entry["path"] for entry in info.get("denied", ()) if isinstance(entry, dict) and entry.get("path")
        }
        if expected_denied - denied:
            drift.append(f"{Path(mp).parent.name}: missing information-set denial {sorted(expected_denied - denied)}")
        expected_condition = info.get("condition")
        if expected_condition and doc.get("condition") != expected_condition:
            drift.append(f"{Path(mp).parent.name}: condition {doc.get('condition')!r} != {expected_condition!r}")
        expected_pins = list(info.get("source_pins") or ())
        if list(doc.get("source_pins") or ()) != expected_pins:
            drift.append(f"{Path(mp).parent.name}: source_pins {doc.get('source_pins')!r} != {expected_pins!r}")
    return drift


# ---- corpora -----------------------------------------------------------------------------------
# The single place that resolves corpus locations the LIBRARY reads (reference-by-location).
#
# No library module should hardcode a corpus path; they call :func:`capsule_corpus_roots` /
# :func:`find_capsule` / :func:`kernel_corpus_root` instead. The locations themselves are DATA — the corpus
# registry ``src/merlin/resources/contracts/corpora.yaml`` — and this module is its one reader, so adding or moving a corpus
# is a registry edit. It is also the ONE sanctioned indirection to a corpus that still lives under
# ``experiments/`` (the boundary lint allowlists this module).
#
# Capsule inputs are historical read-only diagnostics, kept under ``experiments/``
# and excluded from installed distributions. Expert kernel corpora are external
# framework checkouts keyed by source spelling, layout, and selected owner.

_REGISTRY = ("contract", "corpora.yaml")  # checkout resource or bundled core data


@cache
def _load_registry(path: Path) -> dict:
    from merlin.common.record_formats import load_yaml

    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"corpus registry {path} is not a mapping")
    return data


def _registry() -> dict:
    """The corpus registry. A missing or malformed one RAISES: read as empty, it would make every corpus
    reader report "nothing to screen", which looks exactly like a clean result."""
    return _load_registry(data_path(*_REGISTRY))


def capsule_corpus_roots(*, owner_root: Path | None = None) -> list[Path]:
    """Existing historical capsule-corpus roots for read-only diagnostics.

    New grading uses a descriptor-selected, reviewed release; callers must not
    treat this broad registry as an implicit corpus for a target.

    ``owner_root`` selects an explicit source checkout for historical-corpus
    protection when a process is otherwise rooted in an external experiment
    workspace. The registry remains mandatory and is read from that owner.
    """
    if owner_root is None:
        registry = _registry()
        selected = os.environ.get("MERLIN_REPO_ROOT") or checkout_root()
        if selected is None:
            return []
        root = Path(selected)
    else:
        root = Path(owner_root)
        registry = _load_registry(resolve_grant("merlin/contract/corpora.yaml", root=root))
    roots = [resolve_grant(str(rel), root=root) for rel in registry.get("capsule_corpora") or []]
    return [r for r in roots if r.is_dir()]


def kernel_corpora() -> dict[str, dict]:
    """Every expert kernel corpus the registry declares, ``{framework: spec}``, in declaration order."""
    return {str(k): dict(v or {}) for k, v in (_registry().get("kernel_corpora") or {}).items()}


def sched_corpora() -> dict[str, dict]:
    """Every corpus the scheduling-IR expressiveness measurement may read, ``{name: spec}``.

    Separate from :func:`kernel_corpora` because the consumer is: the mining layer BUILDS a corpus,
    this one only reads a corpus already on disk and asks whether it can stand as expert evidence.
    Each spec names a ``pin`` in the hardware-pin registry rather than a path, so an unresolvable
    corpus can report the ENVIRONMENT VARIABLE it needs instead of a zero denominator.
    """
    return {str(k): dict(v or {}) for k, v in (_registry().get("sched_corpora") or {}).items()}


def sched_corpus_record(spec: dict) -> Path | None:
    """The project-relative eligibility record a ``sched_corpora`` spec declares, or None."""
    rel = str(spec.get("provenance_record") or "")
    return resolve_grant(rel) if rel else None


def kernel_corpus_for_source(source: str | None) -> str | None:
    """The corpus (framework name) a ``kernel.source`` spelling names, matched case-insensitively, or None."""
    want = (source or "").lower()
    if not want:
        return None
    for name, spec in kernel_corpora().items():
        if want == name.lower() or want in {str(s).lower() for s in spec.get("sources") or []}:
            return name
    return None


def kernel_corpus_target(name: str | None) -> str | None:
    """The target that owns corpus ``name``'s kernel-ceiling bench config (its registry ``target:``), or
    None when the corpus is unregistered or declares no owner."""
    spec = kernel_corpora().get(name or "")
    owner = str((spec or {}).get("target") or "").strip()
    return owner or None


def kernel_corpus_env(name: str) -> str:
    """The environment variable pointing at corpus ``name``'s checkout: ``MERLIN_<NAME>_REPO`` -- the
    convention the kernel-index CLI already reads, derived from the name rather than listed."""
    return f"MERLIN_{name.upper()}_REPO"


def kernel_corpus_root(name: str) -> Path | None:
    """Root of expert corpus ``name``'s checkout, or None when the registry declares no such corpus.

    Precedence: :func:`kernel_corpus_env` when set; then a local clone at ``<repo>/tmp/kernels/<checkout>``
    if one is actually there; then the corpus's declared ``ext`` fallback (a directory inside an external
    checkout this repo already configures, via ``ext_path``) when that is on disk; else the local-clone
    path, so a fresh checkout's error names the location a clone is expected at rather than someone else's
    machine.
    """
    spec = kernel_corpora().get(name)
    if spec is None:
        return None
    env = os.environ.get(kernel_corpus_env(name))
    if env:
        return Path(env)
    from merlin.common.paths import repo_root

    local = repo_root() / "tmp" / "kernels" / str(spec.get("checkout") or name)
    if local.is_dir():
        return local
    ext = spec.get("ext") or {}
    if ext.get("name"):
        try:
            from merlin.common.paths import ext_path

            cand = ext_path(str(ext["name"])) / str(ext.get("subpath") or "")
            if cand.is_dir():
                return cand
        except (KeyError, ImportError):
            pass
    return local


def capsule_store_targets() -> list[str]:
    """Every target that owns a capsule store, discovered from the tree.

    A caller that has no target name yet needs this before it can call anything keyed on one, and the
    alternative — each caller listing the directory itself — is how a target path gets duplicated in
    five modules and how a target NAME gets typed into library code. It is deliberately NOT
    :func:`~merlin.target.capabilities.manifest.discovered_targets`: that set is targets shipping a
    capability residual, which is a different (and differently-sized) population. Ordered, so a report
    built from it is stable across runs.
    """
    root = repo_root() / "experiments" / "studies" / "capsule_bench" / "targets"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def descriptor_paths(target: str = "*") -> list[Path]:
    """Retained experiment descriptors, in the same deterministic owner order."""
    root = repo_root()
    patterns = (
        "examples/*/{target}/target/descriptor.yaml",
        "experiments/studies/*/targets/{target}/target_experiment.yaml",
        "experiments/inputs/target-inputs/{target}/target_experiment.yaml",
    )
    return [path for pattern in patterns for path in sorted(root.glob(pattern.format(target=target))) if path.is_file()]


def descriptor_path(target: str) -> Path:
    """Where ``target``'s experiment descriptor lives — the per-target convention path, or whatever
    ``MERLIN_TARGET_EXPERIMENT`` overrides it to. Lives here because this module is the one allowed to
    know the ``experiments/`` layout; callers elsewhere took the convention path directly and silently
    ignored the override, so an out-of-tree descriptor resolved for some readers and not others."""
    import os

    override = os.environ.get("MERLIN_TARGET_EXPERIMENT", "").strip()
    if override:
        return Path(override)
    return standard_descriptor_path(target)


def standard_descriptor_path(target: str) -> Path:
    """The CONVENTION path for ``target``'s descriptor, ignoring ``MERLIN_TARGET_EXPERIMENT``.

    Separate from :func:`descriptor_path` because the override names ONE descriptor, not a mapping: a
    caller asking for target *X* while the variable points at *Y* must not be handed *Y*. A caller that
    needs the override validated against the target it asked for reads the variable itself and falls
    back to this. Both spellings live here so the ``experiments/`` layout has exactly one owner.
    """
    from merlin.common.paths import repo_root

    for candidate in descriptor_paths():
        if _declared_target(candidate) == target:
            return candidate
    return repo_root() / "experiments" / "studies" / "capsule_bench" / "targets" / target / "target_experiment.yaml"


def source_experiment_env(target: str | None = None, *, descriptor: str | Path | None = None) -> list[str]:
    """Load ``targets/<target>/experiment.env`` into ``os.environ``, setting ONLY keys not already
    present, and return the keys it set.

    The harness does this for every arm before a run; a caller that grades a package WITHOUT going
    through the harness did not, and the difference is not cosmetic. One target's certifying tier is a
    program-driven Verilator sim registered through ``MERLIN_EXT_<TARGET>_VSIM``; with the variable
    absent there is no adapter, and its own profile spells out the consequence -- every capsule reports
    ``incomplete``, never a pass. Fail-closed is right, but reporting a whole suite incomplete because a
    path was not sourced is a tooling artifact wearing a verdict's clothes.

    The process environment always WINS, so an exported var is never overridden. Structured KEY=VALUE
    parse, ``#`` comments; target-agnostic (keyed off declared resource ownership, defaulting to
    descriptor siblings). An explicit descriptor lets invocation setup
    source tooling before parsing the target identity, without guessing an in-tree location.
    """
    import os

    if descriptor is None:
        if target is None:
            raise ValueError("source_experiment_env requires a target or explicit descriptor")
        desc = descriptor_path(target)
    else:
        desc = Path(descriptor)

    f = descriptor_resources_root(desc) / "experiment.env"
    set_keys: list[str] = []
    if not f.is_file():
        return set_keys
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            set_keys.append(k)
    return set_keys


def experiment_for(target: str):
    """``target``'s parsed descriptor, or None when it ships none / names a different target."""
    desc = descriptor_path(target)
    if not desc.is_file():
        return None
    try:
        te = load_target_experiment(desc)
    except Exception:  # noqa: BLE001 — unreadable descriptor
        return None
    return te if str(getattr(te, "target", "")) == target else None


def graded_capsule_roots(target: str, *, hidden: bool = False) -> list[Path]:
    """The roots that make up ``target``'s descriptor-selected GRADED suite.

    Missing, mismatched, unreadable or empty descriptors return no roots: a
    cross-target historical corpus is never an implicit grading default.

    ``hidden=True`` returns the holdout roots instead. They are a SEPARATE tree, deliberately excluded
    from the public roots (``corpus_siblings`` skips ``hidden/``), so a hidden grade that reused the
    public roots would match nothing and report a 0/0 "pass" that never ran. Empty is a real answer for a
    target that ships no holdouts, and is returned as empty rather than papered over with a fallback.

    A target's suite is not one directory: the capsules are split by kind into sibling categories
    (``isa`` / ``layers`` / ``model`` / ``model_slices``), and different targets keep those siblings in
    different places -- one target's sit at the corpus root, another's under ``<target>/``. Passing their common
    parent is NOT the fix, because that parent holds every target's corpus at once; see the warning in
    :func:`merlin.development.targetgen.corpus.capsule_io.discover_capsules`.

    Measured consequence of getting this wrong: grading the gemmini package against
    ``experiments/corpora`` pulled in 173 capsules from seven targets, marked 89 of them "outside
    this target's declared capability", and reported ``1/84`` -- a number that reads like a catastrophic
    regression and means nothing. The target's own suite is 36.

    ``TargetExperiment.graded_roots()`` is the resolution the A/B launchers and ``readiness_check.py``
    already use; this exposes it to library callers that have only a target NAME. Honours
    ``MERLIN_TARGET_EXPERIMENT`` (the same override ``capsule_bench/harness/_common.py`` reads) so a
    target whose descriptor lives out of tree still resolves.
    """
    desc = descriptor_path(target)
    if not desc.is_file():
        return []
    try:
        te = load_target_experiment(desc)
        if str(getattr(te, "target", "")) != target:  # an override naming a DIFFERENT target
            return []
        roots = [r for r in (te.hidden_roots() if hidden else te.graded_roots()) if r.is_dir()]
    except Exception:  # noqa: BLE001 — unreadable descriptor
        return []
    if hidden:
        return roots  # empty == this target ships no holdouts
    return roots


def perf_capsule_roots(target: str) -> list[Path]:
    """The roots holding ``target``'s PERFORMANCE capsules, or ``[]`` when it ships none.

    The library-facing form of :meth:`TargetExperiment.perf_roots`, resolved from a target NAME like
    its graded and hidden siblings here. Performance capsules are excluded from the functional suite by
    the underscore convention, so asking the graded roots for them reports that they do not exist --
    which reads as "this target has no performance families" and is how an optimization run comes to
    refuse on capsules that are sitting right there.
    """
    desc = descriptor_path(target)
    if not desc.is_file():
        return []
    try:
        te = load_target_experiment(desc)
        if str(getattr(te, "target", "")) != target:  # an override naming a DIFFERENT target
            return []
        return [r for r in te.perf_roots() if r.is_dir()]
    except Exception:  # noqa: BLE001 — unreadable descriptor
        return []


def find_capsule(name: str) -> Path | None:
    """Locate a capsule directory by name across the corpus roots (first match wins)."""
    for root in capsule_corpus_roots():
        for cy in root.rglob("capsule.yaml"):
            if cy.parent.name == name:
                return cy.parent
    return None
