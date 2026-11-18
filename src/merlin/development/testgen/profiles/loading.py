"""Host-owned Phase 0 profiles implementation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.analysis.perf.contract.profile import TRAITS  # noqa: E402
from merlin.common.digest import is_sha256
from merlin.common.paths import repo_root  # noqa: E402
from merlin.development.testgen.generation.provenance import _document_digest
from merlin.experiments.definition import ExperimentSpec, SpecError, catalog, engine_definitions, load_spec
from merlin.runtime.backends.registry import EXECUTION_CAPABILITIES  # noqa: E402
from merlin.target.spec.software import (
    load_software_spec,
    numerical_datapath,
    software_spec_identity,
    software_spec_path_for_recipe,
)

# ---- profiles ----------------------------------------------------------------------------------


def validate_host_source_basis(value, *, owner):
    """A source/execution selection is distinct from an accelerator RTL trait."""
    from merlin.development.testgen.profiles.synthesis_selection import HOST_SOURCE_BASIS
    from merlin.development.testgen.public_sources.programs import BASIS_SCHEMA

    if isinstance(value, dict) and value.get("schema") == BASIS_SCHEMA:
        if (
            set(value)
            != {"schema", "kind", "capture_sha256", "receipt_sha256", "operations", "numerical_reference", "boundary"}
            or value["kind"] != "public_program"
            or any(not is_sha256(value[key]) for key in ("capture_sha256", "receipt_sha256"))
            or value["numerical_reference"] != "host_torch_eager"
            or value["boundary"] != "complete_program_call"
            or not isinstance(value["operations"], list)
            or not value["operations"]
            or any(not isinstance(op, str) or not op for op in value["operations"])
            or len(set(value["operations"])) != len(value["operations"])
        ):
            raise ValueError(f"{owner}: public program basis requires its exact source/reference selection")
        return value
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "kind", "software_intake_sha256", "operations", "numerical_reference", "boundary"}
        or value["schema"] != HOST_SOURCE_BASIS
        or value["kind"] != "host"
        or not is_sha256(value["software_intake_sha256"])
        or value["numerical_reference"] != "integer_reference"
        or value["boundary"] != "complete_host_call"
        or not isinstance(value["operations"], list)
        or not value["operations"]
        or any(not isinstance(op, str) or not op for op in value["operations"])
        or len(set(value["operations"])) != len(value["operations"])
    ):
        raise ValueError(f"{owner}: host source basis requires its exact reviewed source/reference/execution selection")
    return value


def pending_host_source_basis(gate):
    """Validate a source selection without accepting saved execution labels."""
    selected = gate.get("source_execution_basis")
    if selected is None:
        return None
    validate_host_source_basis(selected, owner="performance gate")
    # No independent host execution issuer is selected by source construction.
    # Saved labels cannot qualify the experiment or its timing.
    return {
        "selection": copy.deepcopy(selected),
        "satisfied": None,
        "tier": "not_established",
        "evidence": "host source selection has no observed execution owner",
        "missing": ["independently observed execution of the same complete host source boundary"],
    }


_PUBLIC_CATEGORY_BY_KIND = {
    "isa": "isa",
    "layer": "layers",
    "model_slice": "model_slices",
    "model": "model",
}


def _normalize_public_capsules(profile: dict, *, source: Path) -> None:
    """Fill redundant metadata only for entries in the selected public recipe.

    Sidecars (especially private holdouts) and the shared performance template
    are merged later and retain their own explicit labels and categories.
    Explicit values on a public entry also remain authoritative.
    """
    entries = profile.get("capsules") or []
    if not isinstance(entries, list):
        raise ValueError(f"{source}: capsules must be a list")
    normalized = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: capsule {index} must be a mapping")
        row = dict(entry)
        row.setdefault("label", "public")
        category = _PUBLIC_CATEGORY_BY_KIND.get(row.get("kind"))
        if category is not None:
            row.setdefault("cat", category)
        normalized.append(row)
    profile["capsules"] = normalized


def _validate_performance_block(block, *, owner: str, blocked: bool = False) -> dict:
    """Validate the claim-bearing contract before a family can be admitted.

    Performance metadata is consumed later than corpus generation, so a partial
    block otherwise succeeds here and turns into an unmeasurable family only
    after an expensive run.  All claim, comparison, falsifier, applicability,
    regime, emitter, and cost fields therefore fail closed at profile load.
    """
    from merlin.development.testgen.profiles.synthesis_selection import (
        _MEMBER_CLASSES,
        _PERFORMANCE_CLAIMS,
        _PERFORMANCE_FIELDS,
        _PERFORMANCE_NESTED_FIELDS,
        _STREAM_CLAIMS,
    )

    if not isinstance(block, dict):
        raise ValueError(f"{owner}: `performance` must be a mapping")
    missing = sorted(_PERFORMANCE_FIELDS - block.keys())
    if missing:
        raise ValueError(f"{owner}: performance block missing required field(s) {missing}")
    for field in ("level", "family", "lever"):
        if not isinstance(block[field], str) or not block[field].strip():
            raise ValueError(f"{owner}: performance.{field} must be a non-empty string")
    claim = block.get("claim")
    if claim not in _PERFORMANCE_CLAIMS:
        raise ValueError(f"{owner}: performance.claim must be one of {sorted(_PERFORMANCE_CLAIMS)}, got {claim!r}")
    if claim in _STREAM_CLAIMS and not blocked:
        acceptance = block.get("acceptance")
        analyzer = acceptance.get("analyzer") if isinstance(acceptance, dict) else None
        if not isinstance(analyzer, str) or not analyzer.strip():
            raise ValueError(
                f"{owner}: an {claim} claim is decided only by the analyzer its contract names, so "
                "performance.acceptance.analyzer is required -- without one the family reads its "
                "stream and nothing ever turns that reading into a verdict"
            )
    member_class = block.get("member_class")
    if member_class not in _MEMBER_CLASSES:
        raise ValueError(
            f"{owner}: performance.member_class must be one of {sorted(_MEMBER_CLASSES)}, got "
            f"{member_class!r} -- a member with no declared job is measured every candidate AND "
            f"summed into the objective, which is how a law-fitting cohort came to weigh as much as "
            f"the workload it was never meant to represent"
        )
    if member_class == "REFERENCE" and claim != "RECOVERS":
        raise ValueError(
            f"{owner}: a REFERENCE member is measured against a shipped implementation, so its claim "
            f"must be RECOVERS; got {claim!r}"
        )
    gate_value = block.get("gate")
    if isinstance(gate_value, dict) and "requires" in gate_value:
        raise ValueError(f"{owner}: performance.gate.requires is not accepted; use canonical `gate.traits`")
    for field, required in _PERFORMANCE_NESTED_FIELDS.items():
        value = block[field]
        if not isinstance(value, dict):
            raise ValueError(f"{owner}: performance.{field} must be a mapping")
        absent = sorted(required - value.keys())
        if absent:
            raise ValueError(f"{owner}: performance.{field} missing required field(s) {absent}")
        for key in required - {"knobs", "traits"}:
            nested = value[key]
            if (
                nested is None
                or (isinstance(nested, str) and not nested.strip())
                or (isinstance(nested, (list, tuple, dict)) and not nested)
            ):
                raise ValueError(f"{owner}: performance.{field}.{key} must be non-empty")
    gate = block["gate"]
    names = gate["traits"]
    basis = gate.get("source_execution_basis")
    if basis is not None:
        validate_host_source_basis(basis, owner=owner)
    if (
        not isinstance(names, list)
        or (not names and basis is None)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError(f"{owner}: performance.gate.traits must be a non-empty list of trait names")
    if len(set(names)) != len(names):
        raise ValueError(f"{owner}: performance.gate.traits contains duplicate names {names}")
    unknown = sorted(set(names) - set(TRAITS))
    if unknown:
        raise ValueError(f"{owner}: unknown performance trait(s) {unknown}; canonical traits are {list(TRAITS)}")
    execution_names = gate.get("execution_capabilities", [])
    if not isinstance(execution_names, list) or any(not isinstance(name, str) or not name for name in execution_names):
        raise ValueError(f"{owner}: performance.gate.execution_capabilities must be a list of names")
    if len(set(execution_names)) != len(execution_names):
        raise ValueError(f"{owner}: performance.gate.execution_capabilities contains duplicate names {execution_names}")
    unknown_execution = sorted(set(execution_names) - set(EXECUTION_CAPABILITIES))
    if unknown_execution:
        raise ValueError(
            f"{owner}: unknown execution capability(s) {unknown_execution}; canonical capabilities "
            f"are {list(EXECUTION_CAPABILITIES)}"
        )
    if gate["on_missing"] != "skip_with_evidence":
        raise ValueError(f"{owner}: performance.gate.on_missing must be 'skip_with_evidence'")
    emitter = block["emitter"]
    if not isinstance(emitter["status"], str) or not emitter["status"]:
        raise ValueError(f"{owner}: performance.emitter.status must be non-empty")
    if not isinstance(emitter["entry"], str) or not emitter["entry"]:
        raise ValueError(f"{owner}: performance.emitter.entry must be non-empty")
    if not isinstance(emitter["knobs"], dict):
        raise ValueError(f"{owner}: performance.emitter.knobs must be a mapping")
    return block


def _comparison_roles(sweep: dict) -> list[str]:
    roles = {str(role) for role in (sweep.get("comparison_roles") or []) if str(role)}
    roles |= {
        str(group["role"])
        for variant in (sweep.get("variants") or [])
        if isinstance(variant, dict)
        if isinstance((group := variant.get("comparison_group")), dict)
        if group.get("role")
    }
    return sorted(roles)


def _validate_declared_fit_axes(sweep: dict, *, owner: str) -> None:
    fit_axes = sweep.get("fit_axes") or []
    if (
        not isinstance(fit_axes, list)
        or not fit_axes
        or any(not isinstance(axis, str) or not axis for axis in fit_axes)
    ):
        raise ValueError(f"{owner}: fit_axes must be a non-empty list of axis names")
    axes = sweep.get("axes") or {}
    unknown = [axis for axis in fit_axes if axis not in axes]
    if unknown:
        raise ValueError(f"{owner}: fitted axes {unknown} are not declared in axes")
    for axis in fit_axes:
        points = axes[axis]
        if isinstance(points, dict):
            # A DERIVED axis cannot be checked here -- its points are a function of the target's own
            # facts and no target is bound at profile load. What IS checkable at load is the promise:
            # the declaration must ask for at least two points per band, and `expand_sweeps` re-checks
            # the resolved values. A derivation that silently returned one point would otherwise ship a
            # one-point fit under a contract that says it is fitted.
            if str(points.get("derive") or "") not in _DERIVED_AXES:
                raise ValueError(
                    f"{owner}: fitted axis {axis} declares an unknown derivation "
                    f"{points.get('derive')!r}; known derivations are {list(_DERIVED_AXES)}"
                )
            if points.get("derive") == "resident_allocation_boundary":
                from merlin.development.testgen.profiles.resource_boundaries import validate

                validate(points, owner=owner)
                continue
            if int(points.get("points_per_regime", 0)) < 2:
                raise ValueError(
                    f"{owner}: fitted axis {axis} derives fewer than two points per regime; a rate "
                    "and a fixed intercept are two parameters and one point cannot separate them"
                )
            continue
        if not isinstance(points, list) or len({repr(point) for point in points}) < 2:
            raise ValueError(f"{owner}: fitted axis {axis} must declare at least two distinct points")


def _performance_family_record(sweep: dict) -> dict:
    performance = (sweep.get("base") or {}).get("performance") or sweep.get("performance") or {}
    return {
        "family": performance.get("family") or sweep.get("id"),
        "claim": performance.get("claim"),
        "fit_axes": list(sweep.get("fit_axes") or []),
        "comparison_roles": _comparison_roles(sweep),
    }


def _target_local_perf_declarations(profile: dict) -> list[str]:
    """Names of performance entries embedded in a target-owned profile.

    Functional sweeps remain target-owned: they describe operation coverage for
    that target.  Performance families do not.  Their shapes and comparison
    structure must come from the one shared ``_perf.yaml`` template, otherwise
    onboarding a target can quietly fork the experiment it is compared under.
    """
    found: list[str] = []
    for entry in profile.get("capsules") or []:
        if isinstance(entry, dict) and (entry.get("cat") in {"perf", "_perf"} or "performance" in entry):
            found.append(str(entry.get("name") or "<unnamed capsule>"))
    for sweep in profile.get("sweeps") or []:
        if not isinstance(sweep, dict):
            continue
        base = sweep.get("base") or {}
        variants = sweep.get("variants") or []
        if (isinstance(base, dict) and (base.get("cat") in {"perf", "_perf"} or "performance" in base)) or any(
            isinstance(v, dict) and (v.get("cat") in {"perf", "_perf"} or "performance" in v) for v in variants
        ):
            found.append(str(sweep.get("id") or "<unnamed sweep>"))
    return found


def _select_performance_withdrawals(profile: dict, *, source: Path) -> None:
    """Validate a target recipe's reviewed withdrawals of shared performance families.

    The shared template stays the one definition of every family; a target may only decline one,
    by name, with the reason and the decision that withdrew it. Both are recorded beside the
    generation's other skipped families, so a withdrawn family is never mistaken for a gate refusal
    or for a family that was never declared. A withdrawal naming no declared family is an error: it
    would otherwise persist silently after the family was renamed or removed.
    """
    rows = profile.pop("performance_withdrawals", None)
    if rows is None:
        return
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}: performance_withdrawals must be a nonempty list")
    declared = {row["family"] for row in (profile.get("_performance_template") or {}).get("families") or []}
    selected: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"family", "reason", "decided_by"}:
            raise ValueError(f"{source}: each performance withdrawal needs exactly family, reason and decided_by")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in row):
            raise ValueError(f"{source}: performance withdrawal fields must be nonempty strings")
        if row["family"] not in declared:
            raise ValueError(f"{source}: performance withdrawal names undeclared family {row['family']!r}")
        if row["family"] in selected:
            raise ValueError(f"{source}: performance family {row['family']!r} is withdrawn twice")
        selected[row["family"]] = copy.deepcopy(row)
    profile["_performance_withdrawals"] = selected
    profile["_performance_template"]["withdrawn"] = [selected[name] for name in sorted(selected)]


TARGET_DECLARED = "declared_by_the_selected_target"


TARGET_UNDECLARED = "UNDECLARED_by_the_selected_target"


_REGIME_EDGES = ("load_critical_at_or_below", "mesh_critical_at_or_above")


def _target_regime(row: object, *, family: str, source: Path) -> dict:
    """Validate one ``performance_regimes`` entry: both edges, lower strictly below upper, and a basis."""
    if not isinstance(row, dict) or set(row) != {*_REGIME_EDGES, "basis"}:
        raise ValueError(f"{source}: performance_regimes.{family} needs exactly {', '.join(_REGIME_EDGES)} and basis")
    low, high = (row[edge] for edge in _REGIME_EDGES)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (low, high)):
        raise ValueError(f"{source}: performance_regimes.{family} edges must be positive integers")
    if low >= high:
        raise ValueError(f"{source}: performance_regimes.{family} lower edge must be below the upper edge")
    if not isinstance(row["basis"], str) or not row["basis"].strip():
        raise ValueError(f"{source}: performance_regimes.{family} must state the measurement as its basis")
    return row


def _apply_target_regimes(profile: dict, sweeps: list[dict], *, source: Path) -> None:
    """Fill each shared family's target-declared regime bracket from the selected target's recipe.

    Where a lever stops paying is MEASURED on one device, so the shared template names only the
    discriminant and marks the edges and their basis :data:`TARGET_DECLARED`. The recipe supplies them
    under ``performance_regimes.<family>``; the merged contract records them in the template's own key
    order, the edges where the marker stood. A target that declares none for such a family gets both
    markers replaced by :data:`TARGET_UNDECLARED` and no edges, so the family's analyzer refuses the
    cohort with that reason. A recipe entry for a family whose template takes no bracket is an error:
    it would otherwise persist silently after the family changed.
    """
    rows = profile.pop("performance_regimes", None)
    if rows is not None and (not isinstance(rows, dict) or not rows):
        raise ValueError(f"{source}: performance_regimes must be a nonempty mapping of family to bracket")
    rows = dict(rows or {})
    wanting: dict[str, dict] = {}
    for sweep in sweeps:
        acceptance = ((sweep.get("base") or {}).get("performance") or {}).get("acceptance")
        regime = acceptance.get("regime") if isinstance(acceptance, dict) else None
        if isinstance(regime, dict) and regime.get("bracket") == TARGET_DECLARED:
            wanting[str(sweep.get("id"))] = acceptance
    unknown = sorted(set(rows) - set(wanting))
    if unknown:
        raise ValueError(
            f"{source}: performance_regimes names {unknown}, whose shared template declares no "
            "target-supplied regime bracket"
        )
    for family, acceptance in wanting.items():
        declared = _target_regime(rows[family], family=family, source=source) if family in rows else None
        merged: dict = {}
        for key, value in acceptance["regime"].items():
            if declared is None:
                merged[key] = TARGET_UNDECLARED if value == TARGET_DECLARED else value
            elif key == "bracket":
                merged.update({edge: declared[edge] for edge in _REGIME_EDGES})
            elif key == "basis" and value == TARGET_DECLARED:
                merged[key] = declared["basis"]
            else:
                merged[key] = value
        acceptance["regime"] = merged


def _profiles_root(profiles_root: str | Path | None) -> Path:
    if profiles_root is None:
        raise ValueError("Phase 0 requires explicit recipe inputs or profiles_root; select an experiment definition")
    return Path(profiles_root).expanduser().resolve()


def _merge_shared_perf(
    profile: dict,
    *,
    source: Path,
    profiles_root: str | Path | None = None,
    performance_template: str | Path | None = None,
) -> None:
    """Merge the sole shared performance template into one target profile."""
    shared_path = (
        Path(performance_template).expanduser().absolute()
        if performance_template is not None
        else _profiles_root(profiles_root) / "_perf.yaml"
    )
    shared = yaml.safe_load(shared_path.read_text(encoding="utf-8")) or {}
    misplaced = _target_local_perf_declarations(profile)
    if misplaced:
        raise ValueError(
            f"{source} declares target-local performance template entries {misplaced}; "
            f"move them to the shared {shared_path}"
        )
    if shared.get("capsules"):
        raise ValueError(
            f"shared performance template {shared_path} must generate entries through `sweeps`, "
            "not hand-author `capsules`"
        )
    non_perf = [
        str(s.get("id") or "<unnamed sweep>")
        for s in (shared.get("sweeps") or [])
        if not isinstance(s, dict) or (s.get("base") or {}).get("cat") != "_perf"
    ]
    if non_perf:
        raise ValueError(f"shared performance template {shared_path} contains non-performance sweeps {non_perf}")
    # A copy: the selected target's data is merged into the sweeps below, while the recorded template
    # digest stays the digest of the shared template itself.
    sweeps = copy.deepcopy(list(shared.get("sweeps") or []))
    blocked = list(shared.get("blocked_unimplemented") or [])
    family_records: list[dict] = []
    seen: set[str] = set()
    for sweep in sweeps:
        sweep_id = str(sweep.get("id") or "").strip()
        if not sweep_id:
            raise ValueError(f"shared performance template {shared_path} has a sweep without an id")
        performance = _validate_performance_block(
            (sweep.get("base") or {}).get("performance"), owner=f"shared sweep {sweep_id}"
        )
        if sweep.get("requires_form_scope") is not None:
            from merlin.development.testgen.performance.form_scope import validate_form_scope_declaration

            validate_form_scope_declaration(sweep, owner=f"shared sweep {sweep_id}")
        elif sweep.get("requires_capacity_scope") is not None:
            from merlin.development.testgen.performance.capacity_families import (
                validate_capacity_scope_declaration,
            )

            validate_capacity_scope_declaration(sweep, owner=f"shared sweep {sweep_id}")
        else:
            _validate_declared_fit_axes(sweep, owner=f"shared sweep {sweep_id}")
        if performance["family"] != sweep_id:
            raise ValueError(f"shared sweep {sweep_id}: performance.family must equal the sweep id")
        if sweep_id in seen:
            raise ValueError(f"shared performance template repeats family {sweep_id!r}")
        seen.add(sweep_id)
        family_records.append(_performance_family_record(sweep))
    for item in blocked:
        if not isinstance(item, dict):
            raise ValueError(f"shared {shared_path}: blocked_unimplemented entries must be mappings")
        family = str(item.get("family") or "").strip()
        if not family or not str(item.get("reason") or "").strip():
            raise ValueError(f"shared {shared_path}: blocked_unimplemented needs family and reason")
        performance = _validate_performance_block(
            item.get("performance"), owner=f"blocked performance family {family}", blocked=True
        )
        if performance["family"] != family:
            raise ValueError(f"blocked family {family}: performance.family must equal its family")
        if family in seen:
            raise ValueError(f"shared performance template repeats family {family!r}")
        seen.add(family)
        family_records.append(
            {
                "family": family,
                "claim": performance.get("claim"),
                "fit_axes": list(item.get("fit_axes") or []),
                "comparison_roles": list(item.get("comparison_roles") or []),
            }
        )
    _apply_target_regimes(profile, sweeps, source=source)
    profile["sweeps"] = list(profile.get("sweeps") or []) + sweeps
    # Recorded repo-root-relative so a consumer can resolve it against `repo_root()` alone.
    # It is derived from the same `shared_path` this function read, never re-spelled by hand: a
    # hand-typed (or mis-rooted) prefix is exactly how this record went stale after the tree moved
    # under `merlin/` and the campaign gate started refusing its own corpus.
    try:
        template_path = shared_path.relative_to(repo_root()).as_posix()
    except ValueError:
        # A template outside the checkout (test fixtures point PROFILES at a tmp dir) can only be
        # named absolutely. Consumers resolve an absolute record as-is and refuse it when it does
        # not exist, so this stays fail-closed rather than becoming a second relative spelling.
        template_path = str(shared_path)
    profile["_performance_template"] = {
        "path": template_path,
        "sha256": _document_digest(shared),
        "families": family_records,
        "blocked_unimplemented": copy.deepcopy(blocked),
    }
    if "curriculum_derivation" in shared:
        profile["_performance_template"]["curriculum_derivation"] = copy.deepcopy(shared["curriculum_derivation"])
    if "public_boundary_factory" in shared:
        profile["_performance_template"]["public_boundary_factory"] = copy.deepcopy(shared["public_boundary_factory"])
    if "public_boundary_factories" in shared:
        profile["_performance_template"]["public_boundary_factories"] = copy.deepcopy(
            shared["public_boundary_factories"]
        )


def validate_profile_inputs(
    *,
    profiles_root=None,
    recipe=None,
    performance_template=None,
    conformance_spec=None,
    synth_profile=None,
    smt_profile=None,
    hidden_profile=None,
    software_spec=None,
    chain_profile=None,
) -> None:
    """Reject mixed ownership modes without touching any input, especially private sidecars."""
    if recipe is not None:
        if profiles_root is not None:
            raise ValueError("recipe and profiles_root are mutually exclusive")
        if performance_template is None:
            raise ValueError("explicit recipe requires performance_template")
    elif any(
        value is not None
        for value in (
            performance_template,
            conformance_spec,
            synth_profile,
            smt_profile,
            hidden_profile,
            software_spec,
            chain_profile,
        )
    ):
        raise ValueError("explicit performance_template and sidecar paths require recipe")


def chain_profile_entries(path: str | Path) -> list[dict]:
    """The public correctness members a chain-demand stage derived (:mod:`.chain_demand`).

    A CHAIN PROFILE IS PUBLIC AND DERIVED. Each entry is a model-derived connected case of the
    generated cohort, bound to its digest-bound plan; nothing here is a holdout, so it is read for
    public outputs too, and an entry that claims another origin or category is refused rather than
    published under the chain stage's name.
    """
    from merlin.development.testgen.performance.chain_demand import BLOCK

    selected = Path(path).expanduser().absolute()
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"declared chain profile is not an ordinary file: {selected}")
    doc = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict) or set(doc) != {"capsules"} or not isinstance(doc["capsules"], list):
        raise ValueError(f"{selected}: a chain profile holds exactly a capsules list")
    entries = []
    for entry in doc["capsules"]:
        block = entry.get(BLOCK) if isinstance(entry, dict) else None
        if (
            not isinstance(block, dict)
            or len(str(block.get("plan_sha256") or "")) != 64
            or entry.get("source_role") != "model_derived"
            or entry.get("label") != "public"
            or entry.get("cat") in {None, "hidden"}
            or str(entry.get("cat")).startswith("_")
        ):
            label = entry.get("name") if isinstance(entry, dict) else entry
            raise ValueError(
                f"{selected}: chain profile entry {label!r} "
                "must be a public model-derived member bound to its chain plan"
            )
        entries.append(copy.deepcopy(entry))
    return entries


def load_profile(
    target: str,
    *,
    include_holdouts: bool = True,
    profiles_root: str | Path | None = None,
    recipe: str | Path | None = None,
    performance_template: str | Path | None = None,
    conformance_spec: str | Path | None = None,
    synth_profile: str | Path | None = None,
    smt_profile: str | Path | None = None,
    hidden_profile: str | Path | None = None,
    descriptor: str | Path | None = None,
    software_spec: str | Path | None = None,
    diagnostic: bool = False,
    chain_profile: str | Path | None = None,
) -> dict:
    """The target's functional profile plus shared perf and the private holdout sidecar.

    The holdout spec (op + dtype + exact shape) is an answer, not a contract: the tracked profile lives
    inside the ``src/merlin/resources/contracts/`` tree every arm is granted read-only, so a holdout declared there is
    readable by the agent under test. It therefore lives in ``profiles/<target>.hidden.yaml``, which is
    gitignored and masked by :mod:`merlin.development.targetgen.sandbox.answer_surfaces`. When the sidecar is absent
    -- a public clone, or a sandbox where it is masked -- this returns the public profile unchanged and
    the run simply emits no hidden capsules, which is the correct behaviour rather than an error.

    Explicit ``recipe`` mode reads only the named files; no sibling is discovered. Its shared
    ``performance_template`` is mandatory. Optional sidecars may be declared but absent. The legacy
    ``profiles_root`` directory interface retains its established basename conventions.

    ``include_holdouts=False`` for any caller whose OUTPUT is public: it does not read or stat the
    hidden path. A published artifact enumerating the holdouts would leak them just as the profile did.
    """
    from merlin.development.testgen.profiles.synthesis_selection import (
        selected_software_spec_path,
        verify_selected_synthesis,
    )

    validate_profile_inputs(
        profiles_root=profiles_root,
        recipe=recipe,
        performance_template=performance_template,
        conformance_spec=conformance_spec,
        synth_profile=synth_profile,
        smt_profile=smt_profile,
        hidden_profile=hidden_profile,
        software_spec=software_spec,
        chain_profile=chain_profile,
    )
    if recipe is None:
        profiles = _profiles_root(profiles_root)
        public = profiles / f"{target}.yaml"
        shared = profiles / "_perf.yaml"
        synth = profiles / f"{target}.synth.yaml"
        smt = profiles / f"{target}.smt.yaml"
        side = profiles / f"{target}.hidden.yaml"
    else:
        public = Path(recipe).expanduser().absolute()
        shared = Path(performance_template).expanduser().absolute()
        synth = Path(synth_profile).expanduser().absolute() if synth_profile is not None else None
        smt = Path(smt_profile).expanduser().absolute() if smt_profile is not None else None
        side = Path(hidden_profile).expanduser().absolute() if include_holdouts and hidden_profile is not None else None
        for optional in (synth, smt, side):
            if optional is not None and (optional.exists() or optional.is_symlink()) and not optional.is_file():
                raise ValueError(f"declared optional profile is not a file: {optional}")
    prof = yaml.safe_load(public.read_text(encoding="utf-8")) or {}
    if not isinstance(prof, dict):
        raise ValueError(f"{public}: recipe must be a mapping")
    if recipe is not None:
        # Explicit overrides resolve abstract contract fidelities. Concrete
        # contract tier names need no override; neither route probes the build
        # host, which would make identical derivations depend on installation.
        oracles = prof.get("performance_oracles", {})
        if not isinstance(oracles, dict) or any(
            not isinstance(tier, str)
            or not tier.startswith("L")
            or not tier[1:]
            or not all("0" <= digit <= "9" for digit in tier[1:])
            or not isinstance(engine, str)
            or not engine
            or not ("A" <= engine[0] <= "Z" or "a" <= engine[0] <= "z")
            or not all(
                "A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9" or char == "_" for char in engine[1:]
            )
            or engine == "elaborated_rtl"
            for tier, engine in oracles.items()
        ):
            raise ValueError(f"{public}: performance_oracles must map tiers to concrete simulator names")
        prof["_performance_oracles"] = copy.deepcopy(oracles)
    if prof.get("capsule_policy", "compatibility") not in {"compatibility", "derived_only"}:
        raise ValueError(f"{public}: capsule_policy must be compatibility or derived_only")
    if prof.get("capsule_policy") == "derived_only" and (prof.get("capsules") or prof.get("sweeps")):
        raise ValueError(f"{public}: derived-only recipes cannot author capsule membership or sweeps")
    selected_spec = selected_software_spec_path(public, software_spec)
    if selected_spec is not None:
        declaration = load_software_spec(selected_spec, target=target)
        axes = numerical_datapath(declaration)
        datapath = prof.get("datapath") or {}
        if not isinstance(datapath, dict):
            raise ValueError(f"{public}: datapath must be a mapping")
        conflicting = sorted(key for key, value in axes.items() if key in datapath and datapath[key] != value)
        if conflicting:
            raise ValueError(f"{public}: recipe conflicts with selected software spec: {conflicting}")
        prof["datapath"] = {**datapath, **axes}
        prof["_software_spec_identity"] = software_spec_identity(selected_spec, declaration)
        prof["_software_spec_path"] = str(selected_spec)
    else:
        prof["_software_spec_identity"] = {"status": "unverified_legacy", "reason": "no selected software spec"}
    _normalize_public_capsules(prof, source=public)
    _merge_shared_perf(prof, source=public, performance_template=shared)
    _select_performance_withdrawals(prof, source=public)
    # SYNTHESIZED ENTRIES, appended after the hand-authored ones. They come from the target's own
    # derived conformance requirement and carry
    # the cell each was synthesized for in `source_reference`. Appended rather than prepended so
    # `expand_sweeps`' documented declaration-order semantics are untouched; its `seen` set already
    # raises on a duplicate name, and every synthesized name is `SY_`-prefixed, so a collision with a
    # hand-authored capsule is impossible rather than merely unlikely.
    if synth is not None and synth.is_file():
        doc = yaml.safe_load(synth.read_text(encoding="utf-8")) or {}
        prof["_synth_verification"] = verify_selected_synthesis(
            synth,
            conformance_spec=conformance_spec,
            recipe=public,
            descriptor=descriptor,
            software_spec=selected_spec,
            document=doc,
            diagnostic=diagnostic,
        )
        extra = list(doc.get("capsules") or ())
        for entry in extra:
            if entry.get("public_boundary"):
                from merlin.development.testgen.public_sources import boundaries as PB

                version = entry["public_boundary"]["schema"]
                if version == PB.SCHEMA_V3:
                    factory = (prof["_performance_template"].get("public_boundary_factories") or {}).get(version)
                    expected_factory = PB.TEMPLATE_V3
                else:
                    factory, expected_factory = (
                        prof["_performance_template"].get("public_boundary_factory"),
                        PB.TEMPLATE,
                    )
                if factory != expected_factory:
                    raise ValueError("public boundary synthesis requires its normal shared template factory")
                entry["loader"] = str(PB.selected_loader(entry, synth.parent))
            selection = entry.get("materialized_capture")
            if selection is not None:
                if not isinstance(selection, dict) or not isinstance(selection.get("path"), str):
                    raise ValueError(f"{synth}: materialized capture must declare a path and exact identities")
                resolved = dict(selection)
                for key in ("path", "loader_path", "profile_path"):
                    if key not in selection:
                        continue
                    if not isinstance(selection[key], str):
                        raise ValueError(f"{synth}: materialized {key} must be a path")
                    location = Path(selection[key])
                    if location.is_absolute() or ".." in location.parts:
                        raise ValueError(f"{synth}: materialized {key} must remain inside its selected profile bundle")
                    resolved[key] = str(synth.parent / location)
                entry["materialized_capture"] = resolved
        prof["_claim_model_evaluation"] = (doc.get("provenance") or {}).get("claim_model_evaluation")
        if extra:
            prof["capsules"] = list(prof.get("capsules") or []) + extra
    else:
        prof["_synth_verification"] = {"status": "absent"}
    # SOLVER-DERIVED ENTRIES. `verify.counterexamples` writes `<target>.smt.yaml` -- a counterexample
    # the deterministic fill cannot reach, found by the SMT layer at a shape it can still decide. It
    # was written to a filename nothing read: this chain was three hardcoded names, and
    # `counterexamples.py` documented a glob that does not exist ("load_profile already merges
    # profiles/<target>.*.yaml sidecars"). No `.smt.yaml` has ever been committed, which is consistent
    # with the path never having worked end to end.
    if smt is not None and smt.is_file():
        doc = yaml.safe_load(smt.read_text(encoding="utf-8")) or {}
        extra = list(doc.get("capsules") or ())
        if extra:
            prof["capsules"] = list(prof.get("capsules") or []) + extra

    if chain_profile is not None:
        prof["capsules"] = list(prof.get("capsules") or []) + chain_profile_entries(chain_profile)
    if include_holdouts and side is not None and side.is_file():
        held = yaml.safe_load(side.read_text(encoding="utf-8")) or {}
        misplaced = _target_local_perf_declarations(held)
        if misplaced:
            raise ValueError(
                f"{side} declares target-local performance template entries {misplaced}; "
                f"move them to the shared {shared}"
            )
        prof["capsules"] = list(prof.get("capsules") or []) + list(held.get("capsules") or [])
        # HOLDOUTS ARE GENERATED TOO, and without this line the sidecar's `sweeps:` block is read and
        # silently dropped -- the points expand for the disjointness gate, which reads the sidecar
        # itself, and then never become capsules. Generating them is the point: a hand-authored holdout
        # is written by someone who has just read the public profile, and an audit of this repo found
        # holdouts that were public capsules under another name, scoring memorisation as transfer. A
        # sweep states the OBLIGATION and lets the tile edge compute the points.
        prof["sweeps"] = list(prof.get("sweeps") or []) + list(held.get("sweeps") or [])
    if descriptor is not None:
        from merlin.development.testgen.generation.claim_boundary import (
            assert_no_claim_capsules,
            held_out_models,
        )
        from merlin.target.registry.descriptor_loading import load_target_experiment

        assert_no_claim_capsules(list(prof.get("capsules") or ()), held_out_models(load_target_experiment(descriptor)))
    return prof


def profile_targets(*, profiles_root: str | Path | None = None) -> list[str]:
    """Target profile stems, excluding shared templates, sidecars, and every DOTTED stem.

    ⚠️ A DOTTED STEM IS A SIDECAR, NEVER A TARGET. `Path.stem` strips only the LAST suffix, so
    `gemmini.synth.yaml` yields the stem `gemmini.synth` -- and this function returned it as a target
    beside `gemmini`. Measured: six real targets came back as twelve, and `main()` uses this list as
    the default when `--target` is absent, so a bare run generated six phantom corpora whose profiles
    are fragments. Excluding `.hidden` by filename alone was the same bug avoided one case at a time;
    the rule is now structural -- a target name has no dot in it, which is also what
    `check_phase_split` and `retire_hand_capsules` already assume when they enumerate targets.
    """
    return sorted(
        path.stem
        for path in _profiles_root(profiles_root).glob("*.yaml")
        if not path.name.startswith("_") and "." not in path.stem
    )


_DERIVED_AXES = ("memory_regime_reduction_depth", "resident_allocation_boundary")


def build_comparison_manifest(targets: list[str], *, profiles_root: str | Path | None = None) -> dict:
    """Group capsules that exercise the SAME op across targets into comparison sets, so a shared op (e.g.
    rmsnorm/gelu/gemv_batched) can be compared across each target's own precision (e.g. MXFP8 on one target vs
    FP8-E4M3 or fp16 on another). Keyed by ``comparison_group`` when the profile declares one, else by op."""
    if not targets:
        raise ValueError("comparison requires explicitly selected public profiles")
    return compare_public_profiles(
        (t, load_profile(t, include_holdouts=False, profiles_root=profiles_root)) for t in targets
    )


def compare_public_profiles(profiles) -> dict:
    """Compare explicitly loaded public profiles without discovering or reading files."""
    groups: dict[str, list[dict]] = {}
    for t, prof in profiles:
        for e in prof["capsules"]:
            if e.get("kind") == "model" or e.get("op") == "model":
                continue
            key = e.get("comparison_group") or e.get("op", "unknown")
            groups.setdefault(key, []).append(
                {
                    "target": t,
                    "name": e["name"],
                    "dtype": e.get("operand_dtype", prof.get("datapath", {}).get("operand_dtype", "")),
                    "label": e.get("label", "public"),
                }
            )
    # a comparison set is only interesting when >1 target covers the op
    cross = {k: v for k, v in sorted(groups.items()) if len({m["target"] for m in v}) > 1}
    return {
        "comparison_sets": cross,
        "note": "each set is one op exercised across multiple targets in each target's own precision; "
        "same inner op name across targets makes target-vs-target numerics directly comparable",
    }


def write_comparison_manifest(targets: list[str], *, profiles_root: str | Path | None = None) -> Path:
    from merlin.common.paths import artifacts_dir

    manifest = build_comparison_manifest(targets, profiles_root=profiles_root)
    out = Path(artifacts_dir()) / "compare" / "capsule_comparison_manifest.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return out


# ---- declarations ------------------------------------------------------------------------------
# Explicit derivation inputs declared by experiments, without reading private sidecars.


@dataclass(frozen=True)
class DerivationDeclaration:
    definition: Path
    id: str
    target: str
    profile: str
    descriptor: Path
    recipe: Path
    performance_template: Path
    conformance_spec: Path | None
    synth_profile: Path | None
    smt_profile: Path | None
    hidden_profile: Path | None

    def profile_inputs(self) -> dict[str, Path | None]:
        return {
            name: getattr(self, name)
            for name in (
                "recipe",
                "performance_template",
                "conformance_spec",
                "synth_profile",
                "smt_profile",
                "hidden_profile",
            )
        }


def _declaration(spec: ExperimentSpec) -> DerivationDeclaration:
    phase = engine_definitions(spec.document).get("0")
    if spec.document.get("kind") == "template" or not phase or phase["adapter"] != "capsule_derivation":
        raise SpecError(f"{spec.path}: not a concrete capsule derivation definition")
    config = phase["config"]
    for name in ("descriptor", "recipe", "performance_template"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise SpecError(f"{spec.path}: explicit phase-0 {name} is required")
    # Declarations are lexical inputs, not an admission check. Path.resolve() probes
    # the filesystem (including private holdouts) and erases dangling aliases.
    # The consuming loader/verifier checks the inputs it is authorized to open.
    paths = {
        name: Path(os.path.abspath(spec.path.parent / Path(config[name]).expanduser())) if config.get(name) else None
        for name in (
            "descriptor",
            "recipe",
            "performance_template",
            "conformance_spec",
            "synth_profile",
            "smt_profile",
            "hidden_profile",
        )
    }
    return DerivationDeclaration(spec.path, spec.id, spec.target, config.get("profile", spec.target), **paths)


def from_definition(path: str | Path) -> DerivationDeclaration:
    return _declaration(load_spec(path))


def all_declarations(*, catalog_path: Path | None = None) -> tuple[DerivationDeclaration, ...]:
    declarations = []
    for name, path in catalog(catalog_path).items():
        spec = load_spec(path)
        if spec.id != name:
            raise SpecError(f"catalog id {name!r} does not match definition id {spec.id!r}")
        phase = engine_definitions(spec.document).get("0")
        if spec.document.get("kind") == "template" or not phase or phase["adapter"] != "capsule_derivation":
            continue
        declarations.append(_declaration(spec))
    return tuple(declarations)


def for_target(selector: str, *, catalog_path: Path | None = None) -> DerivationDeclaration:
    matches = [item for item in all_declarations(catalog_path=catalog_path) if selector in (item.target, item.profile)]
    if len(matches) != 1:
        raise SpecError(f"phase-0 selector {selector!r} requires one declaration; found {len(matches)}")
    return matches[0]
