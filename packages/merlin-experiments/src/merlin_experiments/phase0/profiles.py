"""Host-owned Phase 0 profiles implementation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.paths import repo_root  # noqa: E402
from merlin.perf.profile import TRAITS  # noqa: E402
from merlin.runtime.backends.base import EXECUTION_CAPABILITIES  # noqa: E402

from .provenance import _document_digest

_SYNTHESIS_INPUT_KEYS = frozenset({"conformance_spec_sha256", "recipe_sha256", "workload_spec_sha256"})


def application_inventory_path(conformance_spec: str | Path) -> Path | None:
    """Resolve only the requirement's adjacent generated sidecar, never an arbitrary path."""
    spec = Path(conformance_spec)
    document = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{spec}: conformance spec must be a mapping")
    demands = document.get("application_demands") or {}
    if not isinstance(demands, dict):
        raise ValueError(f"{spec}: application_demands must be a mapping")
    name = demands.get("sidecar")
    if name is None:
        return None
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or Path(name).name != name
        or "\\" in name
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(f"{spec}: application-demand sidecar must be one adjacent basename")
    path = spec.parent / name
    if path.is_symlink():
        raise ValueError(f"{spec}: application-demand sidecar may not be a symlink")
    return path


def synthesis_input_identity(*, conformance_spec: str | Path, recipe: str | Path, descriptor: str | Path) -> dict:
    """The exact requirement and authored software inputs a synthesis used.

    The requirement and recipe are byte identities. ``workload_spec`` is a parsed
    declaration inside the descriptor, so its canonical document digest avoids
    invalidating synthesis for unrelated descriptor comments or hardware setup.
    """
    source = yaml.safe_load(Path(descriptor).read_text(encoding="utf-8")) or {}
    if not isinstance(source, dict) or not isinstance(source.get("workload_spec") or {}, dict):
        raise ValueError(f"{descriptor}: workload_spec must be a mapping")
    return {
        "conformance_spec_sha256": hashlib.sha256(Path(conformance_spec).read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(recipe).read_bytes()).hexdigest(),
        "workload_spec_sha256": _document_digest(source.get("workload_spec") or {}),
    }


def verify_selected_synthesis(
    synth_profile: str | Path | None,
    *,
    conformance_spec: str | Path | None = None,
    recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    document: dict | None = None,
) -> dict:
    """Verify a reviewed synth sidecar against its selected, frozen inputs.

    Historical profiles predate the identity block. They remain inspectable as
    diagnostics, but are explicitly *unverified*, never upgraded by an ambient
    checkout reference. Verified execution needs newly frozen selected inputs.
    """
    if synth_profile is None or not Path(synth_profile).is_file():
        return {"status": "absent"}
    selected = document if document is not None else yaml.safe_load(Path(synth_profile).read_text(encoding="utf-8"))
    if not isinstance(selected, dict):
        raise ValueError(f"{synth_profile}: synthesized profile must be a mapping")
    provenance = selected.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError(f"{synth_profile}: provenance must be a mapping")
    identity = provenance.get("selected_inputs")
    if identity is None:
        return {
            "status": "unverified_legacy",
            "reason": "selected synthesis predates conformance/recipe/workload input digests; regenerate and review it",
        }
    if (
        not isinstance(identity, dict)
        or set(identity) != _SYNTHESIS_INPUT_KEYS
        or any(
            not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
            for value in identity.values()
        )
    ):
        raise ValueError(f"{synth_profile}: selected_inputs must contain three SHA-256 digests")
    if conformance_spec is None or recipe is None or descriptor is None:
        raise ValueError(
            f"{synth_profile}: verified synthesis requires explicit conformance_spec, recipe, and descriptor inputs"
        )
    expected = synthesis_input_identity(conformance_spec=conformance_spec, recipe=recipe, descriptor=descriptor)
    changed = sorted(key for key in _SYNTHESIS_INPUT_KEYS if identity[key] != expected[key])
    if changed:
        raise ValueError(
            f"{synth_profile}: stale selected synthesis ({', '.join(changed)} changed); "
            "regenerate a versioned sidecar and review/select it explicitly"
        )
    descriptor_doc = yaml.safe_load(Path(descriptor).read_text(encoding="utf-8")) or {}
    if not isinstance(descriptor_doc, dict):
        raise ValueError(f"{descriptor}: descriptor must be a mapping")
    applications = (descriptor_doc.get("workload_spec") or {}).get("applications") or {}
    spec_doc = yaml.safe_load(Path(conformance_spec).read_text(encoding="utf-8")) or {}
    if not isinstance(spec_doc, dict):
        raise ValueError(f"{conformance_spec}: conformance spec must be a mapping")
    demands = spec_doc.get("application_demands") or {}
    if not isinstance(demands, dict):
        raise ValueError(f"{conformance_spec}: application_demands must be a mapping")
    sidecar = application_inventory_path(conformance_spec)
    if applications and sidecar is None:
        raise ValueError(f"{conformance_spec}: declared applications require a selected detailed-demand sidecar")
    if sidecar is not None:
        if not sidecar.is_file():
            raise ValueError(f"{sidecar}: selected detailed-demand sidecar is missing")
        try:
            detailed = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{sidecar}: invalid detailed-demand JSON: {exc}") from exc
        if not isinstance(detailed, dict):
            raise ValueError(f"{sidecar}: detailed-demand inventory must be a mapping")
        actual = hashlib.sha256(json.dumps(detailed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if actual != demands.get("full_inventory_sha256"):
            raise ValueError(f"{sidecar}: detailed-demand content differs from selected conformance spec")
        if applications and (
            detailed.get("status") != "inventoried" or detailed.get("coverage_status") != "unverified"
        ):
            raise ValueError(f"{sidecar}: detailed-demand inventory must be inventoried with coverage unverified")
    if applications and demands.get("status") != "inventoried":
        raise ValueError(
            f"{conformance_spec}: declared applications require an inventoried operation-demand requirement; "
            "regenerate and review the conformance spec and selected synthesis"
        )
    coverage_status = demands.get("coverage_status", "unverified")
    if applications and coverage_status != "unverified":
        raise ValueError(f"{conformance_spec}: application-demand coverage must remain explicitly unverified")
    return {"status": "verified", "selected_inputs": dict(identity), "application_coverage_status": coverage_status}


_PERFORMANCE_FIELDS = frozenset(
    {
        "level",
        "family",
        "lever",
        "member_class",
        "comparand",
        "falsifier",
        "gate",
        "regime",
        "emitter",
        "cost",
    }
)
_PERFORMANCE_CLAIMS = frozenset({"RECOVERS", "PREDICTS", "DIFFERENTIAL"})
#: WHAT A MEMBER IS FOR, which is not the same question as what it CLAIMS. A family declares a claim
#: (does the law hold?); a member additionally has a job in the search, and the two had been conflated
#: with a measurable cost.
#:
#: ``LAW``
#:     The member exists to make a family's claim decidable -- it is a point in a fitted cohort. Its
#:     cohort is EXACT (the analyzers refuse a partial one), so it can never be sampled; but its cycle
#:     count is a property of the machine, not of how good today's candidate is, so re-measuring it on
#:     every candidate buys nothing. Measured on a real sweep: the parallel-extents family is 16 of 38
#:     members and 33% of the 36.9-minute serial sweep, for members worth under 2% of the objective.
#: ``OBJECTIVE``
#:     The member exists to BE optimised. Its cycles are the thing the search minimises, so it is
#:     re-measured every candidate and it is what the total is summed over.
#: ``REFERENCE``
#:     The member is measured against a shipped hand-written implementation of the same shape, which
#:     is the only place a RECOVERS claim has a denominator. Reported beside the objective, never
#:     folded into it -- a fraction-of-reference and a sum-of-cycles are different quantities.
#:
#: Declared rather than inferred from the claim, because the mapping is not one-to-one: a DIFFERENTIAL
#: family can be either, and inferring it would silently reclassify a family when its claim changed.
_MEMBER_CLASSES = frozenset({"LAW", "OBJECTIVE", "REFERENCE"})
#: The optimization LEVELS a performance family may declare. DOCUMENTED rather than enforced: several
#: test fixtures declare synthetic levels to prove the validator is generic, so closing this set is a
#: change to make deliberately alongside those fixtures rather than as a side effect. What it is for:
#:
#: The ladder is tile -> layer -> inter-layer -> global, and two rungs were missing from it. `L4_boundary`
#: is the host/accelerator seam -- the H->A break-even and the cost of an A->H->A island -- which is
#: where a placement decision is actually made and paid for. `L6_global` is the whole-program decision:
#: quantization, packing, encoding, layout propagation. Declaring them here does not create the
#: families; it makes them nameable, so a family that needs one is not forced to file under a level
#: that means something else. `merlin.perf.profile` now carries the canonical trait an L6 family needs
#: (`multiple_operand_encodings`), which was the part that could not be added in YAML at all.
_PERFORMANCE_LEVELS = frozenset(
    {
        "L1_tile",
        "L1_separation_floor",
        "L2_intra_layer",
        "L3_inter_layer",
        "L4_boundary",
        "L5_fusion",
        "L6_global",
    }
)
_PERFORMANCE_NESTED_FIELDS = {
    "comparand": frozenset({"kind", "against", "cancels", "demand_equal"}),
    "falsifier": frozenset({"observation", "fires_when", "negative_control"}),
    "gate": frozenset({"traits", "instrument", "capacity", "on_missing"}),
    "regime": frozenset({"separation", "layout"}),
    "emitter": frozenset({"status", "entry", "knobs"}),
    "cost": frozenset({"tier", "runs", "projected_cycles", "basis"}),
}

# Public recipes describe workload intent. These directory categories are a
# presentation of the existing capsule kind, not target-authored facts.
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


def _validate_performance_block(block, *, owner: str) -> dict:
    """Validate the claim-bearing contract before a family can be admitted.

    Performance metadata is consumed later than corpus generation, so a partial
    block otherwise succeeds here and turns into an unmeasurable family only
    after an expensive run.  All claim, comparison, falsifier, applicability,
    regime, emitter, and cost fields therefore fail closed at profile load.
    """
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
        for key in required - {"knobs"}:
            nested = value[key]
            if (
                nested is None
                or (isinstance(nested, str) and not nested.strip())
                or (isinstance(nested, (list, tuple, dict)) and not nested)
            ):
                raise ValueError(f"{owner}: performance.{field}.{key} must be non-empty")
    gate = block["gate"]
    names = gate["traits"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
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
    sweeps = list(shared.get("sweeps") or [])
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
        performance = _validate_performance_block(item.get("performance"), owner=f"blocked performance family {family}")
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


def validate_profile_inputs(
    *,
    profiles_root=None,
    recipe=None,
    performance_template=None,
    conformance_spec=None,
    synth_profile=None,
    smt_profile=None,
    hidden_profile=None,
) -> None:
    """Reject mixed ownership modes without touching any input, especially private sidecars."""
    if recipe is not None:
        if profiles_root is not None:
            raise ValueError("recipe and profiles_root are mutually exclusive")
        if performance_template is None:
            raise ValueError("explicit recipe requires performance_template")
    elif any(
        value is not None
        for value in (performance_template, conformance_spec, synth_profile, smt_profile, hidden_profile)
    ):
        raise ValueError("explicit performance_template and sidecar paths require recipe")


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
) -> dict:
    """The target's functional profile plus shared perf and the private holdout sidecar.

    The holdout spec (op + dtype + exact shape) is an answer, not a contract: the tracked profile lives
    inside the ``merlin/contract/`` tree every arm is granted read-only, so a holdout declared there is
    readable by the agent under test. It therefore lives in ``profiles/<target>.hidden.yaml``, which is
    gitignored and masked by :mod:`merlin.targetgen.sandbox.answer_surfaces`. When the sidecar is absent
    -- a public clone, or a sandbox where it is masked -- this returns the public profile unchanged and
    the run simply emits no hidden capsules, which is the correct behaviour rather than an error.

    Explicit ``recipe`` mode reads only the named files; no sibling is discovered. Its shared
    ``performance_template`` is mandatory. Optional sidecars may be declared but absent. The legacy
    ``profiles_root`` directory interface retains its established basename conventions.

    ``include_holdouts=False`` for any caller whose OUTPUT is public: it does not read or stat the
    hidden path. A published artifact enumerating the holdouts would leak them just as the profile did.
    """
    validate_profile_inputs(
        profiles_root=profiles_root,
        recipe=recipe,
        performance_template=performance_template,
        conformance_spec=conformance_spec,
        synth_profile=synth_profile,
        smt_profile=smt_profile,
        hidden_profile=hidden_profile,
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
    _normalize_public_capsules(prof, source=public)
    _merge_shared_perf(prof, source=public, performance_template=shared)
    # SYNTHESIZED ENTRIES, appended after the hand-authored ones. They come from the target's own
    # derived conformance requirement (build_tools/scripts/synth_capsule_corpus.py --write) and carry
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
            document=doc,
        )
        extra = list(doc.get("capsules") or ())
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


#: Axis DERIVATIONS a sweep may name in place of writing extents down. A derived axis exists for the
#: same reason ``encoding<N>`` does: the points the family needs are a function of the target's own
#: derived facts, and any literal the one shared template wrote down would describe a single machine.
#:
#: ``memory_regime_reduction_depth`` is the residency axis. A performance model is only valid in the
#: regimes it was fitted in, and a rate cannot be separated from a fixed fill/drain intercept from one
#: point -- so this axis asks the target's OWN operand store for several reduction depths inside each
#: residency band, spread across the band rather than piled against its edge. Measured on the
#: interlocked target here before it existed: the whole `_perf` corpus sat in ``fits_double`` while the
#: overwhelming majority of contraction regions in the real captures land in ``spills``, so every
#: coefficient was fitted where almost no real work lands.
_DERIVED_AXES = ("memory_regime_reduction_depth",)


def build_comparison_manifest(targets: list[str], *, profiles_root: str | Path | None = None) -> dict:
    """Group capsules that exercise the SAME op across targets into comparison sets, so a shared op (e.g.
    rmsnorm/gelu/gemv_batched) can be compared across each target's own precision (MXFP8 on mx vs FP8-E4M3
    on atlas vs fp16 on radiance). Keyed by ``comparison_group`` when the profile declares one, else by op."""
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
