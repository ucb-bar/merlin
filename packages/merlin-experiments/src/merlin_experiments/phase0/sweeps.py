"""Host-owned Phase 0 sweeps implementation."""

from __future__ import annotations

import copy

from merlin.perf import workload_gen as WG  # noqa: E402
from merlin.perf.profile import TRAITS, derive_profile  # noqa: E402
from merlin.runtime.backends.base import EXECUTION_CAPABILITIES, execution_capability_facts  # noqa: E402
from merlin.targetgen import corpus_spec as CS  # noqa: E402

from .profiles import _DERIVED_AXES, _comparison_roles, _validate_performance_block
from .provenance import _document_digest

#: Extent tokens a sweep axis may use, resolved against the binding's tile edge.
#: Spelled the way `kernels/opu_corpus.py` already spells tile-relative extents
#: ("tile", "tile/2"), so one convention covers both corpora.
_TILE_TOKEN = "tile"

#: A sweep may not silently become a thousand capsules. Exceeding this raises;
#: it is never truncated, because a corpus that quietly dropped points reads as
#: "covered everything" when it did not.
_MAX_SWEEP_CAPSULES = 128


#: Entry keys whose value is a SHAPE EXTENT, and may therefore be written tile-relative.
_EXTENT_KEYS = ("M", "K", "N", "H", "Skv", "Dv", "B")


def _resolve_flat_extents(entry: dict, binding) -> dict:
    """Resolve tile-relative extent tokens on an entry and its matmul commands.

    ``resolve_extent`` was reachable only from sweep axes, so a flat entry had to spell its shape as
    integers -- which bakes one target's geometry into a file that is supposed to describe a shape
    RELATIVE to whatever edge the hardware has. Synthesized entries are written ``tile`` / ``tile-1``
    precisely so the same entry means the same thing on a target with a different edge, and this is
    where that promise is kept.

    Integer extents and absent keys are untouched. A resident-reuse entry can
    also declare a separate M for each matmul; those commands share the same
    target-derived tile edge as the top-level K and N.
    """
    tile = getattr(binding, "tile_dim", None)
    if not tile:
        return entry
    out = None
    for key in _EXTENT_KEYS:
        value = entry.get(key)
        if not isinstance(value, str):
            continue
        if out is None:
            out = dict(entry)
        out[key] = resolve_extent(value, int(tile))
    matmuls = (out or entry).get("matmuls")
    if isinstance(matmuls, list) and any(
        isinstance(mm, dict) and isinstance(mm.get(key), str) for mm in matmuls for key in _EXTENT_KEYS
    ):
        if out is None:
            out = dict(entry)
        out["matmuls"] = [
            {
                key: (resolve_extent(value, int(tile)) if key in _EXTENT_KEYS and isinstance(value, str) else value)
                for key, value in mm.items()
            }
            if isinstance(mm, dict)
            else mm
            for mm in matmuls
        ]
    resolved = out if out is not None else entry
    # A POOLING EPILOGUE NEEDS ITS SPATIAL SHAPE, and this is the first point at which it can be had:
    # the entry's rows are known only after the tile-relative tokens resolve. A synthesized entry
    # declares the pool WINDOW (a 2x2 with stride 2 is the shape every pooling datapath has) and leaves
    # the input geometry to be derived, because the number of rows to factor is the target's tile edge.
    if "maxpool" in [str(x) for x in (resolved.get("epilogue") or ())] and not resolved.get("pool_in_dims"):
        rows = int(resolved.get("M") or 0)
        side = int(rows**0.5)
        if side >= 2 and side * side == rows:
            resolved = {**resolved, "pool_in_dims": [side, side]}
        else:
            # NOT a square number of rows: there is no H x W the rows can mean. Left absent so the
            # builder refuses with its own message rather than this inventing a geometry that silently
            # pools over the wrong axis.
            pass
    return resolved


def target_encodings(target: str) -> list[str]:
    """The operand encodings this target can compute a contraction in, in capsule spelling.

    The axis an ``L6_global`` (encoding / packing / layout) family sweeps. Derived from the same place
    the gate trait ``multiple_operand_encodings`` reads -- the contraction family's declared dtypes --
    so the family's members and the gate that admits it cannot disagree about how many encodings exist.

    Sorted, because the corpus must regenerate byte-identically and a capability map is a mapping.
    Empty when the map cannot be resolved: a family that needs a choice then has none to make, and the
    gate refuses it with evidence rather than this inventing one.
    """
    try:
        from merlin.targetgen.conformance import capsule_dtype
        from merlin.targetgen.eligibility import capability_map_for_target

        cap = (capability_map_for_target(target) or {}).get("contraction")
    except Exception:  # noqa: BLE001 -- an unresolvable map is no choice
        return []
    out = []
    for d in sorted(getattr(cap, "dtypes", ()) or ()):
        try:
            out.append(capsule_dtype(str(d)))
        except Exception:  # noqa: BLE001 -- keep an unmappable token visible
            out.append(str(d))
    return list(dict.fromkeys(out))


def resolve_encoding(token, encodings: list[str]) -> str:
    """Resolve a sweep ENCODING token against the target's own encodings.

    ``encoding<N>`` selects the Nth encoding the target declares for a contraction, exactly as
    ``resolve_extent`` selects a geometry relative to the tile edge -- and for the same reason: a
    profile must never write down a dtype. `_perf.yaml` is shared by every target, so a family that
    named ``int8`` and ``bf16`` would be describing one machine and silently mis-describing the rest.

    An index past the end RAISES. That case means the family was admitted on a target with fewer
    encodings than it compares, which the gate is supposed to have refused; resolving it to the last
    one instead would emit a comparison group whose two members are the same encoding, and a
    differential over identical work always reads as "no effect".
    """
    if not isinstance(token, str):
        raise ValueError(f"sweep encoding {token!r} is not an encoding token")
    head, digits = token.strip(), ""
    prefix = "encoding"
    if not head.startswith(prefix) or not head[len(prefix) :].isdigit():
        raise ValueError(f"sweep encoding {token!r} is not of the form 'encoding<N>'")
    digits = head[len(prefix) :]
    idx = int(digits)
    if idx >= len(encodings):
        raise ValueError(
            f"sweep encoding {token!r} asks for encoding {idx} of a target declaring "
            f"{len(encodings)} ({encodings}); the family's gate should have refused it here rather "
            f"than letting two members resolve to the same encoding"
        )
    return encodings[idx]


def resolve_extent(token, tile: int) -> int:
    """Resolve a sweep extent token against *tile* (the binding's tile edge).

    Accepts a plain int, or a tile-relative expression of the form
    ``[<mult>*]tile[+<n>|-<n>|/<div>]`` — e.g. ``tile``, ``tile-1``, ``tile+1``,
    ``tile/2``, ``2*tile``, ``2*tile-1``. Parsed structurally (``partition``), not
    by pattern-matching, so an unsupported spelling raises instead of silently
    resolving to something plausible.

    The point of tile-relative tokens is that a profile never hardcodes a
    geometry: the same sweep produces edge cases for a 16-wide command-buffer
    tile and for a 64-wide VLMAX tile without being edited.
    """
    if isinstance(token, bool):
        raise ValueError(f"sweep extent {token!r} is a bool, not an extent")
    if isinstance(token, int):
        if token < 1:
            raise ValueError(f"sweep extent {token!r} must be >= 1")
        return token
    if not isinstance(token, str):
        raise ValueError(f"sweep extent {token!r} is neither an int nor a tile expression")

    text = token.strip()
    mult_text, star, rest = text.partition("*")
    if star:
        mult = int(mult_text.strip())
        rest = rest.strip()
    else:
        mult, rest = 1, text

    for op in ("+", "-", "/"):
        head, found, tail = rest.partition(op)
        if found:
            if head.strip() != _TILE_TOKEN:
                raise ValueError(f"sweep extent {token!r}: expected {_TILE_TOKEN!r} before {op!r}")
            operand = int(tail.strip())
            base = mult * tile
            if op == "+":
                value = base + operand
            elif op == "-":
                value = base - operand
            else:
                if operand == 0:
                    raise ValueError(f"sweep extent {token!r}: division by zero")
                value = base // operand
            break
    else:
        if rest.strip() != _TILE_TOKEN:
            raise ValueError(f"sweep extent {token!r} is not a recognized tile expression")
        value = mult * tile

    if value < 1:
        raise ValueError(f"sweep extent {token!r} resolves to {value} at tile={tile}; must be >= 1")
    return value


class AxisDerivationUnavailable(ValueError):
    """The declaration is well formed but THIS target cannot host the axis it derives.

    Kept distinct from a plain ``ValueError`` on purpose. A malformed declaration is an authoring
    error and must stop generation; a target that declares no operand store simply cannot be given a
    residency ladder, and that is the same kind of answer a refuted trait gate gives -- a skip with
    evidence. Conflating them meant one such target aborted `expand_sweeps` outright and took every
    OTHER family's members down with it, which reads downstream as "the corpus has no perf families"
    rather than "this family does not apply here".
    """


def _memory_regime_axis(
    spec: dict, *, owner: str, target: str, tile: int, dtype: str | None, fixed: dict[str, list[int]]
) -> tuple[list[int], dict, dict]:
    """``(K extents, {extent: regime}, derivation record)`` for a residency-banded reduction sweep.

    The parallel extents are read from the sweep's OWN ``M``/``N`` axes so the two cannot disagree
    about the shape whose residency is being banded; each must be single-valued, because a band is a
    property of one shape family and crossing it with a second parallel extent would put two different
    residency ladders under one name.
    """
    from merlin.targetgen import memory_regime as MR

    regimes = [str(r) for r in (spec.get("regimes") or MR.ORDER)]
    points_per_regime = int(spec.get("points_per_regime", 2))
    if points_per_regime < 2:
        raise ValueError(
            f"{owner}: points_per_regime={points_per_regime} -- a rate and a fixed intercept are two "
            "parameters and cannot be separated by fewer than two points in the same regime"
        )
    ceiling = float(spec.get("spills_max_fraction_of_capacity", 2.0))
    tiles = {}
    for axis in ("M", "N"):
        points = fixed.get(axis) or []
        if len(set(points)) != 1:
            raise ValueError(
                f"{owner}: a memory-regime reduction axis needs a single-valued {axis} axis "
                f"(got {points}); the band is a property of one parallel shape"
            )
        extent = int(points[0])
        if extent % tile:
            raise ValueError(f"{owner}: {axis}={extent} is not a whole number of {tile}-wide tiles")
        tiles[axis] = extent // tile
    record = MR.reduction_depth_regimes(
        target,
        regimes,
        tile_dim=tile,
        dtype=dtype,
        m_tiles=tiles["M"],
        n_tiles=tiles["N"],
        points_per_regime=points_per_regime,
        spills_max_fraction=ceiling,
    )
    values: list[int] = []
    labels: dict[int, str] = {}
    for regime in regimes:
        got = (record["by_regime"] or {}).get(regime) or {}
        for point in got.get("points") or []:
            k = int(point["K"])
            if k in labels:  # bands cannot overlap; check, do not trust
                raise ValueError(f"{owner}: reduction depth {k} was assigned to both {labels[k]!r} and {regime!r}")
            labels[k] = regime
            values.append(k)
    if not values:
        raise AxisDerivationUnavailable(
            f"{owner}: no regime in {regimes} is reachable on {target!r}: "
            + "; ".join(f"{r}: {(record['by_regime'].get(r) or {}).get('unreachable')}" for r in regimes)
        )
    return values, labels, record


def _resolve_derived_axis(
    spec: dict, *, owner: str, axis: str, target: str, tile: int, dtype: str | None, fixed: dict[str, list[int]]
):
    """Dispatch one ``axes: {<name>: {derive: ...}}`` declaration to its derivation."""
    kind = str(spec.get("derive") or "")
    if kind not in _DERIVED_AXES:
        raise ValueError(
            f"{owner}: axis {axis!r} declares derive={kind!r}; known derivations are {list(_DERIVED_AXES)}"
        )
    if kind == "memory_regime_reduction_depth":
        if axis != "K":
            raise ValueError(
                f"{owner}: {kind!r} derives the REDUCTION depth, so it must be declared on K, not {axis!r}"
            )
        return _memory_regime_axis(spec, owner=owner, target=target, tile=tile, dtype=dtype, fixed=fixed)
    raise ValueError(f"{owner}: axis derivation {kind!r} has no resolver")  # unreachable; fail closed


def _performance_facts(target: str) -> dict:
    """Canonical hardware-trait and backend-execution facts, derived once for this target."""
    profile = derive_profile(target).to_dict()
    execution = execution_capability_facts(target)
    document = {"target_profile": profile, "execution_capabilities": execution}
    return {
        "target": target,
        "traits": profile["traits"],
        "execution_capabilities": execution,
        "target_profile_sha256": _document_digest(profile),
        "execution_capabilities_sha256": _document_digest(execution),
        "sha256": _document_digest(document),
    }


def evaluate_gate(gate: dict, trait_facts: dict | None) -> tuple[bool, dict]:
    """Require every hardware trait and software execution capability to be true with evidence.

    The structured decision deliberately carries False and None separately. A
    refuted capability makes a family inapplicable; an unestablished one means
    the instrument/fact coverage is incomplete. Neither is admitted, and
    neither is silently reduced to Python truthiness.
    """
    if not isinstance(gate, dict):
        raise ValueError("performance gate must be a mapping")
    if "requires" in gate:
        raise ValueError("performance gate.requires is not accepted; use canonical gate.traits")
    names = gate.get("traits")
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("performance gate.traits must be a non-empty list")
    unknown = sorted(set(names) - set(TRAITS))
    if unknown:
        raise ValueError(f"unknown performance trait(s) {unknown}; canonical traits are {list(TRAITS)}")
    execution_names = gate.get("execution_capabilities", [])
    if not isinstance(execution_names, list) or any(not isinstance(name, str) or not name for name in execution_names):
        raise ValueError("performance gate.execution_capabilities must be a list")
    unknown_execution = sorted(set(execution_names) - set(EXECUTION_CAPABILITIES))
    if unknown_execution:
        raise ValueError(
            f"unknown execution capability(s) {unknown_execution}; canonical capabilities are "
            f"{list(EXECUTION_CAPABILITIES)}"
        )
    facts = (trait_facts or {}).get("traits", trait_facts or {})
    selected: dict[str, dict] = {}
    for name in names:
        raw = facts.get(name)
        if not isinstance(raw, dict):
            raw = {
                "satisfied": None,
                "tier": "not_established",
                "evidence": "canonical trait fact was not supplied",
                "missing": ["derive_profile(target) result for this trait"],
            }
        selected[name] = {
            "satisfied": raw.get("satisfied") if raw.get("satisfied") in (True, False) else None,
            "tier": raw.get("tier") or "not_established",
            "evidence": raw.get("evidence") or "no evidence recorded",
            "missing": list(raw.get("missing") or []),
        }
    refuted = [name for name, fact in selected.items() if fact["satisfied"] is False]
    unestablished = [name for name, fact in selected.items() if fact["satisfied"] is None]
    satisfied = [name for name, fact in selected.items() if fact["satisfied"] is True]
    execution_source = (trait_facts or {}).get("execution_capabilities", {})
    selected_execution: dict[str, dict] = {}
    for name in execution_names:
        raw = execution_source.get(name) if isinstance(execution_source, dict) else None
        if not isinstance(raw, dict):
            raw = {
                "satisfied": None,
                "tier": "not_established",
                "evidence": "canonical execution capability fact was not supplied",
                "missing": ["execution_capability_facts(target) result for this capability"],
            }
        selected_execution[name] = {
            "satisfied": raw.get("satisfied") if raw.get("satisfied") in (True, False) else None,
            "tier": raw.get("tier") or "not_established",
            "evidence": raw.get("evidence") or "no evidence recorded",
            "missing": list(raw.get("missing") or []),
        }
    execution_refuted = [name for name, fact in selected_execution.items() if fact["satisfied"] is False]
    execution_unestablished = [name for name, fact in selected_execution.items() if fact["satisfied"] is None]
    execution_satisfied = [name for name, fact in selected_execution.items() if fact["satisfied"] is True]
    any_refuted = bool(refuted or execution_refuted)
    any_unknown = bool(unestablished or execution_unestablished)
    outcome = "refuted" if any_refuted else ("unestablished" if any_unknown else "satisfied")
    decision = {
        "outcome": outcome,
        "required_traits": list(names),
        "satisfied": satisfied,
        "refuted": refuted,
        "unestablished": unestablished,
        "facts": selected,
        "required_execution_capabilities": list(execution_names),
        "satisfied_execution_capabilities": execution_satisfied,
        "refuted_execution_capabilities": execution_refuted,
        "unestablished_execution_capabilities": execution_unestablished,
        "execution_capability_facts": selected_execution,
    }
    return outcome == "satisfied", decision


def _accum_for_encoding(target: str, operand: str, fallback: str | None) -> str:
    """The accumulate format ``target`` declares for a contraction whose operands are ``operand``.

    ASKED OF THE ROUTER, not re-derived. `routing._legal_on` is the one predicate that answers "what
    does this unit accumulate this contraction in", and it answers by taking the FIRST declared
    accumulate rule that matches -- which is what every other path in the repo computes for the same
    contraction. A second derivation here would be a second answer to one question, and on a target
    declaring several rules for one operand (atlas declares both bf16 and f32 for fp8_e4m3, which is a
    real capability rather than an ambiguity) the two would disagree.

    Falls back to the corpus binding's accumulator only when no unit accepts the operand at all, and
    raises when there is no fallback either: a member whose accumulator is unknown cannot be built,
    and guessing one emits a capsule that measures arithmetic the target does not perform.
    """
    from merlin.targetgen import compute_units as _cu
    from merlin.targetgen.routing import OpDemand, _legal_on  # noqa: PLC2701 -- one predicate
    from merlin.targetgen.target_registry import load_contract

    demand = OpDemand(op="matmul", in_fmt=operand, weight_fmt=operand, site="encoding_probe")
    try:
        units = list(_cu.compute_units(load_contract(target)))
    except Exception:  # noqa: BLE001 -- an unreadable contract is no answer
        units = []
    for unit in (_cu.effective(u, units) for u in units):
        legal, acc = _legal_on(unit, demand)
        if legal and acc:
            return str(acc)
    if fallback:
        return str(fallback)
    raise ValueError(
        f"no unit of {target!r} declares an accumulate rule for operand {operand!r}, and no corpus "
        f"binding accumulator is available; an encoding member cannot be built without knowing which "
        f"datapath it accumulates in"
    )


def _resolve_target_oracle_evidence(performance: dict, target: str) -> dict:
    """Resolve ``$target_oracle:<tier>`` evidence placeholders from the target's own oracle route.

    The shared profile must not name one target's simulator binary.  At generation time the target is
    known, so its contract supplies ordinary tiers and its RTL-engine policy supplies the concrete L3
    implementation selected for an elaborated-RTL fidelity.  The resolved names are frozen into the
    capsule acceptance contract together with the placeholders they came from.
    """
    acceptance = performance.get("acceptance")
    evidence = acceptance.get("evidence") if isinstance(acceptance, dict) else None
    if not isinstance(evidence, dict):
        return performance
    prefix = "$target_oracle:"
    pending = {key: value for key, value in evidence.items() if isinstance(value, str) and value.startswith(prefix)}
    if not pending:
        return performance
    from merlin.targetgen.target_experiment import load_capability_manifest

    contract = load_capability_manifest(target).contract
    declared = (contract.get("runner") or {}).get("tier_sim") or {}
    resolved_from: dict[str, str] = {}
    for key, placeholder in pending.items():
        tier = placeholder[len(prefix) :]
        if not tier:
            raise ValueError(f"{target}: empty tier in performance evidence placeholder {placeholder!r}")
        concrete = None
        if tier == "L3":
            # L3 is a fidelity and may have several implementations. Resolve it through the same
            # target-neutral policy grading uses, so a faster available engine changes the frozen
            # evidence by derivation rather than by editing the shared profile.
            from merlin.targetgen.capsule_runner import describe_l3_engine

            selection = describe_l3_engine(target)
            if selection.get("available") and selection.get("engine"):
                concrete = str(selection["engine"])
        if concrete is None and declared.get(tier):
            concrete = str(declared[tier])
        if not concrete or concrete == "elaborated_rtl":
            raise ValueError(
                f"{target}: target oracle route does not resolve {tier} to a concrete simulator "
                f"(declared={declared.get(tier)!r})"
            )
        evidence[key] = concrete
        resolved_from[key] = placeholder
    evidence["resolved_from"] = resolved_from
    return performance


def _materialize_performance_entry(entry: dict, binding) -> dict:
    """Resolve a performance member onto a runnable direct corpus path.

    Dtypes come from workload_gen's capability-manifest accessor and must agree
    with the corpus binding selected for the same target.  The shared template
    therefore contains neither a target dtype nor a frontend choice.
    """
    target = str(getattr(binding, "target", "") or "")
    if not target:
        raise ValueError("performance materialization needs binding.target")
    binding_operand = getattr(binding, "operand_dtype", None)
    binding_accum = getattr(binding, "accum_dtype", None)
    declared = entry.pop("_encoding_variant", None)
    derived_axes = entry.pop("_derived_axes", None)
    if declared:
        # AN ENCODING FAMILY'S WHOLE CONTENT IS THAT ITS MEMBERS DIFFER HERE. This function otherwise
        # overwrites `operand_dtype` with the target's single corpus binding, which is right for every
        # other family -- they compare geometry or fusion at ONE datapath and must not drift off it --
        # and fatal for an L6_global comparison: both members would collapse onto the same encoding and
        # the differential would read "no effect" for a lever that was never exercised.
        #
        # The accumulate format is DERIVED for that operand rather than carried over from the binding:
        # an accumulator belongs to the datapath the operand feeds, and pairing one encoding's operand
        # with another's accumulator describes a machine that does not exist.
        operand_dtype = str(declared)
        accum_dtype = _accum_for_encoding(target, operand_dtype, binding_accum)
        datatype_basis = "declared_encoding_variant"
    elif binding_operand and binding_accum:
        # This is the binding _write_capsule will actually consume, derived by
        # corpus_spec from the target experiment + numeric profile. Prefer it
        # over re-deriving through a second capability-manifest representation.
        operand_dtype, accum_dtype = str(binding_operand), str(binding_accum)
        datatype_basis = "corpus_binding"
    else:
        operand_dtype, accum_dtype = WG.datapath_formats(target, accum_dtype=binding_accum)
        datatype_basis = "merlin.perf.workload_gen.datapath_formats"
    op = str(entry.get("op") or "")
    if op not in CS.BUILDERS:
        raise ValueError(
            f"performance member {entry.get('name', '?')}: direct corpus source has no runnable builder for {op!r}"
        )
    entry["source"] = "direct"
    entry["operand_dtype"] = operand_dtype
    performance = copy.deepcopy(entry["performance"])
    performance = _resolve_target_oracle_evidence(performance, target)
    performance["emitter"] = copy.deepcopy(performance["emitter"])
    performance["emitter"]["resolved"] = {
        "source": "direct",
        "operand_dtype": operand_dtype,
        "accum_dtype": accum_dtype,
        "datatype_basis": datatype_basis,
        "builder": "merlin.targetgen.corpus_spec.build",
    }
    if derived_axes:
        # THE UNREACHABLE REGIMES TRAVEL WITH THE CAPSULE. The derivation knows which bands it could
        # not reach and why, and that answer is worth exactly as much as the points it did reach: a
        # regime nothing covers because nothing CAN is a result, while one that is merely absent from
        # the corpus is a hole. Recorded on every member so a corpus-side gate can read it out of the
        # tracked capsule rather than out of a generation report that may not have been rewritten.
        performance["emitter"]["derived_axes"] = copy.deepcopy(derived_axes)
    entry["performance"] = performance
    return entry


def expand_sweeps(
    profile: dict,
    binding,
    *,
    trait_facts: dict | None = None,
    skipped: list | None = None,
    blocked_unimplemented: list | None = None,
    errors: list | None = None,
    traits: dict | None = None,
) -> list[dict]:
    """Return the profile's capsule entries with any ``sweeps:`` block expanded.

    A performance family's ``performance.gate.traits`` are evaluated against
    :func:`derive_profile`'s canonical tri-state records. Every trait must be
    exactly True. False and None both skip admission but remain distinct, with
    their evidence and evidence tier, in ``skipped``.

    A sweep is a cross-product over named axes plus a shared ``base``, producing
    exactly the flat entry dicts the per-capsule pipeline already consumes — so
    ``_write_capsule`` and every golden path are untouched by this feature.

    Two rules are enforced rather than documented:

    * **Every fitted axis needs at least two distinct points.** K is always
      fitted when present because one reduction depth cannot separate a tiled
      unit's rate from fixed overhead. Other axes opt in through ``fit_axes``;
      a one-point fit prices a parameter confidently and wrongly.
    * **Names must be unique across generated and hand-authored entries.** A
      collision would have one capsule overwrite another's directory, silently
      shrinking the corpus.

    Hand-written entries in ``capsules:`` are kept verbatim and come first, so a
    profile can mix a sweep with cases whose prose is worth writing by hand.
    """
    entries = list(profile.get("capsules") or [])
    sweeps = profile.get("sweeps") or []
    if not sweeps:
        return entries
    # Compatibility for the public/holdout disjointness checker, which passes
    # this old keyword even for purely functional profiles. It is never used to
    # admit performance: the first performance sweep below rejects it.
    legacy_traits_supplied = traits is not None

    tile = int(getattr(binding, "tile_dim", 0) or 0)
    if tile < 1:
        raise ValueError("sweeps need a tile edge; the binding reports none")

    seen = {e.get("name") for e in entries if isinstance(e, dict)}
    generated: list[dict] = []

    for sweep in sweeps:
        if not isinstance(sweep, dict):
            raise ValueError(f"sweep entry {sweep!r} is not a mapping")
        sweep_id = str(sweep.get("id") or "").strip()
        if not sweep_id:
            raise ValueError("every sweep needs an `id` (it prefixes the generated names)")
        base = dict(sweep.get("base") or {})
        variant_performance = [
            i
            for i, variant in enumerate(sweep.get("variants") or [])
            if isinstance(variant, dict) and "performance" in variant
        ]
        if variant_performance:
            raise ValueError(
                f"sweep {sweep_id!r}: performance blocks belong on `base`, not variants "
                f"{variant_performance}; every member of a family shares one claim contract"
            )
        performance = base.get("performance")
        is_performance = base.get("cat") in {"perf", "_perf"} or performance is not None
        gate_decision = None
        if is_performance:
            if legacy_traits_supplied:
                raise ValueError(
                    f"performance sweep {sweep_id}: legacy ad-hoc `traits` cannot gate performance; "
                    "pass canonical derive_profile(target) records through `trait_facts`"
                )
            performance = _validate_performance_block(performance, owner=f"performance sweep {sweep_id}")
            if performance["family"] != sweep_id:
                raise ValueError(f"performance sweep {sweep_id}: performance.family must equal the sweep id")
            facts = trait_facts
            if facts is None:
                target = str(getattr(binding, "target", "") or "")
                if not target:
                    raise ValueError(
                        f"performance sweep {sweep_id}: no trait facts supplied and binding.target is absent"
                    )
                facts = _performance_facts(target)
            ok, decision = evaluate_gate(performance["gate"], facts)
            gate_decision = decision
            if not ok:
                if skipped is not None:
                    skipped.append(
                        {
                            "family": sweep_id,
                            "sweep": sweep_id,
                            "status": "skipped_inapplicable",
                            "gate": decision,
                            "fit_axes": list(sweep.get("fit_axes") or []),
                            "comparison_roles": _comparison_roles(sweep),
                        }
                    )
                continue
            emitter_status = str(performance["emitter"]["status"])
            if emitter_status != "existing":
                if blocked_unimplemented is not None:
                    blocked_unimplemented.append(
                        {
                            "family": sweep_id,
                            "status": "blocked_unimplemented",
                            "reason": f"declared emitter {emitter_status!r} is not implemented",
                            "emitter": copy.deepcopy(performance["emitter"]),
                            "fit_axes": list(sweep.get("fit_axes") or []),
                            "comparison_roles": _comparison_roles(sweep),
                        }
                    )
                continue
        axes = sweep.get("axes") or {}
        if not isinstance(axes, dict) or not axes:
            raise ValueError(f"sweep {sweep_id!r} declares no axes")
        encodings = target_encodings(str(getattr(binding, "target", "") or ""))

        # Resolve each axis to concrete extents, preserving declaration order so
        # the generated corpus is reproducible.
        resolved: dict[str, list[int]] = {}
        derived_axes = {a: t for a, t in axes.items() if isinstance(t, dict)}
        for axis, tokens in axes.items():
            if axis in derived_axes:
                continue
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(f"sweep {sweep_id!r} axis {axis!r} must be a non-empty list")
            resolved[axis] = [resolve_extent(t, tile) for t in tokens]
        # DERIVED axes come second, because a derivation reads the literal axes it is banded against
        # (a residency band is a property of one parallel shape). The order is a dependency, not a
        # convenience: resolving them together would make the M/N a derivation sees depend on dict
        # iteration order.
        axis_labels: dict[str, dict] = {}
        axis_records: dict[str, dict] = {}
        unhostable = None
        for axis, spec in derived_axes.items():
            try:
                values, labels, record = _resolve_derived_axis(
                    spec,
                    owner=f"sweep {sweep_id!r}",
                    axis=axis,
                    target=str(getattr(binding, "target", "") or ""),
                    tile=tile,
                    dtype=getattr(binding, "operand_dtype", None),
                    fixed=resolved,
                )
            except AxisDerivationUnavailable as exc:
                unhostable = {"axis": axis, "derive": str(spec.get("derive")), "detail": str(exc)}
                break
            resolved[axis] = values
            axis_labels[axis] = labels
            axis_records[axis] = record
        if unhostable is not None:
            if skipped is not None:
                skipped.append(
                    {
                        "family": sweep_id,
                        "sweep": sweep_id,
                        "status": "skipped_inapplicable",
                        "reason": unhostable["detail"],
                        "axis_derivation": unhostable,
                        "gate": gate_decision,
                        "fit_axes": list(sweep.get("fit_axes") or []),
                        "comparison_roles": _comparison_roles(sweep),
                    }
                )
            continue

        declared_fit_axes = sweep.get("fit_axes") or []
        if not isinstance(declared_fit_axes, list) or any(
            not isinstance(axis, str) or not axis for axis in declared_fit_axes
        ):
            raise ValueError(f"sweep {sweep_id!r}: `fit_axes` must be a list of axis names")
        fitted_axes = list(dict.fromkeys((["K"] if "K" in resolved else []) + declared_fit_axes))
        unknown_fit_axes = [axis for axis in fitted_axes if axis not in resolved]
        if unknown_fit_axes:
            raise ValueError(
                f"sweep {sweep_id!r} fits undeclared axis/axes {unknown_fit_axes}; declared axes are {sorted(resolved)}"
            )
        for axis in fitted_axes:
            if len(set(resolved[axis])) < 2:
                detail = (
                    "a tiled unit needs at least TWO distinct K points, because one cannot "
                    "separate the rate from the per-tile overhead"
                    if axis == "K"
                    else "every fitted parameter needs at least TWO distinct points"
                )
                raise ValueError(f"sweep {sweep_id!r} fits {axis} over {resolved[axis]} — {detail}")

        combos = _cross_product(resolved)
        if len(combos) > _MAX_SWEEP_CAPSULES:
            raise ValueError(
                f"sweep {sweep_id!r} would generate {len(combos)} capsules (cap {_MAX_SWEEP_CAPSULES}); "
                f"narrow the axes rather than letting it be truncated"
            )

        # A sweep may also cross the extents with a list of NON-EXTENT overrides. The motivating case is
        # a fusion comparison: three capsules that must sit at the IDENTICAL shape and differ only in
        # which op they ask for, so that `cycles(fused)` and `cycles(part) + cycles(part)` are about the
        # same work. Expressing that as an axis is impossible -- an axis value is an extent, resolved
        # against the tile -- and expressing it as three hand-authored entries would hardcode the shape
        # in three places, where the whole point is that the three shapes are the same one.
        variants = sweep.get("variants") or [{}]
        if not isinstance(variants, list) or any(not isinstance(v, dict) for v in variants):
            raise ValueError(f"sweep {sweep_id!r}: `variants` must be a list of mappings")

        # A comparison group whose members cannot be compared is a declaration with no content. The
        # field was carried on four shipped capsules for a year while every one of them sat alone in
        # its group, so nothing could ever do the arithmetic it exists for.
        _groups: dict[str, int] = {}
        for v in variants:
            g = v.get("comparison_group")
            gname = g.get("name") if isinstance(g, dict) else g
            if gname:
                _groups[str(gname)] = _groups.get(str(gname), 0) + 1
        _lonely = sorted(g for g, n in _groups.items() if n < 2)
        if _lonely:
            raise ValueError(
                f"sweep {sweep_id!r} declares comparison group(s) {_lonely} with a single member; a "
                "group of one cannot be compared to anything, so declare the other members or drop "
                "the group"
            )

        template = str(sweep.get("name") or "{id}_{i:02d}")
        index = 0
        for combo in combos:
            for variant in variants:
                entry = copy.deepcopy(base)
                entry.update(combo)
                rendered = _render_variant(variant, combo)
                # An `operand_dtype: encoding<N>` variant names an encoding POSITIONALLY, so the shared
                # template never writes a dtype down. Resolved here, where the target is known, and
                # stashed under a private key so `_materialize_performance_entry` can tell "this member
                # chose its encoding" from "this member inherited the corpus binding".
                enc_token = rendered.get("operand_dtype")
                if isinstance(enc_token, str) and enc_token.startswith("encoding"):
                    rendered["operand_dtype"] = resolve_encoding(enc_token, encodings)
                    rendered["_encoding_variant"] = rendered["operand_dtype"]
                entry.update(rendered)
                # A derived axis contributes a LABEL as well as a value, so a member can be named by
                # the band it was derived for (`{K_label}` -> "spills") rather than only by an extent
                # whose meaning a reader would have to recompute against the store.
                name_ns = {**combo, **variant}
                for _axis, _labels in axis_labels.items():
                    name_ns[f"{_axis}_label"] = _labels.get(combo.get(_axis), "unknown")
                entry["name"] = template.format(id=sweep_id, i=index, **name_ns)
                index += 1
                entry.setdefault("source_role", "derived_sweep")
                # Provenance survives generation: say which sweep and which point.
                reference = sweep.get("source_reference") or f"generated by sweep {sweep_id!r}"
                axis_note = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
                if variant:
                    axis_note += "; " + ", ".join(
                        f"{k}={v}" for k, v in sorted(variant.items()) if not isinstance(v, dict)
                    )
                if axis_records:
                    entry["_derived_axes"] = {
                        a: {
                            "derive": str(axes[a].get("derive")),
                            "value": combo.get(a),
                            "label": axis_labels[a].get(combo.get(a)),
                            "derivation": r,
                        }
                        for a, r in axis_records.items()
                    }
                    axis_note += "; " + ", ".join(
                        f"{a}={combo.get(a)} is {axis_labels[a].get(combo.get(a))}" for a in sorted(axis_records)
                    )
                entry["source_reference"] = f"{reference} (tile={tile}; {axis_note})"
                if entry["name"] in seen:
                    raise ValueError(f"sweep {sweep_id!r} generated duplicate capsule name {entry['name']!r}")
                seen.add(entry["name"])
                if is_performance:
                    try:
                        _materialize_performance_entry(entry, binding)
                    except Exception as exc:  # noqa: BLE001 - persisted as a generation error
                        if errors is None:
                            raise
                        errors.append(
                            {
                                "family": sweep_id,
                                "member": entry["name"],
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "detail": str(exc)[:500],
                            }
                        )
                        continue
                generated.append(entry)

    # A GROUP REDUCED TO ONE MEMBER IS NOT A COMPARISON. The declaration-time check above refuses a
    # group AUTHORED with a single member, but a member can also disappear afterwards: materialization
    # is allowed to fail per member and record an error, and the survivors were shipped anyway. A lone
    # member then carries a `comparison_group` and a DIFFERENTIAL claim with nothing to difference
    # against -- which is the same silent vacuity the single-member check exists to prevent, arriving by
    # a different route. Measured while opening the encoding family: one target's second encoding could
    # not resolve its accumulator, and the first shipped alone.
    if generated:
        members: dict[str, int] = {}
        for e in generated:
            g = e.get("comparison_group")
            name = g.get("name") if isinstance(g, dict) else g
            if name:
                members[str(name)] = members.get(str(name), 0) + 1
        lonely = {g for g, n in members.items() if n < 2}
        if lonely:
            kept = []
            for e in generated:
                g = e.get("comparison_group")
                name = str(g.get("name") if isinstance(g, dict) else g or "")
                if name in lonely:
                    fam = (e.get("performance") or {}).get("family")
                    if errors is not None:
                        errors.append(
                            {
                                "family": fam,
                                "member": e.get("name", "?"),
                                "status": "error",
                                "error_type": "IncompleteComparisonGroup",
                                "detail": (
                                    f"comparison group {name!r} kept only this member; a "
                                    f"differential needs both, so it is dropped rather than "
                                    f"shipped as a comparison against nothing"
                                ),
                            }
                        )
                    continue
                kept.append(e)
            generated = kept

    return entries + generated


def _render_variant(variant: dict, combo: dict) -> dict:
    """A variant override with its string fields resolved against the shape point it is paired with.

    Only ``{extent}`` substitution, one level into a nested mapping -- enough for a comparison group to
    name the shape its members share (``fmb_{M}x{K}x{N}``) without any entry writing that shape down.
    A field with no placeholder is copied through untouched.
    """
    out: dict = {}
    for key, value in variant.items():
        if isinstance(value, str):
            out[key] = value.format(**combo)
        elif isinstance(value, dict):
            out[key] = {k: (v.format(**combo) if isinstance(v, str) else v) for k, v in value.items()}
        else:
            out[key] = value
    return out


def _cross_product(axes: dict) -> list[dict]:
    """Cross-product of ``{axis: [values]}`` preserving declaration order."""
    combos: list[dict] = [{}]
    for axis, values in axes.items():
        combos = [{**combo, axis: value} for combo in combos for value in values]
    return combos
