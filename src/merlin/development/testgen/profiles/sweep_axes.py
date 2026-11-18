"""Host-owned Phase 0 sweeps implementation."""

from __future__ import annotations

import copy
import hashlib

from merlin.analysis.perf.contract import workload_programs as WG  # noqa: E402
from merlin.analysis.perf.contract.profile import TRAITS, derive_profile  # noqa: E402
from merlin.development.targetgen.corpus import binding as CS  # noqa: E402
from merlin.development.testgen.generation.provenance import _document_digest
from merlin.development.testgen.performance.capture_residency import (
    _bound_capture_inventory,
    _capture_residency_sweep,
)
from merlin.development.testgen.profiles.loading import (
    _DERIVED_AXES,
    _comparison_roles,
    _validate_performance_block,
    pending_host_source_basis,
)
from merlin.development.testgen.profiles.synthesis_selection import _cross_product, _render_variant
from merlin.runtime.backends.registry import EXECUTION_CAPABILITIES, execution_capability_facts  # noqa: E402

# ---- sweeps ------------------------------------------------------------------------------------

_TILE_TOKEN = "tile"


_MAX_SWEEP_CAPSULES = 128


_EXTENT_KEYS = ("M", "K", "N", "H", "Skv", "Dv", "B")


class OracleSelectionError(ValueError):
    """Required performance evidence has no consistent concrete execution route."""


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
    out = None
    for key in _EXTENT_KEYS:
        value = entry.get(key)
        if isinstance(value, bool):
            raise ValueError(f"sweep extent {value!r} is a bool, not an extent")
        if not isinstance(value, str):
            continue
        if out is None:
            out = dict(entry)
        out[key] = resolve_extent(value, tile)
    matmuls = (out or entry).get("matmuls")
    if isinstance(matmuls, list) and any(
        isinstance(mm, dict) and isinstance(mm.get(key), str) for mm in matmuls for key in _EXTENT_KEYS
    ):
        if out is None:
            out = dict(entry)
        out["matmuls"] = [
            {
                key: (resolve_extent(value, tile) if key in _EXTENT_KEYS and isinstance(value, str) else value)
                for key, value in mm.items()
            }
            if isinstance(mm, dict)
            else mm
            for mm in matmuls
        ]
    resolved = out if out is not None else entry
    prelude = resolved.get("prelude")
    if isinstance(prelude, dict) and any(isinstance(prelude.get(key), str) for key in _EXTENT_KEYS):
        # A residual-state prelude is sized in the same tile-relative tokens as the entry it precedes.
        resolved = {
            **resolved,
            "prelude": {
                key: (resolve_extent(value, tile) if key in _EXTENT_KEYS and isinstance(value, str) else value)
                for key, value in prelude.items()
            },
        }
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


def target_encodings(target: str, *, contract: dict | None = None) -> list[str]:
    """The operand encodings this target can compute a contraction in, in capsule spelling.

    The axis an ``L6_global`` (encoding / packing / layout) family sweeps. Derived from the same place
    the gate trait ``multiple_operand_encodings`` reads -- the contraction family's declared dtypes --
    so the family's members and the gate that admits it cannot disagree about how many encodings exist.

    Sorted, because the corpus must regenerate byte-identically and a capability map is a mapping.
    Empty when the map cannot be resolved: a family that needs a choice then has none to make, and the
    gate refuses it with evidence rather than this inventing one.
    """
    try:
        from merlin.development.targetgen.conformance.requirements import capsule_dtype
        from merlin.target.capabilities.eligibility import capability_map_for_target

        if contract is None:
            cap = (capability_map_for_target(target) or {}).get("contraction")
        else:
            from merlin.target.capabilities.compute_units import compute_units, semantic_capability_map

            cap = semantic_capability_map(compute_units(contract)).get("contraction")
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


def resolve_extent(token, tile: int | None) -> int:
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
    if type(tile) is not int or tile < 1:
        raise ValueError("tile-relative extent requires independently selected tile geometry")

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

    def __init__(self, detail: str, *, record: dict | None = None):
        super().__init__(detail)
        self.record = record


def _memory_regime_axis(
    spec: dict,
    *,
    owner: str,
    target: str,
    tile: int,
    dtype: str | None,
    fixed: dict[str, list[int]],
    evidence=None,
) -> tuple[list[int], dict, dict]:
    """``(K extents, {extent: regime}, derivation record)`` for a residency-banded reduction sweep.

    The parallel extents are read from the sweep's OWN ``M``/``N`` axes so the two cannot disagree
    about the shape whose residency is being banded; each must be single-valued, because a band is a
    property of one shape family and crossing it with a second parallel extent would put two different
    residency ladders under one name.
    """
    from merlin.development.targetgen.policies import memory_regime as MR

    regimes = [str(r) for r in (spec.get("regimes") or MR.ORDER)]
    points_per_regime = int(spec.get("points_per_regime", 2))
    if points_per_regime < 2:
        raise ValueError(
            f"{owner}: points_per_regime={points_per_regime} -- a rate and a fixed intercept are two "
            "parameters and cannot be separated by fewer than two points in the same regime"
        )
    ceiling = float(spec.get("spills_max_fraction_of_capacity", 2.0))
    extents = {}
    for axis in ("M", "N"):
        points = fixed.get(axis) or []
        if len(set(points)) != 1:
            raise ValueError(
                f"{owner}: a memory-regime reduction axis needs a single-valued {axis} axis "
                f"(got {points}); the band is a property of one parallel shape"
            )
        extent = int(points[0])
        if extent < 1:
            raise ValueError(f"{owner}: {axis}={extent} is not a positive parallel extent")
        extents[axis] = extent
    selected_store = {}
    if evidence is not None:
        from merlin.target.capabilities import address_space as AS

        space = AS.derive_address_space(target, facts=evidence.refreshed_facts)
        resolved_store = AS.operand_store(space, dtype=dtype)
        selected_store = {"store": resolved_store.store, "capacity": resolved_store.capacity_rows(dtype) or 0}
    parallel = (
        {"m_tiles": extents["M"] // tile, "n_tiles": extents["N"] // tile}
        if extents["M"] % tile == 0 and extents["N"] % tile == 0
        else {"m_extent": extents["M"], "n_extent": extents["N"]}
    )
    record = MR.reduction_depth_regimes(
        target,
        regimes,
        tile_dim=tile,
        dtype=dtype,
        **parallel,
        points_per_regime=points_per_regime,
        spills_max_fraction=ceiling,
        **selected_store,
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
    spec: dict,
    *,
    owner: str,
    axis: str,
    target: str,
    tile: int,
    dtype: str | None,
    fixed: dict[str, list[int]],
    evidence=None,
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
        return _memory_regime_axis(
            spec, owner=owner, target=target, tile=tile, dtype=dtype, fixed=fixed, evidence=evidence
        )
    if kind == "resident_allocation_boundary":
        from merlin.development.testgen.profiles.resource_boundaries import BoundaryUnavailable, derive

        try:
            return derive(
                spec,
                owner=owner,
                axis=axis,
                target=target,
                tile=tile,
                dtype=dtype,
                fixed=fixed,
                evidence=evidence,
                resolve_extent=resolve_extent,
            )
        except BoundaryUnavailable as exc:
            raise AxisDerivationUnavailable(str(exc), record=exc.record) from exc
    raise ValueError(f"{owner}: axis derivation {kind!r} has no resolver")  # unreachable; fail closed


def _performance_facts(target: str, *, evidence=None) -> dict:
    """Canonical hardware-trait and backend-execution facts, derived once for this target."""
    if evidence is not None:
        if evidence.target != target:
            raise ValueError("selected evidence target differs from sweep target")
        return evidence.performance_facts
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
    source_decision = pending_host_source_basis(gate)
    if not isinstance(names, list) or (not names and source_decision is None):
        raise ValueError("performance gate.traits must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in names):
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
    any_unknown = bool(unestablished or execution_unestablished or source_decision)
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
    if source_decision is not None:
        decision["source_execution_basis"] = source_decision
    return outcome == "satisfied", decision


def _accum_for_encoding(target: str, operand: str, fallback: str | None) -> str:
    """The accumulate format ``target`` declares for a contraction whose operands are ``operand``.

    ASKED OF THE ROUTER, not re-derived. `routing._legal_on` is the one predicate that answers "what
    does this unit accumulate this contraction in", and it answers by taking the FIRST declared
    accumulate rule that matches -- which is what every other path in the repo computes for the same
    contraction. A second derivation here would be a second answer to one question, and on a target
    declaring several rules for one operand (e.g. both bf16 and f32 accumulation for fp8_e4m3, which is a
    real capability rather than an ambiguity) the two would disagree.

    Falls back to the corpus binding's accumulator only when no unit accepts the operand at all, and
    raises when there is no fallback either: a member whose accumulator is unknown cannot be built,
    and guessing one emits a capsule that measures arithmetic the target does not perform.
    """
    from merlin.target.capabilities import compute_units as _cu
    from merlin.target.capabilities.routing import OpDemand, _legal_on  # noqa: PLC2701 -- one predicate
    from merlin.target.registry.identity import load_contract

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


def _resolve_target_oracle_evidence(
    performance: dict, target: str, *, oracle_selection: dict[str, str] | None = None
) -> dict:
    """Resolve ``$target_oracle:<tier>`` evidence placeholders from the target's own oracle route.

    The shared profile must not name one target's simulator binary. An explicit
    recipe may select concrete engines over abstract contract tiers without
    probing the build host; concrete contract tiers are used directly. Legacy
    profiles use the target's contract and RTL-engine policy. Resolved engine,
    metric and oracle-kind names are frozen with their source placeholders.
    """
    acceptance = performance.get("acceptance")
    evidence = acceptance.get("evidence") if isinstance(acceptance, dict) else None
    if not isinstance(evidence, dict):
        return performance
    prefix = "$target_oracle:"
    pending = {key: value for key, value in evidence.items() if isinstance(value, str) and value.startswith(prefix)}
    kind_placeholder = evidence.get("timing_oracle_kind")
    kind_pending = isinstance(kind_placeholder, str) and kind_placeholder.startswith("$target_oracle_kind:")
    if not pending and not kind_pending:
        return performance
    from merlin.target.registry.experiment_descriptor import load_capability_manifest

    contract = load_capability_manifest(target).contract
    declared = (contract.get("runner") or {}).get("tier_sim") or {}
    resolved_from: dict[str, str] = {}
    for key, placeholder in pending.items():
        tier = placeholder[len(prefix) :]
        if not tier:
            raise OracleSelectionError(f"{target}: empty tier in performance evidence placeholder {placeholder!r}")
        concrete = None
        if oracle_selection is not None:
            concrete = oracle_selection.get(tier) or declared.get(tier)
            if not isinstance(concrete, str) or not concrete or concrete == "elaborated_rtl":
                raise OracleSelectionError(f"{target}: explicit Phase 0 inputs have no concrete {tier} oracle")
            declared_engine = declared.get(tier)
            if declared_engine and declared_engine != "elaborated_rtl" and concrete != declared_engine:
                raise OracleSelectionError(
                    f"{target}: selected {tier} oracle {concrete!r} conflicts with target contract {declared_engine!r}"
                )
        elif tier == "L3":
            # L3 is a fidelity and may have several implementations. Resolve it through the same
            # target-neutral metadata policy grading uses. The evaluator imports optional AET;
            # Phase 0's frozen derivation wheel does not and must not import that owner merely
            # to name the engine. An unavailable metadata route still fails below rather than
            # substituting the generic fidelity label for a concrete simulator.
            from merlin.target.simulators.oracle_policy import selected_l3_engine_report

            selection = selected_l3_engine_report(target)
            if selection.get("available") and selection.get("engine"):
                concrete = str(selection["engine"])
        if oracle_selection is None and concrete is None and declared.get(tier):
            concrete = str(declared[tier])
        if not concrete or concrete == "elaborated_rtl":
            raise OracleSelectionError(
                f"{target}: target oracle route does not resolve {tier} to a concrete simulator "
                f"(declared={declared.get(tier)!r})"
            )
        evidence[key] = concrete
        resolved_from[key] = placeholder
    if kind_pending:
        kind_tier = kind_placeholder.partition(":")[2]
        simulator = evidence.get("timing_simulator")
        if kind_tier != "L3" or evidence.get("timing_tier") != kind_tier:
            raise OracleSelectionError(f"{target}: timing oracle kind must name the selected L3 tier")
        if (
            not isinstance(simulator, str)
            or not simulator
            or simulator.startswith("$")
            or simulator == "elaborated_rtl"
        ):
            raise OracleSelectionError(f"{target}: selected L3 timing simulator cannot name an oracle kind")
        evidence["timing_oracle_kind"] = f"rtl_{simulator}"
        resolved_from["timing_oracle_kind"] = kind_placeholder
    evidence["resolved_from"] = resolved_from
    fit = acceptance.get("fit")
    if isinstance(fit, dict) and fit.get("dependent_metric") == "$target_oracle_metric:L3":
        simulator = evidence.get("timing_simulator")
        if not isinstance(simulator, str) or evidence.get("timing_tier") != "L3":
            raise OracleSelectionError(f"{target}: selected L3 timing oracle cannot name an affine metric")
        fit["dependent_metric"] = f"{simulator}_L3_cycles"
    return performance
