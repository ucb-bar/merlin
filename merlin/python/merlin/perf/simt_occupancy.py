"""Lane and warp occupancy on a SIMT cluster, against the machine's OWN ELABORATED geometry.

On every other archetype in this layer "occupancy" is a busy bit: a unit either carried work in a
cycle or it did not, and :mod:`merlin.perf.occupancy` counts the cycles in which two busy bits were
high together. **A SIMT cluster breaks that reading.** A warp whose predicate mask enabled one lane
of sixteen is 100% busy for the whole issue, and every busy-bit instrument on this layer will report
it as a fully utilised machine. Occupancy here is a WIDTH -- active lanes over the lane width, and
resident warps over the warps a core can hold -- so it needs a denominator the busy bit never needed.

⚠️ **Divergence is the SIMT analogue of the interlocked target's "every reordering is correct".**
There, a machine that serialises whatever it is handed makes a correctness gate unable to fail on a
scheduling mistake, which is why η exists: the gate proves the answer, and only the overlap figure
proves the schedule. Here the same hole opens one level down. A SIMT kernel that branches its threads
apart computes exactly the right answer on one lane at a time, passes every functional and numerical
capsule, and wastes fifteen sixteenths of the datapath. **A correctness gate proves nothing about a
SIMT kernel's use of the machine**, and lane occupancy is the falsifier that does -- the reason this
module exists beside the grader rather than inside it.

WHERE THE DENOMINATOR COMES FROM -- AND WHY IT IS THE RTL
---------------------------------------------------------
The first version of this module took the denominator from a **cycle model's TOML config**, on the
stated grounds that "the RTL cannot supply this". That was wrong, and the correction is the point of
the current file. The fact bundles for these designs are empty because nobody pointed an extractor at
them, not because the elaboration lacks the geometry -- and a cycle model's config is a statement about
the *model*, which is precisely the thing an RTL-grounded figure exists to be checkable against. A
denominator taken from the model makes "the model agrees with itself" the only claim the figure can
support.

So the geometry now comes from :mod:`merlin.targetgen.rtl.simt_geometry`, which reads it out of what
was ELABORATED -- a CIRCT HW-dialect / arc-model artifact, or the design's own FIRRTL -- as a width, a
table depth and an instance-path count, never as a literal. The model config is still read, but only as
a :func:`model_cross_check`: it is reported beside the derived numbers, disagreements named, and it can
never become the answer. **When no elaboration can be read, this module refuses** (``unavailable``,
with the reason) even if a model config is sitting right there, because falling back to it would
reinstate the substitution silently.

WHAT THIS DOES NOT DO
---------------------
It derives the instrument and computes the ratios from supplied readings. It does not run a
simulation, install a per-cycle callback, or claim a measurement. **With no execution trace there is
no occupancy**, and this module says so with the reason rather than returning a full-lane 1.0 -- the
reading that flatters a divergent kernel into looking dense, and the exact failure this instrument
was built to catch.

Three states throughout, never two: ``derived`` (read from an elaboration), ``absent`` (an elaboration
was read and describes no SIMT geometry -- a fact about that design), and ``unavailable`` (nothing
could be located or read -- UNKNOWN, which is not absent).
"""
from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.targetgen.rtl.simt_geometry import (
    CORE as _CORE,
    LANE as _LANE,
    ROLES,
    WARP as _WARP,
    SimtGeometry,
    geometry_from_dict,
    role_tokens,
)
from merlin.targetgen.rtl import simt_geometry as _rtl_geometry

__all__ = [
    "ACTIVE_LANES_PER_ISSUE",
    "ACTIVE_LANE_CYCLES",
    "RESIDENT_WARPS",
    "ROLES",
    "WARP_ISSUES",
    "SimtGeometry",
    "geometry_for_target",
    "model_cross_check",
    "occupancy_for_target",
    "occupancy_from_readings",
]

# --------------------------------------------------------------------------------------------
# Reading names. A caller supplies these from whatever produced the trace; they are named here so a
# missing one can be reported BY NAME rather than as a bare "no readings".
# --------------------------------------------------------------------------------------------
#: Sum, over warp issues, of the lanes the thread mask enabled. NOT the cycles a warp was busy.
ACTIVE_LANE_CYCLES = "active_lane_cycles"
#: How many warp issues that sum is over -- the denominator's multiplier.
WARP_ISSUES = "warp_issues"
#: Per-issue active-lane counts, when the trace carries them individually. Both figures above are
#: derived from it, so a caller with the masks need not pre-reduce them (and cannot mis-pair them).
ACTIVE_LANES_PER_ISSUE = "active_lanes_per_issue"
#: Warps resident on one core while the kernel ran. The warp-occupancy numerator.
RESIDENT_WARPS = "resident_warps"


# --------------------------------------------------------------------------------------------
# The geometry -- DERIVED, from the elaborated design
# --------------------------------------------------------------------------------------------


def geometry_for_target(target: str, *, artifact_path: str | Path | None = None,
                        artifact_text: str | None = None,
                        cross_check: bool = True) -> dict[str, Any]:
    """``target``'s SIMT geometry, read out of its own elaborated design.

    A thin pass-through to :func:`merlin.targetgen.rtl.simt_geometry.geometry_for_target` -- the
    derivation lives there because it is an RTL fact, not a performance one -- plus the model
    cross-check attached under ``model_cross_check``. The cross-check is REPORTED, never consulted:
    ``status`` and every number in ``geometry`` come from the elaboration alone, and a target with a
    model config and no readable elaboration reports ``unavailable``.

    ``artifact_path`` / ``artifact_text`` override the search, which is how a caller prices a machine
    whose elaboration is not on this host -- and how a test pins that an elaboration missing a
    dimension reports it missing instead of substituting a plausible one.
    """
    out = _rtl_geometry.geometry_for_target(target, artifact_path=artifact_path,
                                            artifact_text=artifact_text)
    out.setdefault("target", target)
    if cross_check:
        out["model_cross_check"] = model_cross_check(target, derived=out.get("geometry"))
    return out


def occupancy_for_target(target: str, readings: Mapping[str, Any] | None = None, *,
                         artifact_path: str | Path | None = None,
                         artifact_text: str | None = None) -> dict[str, Any]:
    """Derive ``target``'s geometry and, if readings are supplied, its occupancy.

    The end-to-end call, and the one that most often returns UNKNOWN: on a target with an elaboration
    but no trace the geometry resolves and every ratio refuses, which is the honest report and not a
    failure. Nothing here runs the machine.
    """
    geo = geometry_for_target(target, artifact_path=artifact_path, artifact_text=artifact_text)
    if geo.get("status") != "derived":
        return {"target": target, "state": "unknown", "geometry_status": geo.get("status"),
                "geometry": geo,
                "why": f"no SIMT geometry is derived from {target!r}'s elaborated design "
                       f"({geo.get('why')}), so no occupancy denominator exists. The cycle model's "
                       f"config is NOT a substitute: it would price the run against the model"}
    geom = geometry_from_dict(geo["geometry"])
    out = occupancy_from_readings(readings, geom)
    out["target"] = target
    out["geometry_status"] = "derived"
    out["source"] = geom.source
    out["authority"] = geo.get("authority")
    for key in ("contested", "why_contested", "model_cross_check"):
        if geo.get(key) is not None:
            out[key] = geo[key]
    return out


# --------------------------------------------------------------------------------------------
# The CROSS-CHECK -- a cycle model's own declaration, reported and never believed
# --------------------------------------------------------------------------------------------

#: The accessor a target's own package exposes to say where its cycle-model declaration lives. Asking
#: the target rather than holding a path keeps this target-agnostic: the value is resolved by that
#: package from the descriptor / dotenv, so moving the checkout moves the answer with it.
_CONFIG_ACCESSOR = "config_path"   # fact-source-ok: names the CROSS-CHECK accessor; no geometry is read through it


def _role_of(key: str) -> str | None:
    """The SIMT dimension a model-config key names, or ``None``.

    Structural, per the repo's no-regex rule: tokenize the key (underscores and camel-case humps
    alike), compare against the role words. A key matching two roles at once (``lanes_per_warp``) is
    deliberately NOT resolved -- it is a ratio, not a count, and reading it as either dimension is off
    by the other one.
    """
    toks = set(role_tokens(key))
    hits = {r for r in ROLES if r in toks}
    return next(iter(hits)) if len(hits) == 1 else None


def _dims_from_toml(text: str) -> tuple[dict[str, int], dict[str, str], list[str], str]:
    """``(role -> value, role -> key read, ambiguous roles, table path)`` for a model declaration.

    Parsed with :mod:`tomllib` -- a real parser, not line matching. The table resolving the most
    dimensions wins; a tie resolves to nothing, because a tie means the file describes two machines.
    """
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}, {}, [], ""

    def tables(node: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Mapping[str, Any]]]:
        out = [(prefix, node)]
        for k, v in node.items():
            if isinstance(v, Mapping):
                out.extend(tables(v, f"{prefix}.{k}" if prefix else str(k)))
        return out

    scored: list[tuple[int, str, dict[str, int], dict[str, str], list[str]]] = []
    for path, table in tables(doc):
        found: dict[str, dict[str, int]] = {r: {} for r in ROLES}
        for key, val in table.items():
            # bool is an int subclass and a flag is not a count -- admitting one would make
            # `warp_sync = true` a warp count of 1, which reads as a fully occupied core.
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                continue
            role = _role_of(str(key))
            if role is not None:
                found[role][str(key)] = int(val)
        vals: dict[str, int] = {}
        keys: dict[str, str] = {}
        ambiguous: list[str] = []
        for role in ROLES:
            distinct = set(found[role].values())
            if len(distinct) == 1:
                k = sorted(found[role])[0]
                vals[role], keys[role] = found[role][k], k
            elif len(distinct) > 1:
                ambiguous.append(role)
        scored.append((len(vals), path, vals, keys, ambiguous))
    best = max((n for n, *_ in scored), default=0)
    winners = [s for s in scored if s[0] == best and best > 0]
    if len(winners) != 1:
        contested = [s for s in scored if s[4]]
        if contested:
            _n, path, vals, keys, ambiguous = contested[0]
            return vals, keys, ambiguous, path
        return {}, {}, [], ""
    _n, path, vals, keys, ambiguous = winners[0]
    return vals, keys, ambiguous, path


def _model_config_path(target: str) -> tuple[Path | None, str]:   # fact-source-ok: locates the model declaration for model_cross_check() only
    """``(path, how)`` for ``target``'s cycle-model declaration, via its own runtime backend."""
    try:
        from merlin.runtime.backends.base import get_backend
        mod = get_backend(target)
    except Exception as e:                                     # noqa: BLE001 -- no such backend
        return None, f"no runtime backend is registered for {target!r} ({type(e).__name__})"
    accessor = getattr(mod, _CONFIG_ACCESSOR, None)
    if accessor is None:
        return None, f"the {target!r} backend exposes no {_CONFIG_ACCESSOR}()"
    try:
        return Path(str(accessor())), f"{target!r} backend {_CONFIG_ACCESSOR}()"
    except Exception as e:                                     # noqa: BLE001
        return None, f"{target!r} backend {_CONFIG_ACCESSOR}() raised {type(e).__name__}"


def model_cross_check(target: str, *, derived: Mapping[str, Any] | None = None,
                      config_text: str | None = None,
                      config_path: str | Path | None = None) -> dict[str, Any]:
    """What the CYCLE MODEL declares, compared against the geometry derived from the RTL.

    ⚠️ **This is not a source of hardware facts and must never become one.** A Rust cycle model, a
    functional simulator and a hand-written NPU model are all downstream of the hardware: they may
    corroborate a derived number or contradict it -- and a contradiction is a finding worth surfacing,
    which is why this exists -- but the fact itself comes from CIRCT or from the target's RTL repo. The
    return value carries no ``lane_width`` the caller could mistake for a denominator; it carries a
    comparison, and ``agrees``/``disagrees`` lists naming exactly which dimensions matched.

    ``status`` is ``unavailable`` when no model declaration could be read; that is a fact about the
    cross-check, and it never changes the derived geometry's own status.
    """
    if config_text is None:
        path = Path(config_path) if config_path is not None else None
        how = "supplied by the caller"
        if path is None:
            path, how = _model_config_path(target)   # fact-source-ok: the cross-check's own lookup; the geometry came from the elaboration
        if path is None:
            return {"status": "unavailable", "why": how,
                    "note": "no cross-check was possible; the derived geometry stands on the RTL alone"}
        try:
            config_text = path.read_text(encoding="utf-8")
        except OSError as e:
            return {"status": "unavailable", "source": str(path),
                    "why": f"{type(e).__name__}: {e}",
                    "note": "no cross-check was possible; the derived geometry stands on the RTL alone"}
        source = str(path)
    else:
        source, how = (str(config_path) if config_path else "<supplied text>"), "supplied by the caller"

    vals, keys, ambiguous, table = _dims_from_toml(config_text)
    out: dict[str, Any] = {
        "status": "read" if vals or ambiguous else "declares_nothing",
        "source": source, "how": how, "table": table,
        "model_says": {r: vals.get(r) for r in ROLES}, "keys": keys,
        "ambiguous": ambiguous,
        "note": "a cycle model's configuration describes the MODEL. It is reported here to be compared "
                "with the RTL-derived geometry, never to supply it",
    }
    if derived is None:
        return out
    # The derived block spells its dimensions out (`lane_width`); the roles are the shared vocabulary.
    _FIELD = {_LANE: "lane_width", _WARP: "warps_per_core", _CORE: "cores"}
    agrees, disagrees, unchecked = [], [], []
    for role in ROLES:
        d, m = derived.get(_FIELD[role]), vals.get(role)
        if d is None or m is None:
            unchecked.append(role)
        elif int(d) == int(m):
            agrees.append(role)
        else:
            disagrees.append({"role": role, "rtl_derived": int(d), "model_says": int(m)})
    out["agrees"] = agrees
    out["disagrees"] = disagrees
    out["unchecked"] = unchecked
    if disagrees:
        out["why"] = (f"the cycle model's declaration contradicts the elaborated design on "
                      f"{[d['role'] for d in disagrees]}. The RTL-derived value is the fact; the model "
                      f"is running a machine that was not elaborated, and any cycle count it produced "
                      f"belongs to THAT machine")
    return out


# --------------------------------------------------------------------------------------------
# The arithmetic (unchanged -- only the geometry's provenance moved)
# --------------------------------------------------------------------------------------------


def _lane_figure(readings: Mapping[str, Any], geom: SimtGeometry) -> dict[str, Any]:
    """Active lanes over lane slots offered, with every refusal carrying its reason."""
    if geom.lane_width is None:
        return {"state": "unknown", "value": None,
                "why": "no lane width is derived, so the ratio has no denominator"}
    per_issue = readings.get(ACTIVE_LANES_PER_ISSUE)
    active = readings.get(ACTIVE_LANE_CYCLES)
    issues = readings.get(WARP_ISSUES)
    if isinstance(per_issue, Sequence) and not isinstance(per_issue, (str, bytes)):
        derived_active, derived_issues = sum(int(v) for v in per_issue), len(per_issue)
        # A caller may hand both the masks and a pre-reduced pair. If they disagree, one of them
        # describes a different run, and averaging or preferring either publishes a figure no trace
        # supports -- so say so instead.
        if active is not None and int(active) != derived_active:
            return {"state": "unknown", "value": None,
                    "why": f"{ACTIVE_LANE_CYCLES}={int(active)} contradicts the per-issue masks, which "
                           f"sum to {derived_active}: the two readings are of different runs"}
        if issues is not None and int(issues) != derived_issues:
            return {"state": "unknown", "value": None,
                    "why": f"{WARP_ISSUES}={int(issues)} contradicts the {derived_issues} per-issue "
                           f"masks supplied: the two readings are of different runs"}
        active, issues = derived_active, derived_issues
    missing = [n for n, v in ((ACTIVE_LANE_CYCLES, active), (WARP_ISSUES, issues)) if v is None]
    if missing:
        return {"state": "unknown", "value": None,
                "why": f"no execution trace supplied {missing}; a kernel whose mask was never observed "
                       f"is UNKNOWN, and reading it as full lanes would report a divergent kernel as a "
                       f"dense one -- the exact mistake this instrument exists to catch"}
    active, issues = int(active), int(issues)
    if issues <= 0:
        return {"state": "unknown", "value": None,
                "why": "the trace records no warp issues, so no lane slots were OFFERED; 0/0 is "
                       "undefined, not 0.0"}
    offered = issues * geom.lane_width
    if active < 0 or active > offered:
        return {"state": "unknown", "value": None,
                "why": f"{active} active lane-cycles over {issues} issues exceeds the {offered} the "
                       f"derived width of {geom.lane_width} offers: the trace and the geometry "
                       f"describe different machines, and neither can be trusted to price the other"}
    frac = active / float(offered)
    return {"state": "measured", "value": frac, "divergence": 1.0 - frac,
            "active_lane_cycles": active, "warp_issues": issues, "lane_slots_offered": offered,
            "lane_width": geom.lane_width,
            "note": "divergence is the fraction of offered lane slots the thread mask turned off; a "
                    "functional gate cannot see it, because the disabled lanes still produce the "
                    "right answer on the enabled ones"}


def _warp_figure(readings: Mapping[str, Any], geom: SimtGeometry) -> dict[str, Any]:
    """Resident warps over the warps one core can hold."""
    if geom.warps_per_core is None:
        return {"state": "unknown", "value": None,
                "why": "the elaboration names no warps-per-core, so the ratio has no denominator"}
    resident = readings.get(RESIDENT_WARPS)
    if resident is None:
        return {"state": "unknown", "value": None,
                "why": f"no trace supplied {RESIDENT_WARPS!r}; an unobserved warp count is UNKNOWN, and "
                       f"assuming a full core would hide a kernel that launched one warp"}
    resident = int(resident)
    if resident < 0 or resident > geom.warps_per_core:
        return {"state": "unknown", "value": None,
                "why": f"{resident} resident warp(s) is outside the {geom.warps_per_core} the "
                       f"elaborated warp-slot table holds: the trace and the geometry disagree"}
    return {"state": "measured", "value": resident / float(geom.warps_per_core),
            "resident_warps": resident, "warps_per_core": geom.warps_per_core}


def occupancy_from_readings(readings: Mapping[str, Any] | None,
                            geometry: SimtGeometry) -> dict[str, Any]:
    """Lane and warp occupancy from a set of trace READINGS against a derived geometry.

    ``lane`` is the divergence figure: active lane-cycles over the lane slots the issues offered.
    ``warp`` is residency: warps on a core over the warps it holds. ``simt`` multiplies them, which is
    the fraction of the core's whole lane-slot capacity that carried work -- and it is reported only
    when BOTH factors were measured, because a product with an assumed factor is an assumption wearing
    a measurement's units.

    The headline ``state`` follows the LANE figure alone. That is the falsifier: a fully resident core
    running fully divergent warps is the case this instrument exists to name, and calling the result
    "measured" on the strength of the residency figure would let it pass with the lane figure unknown.
    """
    r = dict(readings or {})
    lane = _lane_figure(r, geometry)
    warp = _warp_figure(r, geometry)
    if lane["state"] == "measured" and warp["state"] == "measured":
        simt = {"state": "measured", "value": lane["value"] * warp["value"],
                "note": "lane occupancy times warp residency: the fraction of a core's lane-slot "
                        "capacity that carried work"}
    else:
        unmeasured = [n for n, f in (("lane", lane), ("warp", warp)) if f["state"] != "measured"]
        simt = {"state": "unknown", "value": None,
                "why": f"the {unmeasured} figure(s) are unknown, and a product with an assumed factor "
                       f"is an assumption wearing a measurement's units"}
    return {"state": lane["state"], "lane": lane, "warp": warp, "simt": simt,
            "geometry": geometry.to_dict(),
            "why": lane.get("why", "")}
