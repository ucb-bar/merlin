"""Lane and warp occupancy on a SIMT cluster, from the machine's OWN geometry declaration.

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

WHERE THE DENOMINATOR COMES FROM, and why it is not the RTL
-----------------------------------------------------------
The elaborated RTL is the usual authority for geometry, and for this class of target it is not the
available one: the fact bundles for the SIMT targets here are empty, and the substrate that produces
cycles for them at all is a configurable cycle model, not an elaboration. That model reads a
**versioned TOML declaration of the machine** -- lanes, warps per core, cores -- and that declaration
is what the measured cycles actually belong to. So the geometry is derived from the config the
simulator itself is driven by, and the derived figure is labelled with the file and the keys it came
from, because a lane count that does not name its source cannot be checked against the run it prices.

The keys are FACTORED OUT rather than typed, exactly as :mod:`merlin.perf.hw_counters` factors engine
tokens out of counter names: a key names a SIMT dimension when one of its underscore-separated tokens
is a lane / warp / core word, so ``num_lanes``, ``lanes``, ``lane_width`` and ``lane_count`` all
resolve, and a config that spells its dimensions differently is served by the same derivation. Two
keys in one table claiming the same dimension with different values resolve to NOTHING -- an
ambiguous declaration is undeterminable, never a coin flip.

WHAT THIS DOES NOT DO
---------------------
It derives the instrument and computes the ratios from supplied readings. It does not run a
simulation, install a per-cycle callback, or claim a measurement. **With no execution trace there is
no occupancy**, and this module says so with the reason rather than returning a full-lane 1.0 -- the
reading that flatters a divergent kernel into looking dense, and the exact failure this instrument
was built to catch.

Three states throughout, never two: ``derived`` (read from a declaration), ``absent`` (the
declaration was read and describes no SIMT geometry -- a fact about that config), and ``unavailable``
(nothing could be located or read -- UNKNOWN, which is not absent).
"""
from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = [
    "ACTIVE_LANES_PER_ISSUE",
    "ACTIVE_LANE_CYCLES",
    "RESIDENT_WARPS",
    "WARP_ISSUES",
    "SimtGeometry",
    "derive_simt_geometry",
    "geometry_for_target",
    "occupancy_for_target",
    "occupancy_from_readings",
]

# --------------------------------------------------------------------------------------------
# The dimension vocabulary.
#
# These are ROLE words, not key names and not a target's spelling: a config key names a SIMT
# dimension when one of its underscore-separated tokens is one of these (singular, after a trailing
# plural "s" is stripped). Writing this config's exact key names here instead would make the module
# that config's reader, which is the overfit the repo's cardinal rule exists to prevent -- and it
# would silently return UNKNOWN for the next SIMT machine that spells `lane_count`.
# --------------------------------------------------------------------------------------------
_LANE = "lane"
_WARP = "warp"
_CORE = "core"
#: Reported in this order; ``_LANE`` first because it is the divergence denominator.
ROLES: tuple[str, ...] = (_LANE, _WARP, _CORE)

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


def _role_of(key: str) -> str | None:
    """The SIMT dimension ``key`` names, or ``None``.

    Structural, per the repo's no-regex rule: split on ``_``, strip a trailing plural, compare to the
    role words. A key matching two roles at once (``lanes_per_warp``) is deliberately NOT resolved
    here -- it is a ratio, not a count, and reading it as either dimension is off by the other one.
    """
    hits = {r for tok in key.split("_") for r in ROLES
            if (tok[:-1] if tok.endswith("s") and len(tok) > 1 else tok) == r}
    return next(iter(hits)) if len(hits) == 1 else None


# --------------------------------------------------------------------------------------------
# The geometry
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimtGeometry:
    """The SIMT dimensions one machine declaration establishes, and which keys established them.

    Every dimension is tri-state. ``None`` is "this declaration does not say", never a default: a
    lane width nobody wrote down is the one number an occupancy figure cannot borrow from elsewhere,
    and substituting a plausible sixteen would price a run against a machine that was never run.
    """

    #: Dotted path of the table the dimensions were read from (``""`` for the document root).
    table: str = ""
    lane_width: int | None = None
    warps_per_core: int | None = None
    cores: int | None = None
    #: ``role -> the key name actually read``, so a reader can check the derivation against the file.
    keys: dict[str, str] = field(default_factory=dict)
    #: Roles refused because two keys in the same table claimed them with different values.
    ambiguous: tuple[str, ...] = ()
    #: Where the declaration was read from, when it came from a file.
    source: str | None = None

    def value(self, role: str) -> int | None:
        return {_LANE: self.lane_width, _WARP: self.warps_per_core, _CORE: self.cores}.get(role)

    def resolved(self) -> tuple[str, ...]:
        return tuple(r for r in ROLES if self.value(r) is not None)

    @property
    def threads_per_core(self) -> int | None:
        """Lane slots one core can present per cycle. ``None`` unless BOTH factors are declared."""
        if self.lane_width is None or self.warps_per_core is None:
            return None
        return self.lane_width * self.warps_per_core

    @property
    def lane_slots_per_cycle(self) -> int | None:
        """Lane slots the whole cluster presents per cycle. ``None`` unless all three are declared."""
        per_core = self.threads_per_core
        if per_core is None or self.cores is None:
            return None
        return per_core * self.cores

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table, "lane_width": self.lane_width,
                "warps_per_core": self.warps_per_core, "cores": self.cores,
                "threads_per_core": self.threads_per_core,
                "lane_slots_per_cycle": self.lane_slots_per_cycle,
                "keys": dict(sorted(self.keys.items())), "ambiguous": list(self.ambiguous),
                "source": self.source, "resolved": list(self.resolved())}


def _tables(doc: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Mapping[str, Any]]]:
    """Every table in the document, the root included, as ``(dotted path, table)``.

    Walked rather than assumed to be one level deep: a config may declare the machine at the root, in
    a top-level table, or nested inside one, and picking a fixed depth would report UNKNOWN for a
    layout that differs only in nesting.
    """
    out = [(prefix, doc)]
    for key, val in doc.items():
        if isinstance(val, Mapping):
            out.extend(_tables(val, f"{prefix}.{key}" if prefix else str(key)))
    return out


def _geometry_of_table(path: str, table: Mapping[str, Any]) -> SimtGeometry:
    """Resolve the SIMT dimensions one table declares, refusing where it contradicts itself."""
    found: dict[str, dict[str, int]] = {r: {} for r in ROLES}
    for key, val in table.items():
        # bool is an int subclass and a flag is not a count -- admitting one would make `warp_sync =
        # true` a warp count of 1, which reads as a fully occupied core.
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            continue
        role = _role_of(str(key))
        if role is not None:
            found[role][str(key)] = int(val)
    vals: dict[str, int | None] = {}
    keys: dict[str, str] = {}
    ambiguous: list[str] = []
    for role in ROLES:
        distinct = set(found[role].values())
        if len(distinct) == 1:
            key = sorted(found[role])[0]
            vals[role], keys[role] = found[role][key], key
        elif len(distinct) > 1:
            # TWO ANSWERS IS NOT AN ANSWER. Picking the first (or the largest) would publish a
            # denominator nobody wrote, and an occupancy figure is only as honest as its denominator.
            ambiguous.append(role)
    return SimtGeometry(table=path, lane_width=vals.get(_LANE), warps_per_core=vals.get(_WARP),
                        cores=vals.get(_CORE), keys=keys, ambiguous=tuple(ambiguous))


def derive_simt_geometry(text: str) -> SimtGeometry:
    """Factor a machine declaration's TOML into SIMT dimensions.

    Parsed with :mod:`tomllib` -- a real parser, not line matching, so a value spread over lines or
    quoted unusually is read as the file's author wrote it. Returns the table that resolves the most
    dimensions; an empty :class:`SimtGeometry` when the document declares none, and also when two
    tables tie, because a tie means the file describes two machines and nothing here can say which
    one the cycles belong to.
    """
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return SimtGeometry()
    scored = [(len(g.resolved()), path, g) for path, table in _tables(doc)
              for g in (_geometry_of_table(path, table),)]
    best = max((n for n, _p, _g in scored), default=0)
    if best == 0:
        # Nothing resolved -- but a table that CONTRADICTED ITSELF is a different report from a file
        # that simply says nothing about lanes, and the caller separates absent from undeterminable on
        # exactly that distinction. Carry the ambiguity out rather than flattening both to empty.
        contested = [g for _n, _p, g in scored if g.ambiguous]
        return contested[0] if contested else SimtGeometry()
    winners = [g for n, _p, g in scored if n == best]
    if len(winners) > 1:
        return SimtGeometry()
    return winners[0]


# --------------------------------------------------------------------------------------------
# Locating the declaration, without naming a target or typing a path
# --------------------------------------------------------------------------------------------

#: The accessor a target's own package exposes to say where its machine declaration lives. Asking the
#: target rather than holding a path is what keeps this module target-agnostic: the value is resolved
#: by that package from the descriptor / dotenv, and moving the checkout moves the answer with it.
_CONFIG_ACCESSOR = "config_path"

#: Suffix of a machine declaration, used to recognise one among the paths a facts artifact records.
_CONFIG_SUFFIX = ".toml"


def _from_backend(name: str) -> tuple[Path | None, str]:
    """``(path, how)`` from the runtime backend registered under ``name``, if it declares one."""
    try:
        from merlin.runtime.backends.base import get_backend
        mod = get_backend(name)
    except Exception as e:                                     # noqa: BLE001 -- no such backend
        return None, f"no runtime backend is registered for {name!r} ({type(e).__name__})"
    accessor = getattr(mod, _CONFIG_ACCESSOR, None)
    if accessor is None:
        return None, f"the {name!r} backend exposes no {_CONFIG_ACCESSOR}()"
    try:
        return Path(str(accessor())), f"{name!r} backend {_CONFIG_ACCESSOR}()"
    except Exception as e:                                     # noqa: BLE001
        return None, f"{name!r} backend {_CONFIG_ACCESSOR}() raised {type(e).__name__}"


def _config_candidates(target: str) -> tuple[list[tuple[Path, str]], list[str]]:
    """``([(path, how)], [why each route failed])`` -- where this target's declaration might live.

    Three routes, each of which asks something that already knows, and none of which holds a path:

    1. **The target's own runtime backend.** A reference target's package resolves its external
       toolchain from the descriptor/dotenv already; if it exposes the accessor, it is the authority.
    2. **The SIMT RTL introspect registered for this target**, then the backend of the identity that
       introspect DECLARES it serves. This is the route a composite target needs: a cluster whose
       machine declaration belongs to the core inside it resolves through that core rather than
       reporting UNKNOWN because nothing is registered under the cluster's own name.
    3. **The target's facts artifact**, when its recorded inputs name a declaration file. Weakest but
       most direct: it is what an extraction says it actually read.
    """
    found: list[tuple[Path, str]] = []
    why: list[str] = []

    path, how = _from_backend(target)
    if path is not None:
        found.append((path, how))
    else:
        why.append(how)

    intro = None
    try:
        from merlin.targetgen.rtl import mlc_bridge
        intro = mlc_bridge._resolve_simt_introspect(target)
    except Exception as e:                                     # noqa: BLE001 -- registry unreachable
        why.append(f"the SIMT introspect registry is unreachable ({type(e).__name__})")
    if intro is None:
        why.append(f"no SIMT RTL introspect is registered for {target!r}")
    else:
        served = str(getattr(intro, "TARGET", "") or "")
        if served and served != target:
            path, how = _from_backend(served)
            if path is not None:
                found.append((path, f"{how}, via the SIMT introspect serving {target!r}"))
            else:
                why.append(how)

    try:
        from merlin.targetgen.rtl.facts import rtl_facts_path
        doc = json.loads(Path(rtl_facts_path(target)).read_text(encoding="utf-8"))
        inputs = doc.get("inputs") if isinstance(doc.get("inputs"), Mapping) else {}
        hits = [str(v) for v in (inputs or {}).values()
                if isinstance(v, str) and v.endswith(_CONFIG_SUFFIX)]
        if hits:
            found.extend((Path(h), "recorded in this target's facts artifact inputs") for h in hits)
        else:
            why.append("this target's facts artifact records no declaration file among its inputs")
    except Exception as e:                                     # noqa: BLE001 -- absent facts are an answer
        why.append(f"no readable facts artifact for {target!r} ({type(e).__name__})")

    seen: set[str] = set()
    uniq = [(p, how) for p, how in found if not (str(p) in seen or seen.add(str(p)))]
    return uniq, why


def geometry_for_target(target: str, *, config_text: str | None = None,
                        config_path: str | Path | None = None) -> dict[str, Any]:
    """Derive ``target``'s SIMT geometry from the declaration its own package points at.

    Three states. ``derived`` when a declaration resolved at least the lane width -- the one dimension
    without which no occupancy figure has a denominator; ``absent`` when a declaration WAS READ and
    declares no SIMT dimensions (a real fact about that file: it does not describe a SIMT machine);
    ``unavailable`` when nothing could be located or read, which is UNKNOWN and not absent.

    ``config_text`` / ``config_path`` override the search, which is how a caller supplies a machine
    that is not on this host -- and how a test pins that a declaration missing a dimension reports it
    missing instead of falling back to a plausible one.
    """
    if config_text is not None:
        geom = derive_simt_geometry(config_text)
        return _state(geom, source=(str(config_path) if config_path else "<supplied text>"),
                      read=["<supplied text>"], unread={}, routes=[])

    if config_path is not None:
        candidates: list[tuple[Path, str]] = [(Path(config_path), "supplied by the caller")]
        routes: list[str] = []
    else:
        candidates, routes = _config_candidates(target)
    if not candidates:
        return {"status": "unavailable", "target": target, "routes_tried": routes,
                "why": f"no machine declaration could be located for {target!r}; whether it declares a "
                       f"SIMT geometry is UNKNOWN, not absent"}

    best, where = SimtGeometry(), None
    read: list[str] = []
    unread: dict[str, str] = {}
    for path, how in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            # A placeholder path a package emits when its toolchain env is unset lands here, and it
            # must read as "we could not look", never as "the machine declares no lanes".
            unread[str(path)] = f"{type(e).__name__}: {e}"
            continue
        read.append(f"{path} ({how})")
        got = derive_simt_geometry(text)
        # Rank by dimensions resolved, and -- at equal rank -- prefer a candidate that recorded a
        # CONTRADICTION over one that recorded nothing, so the refusal keeps its reason instead of
        # degrading into "this file declares no lanes", which is a different report entirely.
        rank, incumbent = (len(got.resolved()), bool(got.ambiguous)), \
                          (len(best.resolved()), bool(best.ambiguous))
        if rank > incumbent:
            best, where = got, path
    return _state(best, source=(str(where) if where else None), read=read, unread=unread,
                  routes=routes)


def _state(geom: SimtGeometry, *, source: str | None, read: list[str], unread: dict[str, str],
           routes: list[str]) -> dict[str, Any]:
    """Wrap a derivation in the three-state envelope, keeping ABSENT and UNAVAILABLE apart."""
    # Stamp WHICH declaration this geometry is, onto the geometry itself: a lane count that travels
    # without its source cannot be checked against the run it prices, and these numbers are quoted.
    geom = replace(geom, source=source)
    if geom.lane_width is None:
        # ABSENT REQUIRES HAVING READ SOMETHING. Falling through to "this machine declares no SIMT
        # geometry" when every candidate failed to open would report our inability to look as a
        # property of the machine -- the same collapse hw_counters guards against.
        if not read:
            return {"status": "unavailable", "read": read, "unreadable": unread,
                    "routes_tried": routes,
                    "why": "no candidate declaration could be READ, so whether this target declares a "
                           "SIMT geometry is UNKNOWN, not absent"}
        if geom.ambiguous:
            return {"status": "unavailable", "read": read, "unreadable": unread,
                    "geometry": geom.to_dict(),
                    "why": f"the declaration gives more than one value for {list(geom.ambiguous)}; two "
                           f"answers is not an answer, and a guessed denominator would price the run "
                           f"against a machine that was never run"}
        return {"status": "absent", "read": read, "unreadable": unread,
                "why": "the declaration was read and names no lane dimension, so it does not describe "
                       "a SIMT machine and there is no lane denominator to occupy"}
    return {"status": "derived", "source": source, "read": read, "unreadable": unread,
            "geometry": geom.to_dict(),
            "missing": [r for r in ROLES if geom.value(r) is None],
            "note": ("lane_width is the divergence denominator; warps_per_core and cores extend it to "
                     "the core and the cluster, and each is UNKNOWN on its own when undeclared")}


# --------------------------------------------------------------------------------------------
# The arithmetic
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
                "why": "the declaration names no warps-per-core, so the ratio has no denominator"}
    resident = readings.get(RESIDENT_WARPS)
    if resident is None:
        return {"state": "unknown", "value": None,
                "why": f"no trace supplied {RESIDENT_WARPS!r}; an unobserved warp count is UNKNOWN, and "
                       f"assuming a full core would hide a kernel that launched one warp"}
    resident = int(resident)
    if resident < 0 or resident > geom.warps_per_core:
        return {"state": "unknown", "value": None,
                "why": f"{resident} resident warp(s) is outside the {geom.warps_per_core} the "
                       f"declaration says a core holds: the trace and the geometry disagree"}
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


def occupancy_for_target(target: str, readings: Mapping[str, Any] | None = None, *,
                         config_text: str | None = None,
                         config_path: str | Path | None = None) -> dict[str, Any]:
    """Derive ``target``'s geometry and, if readings are supplied, its occupancy.

    The end-to-end call, and the one that most often returns UNKNOWN: on a target with a declaration
    but no trace the geometry resolves and every ratio refuses, which is the honest report and not a
    failure. Nothing here runs the machine.
    """
    geo = geometry_for_target(target, config_text=config_text, config_path=config_path)
    if geo.get("status") != "derived":
        return {"target": target, "state": "unknown", "geometry_status": geo.get("status"),
                "geometry": geo,
                "why": f"no SIMT geometry is derived for {target!r} ({geo.get('why')}), so no "
                       f"occupancy denominator exists"}
    g = geo["geometry"]
    geom = SimtGeometry(table=g["table"], lane_width=g["lane_width"],
                        warps_per_core=g["warps_per_core"], cores=g["cores"],
                        keys=dict(g["keys"]), ambiguous=tuple(g["ambiguous"]),
                        source=geo.get("source"))
    out = occupancy_from_readings(readings, geom)
    out["target"] = target
    out["geometry_status"] = "derived"
    out["source"] = geo.get("source")
    return out
