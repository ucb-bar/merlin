"""Joint occupancy from the target's OWN hardware counters, when it counts overlap itself.

Realised overlap normally needs a per-cycle trace, and a per-cycle trace normally needs a waveform
build or a co-simulation model. On a target whose RTL carries performance counters it needs neither:
the hardware can already count the cycles in which each combination of engines was busy, and a
combination counter IS a joint-occupancy reading.

Measured on the interlocked target here, from its own shipped counter header — seven counters that
between them partition busy time over three engines:

    <prefix>_LD_CYCLES  <prefix>_ST_CYCLES  <prefix>_EX_CYCLES          the three singles
    <prefix>_LD_ST_CYCLES  <prefix>_LD_EX_CYCLES  <prefix>_ST_EX_CYCLES the three pairs
    <prefix>_LD_ST_EX_CYCLES                                            all three at once

⚠️ **The engine tokens are FACTORED OUT of the counter names, not typed here.** A target that spells
its engines differently, or has two of them, or five, is served by the same derivation: find the
counters whose names share a prefix and a suffix and differ only in a set of tokens, then the tokens
ARE the engines and the token-set size is the combination order. Writing this target's spellings into
this module would make it that target's counter reader, which is the overfit the repo's cardinal rule
exists to prevent.

**What this does NOT do.** It reads the header to learn what the hardware can count, and it computes η
from a set of counter VALUES. It does not run a program, emit the counter-read commands, or claim a
measurement: a caller supplies the values, and an absent value is UNKNOWN rather than zero. Wiring the
reads into a capsule's command stream is a separate, invasive step in the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

#: A counter whose name ends in this token counts CYCLES, which is the only kind this module reads: a
#: joint-occupancy figure is a duration, and an event count is not one.
_CYCLES = "CYCLES"


@dataclass(frozen=True)
class OccupancyCounters:
    """The combination counters one target exposes, and the engines they are over."""

    prefix: str = ""
    engines: tuple = ()
    #: ``frozenset(engine tokens) -> counter name``
    by_combination: dict = field(default_factory=dict)

    def singles(self) -> dict:
        return {next(iter(k)): v for k, v in self.by_combination.items() if len(k) == 1}

    def overlaps(self) -> dict:
        return {k: v for k, v in self.by_combination.items() if len(k) >= 2}

    def complete(self) -> bool:
        """True when every combination of the derived engines has a counter.

        A partial set is usable but not complete, and the difference matters: with a pair missing, the
        realised-overlap total is a LOWER BOUND, and reporting it as the total understates η.
        """
        want = sum(1 for r in range(1, len(self.engines) + 1)
                   for _ in combinations(self.engines, r))
        return len(self.by_combination) == want and bool(self.engines)

    def to_dict(self) -> dict:
        return {"prefix": self.prefix, "engines": list(self.engines),
                "by_combination": {"+".join(sorted(k)): v
                                   for k, v in sorted(self.by_combination.items(),
                                                      key=lambda kv: sorted(kv[0]))},
                "complete": self.complete()}


def _defines(text: str) -> dict:
    """``NAME -> int`` for every object-like ``#define NAME <integer>``, parsed structurally."""
    out: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("#define"):
            continue
        parts = line.split()
        if len(parts) < 3 or "(" in parts[1]:
            continue
        name, value = parts[1], parts[2]
        neg = value.startswith("-")
        digits = value[1:] if neg else value
        if digits.isdigit():
            out[name] = -int(digits) if neg else int(digits)
    return out


def derive_occupancy_counters(text: str) -> OccupancyCounters:
    """Factor a counter header's combination counters into engines and combinations.

    The derivation: among ``<PREFIX>_<tokens...>_CYCLES`` names sharing one prefix, the union of the
    single-token names' tokens is the engine set, and every other name over those tokens is a
    combination. Requiring the singles to exist is deliberate — without them there is no per-engine
    busy figure, so η has no denominator and the reading is not an occupancy vector at all.
    """
    names = [n for n in _defines(text) if n.endswith("_" + _CYCLES)]
    # Group by first token, which is the family prefix the header uses for one counter block.
    groups: dict = {}
    for n in names:
        toks = n.split("_")
        if len(toks) < 3:
            continue                                            # <PREFIX>_CYCLES carries no tokens
        groups.setdefault(toks[0], []).append((n, tuple(toks[1:-1])))
    best = OccupancyCounters()
    for prefix, entries in sorted(groups.items()):
        singles = {t[0] for _n, t in entries if len(t) == 1}
        if len(singles) < 2:
            continue                                            # one engine cannot overlap with itself
        combos: dict = {}
        for name, toks in entries:
            if toks and set(toks) <= singles and len(set(toks)) == len(toks):
                combos[frozenset(toks)] = name
        got = OccupancyCounters(prefix=prefix, engines=tuple(sorted(singles)), by_combination=combos)
        # Prefer the block that resolves the most combinations; a header may carry several.
        if len(got.by_combination) > len(best.by_combination):
            best = got
    return best


def counters_for_target(target: str, *, sources=None) -> dict:
    """Derive the combination counters from ``target``'s own shipped counter header.

    Three states. ``derived`` when a block resolved; ``absent`` when the headers were read and expose
    no combination counters (a real fact about that target); ``unavailable`` when no header could be
    read at all — which is NOT the same as the target having none.
    """
    paths = list(sources or ())
    if not paths:
        try:
            from merlin.targetgen import capability_discovery as CD
            for s in CD.isa_sources(target) or ():
                p = Path(str(getattr(s, "path", s)))
                if p.is_file():
                    paths.append(p)
                    # Sibling headers of the same shipped set, e.g. a dedicated counter header.
                    paths.extend(sorted(q for q in p.parent.glob("*.h") if q.is_file()))
        except Exception as e:                                 # noqa: BLE001
            return {"status": "unavailable", "why": f"{type(e).__name__}: {str(e)[:120]}"}
    seen, uniq = set(), []
    for p in paths:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(Path(p))
    if not uniq:
        return {"status": "unavailable",
                "why": "no shipped header could be located for this target; whether it exposes "
                       "combination counters is UNKNOWN, not absent"}
    best, where = OccupancyCounters(), None
    read, unread = [], {}
    for p in uniq:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            unread[str(p)] = f"{type(e).__name__}: {e}"
            continue
        read.append(str(p))
        got = derive_occupancy_counters(text)
        if len(got.by_combination) > len(best.by_combination):
            best, where = got, p
    if not best.by_combination:
        # ABSENT REQUIRES HAVING READ SOMETHING. Falling through to "this target does not count
        # overlap in hardware" when every candidate header failed to open is precisely the collapse
        # this module exists to prevent -- our inability to read reported as a property of the
        # machine. Caught by its own test, which passed a path that does not exist.
        if not read:
            return {"status": "unavailable", "unreadable": unread,
                    "why": "no candidate header could be READ, so whether this target exposes "
                           "combination counters is UNKNOWN, not absent"}
        return {"status": "absent", "read": read, "unreadable": unread,
                "why": "the shipped headers expose no counter block with per-engine singles and a "
                       "combination over them, so this target does not count overlap in hardware"}
    return {"status": "derived", "header": str(where), "counters": best.to_dict()}


def eta_from_counters(values: dict, counters: OccupancyCounters) -> dict:
    """η from a set of counter READINGS, with every refusal carrying its reason.

    ``realised`` is the total cycles in which two or more engines were busy at once. ``available`` is
    the second-largest per-engine busy total — deliberately the same quantity ``headroom`` and the
    falsifier use, so this η and theirs are one number rather than two that share a name.

    A per-engine busy total must include the cycles that engine spent overlapping, so a single counter
    that counts "this engine ALONE" is summed with every combination containing it. Reading the singles
    as whole-engine totals instead understates the busiest engine and inflates η.
    """
    missing = sorted(n for n in counters.by_combination.values() if n not in (values or {}))
    if missing:
        return {"state": "unknown", "eta": None,
                "why": f"{len(missing)} counter reading(s) absent ({missing[:4]}); a missing counter "
                       f"is UNKNOWN, and treating it as zero would report overlap that was never "
                       f"measured as overlap that did not happen"}
    busy: dict = {e: 0 for e in counters.engines}
    realised = 0
    for combo, name in counters.by_combination.items():
        v = int((values or {})[name])
        for e in combo:
            busy[e] = busy.get(e, 0) + v
        if len(combo) >= 2:
            realised += v
    ordered = sorted(busy.values(), reverse=True)
    available = ordered[1] if len(ordered) >= 2 else 0
    if available <= 0:
        return {"state": "unknown", "eta": None, "busy_cycles": busy, "realised_cycles": realised,
                "why": "the second-busiest engine has no busy cycles, so no overlap was AVAILABLE; "
                       "0/0 is undefined, not 0.0"}
    return {"state": "measured", "eta": realised / float(available),
            "busy_cycles": busy, "realised_cycles": realised, "available_cycles": available,
            "complete": counters.complete(),
            "note": ("realised counts every cycle with two or more engines busy; available is the "
                     "second-largest per-engine busy total, the same quantity headroom and the "
                     "falsifier use")}
