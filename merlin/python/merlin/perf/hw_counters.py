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

import ast
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
from pathlib import Path
from collections.abc import Mapping

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


def _module_lines(hw_text: str, module: str) -> tuple[list[str] | None, str | None]:
    """Return one balanced CIRCT ``hw.module`` body without regex or target assumptions."""
    marker = "@" + module + "("
    lines = (hw_text or "").splitlines()
    for at, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("hw.module") or marker not in line:
            continue
        depth = raw.count("{") - raw.count("}")
        body = [raw]
        cursor = at + 1
        while cursor < len(lines) and depth > 0:
            body.append(lines[cursor])
            depth += lines[cursor].count("{") - lines[cursor].count("}")
            cursor += 1
        if depth:
            return None, f"hw.module @{module} has unbalanced braces"
        return body, None
    return None, f"hw.module @{module} is absent"


def _counter_fingerprint(counters: OccupancyCounters, codes: Mapping[str, int]) -> str:
    rows = []
    for combo, name in sorted(counters.by_combination.items(), key=lambda item: sorted(item[0])):
        rows.append("+".join(sorted(combo)) + "=" + name + "=" + str(codes.get(name)))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _operand_refs(rhs: str, opcode: str) -> tuple[str, ...] | None:
    prefix = opcode + " "
    if not rhs.startswith(prefix):
        return None
    operands = rhs[len(prefix):]
    for marker in (" {", " :"):
        operands = operands.split(marker, 1)[0]
    if operands.startswith("bin "):
        operands = operands[4:]
    refs = tuple(part.strip() for part in operands.split(",") if part.strip())
    return refs or None


def prove_occupancy_partition_from_circt(
        hw_text: str, counters: OccupancyCounters, codes: Mapping[str, int], *,
        module: str, counter_module: str, source: str | None = None) -> dict:
    """Prove that counter events are the exhaustive one-hot partition of engine-busy states.

    Header names only suggest combination semantics.  This verifier follows the corresponding numeric
    event ports into elaborated CIRCT, symbolically evaluates their ``comb.and/or/xor`` cones, and
    requires every combination counter to be true on exactly its named busy-bit valuation.  Engine and
    signal names are inferred; callers provide only target-owned module identities and artifacts.
    """
    digest = hashlib.sha256((hw_text or "").encode("utf-8")).hexdigest()
    base = {"status": "unknown", "method": "circt_boolean_partition_v1",
            "source": source, "sha256": digest, "module": module,
            "counter_module": counter_module,
            "counter_fingerprint": _counter_fingerprint(counters, codes)}
    if not counters.complete():
        return {**base, "why": "the header does not expose every non-empty engine combination"}
    if any(name not in codes for name in counters.by_combination.values()):
        return {**base, "why": "one or more occupancy counters has no derived event code"}
    selected_codes = [codes[name] for name in counters.by_combination.values()]
    if any(isinstance(code, bool) or not isinstance(code, int) or code < 0
           for code in selected_codes):
        return {**base, "why": "one or more occupancy event codes is not a non-negative integer"}
    if len(set(selected_codes)) != len(selected_codes):
        return {**base, "why": "occupancy counters do not have unique event codes"}
    body, error = _module_lines(hw_text, module)
    if body is None:
        return {**base, "why": error}

    definitions: dict[str, str] = {}
    instance_line = None
    instance_marker = "@" + counter_module + "("
    for raw in body:
        stripped = raw.strip()
        lhs, sep, rhs = stripped.partition(" = ")
        if sep and lhs.startswith("%"):
            definitions[lhs] = rhs
        if "hw.instance" in stripped and instance_marker in stripped:
            if instance_line is not None:
                return {**base, "why": f"multiple @{counter_module} instances are ambiguous"}
            instance_line = stripped
    if instance_line is None:
        return {**base, "why": f"no @{counter_module} instance exists in @{module}"}

    event_refs: dict[frozenset, str] = {}
    for combo, name in counters.by_combination.items():
        port = "io_event_io_event_signal_" + str(codes[name]) + ": "
        if instance_line.count(port) != 1:
            return {**base, "why": f"event port for {name!r} is absent or ambiguous"}
        tail = instance_line.split(port, 1)[1]
        ref = tail.split(":", 1)[0].strip()
        if not ref.startswith("%"):
            return {**base, "why": f"event port for {name!r} is not driven by an SSA value"}
        event_refs[combo] = ref

    leaves: set[str] = set()
    visiting: set[str] = set()

    def discover(ref: str) -> bool:
        if ref in ("%true", "%false"):
            return True
        if ref in visiting:
            return False
        rhs = definitions.get(ref)
        if rhs is None:
            leaves.add(ref)
            return True
        visiting.add(ref)
        refs = (_operand_refs(rhs, "comb.and") or _operand_refs(rhs, "comb.or")
                or _operand_refs(rhs, "comb.xor"))
        if refs is None:
            visiting.remove(ref)
            return False
        ok = all(discover(operand) for operand in refs)
        visiting.remove(ref)
        return ok

    if not all(discover(ref) for ref in event_refs.values()):
        return {**base, "why": "an occupancy event cone uses unsupported or cyclic CIRCT logic"}
    if len(leaves) != len(counters.engines):
        return {**base, "why": (f"event cones have {len(leaves)} independent boolean leaves for "
                                  f"{len(counters.engines)} header-derived engines")}
    ordered_leaves = tuple(sorted(leaves))

    def evaluate(ref: str, values: Mapping[str, bool], active: set[str]) -> bool:
        if ref == "%true":
            return True
        if ref == "%false":
            return False
        if ref in values:
            return values[ref]
        if ref in active:
            raise ValueError("cyclic boolean cone")
        rhs = definitions[ref]
        active.add(ref)
        refs = _operand_refs(rhs, "comb.and")
        if refs is not None:
            result = all(evaluate(item, values, active) for item in refs)
        else:
            refs = _operand_refs(rhs, "comb.or")
            if refs is not None:
                result = any(evaluate(item, values, active) for item in refs)
            else:
                refs = _operand_refs(rhs, "comb.xor")
                if refs is None:
                    raise ValueError("unsupported boolean operation")
                result = False
                for item in refs:
                    result ^= evaluate(item, values, active)
        active.remove(ref)
        return result

    satisfying: dict[frozenset, list[frozenset[str]]] = {combo: [] for combo in event_refs}
    for mask in range(1 << len(ordered_leaves)):
        values = {leaf: bool(mask & (1 << index)) for index, leaf in enumerate(ordered_leaves)}
        enabled = frozenset(leaf for leaf, value in values.items() if value)
        for combo, ref in event_refs.items():
            try:
                if evaluate(ref, values, set()):
                    satisfying[combo].append(enabled)
            except (KeyError, ValueError) as exc:
                return {**base, "why": f"could not evaluate event cone: {exc}"}

    engine_to_leaf: dict[str, str] = {}
    for combo, states in satisfying.items():
        if len(combo) != 1:
            continue
        if len(states) != 1 or len(states[0]) != 1:
            return {**base, "why": "a singleton counter is not one exact busy-bit valuation"}
        engine_to_leaf[next(iter(combo))] = next(iter(states[0]))
    if len(engine_to_leaf) != len(counters.engines) or len(set(engine_to_leaf.values())) != len(leaves):
        return {**base, "why": "singleton counters do not bijectively identify engine busy bits"}
    for combo, states in satisfying.items():
        expected = frozenset(engine_to_leaf[engine] for engine in combo)
        if states != [expected]:
            return {**base, "why": (f"counter for {sorted(combo)} is not exactly its named "
                                      "busy-bit valuation")}
    return {**base, "status": "proved", "engine_leaves": engine_to_leaf,
            "events": {"+".join(sorted(combo)): event_refs[combo]
                       for combo in sorted(event_refs, key=lambda item: sorted(item))}}


def _define_int_expr(node: ast.AST, values: dict[str, int]) -> int:
    """Evaluate the integer-only subset used by shipped counter headers."""
    if isinstance(node, ast.Expression):
        return _define_int_expr(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
        value = _define_int_expr(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        return ~value
    if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.LShift, ast.RShift,
                      ast.BitOr, ast.BitAnd, ast.BitXor)):
        lhs, rhs = _define_int_expr(node.left, values), _define_int_expr(node.right, values)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.FloorDiv):
            return lhs // rhs
        if isinstance(node.op, ast.LShift):
            return lhs << rhs
        if isinstance(node.op, ast.RShift):
            return lhs >> rhs
        if isinstance(node.op, ast.BitOr):
            return lhs | rhs
        if isinstance(node.op, ast.BitAnd):
            return lhs & rhs
        return lhs ^ rhs
    raise ValueError("unsupported counter-header integer expression")


def _defines(text: str) -> dict:
    """``NAME -> int`` for object-like integer defines, including derived expressions.

    Counter headers commonly allocate an incremental event range as ``BASE + offset``.  Reading only
    decimal literals silently drops precisely those later counters (including byte-volume events on
    the current RTL).  Expressions are resolved iteratively through a deliberately tiny AST evaluator;
    no C preprocessor, Python ``eval``, target spelling, or numeric fallback is involved.
    """
    pending: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line.startswith("#define"):
            continue
        parts = line.split()
        if len(parts) < 3 or "(" in parts[1]:
            continue
        pending[parts[1]] = " ".join(parts[2:])
    out: dict[str, int] = {}
    progressed = True
    while pending and progressed:
        progressed = False
        for name, expression in list(pending.items()):
            try:
                value = _define_int_expr(ast.parse(expression, mode="eval"), out)
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
                continue
            out[name] = int(value)
            del pending[name]
            progressed = True
    return out


def counters_with_unit(text: str, unit: str) -> dict[str, int]:
    """Return counters whose underscore-delimited name declares ``unit``.

    This is a unit query, not a target-specific name table: callers may ask for ``BYTES``, ``CYCLES``,
    or another unit the target itself puts in its counter API.  Direction/engine spellings remain in
    the returned names and are not interpreted here.
    """
    wanted = str(unit or "").strip().upper()
    if not wanted:
        raise ValueError("a non-empty counter unit is required")
    return {name: code for name, code in _defines(text).items()
            if wanted in name.upper().split("_")}


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
    best, where, where_digest, where_codes = OccupancyCounters(), None, None, None
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
            where_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            definitions = _defines(text)
            where_codes = {name: definitions[name] for name in got.by_combination.values()}
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
    return {"status": "derived", "header": str(where), "header_sha256": where_digest,
            "event_codes": where_codes, "counters": best.to_dict()}


def counter_slots_from_circt(
    hw_text: str,
    *,
    module: str,
    state_families: tuple[str, ...],
    source: str | None = None,
) -> dict:
    """Derive a counter-file's physical slot count from elaborated CIRCT HW.

    ``module`` and ``state_families`` identify target-owned structures; the caller obtains those
    identities at the target boundary.  This generic reader supplies no target name, register name,
    width, or count.  It finds ``seq.firreg`` SSA results named ``%<family>_<index>`` inside the named
    ``hw.module`` and accepts the result only when every requested family has the same dense index
    set starting at zero.  Requiring multiple independently elaborated state arrays prevents an
    address width (which can encode unused values) or one coincidental register name from being
    mistaken for capacity.

    The result deliberately has three states.  ``derived`` carries the exact count and evidence;
    ``unknown`` carries the refusal reason.  There is no numeric fallback.
    """
    digest = hashlib.sha256((hw_text or "").encode("utf-8")).hexdigest()
    provenance = {"source": source, "sha256": digest, "module": module,
                  "state_families": list(state_families), "method": "dense_seq_firreg_families"}
    if not module or not state_families:
        return {"status": "unknown", "slots": None, "provenance": provenance,
                "why": "a CIRCT module and at least one target-owned state family are required"}

    body, error = _module_lines(hw_text, module)
    if body is None:
        return {"status": "unknown", "slots": None, "provenance": provenance,
                "why": error}

    indices = {family: set() for family in state_families}
    for raw in body:
        line = raw.strip()
        lhs, sep, rhs = line.partition(" = ")
        if not sep or not rhs.startswith("seq.firreg "):
            continue
        for family in state_families:
            prefix = "%" + family + "_"
            if not lhs.startswith(prefix):
                continue
            suffix = lhs[len(prefix):]
            if suffix.isdigit():
                indices[family].add(int(suffix))

    missing = [family for family, got in indices.items() if not got]
    if missing:
        return {"status": "unknown", "slots": None, "provenance": provenance,
                "why": f"no indexed seq.firreg state was found for family/families {missing}"}
    sets = list(indices.values())
    if any(got != sets[0] for got in sets[1:]):
        return {"status": "unknown", "slots": None, "provenance": provenance,
                "why": "the independently elaborated counter state families disagree on their indices",
                "indices": {name: sorted(got) for name, got in indices.items()}}
    expected = set(range(len(sets[0])))
    if sets[0] != expected:
        return {"status": "unknown", "slots": None, "provenance": provenance,
                "why": "counter state indices are not a dense zero-based slot set",
                "indices": {name: sorted(got) for name, got in indices.items()}}
    return {"status": "derived", "slots": len(expected), "provenance": provenance,
            "evidence": {"indices": sorted(expected),
                         "families": {name: len(got) for name, got in indices.items()}}}


def eta_from_counters(values: dict, counters: OccupancyCounters, *, hw_text: str,
                      codes: Mapping[str, int], module: str, counter_module: str,
                      measurement_cycles: int | None = None,
                      source: str | None = None) -> dict:
    """η from a set of counter READINGS, with every refusal carrying its reason.

    ``realised`` is the total cycles in which two or more engines were busy at once. ``available`` is
    the second-largest per-engine busy total — deliberately the same quantity ``headroom`` and the
    falsifier use, so this η and theirs are one number rather than two that share a name.

    A per-engine busy total must include the cycles that engine spent overlapping, so a single counter
    is summed with every combination containing it. Reading the singles as whole-engine totals instead
    understates the busiest engine and inflates η.

    ⚠️ THAT SUM IS EXACT, NOT AN APPROXIMATION, and it is worth knowing why. Verified in the pinned RTL
    of the target measured here: the increment conditions are mutually exclusive by construction --
    each counter fires only on its own combination, with the other engines' busy bits explicitly
    negated (``ld.busy && !st.busy && !ex.busy`` for the load single, and so on up to
    ``ld.busy && st.busy && ex.busy``). So the seven counters PARTITION busy time rather than
    overlapping in their own accounting, no cycle is counted twice, and adding a single to the
    combinations containing it recovers that engine's true busy total.

    A target whose counters are NOT mutually exclusive would double-count, and this arithmetic would
    silently over-report both the per-engine totals and the realised overlap. Nothing here can detect
    that from the header alone -- the exclusivity is a property of the RTL, not of the ``#define`` list
    -- so a new target's counter block should be read once before its η is cited.
    """
    proof = prove_occupancy_partition_from_circt(
        hw_text, counters, codes, module=module, counter_module=counter_module, source=source)
    if proof.get("status") != "proved":
        return {"state": "unknown", "eta": None, "complete": counters.complete(),
                "partition_proof": proof,
                "why": "counter exclusivity/exhaustiveness was not proved from CIRCT: "
                       + str(proof.get("why", "unknown proof failure"))}
    required = set(counters.by_combination.values())
    supplied = set(values or {})
    missing = sorted(required - supplied)
    if missing:
        return {"state": "unknown", "eta": None, "partition_proof": proof,
                "why": f"{len(missing)} counter reading(s) absent ({missing[:4]}); a missing counter "
                       f"is UNKNOWN, and treating it as zero would report overlap that was never "
                       f"measured as overlap that did not happen"}
    extra = sorted(supplied - required)
    if extra:
        return {"state": "unknown", "eta": None, "partition_proof": proof,
                "why": f"unexpected counter reading(s) {extra[:4]}; the occupancy pass must contain "
                       "exactly its proved partition so readings from another window cannot be mixed"}
    invalid = sorted(name for name in required
                     if (isinstance(values[name], bool) or not isinstance(values[name], int)
                         or values[name] < 0))
    if invalid:
        return {"state": "unknown", "eta": None, "partition_proof": proof,
                "why": f"counter reading(s) {invalid[:4]} are not non-negative integers"}
    if (isinstance(measurement_cycles, bool) or not isinstance(measurement_cycles, int)
            or measurement_cycles <= 0):
        return {"state": "unknown", "eta": None, "partition_proof": proof,
                "why": "a positive integer cycle measurement for the identical counter window is "
                       "required to exclude impossible or wrapped occupancy readings"}
    partition_cycles = sum(values[name] for name in required)
    if partition_cycles > measurement_cycles:
        return {"state": "unknown", "eta": None, "partition_proof": proof,
                "why": f"the exclusive occupancy partition totals {partition_cycles} cycles, more "
                       f"than its measured {measurement_cycles}-cycle window; the readings are "
                       "mixed, corrupt, or wrapped"}
    busy: dict = {e: 0 for e in counters.engines}
    realised = 0
    for combo, name in counters.by_combination.items():
        v = values[name]
        for e in combo:
            busy[e] = busy.get(e, 0) + v
        if len(combo) >= 2:
            realised += v
    # HOW MUCH OVERLAP WAS AVAILABLE, for any number of engines.
    #
    # The second-largest per-engine total is the right bound for TWO engines and is wrong for three:
    # with three engines overlapping in disjoint pairs the numerator counts every pair while the
    # denominator only admits the top pair's ceiling, and η comes out above 1. Measured on the first
    # real run of this instrument -- a bit-exact A/B on the pinned RTL reported η of 1.1726 and 1.0253,
    # which is not a fraction and cannot be quoted as one.
    #
    # Two facts bound it, and the binding one is whichever is smaller. Every cycle in which two or more
    # engines are busy consumes at least TWO engine-busy-cycles, so overlap <= floor(total / 2). And in
    # every such cycle at least one engine other than the busiest is busy, so overlap <= total minus
    # the busiest engine's own total.
    #
    # For two engines this REDUCES to the second-largest exactly (checked: [100,60] -> 60, [80,80] ->
    # 80, [500,7] -> 7), so it generalises the existing convention rather than replacing it, and the
    # falsifier's denominator and this one stay the same quantity wherever they were already equal.
    ordered = sorted(busy.values(), reverse=True)
    total = sum(ordered)
    available = min(total - ordered[0], total // 2) if len(ordered) >= 2 else 0
    if available <= 0:
        return {"state": "unknown", "eta": None, "busy_cycles": busy, "realised_cycles": realised,
                "partition_proof": proof,
                "why": "the second-busiest engine has no busy cycles, so no overlap was AVAILABLE; "
                       "0/0 is undefined, not 0.0"}
    return {"state": "measured", "eta": realised / float(available),
            "busy_cycles": busy, "realised_cycles": realised, "available_cycles": available,
            "measurement_cycles": measurement_cycles,
            "partition_cycles": partition_cycles,
            "complete": counters.complete(),
            "partition_proof": proof,
            "note": ("realised counts every cycle with two or more engines busy; available is the "
                     "most overlap the per-engine totals admit -- min(total - busiest, total // 2) -- "
                     "which equals the second-largest total when there are two engines, the case the "
                     "falsifier and headroom were written for")}


# ---------------------------------------------------------------------------------------------------
# turning the derived counter set into something a host harness can run
# ---------------------------------------------------------------------------------------------------

#: The marker a bracketed run prints before each counter value. Chosen to be greppable out of a
#: console log that also carries the capsule's own output, and parsed by exact prefix rather than by
#: position, because a simulator console interleaves writers and a positional reader silently
#: mis-attributes when something else prints mid-run.
COUNTER_MARKER = "MERLIN_HWCOUNTER"
COUNTER_SCHEMA_MARKER = "MERLIN_COUNTER_SCHEMA"


def counter_bracket_for_names(names: list[str] | tuple[str, ...], codes: dict, *, slots: int,
                              padding_code: int | None = None) -> dict:
    """C for configuring and reading an explicit derived counter set around a kernel.

    The names are supplied by another structural selector (for example
    :func:`derive_occupancy_counters` or :func:`counters_with_unit`).  This function assigns physical
    slots and refuses a partial set; it does not know any target event name or code.
    """
    ordered = list(dict.fromkeys(str(name) for name in names))
    missing = [name for name in ordered if name not in (codes or {})]
    if missing:
        raise ValueError(
            f"no event code for {missing}; a counter read at an unconfigured slot returns whatever "
            f"that slot last held, which would be reported as this run's measurement")
    selected_codes = [codes[name] for name in ordered]
    if any(isinstance(code, bool) or not isinstance(code, int) or code < 0
           for code in selected_codes):
        raise ValueError("counter event codes must be non-negative integers derived from the header")
    if len(set(selected_codes)) != len(selected_codes):
        raise ValueError(
            "selected counter names share an event code; assigning both to slots would label one "
            "hardware event as two different measurements")
    if len(ordered) > int(slots):
        raise ValueError(
            f"{len(ordered)} counter(s) need slots but the target exposes {slots}; refusing to emit a "
            f"partial bracket, because omitted events would turn an exact measurement into an "
            f"unlabelled lower bound")
    if padding_code is not None and (
            isinstance(padding_code, bool) or not isinstance(padding_code, int) or padding_code < 0):
        raise ValueError("counter padding code must be a non-negative integer derived from the header")
    slot_of = {name: index for index, name in enumerate(ordered)}
    pro = ["  // merlin: counters configured from this target's own event codes.",
           "  counter_reset();"]
    for name in ordered:
        pro.append(f"  counter_configure({slot_of[name]}, {int(codes[name])});  // {name}")
    # Multi-pass measurements must execute the same number of pre-window counter commands.  Otherwise
    # selecting two events in one pass and seven in another perturbs instruction/cache state before
    # the timer starts, even though both brackets surround the same kernel.  A target may provide its
    # own header-derived disabled-event code to fill the remaining physical slots; absence means no
    # padding rather than an assumed zero code.
    if padding_code is not None:
        for slot in range(len(ordered), int(slots)):
            pro.append(f"  counter_configure({slot}, {padding_code});  // padding: disabled event")
    epi = ["  // merlin: read the counters back. Marker-prefixed so a reader attributes by NAME, not",
           "  // by position -- a simulator console interleaves writers.",
           "  counter_snapshot_take();"]
    for name in ordered:
        epi.append(f'  printf("{COUNTER_MARKER} {name} %u\\n", counter_read({slot_of[name]}));')
    return {"prologue": "\n".join(pro) + "\n", "epilogue": "\n".join(epi) + "\n",
            "slot_of": slot_of,
            "configured_slots": int(slots) if padding_code is not None else len(ordered)}


def counter_bracket_c(counters: OccupancyCounters, codes: dict, *, slots: int,
                      padding_code: int | None = None) -> dict:
    """C for configuring, then reading, this target's combination counters around a kernel.

    Returns ``{"prologue": str, "epilogue": str, "slot_of": {...}}`` — text a harness generator can
    place before and after the work, plus which slot each counter was assigned so the epilogue's
    output can be attributed. Emitting text rather than editing a harness is deliberate: nothing on
    the graded path changes until a caller chooses to place it.

    ``codes`` maps a counter NAME to the numeric event code the target's own header declares, and
    ``slots`` is how many counter slots the hardware exposes — on the target measured here the config
    register masks the index with three bits and the RTL's counter bundle carries an
    ``external_values`` array, so eight, and the seven combination counters fit exactly. Both are
    supplied rather than assumed: a target with fewer slots than counters must be told so it can
    FAIL rather than silently read whichever ones happened to fit.
    """
    names = [counters.by_combination[key]
             for key in sorted(counters.by_combination, key=lambda combo: sorted(combo))]
    try:
        return counter_bracket_for_names(
            names, codes, slots=slots, padding_code=padding_code)
    except ValueError as exc:
        if len(names) > int(slots):
            raise ValueError(str(exc) + "; a missing combination makes realised overlap a lower "
                             "bound that could be reported as the total") from exc
        raise


def parse_counter_output(console: str) -> dict:
    """``counter name -> value`` from a bracketed run's console output.

    Attributes by the NAME the line carries, so an interleaved or reordered console still reads
    correctly, and a line whose value is not an integer is skipped rather than coerced -- a truncated
    console is a missing reading, which η already refuses on, and never a zero.
    """
    out: dict = {}
    ambiguous: set[str] = set()
    for raw in (console or "").splitlines():
        line = raw.strip()
        at = line.find(COUNTER_MARKER)
        if at == -1:
            continue
        parts = line[at + len(COUNTER_MARKER):].split()
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[1]
        if value.isdigit():
            if name in out:
                ambiguous.add(name)
            else:
                out[name] = int(value)
    for name in ambiguous:
        out.pop(name, None)
    return out


def parse_counter_schema(console: str) -> str | None:
    """Return the unique header digest embedded in the measured ELF, or UNKNOWN."""
    values: list[str] = []
    for raw in (console or "").splitlines():
        parts = raw.strip().split()
        if len(parts) != 2 or parts[0] != COUNTER_SCHEMA_MARKER:
            continue
        digest = parts[1]
        if len(digest) == 64 and all(char in "0123456789abcdefABCDEF" for char in digest):
            values.append(digest.lower())
    return values[0] if values and len(set(values)) == 1 else None


def event_codes(text: str) -> dict:
    """``NAME -> code`` for every integer ``#define`` in a counter header, for :func:`counter_bracket_c`."""
    return _defines(text)


def observations_from_counters(values: Mapping[str, int], counters: "OccupancyCounters", *,
                               total_cycles: int | None = None, source: str | None = None,
                               kind_of: Mapping[str, str] | None = None) -> dict:
    """A :mod:`merlin.perf.observations` timing block from one bracketed run's counter readings.

    This is the hop that was missing. The bracket emitter, the console parser, the wire contract and
    every consumer of a per-unit activity vector all existed; nothing turned a console full of
    ``<PREFIX>_<engines>_CYCLES`` readings into the block the tier record carries, so a target that
    counts overlap in hardware still reported ``missing: ['at least one activity source']`` and no
    composition operator, headroom or η could resolve from it.

    Three rules, each of which changes the numbers:

    **A per-engine busy total is the single counter PLUS every combination containing it.** An engine
    is busy during the cycles it shares with another engine, so reading the singles as whole-engine
    totals understates the busiest engine — which is η's denominator, so it inflates η. The sum is
    exact rather than approximate because the increment conditions are mutually exclusive by
    construction: every cycle is charged to exactly one subset.

    **The emitted totals therefore do NOT partition the timeline**, and the block says so. That is
    what licenses the overlap reading: ``partitioned=True`` exists to reject an overlap inferred from
    buckets that charge each cycle once and so report zero overlap whether or not the hardware
    overlaps. Here the overlap is not inferred at all — the multi-engine combinations measure it
    directly, which is the independent observation :func:`~merlin.perf.headroom.composition_operator`
    requires.

    **A counter the bracket configured but whose reading did not come back is UNMEASURED, never
    zero.** Its engines lose their busy entry and are named in ``unmeasured_units``, because a total
    computed from a partial combination set is a lower bound that would be read as the total.
    ``alias_collisions`` is deliberately omitted rather than reported as 0: it is a property of the
    address span a program touched, and this instrument does not establish it.

    ``kind_of`` maps each engine to its resource KIND and unlocks the second overlap quantity. A unit's
    kind cannot be derived from a counter name, so it is declared or it is absent -- and the distinction
    is load-bearing rather than cosmetic. ``overlap_cycles.observed`` counts any two engines busy
    together; the eta a kind-axis consumer needs counts only cycles spanning two DIFFERENT kinds,
    because two movement engines running together is not movement/compute overlap. On a machine whose
    engines are not all distinct kinds the two numbers differ, and reporting the first where the second
    is meant overstates the overlap that a compute/movement pairing achieved.
    """
    from .observations import (BUSY_PREFIX, IDLE_QUANTITY, IN_PROGRAM_SUFFIX, OVERLAP_ACROSS_KINDS,
                              OVERLAP_OBSERVED, PARTITIONED_KEY, TIMING_OBSERVATIONS_KEY,
                              UNMEASURED_UNITS_KEY)

    by_combo = dict(counters.by_combination)
    missing = sorted(name for name in by_combo.values() if name not in values)
    # An engine is unreadable if ANY combination naming it is missing: its total would be a lower bound.
    unmeasured: set[str] = set()
    for combo, name in by_combo.items():
        if name in missing:
            unmeasured.update(combo)

    prov = source or "hardware combination counters"
    entries: list[dict] = []
    for engine in counters.engines:
        if engine in unmeasured:
            continue
        busy = sum(int(values[name]) for combo, name in by_combo.items() if engine in combo)
        entries.append({"quantity": f"{BUSY_PREFIX}{engine}{IN_PROGRAM_SUFFIX}", "value": busy,
                        "unit": "cycles", "source": prov})

    # Overlap: the cycles two or more engines were busy at once, read off the combination counters
    # themselves. Refused outright if any combination is missing -- a partial sum is a lower bound.
    if not missing:
        realised = sum(int(values[name]) for combo, name in by_combo.items() if len(combo) >= 2)
        entries.append({"quantity": OVERLAP_OBSERVED, "value": realised, "unit": "cycles",
                        "source": prov})
        if kind_of:
            missing_kinds = sorted(e for e in counters.engines if e not in kind_of)
            if not missing_kinds:        # a partial kind map cannot classify every combination
                across = sum(int(values[name]) for combo, name in by_combo.items()
                             if len({kind_of[e] for e in combo}) >= 2)
                entries.append({"quantity": OVERLAP_ACROSS_KINDS, "value": across,
                                "unit": "cycles", "source": prov})
        if total_cycles is not None:
            charged = sum(int(values[name]) for name in by_combo.values())
            idle = int(total_cycles) - charged
            if idle >= 0:            # a negative idle means the window and the counters disagree: drop it
                entries.append({"quantity": IDLE_QUANTITY, "value": idle, "unit": "cycles",
                                "source": prov})

    return {TIMING_OBSERVATIONS_KEY: entries,
            UNMEASURED_UNITS_KEY: sorted(unmeasured),
            PARTITIONED_KEY: False}
