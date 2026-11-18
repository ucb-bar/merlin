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
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

# ---- hw_counters -------------------------------------------------------------------------------

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
        want = sum(1 for r in range(1, len(self.engines) + 1) for _ in combinations(self.engines, r))
        return len(self.by_combination) == want and bool(self.engines)

    def to_dict(self) -> dict:
        return {
            "prefix": self.prefix,
            "engines": list(self.engines),
            "by_combination": {
                "+".join(sorted(k)): v for k, v in sorted(self.by_combination.items(), key=lambda kv: sorted(kv[0]))
            },
            "complete": self.complete(),
        }


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
    operands = rhs[len(prefix) :]
    for marker in (" {", " :"):
        operands = operands.split(marker, 1)[0]
    if operands.startswith("bin "):
        operands = operands[4:]
    refs = tuple(part.strip() for part in operands.split(",") if part.strip())
    return refs or None


def prove_occupancy_partition_from_circt(
    hw_text: str,
    counters: OccupancyCounters,
    codes: Mapping[str, int],
    *,
    module: str,
    counter_module: str,
    source: str | None = None,
) -> dict:
    """Prove that counter events are the exhaustive one-hot partition of engine-busy states.

    Header names only suggest combination semantics.  This verifier follows the corresponding numeric
    event ports into elaborated CIRCT, symbolically evaluates their ``comb.and/or/xor`` cones, and
    requires every combination counter to be true on exactly its named busy-bit valuation.  Engine and
    signal names are inferred; callers provide only target-owned module identities and artifacts.
    """
    digest = hashlib.sha256((hw_text or "").encode("utf-8")).hexdigest()
    base = {
        "status": "unknown",
        "method": "circt_boolean_partition_v1",
        "source": source,
        "sha256": digest,
        "module": module,
        "counter_module": counter_module,
        "counter_fingerprint": _counter_fingerprint(counters, codes),
    }
    if not counters.complete():
        return {**base, "why": "the header does not expose every non-empty engine combination"}
    if any(name not in codes for name in counters.by_combination.values()):
        return {**base, "why": "one or more occupancy counters has no derived event code"}
    selected_codes = [codes[name] for name in counters.by_combination.values()]
    if any(isinstance(code, bool) or not isinstance(code, int) or code < 0 for code in selected_codes):
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
        refs = _operand_refs(rhs, "comb.and") or _operand_refs(rhs, "comb.or") or _operand_refs(rhs, "comb.xor")
        if refs is None:
            visiting.remove(ref)
            return False
        ok = all(discover(operand) for operand in refs)
        visiting.remove(ref)
        return ok

    if not all(discover(ref) for ref in event_refs.values()):
        return {**base, "why": "an occupancy event cone uses unsupported or cyclic CIRCT logic"}
    if len(leaves) != len(counters.engines):
        return {
            **base,
            "why": (
                f"event cones have {len(leaves)} independent boolean leaves for "
                f"{len(counters.engines)} header-derived engines"
            ),
        }
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
            return {**base, "why": (f"counter for {sorted(combo)} is not exactly its named busy-bit valuation")}
    return {
        **base,
        "status": "proved",
        "engine_leaves": engine_to_leaf,
        "events": {
            "+".join(sorted(combo)): event_refs[combo] for combo in sorted(event_refs, key=lambda item: sorted(item))
        },
    }


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
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor)
    ):
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
    return {name: code for name, code in _defines(text).items() if wanted in name.upper().split("_")}


def derive_occupancy_counters(text: str) -> OccupancyCounters:
    """Factor a counter header's combination counters into engines and combinations.

    The derivation: among ``<PREFIX>_<tokens...>_CYCLES`` names sharing one prefix, the union of the
    single-token names' tokens is the engine set, and every other name over those tokens is a
    combination. Requiring the singles to exist is deliberate — without them there is no per-engine
    busy figure, so η has no denominator and the reading is not an occupancy vector at all.
    """
    return derive_occupancy_from_codes(_defines(text))


def derive_occupancy_from_codes(codes: Mapping[str, int]) -> OccupancyCounters:
    """:func:`derive_occupancy_counters` over an explicit ``name -> code`` table (a header's defines, or
    a counter set the target contract declares) -- the same structural factoring, one implementation."""
    names = [n for n in codes if n.endswith("_" + _CYCLES)]
    # Group by first token, which is the family prefix the header uses for one counter block.
    groups: dict = {}
    for n in names:
        toks = n.split("_")
        if len(toks) < 3:
            continue  # <PREFIX>_CYCLES carries no tokens
        groups.setdefault(toks[0], []).append((n, tuple(toks[1:-1])))
    best = OccupancyCounters()
    for prefix, entries in sorted(groups.items()):
        singles = {t[0] for _n, t in entries if len(t) == 1}
        if len(singles) < 2:
            continue  # one engine cannot overlap with itself
        combos: dict = {}
        for name, toks in entries:
            if toks and set(toks) <= singles and len(set(toks)) == len(toks):
                combos[frozenset(toks)] = name
        got = OccupancyCounters(prefix=prefix, engines=tuple(sorted(singles)), by_combination=combos)
        # Prefer the block that resolves the most combinations; a header may carry several.
        if len(got.by_combination) > len(best.by_combination):
            best = got
    return best


def counter_set_digest(events: Mapping[str, Any]) -> str:
    """Content digest of a declared counter-event table; it plays the role a header's SHA-256 plays."""
    import json

    return hashlib.sha256(json.dumps(events, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def declared_counter_set(target: str, *, contract: Mapping[str, Any] | None = None) -> dict:
    """The counter-event table ``target``'s contract declares (``logical_harness.counter_bracket.events``).

    ``{"internal": {NAME: code}, "external": {NAME: code}}``: two independent event spaces, an external
    event being selected with the command's external flag. The codes are reviewed data citing the RTL
    enumeration they come from; nothing here knows any name or code.
    """
    if contract is None:
        try:
            from merlin.target.registry.identity import resolve as resolve_target

            contract = resolve_target(target).load_contract()
        except Exception as e:  # noqa: BLE001
            return {"status": "unavailable", "why": f"the target contract is unreadable ({type(e).__name__})"}
    block = ((contract or {}).get("logical_harness") or {}).get("counter_bracket")
    events = block.get("events") if isinstance(block, Mapping) else None
    if not isinstance(events, Mapping):
        return {"status": "unavailable", "why": "the target contract declares no counter-event table"}
    flat: dict[str, int] = {}
    external: list[str] = []
    for space in ("internal", "external"):
        table = events.get(space) or {}
        if not isinstance(table, Mapping):
            return {"status": "unavailable", "why": f"counter events.{space} is not a mapping"}
        for name, code in table.items():
            if isinstance(code, bool) or not isinstance(code, int) or code < 0 or str(name) in flat:
                return {"status": "unavailable", "why": f"counter event {name!r} is malformed or duplicated"}
            flat[str(name)] = code
            if space == "external":
                external.append(str(name))
    occupancy = derive_occupancy_from_codes({n: c for n, c in flat.items() if n not in external})
    if not occupancy.by_combination:
        return {"status": "absent", "why": "the declared counter events form no combination block"}
    return {
        "status": "derived",
        "source": "contract",
        "header": None,
        "header_sha256": counter_set_digest(events),
        "codes": flat,
        "external": sorted(external),
        "event_codes": {name: flat[name] for name in occupancy.by_combination.values()},
        "counters": occupancy.to_dict(),
    }


def _counter_block(target: str) -> Mapping[str, Any]:
    try:
        from merlin.target.registry.identity import resolve as resolve_target

        contract = resolve_target(target).load_contract()
    except Exception:  # noqa: BLE001 -- an unreadable contract declares nothing
        return {}
    block = ((contract or {}).get("logical_harness") or {}).get("counter_bracket")
    return block if isinstance(block, Mapping) else {}


def declared_engine_kinds(target: str) -> dict[str, str] | None:
    """Which resource kind each counted engine is, as the target contract declares it (``engine_kinds``);
    None when undeclared -- a kind is never inferred from a counter name."""
    kinds = _counter_block(target).get("engine_kinds")
    if not isinstance(kinds, Mapping) or not kinds:
        return None
    return {str(k): str(v) for k, v in kinds.items()}


def declared_partition_inputs(target: str) -> dict:
    """The CIRCT inputs of the occupancy-partition proof, from the contract's ``partition`` module
    identities and the target's elaborated core HW. The proof itself still runs on the artifact."""
    partition = _counter_block(target).get("partition")
    if not isinstance(partition, Mapping) or not partition.get("module") or not partition.get("counter_module"):
        return {"status": "unknown", "why": "the target declares no counter-partition module identities"}
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

    path = mlc_bridge.core_hw_mlir(target)
    if path is None or not Path(path).is_file():
        return {"status": "unknown", "why": "elaborated CIRCT core HW is unavailable"}
    return {
        "status": "available",
        "hw_text": Path(path).read_text(encoding="utf-8", errors="replace"),
        "module": str(partition["module"]),
        "counter_module": str(partition["counter_module"]),
        "source": str(path),
    }


def discovered_codes(discovery: Mapping[str, Any]) -> dict[str, int]:
    """Every ``name -> code`` of a derived discovery: the header's defines (re-read and checked against
    the discovery digest) or the contract's declared table."""
    if discovery.get("source") == "contract":
        return dict(discovery.get("codes") or {})
    text = Path(str(discovery["header"])).read_text(encoding="utf-8", errors="replace")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != discovery.get("header_sha256"):
        raise ValueError("the counter header changed after discovery")
    return _defines(text)


def occupancy_for_discovery(discovery: Mapping[str, Any]) -> tuple[OccupancyCounters, dict[str, int]]:
    """``(occupancy counters, every event code)`` of a derived discovery, header or contract alike."""
    codes = discovered_codes(discovery)
    external = set(discovery.get("external") or ())
    return derive_occupancy_from_codes({n: c for n, c in codes.items() if n not in external}), codes


def counter_source_for_target(target: str) -> dict:
    """The target's counter-event source: its shipped counter header when it declares one
    (:func:`counters_for_target`), else the contract's reviewed counter-event table
    (:func:`declared_counter_set`, marked ``source: contract``). Header consumers keep calling
    :func:`counters_for_target`, which never answers from the contract."""
    found = counters_for_target(target)
    if found.get("status") == "unavailable":
        declared = declared_counter_set(target)
        if declared.get("status") != "unavailable":
            return declared
    return found


def counters_for_target(target: str, *, sources=None) -> dict:
    """Derive the combination counters from ``target``'s own shipped counter header.

    Three states. ``derived`` when a block resolved; ``absent`` when the headers were read and expose
    no combination counters (a real fact about that target); ``unavailable`` when no header could be
    read at all — which is NOT the same as the target having none.
    """
    paths = list(sources or ())
    if not paths:
        try:
            from merlin.target.capabilities.discovery import sources as CD

            for s in CD.isa_sources(target) or ():
                p = Path(str(getattr(s, "path", s)))
                if p.is_file():
                    paths.append(p)
                    # Sibling headers of the same shipped set, e.g. a dedicated counter header.
                    paths.extend(sorted(q for q in p.parent.glob("*.h") if q.is_file()))
        except Exception as e:  # noqa: BLE001
            return {"status": "unavailable", "why": f"{type(e).__name__}: {str(e)[:120]}"}
    seen, uniq = set(), []
    for p in paths:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(Path(p))
    if not uniq:
        return {
            "status": "unavailable",
            "why": "no shipped header could be located for this target; whether it exposes "
            "combination counters is UNKNOWN, not absent",
        }
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
            return {
                "status": "unavailable",
                "unreadable": unread,
                "why": "no candidate header could be READ, so whether this target exposes "
                "combination counters is UNKNOWN, not absent",
            }
        return {
            "status": "absent",
            "read": read,
            "unreadable": unread,
            "why": "the shipped headers expose no counter block with per-engine singles and a "
            "combination over them, so this target does not count overlap in hardware",
        }
    return {
        "status": "derived",
        "header": str(where),
        "header_sha256": where_digest,
        "event_codes": where_codes,
        "counters": best.to_dict(),
    }


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
    provenance = {
        "source": source,
        "sha256": digest,
        "module": module,
        "state_families": list(state_families),
        "method": "dense_seq_firreg_families",
    }
    if not module or not state_families:
        return {
            "status": "unknown",
            "slots": None,
            "provenance": provenance,
            "why": "a CIRCT module and at least one target-owned state family are required",
        }

    body, error = _module_lines(hw_text, module)
    if body is None:
        return {"status": "unknown", "slots": None, "provenance": provenance, "why": error}

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
            suffix = lhs[len(prefix) :]
            if suffix.isdigit():
                indices[family].add(int(suffix))

    missing = [family for family, got in indices.items() if not got]
    if missing:
        return {
            "status": "unknown",
            "slots": None,
            "provenance": provenance,
            "why": f"no indexed seq.firreg state was found for family/families {missing}",
        }
    sets = list(indices.values())
    if any(got != sets[0] for got in sets[1:]):
        return {
            "status": "unknown",
            "slots": None,
            "provenance": provenance,
            "why": "the independently elaborated counter state families disagree on their indices",
            "indices": {name: sorted(got) for name, got in indices.items()},
        }
    expected = set(range(len(sets[0])))
    if sets[0] != expected:
        return {
            "status": "unknown",
            "slots": None,
            "provenance": provenance,
            "why": "counter state indices are not a dense zero-based slot set",
            "indices": {name: sorted(got) for name, got in indices.items()},
        }
    return {
        "status": "derived",
        "slots": len(expected),
        "provenance": provenance,
        "evidence": {"indices": sorted(expected), "families": {name: len(got) for name, got in indices.items()}},
    }


PROVED_FROM_ARTIFACT = "proved_from_artifact"


DECLARED_BY_PRODUCER = "declared_by_producer"
