"""How much of the target's local storage a case's emitted kernel uses, and how often it re-reads operands.

Measured on one run: the author hand-picked storage windows (hundreds of rows, "conservative ... the
target's stores are much larger") and shape cutoffs, and one large dense case read its operands about
8.5x the compulsory bytes while the scratchpad could hold an operand panel that reads them about 1.5x.
Nothing in the feedback related the kernel's footprint to the capacity. This block does, per case:

* ``scratchpad`` / ``accumulator`` -- rows (and bytes) the kernel's commands address, against the
  capacity derived from the target's RTL facts (:func:`merlin.targetgen.address_space.operand_store` /
  ``accumulator_store``). The rows come from the executed local-address activity of a dynamic census
  when the row carries one, else from the case's copied static instruction trace decoded through the
  generic fact-driven RoCC decoder (:mod:`merlin.targetgen.rocc.semantics`) and the target contract's
  local-address protocol (:func:`merlin.perf.executed_trace.local_high_water`). A static trace states
  only the addresses that are constants in the emitted code: when some addressed commands carry run-time
  addresses (loop-emitted code) the count is a lower bound and says so, and a store none of whose
  commands resolve is UNKNOWN, never zero;
* ``operand_reads`` -- device bytes read in the timed scope (the timing observation's own request
  counter) over the compulsory bytes of the declared operands, and for a dense contraction the ratio a single-buffered
  blocking could reach if an operand panel filled the derived capacity (:func:`reuse_bound`, a simple
  analytic bound from M, K, N, the tile edge and the two capacities; no double buffering, banks or
  alignment);
* ``reading`` -- one line, e.g. "uses 1.1% of scratchpad rows (180 of 16384); reads 8.50x the
  compulsory operand bytes; capacity allows ~1.50x".

Structural and advisory: no cycles, no status, no threshold. No ``re``, no target name, no assumed
field or capacity: every number is read from the facts, the contract or the emitted artifacts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

SCHEMA = "merlin.capacity_use.v1"
TRACE = "instruction_trace.json"
BUFFER = "command_buffer.json"
_MAX_BYTES = 32 << 20
_CAPACITY: dict[tuple, dict | None] = {}


def _load(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def capacity(target: str | None) -> dict | None:
    """``{"scratchpad": {...}, "accumulator": {...}, "edge"}`` from the target's RTL facts, or None."""
    if not isinstance(target, str) or not target:
        return None
    key = (target, os.environ.get("MERLIN_RTL_FACTS"))
    if key not in _CAPACITY:
        try:
            from merlin.targetgen import address_space as AS
            from merlin.targetgen.rtl.facts import load_facts

            # Selected facts only: advisory feedback never regenerates RTL facts.
            space = AS.derive_address_space(target, facts=load_facts(target, regenerate=False))
            out: dict = {"edge": space.array_cols or space.array_rows}
            for name, resolver in (("scratchpad", AS.operand_store), ("accumulator", AS.accumulator_store)):
                store = resolver(space).store
                out[name] = (
                    {
                        "store": store.name,
                        "rows": store.total_rows,
                        "row_bytes": store.row_bytes,
                        "row_elems": store.row_elems,
                        "bytes": store.nbytes,
                    }
                    if store is not None and store.total_rows
                    else None
                )
            _CAPACITY[key] = out
        except Exception:  # noqa: BLE001 -- an underivable capacity leaves the block absent
            _CAPACITY[key] = None
    return _CAPACITY[key]


def decode(trace: dict, target: str) -> dict:
    """``trace`` with every RoCC instruction (re)decoded by the generic fact-driven decoder."""
    from merlin.targetgen.rocc import semantics

    isa = semantics.isa_constants(target)
    out = []
    for instruction in trace.get("instructions") or ():
        if not isinstance(instruction, dict):
            continue
        funct = instruction.get("funct")
        if type(funct) is int and isinstance(instruction.get("rs1"), dict) and isinstance(instruction.get("rs2"), dict):
            klass, decoded = semantics.decode_instruction(funct, instruction["rs1"], instruction["rs2"], isa)
            instruction = {**instruction, "class": klass, "decoded": decoded}
        out.append(instruction)
    return {**trace, "instructions": out}


def _addressed(trace: dict, target: str) -> tuple[int, int]:
    """``(addressed commands, of which statically resolved)`` under the contract's local-address protocol."""
    from merlin.targetgen import rtl_checks_generic as G

    protocol = G.protocol_for(target)
    total = resolved = 0
    for instruction in trace.get("instructions") or ():
        decoded = instruction.get("decoded") or {}
        for entry in protocol.local_addresses:
            if instruction.get("class") in protocol.classes(str(entry["role"])):
                total += 1
                value = decoded.get(entry["address"])
                resolved += isinstance(value, int) and not isinstance(value, bool)
    return total, resolved


def _census_trace(row: dict) -> dict | None:
    """The executed local-address requests of a dynamic census, as decoded pseudo-instructions."""
    census = row.get("dynamic_census") if isinstance(row, dict) else None
    observations = (census.get("observations") if isinstance(census, dict) else None) or {}
    activity = observations.get("local_address_activity") if isinstance(observations, dict) else None
    if not isinstance(activity, dict) or activity.get("status") != "observed":
        return None
    out = []
    for request in activity.get("requests") or ():
        if isinstance(request, dict) and isinstance(request.get("address_field"), str):
            decoded = {request["address_field"]: request.get("raw_address")}
            if isinstance(request.get("rows_field"), str):
                decoded[request["rows_field"]] = request.get("raw_rows")
            out.append({"class": request.get("class"), "decoded": decoded})
    return {"instructions": out} if out else None


def _store_use(rows, cap: dict | None, *, lower_bound: bool) -> dict | None:
    if cap is None:
        return None
    out = {"rows": rows, "capacity_rows": cap["rows"], "capacity_bytes": cap["bytes"]}
    if isinstance(rows, int):
        out["bytes"] = rows * cap["row_bytes"] if cap.get("row_bytes") else None
        out["pct"] = round(100.0 * rows / cap["rows"], 1)
        if lower_bound:
            out["lower_bound"] = True
    return out


def footprint(row: dict, folder: Path | None, target: str, cap: dict) -> dict | None:
    """Rows each store's commands address, with the basis they were read from."""
    from merlin.perf.executed_trace import local_high_water

    executed = _census_trace(row)
    if executed is not None:
        water, basis, partial = local_high_water(executed, target=target), "executed (dynamic census)", False
        total = resolved = len(executed["instructions"])
    else:
        trace = _load(folder / TRACE) if folder is not None else None
        if not isinstance(trace, dict):
            return None
        trace = decode(trace, target)
        total, resolved = _addressed(trace, target)
        if not total:
            return None
        water = local_high_water(trace, target=target)
        partial = resolved < total
        basis = "static decoded trace" + (
            f" ({resolved} of {total} addressed commands have constant addresses; the rest are run-time values)"
            if partial
            else ""
        )
    if water.get("status") != "measured":
        return {"basis": basis, "scratchpad": None, "accumulator": None, "why": water.get("why")}
    out = {"basis": basis}
    for store in ("scratchpad", "accumulator"):
        rows = water.get(f"{store}_rows_touched")
        # A partial static view that resolved none of a store's addresses knows nothing about it.
        if partial and not rows:
            rows = None
        out[store] = _store_use(rows, cap.get(store), lower_bound=partial)
    return out


def _bytes(shape, dtype) -> int | None:
    from merlin.targetgen.address_space import element_bits

    bits = element_bits(dtype) if isinstance(dtype, str) else None
    if not bits or not isinstance(shape, (list, tuple)) or any(type(v) is not int or v <= 0 for v in shape):
        return None
    return math.prod(shape) * bits // 8


def compulsory_bytes(declaration: dict | None, buffer: dict | None) -> int | None:
    """Bytes of every declared operand read once (the declaration's inputs, else the buffer's tensors)."""
    tensors = []
    if isinstance(declaration, dict) and declaration.get("inputs"):
        tensors = [t for t in declaration["inputs"] if isinstance(t, dict)]
    elif isinstance(buffer, dict) and isinstance(buffer.get("tensors"), dict):
        tensors = [t for t in buffer["tensors"].values() if isinstance(t, dict) and t.get("role") != "output"]
    sizes = [_bytes(t.get("shape"), t.get("dtype")) for t in tensors]
    return sum(sizes) if sizes and all(s is not None for s in sizes) else None


def reuse_bound(m: int, k: int, n: int, *, operand_elems: int, accumulator_elems: int, edge: int):
    """Fewest operand element reads over ``M*K + K*N`` a single-buffered blocking reaches, or None.

    Three schedules, each feasible only within the derived capacities: keep a ``K x Nb`` right-operand
    panel resident beside one ``edge x K`` left strip (outputs drain ``edge x Nb`` at a time), the
    symmetric left-panel schedule, and an output-stationary ``Mb x Nb`` accumulator block fed by
    ``edge``-deep operand strips. Block sides are whole tiles.
    """
    if min(m, k, n, operand_elems, accumulator_elems, edge) <= 0:
        return None

    def up(value):
        return -(-value // edge) * edge

    big_m, big_k, big_n = up(m), up(k), up(n)
    compulsory = m * k + k * n
    best = None

    def keep(reads):
        nonlocal best
        best = reads if best is None or reads < best else best

    # Right panel resident: K x Nb in the operand store with one edge x K left strip.
    panel = (operand_elems - edge * big_k) // max(1, big_k) // edge * edge  # widest resident panel
    drain = accumulator_elems // edge // edge * edge  # widest edge-row output strip
    nb = min(big_n, panel, drain)
    if nb >= edge:
        keep(k * n + m * k * math.ceil(n / nb))
    mb = min(big_m, panel, drain)
    if mb >= edge:
        keep(m * k + k * n * math.ceil(m / mb))
    for mb in range(edge, big_m + 1, edge):
        nb = min(big_n, accumulator_elems // mb // edge * edge, (operand_elems // edge - mb) // edge * edge)
        if nb >= edge:
            keep(m * k * math.ceil(n / nb) + k * n * math.ceil(m / mb))
    return None if best is None else round(best / compulsory, 2)


def _bound(declaration, cap) -> float | None:
    from . import shape_classes

    description = shape_classes.describe(declaration) if isinstance(declaration, dict) else None
    if description is None or description["family"] != "dense" or not cap.get("edge"):
        return None
    scratch, acc = cap.get("scratchpad"), cap.get("accumulator")
    if not scratch or not acc or not scratch.get("row_elems") or not acc.get("row_elems"):
        return None
    extents = description["extents"]
    return reuse_bound(
        extents["m"],
        extents["k"],
        extents["n"],
        operand_elems=scratch["rows"] * scratch["row_elems"],
        accumulator_elems=acc["rows"] * acc["row_elems"],
        edge=cap["edge"],
    )


#: The timing observation's count of bytes the device requested from memory (its own counter name).
DEVICE_READ = "device.requested_read_bytes"


def _reads(row: dict) -> int | None:
    """Device bytes read in the timed scope: the observation's counter, else its traffic boundary."""
    performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
    value = (performance.get("feature_counts") or {}).get(DEVICE_READ)
    if type(value) is int and value > 0:
        return value
    for boundary in (performance.get("scoped_resource_feedback") or {}).get("traffic") or ():
        if isinstance(boundary, dict) and boundary.get("boundary") == "device_requested_read":
            value = boundary.get("bytes")
            return value if type(value) is int and value > 0 else None
    return None


def reading(block: dict) -> str | None:
    parts = []
    for store in ("scratchpad", "accumulator"):
        use = block.get(store)
        if use is None:
            continue
        if use.get("pct") is None:
            parts.append(f"{store} use UNKNOWN (run-time addresses)")
        else:
            prefix = "at least " if use.get("lower_bound") else ""
            parts.append(f"uses {prefix}{use['pct']}% of {store} rows ({use['rows']} of {use['capacity_rows']})")
    reads = block.get("operand_reads") or {}
    if reads.get("ratio") is not None:
        parts.append(f"reads {reads['ratio']:.2f}x the compulsory operand bytes")
    if reads.get("capacity_bound_ratio") is not None:
        parts.append(f"capacity allows ~{reads['capacity_bound_ratio']:.2f}x")
    return "; ".join(parts) or None


def for_case(name: str, row: dict, workspace, *, artifact_root=None, target=None) -> dict | None:
    """``capacity_use`` for one case (module docstring), or None when nothing is known; never raises."""
    try:
        from . import case_declarations
        from .fence_overlap import case_dir

        folder = case_dir(name, Path(workspace), artifact_root)
        buffer = _load(folder / BUFFER) if folder is not None else None
        if target is None:
            target = case_declarations.target() or (buffer.get("target") if isinstance(buffer, dict) else None)
        cap = capacity(target)
        if cap is None:
            return None
        declaration = case_declarations.load(name, workspace)
        block = {"schema": SCHEMA}
        used = footprint(row or {}, folder, target, cap)
        if used is not None:
            block.update(used)
        compulsory, read = compulsory_bytes(declaration, buffer), _reads(row or {})
        reads = {"device_read_bytes": read, "compulsory_bytes": compulsory}
        if read and compulsory:
            reads["ratio"] = round(read / compulsory, 2)
        bound = _bound(declaration, cap)
        if bound is not None:
            reads["capacity_bound_ratio"] = bound
        if read or bound is not None:
            block["operand_reads"] = reads
        if len(block) == 1:
            return None
        line = reading(block)
        if line:
            block["reading"] = line
        return block
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError):
        return None
    except Exception:  # noqa: BLE001 -- an unavailable protocol or decoder leaves the block absent
        return None


__all__ = ["SCHEMA", "capacity", "compulsory_bytes", "decode", "footprint", "for_case", "reading", "reuse_bound"]
