"""Per-group DERIVED ROOFLINE: the fewest cycles any schedule of a group can take on this machine.

    machine = roofline_machine(target)                    # array geometry + memory path, both derived
    shapes = group_shapes(model_capsule, target=target)   # each group's extents and compulsory bytes
    bound = group_roofline(shapes[group], machine)         # {"roofline_cycles", "min_computes", ...}
    bound = confront(bound, [("ours", 412_000)])           # a measurement below it REFUTES it

WHY. A bar made from another implementation's cycles (the vendor library's) is a fact about that
implementation: where it is weak, the bar is weak, and a form whose bar is already beaten looks done
while most of its headroom is left. This bound is about the MACHINE, from the target's own facts and
the group's own shape, so it is the same for every implementation and shows each form's real headroom.

TWO FLOORS, AND THE ROOFLINE IS THE LARGER.

* **Compute.** A contraction ``M x K x N`` on an ``R x C`` array is cut into stationary blocks of at
  most ``R`` deep by ``C`` wide; every row of the streamed operand passes every block once, one row
  per cycle (the sequencer's own loop, :mod:`merlin.perf.mesh_occupancy`), and a block itself enters
  the array one row per cycle, which at best overlaps the previous block's stream. So a block costs at
  least ``max(streamed rows, block depth)`` cycles, and the floor is the sum over blocks, minimised
  over which operand is held (``C = A B`` or ``C^T = B^T A^T``). ``min_computes`` is the same cut
  counted in full ``R x R x C`` compute units. A convolution is its own im2col contraction
  (``M = Hout * Wout``, ``K = ci * kh * kw``, ``N = co``); nothing is charged for forming that matrix.
* **Movement.** Every byte a group reads (its operand tensors, as the buffer declares them) must come in
  over the accelerator's memory path, and every byte it writes must go out; read and write are
  separate channels, so the floor is ``max(read bytes / read width, write bytes / write width)``. The
  widths come from the elaborated circuit itself (:func:`merlin.targetgen.rtl.introspect.memory_path`,
  checked against the emulator's receipt), never from a configuration name.

WHAT IS LEFT OUT LOOSENS THE BOUND AND IS RECORDED. The fill/drain delay line, any bias vector (its
device width is a lowering choice), and a window-mean's all-ones operand are not charged. Each
omission makes the floor LOWER -- still a floor. A term whose fact is UNKNOWN is dropped the same way
and named in ``unresolved``; a group with neither term has no roofline, never a zero one.

IT IS REFUTABLE. Any measured cycle count below the roofline means an input it was derived from does
not describe the machine; :func:`confront` then marks it ``refuted`` and it is not a bar. Nothing here
names a target, an instruction or a size.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "compute_floor",
    "confront",
    "form_table",
    "group_roofline",
    "group_shapes",
    "measured_groups",
    "report",
    "roofline_machine",
]

SCHEMA = "merlin_group_roofline_v1"
REPORT_SCHEMA = "merlin_roofline_report_v1"
#: The perf-coverage convention for a window mean's constant operand (see forms.statement_forms).
_CONSTANT_ONES_PREFIX = "ONES_"
_READ_ROLES = ("lhs", "rhs")
_WRITE_ROLES = ("dst",)


def _ceil(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------- the machine


def engine_memory_path(target: str, *, emulator: str | Path | None = None) -> dict[str, Any]:
    """The memory path of the elaboration the dump-capable emulator was built from (its receipt names the
    FIRRTL and its digest; a file that no longer hashes to it is refused, never read as if it did)."""
    from merlin.perf import whole_model_gsim as G
    from merlin.targetgen.rtl import introspect

    try:
        engine = G._engine(target, emulator)
        receipt = json.loads(Path(engine["receipt"]).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 -- no engine: the movement term is UNKNOWN, with why
        return {"status": "unknown", "reason": f"no receipted emulator to read the elaboration of: {error}"}
    firrtl = (receipt.get("artifacts") or {}).get("firrtl") or {}
    path, declared = Path(str(firrtl.get("path") or "")), firrtl.get("sha256") or receipt.get("firrtl_sha256")
    if not path.is_file() or not declared:
        return {"status": "unknown", "reason": f"the emulator's receipt names no readable elaboration ({path})"}
    actual = _sha256(path)
    if actual != declared:
        return {
            "status": "unknown",
            "reason": f"{path} hashes to {actual[:12]}, the receipt names {str(declared)[:12]}: "
            "not the same elaboration",
        }
    found = introspect.memory_path(path)
    found["firrtl"] = {"path": str(path), "sha256": actual}
    found["emulator_sha256"] = engine["sha256"]
    return found


def roofline_machine(target: str, *, emulator: str | Path | None = None) -> dict[str, Any]:
    """``{"array_rows", "array_cols", "memory", "provenance", "unresolved"}`` -- every value derived."""
    from merlin.perf.decompose import is_unknown
    from merlin.perf.derived_bound import machine_from_facts

    facts = machine_from_facts(target, measure_fill=False)
    unresolved: dict[str, str] = {}
    rows, cols = facts.array_rows, facts.array_cols
    if is_unknown(rows) or is_unknown(cols):
        unresolved["compute"] = facts.refusals.get("array_rows") or "the array geometry is UNKNOWN"
        rows = cols = None
    memory = engine_memory_path(target, emulator=emulator)
    if memory.get("status") != "derived":
        unresolved["movement"] = str(memory.get("reason"))
    return {
        "target": target,
        "array_rows": None if rows is None else int(rows),
        "array_cols": None if cols is None else int(cols),
        "memory": memory,
        "provenance": {
            "array": facts.provenance.get("array_rows", ""),
            "memory": memory.get("evidence", ""),
            "firrtl_sha256": (memory.get("firrtl") or {}).get("sha256"),
        },
        "unresolved": unresolved,
    }


# -------------------------------------------------------------------------------------- the shapes


def _tensor_bytes(tensor: Mapping[str, Any] | None) -> int | None:
    from merlin.perf.derived_bound import _width_bits

    if not isinstance(tensor, Mapping):
        return None
    bits = _width_bits(tensor.get("dtype"))
    shape = tensor.get("shape")
    if bits is None or not isinstance(shape, Sequence) or not shape:
        return None
    count = 1
    for extent in shape:
        if not isinstance(extent, int) or extent < 0:
            return None
        count *= extent
    return count * _ceil(bits, 8)


def _extents(op: str, entry: Mapping[str, Any]) -> dict[str, int] | None:
    """The contraction ``{"M", "K", "N"}`` a group's entry declares, or its output matrix for an
    elementwise group (``{"M", "N"}``); None when a needed extent is missing."""

    def ints(*names: str) -> list[int] | None:
        values = [entry.get(n) for n in names]
        return [int(v) for v in values] if all(isinstance(v, int) and v > 0 for v in values) else None

    if op == "matmul":
        got = ints("M", "K", "N")
        return None if got is None else {"M": got[0], "K": got[1], "N": got[2]}
    if op == "conv2d":
        got = ints("N", "ci", "Himg", "Wimg", "kh", "kw")
        stride, padding = entry.get("stride") or [1, 1], entry.get("padding") or [0, 0, 0, 0]
        if got is None or len(stride) != 2 or len(padding) != 4:
            return None
        co, ci, himg, wimg, kh, kw = got
        hout = (himg + int(padding[0]) + int(padding[2]) - kh) // int(stride[0]) + 1
        wout = (wimg + int(padding[1]) + int(padding[3]) - kw) // int(stride[1]) + 1
        if hout <= 0 or wout <= 0:
            return None
        return {"M": hout * wout, "K": ci * kh * kw, "N": co}
    got = ints("M", "N")
    return None if got is None else {"M": got[0], "N": got[1]}


def group_shapes(model_capsule: str | Path, *, target: str) -> dict[str, dict[str, Any]]:
    """``{group: {"op", "form_text", "extents", "reads", "writes", "omitted"}}`` from the capture's own
    buffer: each read/write is a tensor the group names, at the size and dtype the buffer declares."""
    from merlin.perf import whole_model_build as W

    from . import forms as PC

    buffer = W.state(W.load_model_capsule(model_capsule), target=target)
    tensors = buffer.get("tensors") or {}
    out: dict[str, dict[str, Any]] = {}
    for row in (buffer.get("whole_program") or {}).get("per_group") or ():
        entry = row.get("entry")
        if not isinstance(entry, Mapping):
            continue
        operands = dict(row.get("operands") or {})
        window_mean = str(operands.get("lhs") or "").startswith(_CONSTANT_ONES_PREFIX)
        _key, text = PC.form_of_entry(entry, window_mean=window_mean)
        reads, writes, omitted = {}, {}, []
        for role, name in operands.items():
            name = str(name)
            if role in _READ_ROLES and not name.startswith(_CONSTANT_ONES_PREFIX):
                reads[name] = _tensor_bytes(tensors.get(name))
            elif role in _WRITE_ROLES:
                writes[name] = _tensor_bytes(tensors.get(name))
            else:
                omitted.append(f"{role}:{name}")
        out[str(row["group"])] = {
            "op": str(entry.get("op")),
            "form_text": text,
            "extents": _extents(str(entry.get("op")), entry),
            "reads": reads,
            "writes": writes,
            "omitted": omitted,
        }
    return out


# ------------------------------------------------------------------------------------- the floors


def compute_floor(m: int, k: int, n: int, *, array_rows: int, array_cols: int) -> dict[str, Any]:
    """The array's floor for ``M x K x N`` (see the module docstring), minimised over orientation."""

    def oriented(streamed: int, held: int) -> tuple[int, int]:
        depths = [min(array_rows, k - start) for start in range(0, k, array_rows)]
        blocks_wide = _ceil(held, array_cols)
        cycles = blocks_wide * sum(max(streamed, depth) for depth in depths)
        computes = _ceil(streamed, array_rows) * len(depths) * blocks_wide
        return cycles, computes

    options = {"stream_m": oriented(m, n), "stream_n": oriented(n, m)}
    best = min(options, key=lambda o: options[o][0])
    return {
        "cycles": options[best][0],
        "min_computes": min(c for _, c in options.values()),
        "orientation": best,
        "options": {o: {"cycles": c, "computes": u} for o, (c, u) in options.items()},
    }


def group_roofline(shape: Mapping[str, Any], machine: Mapping[str, Any]) -> dict[str, Any]:
    """One group's roofline document. ``roofline_cycles`` is None only when no term resolved."""
    unresolved = dict(machine.get("unresolved") or {})
    extents = shape.get("extents") or {}
    rows, cols = machine.get("array_rows"), machine.get("array_cols")
    doc: dict[str, Any] = {"schema": SCHEMA, "op": shape.get("op"), "extents": dict(extents) or None}
    terms: dict[str, float] = {}
    if rows and cols and {"M", "K", "N"} <= set(extents):
        floor = compute_floor(extents["M"], extents["K"], extents["N"], array_rows=rows, array_cols=cols)
        terms["compute"] = float(floor["cycles"])
        doc.update(
            min_computes=floor["min_computes"],
            compute_floor_cycles=floor["cycles"],
            orientation=floor["orientation"],
            cycles_per_compute_floor=round(floor["cycles"] / floor["min_computes"], 3),
        )
    elif {"M", "K", "N"} <= set(extents):
        doc["compute_floor_cycles"] = None
    else:
        doc.update(min_computes=0, compute_floor_cycles=0)  # no contraction: the array does no work here
    if rows and cols and {"M", "N"} <= set(extents):
        doc["output_tiles"] = _ceil(extents["M"], rows) * _ceil(extents["N"], cols)
    reads, writes = shape.get("reads") or {}, shape.get("writes") or {}
    unknown_bytes = [n for n, b in {**reads, **writes}.items() if not isinstance(b, int)]
    memory = machine.get("memory") or {}
    if unknown_bytes:
        unresolved["movement"] = f"tensor size not declared for {unknown_bytes}"
    elif memory.get("status") == "derived":
        read_bytes, write_bytes = sum(reads.values()), sum(writes.values())
        movement = max(read_bytes / memory["read_bytes_per_cycle"], write_bytes / memory["write_bytes_per_cycle"])
        terms["movement"] = movement
        doc.update(read_bytes=read_bytes, write_bytes=write_bytes, movement_floor_cycles=round(movement, 1))
    doc["omitted"] = list(shape.get("omitted") or ()) + ["fill/drain delay line"]
    doc["unresolved"] = unresolved
    if terms:
        limiter = max(terms, key=lambda t: terms[t])
        doc.update(roofline_cycles=int(-(-terms[limiter] // 1)), limiter=limiter, status="derived")
    else:
        doc.update(roofline_cycles=None, limiter=None, status="unknown")
    return doc


def confront(doc: Mapping[str, Any], samples: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """``doc`` with every measured ``(label, cycles)`` held against it. Any count BELOW the roofline
    refutes it (``status: refuted``, the violators named): a bound a real schedule beats is false."""
    out = dict(doc)
    bound = doc.get("roofline_cycles")
    counted = [(str(label), int(c)) for label, c in samples if isinstance(c, int) and not isinstance(c, bool)]
    out["confronted_with"] = len(counted)
    if isinstance(bound, int):
        below = [{"label": label, "cycles": c} for label, c in counted if c < bound]
        if below:
            out.update(status="refuted", refuted_by=below)
    return out


# ------------------------------------------------------------------------------------- the report


def form_table(
    shapes: Mapping[str, Mapping[str, Any]],
    rooflines: Mapping[str, Mapping[str, Any]],
    measured: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One row per FORM: its groups, the summed roofline, and each measured arm's summed cycles with its
    ratio to the roofline (``measured`` is ``{arm: {group: cycles}}``). A form with any group lacking a
    derived roofline, or any arm lacking a group's cycles, states which rather than summing a subset."""
    by_form: dict[str, list[str]] = {}
    for group, shape in shapes.items():
        by_form.setdefault(str(shape.get("form_text")), []).append(str(group))
    rows = []
    for text, groups in by_form.items():
        groups = sorted(groups, key=int)
        bounds = [rooflines.get(g, {}).get("roofline_cycles") for g in groups]
        complete = all(isinstance(b, int) for b in bounds) and all(
            rooflines.get(g, {}).get("status") == "derived" for g in groups
        )
        row: dict[str, Any] = {
            "form_text": text,
            "op": shapes[groups[0]].get("op"),
            "groups": [int(g) for g in groups],
            "roofline_cycles": sum(bounds) if complete else None,
            "limiters": sorted({str(rooflines.get(g, {}).get("limiter")) for g in groups}),
            "min_computes": sum(int(rooflines.get(g, {}).get("min_computes") or 0) for g in groups),
        }
        if not complete:
            row["roofline_missing"] = [g for g in groups if rooflines.get(g, {}).get("status") != "derived"]
        for arm, cycles in measured.items():
            got = [cycles.get(g) if cycles.get(g) is not None else cycles.get(int(g)) for g in groups]
            if all(isinstance(c, int) for c in got):
                row[arm] = sum(got)
                if row["roofline_cycles"]:
                    row[f"{arm}_over_roofline"] = round(row[arm] / row["roofline_cycles"], 3)
            else:
                row[arm] = None
        rows.append(row)
    return sorted(rows, key=lambda r: -(r.get("roofline_cycles") or 0))


def measured_groups(result: Mapping[str, Any]) -> dict[str, int]:
    """``{group: cycles}`` of a whole-model measurement result (its verdict's per-group rows)."""
    rows = ((result.get("verdict") or {}).get("groups")) or ()
    return {str(row["group"]): int(row["cycles"]) for row in rows if isinstance(row.get("cycles"), int)}


def report(
    target: str,
    model_capsule: str | Path,
    results: Mapping[str, str | Path],
    *,
    emulator: str | Path | None = None,
) -> dict[str, Any]:
    """Every group's derived roofline confronted with each labelled measurement ``results`` (a
    whole-model ``result.json`` per arm), and the per-form table.  The measured cycles are each
    result's own -- the device that produced them named beside them -- compared, never certified."""
    from merlin.common import provenance as PROV

    measured, devices = {}, {}
    for label, path in results.items():
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        measured[str(label)] = measured_groups(document)
        devices[str(label)] = {
            "result": str(path),
            "objective_cycles": document.get("objective_cycles"),
            "timing_status": document.get("timing_status"),
            "artifact": (document.get("device") or {}).get("artifact"),
            "package_sha256": document.get("package_sha256"),
        }
    machine = roofline_machine(target, emulator=emulator)
    shapes = group_shapes(model_capsule, target=target)
    rooflines = {
        group: confront(group_roofline(shape, machine), [(arm, cycles.get(group)) for arm, cycles in measured.items()])
        for group, shape in shapes.items()
    }
    pins = {}
    for name, pin in PROV.load_pins().items():
        if target in (pin.targets or ()):
            try:
                pins[name] = PROV.verify(name)
            except Exception:  # noqa: BLE001 -- an unverifiable pin is recorded by name, never omitted
                continue
    firrtl = (machine.get("memory") or {}).get("firrtl") or {}
    return {
        "schema": REPORT_SCHEMA,
        "target": target,
        "model_capsule": str(model_capsule),
        "claim": "derived lower bounds per group; the measured cycles are each result's own, compared, not certified",
        "machine": machine,
        "measured_on": devices,
        "refuted_groups": sorted(g for g, doc in rooflines.items() if doc.get("status") == "refuted"),
        "rooflines": rooflines,
        "table": form_table(shapes, rooflines, measured),
        "provenance": PROV.record(pins=pins, artifacts={"firrtl": firrtl["path"]} if firrtl.get("path") else {}),
    }
