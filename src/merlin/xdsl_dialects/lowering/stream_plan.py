"""The stream-level plan of a grouped model: lifetimes, staging, fences and overlap.

A dispatch program over compute groups says what runs. This says what it costs BETWEEN the things
that run, and what a scheduler could do about it, before anything is emitted:

* **lifetimes** — every buffer is ``external`` (bound by the caller), ``constant`` (a weight, or
  computed only from constants, so computable once and offline), ``transient`` (produced and
  consumed on one side of the host/device boundary) or ``staging`` (crosses it);
* **prepack** — the work whose inputs are all constant. It is run per inference today and need
  not be run at all at inference time;
* **fences** — where the program has to wait for the device: a true dependence of a host node (or
  the program's result) on a device node's output. Everything else is a fence the emission adds
  because nothing told it otherwise;
* **overlap** — host work that depends on no in-flight device group, so could run while it does.

Pure analysis over :class:`~.dispatch_program.DispatchProgram` and the outliner's dispatch table.
See ``docs/design/macro_scheduling.md`` for what each number is for.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any

from .arena_plan import ArenaPlanError, _buf_bytes
from .dispatch_program import DispatchProgram

SCHEMA = "stream_plan_v1"
HOST = "host"

EXTERNAL, CONSTANT, TRANSIENT, STAGING = "external", "constant", "transient", "staging"


def _bytes(buffer) -> int | None:
    try:
        return _buf_bytes(list(buffer.shape), str(buffer.dtype))
    except ArenaPlanError:
        return None


def _elements(buffer) -> int:
    total = 1
    for extent in buffer.shape:
        total *= max(int(extent), 0)
    return total if buffer.shape else 1


def plan(program: DispatchProgram, dispatches: Iterable[Any], *, weight_args: Collection[int] = ()) -> dict[str, Any]:
    """The stream plan of ``program``.

    ``dispatches`` is the outliner's table (``symbol``, ``placement``). ``weight_args`` are the
    model-argument indices that are weights (from the capture's weights manifest); with none, every
    argument is external and nothing is provably constant, which understates prepack and never
    overstates it.
    """
    placement = {str(d.symbol): (str(d.placement) if getattr(d, "placement", None) else HOST) for d in dispatches}
    weights = set(int(i) for i in weight_args)

    def side(node) -> str:
        if node.kind != "dispatch":
            return HOST
        return "device" if placement.get(node.op, HOST) != HOST else HOST

    producer: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(program.nodes):
        for out in node.outputs:
            producer[out] = index
        for used in node.inputs:
            consumers.setdefault(used, []).append(index)

    # --- constants: weights, literals, and anything computed only from them ----------------------
    constant: set[str] = {
        bid
        for bid, buffer in program.buffers.items()
        if buffer.kind == "const" or (buffer.kind == "arg" and buffer.arg_index in weights)
    }
    prepack_nodes: list[int] = []
    for index, node in enumerate(program.nodes):
        if node.inputs and all(used in constant for used in node.inputs):
            constant.update(node.outputs)
            if node.kind == "dispatch":
                prepack_nodes.append(index)
        elif not node.inputs and node.kind == "view":
            constant.update(node.outputs)  # a literal or an empty: no runtime input

    # --- lifetimes ------------------------------------------------------------------------------
    kinds: dict[str, str] = {}
    for bid, buffer in program.buffers.items():
        if bid in constant:
            kinds[bid] = CONSTANT
        elif buffer.kind == "arg" or bid in program.results:
            kinds[bid] = EXTERNAL
        else:
            made_on = side(program.nodes[producer[bid]]) if bid in producer else HOST
            read_on = {side(program.nodes[i]) for i in consumers.get(bid, ())}
            kinds[bid] = STAGING if read_on - {made_on} else TRANSIENT
    by_kind: dict[str, dict[str, int]] = {}
    unsized = 0
    for bid, kind in kinds.items():
        size = _bytes(program.buffers[bid])
        row = by_kind.setdefault(kind, {"buffers": 0, "bytes": 0})
        row["buffers"] += 1
        if size is None:
            unsized += 1
        else:
            row["bytes"] += size

    # --- fences: a wait is owed only where the host (or the result) reads a device output ---------
    device_nodes = [i for i, node in enumerate(program.nodes) if side(node) == "device"]
    results = set(program.results)
    fenced = [
        i
        for i in device_nodes
        if any(
            out in results or any(side(program.nodes[c]) == HOST for c in consumers.get(out, ()))
            for out in program.nodes[i].outputs
        )
    ]

    # --- overlap: host work between a device node and the first node that needs its result --------
    depends: list[set[int]] = []  # per node, the device nodes it transitively depends on
    for index, node in enumerate(program.nodes):
        found: set[int] = set()
        for used in node.inputs:
            source = producer.get(used)
            if source is None:
                continue
            found |= depends[source]
            if side(program.nodes[source]) == "device":
                found.add(source)
        depends.append(found)

    host_elements = overlappable = 0
    last_device: int | None = None
    for index, node in enumerate(program.nodes):
        if side(node) == "device":
            last_device = index
            continue
        if node.kind != "dispatch" or all(out in constant for out in node.outputs):
            continue  # glue, or work that is not per-inference
        work = sum(_elements(program.buffers[out]) for out in node.outputs)
        host_elements += work
        if last_device is not None and last_device not in depends[index]:
            overlappable += work

    prepack_elements = sum(_elements(program.buffers[out]) for i in prepack_nodes for out in program.nodes[i].outputs)
    return {
        "schema": SCHEMA,
        "entry": program.entry,
        "nodes": {
            "dispatch": program.n_dispatches,
            "device": len(device_nodes),
            "host": program.n_dispatches - len(device_nodes),
        },
        "lifetimes": dict(sorted(by_kind.items())),
        "unsized_buffers": unsized,
        "staging": {
            "buffers": by_kind.get(STAGING, {}).get("buffers", 0),
            "bytes": by_kind.get(STAGING, {}).get("bytes", 0),
            "per_device_node": (
                round(by_kind.get(STAGING, {}).get("buffers", 0) / len(device_nodes), 3) if device_nodes else None
            ),
        },
        "prepack": {"dispatches": len(prepack_nodes), "elements": prepack_elements, "weights_known": bool(weights)},
        "fences": {
            "device_nodes": len(device_nodes),
            "true_dependences": len(fenced),
            "avoidable": len(device_nodes) - len(fenced),
        },
        "overlap": {
            "host_elements": host_elements,
            "independent_of_in_flight_device": overlappable,
            "share": round(overlappable / host_elements, 6) if host_elements else None,
        },
    }


def weight_args_of(manifest: Any) -> set[int]:
    """Model-argument indices a capture's weights manifest binds to stored tensors.

    The manifest maps each ``@forward`` argument index to what fills it. An argument the manifest
    calls an input is the caller's; every other bound argument is a stored tensor, constant for
    the life of the program.
    """
    found: set[int] = set()
    if not isinstance(manifest, dict):
        return found
    for key, entry in manifest.items():
        if str(key).isdigit() and isinstance(entry, dict) and entry.get("kind") != "input":
            found.add(int(key))
    return found


def weight_args_beside(capture: Any) -> set[int] | None:
    """The stored arguments of the capture at ``capture``, from the weights manifest beside it.

    ``None`` when there is no readable manifest, which is not the same as "no stored argument": a
    consumer that needs to tell a first layer's activation from its weight has to refuse then.
    """
    import json
    from pathlib import Path

    for manifest in sorted(Path(str(capture)).parent.glob("*.manifest.json")):
        try:
            return weight_args_of(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return None
