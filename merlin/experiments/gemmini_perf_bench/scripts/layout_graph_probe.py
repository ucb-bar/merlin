#!/usr/bin/env python3
"""How large a search a whole model's layout plan actually is, measured on a recorded emission.

`perf.physical_layout` plans physical encodings EXHAUSTIVELY: it takes the cartesian product of the
per-component encoding domains and refuses, rather than approximating, when that product exceeds the
graph's declared `max_assignments`. Whether it can plan a fifty-layer model is therefore arithmetic,
and the arithmetic is over COUPLED COMPONENTS, not values.

Exactly ONE mechanism shrinks the search: a `couple=True` operation unions its propagated ports into
one component. A non-coupled operation does NOT narrow anything -- `propagated_ports` is empty when
`couple` is false, the domain intersection in `_component_domains` is guarded on it, and at
`physical_layout.py:629-630` such an operation simply takes `options[0].encoding_id` while `_evaluate`
inserts conversions wherever ports then disagree. So a convolution being a fixed NHWC boundary buys no
reduction at all; it only guarantees conversions around itself. (This probe was written believing the
opposite, and reporting `0 components pinned` is what disproved it.)

WHAT THIS IS NOT. The encodings are a STATED MODEL, not the target's own facts: two layouts per value,
a convolution as a non-coupled NHWC boundary, everything else coupled. A real adapter derives them from
the readout facet and the target's conversion capabilities.

AND WHAT ITS NUMBERS ARE NOT. On the recorded ResNet-50 emission this keeps only the 53 convolution
commands of 109, because the `RES_PACK` operands carry no shape or dtype there and a `LayoutValue`
requires positive static extents. The result is a disconnected set of boundaries with nothing to
couple, so the search it reports is the trivial product over unmerged values -- a statement about this
probe's input, not about the model. A representative measurement needs a graph whose coupled
operations are present, which needs shapes the emission does not record for them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_root = os.environ.get("MERLIN_REPO_ROOT", "").strip()
REPO = Path(_root).expanduser().resolve() if _root else _HERE.parents[4]
sys.path.insert(0, str(REPO / "merlin/python"))

#: The two layouts a 4-D activation can carry in this model, and the one the sequencer accepts.
_LAYOUTS = ("nhwc", "nchw")
_NATIVE = "nhwc"
#: Opcodes that are a fixed physical boundary: they accept one layout and convert nothing.
_BOUNDARY_OPS = ("CONV2D",)


def _encodings(name: str, shape, dtype: str):
    from merlin.perf.physical_layout import PhysicalEncoding

    elems = 1
    for dim in shape:
        elems *= int(dim)
    width = {"i8": 1, "i16": 2, "i32": 4, "f32": 4, "bf16": 2}.get(dtype, 1)
    return tuple(
        PhysicalEncoding(
            id=f"{layout}",
            layout=layout,
            storage_space="dram",
            storage_bytes=max(1, elems * width),
            provenance=(f"emission:{name}",),
        )
        for layout in _LAYOUTS
    )


def build_graph(doc, *, max_assignments: int):
    from merlin.perf.physical_layout import (
        ConversionCapability,
        LayoutGraph,
        LayoutOp,
        LayoutValue,
        OperatorEncoding,
    )

    tensors = doc["tensors"]
    used: dict[str, dict] = {}
    for command in doc["commands"]:
        for name in (command.get("operands") or {}).values():
            record = tensors.get(name) or {}
            if record.get("shape") and record.get("dtype"):
                used[name] = record

    values = tuple(
        LayoutValue(
            name=n,
            shape=tuple(int(d) for d in r["shape"]),
            dtype=r["dtype"],
            encodings=_encodings(n, r["shape"], r["dtype"]),
        )
        for n, r in sorted(used.items())
    )
    known = {v.name for v in values}

    ops = []
    for index, command in enumerate(doc["commands"]):
        operands = {role: name for role, name in (command.get("operands") or {}).items() if name in known}
        if not operands:
            continue
        outs = tuple(name for role, name in operands.items() if role == "dst")
        ins = tuple(name for role, name in operands.items() if role != "dst")
        if not ins or not outs:
            continue
        boundary = command["opcode"] in _BOUNDARY_OPS
        options = (
            (
                OperatorEncoding(
                    id=f"{command['opcode']}_{index}_native",
                    encoding_id=_NATIVE,
                    provenance=(f"emission:{command['opcode']}",),
                ),
            )
            if boundary
            else tuple(
                OperatorEncoding(
                    id=f"{command['opcode']}_{index}_{layout}",
                    encoding_id=layout,
                    provenance=(f"emission:{command['opcode']}",),
                )
                for layout in _LAYOUTS
            )
        )
        ops.append(
            LayoutOp(
                name=f"{command['opcode']}_{index}",
                kind=command["opcode"].lower(),
                inputs=ins,
                outputs=outs,
                couple=not boundary,
                options=options,
                provenance=(f"emission:index{index}",),
            )
        )

    conversions = tuple(
        ConversionCapability(
            id=f"{src}_to_{dst}",
            source_encoding=src,
            destination_encoding=dst,
            read_scope="logical_payload",
            write_scope="logical_payload",
            provenance=("probe:stated_model",),
        )
        for src in _LAYOUTS
        for dst in _LAYOUTS
        if src != dst
    )
    return LayoutGraph(
        values=values, ops=tuple(ops), capabilities=("dram",), conversions=conversions, max_assignments=max_assignments
    )


#: Step kinds that are a fixed physical boundary on this device, ASSUMED. Prefer `derive_boundary_kinds`,
#: which reads the same fact off the target's instruction set; this remains only as the fallback for a
#: target whose schedule ISA cannot be loaded, and as the knob the sensitivity sweep varies.
_BOUNDARY_KINDS = ("conv2d", "matmul")

#: Step kind -> the schedule instruction that performs it. A kind with no instruction here is not a
#: contraction on this unit (an integer sum, a pooled mean) and is elementwise over whatever layout its
#: operands carry, so it couples.
_KIND_INSTRUCTION = {"conv2d": "loop_conv_ws", "matmul": "loop_ws"}

#: Operand-name fragments that name a SPATIAL EXTENT rather than a tile count. An instruction whose
#: operands name these is addressing the layout's own axes.
_SPATIAL_OPERAND_MARKERS = ("row_dim", "col_dim", "channels", "orows", "ocols", "porows", "pocols")


def derive_boundary_kinds(target: str) -> tuple[str, ...]:
    """Which step kinds are layout BOUNDARIES, read off the target's own instruction operands.

    The distinction is in the ISA and does not need assuming. An instruction that walks tiles and is
    handed explicit row strides consumes any row-major view of its operands, so the layout its data
    arrived in is not its business and it propagates whatever it is given -- it COUPLES. An instruction
    whose operand list names the layout's own axes has a descriptor that presupposes one ordering, and
    is a boundary.

    Measured on this target: ``loop_ws`` takes 23 operands and names no spatial extent (it is I/J/K
    plus strides), while ``loop_conv_ws`` takes 54 and names ``in_row_dim``, ``in_col_dim``,
    ``in_channels``, ``out_channels``, ``out_row_dim`` and ``out_col_dim``. So the mesh is
    layout-polymorphic and the convolution sequencer is not, which is the difference between a search
    of 2^25 and one of 2^93 over this model.

    A target whose schedule instruction set cannot be loaded gets the assumed set, flagged by the
    caller, rather than a silent guess.
    """
    from merlin.runtime.backends import base

    iset = base.get_backend(target).sched_instruction_set()
    boundaries = []
    for kind, instruction in _KIND_INSTRUCTION.items():
        if instruction not in iset.instrs:
            continue
        names = [operand.name for operand in iset.instr(instruction).operands]
        if any(marker in name for name in names for marker in _SPATIAL_OPERAND_MARKERS):
            boundaries.append(kind)
    return tuple(sorted(boundaries))


def build_graph_from_groups(model, *, max_assignments: int, boundary_kinds=None):
    """A layout graph over the model's COMPUTE GROUPS rather than a legacy command buffer.

    This is the representative source: every group states its operand and result buffers with real
    extents, so the coupled operations survive instead of being dropped for want of a shape. Buffer
    element counts come from the extractor; a 1-D extent is enough here because the planner's search
    depends on the graph's connectivity, not on the byte cost of any one encoding.
    """
    from merlin.perf.physical_layout import (
        ConversionCapability,
        LayoutGraph,
        LayoutOp,
        LayoutValue,
        OperatorEncoding,
    )

    sizes = {b["name"]: max(1, int(b.get("elements") or 1)) for b in model.get("buffers", [])}
    used: set[str] = set()
    plan = []
    boundaries = boundary_kinds if boundary_kinds is not None else _BOUNDARY_KINDS
    for index, step in enumerate(model.get("steps", [])):
        out = step.get("out")
        ins = [step[role] for role in ("in", "lhs", "rhs", "weight", "window") if step.get(role)]
        ins = [n for n in ins if isinstance(n, str)]
        if not out or not ins:
            continue
        plan.append((index, step, ins, out))
        used.update(ins)
        used.add(out)

    values = tuple(
        LayoutValue(name=n, shape=(sizes.get(n, 1),), dtype="i8", encodings=_encodings(n, (sizes.get(n, 1),), "i8"))
        for n in sorted(used)
    )
    ops = []
    for index, step, ins, out in plan:
        boundary = step.get("kind") in boundaries
        options = (
            (
                OperatorEncoding(
                    id=f"{step['kind']}_{index}_native", encoding_id=_NATIVE, provenance=(f"group:{step.get('group')}",)
                ),
            )
            if boundary
            else tuple(
                OperatorEncoding(
                    id=f"{step['kind']}_{index}_{layout}",
                    encoding_id=layout,
                    provenance=(f"group:{step.get('group')}",),
                )
                for layout in _LAYOUTS
            )
        )
        ops.append(
            LayoutOp(
                name=f"{step['kind']}_{index}",
                kind=step["kind"],
                inputs=tuple(ins),
                outputs=(out,),
                couple=not boundary,
                options=options,
                provenance=(f"group:{step.get('group')}",),
            )
        )
    conversions = tuple(
        ConversionCapability(
            id=f"{s}_to_{d}",
            source_encoding=s,
            destination_encoding=d,
            read_scope="logical_payload",
            write_scope="logical_payload",
            provenance=("probe:stated_model",),
        )
        for s in _LAYOUTS
        for d in _LAYOUTS
        if s != d
    )
    return LayoutGraph(
        values=values, ops=tuple(ops), capabilities=("dram",), conversions=conversions, max_assignments=max_assignments
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emission", type=Path, help="a recorded command_buffer.json")
    ap.add_argument("--capture", type=Path, help="a recapture directory; uses its COMPUTE GROUPS")
    ap.add_argument("--target", default="gemmini")
    ap.add_argument("--max-assignments", type=int, default=100_000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from math import prod

    from merlin.perf.physical_layout import _component_domains

    if args.capture:
        import importlib.util

        spec = importlib.util.spec_from_file_location("gmp", str(_HERE.with_name("group_model_program.py")))
        gmp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gmp)
        model = gmp.extract(args.capture, args.target)
        try:
            derived = derive_boundary_kinds(args.target)
            how = "derived from the target's instruction operands"
        except Exception as exc:  # noqa: BLE001 -- a target with no schedule ISA is named, not guessed
            derived, how = _BOUNDARY_KINDS, f"ASSUMED ({type(exc).__name__}); the ISA could not be read"
        graph = build_graph_from_groups(model, max_assignments=args.max_assignments, boundary_kinds=derived)
        source = {
            "capture": str(args.capture),
            "steps": len(model.get("steps", [])),
            "boundary_kinds": list(derived),
            "boundary_kinds_how": how,
        }
    elif args.emission:
        graph = build_graph(json.loads(args.emission.read_text()), max_assignments=args.max_assignments)
        source = {"emission": str(args.emission)}
    else:
        ap.error("one of --capture or --emission is required")
    domains, _index = _component_domains(graph)
    free = [d for d in domains if len(d) > 1]
    pinned = [d for d in domains if len(d) == 1]
    empty = [d for d in domains if not d]
    search = prod(len(d) for d in domains) if not empty else 0

    report = {
        "schema": "layout_graph_shape_v1",
        "source": source,
        "model": {
            "layouts": list(_LAYOUTS),
            "native": _NATIVE,
            "boundary_opcodes": list(_BOUNDARY_OPS),
            "note": "a STATED model, not the target's own facts; measures graph shape only",
        },
        "values": len(graph.values),
        "ops": len(graph.ops),
        "components": len(domains),
        "components_pinned_to_one_encoding": len(pinned),
        "components_with_a_free_choice": len(free),
        "components_with_no_common_encoding": len(empty),
        "search_size": search,
        "max_assignments": args.max_assignments,
        "would_refuse": bool(empty) or search > args.max_assignments,
    }
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
