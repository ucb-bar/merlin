"""Actual pinned five-stage producer and independent complete-buffer projection.

Run in an isolated process with the pinned source as the sole Merlin owner.
Every submitted stage is re-parsed/verified; metadata and all SSA uses are kept.
Python projection/host delivery remain explicit trust boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from functools import cache
from pathlib import Path

if __package__:
    from . import lean_graph_input_admission as admission
else:
    import lean_graph_input_admission as admission

if __package__:
    from .lean_graph_state_scalar_capture import REVISION, canonical, file_ref, metadata, project
else:
    from lean_graph_state_scalar_capture import REVISION, canonical, file_ref, metadata, project


def index_operation(op):
    """Closed, independent control whitelist; flags/extra metadata are refused."""
    from xdsl.dialects import arith, builtin
    from xdsl.traits import get_effects

    if op.regions or op.successors or op.attributes or len(op.results) != 1 or get_effects(op) != set():
        return None
    if any(type(v.type) is not builtin.IndexType for v in (*op.operands, *op.results)):
        return None
    if type(op) is arith.ConstantOp and not op.operands and set(op.properties) == {"value"}:
        if type(op.value) is builtin.IntegerAttr:
            return {"kind": "index-constant", "value": op.value.value.data}
    if type(op) is arith.AddiOp and len(op.operands) == 2:
        if set(op.properties) <= {"overflowFlags"} and op.overflow_flags == arith.IntegerOverflowAttr("none"):
            return {"kind": "index-add"}
    if type(op) is arith.MinUIOp and len(op.operands) == 2 and not op.properties:
        return {"kind": "index-minui"}
    return None


def project_modules(modules):
    if len(modules) != 5:
        raise ValueError("five complete actual programs required")
    from xdsl.dialects import func

    for module in modules:
        admission.admit_module(module)
        module.verify()
        declarations = {
            op.sym_name.data: op for op in module.body.block.ops if type(op) is func.FuncOp and not op.body.blocks
        }
        for call in module.walk():
            if type(call) is not func.CallOp:
                continue
            if len(call.callee.nested_references):
                raise ValueError("nested external symbol references require a separate call bridge")
            declaration = declarations.get(call.callee.root_reference.data)
            if declaration is None:
                raise ValueError("only explicitly declared external calls are admitted")
            if tuple(value.type for value in call.operands) != tuple(declaration.function_type.inputs) or tuple(
                value.type for value in call.results
            ) != tuple(declaration.function_type.outputs):
                raise ValueError("actual call operand/result signature differs from its explicit declaration")
    stages = [project(modules[i], modules[i + 1], context=tuple(modules)) for i in range(4)]
    for stage in stages[1:]:
        if stage["registry"] != stages[0]["registry"] or stage["types"] != stages[0]["types"]:
            raise ValueError("one common complete descriptor/type registry required")
    index_ops = {}
    for module in modules:
        for op in module.walk():
            kind = index_operation(op)
            if kind is None:
                continue
            token = canonical(metadata(op))
            if token in index_ops and index_ops[token] != kind:
                raise ValueError("inconsistent concrete control descriptor")
            index_ops[token] = kind
    concrete = [
        {"identity": i, **index_ops[canonical(row["descriptor"])]}
        for i, row in enumerate(stages[0]["registry"])
        if canonical(row["descriptor"]) in index_ops
    ]
    return stages, concrete


def selected_source(reference):
    observed = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    if observed != REVISION:
        raise ValueError("pinned actual producer/parser revision differs")
    return observed


@cache
def expected_producer(reference, path):
    relative = path.relative_to(reference)
    data = subprocess.run(
        ["git", "-C", str(reference), "show", REVISION + ":" + str(relative)], capture_output=True, check=True
    ).stdout
    return hashlib.sha256(data).hexdigest()


def loaded_sources(reference):
    producer, runtime = {}, {}
    for name, loaded in tuple(sys.modules.items()):
        path = getattr(loaded, "__file__", None)
        if not path or not Path(path).is_file():
            continue
        path = Path(path).resolve()
        if name == "merlin" or name.startswith("merlin."):
            if not path.is_relative_to(reference / "src"):
                raise ValueError("actual producer import escaped pinned reference")
            producer[str(path)] = file_ref(path)
        else:
            runtime[str(path)] = file_ref(path)
    for name in ("pure_scalar", "unrolling", "loops", "certificates"):
        path = reference / "src/merlin/transforms" / (name + ".py")
        producer[str(path)] = file_ref(path)
    for row in producer.values():
        if row["sha256"] != expected_producer(reference, Path(row["path"])):
            raise ValueError("loaded pinned producer/parser source differs from selected commit")
    return sorted(producer.values(), key=lambda r: r["path"]), {
        "executable": file_ref(sys.executable),
        "loaded": sorted(runtime.values(), key=lambda r: r["path"]),
    }


def write_projection(reference, buffers, destination, *, factor, tile, bits, mode, diagnostics=(), input_path=None):
    from merlin.transforms.certificates import parse_ir as producer_parse_ir

    def parse_ir(raw):
        return admission.parse_admitted(raw, producer_parse_ir)

    revision = selected_source(reference)
    modules = [parse_ir(data.decode()) for data in buffers]
    stages, concrete = project_modules(modules)
    producer, runtime = loaded_sources(reference)
    result = {
        "schema": "merlin.graph-state-pipeline.actual-five-buffer.v2",
        "revision": revision,
        "factor": factor,
        "tile": tile,
        "index_bits": bits,
        "stages": stages,
        "index_operations": concrete,
        "buffers": [file_ref(destination / f"stage{i}.mlir") for i in range(5)],
        "each_stage_changed": [buffers[i] != buffers[i + 1] for i in range(4)],
        "diagnostics": list(diagnostics),
        "input": file_ref(input_path) if input_path else None,
        "producer_sources": producer,
        "python_runtime": runtime,
        "owned_projection": file_ref(__file__),
        "input_admission_source": admission.source_reference(),
        "selected_scalar_projection": file_ref(Path(__file__).with_name("lean_graph_state_scalar_capture.py")),
        "mode": mode,
        "projection_trusted": True,
        "lean_checked": False,
        "qualified": False,
        "full_pipeline_claim": False,
        "native_fidelity_proved": False,
        "review_approved": False,
    }
    if any((destination / f"stage{i}.mlir").read_bytes() != data for i, data in enumerate(buffers)):
        raise ValueError("stage buffers changed during projection")
    if loaded_sources(reference) != (producer, runtime):
        raise ValueError("selected source/runtime identity changed during projection")
    (destination / "CAPTURE.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def capture(reference, source, destination, *, factor=2, tile=3, bits=32):
    sys.dont_write_bytecode = True
    from merlin.transforms.certificates import ir_text
    from merlin.transforms.certificates import parse_ir as producer_parse_ir

    def parse_ir(raw):
        return admission.parse_admitted(raw, producer_parse_ir)

    from merlin.transforms.loops import segment_loops
    from merlin.transforms.pure_scalar import eliminate_scalar_redundancy, hoist_scalar_invariants
    from merlin.transforms.unrolling import unroll_ordered_loops

    reference, source, destination = reference.resolve(), source.resolve(), destination.resolve()
    selected_source(reference)
    destination.mkdir(parents=True, exist_ok=False)
    original = source.read_bytes()
    (destination / "input.mlir").write_bytes(original)
    module = parse_ir(original.decode())
    module.verify()
    buffers = [ir_text(module).encode()]
    witnesses = []
    (destination / "stage0.mlir").write_bytes(buffers[0])
    for index, (name, run) in enumerate(
        (
            ("cse-dce", lambda: eliminate_scalar_redundancy(module)),
            ("licm", lambda: hoist_scalar_invariants(module)),
            ("unroll", lambda: unroll_ordered_loops(module, (factor,), index_bits=bits)),
            ("segment", lambda: segment_loops(module, (tile,), index_bits=bits)),
        ),
        1,
    ):
        diagnostics = run()
        module.verify()
        witnesses.append({"stage": name, "diagnostic_only": [asdict(w) for w in diagnostics]})
        buffers.append(ir_text(module).encode())
        (destination / f"stage{index}.mlir").write_bytes(buffers[-1])
    result = write_projection(
        reference,
        buffers,
        destination,
        factor=factor,
        tile=tile,
        bits=bits,
        mode="actual-producer",
        diagnostics=witnesses,
        input_path=source,
    )
    if source.read_bytes() != original:
        raise ValueError("input changed during actual production")
    return result


def reopen(reference, paths, destination, *, factor=2, tile=3, bits=32):
    sys.dont_write_bytecode = True
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    if len(paths) != 5:
        raise ValueError("five exact submitted buffers required")
    buffers = [p.read_bytes() for p in paths]
    for i, data in enumerate(buffers):
        (destination / f"stage{i}.mlir").write_bytes(data)
    result = write_projection(
        reference.resolve(), buffers, destination, factor=factor, tile=tile, bits=bits, mode="independent-reprojection"
    )
    if any(path.read_bytes() != data for path, data in zip(paths, buffers, strict=True)):
        raise ValueError("submitted originals changed during independent projection")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--buffers", nargs=5, type=Path)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--tile", type=int, default=3)
    parser.add_argument("--bits", type=int, default=32)
    args = vars(parser.parse_args())
    source = args.pop("source")
    paths = args.pop("buffers")
    if (source is None) == (paths is None):
        parser.error("select source production OR five complete buffers")
    try:
        result = capture(source=source, **args) if source is not None else reopen(paths=paths, **args)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        dest = args["destination"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "REJECTION.json").write_text(
            json.dumps(
                {
                    "status": "unsupported",
                    "reason": str(error),
                    "kind": type(error).__name__,
                    "qualified": False,
                    "projection_trusted": True,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"status": "unsupported", "reason": str(error)}))
        raise SystemExit(1) from error
    print(json.dumps({"each_stage_changed": result["each_stage_changed"], "lean_checked": False}))


if __name__ == "__main__":
    main()
