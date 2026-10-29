"""Actual pinned exact-integer producer and independent complete buffer intake.

Runs with the selected read-only producer as the only Merlin import owner.
Diagnostic witnesses never authorize equality. All new code belongs to primary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__:
    from . import lean_graph_input_admission as admission
else:
    import lean_graph_input_admission as admission

if __package__:
    from .lean_graph_state_pipeline_capture import loaded_sources, project_modules, selected_source
    from .lean_graph_state_scalar_capture import file_ref, project
else:
    from lean_graph_state_pipeline_capture import loaded_sources, project_modules, selected_source
    from lean_graph_state_scalar_capture import file_ref, project


def validate_calls(modules):
    from xdsl.dialects import func

    for module in modules:
        admission.admit_module(module)
        module.verify()
        declarations = {
            op.sym_name.data: op for op in module.body.block.ops if type(op) is func.FuncOp and not op.body.blocks
        }
        for op in module.walk():
            if type(op) is not func.CallOp:
                continue
            selected = declarations.get(op.callee.root_reference.data)
            if selected is None or op.callee.nested_references:
                raise ValueError("explicit nonnested external declaration required")
            if tuple(v.type for v in op.operands) != tuple(selected.function_type.inputs) or tuple(
                v.type for v in op.results
            ) != tuple(selected.function_type.outputs):
                raise ValueError("actual call signature differs from declaration")


def projection(reference, buffers, destination, *, factor, tile, bits, mode, diagnostics=()):
    from merlin.transforms.certificates import parse_ir as producer_parse_ir

    def parse_ir(raw):
        return admission.parse_admitted(raw, producer_parse_ir)

    revision = selected_source(reference)
    if len(buffers) not in (2, 6):
        raise ValueError("two standalone or six composed exact buffers required")
    modules = [parse_ir(data.decode()) for data in buffers]
    validate_calls(modules)
    prefix = project(modules[0], modules[1], context=tuple(modules[2:]))
    pipeline = None
    if len(buffers) == 6:
        stages, controls = project_modules(modules[1:])
        if any(s["registry"] != prefix["registry"] or s["types"] != prefix["types"] for s in stages):
            raise ValueError("one exact six-buffer common registry/type projection required")
        pipeline = {
            "stages": stages,
            "index_operations": controls,
            "factor": factor,
            "tile": tile,
            "index_bits": bits,
        }
    producer, runtime = loaded_sources(reference)
    result = {
        "schema": "merlin.graph-state-integer.actual-boundary.v1",
        "revision": revision,
        "prefix": prefix,
        "pipeline": pipeline,
        "buffers": [file_ref(destination / f"stage{i}.mlir") for i in range(len(buffers))],
        "each_stage_changed": [first != second for first, second in zip(buffers, buffers[1:])],
        "mode": mode,
        "diagnostic_only": list(diagnostics),
        "producer_sources": producer,
        "python_runtime": runtime,
        "owned_projection": file_ref(__file__),
        "input_admission_source": admission.source_reference(),
        "projection_trusted": True,
        "lean_checked": False,
        "qualified": False,
        "native_fidelity_proved": False,
        "poison_freedom_proved": False,
        "review_approved": False,
    }
    if any((destination / f"stage{i}.mlir").read_bytes() != data for i, data in enumerate(buffers)):
        raise ValueError("submitted buffers changed during intake")
    if loaded_sources(reference) != (producer, runtime):
        raise ValueError("producer/runtime identity changed during intake")
    (destination / "CAPTURE.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def capture(reference, source, destination, *, chain=False, factor=2, tile=3, bits=32):
    from merlin.transforms.certificates import ir_text
    from merlin.transforms.certificates import parse_ir as producer_parse_ir

    def parse_ir(raw):
        return admission.parse_admitted(raw, producer_parse_ir)

    from merlin.transforms.normalization import simplify_integers

    reference, source, destination = reference.resolve(), source.resolve(), destination.resolve()
    selected_source(reference)
    destination.mkdir(parents=True, exist_ok=False)
    original = source.read_bytes()
    (destination / "input.mlir").write_bytes(original)
    module = parse_ir(original.decode())
    module.verify()
    buffers, diagnostics = [ir_text(module).encode()], []
    runs = [("exact-integer", lambda: simplify_integers(module))]
    if chain:
        from merlin.transforms.loops import segment_loops
        from merlin.transforms.pure_scalar import eliminate_scalar_redundancy, hoist_scalar_invariants
        from merlin.transforms.unrolling import unroll_ordered_loops

        runs += [
            ("cse-dce", lambda: eliminate_scalar_redundancy(module)),
            ("licm", lambda: hoist_scalar_invariants(module)),
            ("unroll", lambda: unroll_ordered_loops(module, (factor,), index_bits=bits)),
            ("segment", lambda: segment_loops(module, (tile,), index_bits=bits)),
        ]
    for name, run in runs:
        witnesses = run()
        module.verify()
        diagnostics.append({"stage": name, "witnesses": [asdict(w) for w in witnesses]})
        buffers.append(ir_text(module).encode())
    for index, data in enumerate(buffers):
        (destination / f"stage{index}.mlir").write_bytes(data)
    result = projection(
        reference,
        buffers,
        destination,
        factor=factor,
        tile=tile,
        bits=bits,
        mode="actual-producer",
        diagnostics=diagnostics,
    )
    if source.read_bytes() != original:
        raise ValueError("producer input changed")
    return result


def reopen(reference, paths, destination, *, factor=2, tile=3, bits=32):
    destination.mkdir(parents=True, exist_ok=False)
    buffers = [p.read_bytes() for p in paths]
    for index, data in enumerate(buffers):
        (destination / f"stage{index}.mlir").write_bytes(data)
    result = projection(
        reference, buffers, destination, factor=factor, tile=tile, bits=bits, mode="independent-reprojection"
    )
    if any(path.read_bytes() != data for path, data in zip(paths, buffers, strict=True)):
        raise ValueError("submitted original changed during reprojection")
    return result


def main():
    from xdsl.utils.exceptions import ParseError, VerifyException

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--buffers", type=Path, nargs="+")
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--tile", type=int, default=3)
    parser.add_argument("--bits", type=int, default=32)
    args = vars(parser.parse_args())
    source, paths = args.pop("source"), args.pop("buffers")
    if (source is None) == (paths is None):
        parser.error("select actual source OR original buffers")
    try:
        if source is not None:
            result = capture(source=source, **args)
        else:
            args.pop("chain")
            result = reopen(paths=paths, **args)
    except (ValueError, ParseError, VerifyException) as error:
        args["destination"].mkdir(parents=True, exist_ok=True)
        (args["destination"] / "REJECTION.json").write_text(
            json.dumps(
                {
                    "schema": "merlin.graph-state-integer.host-domain-refusal.v1",
                    "status": "malformed" if isinstance(error, (ParseError, VerifyException)) else "unsupported",
                    "reason": str(error),
                    "kind": type(error).__name__,
                    "qualified": False,
                },
                indent=2,
            )
            + "\n"
        )
        raise SystemExit(1) from error
    print(json.dumps({"each_stage_changed": result["each_stage_changed"], "lean_checked": False}))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
