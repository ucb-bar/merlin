"""Five actual pinned producer buffers and one common complete scalar registry.

Run as an isolated script with only the pinned reference source on PYTHONPATH.
New generator/projection code lives in primary; producer files remain read-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .lean_graph_state_scalar_capture import REVISION, canonical, file_ref, metadata, project
else:
    from lean_graph_state_scalar_capture import REVISION, canonical, file_ref, metadata, project


def capture(reference, source, destination, *, factor=2, tile=3, bits=32):
    sys.dont_write_bytecode = True
    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.loops import segment_loops
    from merlin.transforms.pure_scalar import eliminate_scalar_redundancy, hoist_scalar_invariants
    from merlin.transforms.unrolling import unroll_ordered_loops
    from xdsl.dialects import arith, builtin

    reference, source, destination = reference.resolve(), source.resolve(), destination.resolve()
    revision = subprocess.run(["git", "-C", str(reference), "rev-parse", "HEAD"],
                              text=True, capture_output=True, check=True).stdout.strip()
    if revision != REVISION:
        raise ValueError("actual producer revision differs from pinned intake")
    destination.mkdir(parents=True, exist_ok=False)
    raw_input = source.read_bytes()
    module = parse_ir(raw_input.decode())
    module.verify()
    buffers, witnesses = [ir_text(module)], []
    for name, run in (
        ("cse-dce", lambda: eliminate_scalar_redundancy(module)),
        ("licm", lambda: hoist_scalar_invariants(module)),
        ("unroll", lambda: unroll_ordered_loops(module, (factor,), index_bits=bits)),
        ("segment", lambda: segment_loops(module, (tile,), index_bits=bits)),
    ):
        result = run()
        module.verify()
        witnesses.append({"stage": name, "diagnostic_only": [asdict(w) for w in result]})
        buffers.append(ir_text(module))
    (destination / "input.mlir").write_bytes(raw_input)
    for stage, text in enumerate(buffers):
        (destination / f"stage{stage}.mlir").write_text(text)
    # Reopen every exact retained middle; no independently numbered registry is equated.
    modules = [parse_ir(text) for text in buffers]
    stages = [project(modules[i], modules[i+1], context=tuple(modules)) for i in range(4)]
    for p in stages[1:]:
        if p["registry"] != stages[0]["registry"] or p["types"] != stages[0]["types"]:
            raise ValueError("one complete common registry/type table across all five stages required")
    index_ops = {}
    for m in modules:
        for op in m.walk():
            if type(op) is arith.ConstantOp and type(op.result.type) is builtin.IndexType:
                kind = {"kind": "index-constant", "value": op.value.value.data}
            elif type(op) in (arith.AddiOp, arith.MinUIOp) and all(
                type(v.type) is builtin.IndexType for v in (*op.operands, *op.results)
            ):
                kind = {"kind": "index-add" if type(op) is arith.AddiOp else "index-minui"}
            else:
                continue
            key = canonical(metadata(op))
            if key in index_ops and index_ops[key] != kind:
                raise ValueError("inconsistent concrete control descriptor")
            index_ops[key] = kind
    registry = stages[0]["registry"]
    concrete = [{"identity": i, **index_ops[canonical(row["descriptor"])]}
                for i, row in enumerate(registry) if canonical(row["descriptor"]) in index_ops]
    producer, runtime = {}, {}
    for name, loaded in tuple(sys.modules.items()):
        path = getattr(loaded, "__file__", None)
        if not path or not Path(path).is_file():
            continue
        path = Path(path).resolve()
        if name == "merlin" or name.startswith("merlin."):
            if not path.is_relative_to(reference / "src"):
                raise ValueError("actual producer import escaped pinned reference: " + str(path))
            producer[str(path)] = file_ref(path)
        else:
            runtime[str(path)] = file_ref(path)
    result = {
        "schema": "merlin.graph-state-pipeline.actual-five-buffer.v1",
        "revision": revision, "factor": factor, "tile": tile, "index_bits": bits,
        "stages": stages, "index_operations": concrete,
        "buffers": [file_ref(destination / f"stage{i}.mlir") for i in range(5)],
        "each_stage_changed": [buffers[i] != buffers[i+1] for i in range(4)],
        "diagnostics": witnesses, "input": file_ref(source),
        "producer_sources": sorted(producer.values(), key=lambda r: r["path"]),
        "python_runtime": {"executable": file_ref(sys.executable), "loaded": sorted(runtime.values(), key=lambda r: r["path"])},
        "owned_projection": file_ref(__file__),
        "projection_trusted": True, "lean_checked": False, "qualified": False,
        "full_pipeline_claim": False, "native_fidelity_proved": False, "review_approved": False,
    }
    if source.read_bytes() != raw_input:
        raise ValueError("source changed during capture")
    (destination / "CAPTURE.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "source", "destination"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--tile", type=int, default=3)
    parser.add_argument("--bits", type=int, default=32)
    result = capture(**vars(parser.parse_args()))
    print(json.dumps({"each_stage_changed": result["each_stage_changed"], "lean_checked": False}))


if __name__ == "__main__":
    main()
