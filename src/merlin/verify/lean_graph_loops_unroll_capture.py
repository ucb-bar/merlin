"""Actual pinned ordered-unrolling production and complete SSA tree evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .lean_graph_loops_capture import _producer_sources, _reference
    from .lean_graph_loops_segment_capture import loaded_xdsl_sources, operation_tree
else:
    from lean_graph_loops_capture import _producer_sources, _reference
    from lean_graph_loops_segment_capture import loaded_xdsl_sources, operation_tree


def capture(
    *,
    source: Path,
    revision: str,
    before: Path,
    output: Path,
    factor: int,
    index_bits: int,
    max_added_operations: int,
    disabled: bool = False,
):
    sys.dont_write_bytecode = True
    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.unrolling import unroll_ordered_loops

    source = source.resolve(strict=True)
    observed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if observed != revision:
        raise ValueError("actual unrolling producer revision changed")
    original = before.read_bytes()
    module = parse_ir(original.decode())
    before_tree = operation_tree(module)
    before_generic = ir_text(module)
    witnesses = unroll_ordered_loops(
        module, () if disabled else (factor,), index_bits=index_bits, max_added_operations=max_added_operations
    )
    module.verify()
    after_tree = operation_tree(module)
    closure = _producer_sources(source, revision, ())
    parser_sources = loaded_xdsl_sources()
    output.mkdir(parents=True, exist_ok=False)
    products = {
        "before.mlir": original.decode(),
        "before-generic.mlir": before_generic,
        "after.mlir": ir_text(module),
        "before-tree.json": json.dumps(before_tree, sort_keys=True, indent=2) + "\n",
        "after-tree.json": json.dumps(after_tree, sort_keys=True, indent=2) + "\n",
        "witnesses.json": json.dumps([asdict(row) for row in witnesses], indent=2) + "\n",
    }
    for name, text in products.items():
        (output / name).write_text(text)
    (output / "capture-implementation.py").write_bytes(Path(__file__).read_bytes())
    if (
        before.read_bytes() != original
        or closure != _producer_sources(source, revision, ())
        or parser_sources != loaded_xdsl_sources()
    ):
        raise ValueError("actual input or selected producer/parser code changed during capture")
    result = {
        "schema": "merlin.graph.actual_ordered_unroll_capture.v1",
        "producer_revision": revision,
        "producer_source": closure,
        "xdsl_loaded_source": parser_sources,
        "capture_implementation": _reference(output / "capture-implementation.py"),
        "actual_before": _reference(before),
        "products": {name: _reference(output / name) for name in products},
        "factor": factor,
        "disabled": disabled,
        "index_bits": index_bits,
        "max_added_operations": max_added_operations,
        "changed_loops": len(witnesses),
        "changed": bool(witnesses),
        "untrusted_witness_is_authority": False,
        "python_projection_trusted": True,
        "lean_checked": False,
        "pipeline_receipt_issued": False,
        "numerical_gates_discharged": False,
        "renderer_or_hardware_proved": False,
    }
    (output / "CAPTURE.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def project(*, source: Path, revision: str, before: Path, after: Path, output: Path):
    sys.dont_write_bytecode = True
    from merlin.transforms import unrolling as producer_module
    from merlin.transforms.certificates import parse_ir

    source = source.resolve(strict=True)
    observed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if observed != revision:
        raise ValueError("actual unroll/parser source revision changed")
    first, second = before.read_bytes(), after.read_bytes()
    trees = [operation_tree(parse_ir(data.decode())) for data in (first, second)]
    producer_file = Path(producer_module.__file__).resolve()
    closure, parser_sources = _producer_sources(source, revision, (producer_file,)), loaded_xdsl_sources()
    result = {
        "schema": "merlin.graph.actual_unroll_trees.v1",
        "before": _reference(before),
        "after": _reference(after),
        "before_tree": trees[0],
        "after_tree": trees[1],
        "producer_revision": revision,
        "producer_source": closure,
        "xdsl_loaded_source": parser_sources,
        "python_projection_trusted": True,
        "pipeline_receipt_issued": False,
    }
    if (
        before.read_bytes() != first
        or after.read_bytes() != second
        or closure != _producer_sources(source, revision, (producer_file,))
        or parser_sources != loaded_xdsl_sources()
    ):
        raise ValueError("actual unrolling modules or source/parser changed while reopening")
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "project"))
    for name in ("source", "before", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--factor", type=int)
    parser.add_argument("--index-bits", type=int)
    parser.add_argument("--max-added-operations", type=int, default=4096)
    parser.add_argument("--disabled", action="store_true")
    data = vars(parser.parse_args())
    mode = data.pop("mode")
    if mode == "capture":
        data.pop("after")
        result = capture(**data)
        print(json.dumps({key: result[key] for key in ("changed", "changed_loops", "factor", "lean_checked")}))
    else:
        for key in ("factor", "index_bits", "max_added_operations", "disabled"):
            data.pop(key)
        project(**data)
        print(json.dumps({"actual_trees_projected": True, "lean_checked": False}))


if __name__ == "__main__":
    main()
