"""Capture actual pinned plan_storage/reuse_storage and freshly reopen both IRs.

Complete trees and allocation/type/effect facts are parser-derived. Plans and
witnesses are retained as untrusted hints, never used by the independent checker.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

if __package__:
    from . import lean_graph_input_admission as admission
else:
    import lean_graph_input_admission as admission

if __package__:
    from .lean_graph_loops_capture import _producer_sources, _reference
    from .lean_graph_loops_segment_capture import loaded_xdsl_sources, operation_tree
else:
    from lean_graph_loops_capture import _producer_sources, _reference
    from lean_graph_loops_segment_capture import loaded_xdsl_sources, operation_tree


def export(module, *, calls="external-only"):
    from xdsl.dialects import builtin, memref
    from xdsl.traits import get_effects

    tree = operation_tree(module, calls=calls)
    allocations, pure_paths = {}, []

    def visit(op, record, path):
        if not op.regions and get_effects(op) == set():
            pure_paths.append(path)
        if isinstance(op, memref.AllocOp):
            kind = op.memref.type
            element = kind.get_element_type()
            allocations[str(record["results"][0][0])] = dict(
                shape=list(kind.get_shape()),
                element_bits=element.width.data if isinstance(element, builtin.IntegerType) else None,
                contiguous=isinstance(kind.layout, builtin.NoneAttr),
                default_memory_space=isinstance(kind.memory_space, builtin.NoneAttr),
                alignment=op.alignment.value.data if op.alignment is not None else 1,
                dynamic_sizes=len(op.dynamic_sizes),
                symbol_operands=len(op.symbol_operands),
                type=str(kind),
                path=path,
            )
        for ri, (region, records) in enumerate(zip(op.regions, record["regions"], strict=True)):
            for bi, (block, block_record) in enumerate(zip(region.blocks, records, strict=True)):
                for oi, (child, child_record) in enumerate(zip(block.ops, block_record["operations"], strict=True)):
                    visit(child, child_record, [*path, ri, bi, oi])

    visit(module, tree, [])
    return dict(tree=tree, allocations=allocations, pure_paths=pure_paths)


def run(*, source: Path, revision: str, before: Path, output: Path, after: Path | None = None, calls="external-only"):
    sys.dont_write_bytecode = True
    from merlin.transforms import planning as producer
    from merlin.transforms.certificates import ir_text
    from merlin.transforms.certificates import parse_ir as producer_parse_ir

    def parse_ir(raw):
        return admission.parse_admitted(raw, producer_parse_ir, calls=calls)

    source = source.resolve(strict=True)
    observed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if observed != revision:
        raise ValueError("actual storage producer/parser revision changed")
    original = before.read_bytes()
    module = parse_ir(original.decode())
    first = export(module, calls=calls)
    if after is None:
        generic = ir_text(module)
        plan = producer.plan_storage(module)
        witnesses = producer.reuse_storage(module, plan)
        module.verify()
        second = export(module, calls=calls)
        candidate = ir_text(module).encode()
        output.mkdir(parents=True, exist_ok=False)
        products = {
            "before.mlir": original,
            "before-generic.mlir": generic.encode(),
            "after.mlir": candidate,
            "before-export.json": (json.dumps(first, sort_keys=True, indent=2) + "\n").encode(),
            "after-export.json": (json.dumps(second, sort_keys=True, indent=2) + "\n").encode(),
            "untrusted-plan.json": (json.dumps(asdict(plan), indent=2) + "\n").encode(),
            "untrusted-witnesses.json": (json.dumps([asdict(row) for row in witnesses], indent=2) + "\n").encode(),
        }
        for name, data in products.items():
            (output / name).write_bytes(data)
        (output / "capture-implementation.py").write_bytes(Path(__file__).read_bytes())
        result = dict(
            schema="merlin.graph.actual_buffer_reuse_capture.v1",
            producer_revision=revision,
            producer_source=_producer_sources(source, revision, ()),
            xdsl_loaded_source=loaded_xdsl_sources(),
            capture_implementation=_reference(output / "capture-implementation.py"),
            actual_before=_reference(before),
            products={name: _reference(output / name) for name in products},
            changed=bool(witnesses),
            changed_groups=len(witnesses),
            witness_or_plan_is_authority=False,
            input_admission_source=admission.source_reference(),
            input_admission_calls=calls,
            python_projection_trusted=True,
            lean_checked=False,
            allocator_contract_proved=False,
            raw_SCF_bridge_proved=False,
            numerical_gates_discharged=False,
            pipeline_receipt_issued=False,
            renderer_or_hardware_proved=False,
        )
        destination = output / "CAPTURE.json"
    else:
        candidate = after.read_bytes()
        second = export(parse_ir(candidate.decode()), calls=calls)
        result = dict(
            schema="merlin.graph.actual_buffer_reuse_export.v1",
            before=_reference(before),
            after=_reference(after),
            before_export=first,
            after_export=second,
            producer_revision=revision,
            producer_source=_producer_sources(source, revision, (Path(producer.__file__).resolve(),)),
            xdsl_loaded_source=loaded_xdsl_sources(),
            input_admission_source=admission.source_reference(),
            input_admission_calls=calls,
            python_projection_trusted=True,
            pipeline_receipt_issued=False,
        )
        destination = output
    if (
        before.read_bytes() != original
        or (after is not None and after.read_bytes() != candidate)
        or result["producer_source"] != _producer_sources(source, revision, ())
        or result["xdsl_loaded_source"] != loaded_xdsl_sources()
    ):
        raise ValueError("actual modules or selected loaded source/parser changed during capture/export")
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "before", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--calls", choices=("external-only", "shape-direct"), default="external-only")
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("schema", "producer_revision")}))


if __name__ == "__main__":
    main()
