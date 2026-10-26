"""Pinned actual shape normalizer production and fresh complete typed IR export.

Descriptor conformance is a semantic contract; static IR type facts are not
runtime descriptor witnesses. Parsing/projection remains trusted Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__:
    from . import lean_graph_input_admission as admission
else:
    import lean_graph_input_admission as admission

if __package__:
    from .lean_graph_loops_capture import _producer_sources
    from .lean_graph_loops_segment_capture import loaded_xdsl_sources, operation_tree
else:
    from lean_graph_loops_capture import _producer_sources
    from lean_graph_loops_segment_capture import loaded_xdsl_sources, operation_tree


def reference(path, data=None):
    data = path.read_bytes() if data is None else data
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def type_record(kind):
    from xdsl.dialects import builtin

    if isinstance(kind, builtin.IndexType):
        return dict(kind="index", spelling=str(kind))
    if isinstance(kind, builtin.MemRefType):
        shape = list(kind.get_shape())
        contiguous = isinstance(kind.layout, builtin.NoneAttr)
        default_space = isinstance(kind.memory_space, builtin.NoneAttr)
        strides, stride = [], 1
        for extent in reversed(shape):
            strides.insert(0, stride)
            stride *= max(0, extent)
        return dict(
            kind="view",
            spelling=str(kind),
            shape=shape,
            offset=0,
            strides=strides,
            element=str(kind.get_element_type()),
            space=0,
            contiguous=contiguous,
            default_space=default_space,
            actual_layout=str(kind.layout),
            actual_space=str(kind.memory_space),
            static=all(extent >= 0 for extent in shape),
        )
    if isinstance(kind, builtin.UnrankedMemRefType):
        return dict(kind="unsupported_view", spelling=str(kind))
    return dict(kind="scalar", spelling=str(kind))


def export(module):
    tree = operation_tree(module, calls="shape-direct")
    values, functions, calls = {}, [], []

    def visit(op, record):
        from xdsl.dialects import func

        if isinstance(op, func.FuncOp):
            functions.append(
                dict(
                    name=op.sym_name.data,
                    parameters=[type_record(kind) for kind in op.function_type.inputs.data],
                    returned=[type_record(kind) for kind in op.function_type.outputs.data],
                    declaration=not bool(op.body.blocks),
                )
            )
        if isinstance(op, func.CallOp):
            calls.append(
                dict(
                    operation=record,
                    target=op.callee.root_reference.data,
                    nested=bool(op.callee.nested_references.data),
                )
            )
        for value, item in zip(op.results, record["results"], strict=True):
            values[str(item[0])] = type_record(value.type)
        for region, records in zip(op.regions, record["regions"], strict=True):
            for block, block_record in zip(region.blocks, records, strict=True):
                for value, item in zip(block.args, block_record["arguments"], strict=True):
                    values[str(item[0])] = type_record(value.type)
                for child, child_record in zip(block.ops, block_record["operations"], strict=True):
                    visit(child, child_record)

    visit(module, tree)
    return dict(tree=tree, types=values, functions=functions, calls=calls)


def run(*, source, revision, before, output, after=None):
    sys.dont_write_bytecode = True
    from merlin.transforms import normalization
    from merlin.transforms.certificates import ir_text
    from merlin.transforms.certificates import parse_ir as producer_parse_ir

    def parse_ir(raw):
        return admission.parse_admitted(raw, producer_parse_ir, calls="shape-direct")

    observed = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if observed != revision:
        raise ValueError("original selected shape producer revision changed")
    original = before.read_bytes()
    module = parse_ir(original.decode())
    module.verify()
    first = export(module)
    if after is None:
        output.mkdir(parents=True, exist_ok=False)
        changes = normalization.normalize_shapes(module)
        module.verify()
        candidate = ir_text(module).encode()
        after = output / "after.mlir"
        after.write_bytes(candidate)
        second = export(module)
        destination = output / "CAPTURE.json"
        changed = len(changes)
    else:
        candidate = after.read_bytes()
        result = parse_ir(candidate.decode())
        result.verify()
        second = export(result)
        destination, changed = output, None
    provider = _producer_sources(source, revision, (Path(normalization.__file__).resolve(),))
    xdsl = loaded_xdsl_sources()
    result = dict(
        schema="merlin.graph.shape.actual_pair.v1",
        before=reference(before, original),
        after=reference(after, candidate),
        before_export=first,
        after_export=second,
        producer_revision=revision,
        producer_source=provider,
        xdsl_loaded_source=xdsl,
        changed_sites=changed,
        input_admission_source=admission.source_reference(),
        python_projection_trusted=True,
        runtime_descriptor_conformance_proved=False,
        native_or_hardware_proved=False,
    )
    if (
        original != before.read_bytes()
        or candidate != after.read_bytes()
        or provider != _producer_sources(source, revision, ())
        or xdsl != loaded_xdsl_sources()
    ):
        raise ValueError("actual pair or selected provider/parser source changed during export")
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "before", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--revision", required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("schema", "producer_revision", "changed_sites")}))


if __name__ == "__main__":
    main()
