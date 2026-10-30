"""Produce actual ordered SCF segmentation artifacts from pinned Merlin source.

The full operation/SSA tree is exported as data, including unrelated context.
Witnesses are retained for comparison, never treated as proof authority. Parsing,
export, xDSL runtime and source-to-model correspondence remain trusted Python.
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
else:
    from lean_graph_loops_capture import _producer_sources, _reference


def operation_tree(module, *, calls="external-only"):
    admission.admit_module(module, calls=calls)

    from xdsl.dialects import arith, builtin

    identities = {}

    def bind(value):
        if value in identities:
            raise admission.ModelRefusal("SSA definition appears more than once")
        index = len(identities)
        identities[value] = index
        return [index, str(value.type)]

    def operation(op):
        if op.successors:
            raise admission.ModelRefusal("successor-based control is outside this structural family")
        try:
            operands = [identities[value] for value in op.operands]
        except KeyError as error:
            raise admission.ModelRefusal("operation uses an unbound or forward SSA value") from error
        results = [bind(value) for value in op.results]
        regions = []
        for region in op.regions:
            blocks = []
            for block in region.blocks:
                arguments = [bind(value) for value in block.args]
                blocks.append({"arguments": arguments, "operations": [operation(item) for item in block.ops]})
            regions.append(blocks)
        constant = None
        if isinstance(op, arith.ConstantOp) and isinstance(op.result.type, builtin.IndexType):
            if not isinstance(op.value, builtin.IntegerAttr):
                raise admission.ModelRefusal("index constant requires an actual integer attribute")
            constant = op.value.value.data
        return {
            "name": op.name,
            "operands": operands,
            "results": results,
            "attributes": sorted([name, str(value)] for name, value in op.attributes.items()),
            "properties": sorted([name, str(value)] for name, value in op.properties.items()),
            "location": str(op.location),
            "regions": regions,
            "index_constant": constant,
        }

    return operation(module)


def loaded_xdsl_sources():
    paths = set()
    for name, module in tuple(sys.modules.items()):
        if name == "xdsl" or name.startswith("xdsl."):
            location = getattr(module, "__file__", None)
            if location is not None:
                paths.add(Path(location).resolve())
    return [_reference(path) for path in sorted(paths)]


def capture(*, source: Path, revision: str, before: Path, output: Path, tile: int, index_bits: int):
    sys.dont_write_bytecode = True
    from merlin.transforms.certificates import ir_text
    from merlin.transforms.certificates import parse_ir as producer_parse_ir

    def parse_ir(raw):
        return admission.parse_admitted(raw, producer_parse_ir)

    from merlin.transforms.loops import segment_loops

    source = source.resolve(strict=True)
    actual_revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if actual_revision != revision:
        raise ValueError("actual segmentation producer revision differs from the selected source")
    if type(tile) is not int or tile <= 1 or index_bits not in (32, 64):
        raise ValueError("select a changed static loop tile and explicit signed32/64 index domain")
    before = before.resolve(strict=True)
    original = before.read_bytes()
    module = parse_ir(original.decode())
    before_tree = operation_tree(module)
    canonical_before = ir_text(module)
    witnesses = segment_loops(module, (tile,), index_bits=index_bits)
    module.verify()
    after_tree = operation_tree(module)
    canonical_after = ir_text(module)
    closure = _producer_sources(source, revision, ())
    xdsl_sources = loaded_xdsl_sources()
    if not witnesses:
        raise ValueError("selected input produced no actual ordered segmentation change")
    if before.read_bytes() != original:
        raise ValueError("actual source IR changed during production capture")
    output.mkdir(parents=True, exist_ok=False)
    products = {
        "before.mlir": original.decode(),
        "before-generic.mlir": canonical_before,
        "after.mlir": canonical_after,
        "before-tree.json": json.dumps(before_tree, sort_keys=True, indent=2) + "\n",
        "after-tree.json": json.dumps(after_tree, sort_keys=True, indent=2) + "\n",
        "witnesses.json": json.dumps([asdict(witness) for witness in witnesses], indent=2) + "\n",
    }
    for name, text in products.items():
        (output / name).write_text(text)
    (output / "capture-implementation.py").write_bytes(Path(__file__).read_bytes())
    if closure != _producer_sources(source, revision, ()) or xdsl_sources != loaded_xdsl_sources():
        raise ValueError("actual producer/parser code changed during capture")
    result = {
        "schema": "merlin.graph.ordered_scf_segmentation.capture.v1",
        "producer_revision": revision,
        "producer_source": closure,
        "xdsl_loaded_source": xdsl_sources,
        "capture_implementation": _reference(output / "capture-implementation.py"),
        "input_admission_source": admission.source_reference(),
        "actual_before": _reference(before),
        "products": {name: _reference(output / name) for name in products},
        "tile": tile,
        "index_bits": index_bits,
        "changed_loops": len(witnesses),
        "python_projection_trusted": True,
        "lean_checked": False,
        "SCF_countedKernel_bridge_proved": False,
        "LeanMLIR_selected": False,
        "pipeline_receipt_issued": False,
        "numerical_gates_discharged": False,
        "renderer_or_hardware_proved": False,
    }
    (output / "CAPTURE.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "before", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    for name in ("tile", "index-bits"):
        parser.add_argument("--" + name, type=int, required=True)
    result = capture(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("producer_revision", "changed_loops", "tile", "lean_checked")}))


if __name__ == "__main__":
    main()
