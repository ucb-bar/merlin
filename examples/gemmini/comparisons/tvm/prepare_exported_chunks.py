"""Freeze real, full-shape FX regions as semantic TVM tuning inputs.

Region names/node selections are experiment inputs, never compiler decisions.
Constants and all live edges are preserved. References execute source FX with
independent INT64 NumPy integer contractions, without TVM-generated kernels.
Numerical-policy admission and model-quality gates remain separate.
"""

import argparse
import json
import operator
from pathlib import Path

import numpy as np
from prepare_firesim_chunks import identity, write_json


def slice_graph(module, start_node, stop_after):
    import torch

    nodes = list(module.graph.nodes)
    names = {node.name: node for node in nodes}
    start, stop = nodes.index(names[start_node]), nodes.index(names[stop_after]) + 1
    if not 0 <= start < stop <= len(nodes):
        raise ValueError("Require an ordered, nonempty source interval")
    selected = nodes[start:stop]
    selected_set = set(selected)
    for node in selected:
        if node.op != "call_function":
            raise ValueError("Select an interval of functional exported operations")
        schema = getattr(node.target, "_schema", None)
        if schema is not None and schema.is_mutable:
            raise ValueError("Mutable operations are not admitted into a tuning chunk")
        if schema is None and node.target not in (operator.getitem, operator.add, operator.mul):
            raise ValueError("Opaque source call is not admitted into a tuning chunk")
    outputs = [node for node in selected if any(user not in selected_set for user in node.users)]
    if not outputs:
        raise ValueError("Region has no live output")
    graph, mapping, inputs = torch.fx.Graph(), {}, []

    def argument(node):
        if node not in mapping:
            if node.op == "get_attr":
                mapping[node] = graph.node_copy(node, argument)
            else:
                mapping[node] = graph.placeholder(f"input_{len(inputs)}")
                inputs.append(node)
        return mapping[node]

    for node in selected:
        mapping[node] = graph.node_copy(node, argument)
    graph.output(tuple(mapping[node] for node in outputs))
    graph.lint()
    return torch.fx.GraphModule(module, graph), inputs, outputs


def capture_graph(module, requested):
    """Clone only the necessary prefix/dependencies, returning named boundaries."""
    import torch

    graph, mapping = torch.fx.Graph(), {}

    def copy(node):
        if node not in mapping:
            mapping[node] = graph.node_copy(node, copy)
        return mapping[node]

    for node in module.graph.nodes:
        if node.op == "placeholder":
            copy(node)
    graph.output(tuple(copy(node) for node in requested))
    graph.lint()
    return torch.fx.GraphModule(module, graph)


def independent_values(module, inputs):
    import torch

    class IntegerReference(torch.fx.Interpreter):
        def call_function(self, target, args, kwargs):
            if target == torch.ops.aten._int_mm.default:
                a, b = (value.detach().cpu().numpy() for value in args)
                if a.dtype != np.int8 or b.dtype != np.int8 or a.ndim != 2 or b.ndim != 2:
                    raise ValueError("Reference _int_mm requires rank-two signed INT8")
                if a.shape[1] != b.shape[0] or a.shape[1] * 128 * 128 > np.iinfo(np.int64).max:
                    raise ValueError("Reference contraction dimensions/bounds are invalid")
                result = a.astype(np.int64) @ b.astype(np.int64)
                if result.min() < np.iinfo(np.int32).min or result.max() > np.iinfo(np.int32).max:
                    raise ValueError("Reference contraction exceeds INT32")
                return torch.from_numpy(result.astype(np.int32))
            return super().call_function(target, args, kwargs)

    with torch.no_grad():
        return IntegerReference(module).run(*inputs)


def compare(actual, expected):
    import torch

    rows = []
    for observed, reference in zip(actual, expected, strict=True):
        a = observed.detach().cpu().numpy() if isinstance(observed, torch.Tensor) else observed.numpy()
        b = reference.detach().cpu().numpy()
        if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all():
            raise ValueError("Chunk output shape/dtype/finiteness differs from reference")
        if a.dtype.kind in "iu":
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=2e-6)
        rows.append(
            dict(shape=list(a.shape), dtype=a.dtype.name, max_abs_error=float(np.max(np.abs(a.astype(np.float64) - b))))
        )
    return rows


def prepare(args):
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- register serialized Q/DQ operators
    import tvm
    from program_io import import_program

    exported = torch.export.load(args.program)
    parent = exported.module()
    with np.load(args.input_fixture, allow_pickle=False) as archive:
        inputs = tuple(torch.from_numpy(archive[f"input_{i}"].copy()) for i in range(len(exported.example_inputs[0])))
    regions = json.loads(args.regions.read_text())
    if not regions or len({r["name"] for r in regions}) != len(regions):
        raise ValueError("Require distinct nonempty region names")
    prepared = []
    requested = []
    for region in regions:
        if not region["name"] or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in region["name"]):
            raise ValueError("Region name must be a safe lowercase directory name")
        chunk, incoming, outgoing = slice_graph(parent, region["start_node"], region["stop_after"])
        prepared.append((region, chunk, incoming, outgoing))
        for node in incoming + outgoing:
            if node not in requested:
                requested.append(node)
    values = independent_values(capture_graph(parent, requested), inputs)
    boundaries = dict(zip(requested, values, strict=True))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [identity(path) for path in (args.program, args.input_fixture, args.regions, args.policy, __file__)]
    rows = []
    for region, chunk, incoming, outgoing in prepared:
        folder = args.output_dir / region["name"]
        folder.mkdir()
        arguments = tuple(boundaries[node] for node in incoming)
        expected = tuple(boundaries[node] for node in outgoing)
        with torch.no_grad():
            source_check = compare(chunk(*arguments), expected)
        release = torch.export.export(chunk, arguments, strict=True)
        torch.export.save(release, folder / "program.pt2")
        mod = import_program(folder / "program.pt2")
        (folder / "program.json").write_text(tvm.ir.save_json(mod))
        arrays = {f"input_{i}": tensor.detach().cpu().numpy() for i, tensor in enumerate(arguments)}
        arrays.update(
            {
                "expected" if len(expected) == 1 else f"expected_{i}": tensor.detach().cpu().numpy()
                for i, tensor in enumerate(expected)
            }
        )
        np.savez(folder / "fixture.npz", **arrays)
        proof = dict(
            program=identity(folder / "program.json"),
            fixture_sha256=identity(folder / "fixture.npz")["sha256"],
            reference_sources=sources,
            reference_contract="Source FX ops; integer GEMMs use independent checked NumPy INT64 accumulation",
            comparison=dict(rtol=1e-5, atol=2e-6),
            policy=identity(args.policy),
            numerical_policy_admission="REVIEW_REQUIRED",
            model_quality="NOT_RUN",
            source_crosscheck=source_check,
        )
        write_json(folder / "fixture-provenance.json", proof)
        row = dict(
            **region,
            live_inputs=[n.name for n in incoming],
            live_outputs=[n.name for n in outgoing],
            shape_policy="Exact source dimensions, constants, operations and all live edges; no cropping",
            program=identity(folder / "program.json"),
            fixture=identity(folder / "fixture.npz"),
            numerical_policy_admission="REVIEW_REQUIRED",
            firesim="NOT_RUN",
            elf="NOT_BUILT",
        )
        write_json(folder / "chunk.json", row)
        rows.append(row)
    write_json(
        args.output_dir / "chunks.json", dict(schema="exported_full_shape_chunks_v1", sources=sources, chunks=rows)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "input-fixture", "regions", "policy", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
