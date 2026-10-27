"""Freeze source regions with captured live-ins and independent chunk outputs.

Boundary capture uses the prepared source program, not TVM. Only the small
regions use the independent INT64 contraction oracle. Optional row profiles
are restricted to row-local MLPs; attention/cache shapes cannot be rewritten.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from prepare_exported_chunks import capture_graph, compare, independent_values, slice_graph
from prepare_firesim_chunks import identity, write_json


def row_profile(chunk, inputs, source_rows, rows):
    """Specialize a proved row-local [1,M,C] MLP, preserving every feature."""

    if not 0 < rows <= source_rows:
        raise ValueError("Row profile must be a nonempty prefix of source rows")
    allowed = {
        "aten.pow.Tensor_Scalar",
        "aten.mean.dim",
        "aten.add.Tensor",
        "aten.mul.Tensor",
        "aten.rsqrt.default",
        "aten._assert_tensor_metadata.default",
        "aten.view.default",
        "aten.permute.default",
        "aten._int_mm.default",
        "aten._to_copy.default",
        "aten.sigmoid.default",
        "quantized_decomposed.quantize_per_tensor.default",
    }
    result = copy.deepcopy(chunk)
    dependent = set()
    for node in result.graph.nodes:
        if node.op not in ("placeholder", "get_attr", "call_function", "output"):
            raise ValueError("Only functional source regions may be row-profiled")
        if node.op == "placeholder":
            dependent.add(node)
        elif node.op == "call_function":
            name = str(node.target)
            if name not in allowed:
                raise ValueError("Not a proved row-local operation: " + name)
            live = any(n in dependent for n in node.all_input_nodes)
            if name == "aten.mean.dim" and (list(node.args[1]) != [-1] or not node.args[2]):
                raise ValueError("Only last-feature-axis norm reductions are row-local")
            if name == "aten.permute.default" and live:
                raise ValueError("Only constant weight permutations are allowed")
            if name == "aten._int_mm.default" and node.args[1] in dependent:
                raise ValueError("Right-hand GEMM operand must be frozen")
            if live:
                dependent.add(node)
            if name == "aten.view.default" and live:
                shape = list(node.args[1])
                if len(shape) == 3 and shape[:2] == [1, source_rows]:
                    shape[1] = rows
                elif len(shape) == 2 and shape[0] in (-1, source_rows):
                    if shape[0] == source_rows:
                        shape[0] = rows
                else:
                    raise ValueError("View does not preserve independent rows: " + str(shape))
                node.args = (node.args[0], shape)
    arguments = []
    for value in inputs:
        if value.ndim != 3 or tuple(value.shape[:2]) != (1, source_rows):
            raise ValueError("Each live-in must be an explicit [1,M,C] row tensor")
        arguments.append(value[:, :rows, :].contiguous())
    result.graph.lint()
    result.recompile()
    return result, tuple(arguments)


def prepare(args):
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401
    import tvm
    from program_io import import_program

    sys.setrecursionlimit(30000)
    torch.set_num_threads(args.threads)
    print("loading prepared parent", flush=True)
    exported = torch.export.load(args.program)
    parent = exported.module()
    with np.load(args.input_fixture, allow_pickle=False) as archive:
        inputs = tuple(torch.from_numpy(archive[f"in{i}"].copy()) for i in range(len(exported.example_inputs[0])))
    regions = json.loads(args.regions.read_text())
    if not regions or len({r["name"] for r in regions}) != len(regions):
        raise ValueError("Require distinct nonempty region names")
    prepared, requested = [], []
    for region in regions:
        if not region["name"] or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in region["name"]):
            raise ValueError("Region name must be a safe lowercase directory name")
        chunk, incoming, outgoing = slice_graph(parent, region["start_node"], region["stop_after"])
        prepared.append((region, chunk, incoming, outgoing))
        for node in incoming + outgoing:
            if node not in requested:
                requested.append(node)
    print("capturing source boundaries", flush=True)
    with torch.no_grad():
        values = capture_graph(parent, requested)(*inputs)
    boundaries = dict(zip(requested, values, strict=True))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [identity(p) for p in (args.program, args.input_fixture, args.regions, args.policy, __file__)]
    rows = []
    for region, chunk, incoming, outgoing in prepared:
        print("preparing " + region["name"], flush=True)
        arguments = tuple(boundaries[node] for node in incoming)
        expected = tuple(boundaries[node] for node in outgoing)
        with torch.no_grad():
            source_check = compare(chunk(*arguments), expected)
        profile = region.get("row_profile")
        if profile:
            chunk, arguments = row_profile(chunk, arguments, profile["source_rows"], profile["rows"])
            expected = tuple(value[:, : profile["rows"], :].contiguous() for value in expected)
        reference = independent_values(chunk, arguments)
        oracle_check = compare(reference, expected)
        folder = args.output_dir / region["name"]
        folder.mkdir()
        release = torch.export.export(chunk, arguments, strict=True)
        torch.export.save(release, folder / "program.pt2")
        mod = import_program(folder / "program.pt2")
        (folder / "program.json").write_text(tvm.ir.save_json(mod))
        arrays = {f"input_{i}": v.detach().cpu().numpy() for i, v in enumerate(arguments)}
        arrays.update(
            {
                "expected" if len(reference) == 1 else f"expected_{i}": v.detach().cpu().numpy()
                for i, v in enumerate(reference)
            }
        )
        np.savez(folder / "fixture.npz", **arrays)
        proof = dict(
            program=identity(folder / "program.json"),
            fixture_sha256=identity(folder / "fixture.npz")["sha256"],
            reference_sources=sources,
            reference_contract=(
                "Captured source FX live-ins; independent NumPy INT64 GEMMs inside chunk; source host ops"
            ),
            comparison=dict(rtol=1e-5, atol=2e-6),
            policy=identity(args.policy),
            numerical_policy_admission="CHUNK_REFERENCE_CHECKED_E2E_QUALITY_NOT_RUN",
            model_quality="NOT_RUN",
            source_crosscheck=source_check,
            independent_crosscheck=oracle_check,
        )
        write_json(folder / "fixture-provenance.json", proof)
        row = dict(
            **region,
            live_inputs=[n.name for n in incoming],
            live_outputs=[n.name for n in outgoing],
            program=identity(folder / "program.json"),
            fixture=identity(folder / "fixture.npz"),
            firesim="NOT_RUN",
            elf="NOT_BUILT",
            reference_check=oracle_check,
        )
        write_json(folder / "chunk.json", row)
        rows.append(row)
        print("ready " + region["name"], flush=True)
    write_json(
        args.output_dir / "chunks.json", dict(schema="source_bound_shared_chunks_v1", sources=sources, chunks=rows)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "input-fixture", "regions", "policy", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
