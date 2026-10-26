"""Freeze integer-domain bias from an exported integer-contraction program.

This numerical preparation recognizes operation/dataflow patterns, never model
names. It preserves frozen weights/scales and records a new source identity.
Only literal integral bias codes with a proved INT32 bound are admitted. Any
change versus the former FP32 ordering remains explicit in the release report.
"""

import argparse
from pathlib import Path

from prepare_firesim_chunks import identity, write_json


def normalize(module):
    import torch

    graph = module.graph
    constants = {}

    def constant(node):
        if not isinstance(node, torch.fx.Node):
            return node
        if node in constants:
            return constants[node]
        if node.op == "get_attr":
            value = module
            for part in node.target.split("."):
                value = getattr(value, part)
        elif node.op == "call_function":
            args = torch.fx.node.map_arg(node.args, constant)
            kwargs = torch.fx.node.map_arg(node.kwargs, constant)
            value = node.target(*args, **kwargs)
        else:
            raise ValueError("Bias/weight must be frozen source constants")
        constants[node] = value
        return value

    views = {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
    }

    def chain(node):
        if not isinstance(node, torch.fx.Node) or node.op != "call_function":
            return None
        to_float32 = (node.target == torch.ops.aten._to_copy.default and node.kwargs.get("dtype") == torch.float32) or (
            node.target == torch.ops.aten.to.dtype and node.args[1] == torch.float32
        )
        if to_float32:
            raw = node.args[0]
            if isinstance(raw, torch.fx.Node) and raw.target == torch.ops.aten._int_mm.default:
                return raw, []
        if node.target in views:
            previous = chain(node.args[0])
            return (previous[0], previous[1] + [node]) if previous else None
        if node.target == torch.ops.aten.mul.Tensor and node.args[1] == 1.0:
            return chain(node.args[0])
        return None

    records = []
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.add.Tensor or node.kwargs.get("alpha", 1) != 1:
            continue
        candidate = next(((chain(a), b) for a, b in (node.args[:2], tuple(reversed(node.args[:2]))) if chain(a)), None)
        if candidate is None:
            continue
        (raw, path), bias_node = candidate
        bias = constant(bias_node).detach().cpu()
        weight = constant(raw.args[1]).detach().cpu()
        if bias.dtype != torch.float32 or not torch.isfinite(bias).all() or not torch.equal(bias, bias.round()):
            raise ValueError("Require finite integral prepared bias codes; do not invent new calibration")
        limit = torch.iinfo(torch.int32)
        if bias.double().min() < limit.min or bias.double().max() > limit.max:
            raise ValueError("Prepared bias code does not fit INT32")
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("Require frozen rank-two INT8 contraction weights")
        integer_bias = bias.to(torch.int32)
        product_bound = int(weight.long().abs().sum(0).max()) * 128
        combined_bound = product_bound + int(integer_bias.long().abs().max())
        if combined_bound > limit.max:
            raise ValueError("INT32 contraction plus bias could overflow on admitted INT8 inputs")
        name = f"_prepared_accumulator_bias_{len(records)}"
        module.register_buffer(name, integer_bias)
        with graph.inserting_before(node):
            value = raw
            for view in path:
                arguments = (value, *view.args[1:])
                value = graph.call_function(view.target, arguments, dict(view.kwargs))
            stored = graph.get_attr(name)
            added = graph.call_function(torch.ops.aten.add.Tensor, (value, stored))
            result = graph.call_function(torch.ops.aten._to_copy.default, (added,), {"dtype": torch.float32})
        node.replace_all_uses_with(result)
        records.append(
            dict(
                source_bias_add=node.name,
                raw_contraction=raw.name,
                bias_buffer=name,
                product_abs_bound=product_bound,
                biased_abs_bound=combined_bound,
                accumulator_dtype="int32",
                bias_dtype="int32",
                original_fp32_order_proved_exact=combined_bound <= 2**24
                and not bool(((bias == 0) & torch.signbit(bias)).any()),
                note="A new prepared identity is required even when the old FP32 ordering is proved exact",
            )
        )
    if not records:
        raise ValueError("No integral FP32-bias contraction pattern found")
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return module, records


def main():
    import numpy as np
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- register Q/DQ ops

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "input-fixture", "output-dir", "policy"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    original = torch.export.load(args.program)
    module, records = normalize(original.module())
    with np.load(args.input_fixture, allow_pickle=False) as data:
        inputs = tuple(torch.from_numpy(data[f"input_{i}"].copy()) for i in range(len(original.example_inputs[0])))
    prepared = torch.export.export(module, inputs, strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.export.save(prepared, args.output_dir / "program.pt2")
    write_json(
        args.output_dir / "numerical-preparation.json",
        dict(
            schema="prepared_integer_accumulator_bias_v1",
            parent=identity(args.program),
            prepared=identity(args.output_dir / "program.pt2"),
            policy=identity(args.policy),
            preparation_source=identity(__file__),
            transformations=records,
            model_quality="NOT_RUN",
            hardware_execution="NOT_RUN",
            reference_policy="Generate independent references from this source; no inherited admission",
        ),
    )


if __name__ == "__main__":
    main()
