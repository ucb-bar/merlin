"""Extract executable PyTorch regions from the real exported Qwen session.

Validate the exported session against eager PyTorch, then validate each region
on its actual graph-boundary tensors. Generated tensors, Python modules and
MLIR are written only to the ignored output directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.fx import Graph, GraphModule, Interpreter
from torch.utils._pytree import tree_flatten
from safetensors.torch import save_file

from loader import get_model_and_inputs

ROOT = Path(__file__).resolve().parents[5]


def cut(graph, prefix, step=0):
    current = -1
    selected = []
    for node in graph.nodes:
        if node.op == "call_function" and str(node.target) == "aten.embedding.default":
            current += 1
        stack = node.meta.get("nn_module_stack", {})
        module = list(stack.values())[-1][0] if stack else ""
        if (
            node.op == "call_function"
            and current == step
            and (module == prefix or module.startswith(prefix + "."))
        ):
            selected.append(node)
    if not selected:
        raise RuntimeError(f"No exported operations found for {prefix}, step {step}")
    chosen = set(selected)
    boundaries = []
    for node in selected:
        for source in node.all_input_nodes:
            if source not in chosen and source not in boundaries:
                boundaries.append(source)
    outputs = [
        node for node in selected if any(user not in chosen for user in node.users)
    ]
    if not outputs:
        raise RuntimeError(f"Region {prefix} has no observable output")
    extracted = Graph()
    env = {node: extracted.placeholder(node.name) for node in boundaries}
    for node in selected:
        env[node] = extracted.node_copy(node, lambda source: env[source])
    extracted.output(tuple(env[node] for node in outputs))
    extracted.lint()
    return GraphModule(torch.nn.Module(), extracted), boundaries, outputs, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    model, inputs = get_model_and_inputs()
    program = torch.export.export(model, inputs, strict=False)
    definitions = {
        "first_layer_norm": "decoder.model.layers.0.input_layernorm",
        "first_layer_attention": "decoder.model.layers.0.self_attn",
        "first_layer_mlp": "decoder.model.layers.0.mlp",
        "first_gate_projection": "decoder.model.layers.0.mlp.gate_proj",
        "first_down_projection": "decoder.model.layers.0.mlp.down_proj",
    }
    regions = {
        name: cut(program.graph_module.graph, prefix)
        for name, prefix in definitions.items()
    }
    needed = {
        node
        for _, boundaries, outputs, _ in regions.values()
        for node in boundaries + outputs
    }
    observed = {}

    class RecordingInterpreter(Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if node in needed:
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"Non-tensor region boundary: {node.name}")
                observed[node] = value.detach().clone()
            return value

    named = dict(program.state_dict)
    named.update(program.constants)
    flattened = []
    source_parameters = {}
    users = iter(inputs)
    for spec in program.graph_signature.input_specs:
        if spec.kind.name in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
            flattened.append(named[spec.target])
            source_parameters[spec.arg.name] = spec.target
        elif spec.kind.name == "USER_INPUT":
            flattened.append(next(users))
        else:
            raise RuntimeError(f"Unsupported exported input kind: {spec.kind}")
    with torch.no_grad():
        eager = model(*inputs)
        exported = RecordingInterpreter(program.graph_module).run(*flattened)
    expected_leaves, _ = tree_flatten(eager)
    actual_leaves, _ = tree_flatten(exported)
    if len(expected_leaves) != len(actual_leaves) or not all(
        torch.equal(a, b) for a, b in zip(actual_leaves, expected_leaves, strict=True)
    ):
        raise RuntimeError("Exported full-model outputs differ from eager PyTorch")
    save_file(
        {
            "input_ids": inputs[0],
            "generated_tokens": eager[0],
            "last_token_logits": eager[1],
        },
        str(args.out / "session.safetensors"),
    )
    graph_hash = hashlib.sha256(program.graph_module.code.encode()).hexdigest()
    rows = []
    for name, (kernel, boundaries, outputs, selected) in regions.items():
        dest = args.out / name
        dest.mkdir()
        arguments = tuple(observed[node] for node in boundaries)
        expected = tuple(observed[node] for node in outputs)
        with torch.no_grad():
            actual = kernel(*arguments)
        if not all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)):
            raise RuntimeError(f"Extracted PyTorch region differs: {name}")
        kernel.to_folder(str(dest / "pytorch"), "ExtractedKernel")
        save_file(
            {
                node.name: value.contiguous()
                for node, value in zip(boundaries, arguments)
            },
            str(dest / "inputs.safetensors"),
        )
        save_file(
            {str(i): value.contiguous() for i, value in enumerate(expected)},
            str(dest / "golden.safetensors"),
        )
        row = dict(
            name=name,
            module=definitions[name],
            step=0,
            source_graph_sha256=graph_hash,
            source_nodes=[node.name for node in selected],
            inputs=[
                dict(
                    name=node.name,
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    parameter=source_parameters.get(node.name),
                )
                for node, value in zip(boundaries, arguments)
            ],
            outputs=[
                dict(node=node.name, shape=list(value.shape), dtype=str(value.dtype))
                for node, value in zip(outputs, expected)
            ],
            pytorch_exact=True,
        )
        # Linalg capture is diagnostic: a refusal is retained, never replaced
        # by a hand-written implementation of the region.
        sys.path.insert(0, str(ROOT / "python"))
        from atlas_mlir.capture import capture

        try:
            captured = capture(kernel, arguments, dest / "capture")
            row["linalg"] = str(captured.linalg.relative_to(args.out))
            row["capture_status"] = "captured"
        except Exception as error:
            row.update(
                capture_status="refused",
                capture_error=f"{type(error).__name__}: {error}",
            )
            raw = dest / "capture/raw-linalg.mlir"
            if raw.is_file():
                row["raw_linalg"] = str(raw.relative_to(args.out))
        (dest / "region.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(
            f"{name}: {len(selected)} source operations; PyTorch exact; {row['capture_status']}",
            flush=True,
        )
    result = dict(
        full_export_matches_eager=True,
        model_layers=28,
        generated_tokens=eager[0].tolist(),
        regions=rows,
    )
    (args.out / "capture.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
