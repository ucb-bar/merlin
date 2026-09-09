#!/usr/bin/env python3
"""Extract one exact batched partition boundary from the captured SmolVLA graph.

This worker runs under the model2MLIR Torch environment.  It reloads the saved
ExportedProgram, replaces every tensor in its state dictionary with the capture's
own safetensors value, binds the six capture inputs in their recorded order, and
requires the complete graph result to be bit-identical to ``golden.npy``.  Only
then does it retain the operands/result at the first exact p0102-shaped ATen
matmul.  The resulting arrays are inputs to the separate Atlas GSIM dispatcher;
they are not an end-to-end Atlas execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
DEFAULT_APPLICATION = REPO / "out/artifacts/applications/atlas/smolvla_denoise_step.pt2"
DEFAULT_CAPTURE = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent"
PARTITION_ID = "atlas_p0102"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_partition() -> dict:
    path = ROOT / "whole_capture_plan/partition_plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in plan["partitions"] if row.get("partition_id") == PARTITION_ID]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {PARTITION_ID} partition")
    partition = matches[0]
    if (
        partition.get("kind") != "matmul_batched"
        or partition.get("source_semantic") != "batch_matmul"
        or partition.get("capture_regions") != ["matmul_101"]
        or partition.get("geometry") != {"B": 15, "M": 113, "K": 64, "N": 113}
    ):
        raise ValueError("bound p0102 source identity or geometry changed")
    return partition


class _BoundaryReached(Exception):
    pass


class _BoundaryInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        target: torch.fx.Node,
        frontier: torch.fx.Node,
    ) -> None:
        super().__init__(module)
        self.target = target
        self.frontier = frontier
        self.boundary: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self._target_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def run_node(self, node: torch.fx.Node):
        if node is self.target:
            args, _kwargs = self.fetch_args_kwargs_from_env(node)
            if len(args) != 2 or not all(isinstance(value, torch.Tensor) for value in args):
                raise ValueError("bound ATen matmul no longer has two tensor operands")
            result = super().run_node(node)
            if not isinstance(result, torch.Tensor):
                raise ValueError("bound ATen matmul no longer has one tensor result")
            self._target_values = tuple(
                value.detach().cpu().clone() for value in (*args, result)
            )
            return result
        if node is self.frontier:
            result = super().run_node(node)
            if self._target_values is None or not isinstance(result, torch.Tensor):
                raise ValueError("bound p0102 frontier executed without its tensor source")
            self.boundary = (*self._target_values, result.detach().cpu().clone())
            raise _BoundaryReached
        return super().run_node(node)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", type=Path, default=DEFAULT_APPLICATION)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "capture_semantics_text_layer0_attn_qk",
    )
    args = parser.parse_args()
    application = args.application.resolve()
    capture = args.capture.resolve()
    out = args.out.resolve()
    partition = _load_partition()

    input_order = json.loads((capture / "input_order.json").read_text(encoding="utf-8"))
    expected_input_order = {
        "img": 0,
        "img_mask": 1,
        "lang_tokens": 2,
        "lang_masks": 3,
        "state": 4,
        "noise": 5,
    }
    if input_order != expected_input_order:
        raise ValueError("capture input order differs from the exported program binding")

    exported = torch.export.load(application)
    module = exported.module()
    capture_state = load_file(capture / "weights.safetensors", device="cpu")
    module_state = module.state_dict()
    if set(module_state) != set(capture_state):
        raise ValueError("ExportedProgram and capture state dictionaries have different keys")
    for name, value in capture_state.items():
        if module_state[name].shape != value.shape or module_state[name].dtype != value.dtype:
            raise ValueError(f"capture state ABI differs at {name!r}")
    module.load_state_dict(capture_state, strict=True)

    with np.load(capture / "inputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != {f"in{index}" for index in range(6)}:
            raise ValueError("capture inputs archive no longer contains exactly in0..in5")
        input_arrays = [np.ascontiguousarray(archive[f"in{index}"]).copy() for index in range(6)]
    inputs = [torch.from_numpy(value) for value in input_arrays]

    geometry = partition["geometry"]
    expected_shapes = (
        (1, geometry["B"], geometry["M"], geometry["K"]),
        (1, geometry["B"], geometry["K"], geometry["N"]),
        (1, geometry["B"], geometry["M"], geometry["N"]),
    )
    matches = []
    for node in module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.matmul.default:
            continue
        value = node.meta.get("val")
        operand_shapes = tuple(
            tuple(arg.meta["val"].shape)
            for arg in node.args
            if isinstance(arg, torch.fx.Node) and "val" in arg.meta
        )
        if operand_shapes == expected_shapes[:2] and tuple(value.shape) == expected_shapes[2]:
            matches.append(node)
    if not matches or matches[0].name != "matmul":
        raise ValueError("first exact-shape ATen matmul no longer binds captured matmul_101")
    target = matches[0]
    users = list(target.users)
    if len(users) != 1:
        raise ValueError("captured p0102 matmul no longer has one immediate graph consumer")
    frontier = users[0]
    if (
        frontier.op != "call_function"
        or frontier.target != torch.ops.aten.mul.Tensor
        or len(frontier.args) != 2
        or frontier.args[0] is not target
        or frontier.args[1] != 0.125
        or tuple(frontier.meta["val"].shape) != expected_shapes[2]
        or frontier.meta["val"].dtype != torch.float32
    ):
        raise ValueError("captured p0102 frontier is no longer the exact scalar multiply by 0.125")
    output_abi = partition["abi"]["outputs"]
    if (
        len(output_abi) != 1
        or output_abi[0].get("frontier_consumers")
        != [{
            "kind": "host_region",
            "op": "linalg.generic",
            "op_index": 3063,
            "region_id": "mul_32",
            "semantic": "mul",
        }]
    ):
        raise ValueError("partition plan no longer binds p0102 to captured mul_32 frontier")

    with torch.no_grad():
        complete = module(*inputs)
    if not isinstance(complete, torch.Tensor):
        raise ValueError("captured graph no longer has one tensor result")
    complete_f32 = np.ascontiguousarray(complete.detach().float().cpu().numpy(), dtype="<f4")
    golden = np.ascontiguousarray(np.load(capture / "golden.npy"), dtype="<f4")
    if not np.array_equal(complete_f32, golden):
        raise ValueError("ExportedProgram with capture state/inputs is not bit-identical to golden.npy")

    interpreter = _BoundaryInterpreter(module, target, frontier)
    try:
        with torch.no_grad():
            interpreter.run(*inputs)
    except _BoundaryReached:
        pass
    if interpreter.boundary is None:
        raise ValueError("bound capture boundary did not execute")
    activation_t, weight_t, source_output_t, frontier_output_t = interpreter.boundary
    actual_shapes = tuple(tuple(value.shape) for value in interpreter.boundary)
    actual_dtypes = tuple(value.dtype for value in interpreter.boundary)
    if actual_shapes != (*expected_shapes, expected_shapes[2]) or actual_dtypes != (torch.float32,) * 4:
        raise ValueError("captured p0102 boundary tensor ABI changed")

    activation = np.ascontiguousarray(activation_t.numpy()[0], dtype="<f4")
    weight = np.ascontiguousarray(weight_t.numpy()[0], dtype="<f4")
    source_output = np.ascontiguousarray(source_output_t.numpy()[0], dtype="<f4")
    frontier_output = np.ascontiguousarray(frontier_output_t.numpy()[0], dtype="<f4")
    expected_frontier = np.ascontiguousarray(source_output * np.float32(0.125), dtype="<f4")
    if not np.array_equal(frontier_output, expected_frontier):
        raise ValueError("captured p0102 frontier is not bit-exact source matmul times 0.125")
    independent_source = np.matmul(activation, weight, dtype=np.float32)
    independent_frontier = independent_source * np.float32(0.125)
    source_oracle_max_abs = float(np.max(np.abs(independent_source - source_output)))
    frontier_oracle_max_abs = float(np.max(np.abs(independent_frontier - frontier_output)))
    if source_oracle_max_abs > 2.0e-6 or frontier_oracle_max_abs > 2.5e-7:
        raise ValueError("independent NumPy p0102 oracle differs from the captured FX boundary")
    out.mkdir(parents=True, exist_ok=True)
    operands_path = out / "capture_operands.npz"
    np.savez(
        operands_path,
        A0=activation,
        W=weight,
        Y_source=source_output,
        Y_frontier=frontier_output,
    )
    receipt = {
        "schema": "atlas_real_capture_batched_boundary_v1",
        "claim": (
            "exact p0102 operands/result tapped from the capture-equivalent ATen graph after "
            "strict capture state/input binding and a bit-exact complete-graph golden check"
        ),
        "not_claimed": [
            "Atlas execution",
            "whole-model Atlas correctness",
            "whole-model Atlas performance",
        ],
        "partition_id": PARTITION_ID,
        "capture_regions": partition["capture_regions"],
        "geometry": geometry,
        "source_operator": {
            "kind": "torch.ops.aten.matmul.default",
            "fx_node": target.name,
            "exact_shape_occurrence": 0,
            "exact_shape_occurrence_count": len(matches),
            "input_shapes_with_export_batch": [list(shape) for shape in expected_shapes[:2]],
            "output_shape_with_export_batch": list(expected_shapes[2]),
        },
        "source_frontier": {
            "kind": "torch.ops.aten.mul.Tensor",
            "fx_node": frontier.name,
            "scalar": 0.125,
            "scalar_hex": float(0.125).hex(),
            "partition_plan_region": "mul_32",
            "partition_plan_op_index": 3063,
            "sole_immediate_consumer": True,
            "fold": "Y_frontier = Y_source * 0.125",
        },
        "independent_oracle": {
            "implementation": "numpy.matmul(dtype=float32)",
            "derived_from_rtl": False,
            "consumes_gsim_output": False,
            "source_matmul_max_abs_error": source_oracle_max_abs,
            "source_matmul_max_abs_limit": 2.0e-6,
            "frontier_max_abs_error": frontier_oracle_max_abs,
            "frontier_max_abs_limit": 2.5e-7,
        },
        "binding": {
            "application": application.relative_to(REPO).as_posix(),
            "application_sha256": _sha256(application),
            "capture_model": (capture / "model.mlir").relative_to(REPO).as_posix(),
            "capture_model_sha256": _sha256(capture / "model.mlir"),
            "capture_weights": (capture / "weights.safetensors").relative_to(REPO).as_posix(),
            "capture_weights_sha256": _sha256(capture / "weights.safetensors"),
            "state_tensor_count": len(capture_state),
            "capture_inputs": (capture / "inputs.npz").relative_to(REPO).as_posix(),
            "capture_inputs_sha256": _sha256(capture / "inputs.npz"),
            "input_order": input_order,
            "complete_graph_output_bit_exact": True,
            "complete_graph_output_sha256": _array_sha256(complete_f32),
            "capture_golden_sha256": _array_sha256(golden),
        },
        "operands": {
            "path": operands_path.relative_to(ROOT).as_posix(),
            "archive_sha256": _sha256(operands_path),
            "A0": {"shape": list(activation.shape), "dtype": "f32", "raw_sha256": _array_sha256(activation)},
            "W": {"shape": list(weight.shape), "dtype": "f32", "raw_sha256": _array_sha256(weight)},
            "Y_source": {
                "shape": list(source_output.shape),
                "dtype": "f32",
                "raw_sha256": _array_sha256(source_output),
            },
            "Y_frontier": {
                "shape": list(frontier_output.shape),
                "dtype": "f32",
                "raw_sha256": _array_sha256(frontier_output),
            },
        },
    }
    (out / "capture_boundary.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
