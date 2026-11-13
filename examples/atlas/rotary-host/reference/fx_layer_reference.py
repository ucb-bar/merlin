"""Replay saved source FX with explicitly selected independent Atlas arithmetic.

Dispatch is driven by original ATen/custom-op schemas, dtype metadata, shapes,
and scalar arguments. No model2MLIR graph, semantic JSON, emitted program,
compiler evaluator, or ordinary framework numerical kernel is read or called.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

import layer_reference as primitives


@dataclass(frozen=True)
class Tensor:
    raw: np.ndarray
    dtype: str


def raw_tensor(tensor):
    import torch
    if tensor.dtype == torch.bfloat16:
        return Tensor(tensor.detach().contiguous().view(torch.uint16).numpy().copy(), "bf16")
    if tensor.dtype == torch.float32:
        return Tensor(tensor.detach().contiguous().view(torch.uint32).numpy().copy(), "f32")
    if tensor.dtype == torch.uint8:
        return Tensor(tensor.detach().contiguous().numpy().copy(), "u8")
    if tensor.dtype == torch.int32:
        return Tensor(tensor.detach().contiguous().numpy().copy(), "i32")
    raise ValueError("unsupported original FX tensor dtype")


def as_f32(value):
    if isinstance(value, Tensor):
        if value.dtype == "f32":
            return value.raw
        if value.dtype == "bf16":
            return primitives.widen(value.raw)
        raise ValueError("source arithmetic operand is not floating data")
    if type(value) not in {int, float}:
        raise ValueError("source scalar argument is not a declared number")
    return np.asarray(value, np.float32).view(np.uint32)


class Interpreter:
    def __init__(self, program, policy, *, engine="numpy", threads=1):
        for field, expected in primitives.POLICIES.items():
            if not primitives.selected_policy_matches(field, policy.get(field)):
                raise ValueError("original FX selected policy missing or changed: " + field)
        self.program, self.policy = program, policy
        self.engine, self.threads = engine, threads
        self.env, self.saved, self.extra, self.mask_additions = {}, {}, {}, {}
        self.specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
        self.initial_state = {name: raw_tensor(value) for name, value in program.state_dict.items()}
        self.dtype_names = None

    def resolve(self, value):
        import torch
        if isinstance(value, torch.fx.Node):
            return self.env[value]
        if isinstance(value, (list, tuple)):
            return type(value)(self.resolve(item) for item in value)
        if isinstance(value, dict):
            return {key: self.resolve(item) for key, item in value.items()}
        return value

    def dtype(self, node):
        import torch
        value = node.meta.get("val")
        names = {torch.bfloat16: "bf16", torch.float32: "f32", torch.uint8: "u8", torch.int32: "i32"}
        return names.get(getattr(value, "dtype", None))

    def save(self, key, value):
        if isinstance(value, Tensor):
            self.saved[key] = value.raw.copy()
        elif isinstance(value, tuple):
            for index, item in enumerate(value):
                self.save(f"{key}.{index}", item)

    def binary(self, node, args, kwargs, operation):
        if kwargs.get("alpha", 1) != 1 or set(kwargs) - {"alpha"}:
            raise ValueError("source binary operation has an unsupported scalar modifier")
        left, right = args[:2]
        dtype = self.dtype(node)
        if operation == "add" and dtype == "f32" and any(str(user.target) == "aten.softmax.int" for user in node.users):
            if not isinstance(right, Tensor) or right.dtype != "bf16" or np.any((right.raw != 0) & (right.raw != 0xff7f)):
                raise ValueError("original FX softmax mask is not declared BF16 minfinite/zero")
            scores = as_f32(left)
            if np.any(~np.isfinite(scores.view(np.float32))) or np.any(np.abs(scores.view(np.float32)) > 32):
                raise ValueError("original FX pre-mask scores leave the selected bounded mask mapping")
            shape = np.broadcast_shapes(scores.shape, right.raw.shape)
            self.mask_additions[node.name] = (np.broadcast_to(scores, shape).copy(), np.broadcast_to(right.raw == 0, shape).copy())
            value = np.add(scores.view(np.float32), as_f32(right).view(np.float32), dtype=np.float32).view(np.uint32)
        elif dtype == "bf16":
            if not isinstance(left, Tensor) or not isinstance(right, Tensor) or left.dtype != "bf16" or right.dtype != "bf16":
                raise ValueError("original BF16 source operation requires BF16 tensor operands")
            if operation in {"add", "sub"}:
                value = primitives.host_bf16_add(left.raw, right.raw, subtract=operation == "sub")
            elif operation == "mul":
                value = primitives.selected_pointwise("bf16_mul", [left.raw, right.raw], source_bf16=True)
            else:
                raise ValueError("unsupported BF16 source arithmetic")
        elif dtype == "f32":
            value = primitives.selected_pointwise("f32_" + operation, [as_f32(left), as_f32(right)])
        else:
            raise ValueError("original FX arithmetic dtype is unsupported")
        return Tensor(value, dtype)

    def call(self, node):
        import operator
        import torch
        from atlas_ref.activation import quantize_act
        from atlas_ref.contractions import qlinear, contraction_nt

        args, kwargs, target = self.resolve(node.args), self.resolve(node.kwargs), str(node.target)
        if node.target is operator.getitem:
            return args[0][args[1]]
        if target.startswith("atlas."):
            schema = node.target._schema
            ordered = [args[index] if index < len(args) else kwargs.get(spec.name, spec.default_value)
                       for index, spec in enumerate(schema.arguments)]
            if target == "atlas.quantize_act.default":
                x, mode, reduce, rule, static_exp, code_dtype, tag, rows = ordered
                if code_dtype != "uint8" or x.dtype not in {"bf16", "f32"}:
                    raise ValueError("original FX quantizer code or activation dtype differs")
                values = x.raw if x.dtype == "bf16" else x.raw.view(np.float32)
                codes, exponent = quantize_act(values, mode, reduce, rule, static_exp, rows)
                return Tensor(codes, "u8"), Tensor(exponent, "i32")
            if target not in {"atlas.qlinear.default", "atlas.qk_matmul.default", "atlas.pv_matmul.default"}:
                raise ValueError("unsupported original FX custom operation: " + target)
            values = ordered[:4]
            if [value.dtype for value in values] != ["u8", "i32", "u8", "i32"]:
                raise ValueError("original FX contraction code/exponent dtypes differ")
            if target == "atlas.qlinear.default":
                if ordered[4] is not None or ordered[6] != "atlas_e4m3" or ordered[5] not in {"bfloat16", "float32"}:
                    raise ValueError("original FX linear requires unfolded, bias-free source semantics")
                method = qlinear
                output_dtype = {"bfloat16": "bf16", "float32": "f32"}[ordered[5]]
            else:
                if len(ordered) > 4 and ordered[4] != 0:
                    raise ValueError("original FX PV fold is forbidden")
                method = contraction_nt
                output_dtype = "f32"
            result = method(*(value.raw for value in values), unit="mxu1", engine=self.engine, threads=self.threads)
            self.extra[node.name + ".accumulator_bf16"] = result.accumulator_bf16
            self.extra[node.name + ".readout_bf16"] = result.readout_bf16
            return Tensor(result.output_bf16 if output_dtype == "bf16" else primitives.widen(result.output_bf16), output_dtype)
        if target == "aten._assert_tensor_metadata.default":
            if "dtype" in kwargs:
                names = {torch.bfloat16: "bf16", torch.float32: "f32", torch.uint8: "u8", torch.int32: "i32"}
                if args[0].dtype != names.get(kwargs["dtype"]):
                    raise ValueError("source dtype assertion differs from original FX interpretation")
            return None
        if target == "aten.to.dtype":
            value, dtype = args[:2]
            names = {torch.bfloat16: "bf16", torch.float32: "f32"}
            target_dtype = names.get(dtype)
            if target_dtype == value.dtype:
                return Tensor(value.raw.copy(), value.dtype)
            if (value.dtype, target_dtype) == ("bf16", "f32"):
                return Tensor(primitives.widen(value.raw), "f32")
            if (value.dtype, target_dtype) == ("f32", "bf16"):
                return Tensor(primitives.narrow(value.raw), "bf16")
            raise ValueError("original source dtype conversion has no selected independent policy")
        if target in {"aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor"}:
            return self.binary(node, args, kwargs, target.split(".")[1])
        if target == "aten.pow.Tensor_Scalar":
            if args[1] != 2 or args[0].dtype != "f32":
                raise ValueError("original FX power is outside the selected square policy")
            return Tensor(primitives.selected_pointwise("f32_mul", [args[0].raw, args[0].raw]), "f32")
        if target == "aten.mean.dim":
            value, axes, keepdim = args
            if value.dtype != "f32" or tuple(axes) not in {(-1,), (value.raw.ndim - 1,)}:
                raise ValueError("original FX mean requires a selected final-axis reduction")
            summed = primitives.selected_sum(value.raw, np.zeros(value.raw.shape[:-1], np.uint32), axis=value.raw.ndim - 1,
                                             reduction_policy=self.policy["reduction"])
            denominator = np.asarray(value.raw.shape[-1], np.float32).view(np.uint32)
            result = primitives.selected_pointwise("f32_div", [summed, denominator])
            return Tensor(result[..., None] if keepdim else result, "f32")
        if target == "aten.rsqrt.default":
            return Tensor(primitives.selected_pointwise("source_f32_rsqrt_unqualified", [as_f32(args[0])]), "f32")
        if target == "aten.neg.default":
            value = args[0]
            if value.dtype not in {"bf16", "f32"}:
                raise ValueError("source negation dtype differs")
            return Tensor(value.raw ^ np.array(0x8000 if value.dtype == "bf16" else 0x80000000, dtype=value.raw.dtype), value.dtype)
        if target == "aten.silu.default":
            value = args[0]
            x = as_f32(value)
            negated = x ^ np.uint32(0x80000000)
            exponential = primitives.selected_exp(negated)
            denominator = primitives.selected_pointwise("f32_add", [np.full(x.shape, 0x3f800000, np.uint32), exponential])
            quotient = primitives.selected_pointwise("f32_div", [x, denominator])
            self.extra.update({node.name + ".negated_f32": negated, node.name + ".exp_f32": exponential,
                               node.name + ".denominator_f32": denominator, node.name + ".quotient_f32": quotient})
            return Tensor(primitives.narrow(quotient) if self.dtype(node) == "bf16" else quotient, self.dtype(node))
        if target == "aten.softmax.int":
            value, axis = args[:2]
            if axis not in {-1, value.raw.ndim - 1} or kwargs or len(args) > 2 and args[2] is not None:
                raise ValueError("original FX softmax arguments differ from its selected normalized policy")
            producer = node.args[0].name
            if producer not in self.mask_additions:
                raise ValueError("original FX softmax lacks the explicit source causal-mask addition")
            scores, keep = self.mask_additions[producer]
            result = primitives.attention_ref.normalized_softmax(primitives.narrow(scores), keep)
            self.extra.update({node.name + "." + key: value for key, value in result.items()})
            return Tensor(primitives.widen(result["probabilities_bf16"]), "f32")
        if target in {"aten.view.default", "aten.reshape.default"}:
            return Tensor(args[0].raw.reshape(tuple(args[1])).copy(), args[0].dtype)
        if target == "aten.transpose.int":
            return Tensor(np.swapaxes(args[0].raw, args[1], args[2]).copy(), args[0].dtype)
        if target == "aten.unsqueeze.default":
            return Tensor(np.expand_dims(args[0].raw, args[1]).copy(), args[0].dtype)
        if target == "aten.expand.default":
            value, shape = args[:2]
            shape = tuple(value.raw.shape[index] if extent == -1 else extent for index, extent in enumerate(shape))
            return Tensor(np.broadcast_to(value.raw, shape).copy(), value.dtype)
        if target == "aten.slice.Tensor":
            value, axis, begin, end, *step = args
            indices = [slice(None)] * value.raw.ndim
            indices[axis] = slice(begin, end, step[0] if step else 1)
            return Tensor(value.raw[tuple(indices)].copy(), value.dtype)
        if target == "aten.cat.default":
            values, axis = args
            if len({value.dtype for value in values}) != 1:
                raise ValueError("original FX concat source dtypes differ")
            return Tensor(np.concatenate([value.raw for value in values], axis=axis), values[0].dtype)
        if target == "aten.contiguous.default":
            return Tensor(np.ascontiguousarray(args[0].raw).copy(), args[0].dtype)
        raise ValueError("original FX operation has no explicit selected independent arithmetic: " + target)

    def run(self, inputs):
        user = iter(inputs)
        for node in self.program.graph.nodes:
            if node.op == "placeholder":
                spec = self.specs[node.target]
                if spec.kind.name == "USER_INPUT":
                    value = Tensor(np.asarray(next(user)).copy(), "bf16")
                elif spec.kind.name in {"PARAMETER", "BUFFER"}:
                    value = self.initial_state[spec.target]
                elif spec.kind.name == "CONSTANT_TENSOR":
                    value = raw_tensor(self.program.constants[spec.target])
                else:
                    raise ValueError("original FX graph has a mutable or unsupported input role")
            elif node.op == "call_function":
                value = self.call(node)
            elif node.op == "output":
                value = self.resolve(node.args[0])
                self.env[node] = value
                marker = object()
                if next(user, marker) is not marker:
                    raise ValueError("original source graph has fewer runtime inputs than its capsule")
                return value
            else:
                raise ValueError("original FX graph contains an unsupported effect or call")
            if isinstance(value, Tensor):
                expected = node.meta.get("val")
                if tuple(value.raw.shape) != tuple(expected.shape) or value.dtype != self.dtype(node):
                    raise ValueError("original FX result dtype or shape differs at " + node.name)
            self.env[node] = value
            self.save(node.name, value)
        raise ValueError("original FX graph has no ordered output")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--engine", choices=("numpy", "native"), default="numpy")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--native-qualification", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    before = primitives.authenticate_model(args.model_root)
    sys.path.insert(0, str(args.model_root.resolve()))
    sys.path.insert(0, str(args.reference_root.resolve()))
    import torch
    from atlas_ref.ops import register

    torch.set_num_threads(1)
    register()
    evidence = json.loads(args.validation.read_text())
    if (evidence.get("status") != "PASS" or evidence.get("mismatches") != 0
            or evidence.get("adapter_sha256") != sha256(Path(primitives.__file__).read_bytes()).hexdigest()
            or evidence.get("attention_adapter_sha256") != sha256(Path(primitives.attention_ref.__file__).read_bytes()).hexdigest()
            or evidence.get("model_arithmetic_sources") != before
            or evidence.get("reference_sources") != primitives.attention_ref.reference_sources()):
        raise ValueError("selected arithmetic helpers lack a matching independent validation")
    provenance = json.loads((args.capsule / "source.json").read_text())
    model_path = args.capsule / "model.pt2"
    if sha256(model_path.read_bytes()).hexdigest() != provenance["model_sha256"]:
        raise ValueError("original saved FX export differs from extracted source identity")
    inputs = primitives.source_inputs(args.capsule / "tensors.safetensors", provenance)
    program = torch.export.load(model_path)
    engine = primitives.configure_engine(args.engine, args.threads, args.native_qualification)
    interpreter = Interpreter(program, json.loads(args.policy.read_text()), engine=args.engine, threads=args.threads)
    outputs = interpreter.run([inputs[key] for key in provenance["input_keys"]])
    if not isinstance(outputs, tuple) or not all(isinstance(value, Tensor) for value in outputs):
        raise ValueError("original FX source outputs are not ordered tensor values")
    for name, initial in interpreter.initial_state.items():
        current = raw_tensor(program.state_dict[name])
        if current.raw.tobytes() != initial.raw.tobytes():
            raise ValueError("original FX lifted constant changed")
    if primitives.authenticate_model(args.model_root) != before:
        raise ValueError("instruction-model arithmetic changed during original FX replay")
    args.out.mkdir(parents=True, exist_ok=False)
    arrays = {**interpreter.saved, **interpreter.extra, **{f"output_{i}": value.raw for i, value in enumerate(outputs)}}
    np.savez(args.out / "expected.npz", **arrays)
    info = lambda value: {"shape": list(value.shape), "dtype": value.dtype.str, "sha256": sha256(value.tobytes()).hexdigest()}
    record = {"schema": "atlas.independent_original_fx_layer.v1", "status": "EXPECTED", "engine": engine,
              "source_model_sha256": provenance["model_sha256"], "model_arithmetic_sources": before,
              "selected_policy_sha256": sha256(args.policy.read_bytes()).hexdigest(),
              "reference_sources": primitives.attention_ref.reference_sources(),
              "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "primitive_helper_sha256": sha256(Path(primitives.__file__).read_bytes()).hexdigest(),
              "validation_sha256": sha256(args.validation.read_bytes()).hexdigest(),
              "inputs": {key: info(value) for key, value in inputs.items()},
              "original_lifted_constants": {key: info(value.raw) for key, value in interpreter.initial_state.items()},
              "boundaries": {key: info(value) for key, value in arrays.items()}, "outputs": [f"output_{i}" for i in range(len(outputs))],
              "ordinary_framework_kernels_called": False, "semantic_json_or_emitted_program_inputs": False,
              "dispatch": "original saved FX ATen/custom-op attributes, dtypes, shapes, and scalar operands",
              "whole_layer_device_exactness": "PENDING separate generated execution comparison"}
    (args.out / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "EXPECTED", "source": "original saved FX", "outputs": {key: info(arrays[key]) for key in record["outputs"]}}))


if __name__ == "__main__":
    main()
