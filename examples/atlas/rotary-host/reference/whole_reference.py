"""Original-FX sequential oracle with unchanged constants and bounded live state.

This module extends the qualified layer adapter, never a compiler evaluator.
Reference version 3 computes rotary preparation in the interpreter with
independent, rigorously enclosed correctly rounded binary32 sin and cos.
The v2 BF16 linear policy remains exact-product ascending-K binary32 RNE.
Other qualified arithmetic remains delegated to the original source adapters.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fx_layer_reference as base
import layer_reference as primitives
import rowspan_softmax
import rotary_exact

Tensor = base.Tensor
REFERENCE_VERSION = 3
ROTARY_POLICY = rotary_exact.POLICY
BF16_LINEAR_POLICY = "bf16-exact-product-f32-ltr-rne-bf16-rne-v1"


def checked(path):
    path = Path(path)
    return path


def info(value):
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": sha256(value.tobytes()).hexdigest()}


def raw_tensor(value):
    import torch
    if value.dtype == torch.int64:
        return Tensor(value.detach().contiguous().numpy().copy(), "i64")
    if value.dtype == torch.bool:
        return Tensor(value.detach().contiguous().numpy().copy(), "bool")
    return base.raw_tensor(value)


def bf16_linear(activation, weight, *, columns=256):
    """Independent source head: exact BF16 products and ascending-K f32 sums.

    Raw BF16 operands widen by bit shift. Each multiply and add is a distinct
    NumPy float32 ufunc, so no FMA, reduction tree, BF16 product rounding, or
    intermediate BF16 tile rounding participates. Positive zero seeds every
    output; one explicit ties-to-even BF16 conversion follows the final K.
    Column blocks affect memory only. BF16 products are exact when representable
    in f32; tiny products preserve source f32 RNE gradual underflow.
    Nonfinite operands, intermediates, or final BF16 results are refused.
    """
    a, b = np.asarray(activation), np.asarray(weight)
    if a.dtype != np.uint16 or b.dtype != np.uint16 or b.ndim != 2 or a.ndim < 1 or a.shape[-1] != b.shape[-1]:
        raise ValueError("BF16 linear requires raw BF16 [...,K] and [N,K]")
    if columns < 1 or np.any((a & 0x7fff) >= 0x7f80) or np.any((b & 0x7fff) >= 0x7f80):
        raise ValueError("BF16 linear requires finite unchanged operands")
    aa = (a.reshape(-1, a.shape[-1]).astype(np.uint32) << np.uint32(16)).view(np.float32)
    bb = (b.astype(np.uint32) << np.uint32(16)).view(np.float32)
    output = np.empty((aa.shape[0], b.shape[0]), np.uint16)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        for begin in range(0, b.shape[0], columns):
            right = bb[begin:begin + columns]
            total = np.zeros((aa.shape[0], right.shape[0]), np.float32)
            for k in range(aa.shape[-1]):
                product = np.multiply(aa[:, k, None], right[None, :, k], dtype=np.float32)
                np.add(total, product, out=total, dtype=np.float32)
            raw = total.view(np.uint32)
            rounded = (
                (raw + np.uint32(0x7fff) + ((raw >> np.uint32(16)) & np.uint32(1)))
                >> np.uint32(16)).astype(np.uint16)
            if np.any((rounded & 0x7fff) >= 0x7f80):
                raise ValueError("BF16 linear final result must be finite")
            output[:, begin:begin + right.shape[0]] = rounded
    return output.reshape((*a.shape[:-1], b.shape[0]))


class Interpreter(base.Interpreter):
    """Same arithmetic dispatch, with typed inputs and last-consumer release."""
    def __init__(self, program, policy, *, engine="numpy", threads=1, linear_policy=None,
                 retain=(), progress=None):
        for field, expected in primitives.POLICIES.items():
            if not primitives.selected_policy_matches(field, policy.get(field)) and not (field == "already_quantized_mask" and policy.get(field) == rowspan_softmax.POLICY):
                raise ValueError("original FX selected policy missing or changed: " + field)
        self.program, self.policy = program, policy
        self.engine, self.threads, self.linear_policy = engine, threads, linear_policy
        self.env, self.saved, self.extra, self.mask_additions = {}, {}, {}, {}
        self.specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
        self.initial_state = {}
        self.constants = {}
        self.boundaries = {}
        self.retain, self.progress = set(retain), progress

    def dtype(self, node):
        import torch
        dtype = getattr(node.meta.get("val"), "dtype", None)
        return {torch.int64: "i64", torch.bool: "bool"}.get(dtype) or super().dtype(node)

    def save(self, key, value):
        if isinstance(value, Tensor):
            self.boundaries[key] = {**info(value.raw), "logical_dtype": value.dtype}
            if key in self.retain or "*" in self.retain:
                self.saved[key] = value.raw.copy()
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                self.save(f"{key}.{i}", item)

    def call(self, node):
        target = str(node.target)
        if target in {"aten.sin.default", "aten.cos.default"}:
            args, kwargs = self.resolve(node.args), self.resolve(node.kwargs)
            if len(args) != 1 or kwargs or args[0].dtype != "f32" or self.dtype(node) != "f32":
                raise ValueError("rotary trig requires one binary32 source tensor")
            return Tensor(rotary_exact.trig_f32(args[0].raw, target.split(".")[1]), "f32")
        if target in {"aten.matmul.default", "aten.bmm.default", "aten.mm.default"}:
            args, kwargs = self.resolve(node.args), self.resolve(node.kwargs)
            left, right = args
            if (kwargs or left.dtype != "f32" or right.dtype != "f32" or self.dtype(node) != "f32"
                    or left.raw.ndim < 2 or right.raw.ndim < 2
                    or left.raw.shape[-1] != 1 or right.raw.shape[-2] != 1):
                raise ValueError("rotary source matmul requires binary32 reduction extent one")
            # Positive-zero seeded K=1 source reduction equals the rounded product
            # on the captured nonnegative frequency/position operands.
            if np.any(left.raw >> 31) or np.any(right.raw >> 31):
                raise ValueError("rotary K=1 operands must be nonnegative")
            return Tensor(rotary_exact.multiply_f32(left.raw, right.raw), "f32")
        if target == "aten.arange.default":
            args, kwargs = self.resolve(node.args), self.resolve(node.kwargs)
            if len(args) != 1 or type(args[0]) is not int or not 0 <= args[0] <= 2048 or self.dtype(node) != "i64":
                raise ValueError("source position arange outside qualified domain")
            if set(kwargs) - {"dtype", "device", "pin_memory", "layout"}:
                raise ValueError("source position arange arguments differ")
            return Tensor(np.arange(args[0], dtype=np.int64), "i64")
        if target == "aten.to.dtype" and self.resolve(node.args[0]).dtype == "i64":
            import torch
            value, dtype = self.resolve(node.args)[:2]
            if dtype != torch.float32 or np.any(value.raw < 0) or np.any(value.raw > 2047):
                raise ValueError("source integer-to-f32 conversion leaves rotary position domain")
            raw = np.array([rotary_exact.ratio_to_f32(int(x)) for x in value.raw.flat], np.uint32).reshape(value.raw.shape)
            return Tensor(raw, "f32")
        if target == "aten.to.dtype_layout":
            import torch
            args, kwargs = self.resolve(node.args), self.resolve(node.kwargs)
            names = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int64: "i64", torch.bool: "bool"}
            if names.get(kwargs.get("dtype")) != args[0].dtype:
                raise ValueError("source dtype-layout conversion must preserve bytes")
            return Tensor(args[0].raw.copy(), args[0].dtype)
        if target == "aten._assert_tensor_metadata.default":
            import torch
            args, kwargs = self.resolve(node.args), self.resolve(node.kwargs)
            names = {torch.bfloat16: "bf16", torch.float32: "f32", torch.uint8: "u8", torch.int32: "i32", torch.int64: "i64", torch.bool: "bool"}
            if "dtype" in kwargs and names.get(kwargs["dtype"]) != args[0].dtype:
                raise ValueError("source dtype assertion differs")
            return None
        if target == "aten.softmax.int" and self.policy.get("already_quantized_mask") == rowspan_softmax.POLICY:
            args, kwargs = self.resolve(node.args), self.resolve(node.kwargs)
            value, axis = args[:2]
            if axis not in {-1, value.raw.ndim - 1} or kwargs or len(args) > 2 and args[2] is not None:
                raise ValueError("rowspan80 original FX softmax axis or dtype differs")
            producer = node.args[0].name
            if producer not in self.mask_additions:
                raise ValueError("rowspan80 original FX softmax lacks a source mask addition")
            scores, keep = self.mask_additions[producer]
            result = rowspan_softmax.normalized_softmax(primitives.narrow(scores), keep)
            self.extra.update({node.name + "." + key: value for key, value in result.items()})
            return Tensor(primitives.widen(result["probabilities_bf16"]), "f32")
        if target in {"aten.alias.default", "aten.detach.default", "aten.clone.default"}:
            value = self.resolve(node.args[0])
            return Tensor(value.raw.copy(), value.dtype)
        if target == "aten.linear.default":
            args = self.resolve(node.args)
            if self.linear_policy != BF16_LINEAR_POLICY:
                raise ValueError("unquantized BF16 linear requires an explicit selected policy")
            if self.dtype(node) != "bf16" or any(value.dtype != "bf16" for value in args[:2]) or len(args) > 2 and args[2] is not None:
                raise ValueError("selected BF16 linear requires bias-free BF16 operands/output")
            return Tensor(bf16_linear(args[0].raw, args[1].raw), "bf16")
        return super().call(node)

    def binary(self, node, args, kwargs, operation):
        if self.dtype(node) == "i64" and operation in {"add", "sub"}:
            left, right = args[:2]
            if left.dtype != "i64" or type(right) is not int or abs(right) > 2048 or kwargs:
                raise ValueError("source integer position arithmetic differs")
            return Tensor((left.raw + right if operation == "add" else left.raw - right).astype(np.int64), "i64")
        if (operation == "mul" and self.dtype(node) == "f32" and len(args) == 2
                and type(args[1]) in {int, float} and args[1] == 1
                and str(getattr(node.args[0], "target", "")) in {"aten.sin.default", "aten.cos.default"}):
            if kwargs or args[0].dtype != "f32":
                raise ValueError("rotary attention scale must be binary32 one")
            return Tensor(args[0].raw.copy(), "f32")
        if (self.policy.get("already_quantized_mask") == rowspan_softmax.POLICY and operation == "add"
                and self.dtype(node) == "f32" and any(str(user.target) == "aten.softmax.int" for user in node.users)):
            left, right = args[:2]
            if kwargs or not isinstance(right, Tensor) or right.dtype != "bf16" or np.any((right.raw != 0) & (right.raw != 0xff7f)):
                raise ValueError("rowspan80 source mask must retain BF16 minfinite/zero bytes")
            scores = base.as_f32(left)
            values = scores.view(np.float32)
            if np.any(~np.isfinite(values)) or np.any(np.abs(values) > 128):
                raise ValueError("rowspan80 source scores leave the sentinel proof abs<=128 domain")
            shape = np.broadcast_shapes(scores.shape, right.raw.shape)
            expanded, keep = np.broadcast_to(scores, shape).copy(), np.broadcast_to(right.raw == 0, shape).copy()
            rowspan_softmax.check_domain(primitives.narrow(expanded), keep)
            self.mask_additions[node.name] = (expanded, keep)
            return Tensor(np.add(values, base.as_f32(right).view(np.float32), dtype=np.float32).view(np.uint32), "f32")
        return super().binary(node, args, kwargs, operation)

    def run(self, inputs):
        user = iter(inputs)
        nodes = list(self.program.graph.nodes)
        last = {dependency: i for i, node in enumerate(nodes) for dependency in node.all_input_nodes}
        for index, node in enumerate(nodes):
            if node.op == "placeholder":
                spec = self.specs[node.target]
                if spec.kind.name == "USER_INPUT":
                    try:
                        incoming = next(user)
                    except StopIteration as error:
                        raise ValueError("original FX input count is too small") from error
                    value = incoming if isinstance(incoming, Tensor) else Tensor(np.asarray(incoming).copy(), self.dtype(node))
                elif spec.kind.name in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR"}:
                    source = self.program.constants if spec.kind.name == "CONSTANT_TENSOR" else self.program.state_dict
                    value = raw_tensor(source[spec.target])
                    self.constants[spec.target] = {**info(value.raw), "logical_dtype": value.dtype}
                else:
                    raise ValueError("original FX graph has a mutable or unsupported input role")
            elif node.op == "call_function":
                value = self.call(node)
            elif node.op == "output":
                value = self.resolve(node.args[0])
                marker = object()
                if next(user, marker) is not marker:
                    raise ValueError("original FX input count is too large")
                self._verify_constants()
                self.env.clear()
                return value
            else:
                raise ValueError("original FX graph has an unsupported effect or call")
            if isinstance(value, Tensor):
                if tuple(value.raw.shape) != tuple(node.meta["val"].shape) or value.dtype != self.dtype(node):
                    raise ValueError("original FX result shape/dtype differs at " + node.name)
                expected_raw = {"bf16": np.uint16, "f32": np.uint32, "u8": np.uint8,
                                "i32": np.int32, "i64": np.int64, "bool": np.bool_}[value.dtype]
                if value.raw.dtype != np.dtype(expected_raw):
                    raise ValueError("original FX input/result raw encoding differs at " + node.name)
            self.env[node] = value
            self.save(node.name, value)
            for key, extra in self.extra.items():
                self.boundaries[key] = info(extra)
                if key in self.retain or "*" in self.retain:
                    self.saved[key] = extra.copy()
            self.extra.clear()
            for dependency in node.all_input_nodes:
                if last[dependency] == index:
                    self.env.pop(dependency, None)
                    self.mask_additions.pop(dependency.name, None)
            if not node.users:
                self.env.pop(node, None)
            if self.progress and str(node.target).startswith("atlas.q"):
                self.progress(node.name, target=str(node.target))
        raise ValueError("original FX graph has no output")

    def _verify_constants(self):
        for name, record in self.constants.items():
            source = self.program.state_dict if name in self.program.state_dict else self.program.constants
            current = raw_tensor(source[name])
            if {**info(current.raw), "logical_dtype": current.dtype} != record:
                raise ValueError("original FX lifted constant changed: " + name)


def validate_helpers(validation, model_root):
    before = primitives.authenticate_model(model_root)
    evidence = json.loads(checked(validation).read_text())
    if (evidence.get("status") != "PASS" or evidence.get("mismatches") != 0
            or evidence.get("adapter_sha256") != sha256(Path(primitives.__file__).read_bytes()).hexdigest()
            or evidence.get("attention_adapter_sha256") != sha256(Path(primitives.attention_ref.__file__).read_bytes()).hexdigest()
            or evidence.get("selected_policies") != primitives.POLICIES
            or evidence.get("model_arithmetic_sources") != before
            or evidence.get("reference_sources") != primitives.attention_ref.reference_sources()):
        raise ValueError("selected ordinary/attention arithmetic lacks matching validation")
    return before


def run_sequence(capsules, output, policy, *, engine="numpy", threads=1,
                 linear_policy=None, stages=None, retain=(), provenance=None,
                 initial_inputs=None, initial_preparation=None, initial_tokens=None,
                 source_bundle=None, selection_manifest=None,
                 rotary_source_bundle=None, rotary_validation=None):
    import torch
    from safetensors.torch import load_file
    capsule_root, output = checked(capsules), checked(output)
    manifest = json.loads((capsule_root / "manifest.json").read_text())
    whole = manifest["whole_model"]
    stage_names = whole["regions"] if stages is None else stages
    if len(set(stage_names)) != len(stage_names) or any(name not in whole["regions"] for name in stage_names):
        raise ValueError("requested source stages must be unique members of the sequential manifest")
    links = {row["kind"]: row["inputs"] for row in whole["links"]}
    initial_activation, initial_receipt = None, None
    if initial_inputs is not None:
        import initial_embedding
        if initial_preparation is None or source_bundle is None:
            raise ValueError("initial embedding override requires its preparation and unchanged source bundle")
        if stage_names[0] != whole["regions"][0] or stage_names[0] != "layer0":
            raise ValueError("initial embedding override requires the first decoder stage")
        first_capsule = json.loads((capsule_root / stage_names[0] / "source.json").read_text())
        initial_activation, initial_receipt = initial_embedding.authenticate(
            initial_inputs, initial_preparation,
            initial_tokens if initial_tokens is not None else checked(initial_preparation).parent / "tokens.npy",
            source_bundle, manifest, first_capsule, selection_manifest)
    elif any(value is not None for value in (initial_preparation, initial_tokens, source_bundle, selection_manifest)):
        raise ValueError("initial embedding options require --initial-inputs")
    rotary_outputs, rotary_provenance = None, None
    if rotary_source_bundle is not None:
        import rotary_source
        if rotary_validation is None:
            raise ValueError("v3 rotary source replay requires its exhaustive qualification receipt")
        qualification = json.loads(checked(rotary_validation).read_text())
        trace_path = checked(rotary_source_bundle) / "frontend-trace.json"
        if (qualification.get("status") != "PASS" or qualification.get("mismatches") != 0
                or qualification.get("selected_policy") != ROTARY_POLICY
                or qualification.get("oracle_sha256") != sha256(Path(rotary_exact.__file__).read_bytes()).hexdigest()
                or qualification.get("source_adapter_sha256") != sha256(Path(rotary_source.__file__).read_bytes()).hexdigest()
                or qualification.get("whole_interpreter_sha256") != sha256(Path(__file__).read_bytes()).hexdigest()
                or qualification.get("source_slice", {}).get("trace_sha256") != sha256(trace_path.read_bytes()).hexdigest()):
            raise ValueError("v3 rotary qualification differs from source or independent oracle")
        gate_path = checked(qualification["archive_gate_path"])
        if sha256(gate_path.read_bytes()).hexdigest() != qualification["archive_gate_sha256"]:
            raise ValueError("v3 rotary archive gate changed")
        rotary_outputs, rotary_provenance, _ = rotary_source.replay(trace_path, policy)
        rotary_provenance.update(qualification_sha256=sha256(checked(rotary_validation).read_bytes()).hexdigest(),
                                 archive_gate_sha256=qualification["archive_gate_sha256"])
    elif rotary_validation is not None:
        raise ValueError("rotary qualification requires a captured rotary source bundle")
    prior, records = {}, []
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": "atlas.independent_original_fx_sequence.v3", "reference_version": REFERENCE_VERSION, "status": "RUNNING",
              "source_model_sha256": manifest["source_model_sha256"],
              "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "base_adapter_sha256": sha256(Path(base.__file__).read_bytes()).hexdigest(),
              "primitive_helper_sha256": sha256(Path(primitives.__file__).read_bytes()).hexdigest(),
              "attention_adapter_sha256": sha256(Path(primitives.attention_ref.__file__).read_bytes()).hexdigest(),
              "selected_policies": policy, "bf16_linear_policy": linear_policy,
              "rotary_policy": ROTARY_POLICY,
              "rotary_oracle_sha256": sha256(Path(rotary_exact.__file__).read_bytes()).hexdigest(),
              "ordinary_framework_kernels_called": False,
              "compiler_or_emitted_program_inputs": False, "stages": records,
              "initial_embedding_override": initial_receipt,
              "rotary_source_replay": rotary_provenance}
    if provenance:
        record.update(provenance)
    def write():
        (output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    write()
    try:
        for name in stage_names:
            start = time.monotonic()
            capsule = checked(capsule_root / name)
            provenance = json.loads((capsule / "source.json").read_text())
            model_file = capsule / "model.pt2"
            if sha256(model_file.read_bytes()).hexdigest() != provenance["model_sha256"]:
                raise ValueError("source capsule FX hash differs: " + name)
            fixtures = load_file(str(capsule / "tensors.safetensors"))
            inputs, origins = [None] * len(provenance["input_keys"]), []
            stage_links = {row["input_index"]: row for row in links[name]}
            for index, key in enumerate(provenance["input_keys"]):
                link = stage_links.get(index, {})
                if link.get("region") is not None:
                    region, output_index = link["region"], link["output_index"]
                    if region not in prior:
                        raise ValueError("sequential oracle predecessor missing: " + region)
                    inputs[index] = prior[region][output_index]
                    origins.append({"input": key, "reference_stage": region, "output_index": output_index})
                else:
                    if name == whole["regions"][0] and index == 0 and initial_activation is not None:
                        if provenance["source_boundaries"][index] != "embedding":
                            raise ValueError("initial embedding override cannot replace a different source boundary")
                        inputs[index] = Tensor(initial_activation.copy(), "bf16")
                        origins.append({"input": key, "source_boundary": "embedding",
                                        "arithmetic": "independently authenticated raw source BF16 gather",
                                        "preparation_sha256": initial_receipt["preparation_sha256"],
                                        "tokens_sha256": initial_receipt["tokens_sha256"]})
                        continue
                    boundary = provenance["source_boundaries"][index]
                    rotary_name = {"to_7": "cos", "to_8": "sin"}.get(boundary)
                    if rotary_name is not None:
                        if rotary_outputs is None:
                            raise ValueError("v3 refuses rotary fixtures; provide captured rotary source bundle and qualification")
                        value = rotary_outputs[rotary_name]
                        if (value.dtype != "bf16" or list(value.raw.shape) != provenance["tensors"][key]["shape"]):
                            raise ValueError("computed rotary shape differs from source boundary")
                        inputs[index] = Tensor(value.raw.copy(), value.dtype)
                        origins.append({"input": key, "source_boundary": boundary,
                                        "arithmetic": "v3 independent original-ATen rotary interpreter",
                                        "rotary_operation": rotary_name, "rotary_source_replay": rotary_provenance})
                        continue
                    value = fixtures[key]
                    # Fixtures may supply preparation boundaries, never a hidden-state predecessor.
                    if info(raw_tensor(value).raw)["sha256"] != provenance["tensors"][key]["sha256"]:
                        raise ValueError("unchanged preparation fixture differs: " + key)
                    inputs[index] = raw_tensor(value)
                    origins.append({"input": key, "source_boundary": provenance["source_boundaries"][index],
                                    "arithmetic": "unchanged source preparation fixture"})
            program = torch.export.load(model_file)
            folder = output / name
            folder.mkdir()
            np.savez(folder / "runtime_inputs.npz", **{f"input_{i}": value.raw for i, value in enumerate(inputs)})
            (folder / "inputs.json").write_text(json.dumps({
                f"input_{i}": {**info(value.raw), "logical_dtype": value.dtype, "origin": origins[i]}
                for i, value in enumerate(inputs)}, indent=2) + "\n")
            interpreter = Interpreter(program, policy, engine=engine, threads=threads,
                                      linear_policy=linear_policy, retain=retain,
                                      progress=lambda node, **rest: print(json.dumps({"stage": name, "node": node, **rest}), flush=True))
            values = interpreter.run(inputs)
            values = values if isinstance(values, (tuple, list)) else (values,)
            if not all(isinstance(value, Tensor) for value in values):
                raise ValueError("source stage outputs include non-tensors")
            prior[name] = values
            arrays = {**interpreter.saved, **{f"output_{i}": value.raw for i, value in enumerate(values)}}
            np.savez(folder / "expected.npz", **arrays)
            stage_record = {"schema": "atlas.independent_original_fx_layer.v3", "reference_version": REFERENCE_VERSION, "stage": name, "status": "EXPECTED", "source_model_sha256": provenance["model_sha256"],
                            "inputs": origins, "constants": interpreter.constants,
                            "original_lifted_constants": {key: value for key, value in interpreter.constants.items()
                                                          if key in program.state_dict},
                            "ordinary_framework_kernels_called": False,
                            "semantic_json_or_emitted_program_inputs": False,
                            "selected_policy_sha256": record.get("selected_policy_sha256"),
                            "boundaries": interpreter.boundaries,
                            "outputs": {f"output_{i}": {**info(value.raw), "logical_dtype": value.dtype}
                                        for i, value in enumerate(values)},
                            "constant_bytes_identical": True,
                            "elapsed_seconds": round(time.monotonic() - start, 6)}
            (folder / "reference.json").write_text(json.dumps(stage_record, indent=2) + "\n")
            records.append(stage_record)
            write()
            print(json.dumps({"stage": name, "status": "EXPECTED", "elapsed_seconds": stage_record["elapsed_seconds"]}), flush=True)
            del interpreter, program, fixtures
        record["status"] = "EXPECTED"
        record["whole_model_device_exactness"] = "PENDING separate generated execution comparison"
        write()
        return record
    except Exception as error:
        record.update(status="REFUSED", error=str(error))
        write()
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capsules", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--engine", choices=("numpy", "native"), default="numpy")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--native-qualification", type=Path)
    parser.add_argument("--bf16-linear-policy", choices=(BF16_LINEAR_POLICY,))
    parser.add_argument("--bf16-linear-validation", type=Path)
    parser.add_argument("--mask-validation", type=Path)
    parser.add_argument("--initial-inputs", type=Path)
    parser.add_argument("--initial-preparation", type=Path)
    parser.add_argument("--initial-tokens", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--rotary-source-bundle", type=Path)
    parser.add_argument("--rotary-validation", type=Path)
    parser.add_argument("--stage", action="append")
    parser.add_argument("--retain", action="append", default=[])
    args = parser.parse_args()
    sys.path[:0] = [str(checked(args.model_root).resolve()), str(checked(args.reference_root).resolve())]
    import torch
    from atlas_ref.ops import register
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    register()
    before = validate_helpers(args.validation, args.model_root)
    policy = json.loads(checked(args.policy).read_text())
    if policy.get("already_quantized_mask") == rowspan_softmax.POLICY:
        if args.mask_validation is None:
            raise ValueError("rowspan80 softmax requires an independent instruction qualification receipt")
        qualification = json.loads(checked(args.mask_validation).read_text())
        if (qualification.get("status") != "PASS" or qualification.get("mismatches") != 0
                or qualification.get("selected_policy") != rowspan_softmax.POLICY
                or qualification.get("model_arithmetic_sources") != before
                or qualification.get("oracle_sha256") != rowspan_softmax.source_hash()):
            raise ValueError("rowspan80 instruction qualification differs from arithmetic or oracle")
    if args.bf16_linear_policy:
        import inspect
        if args.bf16_linear_validation is None:
            raise ValueError("BF16 linear requires its independent directed qualification receipt")
        qualification = json.loads(checked(args.bf16_linear_validation).read_text())
        if (qualification.get("status") != "PASS" or qualification.get("mismatches") != 0
                or qualification.get("selected_policy") != args.bf16_linear_policy
                or qualification.get("model_arithmetic_sources") != before
                or qualification.get("bf16_linear_oracle_sha256") != sha256(inspect.getsource(bf16_linear).encode()).hexdigest()):
            raise ValueError("BF16 linear qualification differs from selected arithmetic or oracle")
    engine_identity = primitives.configure_engine(args.engine, args.threads, args.native_qualification)
    result = run_sequence(args.capsules, args.out, policy,
                          engine=args.engine, threads=args.threads, linear_policy=args.bf16_linear_policy,
                          stages=args.stage, retain=args.retain,
                          initial_inputs=args.initial_inputs, initial_preparation=args.initial_preparation,
                          initial_tokens=args.initial_tokens, source_bundle=args.source_bundle,
                          selection_manifest=args.selection_manifest,
                          rotary_source_bundle=args.rotary_source_bundle, rotary_validation=args.rotary_validation,
                          provenance={
                              "model_arithmetic_sources": before, "engine": engine_identity,
                              "validation_sha256": sha256(checked(args.validation).read_bytes()).hexdigest(),
                              "selected_policy_sha256": sha256(checked(args.policy).read_bytes()).hexdigest(),
                              "mask_validation_sha256": (sha256(checked(args.mask_validation).read_bytes()).hexdigest()
                                                         if args.mask_validation else None),
                              "bf16_linear_validation_sha256": (sha256(checked(args.bf16_linear_validation).read_bytes()).hexdigest()
                                                                if args.bf16_linear_validation else None),
                              "reference_sources": primitives.attention_ref.reference_sources()})
    if primitives.authenticate_model(args.model_root) != before:
        raise ValueError("instruction-model arithmetic changed during sequential replay")
    result.update(model_arithmetic_sources=before, engine=engine_identity,
                  validation_sha256=sha256(checked(args.validation).read_bytes()).hexdigest(),
                  selected_policy_sha256=sha256(checked(args.policy).read_bytes()).hexdigest(),
                  mask_validation_sha256=(sha256(checked(args.mask_validation).read_bytes()).hexdigest() if args.mask_validation else None),
                  bf16_linear_validation_sha256=(sha256(checked(args.bf16_linear_validation).read_bytes()).hexdigest()
                                                 if args.bf16_linear_validation else None),
                  reference_sources=primitives.attention_ref.reference_sources())
    (args.out / "reference.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
