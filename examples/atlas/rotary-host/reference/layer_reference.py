"""Independent source-semantic layer replay under explicit selected arithmetic.

The inputs are unchanged source tensors, byte-bound constants, and a source
semantic graph. No emitted program or compiler numerical evaluator participates.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from math import prod
from pathlib import Path
import subprocess
import sys

import numpy as np

import attention_reference as attention_ref

REDUCTION_POLICY = "bf16-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1"
NONNEGATIVE_NORMAL_REDUCTION_POLICY = "bf16-nonnegative-normal-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1"

POLICIES = {
    "pointwise": "f32-to-bf16-bounded-pointwise-rne-chop-v1",
    "bf16_pointwise": "bf16-bounded-pointwise-rne-v1",
    "reduction": REDUCTION_POLICY,
    "pointwise_math": "f32-to-bf16-positive-denominator-lut-div-rsqrt-v1",
    "source_exp": "f32-to-bf16-bounded32-exp-lut-v1",
    "already_quantized_mask": "bf16-minfinite-zero-bounded32-nonempty-v1",
}
DTYPES = {"bf16": "<u2", "f32": "<u4", "i8": "u1", "i32": "<i4", "i64": "<i8"}
ENCODINGS = {"bf16": "bf16-le-raw", "f32": "f32-le-raw", "i8": "i8", "i32": "i32-le", "i64": "i64-le"}


def widen(bits):
    return np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)


def narrow(raw):
    """RNE BF16 boundary conversion, without host floating conversion kernels."""
    bits = np.asarray(raw)
    if bits.dtype != np.uint32:
        raise ValueError("f32 boundary must contain raw uint32 words")
    if np.any((bits & 0x7fffffff) > 0x7f800000):
        raise ValueError("unqualified source NaN payload in f32 cast")
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bounded(bits, *, positive=False, exp=False):
    bits = np.asarray(bits, dtype=np.uint16)
    magnitude = bits & 0x7fff
    good = ((magnitude == 0) | ((magnitude >= (0x80 if exp else 0x2000)) & (magnitude <= (0x4200 if exp else 0x5f00))))
    if positive:
        good &= (bits < 0x8000) & (magnitude != 0)
    if not np.all(good):
        raise ValueError("selected operation input leaves its explicit BF16 code domain")
    return bits


def host_bf16_add(a, b, *, subtract=False):
    """Exact integer BF16 addition and one RNE rounding, including zero signs."""
    a, b = np.broadcast_arrays(np.asarray(a, np.uint16), np.asarray(b, np.uint16))
    result = np.empty(a.shape, np.uint16)
    for index in np.ndindex(a.shape):
        left, right = int(a[index]), int(b[index]) ^ (0x8000 if subtract else 0)
        if (left & 0x7fff) > 0x7f80 or (right & 0x7fff) > 0x7f80:
            raise ValueError("source BF16 add/subtract NaN is unqualified")
        if (left & 0x7fff) == 0x7f80 or (right & 0x7fff) == 0x7f80:
            if (left & 0x7fff) == (right & 0x7fff) == 0x7f80 and (left ^ right) & 0x8000:
                raise ValueError("source BF16 infinity cancellation is unqualified")
            result[index] = left if (left & 0x7fff) == 0x7f80 else right
            continue
        def integer(code):
            exponent, mantissa = (code >> 7) & 255, code & 127
            value = mantissa if exponent == 0 else (128 + mantissa) << (exponent - 1)
            return -value if code & 0x8000 else value
        total = integer(left) + integer(right)
        if total == 0:
            result[index] = (left & right) & 0x8000 if not (left & 0x7fff or right & 0x7fff) else 0
            continue
        sign, magnitude = (0x8000 if total < 0 else 0), abs(total)
        shift = max(0, magnitude.bit_length() - 8)
        rounded, remainder = divmod(magnitude, 1 << shift)
        if shift and (remainder > 1 << (shift - 1) or remainder == 1 << (shift - 1) and rounded & 1):
            rounded += 1
        if rounded == 256:
            rounded, shift = 128, shift + 1
        exponent = shift + 1
        result[index] = sign | (rounded if rounded < 128 else 0x7f80 if exponent >= 255 else exponent << 7 | (rounded - 128))
    return result


def selected_pointwise(operation, operands, *, source_bf16=False):
    from npu_model.configs.numerics import RtlNumerics as rtl

    values = [bounded(value if source_bf16 else narrow(value), positive=(operation == "f32_div" and index == 1))
              for index, value in enumerate(operands)]
    tensors = [attention_ref._torch_bf16(value) for value in values]
    if operation in {"f32_add", "f32_sub", "f32_mul", "bf16_mul"}:
        result = {"f32_add": rtl.add, "f32_sub": rtl.sub, "f32_mul": rtl.mul, "bf16_mul": rtl.mul}[operation](*tensors)
    elif operation == "f32_div":
        result = rtl.mul(tensors[0], rtl.unary("rcp", tensors[1]))
    elif operation == "source_f32_rsqrt_unqualified":
        bounded(values[0], positive=True)
        result = rtl.unary("rcp", rtl.unary("sqrt", tensors[0]))
    else:
        raise ValueError("unknown selected pointwise operation")
    bits = attention_ref._bits(result)
    magnitude, exponent = bits & 0x7fff, (bits >> 7) & 255
    if np.any((magnitude != 0) & ((exponent == 0) | (exponent == 255))):
        raise ValueError("selected pointwise result leaves its finite normal/signed-zero domain")
    return bits if source_bf16 else widen(bits)


def selected_exp(raw):
    from npu_model.configs.numerics import RtlNumerics as rtl

    bits = bounded(narrow(raw), exp=True)
    return widen(attention_ref._bits(rtl.unary("exp", attention_ref._torch_bf16(bits))))


def selected_policy_matches(field, actual):
    """Additional reduction domains require an exact explicit selector."""
    return actual == POLICIES[field] or (field == "reduction" and actual == NONNEGATIVE_NORMAL_REDUCTION_POLICY)


def reduction_operands(bits, reduction_policy):
    if reduction_policy == REDUCTION_POLICY:
        result = bounded(bits).copy()
        result[result == 0x8000] = 0
        return result
    if reduction_policy != NONNEGATIVE_NORMAL_REDUCTION_POLICY:
        raise ValueError("unknown explicit selected reduction policy")
    bits = np.asarray(bits, dtype=np.uint16)
    if np.any((bits != 0) & ((bits < 0x0080) | (bits > 0x7f7f))):
        raise ValueError("selected nonnegative reduction requires positive normal BF16 or positive zero")
    return bits


def selected_sum(raw, seed, *, axis, reduction_policy=REDUCTION_POLICY):
    from npu_model.configs.numerics import RtlNumerics as rtl

    bits = reduction_operands(narrow(raw), reduction_policy)
    initial = reduction_operands(narrow(seed), reduction_policy)
    if axis != bits.ndim - 1 or initial.shape != bits.shape[:-1]:
        raise ValueError("selected source sum requires a final-axis reduction with a matching seed")
    total = attention_ref._torch_bf16(initial[..., None])
    for start in range(0, bits.shape[-1], 32):
        width = min(32, bits.shape[-1] - start)
        tile = np.zeros((*bits.shape[:-1], 32), np.uint16)
        tile[..., :width] = bits[..., start:start + width]
        partial = rtl.row_sum(attention_ref._torch_bf16(tile))
        if reduction_policy == NONNEGATIVE_NORMAL_REDUCTION_POLICY:
            # Nonnegative trees cannot cancel or underflow; a finite normal
            # BF16 partial also proves that every earlier FP32 node is finite.
            reduction_operands(attention_ref._bits(partial), reduction_policy)
        total = rtl.add(total, partial)
        if reduction_policy == NONNEGATIVE_NORMAL_REDUCTION_POLICY:
            reduction_operands(attention_ref._bits(total), reduction_policy)
    return widen(attention_ref._bits(total)[..., 0])


def project(value, mapping, shape):
    if mapping["loop_rank"] != len(shape) or len(mapping["coefficients"]) != value.ndim:
        raise ValueError("source affine projection rank differs")
    grid = np.indices(shape, dtype=np.int64)
    indices = []
    for axis, (coefficients, offset) in enumerate(zip(mapping["coefficients"], mapping["offsets"], strict=True)):
        if len(coefficients) != len(shape) or any(type(item) is not int for item in (*coefficients, offset)):
            raise ValueError("source affine projection is not integral")
        index = np.full(shape, offset, np.int64)
        for coefficient, dimension in zip(coefficients, grid, strict=True):
            index += coefficient * dimension
        if np.any(index < 0) or np.any(index >= value.shape[axis]):
            raise ValueError("source affine projection exceeds its exact input bounds")
        indices.append(index)
    return value[tuple(indices)].copy() if indices else np.broadcast_to(value, shape).copy()


def identity_map(mapping, shape):
    rank = len(shape)
    return mapping == {"loop_rank": rank, "coefficients": [[int(i == j) for j in range(rank)] for i in range(rank)], "offsets": [0] * rank}


def source_inputs(archive_path, provenance):
    """Read only declared source input tensors from a safetensors archive."""
    values = {}
    with Path(archive_path).open("rb") as stream:
        header_bytes = int.from_bytes(stream.read(8), "little")
        if not 0 < header_bytes < 1 << 24:
            raise ValueError("source tensor archive header is invalid")
        header = json.loads(stream.read(header_bytes))
        for key in provenance["input_keys"]:
            row, expected = header[key], provenance["tensors"][key]
            if row["dtype"] != "BF16" or expected["dtype"] != "torch.bfloat16" or row["shape"] != expected["shape"]:
                raise ValueError("source runtime input dtype or shape differs")
            begin, end = row["data_offsets"]
            stream.seek(8 + header_bytes + begin)
            raw = stream.read(end - begin)
            if len(raw) != prod(row["shape"]) * 2 or sha256(raw).hexdigest() != expected["sha256"]:
                raise ValueError("source runtime input byte identity differs")
            values[key] = np.frombuffer(raw, dtype="<u2").copy().reshape(row["shape"])
    return values


def check_attributes(node):
    op = node["op"]
    attrs = {k: v for k, v in node["attrs"].items() if not k.startswith("prov.")}
    if node["effect"] in {"input", "constant"} or op in {"reshape", "gather", "transpose"}:
        expected = {}
    elif op == "concat":
        expected = {"axis": attrs.get("axis")}
        if type(expected["axis"]) is not int or not 0 <= expected["axis"] < len(node["type"]["shape"]):
            raise ValueError("source concat axis differs")
    elif op in {"f32_add", "f32_sub", "f32_mul", "f32_div", "bf16_add", "bf16_sub", "bf16_mul"}:
        dtype, operation = op.split("_")
        expected = {"numeric_semantics": f"ieee754-{dtype}-{operation}-rne-v1"}
    elif op in {"f32_neg", "bf16_neg"}:
        expected = {"numeric_semantics": op.split("_")[0] + "-sign-bit-flip-v1"}
    elif op in {"bf16_to_f32", "f32_to_bf16"}:
        expected = {"numeric_semantics": "bf16-to-f32-exact-v1" if op == "bf16_to_f32" else "f32-to-bf16-rne-v1"}
    elif op in {"source_f32_exp_unqualified", "source_f32_rsqrt_unqualified"}:
        expected = {"numeric_semantics": "source-f32-" + op.split("_")[2] + "-unqualified-v1"}
    elif op.startswith("source_f32_reduce_"):
        maximum = op == "source_f32_reduce_max_unqualified"
        order_key = "scalar_operand_order" if maximum else "scalar_add_order"
        expected = {"axis": attrs.get("axis"), "reduction_order": "unresolved", order_key: attrs.get(order_key),
                    "numeric_semantics": "source-f32-maximum-unqualified-v1" if maximum else "source-f32-add-unqualified-v1"}
        if type(attrs.get("axis")) is not int or attrs.get(order_key) not in {"input-accumulator", "accumulator-input"}:
            raise ValueError("source reduction axis or scalar order differs")
    elif op.startswith("atlas.quantize_act."):
        allowed = {"code_dtype", "dequantization", "exponent_dtype", "mode", "reduce", "rows", "rule", "static_exp", "tag", "weight_format", "semantic_kind"}
        if set(attrs) - allowed or attrs.get("dequantization") != "e4m3(code)*2^exponent":
            raise ValueError("source quantizer has unknown numeric metadata")
        return
    elif op in {"atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul"}:
        expected = {"bias_order": "after_output_cast", "code_dtype": "uint8", "dequantization": "e4m3(code)*2^exponent",
                    "exponent_dtype": "int32", "output_dtype": "bfloat16" if node["type"]["dtype"] == "bf16" else "float32",
                    "output_rescale": "2^(lhs_exp+rhs_exp)", "rhs_transposed": "true", "weight_format": "atlas_e4m3",
                    "semantic_kind": "prequantized_contraction"}
        if "pop_shift" in attrs:
            expected["pop_shift"] = 0
    else:
        raise ValueError("source operation has no explicit independent arithmetic policy: " + op)
    if attrs != expected:
        raise ValueError("source numeric attributes differ for " + op)


class SourceInterpreter:
    def __init__(self, source, policy, *, engine="numpy", threads=1):
        for field, expected in POLICIES.items():
            if not selected_policy_matches(field, policy.get(field)):
                raise ValueError(f"selected source policy missing or changed: {field}")
        if not {"bf16_add", "bf16_sub", "bf16_neg", "f32_neg"} <= set(policy.get("host_float_arithmetic", ())):
            raise ValueError("selected exact host arithmetic is incomplete")
        if policy.get("source_sha256") != source.get("source_identity") or policy.get("target_identity") != source.get("target_identity"):
            raise ValueError("selected policy source or target identity differs")
        self.source, self.policy = source, policy
        self.engine, self.threads = engine, threads
        self.nodes = {node["id"]: node for node in source["nodes"]}
        if len(self.nodes) != len(source["nodes"]):
            raise ValueError("duplicate source node identity")
        self.bindings = {item["node_id"]: item for item in source["constants"]}
        self.env, self.quantizers, self.extra, self.overrides = {}, {}, {}, {}
        self.softmax_groups = self._softmax_groups()

    def _unreshape(self, key):
        chain = []
        while self.nodes[key]["op"] == "reshape":
            chain.append(key)
            key = self.nodes[key]["inputs"][0]
        return self.nodes[key], chain

    def _softmax_groups(self):
        groups = {}
        for node in self.source["nodes"]:
            if node["op"] != "f32_div":
                continue
            exp = self.nodes[node["inputs"][0]]
            if exp["op"] != "source_f32_exp_unqualified":
                continue
            summed, sum_views = self._unreshape(node["inputs"][1])
            if summed["op"] != "source_f32_reduce_add_unqualified" or summed["inputs"][0] != exp["id"]:
                continue
            shift = self.nodes[exp["inputs"][0]]
            if shift["op"] != "f32_sub":
                continue
            maximum, max_views = self._unreshape(shift["inputs"][1])
            if maximum["op"] != "source_f32_reduce_max_unqualified" or maximum["inputs"][0] != shift["inputs"][0]:
                continue
            added = self.nodes[shift["inputs"][0]]
            if added["op"] != "f32_add":
                raise ValueError("selected source softmax lacks an explicit mask addition")
            rank = len(node["type"]["shape"])
            identity = {"loop_rank": rank, "coefficients": [[int(i == j) for j in range(rank)] for i in range(rank)], "offsets": [0] * rank}
            projected = {"loop_rank": rank, "coefficients": identity["coefficients"][:-1], "offsets": [0] * (rank - 1)}
            broadcast = {"loop_rank": rank, "coefficients": [*identity["coefficients"][:-1], [0] * rank], "offsets": [0] * rank}
            if (maximum["attrs"].get("axis") != rank - 1 or summed["attrs"].get("axis") != rank - 1
                    or maximum["index_maps"] != [identity, projected, projected]
                    or summed["index_maps"] != [identity, projected, projected]
                    or added["index_maps"] != [identity, identity, identity]
                    or shift["index_maps"] != [identity, broadcast, identity]
                    or exp["index_maps"] != [identity, identity]
                    or node["index_maps"] != [identity, broadcast, identity]):
                raise ValueError("source normalized-softmax axes or maps differ")
            for reduction, seed_bits in ((maximum, 0xff800000), (summed, 0)):
                binding = self.bindings.get(reduction["inputs"][1])
                if (binding is None or binding["encoding"] != "f32-le-raw"
                        or any(int(word) != seed_bits for word in np.frombuffer(bytes.fromhex(binding["data_hex"]), dtype="<u4"))):
                    raise ValueError("source normalized-softmax initializer differs")
            internal = {added["id"], maximum["id"], shift["id"], exp["id"], summed["id"], *sum_views, *max_views}
            if any(key in self.source["outputs"] or any(key in consumer["inputs"] and consumer["id"] not in internal | {node["id"]}
                       for consumer in self.source["nodes"]) for key in internal):
                raise ValueError("source normalized-softmax intermediate escapes its private arithmetic region")
            groups[added["id"]] = {"add": added, "max": maximum, "sub": shift, "exp": exp, "sum": summed, "div": node}
        return groups

    def _softmax(self, group):
        added = group["add"]
        scores, mask = (self.env[key] for key in added["inputs"])
        if np.any((mask != 0) & (mask != 0xff7f0000)):
            raise ValueError("source softmax mask is not the declared BF16 minfinite/zero mapping")
        numbers = scores.view(np.float32)
        if np.any(~np.isfinite(numbers)) or np.any(np.abs(numbers) > 32):
            raise ValueError("source pre-mask scores leave the selected bounded mask mapping")
        softmax = attention_ref.normalized_softmax(narrow(scores), mask == 0)
        source_add = np.add(numbers, mask.view(np.float32), dtype=np.float32).view(np.uint32)
        values = {"add": source_add, "max": widen(softmax["row_max_bf16"]),
                  "sub": widen(softmax["shifted_bf16"]), "exp": widen(softmax["exp_bf16"]),
                  "sum": widen(softmax["denominator_bf16"]), "div": widen(softmax["probabilities_bf16"])}
        for role, value in values.items():
            node = group[role]
            self.overrides[node["id"]] = value.reshape(node["type"]["shape"])
        for name, value in softmax.items():
            self.extra[group["div"]["id"] + "." + name] = value

    def run(self, runtime_inputs):
        from atlas_ref.activation import quantize_act
        from atlas_ref.contractions import qlinear, contraction_nt

        ports = [node for node in self.source["nodes"] if node["effect"] == "input"]
        if len(ports) != len(runtime_inputs):
            raise ValueError("source runtime argument count differs")
        arguments = dict(zip((node["id"] for node in ports), runtime_inputs, strict=True))
        for node in self.source["nodes"]:
            check_attributes(node)
            key, op, type_ = node["id"], node["op"], node["type"]
            shape, dtype = tuple(type_["shape"]), type_["dtype"]
            attrs = {k: v for k, v in node["attrs"].items() if not k.startswith("prov.")}
            if dtype not in DTYPES or node["effect"] not in {"input", "constant", "pure"}:
                raise ValueError("unqualified source type or effect")
            if node["effect"] == "input":
                value = np.asarray(arguments[key])
            elif node["effect"] == "constant":
                binding = self.bindings[key]
                if binding["encoding"] != ENCODINGS[dtype]:
                    raise ValueError("source constant encoding differs from its tensor type")
                value = np.frombuffer(bytes.fromhex(binding["data_hex"]), dtype=DTYPES[dtype]).copy().reshape(shape)
            else:
                if key in self.softmax_groups:
                    self._softmax(self.softmax_groups[key])
                if key in self.overrides:
                    value = self.overrides[key]
                else:
                    inputs = [self.env[item] for item in node["inputs"]]
                    maps = node["index_maps"]
                    if op in {"reshape", "concat"}:
                        value = inputs[0].reshape(shape).copy() if op == "reshape" else np.concatenate(inputs, axis=attrs["axis"])
                    elif op in {"gather", "transpose"}:
                        if len(maps) != 2 or not identity_map(maps[-1], shape):
                            raise ValueError("source data movement lacks its exact logical result map")
                        value = project(inputs[0], maps[0], shape)
                    elif op.startswith("atlas.quantize_act."):
                        if attrs.get("code_dtype") != "uint8" or attrs.get("exponent_dtype") != "int32" or attrs.get("weight_format") != "atlas_e4m3":
                            raise ValueError("source activation code/exponent metadata differs")
                        cache_key = (node["inputs"][0], json.dumps(attrs, sort_keys=True))
                        if cache_key not in self.quantizers:
                            source_dtype = self.nodes[node["inputs"][0]]["type"]["dtype"]
                            if source_dtype not in {"bf16", "f32"}:
                                raise ValueError("source quantizer input is not declared floating data")
                            activation = inputs[0] if source_dtype == "bf16" else inputs[0].view(np.float32)
                            self.quantizers[cache_key] = quantize_act(activation, attrs["mode"], attrs["reduce"], attrs["rule"], attrs.get("static_exp"), attrs["rows"])
                        value = self.quantizers[cache_key][0 if op.endswith(".codes") else 1]
                    elif op in {"atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul"}:
                        if len(inputs) != 4 or attrs.get("weight_format") != "atlas_e4m3" or attrs.get("rhs_transposed") != "true" or attrs.get("output_rescale") != "2^(lhs_exp+rhs_exp)":
                            raise ValueError("source contraction semantics are not the selected unfolded NT contract")
                        result = (qlinear if op == "atlas.qlinear" else contraction_nt)(*inputs, unit="mxu1", engine=self.engine, threads=self.threads)
                        value = result.output_bf16 if dtype == "bf16" else widen(result.output_bf16)
                        self.extra[key + ".accumulator_bf16"] = result.accumulator_bf16
                        self.extra[key + ".readout_bf16"] = result.readout_bf16
                    elif op == "source_f32_reduce_add_unqualified":
                        value = selected_sum(*inputs, axis=attrs["axis"], reduction_policy=self.policy["reduction"])
                    else:
                        if len(maps) != len(inputs) + 1 or not identity_map(maps[-1], shape):
                            raise ValueError("source pointwise result map differs")
                        operands = [project(source, mapping, shape) for source, mapping in zip(inputs, maps[:-1], strict=True)]
                        if op == "bf16_to_f32":
                            value = widen(operands[0])
                        elif op == "f32_to_bf16":
                            value = narrow(operands[0])
                        elif op in {"bf16_neg", "f32_neg"}:
                            value = operands[0] ^ np.array(0x8000 if op == "bf16_neg" else 0x80000000, dtype=operands[0].dtype)
                        elif op in {"bf16_add", "bf16_sub"}:
                            value = host_bf16_add(*operands, subtract=op == "bf16_sub")
                        elif op == "source_f32_exp_unqualified":
                            value = selected_exp(operands[0])
                        else:
                            value = selected_pointwise(op, operands, source_bf16=dtype == "bf16")
            value = np.asarray(value)
            if value.shape != shape or value.dtype != np.dtype(DTYPES[dtype]):
                raise ValueError(f"selected source result ABI differs: {key}, {op}, {value.shape}, {value.dtype}")
            self.env[key] = value.copy()
        if any(self.env[key].tobytes() != bytes.fromhex(binding["data_hex"]) for key, binding in self.bindings.items()):
            raise ValueError("source constant bytes changed during independent replay")
        return self.env, self.extra


def authenticate_model(model_root):
    hashes = attention_ref.authenticate_model(model_root)
    name = "npu_model/configs/data/sqrt.bin.gz"
    raw = (Path(model_root) / name).read_bytes()
    pinned = subprocess.check_output(["git", "-C", str(model_root), "cat-file", "blob", f"{attention_ref.MODEL_PIN}:{name}"])
    if raw != pinned:
        raise ValueError("instruction-model square-root table differs from pin")
    return {**hashes, name: sha256(raw).hexdigest()}


def configure_engine(engine, threads, qualification=None):
    if not 1 <= threads <= 4:
        raise ValueError("reference threads must lie in [1,4]")
    if engine == "numpy":
        return {"engine": engine, "threads": threads}
    if engine != "native" or qualification is None:
        raise ValueError("native independent engine requires its explicit passing qualification")
    from atlas_ref import native
    raw = Path(qualification).read_bytes()
    evidence = json.loads(raw)
    root = Path(native.__file__).resolve().parent.parent
    if evidence.get("pass") is not True or evidence.get("mismatches") != 0 or any(
        case.get("mismatches") != 0 for case in evidence.get("benchmark_exactness", ())
    ):
        raise ValueError("native independent engine qualification is not passing")
    if set(evidence.get("reference_source_sha256", {})) != {"atlas_ref/native.py", "atlas_ref/native.c", "atlas_ref/arithmetic.py"}:
        raise ValueError("native qualification source inventory differs")
    for name, expected in evidence["reference_source_sha256"].items():
        if sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("native independent engine source differs from qualification")
    cache = Path(os.environ["TMPDIR"]).resolve()
    if not cache.is_relative_to(Path.cwd().resolve()):
        raise ValueError("native independent cache must remain inside this job")
    native._TMP = cache
    backend = native.backend_metadata()
    if backend != evidence["backend"]:
        raise ValueError("local native source/compiler/library identity differs from qualification")
    return {"engine": engine, "threads": threads, "backend": backend,
            "qualification_sha256": sha256(raw).hexdigest(), "cache": "invocation TMPDIR; reference source unchanged"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="passing independent ordinary-math and attention validation receipt")
    parser.add_argument("--engine", choices=("numpy", "native"), default="numpy")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--native-qualification", type=Path)
    args = parser.parse_args()
    before = authenticate_model(args.model_root)
    sys.path.insert(0, str(args.model_root.resolve()))
    sys.path.insert(0, str(args.reference_root.resolve()))
    import torch

    torch.set_num_threads(1)
    validation = None
    if args.validation:
        validation = json.loads(args.validation.read_text())
        if (validation.get("status") != "PASS" or validation.get("mismatches") != 0
                or validation.get("adapter_sha256") != sha256(Path(__file__).read_bytes()).hexdigest()
                or validation.get("attention_adapter_sha256") != sha256(Path(attention_ref.__file__).read_bytes()).hexdigest()
                or validation.get("model_arithmetic_sources") != before
                or validation.get("reference_sources") != attention_ref.reference_sources()
                or validation.get("selected_policies") != POLICIES):
            raise ValueError("layer validation differs from selected source arithmetic or adapter identity")
    provenance = json.loads((args.capsule / "source.json").read_text())
    if sha256((args.capsule / "model.pt2").read_bytes()).hexdigest() != provenance["model_sha256"]:
        raise ValueError("source exported model identity changed")
    inputs = source_inputs(args.capsule / "tensors.safetensors", provenance)
    source, policy = json.loads(args.source_request.read_text()), json.loads(args.policy.read_text())
    engine = configure_engine(args.engine, args.threads, args.native_qualification)
    interpreter = SourceInterpreter(source, policy, engine=args.engine, threads=args.threads)
    values, extra = interpreter.run([inputs[key] for key in provenance["input_keys"]])
    if authenticate_model(args.model_root) != before:
        raise ValueError("instruction-model source changed during reference execution")
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / "expected.npz", **values, **extra)
    info = lambda value: {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": sha256(value.tobytes()).hexdigest()}
    record = {"schema": "atlas.independent_selected_layer.v1", "status": "EXPECTED", "model_pin": attention_ref.MODEL_PIN,
              "model_arithmetic_sources": before, "source_model_sha256": provenance["model_sha256"],
              "source_request_sha256": sha256(args.source_request.read_bytes()).hexdigest(),
              "source_request_digest": sha256(json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
              "engine": engine,
              "selected_policy_sha256": sha256(args.policy.read_bytes()).hexdigest(), "selected_policies": {key: policy[key] for key in POLICIES},
              "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "attention_adapter_sha256": sha256(Path(attention_ref.__file__).read_bytes()).hexdigest(),
              "reference_sources": attention_ref.reference_sources(), "inputs": {k: info(v) for k, v in inputs.items()},
              "boundaries": {k: info(v) for k, v in {**values, **extra}.items()}, "outputs": source["outputs"],
              "constant_bytes_identical": True, "compiler_or_emitted_program_inputs": False,
              "whole_layer_device_exactness": "PENDING separate generated execution comparison",
              "validation": ({"status": "PASS", "sha256": sha256(args.validation.read_bytes()).hexdigest(),
                              "case_groups": len(validation["cases"]), "instructions": validation["instructions"],
                              "attention_validation_sha256": validation["attention_validation_sha256"]}
                             if validation else "PENDING separate handmade ordinary-math instruction checks")}
    (args.out / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "EXPECTED", "nodes": len(values), "outputs": {k: record["boundaries"][k] for k in source["outputs"]}}))


if __name__ == "__main__":
    main()
