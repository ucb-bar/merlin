"""Schema-shaped command-buffer emission with explicit Atlas DRAM addresses."""
import json


_DTYPE_BYTES = {"fp8_e4m3": 1, "fp8_e5m2": 1, "i8": 1, "bf16": 2, "f16": 2,
                "i16": 2, "f32": 4, "i32": 4, "i64": 8}


def _size(spec):
    count = 1
    for extent in spec.shape:
        count *= extent
    return count * _DTYPE_BYTES.get(spec.dtype, 4)


def build_command_buffer(workload):
    tensors = {}
    cursor = 0x90000000
    # Atlas' matrix accumulator is natively BF16 and its device ABI has no
    # four-byte vector-result store.  Preserve a logical f32 comparison for
    # BF16 contractions while declaring the actual two-byte device container;
    # the oracle promotes the captured values for the capsule comparison.
    bf16_device_result = bool(workload.tensors) and all(
        tensor.dtype == "bf16" for tensor in workload.tensors
    )
    native_vector_kinds = {"gelu", "silu", "softmax", "reduce_sum",
                           "layernorm", "depthwise_conv2d"}
    native_bf16_vector_graph = (
        bool(workload.ops)
        and all(item["op"] in native_vector_kinds for item in workload.ops)
        and all(tensor.dtype in ("bf16", "f32") for tensor in workload.tensors)
    )
    for spec in [*workload.tensors, *workload.outputs]:
        cursor = (cursor + 31) & ~31
        native_bf16_result = (
            spec.role == "output"
            and ((spec.dtype == "f32" and bf16_device_result)
                 or spec.dtype in ("fp8_e4m3", "fp8_e5m2")
                 or (spec.dtype == "f32" and native_bf16_vector_graph))
        )
        storage_dtype = "bf16" if native_bf16_result else spec.dtype
        tensors[spec.name] = {"shape": spec.shape, "dtype": storage_dtype, "role": spec.role, "base": cursor}
        size_spec = type("StorageSpec", (), {"shape": spec.shape, "dtype": storage_dtype})
        cursor += _size(size_spec)
    for item in workload.ops:
        if item["op"] in ("geglu", "k_chain", "attention_full"):
            out = tensors[item["dst"]]
            cursor = (cursor + 31) & ~31
            temp_name = {"geglu": "__tmp_geglu", "k_chain": "__tmp_chain",
                         "attention_full": "__tmp_attention"}[item["op"]]
            temp_shape = [out["shape"][-2], out["shape"][-2]] \
                if item["op"] == "attention_full" else list(out["shape"])
            tensors[temp_name] = {
                "shape": temp_shape, "dtype": "bf16",
                "role": "intermediate", "base": cursor,
            }
            cursor += _size(type("StorageSpec", (), {
                "shape": out["shape"], "dtype": "bf16"}))
    if native_bf16_vector_graph:
        for tensor in workload.tensors:
            if tensor.dtype != "f32":
                continue
            cursor = (cursor + 31) & ~31
            tensors["__bf16_" + tensor.name] = {
                "shape": list(tensor.shape), "dtype": "bf16",
                "role": "intermediate", "base": cursor,
            }
            cursor += _size(type("StorageSpec", (), {
                "shape": tensor.shape, "dtype": "bf16"}))

    # A direct batched interface may spell its result as the equivalent
    # flattened [batch*M, N] tensor. Commands carry a batch axis, so retain
    # that rank at the host/device boundary whenever the element count proves
    # the two views are identical. Allocation and physical offsets do not
    # change.
    for item in workload.ops:
        if item["op"] not in ("gemv_batched", "matmul_batched"):
            continue
        inputs = item.get("inputs", [])
        if len(inputs) < 2 or item["dst"] not in tensors:
            continue
        lhs, rhs = tensors[inputs[0]], tensors[inputs[1]]
        if len(lhs["shape"]) != 3 or len(rhs["shape"]) != 3:
            continue
        batched_shape = [lhs["shape"][0], lhs["shape"][-2],
                         rhs["shape"][-1]]
        old_count = 1
        new_count = 1
        for extent in tensors[item["dst"]]["shape"]:
            old_count *= extent
        for extent in batched_shape:
            new_count *= extent
        if old_count == new_count:
            tensors[item["dst"]]["shape"] = batched_shape
    commands = []
    declined = None
    for item in workload.ops:
        kind = item["op"]
        if kind == "res_pack":
            commands.append({"opcode": "RES_PACK", "operands": {"src": item["src"], "dst": item["dst"]},
                             "attributes": {"layout": item["attrs"].get("layout", "packed_rhs")}})
        elif kind == "matmul":
            commands.append({"opcode": "MATMUL_RESIDENT", "operands": {"lhs": item["lhs"], "rhs": item["rhs"], "dst": item["dst"]}})
        elif kind == "commit":
            attrs = dict(item["attrs"])
            attrs.pop("name", None)
            commands.append({"opcode": "COMMIT", "operands": {"src": item["src"], "dst": item["dst"]}, "attributes": attrs})
        elif kind == "evict":
            commands.append({"opcode": "EVICT", "operands": {"handle": item["handle"]}})
        elif kind == "movement":
            commands.append({"opcode": "MOVEMENT", "operands": {"src": item["inputs"][0], "dst": item["dst"]}, "attributes": item["attrs"]})
        else:
            if kind == "model":
                declined = {"op": "model", "reason": "whole-model f32 graph exceeds the current Atlas device lowering"}
                continue
            if kind == "layernorm" and len(item.get("inputs", [])) >= 3:
                # Keep both affine operands live in the command shape. The
                # whole-program ABI already lists them; naming the bias here
                # prevents shape binding from treating it as unrelated data.
                commands.append({
                    "opcode": "LAYERNORM",
                    "operands": {"src": item["inputs"][0],
                                 "rhs": item["inputs"][1],
                                 "bias": item["inputs"][2],
                                 "dst": item["dst"]},
                    "attributes": item.get("attrs", {}),
                })
                continue
            if (kind == "add" and len(item.get("inputs", [])) > 1
                    and len(tensors[item["inputs"][1]]["shape"]) == 1):
                commands.append({
                    "opcode": "BIAS_ADD",
                    "operands": {"src": item["inputs"][0],
                                 "bias": item["inputs"][1],
                                 "dst": item["dst"]},
                    "attributes": item.get("attrs", {}),
                })
                continue
            if kind == "geglu":
                commands.append({
                    "opcode": "GEGLU",
                    "operands": {"src": item["inputs"][0],
                                 "weight": item["inputs"][1],
                                 "weight2": item["inputs"][2],
                                 "dst": item["dst"]},
                    "attributes": item.get("attrs", {}),
                })
                continue
            if kind == "attention_full":
                commands.append({
                    "opcode": "ATTENTION_FULL",
                    "operands": {"query": item["inputs"][0],
                                 "key": item["inputs"][1],
                                 "value": item["inputs"][2],
                                 "dst": item["dst"]},
                    "attributes": item.get("attrs", {}),
                })
                continue
            if (kind in ("gemv_batched", "matmul_batched")
                    and tensors[item["inputs"][1]]["role"] == "weight"):
                # Direct batched interfaces have no explicit resident_pack op,
                # but a role=weight operand still needs a host-visible staging
                # declaration.  The generated kernel reads the same source
                # tensor directly; this command only makes its binding live.
                commands.append({
                    "opcode": "RES_PACK",
                    "operands": {"src": item["inputs"][1],
                                 "dst": item["inputs"][1] + "_resident"},
                    "attributes": {"layout": "packed_rhs"},
                })
            opcode = {"add": "VECTOR_MAP", "silu": "VECTOR_MAP", "reduce_sum": "VREDUCE",
                      "gemv_batched": "BATCHED_MATMUL",
                      "matmul_batched": "BATCHED_MATMUL",
                      "fused_matmul_bias": "K_CHAIN"}.get(kind, kind.upper())
            operands = {"src": item["inputs"][0], "dst": item["dst"]} if item.get("inputs") else {"dst": item["dst"]}
            if len(item.get("inputs", [])) > 1:
                operands["rhs"] = item["inputs"][1]
            commands.append({"opcode": opcode, "operands": operands, "attributes": item.get("attrs", {})})
    cb = {"abi_version": "0.1", "target": "atlas", "backend": "atlas-xdsl", "tensors": tensors, "commands": commands}
    if declined is not None:
        cb["commands"] = []
        cb["declined"] = declined
    # Make the host/device binding explicit for every accepted interface
    # grammar.  In particular, direct merlin_iface programs have no enclosing
    # function signature from which the runner could otherwise recover the
    # output argument slot.
    args = [{"tensor": t.name, "access": "read"} for t in workload.tensors]
    args += [{"tensor": t.name, "access": "write"} for t in workload.outputs]
    args += [{"tensor": name, "access": "readwrite"}
             for name, spec in tensors.items()
             if spec["role"] == "intermediate"]
    cb["kernel_abi"] = {"kind": "whole_program", "args": args,
                        "outputs": [t.name for t in workload.outputs]}
    return cb


def write_command_buffer(workload, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(build_command_buffer(workload), handle, indent=2)
