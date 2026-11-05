"""Export captured full Tiny arithmetic to shared immutable Atlas assets.

This assembles instructions and packs operands without constructing a device
Core. Expected words remain arithmetic references until genuine execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked as physical_checked
from atlas_tooling import sha256
from compile_workload_inputs import pack_activation, pack_weights, pack_output
import atlas_host_layout
from atlas_host_layout import ATLAS_DRAM_BASE
from atlas_program import assembly, dma_load_bytes
from atlas_tile_arithmetic import tensor_scale_exponent
from run_workload_contraction import resource_plan, admit_program
from run_model_tile import configure_model, model_binding


def checked(value):
    path = Path(value)
    return physical_checked(path)


class Assets:
    def __init__(self, root):
        self.root = root
        self.sums = {}

    def put(self, kind, data, suffix=".bin"):
        digest = hashlib.sha256(data).hexdigest()
        relative = f"{kind}/{digest}{suffix}"
        destination = self.root / relative
        if relative in self.sums:
            if destination.read_bytes() != data:
                raise ValueError("content-address collision")
        else:
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(data)
            self.sums[relative] = digest
        return relative


def read_array(root, descriptor):
    name = descriptor["file"]
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("capture payload must be relative")
    path = checked(root / name)
    raw = path.read_bytes()
    dtype = np.dtype(descriptor["dtype"])
    shape = tuple(descriptor["shape"])
    if (dtype not in (np.dtype("uint16"), np.dtype("float32"), np.dtype("int64"))
            or any(type(n) is not int or n <= 0 for n in shape)
            or len(raw) != descriptor["bytes"] or len(raw) != int(np.prod(shape)) * dtype.itemsize
            or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]):
        raise ValueError("capture array identity changed")
    return np.frombuffer(raw, dtype=dtype).reshape(shape), path


def float_words(words):
    if words.dtype != np.uint16:
        raise ValueError("BF16 operand words required")
    return (words.astype(np.uint32) << 16).view(np.float32)


class ExactEncoder:
    """Validate every distinct actual BF16 word/scale against selected RTL."""
    def __init__(self, codec, torch, to_fp8):
        self.codec, self.torch, self.to_fp8 = codec, torch, to_fp8
        self.verified = {}

    def encode(self, words):
        packed, scale = self.codec.encode(words)
        maximum = int((words & np.uint16(0x7fff)).max())
        if tensor_scale_exponent([maximum]) != scale:
            raise ValueError("vector tensor scale differs from selected scalar policy")
        seen = self.verified.setdefault(scale, set())
        missing = np.array([int(word) for word in np.unique(words) if int(word) not in seen], dtype=np.uint16)
        if missing.size:
            tensor = self.torch.from_numpy(missing.copy()).view(self.torch.bfloat16)
            actual = self.to_fp8(tensor, 127 + scale).view(self.torch.uint8).numpy()
            expected = self.codec.pack_words(missing.reshape(1, -1), scale).reshape(-1)
            if not np.array_equal(actual, expected):
                raise ValueError("vector FP8 bytes differ from selected RtlNumerics")
            seen.update(int(word) for word in missing)
        return packed, scale


def prepare_encoded(a, b, exponent, assemble):
    """Use unchanged shared packing/resource/assembly admission with exact bytes."""
    m, k = a.shape
    if b.ndim != 2 or b.shape[0] != k:
        raise ValueError("encoded contraction dimensions disagree")
    n = b.shape[1]
    plan = resource_plan(m, k, n)
    if not -126 <= exponent <= 127:
        raise ValueError("VPU rescale factor must be normal finite BF16")
    activation, weights = pack_activation(a), pack_weights(b)
    inputs = activation + weights
    output_bytes, offset = m * n * 2, len(inputs) + 256
    initial = inputs + b"\xa5" * 256 + b"\xcd" * output_bytes + b"\x5a" * 256
    source = assembly(m, k, n, offset, exponent, resident_b=plan["resident_b"],
                      pretranspose_resident_b=plan["resident_b"], resident_a=plan["resident_a"])
    words = list(assemble(io.StringIO(source)).assemble())
    admit_program(words)
    loads = dma_load_bytes(m, k, n, resident_a=plan["resident_a"], resident_b=plan["resident_b"])
    return {"source": source, "program": struct.pack(f"<{len(words)}I", *words), "initial": initial,
            "activation": activation, "weights": weights,
            "memory": {"dram_base": ATLAS_DRAM_BASE, "output_offset": offset, "total_dram_bytes": len(initial)},
            "requested_dma": {"load_bytes": loads, "store_bytes": output_bytes}}


def checked_layout():
    header = checked(Path(__file__).absolute().parents[2] / "runtime/atlas_host_layout.h")
    if checked(atlas_host_layout.HEADER) != header:
        raise ValueError("host layout imported from another header")
    atlas_host_layout.recheck()
    return header


def attention_execution_mode(report):
    """Admit the explicit execution mode and its complete attention ledger."""
    mode = report.get("atlas_instruction_execution", False)
    calls = report.get("attention_calls", [])
    if type(mode) is not bool or not isinstance(calls, list):
        raise ValueError("invalid attention execution mode or ledger")
    if not mode:
        if calls:
            raise ValueError("attention calls contradict disabled instruction execution")
        return False
    layers, decode = report.get("layer_count", 22), report.get("decode_count")
    if (type(layers) is not int or not 1 <= layers <= 22 or type(decode) is not int or not 0 <= decode <= 32
            or not calls or len(calls) != layers * (decode + 1)):
        raise ValueError("attention instruction execution requires complete nonempty layer/step coverage")
    for index, call in enumerate(calls):
        step, layer = divmod(index, layers)
        if (not isinstance(call, dict) or any(type(call.get(name)) is not int for name in ("index", "step", "layer"))
                or (call["index"], call["step"], call["layer"]) != (index, step, layer)):
            raise ValueError("attention capture invocation order changed")
    return True


def export(args):
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    capture = checked(args.capture)
    if sha256(capture) != args.capture_sha256:
        raise ValueError("capture receipt changed")
    report = json.loads(capture.read_text())
    if attention_execution_mode(report):
        from attention_export import export_attention
        return export_attention(args)
    if (report.get("status") != "PASS_FULL_TINY_ARITHMETIC_CAPTURE_REPLAY"
            or report.get("source_bookends_verified") is not True
            or report.get("target_replay_exact") is not True
            or report.get("greedy_feedback_actual") is not True
            or report.get("all22_cache_prefixes_exact") is not True
            or report.get("native_restored_exact") is not True):
        raise ValueError("complete full Tiny arithmetic capture required")
    count = report["decode_count"]
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError("capture decode extent")
    out = checked(args.out)
    if out.exists():
        raise ValueError("fresh full bundle required")
    pins = {str(capture): args.capture_sha256}
    for record in report["source_records"]:
        path = checked(record["path"])
        digest = sha256(path)
        if digest != record["sha256"] or (str(path) in pins and pins[str(path)] != digest):
            raise ValueError("capture source changed")
        pins[str(path)] = digest
    model_root = checked(args.model_root)
    before = model_binding(model_root)
    _, torch, assemble, _ = configure_model(model_root)
    binding = model_binding(model_root)
    if (any(before[key] != binding[key] for key in ("revision", "tracked_source_diff_exit", "curve_sha256"))
            or any(binding["imported_source_sha256"].get(path) != digest
                   for path, digest in before["imported_source_sha256"].items())):
        raise ValueError("model changed during configuration")
    from npu_model.configs.numerics import RtlNumerics
    binding = model_binding(model_root)
    for name in ("export_capture.py",):
        path = checked(Path(__file__).with_name(name))
        pins[str(path)] = sha256(path)
    for name in ("compile_workload_inputs.py", "run_model_tile.py", "run_workload_contraction.py",
                 "atlas_program.py", "atlas_path_guard.py", "atlas_tooling.py", "atlas_tile_arithmetic.py",
                 "atlas_vector_arithmetic.py", "atlas_host_layout.py"):
        path = checked(kernel_library.source_path(name))
        module = sys.modules.get(Path(name).stem)
        if module is None or checked(module.__file__) != path:
            raise ValueError("export helper imported from another source")
        digest = sha256(path)
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("conflicting export helper source pin")
        pins[str(path)] = digest
    header = checked_layout()
    pins[str(header)] = sha256(header)
    codec_paths = [checked(row["path"]) for row in report["source_records"]
                   if Path(row["path"]).name == "tiny_fp8_pack.py"]
    if len(codec_paths) != 1:
        raise ValueError("one capture-bound vector FP8 codec required")
    spec = importlib.util.spec_from_file_location("tiny_capture_fp8_codec", codec_paths[0])
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    encoder = ExactEncoder(codec, torch, RtlNumerics.to_fp8)
    out.mkdir(parents=True)
    assets = Assets(out)
    kernels, host, references = [], {}, []

    def read(desc):
        value, path = read_array(capture.parent, desc)
        digest = sha256(path)
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("conflicting capture payload pin")
        pins[str(path)] = digest
        return value

    for name, desc in report["host_parameters"].items():
        words = read(desc)
        host[name] = {**desc, "file": assets.put("host-data", words.tobytes())}
    for row in report["target_trace"]:
        ref = {"step": row["step"], "next_token": row["next_token"], "input_tokens": row["input_tokens"],
               "logits": {**row["logits_payload"],
                          "file": assets.put("references", read(row["logits_payload"]).tobytes())}, "cache": []}
        for desc in row["cache"]:
            ref["cache"].append({**desc, "file": assets.put("references", read(desc).tobytes())})
        references.append(ref)
    for call_index, call in enumerate(report["calls"]):
        step, index = divmod(call_index, 155)
        if (step > count or call["step"] != step or call["site_index"] != index
                or call["site"] != ([f"model.layers.{layer}.{site}" for layer in range(22)
                    for site in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")] + ["lm_head"])[index]):
            raise ValueError("capture invocation order changed")
        a = read(call["A_bf16"])
        b = read(report["parameters"][call["site"]]["B_bf16"])
        rows = 128 if step == 0 else 1
        if a.ndim != 2 or b.ndim != 2 or a.shape != (rows, b.shape[0]):
            raise ValueError("logical source geometry")
        physical_a = a
        if rows == 1:
            physical_a = np.concatenate((physical_a, np.zeros((31, a.shape[1]), np.uint16)))
        resource_plan(physical_a.shape[0], a.shape[1], 32)
        encoded_a, a_scale = encoder.encode(physical_a)
        column = 0
        for panel in call["panels"]:
            lo, hi = panel["columns"]
            expected = read(panel["expected_bf16"])
            m, k, n = panel["physical_mkn"]
            if (lo != column or hi <= lo or n != hi-lo or k != a.shape[1] or m != physical_a.shape[0]
                    or expected.shape != (m, n) or panel["logical_mkn"] != [rows, k, n]):
                raise ValueError("physical panel coverage")
            resource_plan(m, k, n)
            encoded_b, b_scale = encoder.encode(np.ascontiguousarray(b[:, lo:hi]))
            if (a_scale + b_scale != panel["rescale"]
                    or a_scale != panel["A_scale"] or b_scale != panel["B_scale"]):
                raise ValueError("captured integer-codec scales differ from producer")
            prepared = prepare_encoded(encoded_a, encoded_b, a_scale + b_scale, assemble)
            memory = prepared["memory"]
            kernels.append({"index": len(kernels), "step": step, "site": call["site"],
                "site_index": index, "columns": [lo, hi], "logical_rows": rows, "physical_mkn": [m, k, n],
                "A_scale": panel["A_scale"], "B_scale": panel["B_scale"], "rescale": panel["rescale"],
                "source": assets.put("programs", prepared["source"].encode(), ".S"),
                "original": assets.put("programs", prepared["program"]),
                "activation": assets.put("activations", prepared["activation"]),
                "weights": assets.put("weights", prepared["weights"]),
                "expected": assets.put("expected", pack_output(expected)),
                "initial_sha256": hashlib.sha256(prepared["initial"]).hexdigest(),
                "output_offset": memory["output_offset"], "total_dram_bytes": memory["total_dram_bytes"],
                "dram_base": memory["dram_base"], "requested_dma": prepared["requested_dma"]})
            column = hi
        if column != b.shape[1]:
            raise ValueError("incomplete output columns")
    if len(report["calls"]) != 155 * (count + 1) or len(kernels) != 280 * (count + 1):
        raise ValueError("full model launch count")
    for index, name in enumerate(sorted(host)):
        host[name]["symbol"] = f"tiny_full_param_{index:03d}"
    buffers = []
    for row in references:
        step = row["step"]
        for item in row["cache"]:
            name = f"tiny_full_cache_{step:02d}_{item['layer']:02d}_{item['field']}"
            buffers.append({**item, "actual_symbol": name, "expected_symbol": name + "_expected"})
        name = f"tiny_full_logits_{step:02d}"
        buffers.append({**row["logits"], "actual_symbol": name, "expected_symbol": name + "_expected"})
    tokens = [row["next_token"] for row in references]
    consumed = [row["input_tokens"][0][0] for row in references[1:]]
    if len(tokens) != count + 1 or consumed != tokens[:-1]:
        raise ValueError("actual greedy feedback trace differs")
    for name, values in (("tiny_full_greedy_tokens", tokens), ("tiny_full_decode_inputs", consumed)):
        words = np.array(values, dtype="<u4")
        raw = words.tobytes()
        buffers.append({"actual_symbol": name, "expected_symbol": name + "_expected",
                        "file": assets.put("references", raw), "shape": list(words.shape), "dtype": "uint32",
                        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    contract = {"schema": "atlas.tiny_full_contract.v1", "decode_count": count,
                "kernel_count": len(kernels), "projection_calls": 155 * (count + 1),
                "cycle_cap": 200000000, "kernels": kernels, "host_parameters": host,
                "reference_buffers": buffers, "greedy_tokens": tokens,
                "mailbox_symbol": "tiny_full_mailbox", "panel_symbol": "tiny_full_panels",
                "state_symbol": "tiny_full_state_global", "context_symbol": "tiny_full_projection_state"}
    contract_raw = json.dumps(contract, indent=2).encode() + b"\n"
    (out / "contract.json").write_bytes(contract_raw)
    assets.sums["contract.json"] = hashlib.sha256(contract_raw).hexdigest()
    provenance = {"schema": "atlas.full_bundle.v1", "workload": "tiny-full", "decode_count": count,
        "kernel_count": len(kernels), "kernels": kernels, "host_parameters": host, "references": references,
        "contract": "contract.json", "contract_sha256": assets.sums["contract.json"],
        "capture": {"path": str(capture), "sha256": args.capture_sha256}, "model": binding,
        "initial_layout": "activation+weights+a5*256+cd*output_bytes+5a*256",
        "atlas_instruction_execution": False, "source_sha256": pins}
    if model_binding(model_root) != binding or any(sha256(checked(path)) != digest for path, digest in pins.items()):
        raise ValueError("export source bookend changed")
    data = json.dumps(provenance, indent=2).encode() + b"\n"
    (out / "provenance.json").write_bytes(data)
    assets.sums["provenance.json"] = hashlib.sha256(data).hexdigest()
    stage = {**provenance, "schema": "atlas.full_kernel_assets.v1"}
    stage_raw = json.dumps(stage, indent=2).encode() + b"\n"
    (out / "kernels.json").write_bytes(stage_raw)
    assets.sums["kernels.json"] = hashlib.sha256(stage_raw).hexdigest()
    (out / "SHA256SUMS").write_text("".join(f"{digest}  {path}\n" for path, digest in sorted(assets.sums.items())))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "model-root", "out"):
        parser.add_argument("--" + name, type=checked, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--capture-source-archive", type=checked, help="explicit hash-pinned copies of changed capture sources")
    parser.add_argument("--native-proof-script", type=checked, help="isolated admitted-native provenance provider")
    parser.add_argument("--native-proof-sha256", help="required identity of the native provenance provider")
    parser.add_argument("--native-proof-python", help="venv interpreter for the isolated native provenance checker")
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW",
                        help="explicit physical-path rebinding for attention capture sources")
    parser.add_argument("--qualify-launches", action="store_true", help="execute every v2 launch with selected RTL numerics")
    parser.add_argument("--dram-base", type=lambda value: int(value, 0), default=0x80000000,
                        help="attention platform physical DRAM base (default: 0x80000000)")
    parser.add_argument("--dram-bytes", type=lambda value: int(value, 0),
                        help="attention platform physical DRAM size, required for attention exports")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
