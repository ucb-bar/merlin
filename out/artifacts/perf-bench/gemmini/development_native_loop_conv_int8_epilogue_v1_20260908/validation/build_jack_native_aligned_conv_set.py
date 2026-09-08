#!/usr/bin/env python3
"""Convert and object-build all 53 native-aligned conv sites from Jack's TVM reference.

This consumes the independently produced TVM graph's concrete convolution calls and quantization
receipt, but every convolution instruction stream and object is emitted by this copied Merlin
compiler.  The result is a complete convolution object set, not yet a linked ResNet graph.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
ARTIFACT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ARTIFACT / "compiler"), str(ROOT / "merlin" / "python")]
os.environ["MERLIN_TARGET_PATH"] = str(ARTIFACT / "runtime_target")

from xdsl.printer import Printer
from mlir_oot.codegen import llvm_emit
from mlir_oot.gemmini_opt import Pipeline
from merlin.targetgen.contract.compile import llvm_mlir_to_object


JACK = Path("/scratch/jack/experiments/universal_resnet50_gemmini_20260906")
TVM = JACK / "builds/tvm/compile_coherence_final_20260907"
REFERENCE_JSON = JACK / "builds/tvm/compile_coherence_final_20260907_artifact.json"
REFERENCE_C = TVM / "project/src/model/default_lib1.c"
INVENTORY = (ROOT / "out/artifacts/perf-bench/gemmini/"
             "universal_narrow_epilogue_recovery_v1_20260908/"
             "resnet50_narrow_epilogue_inventory.json")


def balanced_call(text: str, start: int) -> tuple[str, int]:
    open_at = text.index("(", start)
    depth = 0
    for pos in range(open_at, len(text)):
        if text[pos] == "(":
            depth += 1
        elif text[pos] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:pos], pos + 1
    raise ValueError("unterminated tiled_conv_auto call")


def split_args(payload: str) -> list[str]:
    parts, depth, begin = [], 0, 0
    for pos, char in enumerate(payload):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(payload[begin:pos].strip())
            begin = pos + 1
    parts.append(payload[begin:].strip())
    return parts


def calls() -> list[dict]:
    text = REFERENCE_C.read_text()
    result = {}
    for match in re.finditer(r"\btiled_conv_auto\s*\(", text):
        name_at = text.rfind("TVM_DLL int32_t ", 0, match.start())
        name_match = re.match(
            r"TVM_DLL int32_t (tvmgen_default_fused_contrib_gemmini_conv2d(?:_(\d+))?)\(",
            text[name_at:])
        if name_match is None:
            raise ValueError("convolution call is not enclosed by the expected TVM function")
        index = int(name_match.group(2) or 0)
        args = split_args(balanced_call(text, match.start())[0])
        if len(args) != 27:
            raise ValueError(f"conv2d_{index}: expected 27 tiled_conv_auto args, got {len(args)}")
        result[index] = {
            "index": index,
            "geometry": dict(n=int(args[0]), hi=int(args[1]), wi=int(args[2]),
                             ci=int(args[3]), co=int(args[4]), ho=int(args[5]),
                             wo=int(args[6]), stride=int(args[7]),
                             input_dilation=int(args[8]), kernel_dilation=int(args[9]),
                             padding=int(args[10]), kh=int(args[11]), kw=int(args[11])),
            "relu": int(args[21]) == 1,
            "acc_scale_c_literal": args[22],
            "acc_scale": float(args[22].removesuffix("f")),
            "pool_size": int(args[23]), "pool_stride": int(args[24]),
            "pool_padding": int(args[25]),
        }
    if sorted(result) != list(range(53)):
        raise ValueError(f"expected conv2d_0..52, got {sorted(result)}")
    return [result[i] for i in range(53)]


def interface(call: dict, scale: float) -> str:
    g = call["geometry"]
    k = g["kh"] * g["kw"] * g["ci"]
    rows = g["n"] * g["ho"] * g["wo"]
    stages = ["bias"] + (["relu"] if call["relu"] else []) + ["acc_scale"]
    stages_text = ", ".join(f'"{stage}"' for stage in stages)
    return f'''module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"}} {{
  %IFM = merlin_iface.tensor {{name = "IFM", role = "input"}} : tensor<{g['n']}x{g['hi']}x{g['wi']}x{g['ci']}xi8>
  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{k}x{g['co']}xi8>
  %B = merlin_iface.tensor {{name = "B", role = "bias"}} : tensor<{g['co']}xi32>
  %R = merlin_iface.resident_pack %W {{layout = "packed_conv_rhs"}} : (tensor<{k}x{g['co']}xi8>) -> !merlin_iface.resident
  %Y = merlin_iface.conv2d %IFM, %R {{kernel = [{g['kh']}, {g['kw']}, {g['ci']}, {g['co']}], stride = [{g['stride']}, {g['stride']}], padding = [{g['padding']}, {g['padding']}, {g['padding']}, {g['padding']}], dilation = [{g['kernel_dilation']}, {g['kernel_dilation']}], name = "Y", epilogue = [{stages_text}], bias = "B", acc_scale = {scale!r} : f32, output_dtype = "i8", layout = "nhwc"}} : (tensor<{g['n']}x{g['hi']}x{g['wi']}x{g['ci']}xi8>, !merlin_iface.resident) -> tensor<{rows}x{g['co']}xi8>
  merlin_iface.evict %R : (!merlin_iface.resident) -> ()
}}
'''


def main() -> int:
    inv = json.loads(INVENTORY.read_text())["convolutions"]
    qdoc = json.loads(REFERENCE_JSON.read_text())["quantization"]
    qlayers = {int(layer["op"].split("_")[-1]): layer for layer in qdoc["layers"]
               if layer["op"].startswith("conv2d_")}
    root = ARTIFACT / "validation/native_aligned_resnet50_layer_build"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for call in calls():
        index, g = call["index"], call["geometry"]
        oldg = inv[index]["geometry"]
        for key in ("n", "hi", "wi", "ci", "co", "ho", "wo", "kh", "kw"):
            if g[key] != oldg[key]:
                raise ValueError(f"conv2d_{index}: TVM {key}={g[key]} != prepared {oldg[key]}")
        scale = float(qlayers[index]["accumulator_scale"])
        if abs(scale - call["acc_scale"]) > 5e-8:
            raise ValueError(f"conv2d_{index}: C/receipt scale mismatch")
        source = interface(call, scale)
        pipe = Pipeline(source).run()
        if pipe.declined is not None:
            raise RuntimeError(f"conv2d_{index}: {pipe.declined}")
        module = llvm_emit.emit(pipe.plan, pipe.instrs, pipe.staging)
        stream = io.StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        lowered = stream.getvalue() + "\n"
        work = root / f"conv2d_{index:02d}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "native_aligned_iface.mlir").write_text(source)
        (work / "target.mlir").write_text(lowered)
        (work / "command_buffer.json").write_text(
            json.dumps(pipe.plan.command_buffer, indent=2, sort_keys=True) + "\n")
        obj = llvm_mlir_to_object(lowered, work, target="gemmini")
        rows.append({
            **call, "fqn": inv[index]["fqn"], "acc_scale": scale,
            "merlin_native_selected": True,
            "loop_conv_descriptors": sum(i.kind == "loop_conv_ws" for i in pipe.instrs),
            "load3_bias_descriptors": sum(
                i.kind == "loop_conv_ws" and not i.attrs["no_bias"] for i in pipe.instrs),
            "object": str(obj),
            "post_conv_pool_split_required": call["pool_stride"] != 0,
        })
    receipt = {
        "schema": "jack_tvm_native_aligned_to_merlin_conv_object_set_v1",
        "status": "passed" if len(rows) == 53 else "failed",
        "reference": {
            "compiler_stack": "Jack TVM Relay quantize independent reference",
            "quantization": {
                "activation_dtype": qdoc["activation_dtype"],
                "weight_dtype": qdoc["weight_dtype"],
                "accumulator_dtype": qdoc["accumulator_dtype"],
                "weight_scale_method": qdoc["weight_scale_method"],
                "global_scale": qdoc["global_scale"],
            },
            "reference_artifact": str(REFERENCE_JSON),
            "reference_c": str(REFERENCE_C),
        },
        "conv_objects_built": len(rows),
        "native_selected": sum(r["merlin_native_selected"] for r in rows),
        "loop_conv_descriptors": sum(r["loop_conv_descriptors"] for r in rows),
        "load3_bias_descriptors": sum(r["load3_bias_descriptors"] for r in rows),
        "post_conv_pool_splits": sum(r["post_conv_pool_split_required"] for r in rows),
        "qualification": "all_convolution_objects_built_not_linked_full_graph_not_spike_correctness",
        "remaining_full_graph_work": {
            "residual_rescale_add_relu": 20,
            "maxpool_split": sum(r["post_conv_pool_split_required"] for r in rows),
            "global_avgpool_flatten_dense": 1,
            "activation_arena_and_graph_abi": 1,
        },
        "firesim": "not_run",
        "layers": rows,
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "status", "conv_objects_built", "native_selected", "loop_conv_descriptors",
        "load3_bias_descriptors", "post_conv_pool_splits", "qualification")}, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
