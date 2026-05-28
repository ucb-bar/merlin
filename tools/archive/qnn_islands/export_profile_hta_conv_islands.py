#!/usr/bin/env python3
"""Export and profile real HTA conv islands from dumped yolov8 dispatches.

This is the durable version of the manual dispatch_1 bring-up:

* Parse per-dispatch HAL dump MLIR with `iree.compiler.ir`.
* Find int8 conv + first requant tail inside each dispatch.
* Read bias/weight bytes from the real constant arena using the dispatch
  binding offsets.
* Emit a channel-last HWC HTA wrapper with static bias/weight and explicit
  input/output buffer-view arguments.
* Optionally compile the wrappers and profile them on the QRB board.

No timing is synthesized. Profiling requires a captured, real HWC UFIXED8
input file for the source dispatch; islands without captures are exported
but skipped during profiling.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
import re
import shlex
import struct
import subprocess
from collections.abc import Iterable
from typing import Any

from iree.compiler import ir

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SSH_IDENTITY = REPO_ROOT.parent / "DIMA_SLICE"
BENCH_RE = re.compile(
    r"setup_ms=(?P<setup>\S+) iters=\d+ warmup=\d+ "
    r"mean_ms=(?P<mean>\S+) median_ms=(?P<median>\S+) "
    r"p99_ms=(?P<p99>\S+) min_ms=(?P<min>\S+) max_ms=(?P<max>\S+)"
)


@dataclasses.dataclass(frozen=True)
class TensorValue:
    shape: tuple[int, ...]
    elem: str
    source_subspan: str | None = None


@dataclasses.dataclass(frozen=True)
class Subspan:
    name: str
    binding: int
    offset: int
    descriptor_flags: int
    type_text: str


@dataclasses.dataclass(frozen=True)
class ConvIsland:
    source_key: str
    source_dispatch: str
    func_name: str
    input_shape_chw: tuple[int, int, int]
    input_shape_hwc: tuple[int, int, int]
    weight_shape_fchw: tuple[int, int, int, int]
    weight_shape_hwcf: tuple[int, int, int, int]
    output_shape_chw: tuple[int, int, int]
    output_shape_hwc: tuple[int, int, int]
    strides: tuple[int, int]
    acc_scale: float
    out_scale: float
    input_offset: int
    bias_offset: int
    weight_offset: int
    input_bytes: int
    output_bytes: int
    bias_values: tuple[int, ...]
    weight_hwcf: bytes


def _walk_ops(op: Any):
    yield op
    for region in op.operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk_ops(child)


def _all_ops(module: Any):
    for op in module.body.operations:
        yield from _walk_ops(op)


def _attr_int(op: Any, name: str) -> int | None:
    if name not in op.attributes:
        return None
    try:
        return int(ir.IntegerAttr(op.attributes[name]).value)
    except (TypeError, ValueError):
        return None


def _constant_scalar(op: Any) -> int | float | None:
    if op.operation.name != "arith.constant" or "value" not in op.attributes:
        return None
    attr = op.attributes["value"]
    try:
        return int(ir.IntegerAttr(attr).value)
    except (TypeError, ValueError):
        pass
    try:
        return float(ir.FloatAttr(attr).value)
    except (TypeError, ValueError):
        return None


def _shape_elem(value: Any) -> tuple[tuple[int, ...], str] | None:
    ty = value.type
    if not hasattr(ty, "shape") or not hasattr(ty, "element_type"):
        return None
    return tuple(int(d) for d in ty.shape), str(ty.element_type)


def _defining_op(value: Any) -> Any | None:
    owner = value.owner
    return owner if hasattr(owner, "operation") else None


def _const_float_operand(op: Any) -> float | None:
    for operand in op.operands:
        owner = _defining_op(operand)
        if owner is None:
            continue
        value = _constant_scalar(owner)
        if isinstance(value, float):
            return value
    return None


def _extract_requant_scales(tail: Any) -> tuple[float, float] | None:
    """Find the first `mulf(..., acc_scale)` and first `divf(..., out_scale)`.

    The exported HTA island intentionally stops at the first accumulator
    requant. For SiLU dispatches the remaining sigmoid/mul tail is CPU/GPU
    boundary work in the split DAG.
    """
    acc_scale: float | None = None
    out_scale: float | None = None
    for region in tail.operation.regions:
        for block in region.blocks:
            for op in block.operations:
                if op.operation.name == "arith.mulf" and acc_scale is None:
                    acc_scale = _const_float_operand(op)
                if op.operation.name == "arith.divf" and out_scale is None:
                    out_scale = _const_float_operand(op)
                if acc_scale is not None and out_scale is not None:
                    return acc_scale, out_scale
    return None


def _infer_stride(input_size: int, kernel: int, output_size: int) -> int | None:
    for stride in (1, 2, 4):
        if ((input_size - kernel) // stride) + 1 == output_size:
            return stride
    return None


def _product(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _source_dispatch_from_key(key: str) -> str:
    marker = "dispatch_"
    if marker not in key:
        return key
    rest = key.split(marker, 1)[1]
    digits = []
    for ch in rest:
        if not ch.isdigit():
            break
        digits.append(ch)
    return f"dispatch_{int(''.join(digits))}" if digits else key


def _parse_dispatch(mlir_path: pathlib.Path, source_key: str, arena: bytes) -> ConvIsland | None:
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(mlir_path.read_text(), ctx)

    constants: dict[str, int | float] = {}
    subspans: dict[str, Subspan] = {}
    tensors: dict[str, TensorValue] = {}
    conv: Any | None = None
    tail: Any | None = None

    for op in _all_ops(module):
        name = op.operation.name
        if name == "arith.constant" and op.results:
            value = _constant_scalar(op)
            if value is not None:
                constants[op.results[0].get_name()] = value
        elif name == "hal.interface.binding.subspan" and op.results:
            offset_name = op.operands[0].get_name() if op.operands else None
            offset = constants.get(offset_name or "", 0)
            if not isinstance(offset, int):
                offset = 0
            result_name = op.results[0].get_name()
            subspans[result_name] = Subspan(
                name=result_name,
                binding=_attr_int(op, "binding") or 0,
                offset=offset,
                descriptor_flags=_attr_int(op, "descriptor_flags") or 0,
                type_text=str(op.results[0].type),
            )
        elif name == "iree_tensor_ext.dispatch.tensor.load" and op.results:
            meta = _shape_elem(op.results[0])
            if meta is None:
                continue
            src = op.operands[0].get_name() if op.operands else None
            tensors[op.results[0].get_name()] = TensorValue(
                shape=meta[0],
                elem=meta[1],
                source_subspan=src,
            )
        elif name == "linalg.generic" and op.results:
            meta = _shape_elem(op.results[0])
            if meta is None:
                continue
            shape, elem = meta
            operand_metas = [_shape_elem(operand) for operand in op.operands]
            if (
                elem == "i32"
                and len(shape) == 3
                and len(operand_metas) >= 2
                and operand_metas[0] is not None
                and operand_metas[1] is not None
                and operand_metas[0][1] == "i8"
                and operand_metas[1][1] == "i8"
                and len(operand_metas[0][0]) == 3
                and len(operand_metas[1][0]) == 4
            ):
                conv = op

    if conv is None:
        return None

    conv_result = conv.results[0].get_name()
    for op in _all_ops(module):
        if op.operation.name != "linalg.generic" or not op.results:
            continue
        meta = _shape_elem(op.results[0])
        if meta is None or meta[1] != "i8":
            continue
        if op.operands and op.operands[0].get_name() == conv_result:
            tail = op
            break
    if tail is None:
        return None

    input_name = conv.operands[0].get_name()
    weight_name = conv.operands[1].get_name()
    input_tv = tensors.get(input_name)
    weight_tv = tensors.get(weight_name)
    if input_tv is None or weight_tv is None:
        return None
    if input_tv.source_subspan is None or weight_tv.source_subspan is None:
        return None
    input_subspan = subspans.get(input_tv.source_subspan)
    weight_subspan = subspans.get(weight_tv.source_subspan)
    if input_subspan is None or weight_subspan is None:
        return None

    bias_tv: TensorValue | None = None
    bias_subspan: Subspan | None = None
    for operand in tail.operands[1:]:
        meta = _shape_elem(operand)
        if meta is None:
            continue
        shape, elem = meta
        if elem == "i32" and len(shape) == 1:
            tv = tensors.get(operand.get_name())
            if tv and tv.source_subspan:
                bias_tv = tv
                bias_subspan = subspans.get(tv.source_subspan)
                break
    if bias_tv is None or bias_subspan is None:
        return None

    scales = _extract_requant_scales(tail)
    if scales is None:
        return None
    acc_scale, out_scale = scales

    ic, ih, iw = input_tv.shape
    oc, oh, ow = _shape_elem(tail.results[0])[0]
    oc_w, ic_w, kh, kw = weight_tv.shape
    if (oc, ic) != (oc_w, ic_w):
        return None
    stride_h = _infer_stride(ih, kh, oh)
    stride_w = _infer_stride(iw, kw, ow)
    if stride_h is None or stride_w is None:
        return None

    bias_nbytes = oc * 4
    weight_nbytes = oc * ic * kh * kw
    bias_blob = arena[bias_subspan.offset : bias_subspan.offset + bias_nbytes]
    weight_fchw = arena[weight_subspan.offset : weight_subspan.offset + weight_nbytes]
    if len(bias_blob) != bias_nbytes or len(weight_fchw) != weight_nbytes:
        return None
    bias_values = struct.unpack(f"<{oc}i", bias_blob)

    # Convert FCHW bytes (OC,IC,KH,KW) to QNN-native HWCF (KH,KW,IC,OC).
    def weight_at(f: int, c: int, y: int, x: int) -> int:
        return weight_fchw[((f * ic + c) * kh + y) * kw + x]

    weight_hwcf = bytes(
        weight_at(f, c, y, x) for y in range(kh) for x in range(kw) for c in range(ic) for f in range(oc)
    )

    source_dispatch = _source_dispatch_from_key(source_key)
    return ConvIsland(
        source_key=source_key,
        source_dispatch=source_dispatch,
        func_name=f"hta_conv_island_{source_dispatch}",
        input_shape_chw=(ic, ih, iw),
        input_shape_hwc=(ih, iw, ic),
        weight_shape_fchw=(oc, ic, kh, kw),
        weight_shape_hwcf=(kh, kw, ic, oc),
        output_shape_chw=(oc, oh, ow),
        output_shape_hwc=(oh, ow, oc),
        strides=(stride_h, stride_w),
        acc_scale=acc_scale,
        out_scale=out_scale,
        input_offset=input_subspan.offset,
        bias_offset=bias_subspan.offset,
        weight_offset=weight_subspan.offset,
        input_bytes=ic * ih * iw,
        output_bytes=oh * ow * oc,
        bias_values=tuple(int(v) for v in bias_values),
        weight_hwcf=weight_hwcf,
    )


def _emit_wrapper(island: ConvIsland, qnn_backend: str) -> str:
    ih, iw, ic = island.input_shape_hwc
    kh, kw, _, oc = island.weight_shape_hwcf
    oh, ow, _ = island.output_shape_hwc
    sh, sw = island.strides
    bias_values = ", ".join(str(v) for v in island.bias_values)
    weight_hex = island.weight_hwcf.hex().upper()
    exe = f"{island.func_name}_exe"
    kernel = f"{island.func_name}_kernel"
    return f"""#map_bias = affine_map<(d0, d1, d2) -> (d2)>
#map_hwc = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_conv_in = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * {sh} + d3, d1 * {sw} + d4, d5)>
#map_conv_w = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>
#map_conv_out = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#executable_target_qnn_context_binary = #hal.executable.target<"qnn", "qnn-context-binary", {{opaque_binary = true, qnn_backend = "{qnn_backend}"}}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#device_target_qnn = #hal.device.target<"qnn", [#executable_target_qnn_context_binary]> : !hal.device
module {{
  util.global private @__device_0 = #device_target_qnn
  hal.executable private @{exe} {{
    hal.executable.variant public @qnn_context_binary target(#executable_target_qnn_context_binary) {{
      hal.executable.export public @{kernel} ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {{
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }}
      builtin.module {{
        func.func @{kernel}() {{
          %bias = arith.constant dense<[{bias_values}]> : tensor<{oc}xi32>
          %acc_scale = arith.constant {island.acc_scale:.9g} : f32
          %out_scale = arith.constant {island.out_scale:.9g} : f32
          %zp = arith.constant 0.000000e+00 : f32
          %min_i8 = arith.constant -1.280000e+02 : f32
          %max_i8 = arith.constant 1.270000e+02 : f32
          %weight = arith.constant dense<"0x{weight_hex}"> : tensor<{kh}x{kw}x{ic}x{oc}xi8>
          %c0 = arith.constant 0 : index
          %input_ref = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{ih}x{iw}x{ic}xi8>>
          %output_ref = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<{oh}x{ow}x{oc}xi8>>
          %input = iree_tensor_ext.dispatch.tensor.load %input_ref, offsets = [0, 0, 0], sizes = [{ih}, {iw}, {ic}], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{ih}x{iw}x{ic}xi8>> -> tensor<{ih}x{iw}x{ic}xi8>
          %out_i8 = tensor.empty() : tensor<{oh}x{ow}x{oc}xi8>
          %acc = tensor.empty() : tensor<{oh}x{ow}x{oc}xi32>
          %bias_bcast = linalg.generic {{indexing_maps = [#map_bias, #map_hwc], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%bias : tensor<{oc}xi32>) outs(%acc : tensor<{oh}x{ow}x{oc}xi32>) {{
          ^bb0(%in: i32, %out: i32):
            linalg.yield %in : i32
          }} -> tensor<{oh}x{ow}x{oc}xi32>
          %conv = linalg.generic {{indexing_maps = [#map_conv_in, #map_conv_w, #map_conv_out], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}} ins(%input, %weight : tensor<{ih}x{iw}x{ic}xi8>, tensor<{kh}x{kw}x{ic}x{oc}xi8>) outs(%bias_bcast : tensor<{oh}x{ow}x{oc}xi32>) {{
          ^bb0(%in: i8, %w: i8, %out: i32):
            %in_i32 = arith.extsi %in : i8 to i32
            %w_i32 = arith.extsi %w : i8 to i32
            %prod = arith.muli %in_i32, %w_i32 : i32
            %sum = arith.addi %out, %prod : i32
            linalg.yield %sum : i32
          }} -> tensor<{oh}x{ow}x{oc}xi32>
          %requant = linalg.generic {{indexing_maps = [#map_hwc, #map_hwc], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%conv : tensor<{oh}x{ow}x{oc}xi32>) outs(%out_i8 : tensor<{oh}x{ow}x{oc}xi8>) {{
          ^bb0(%in: i32, %out: i8):
            %f = arith.sitofp %in : i32 to f32
            %scaled = arith.mulf %f, %acc_scale : f32
            %qf = arith.divf %scaled, %out_scale : f32
            %round = math.roundeven %qf : f32
            %with_zp = arith.addf %round, %zp : f32
            %lo = arith.maximumf %with_zp, %min_i8 : f32
            %hi = arith.minimumf %lo, %max_i8 : f32
            %q = arith.fptosi %hi : f32 to i8
            linalg.yield %q : i8
          }} -> tensor<{oh}x{ow}x{oc}xi8>
          iree_tensor_ext.dispatch.tensor.store %requant, %output_ref, offsets = [0, 0, 0], sizes = [{oh}, {ow}, {oc}], strides = [1, 1, 1] : tensor<{oh}x{ow}x{oc}xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<{oh}x{ow}x{oc}xi8>>
          return
        }}
      }}
    }}
  }}
  func.func @{island.func_name}(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) attributes {{merlin.dispatch_entry}} {{
    %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
    %buffer_0 = hal.buffer_view.buffer<%arg1 : !hal.buffer_view> : !hal.buffer
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode(OneShot) categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
    %c0 = arith.constant 0 : index
    %input_bytes = arith.constant {island.input_bytes} : index
    %output_bytes = arith.constant {island.output_bytes} : index
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@{exe}::@qnn_context_binary::@{kernel}) : index, index, index
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@{exe}) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@{exe}::@qnn_context_binary::@{kernel}) : index
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) bindings([
      (%buffer : !hal.buffer)[%c0, %input_bytes],
      (%buffer_0 : !hal.buffer)[%c0, %output_bytes]
    ]) flags("None")
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %null_fence = util.null : !hal.fence
    %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%null_fence) signal(%fence) commands(%cmd) flags("None")
    %c-1_i32 = arith.constant -1 : i32
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.status.check_ok %status, "failed to wait on dispatch fence"
    return
  }}
}}
"""


def _ssh_base(host: str, identity: pathlib.Path | None) -> list[str]:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if identity:
        cmd.extend(["-i", str(identity)])
    cmd.append(host)
    return cmd


def _scp_base(identity: pathlib.Path | None) -> list[str]:
    cmd = ["scp", "-q", "-o", "StrictHostKeyChecking=no"]
    if identity:
        cmd.extend(["-i", str(identity)])
    return cmd


def _compile_wrapper(
    mlir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    compile_target: str,
) -> pathlib.Path | None:
    cmd = [
        str(REPO_ROOT / "merlin"),
        "compile",
        str(mlir),
        "--target",
        compile_target,
        "--build-dir",
        "host-merlin-release-qrb",
        "--output-dir",
        str(out_dir),
        "--iree-compile-arg=--iree-plugin=hal_target_qnn",
        "--iree-compile-arg=--iree-execution-model=async-internal",
    ]
    log = out_dir.with_suffix(".compile.log")
    out_dir.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    vmfbs = sorted(out_dir.glob("*.vmfb"))
    if proc.returncode != 0 or not vmfbs:
        return None
    return vmfbs[0]


def _input_for_island(capture_dir: pathlib.Path | None, island: ConvIsland) -> pathlib.Path | None:
    if capture_dir is None:
        return None
    candidates = [
        capture_dir / island.source_dispatch / "input_hwc_u8.bin",
        capture_dir / island.source_dispatch / "input_0_hwc_u8.bin",
        capture_dir / f"{island.source_dispatch}_input_hwc_u8.bin",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size == island.input_bytes:
            return candidate
    return None


def _profile_on_board(
    *,
    host: str,
    identity: pathlib.Path | None,
    board_bench: str,
    board_dir: str,
    device_uri: str,
    vmfb: pathlib.Path,
    input_file: pathlib.Path,
    island: ConvIsland,
    iterations: int,
    warmup: int,
    timeout_s: int,
) -> dict:
    remote_dir = f"{board_dir}/{island.func_name}"
    subprocess.run([*_ssh_base(host, identity), f"mkdir -p {shlex.quote(remote_dir)}"], check=True)
    remote_vmfb = f"{remote_dir}/{vmfb.name}"
    remote_input = f"{remote_dir}/input_hwc_u8.bin"
    remote_output = f"{remote_dir}/output_hwc_u8_zero.bin"
    local_output = vmfb.parent / "output_hwc_u8_zero.bin"
    local_output.write_bytes(bytes(island.output_bytes))
    for local, remote in [(vmfb, remote_vmfb), (input_file, remote_input), (local_output, remote_output)]:
        subprocess.run([*_scp_base(identity), str(local), f"{host}:{remote}"], check=True)
    remote_cmd = (
        "LD_LIBRARY_PATH=/root/qairt/lib/target "
        f"timeout {timeout_s} {shlex.quote(board_bench)} "
        f"--module={shlex.quote(remote_vmfb)} "
        f"--device={shlex.quote(device_uri)} "
        f"--function=module.{island.func_name} "
        f"--input={island.input_bytes}xi8=@{remote_input} "
        f"--input={island.output_bytes}xi8=@{remote_output} "
        f"--iterations={iterations} --warmup={warmup}"
    )
    proc = subprocess.run(
        [*_ssh_base(host, identity), remote_cmd],
        capture_output=True,
        text=True,
        timeout=timeout_s + 10,
        check=False,
    )
    payload = {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }
    match = BENCH_RE.search(proc.stdout)
    if proc.returncode == 0 and match:
        payload.update(
            {
                "setup_us": float(match.group("setup")) * 1000.0,
                "mean_us": float(match.group("mean")) * 1000.0,
                "median_us": float(match.group("median")) * 1000.0,
                "p99_us": float(match.group("p99")) * 1000.0,
                "min_us": float(match.group("min")) * 1000.0,
                "max_us": float(match.group("max")) * 1000.0,
            }
        )
    else:
        payload["error"] = "bench-failed-or-parse-failed"
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=pathlib.Path, required=True)
    parser.add_argument("--constant-arena", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--capture-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--call-graph-json",
        type=pathlib.Path,
        help="Optional call-site graph from "
        "extract_flow_dispatch_call_graph.py. When set, "
        "profile one island per matching dispatch call.",
    )
    parser.add_argument(
        "--profile-target",
        choices=("qnn_hta", "qnn_gpu"),
        default="qnn_hta",
        help="QNN target to compile/profile. The dumped MLIR " "can come from --matrix-cell-target.",
    )
    parser.add_argument(
        "--matrix-cell-target",
        choices=("qnn_hta", "qnn_gpu"),
        default=None,
        help="Target column to read dispatch MLIR from. "
        "Defaults to --profile-target, falling back to "
        "qnn_hta when profiling GPU from an HTA dump.",
    )
    parser.add_argument("--dispatch", action="append", default=[], help="Limit to dispatch_N. May be repeated.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ssh-host", default="root@10.44.120.201")
    parser.add_argument("--ssh-identity", type=pathlib.Path, default=DEFAULT_SSH_IDENTITY)
    parser.add_argument("--board-bench", default="/root/merlin-dispatch-bench")
    parser.add_argument("--board-dir", default="/root/hta_real_islands")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout-per-island-s", type=int, default=120)
    args = parser.parse_args(argv)

    matrix = json.loads(args.matrix.read_text())
    call_graph = json.loads(args.call_graph_json.read_text()) if args.call_graph_json else {}
    calls_by_canonical: dict[str, list[str]] = {}
    for call, call_row in call_graph.get("dispatch_graph", {}).items():
        canonical = str(call_row.get("canonical_dispatch") or call)
        calls_by_canonical.setdefault(canonical, []).append(call)
    arena = args.constant_arena.read_bytes()
    wanted = set(args.dispatch)
    matrix_cell_target = args.matrix_cell_target
    if matrix_cell_target is None:
        matrix_cell_target = args.profile_target
    qnn_backend = "hta" if args.profile_target == "qnn_hta" else "gpu"
    compile_target = "qrb5165_qnn_hta" if args.profile_target == "qnn_hta" else "qrb5165_qnn_gpu"
    device_uri = "qnn://hta" if args.profile_target == "qnn_hta" else "qnn://gpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mlir_dir = args.out_dir / "mlir"
    vmfb_dir = args.out_dir / "vmfb"
    mlir_dir.mkdir(exist_ok=True)
    vmfb_dir.mkdir(exist_ok=True)

    manifest: dict[str, Any] = {
        "matrix": str(args.matrix),
        "constant_arena": str(args.constant_arena),
        "dispatches": {},
    }
    csv_rows: list[dict[str, Any]] = []
    exported = 0
    profiled = 0
    skipped_capture = 0

    for source_key, row in sorted(matrix.get("dispatches", {}).items()):
        source_dispatch = _source_dispatch_from_key(source_key)
        if wanted and source_dispatch not in wanted:
            continue
        cell = row.get(matrix_cell_target) or {}
        if not cell and args.profile_target == "qnn_gpu":
            cell = row.get("qnn_hta") or {}
        mlir_path = pathlib.Path(cell.get("mlir", ""))
        if not mlir_path.is_file():
            continue
        island = _parse_dispatch(mlir_path, source_key, arena)
        if island is None:
            continue
        exported += 1
        profile_islands = [island]
        if calls_by_canonical:
            profile_islands = [
                dataclasses.replace(
                    island,
                    source_dispatch=call,
                    func_name=f"hta_conv_island_{call}",
                )
                for call in calls_by_canonical.get(island.source_dispatch, [])
            ]
        for profile_island in profile_islands:
            out_mlir = mlir_dir / f"{profile_island.func_name}.mlir"
            out_mlir.write_text(_emit_wrapper(profile_island, qnn_backend))
            input_file = _input_for_island(args.capture_dir, profile_island)
            vmfb_path: pathlib.Path | None = None
            compile_error = None
            if args.compile or args.profile:
                vmfb_path = _compile_wrapper(
                    out_mlir,
                    vmfb_dir / profile_island.func_name,
                    compile_target=compile_target,
                )
                if vmfb_path is None:
                    compile_error = f"compile failed for {out_mlir}"
            profile = None
            if args.profile and vmfb_path is not None:
                if input_file is None:
                    skipped_capture += 1
                    profile = {"error": "missing-real-captured-input"}
                else:
                    profile = _profile_on_board(
                        host=args.ssh_host,
                        identity=args.ssh_identity,
                        board_bench=args.board_bench,
                        board_dir=args.board_dir,
                        device_uri=device_uri,
                        vmfb=vmfb_path,
                        input_file=input_file,
                        island=profile_island,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        timeout_s=args.timeout_per_island_s,
                    )
                    if "mean_us" in profile:
                        profiled += 1
            elif compile_error:
                profile = {"error": compile_error}

            entry = {
                "feasible": profile is not None and "mean_us" in profile,
                "source_dispatch": profile_island.source_dispatch,
                "canonical_dispatch": island.source_dispatch,
                "source_key": island.source_key,
                "func": profile_island.func_name,
                "mlir": str(out_mlir),
                "vmfb": str(vmfb_path) if vmfb_path else None,
                "input_shape_hwc": list(island.input_shape_hwc),
                "output_shape_hwc": list(island.output_shape_hwc),
                "weight_shape_hwcf": list(island.weight_shape_hwcf),
                "input_bytes": island.input_bytes,
                "output_bytes": island.output_bytes,
                "input_offset": island.input_offset,
                "bias_offset": island.bias_offset,
                "weight_offset": island.weight_offset,
                "acc_scale": island.acc_scale,
                "out_scale": island.out_scale,
                "captured_input": str(input_file) if input_file else None,
                "profile": profile,
            }
            island_key = f"hta_conv_island_{profile_island.source_dispatch}_dispatch_0"
            manifest["dispatches"][island_key] = {args.profile_target: entry}
            flat = {
                "island": island_key,
                "source_dispatch": profile_island.source_dispatch,
                "canonical_dispatch": island.source_dispatch,
                "func": profile_island.func_name,
                "input_bytes": island.input_bytes,
                "output_bytes": island.output_bytes,
                "input_offset": island.input_offset,
                "mlir": str(out_mlir),
                "vmfb": str(vmfb_path) if vmfb_path else "",
                "captured_input": str(input_file) if input_file else "",
            }
            if profile:
                for key in ("setup_us", "mean_us", "median_us", "p99_us", "min_us", "max_us", "error"):
                    if key in profile:
                        flat[key] = profile[key]
            csv_rows.append(flat)

    manifest["summary"] = {
        "exported": exported,
        "profiled": profiled,
        "skipped_missing_capture": skipped_capture,
        "note": "No synthetic timings; missing captures are skipped.",
    }
    manifest_path = args.out_dir / f"profiled_manifest_{args.profile_target}_real_islands.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    csv_path = args.out_dir / f"profiled_manifest_{args.profile_target}_real_islands.csv"
    if csv_rows:
        keys: list[str] = []
        for row in csv_rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(json.dumps(manifest["summary"], indent=2))
    print(f"wrote {manifest_path}")
    if csv_rows:
        print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
