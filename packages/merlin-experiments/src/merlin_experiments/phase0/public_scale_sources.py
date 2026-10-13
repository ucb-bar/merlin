"""Bounded type-only scale/tail probes from original public interface semantics.

No tensor values, captured models, reference outputs or target schedules are
read or generated. Unsupported compositions/layouts retain only their original
source. These cases establish no numerical, physical or compiler acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from merlin.llvmlower.device_shim import logical_buffers
from merlin.targetgen.capsule_dram import dtype_bits
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir
from merlin_experiments.phase0.public_compile_sources import MAX_CASES, PublicCompileCase, _file


def _abi(source, index_bits):
    buffers, refusal = logical_buffers(source)
    if buffers is None:
        raise ValueError(refusal)
    maximum = (1 << (index_bits - 1)) - 1
    if any(not buf.shape or any(type(size) is not int or size <= 0 for size in buf.shape)
           or (math.prod(buf.shape) * dtype_bits(buf.dtype) + 7) // 8 > maximum for buf in buffers):
        raise ValueError("public scale tensor byte extent exceeds original signed index range")
    return CompileOnlySourceAbi(
        tuple(CompileOnlyTensor(buf.name, buf.shape, buf.dtype) for buf in buffers if buf.kind == "input"),
        tuple(CompileOnlyTensor(buf.name, buf.shape, buf.dtype) for buf in buffers if buf.kind == "output"),
    )


def _family(cb, source):
    work = [cmd for cmd in cb["commands"] if cmd["opcode"] not in {"RES_PACK", "COMMIT", "EVICT"}]
    if len(work) != 1 or work[0]["opcode"] not in {"MATMUL_RESIDENT", "MATMUL", "MOVEMENT", "CONV2D"}:
        return None
    if any(row.get("role") not in {"input", "weight", "output"} for row in cb["tensors"].values()):
        return None
    if any(row["dtype"] not in {"i8", "i16", "i32", "i64"} for row in cb["tensors"].values()):
        return None
    if any((cmd.get("attributes") or {}).get("epilogue") for cmd in cb["commands"]):
        return None
    if any((cmd.get("attributes") or {}).get("layout", "packed_rhs")
           not in {"packed_rhs", "packed_conv_rhs", "NHWC", "nhwc"} for cmd in cb["commands"]):
        return None
    if set(re.findall(r"!merlin_iface\.acc<([^>]+)>", source)) - {"i32"}:
        return None  # The original accumulator precision must survive reconstruction.
    return cb["commands"].index(work[0])


def _resize(original, index, scale, tail, edge):
    cb = copy.deepcopy(original)
    command = cb["commands"][index]
    operands, attrs = command["operands"], command.get("attributes") or {}
    resident = {cmd["operands"]["dst"]: cmd["operands"]["src"] for cmd in cb["commands"]
                if cmd["opcode"] == "RES_PACK"}

    def grow(size):
        if type(size) is not int or size <= 0:
            raise ValueError("public scale source has dynamic or empty input extent")
        value = size * scale
        if tail and edge:
            value = (value + edge - 1) // edge * edge
        return max(1, value + tail)

    if command["opcode"] in {"MATMUL_RESIDENT", "MATMUL"}:
        lhs = cb["tensors"][operands["lhs"]]
        rhs = cb["tensors"][resident.get(operands["rhs"], operands["rhs"])]
        if len(lhs["shape"]) != 2 or len(rhs["shape"]) != 2 or lhs["shape"][1] != rhs["shape"][0]:
            raise ValueError("public contraction has no original rank-two dimension join")
        m, k, n = grow(lhs["shape"][0]), grow(lhs["shape"][1]), grow(rhs["shape"][1])
        lhs["shape"], rhs["shape"] = [m, k], [k, n]
    elif command["opcode"] == "MOVEMENT":
        shape = [grow(size) for size in cb["tensors"][operands["src"]]["shape"]]
        cb["tensors"][operands["src"]]["shape"] = shape
        cb["tensors"][operands["dst"]]["shape"] = shape
    else:
        image = cb["tensors"][operands["ifm"]]
        weight = cb["tensors"][resident.get(operands["weight"], operands["weight"])]
        kh, kw, ci, co = attrs["kernel"]
        if len(image["shape"]) != 4 or attrs.get("layout") not in {"NHWC", "nhwc"}:
            raise ValueError("public convolution lacks original NHWC geometry")
        batch, height, width, channels = image["shape"]
        if channels != ci or weight["shape"] != [kh * kw * ci, co]:
            raise ValueError("public convolution input/weight/kernel do not join")
        height, width, ci, co = grow(height), grow(width), grow(ci), grow(co)
        sh, sw = attrs["stride"]
        pt, pl, pb, pr = attrs["padding"]
        dh, dw = attrs["dilation"]
        if min(sh, sw, dh, dw, kh, kw) <= 0 or min(pt, pl, pb, pr) < 0:
            raise ValueError("public convolution has unsupported original geometry")
        from merlin.runtime.commandbuffer import conv_out_dims

        oh, ow = conv_out_dims(height, width, kh, kw, (sh, sw), (pt, pl, pb, pr), (dh, dw))
        if oh <= 0 or ow <= 0:
            raise ValueError("public convolution has no complete output geometry")
        image["shape"], weight["shape"] = [batch, height, width, ci], [kh * kw * ci, co]
        cb["tensors"][operands["dst"]]["shape"] = [batch * oh * ow, co]
        attrs["kernel"] = [kh, kw, ci, co]
    return emit_interface_mlir(cb)


def derive(capsules, *, selected="all", index_bits, tile_edge=None):
    """The original host owns this fixed scale/tail ladder, never broker data."""
    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise ValueError("public scale probes require the original host index width")
    if tile_edge is not None and (type(tile_edge) is not int or tile_edge <= 0):
        raise ValueError("public scale tails require original positive hardware geometry")
    wanted = {name.strip() for name in selected.split(",") if name.strip()}
    public = {str(cap.get("name") or Path(cap["__dir__"]).name): cap for cap in capsules
              if cap.get("label") in {"public", "dev"}}
    if len(public) != sum(cap.get("label") in {"public", "dev"} for cap in capsules):
        raise ValueError("public scale member identity is ambiguous")
    if wanted != {"all"} and (not wanted or wanted - set(public)):
        raise ValueError("scale probe requested unknown or non-public members")
    cases, unavailable = [], []
    for name, capsule in sorted(public.items()):
        if wanted != {"all"} and name not in wanted:
            continue
        issued = len(cases)
        try:
            root = Path(capsule["__dir__"]).absolute()
            if not capsule.get("interface_mlir") and capsule.get("linalg_mlir"):
                from merlin_experiments.phase0.public_compile_sources import derive as derive_linalg

                siblings, record = derive_linalg([capsule], selected=name, index_bits=index_bits)
                if len(cases) + len(siblings) > MAX_CASES:
                    raise ValueError("public scale roster exceeds its original metadata budget")
                cases.extend(siblings)
                unavailable.extend(record["unavailable"])
                continue
            member = Path(capsule.get("interface_mlir") or "")
            if not member.parts or member.is_absolute() or ".." in member.parts:
                raise ValueError("public member has no retained interface")
            metadata, raw = _file(root / "capsule.yaml"), _file(root / member)
            source = raw.decode()
            if "prov.weights_file" in source:
                raise ValueError("scale probe cannot read captured values")
            original = parse_interface_mlir(source)
            family = _family(original, source)
            variants = [(1, 0)] if family is None else [(1, 0), (8, 0), (8, -1), (8, 1), (64, 0)]
            provenance = ((root / "capsule.yaml", hashlib.sha256(metadata).hexdigest()),
                          (root / member, hashlib.sha256(raw).hexdigest()))
            for scale, tail in variants:
                emitted = source if scale == 1 else _resize(original, family, scale, tail, tile_edge)
                abi = _abi(emitted, index_bits)
                if len(cases) >= MAX_CASES:
                    raise ValueError("public scale roster exceeds its original metadata budget")
                derivation = {"basis": "retained_public_interface" if scale == 1
                              else "public_interface_shape_semantics",
                              "variant": f"scale{scale}_tail{tail:+d}", "extent_ratio": scale, "extent_tail": tail,
                              "hardware_tail_edge": tile_edge, "host_index_bits": index_bits,
                              "tensor_values_allocated": False, "goldens_allocated": False,
                              "scale_family": original["commands"][family]["opcode"] if family is not None
                              else "retained_only",
                              "scope": "public type-only emission; indexing/resources/numerics unproved"}
                cases.append(PublicCompileCase(name, scale, emitted, abi, provenance,
                                               json.dumps(derivation, sort_keys=True)))
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            del cases[issued:]
            unavailable.append({"member": name, "status": "unavailable", "reason": type(error).__name__})
    return tuple(cases), {"schema": "merlin.public_scale_sources.v1", "cases": [case.record() for case in cases],
                          "unavailable": unavailable, "numeric": "not_attempted", "rtl": "not_attempted"}
