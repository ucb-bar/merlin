#!/usr/bin/env python3
"""A captured model as the program its compute groups ARE, run with the vendor's own kernels.

Group formation says which operations a unit takes whole; the prepack says what numbers each group
is issued with; the stream plan says nothing between two device groups has to visit the host. None
of that is a measurement until a program built from exactly those groups runs. This builds one: a
bare-metal program with ONE library call per compute group, in the model's own order, reading the
folded biases, device-layout weights and readout multipliers the compiler derived, and nothing the
compiler did not derive. The vendor library stands in for the per-group kernel, which is the part a
Phase-1 backend is graded on; everything a whole-model route adds on top of a kernel is what this
measures.

It refuses a model that is not closed: any host region other than the quantization of the model's
input is named and the program is not built, because a number for "the groups" that silently runs
the rest on the host is the defect this work exists to end.

The program checks itself against the capture's own golden (the fake-quantized model's output on
the capture's input): it prints the argmax it got and the one it wanted, and the cosine between
the two logit vectors. It is not held to byte equality, and says why: a residual sum on this unit
rounds each operand, within the bound its group declares.

    group_model_program.py --capture <capture dir> --target <target> --vendor-source <snapshot>/source \\
        --compiler <riscv64-unknown-elf-gcc> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "group_model_program_v1"
CFLAGS = (
    "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
    "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
    "-Wa,-march=rv64gc", "-lm", "-lgcc", "-DID_STRING=", "-Wno-incompatible-pointer-types",
    "-nostdlib", "-nostartfiles", "-static", "-DBAREMETAL=1",
)  # fmt: skip


class NotClosed(ValueError):
    """The model has host work between its groups, so "the groups" is not the whole program."""


def _product(extents) -> int:
    out = 1
    for extent in extents:
        out *= int(extent)
    return out


def _through_views(value):
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    owner = getattr(value, "owner", None)
    while owner is not None and mq.op_name(owner) in CG._VIEW_OPS and getattr(owner, "operands", None):
        value = owner.operands[0]
        owner = getattr(value, "owner", None)
    return owner


def extract(capture: Path, target: str, *, oracle=None) -> dict[str, Any]:
    """The model as an ordered list of device steps over named buffers, with every array they read."""
    import numpy as np

    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC
    from merlin.xdsl_dialects.lowering import group_numerics as GN
    from merlin.xdsl_dialects.lowering import group_prepack as GP
    from merlin.xdsl_dialects.lowering import stream_plan as SP

    text = (capture / "linalg.mlir").read_text(encoding="utf-8")
    manifest_path, weights_path = capture / "weights.safetensors.manifest.json", capture / "weights.safetensors"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = CG.form_groups(mq.parse(text), target, oracle=oracle)
    weight_args = SP.weight_args_of(manifest)
    packed = GP.prepack(groups, manifest_path, weights_path, device_layout=True)
    rows = {int(row["group"]): row for row in packed["record"]["groups"]}

    from xdsl.ir import BlockArgument

    def quantizes_an_input(group) -> bool:
        """The one host region a closed model keeps: a model ARGUMENT put on the integer grid."""
        return group.stages == [CG.QUANTIZE] and isinstance(group.members[-1].operands[0], BlockArgument)

    open_host = [g for g in groups if g.placement == CG.HOST and not quantizes_an_input(g)]
    if open_host:
        raise NotClosed(
            f"{len(open_host)} host region(s) compute between the groups, so a program of groups is not "
            f"the model: " + "; ".join(f"group {g.index} {g.stages[:4]} ({g.reason})" for g in open_host[:4])
        )

    buffers: dict[int, dict[str, Any]] = {}  # id(SSA value) -> buffer
    arrays: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    image = None

    def buffer_of(value, *, why: str) -> str:
        found = buffers.get(id(value))
        if found is None:
            raise NotClosed(f"{why} reads a value no group produced")
        return found["name"]

    def produce(value, name: str, elements: int, ctype: str = "elem_t") -> str:
        buffers[id(value)] = {"name": name, "elements": int(elements), "ctype": ctype}
        return name

    for group in groups:
        tag = f"g{group.index}"
        if group.placement == CG.HOST:
            quantize = group.members[-1]
            scale = GN._scale_source(quantize)
            if image is not None or scale.value is None or scale.zero_point:
                raise NotClosed("the model quantizes more than one input, or not under one static symmetric scale")
            image = {"scale": scale.value, "value": quantize.results[0]}
            shape, _ = mq.type_shape_dtype(quantize.results[0].type)
            produce(quantize.results[0], "IMAGE", _product(shape))
            image["shape"] = [int(v) for v in shape]
            continue
        sink = group.members[-1].results[0]
        if group.operand_sum is not None:
            sources = [_through_views(operand) for operand in list(group.root.operands)[:2]]
            shape, _ = mq.type_shape_dtype(group.root.results[0].type)
            factor = float(group.operand_sum["readout_factor"])
            steps.append(
                {
                    "kind": "sum",
                    "group": group.index,
                    "lhs": buffer_of(sources[0].operands[0], why=f"group {group.index}"),
                    "rhs": buffer_of(sources[1].operands[0], why=f"group {group.index}"),
                    "out": produce(sink, f"B_{tag}", _product(shape)),
                    "rows": _product(shape[:-1]),
                    "cols": int(shape[-1]),
                    "lhs_load": float(group.operand_sum["lhs_scale"]) / factor,
                    "rhs_load": float(group.operand_sum["rhs_scale"]) / factor,
                    "readout": factor,
                    "relu": bool(group.operand_sum["relu"]),
                    "bound_lsb": int(group.operand_sum["bound_lsb"]),
                }
            )
            continue
        if group.window_mean is not None:
            dequantize = next(m for m in group.members if CG.classify(m).kind == CG.DEQUANTIZE)
            rows_kept, window = int(group.window_mean["rows"]), int(group.window_mean["window"])
            steps.append(
                {
                    "kind": "mean",
                    "group": group.index,
                    "in": buffer_of(dequantize.operands[0], why=f"group {group.index}"),
                    "out": produce(sink, f"B_{tag}", rows_kept),
                    "rows": rows_kept,
                    "window": window,
                    "multiplier": float(group.window_mean["multiplier"]),
                }
            )
            continue

        try:
            stated = GC.program(group, weight_args=weight_args)
        except CG.NoCapsuleForm as refusal:
            raise NotClosed(f"group {group.index} cannot be stated as a device program: {refusal}") from refusal
        entry = stated.entry
        activation = list(group.root.operands)[1 - int(stated.stored_operand)]
        _adapters, dequantize, _dtype = CG._input_chain(activation)
        step: dict[str, Any] = {
            "kind": entry["op"],
            "group": group.index,
            "in": buffer_of(dequantize.operands[0], why=f"group {group.index}"),
            "relu": "relu" in entry["epilogue"],
            "bias": None,
        }
        row = rows.get(group.index)
        if row is not None:  # a closed group: the prepack already holds its numbers
            step["scale"] = float(row["multiplier"])
            weight = packed["arrays"][row["weight"]["array"]]
            if "bias" in row:
                step["bias"] = f"BIAS_{tag}"
                arrays[step["bias"]] = packed["arrays"][row["bias"]["array"]].astype(np.int32)
            out_ctype = "elem_t"
        else:
            # Not closed: the contraction's result leaves as the accumulator (a model's final
            # classifier). Its bias is folded by the same rule a closed group's is.
            if "acc_scale" in entry["epilogue"] or CG.QUANTIZE in group.stages:
                raise NotClosed(f"group {group.index} is closed and the prepack holds no row for it")
            sources = [GN._scale_source(m) for m in group.members if CG.classify(m).kind == CG.DEQUANTIZE]
            if len(sources) != 2 or any(s.value is None or s.zero_point for s in sources):
                raise NotClosed(f"group {group.index} leaves as an accumulator whose scales are not static")
            divisor = sources[0].value * sources[1].value
            name, stored, _spelled = GP._stored_raw(stated.stored_arg, manifest, weights_path)
            weight = GP.device_weight(group, stated, stored)
            if stated.bias_arg is not None:
                _bias_name, bias = GP._stored_tensor(stated.bias_arg, manifest, weights_path)
                step["bias"] = f"BIAS_{tag}"
                arrays[step["bias"]] = np.rint(bias / divisor).astype(np.int32)
            step.update({"scale": None, "dequantize": divisor})
            out_ctype = "acc_t"
        if entry["op"] == "conv2d":
            if entry["Himg"] != entry["Wimg"] or entry["kh"] != entry["kw"] or len(set(entry["stride"])) != 1:
                raise NotClosed(f"group {group.index} is not a square convolution, which is all the library states")
            pad = entry.get("padding") or [0, 0, 0, 0]
            if len(set(pad)) != 1:
                raise NotClosed(f"group {group.index} pads asymmetrically ({pad})")
            out_dim = (entry["Himg"] + 2 * pad[0] - entry["kh"]) // entry["stride"][0] + 1
            pool = {"size": 0, "stride": 0, "padding": 0}
            final_dim = out_dim
            if "maxpool" in entry["epilogue"]:
                pool = {
                    "size": int(entry["pool_size"][0]),
                    "stride": int(entry["pool_stride"][0]),
                    "padding": int(entry["pool_padding"][0]),
                }
                final_dim = (out_dim + 2 * pool["padding"] - pool["size"]) // pool["stride"] + 1
            # The prepack's device layout IS the library's: [tap_h, tap_w, channel] by output, which
            # is how its convolution indexes a weight (`(krow*k*ci + kcol*ci + kch) * n + och`).
            kh, ci, n = int(entry["kh"]), int(entry["ci"]), int(entry["N"])
            weight = np.ascontiguousarray(weight.reshape(kh, kh, ci, n))
            step.update(
                {
                    "in_dim": int(entry["Himg"]),
                    "ci": ci,
                    "n": n,
                    "out_dim": int(out_dim),
                    "stride": int(entry["stride"][0]),
                    "padding": int(pad[0]),
                    "kernel": kh,
                    "pool": pool,
                }
            )
            elements = final_dim * final_dim * n
        else:
            step.update({"m": int(entry["M"]), "k": int(entry["K"]), "n": int(entry["N"])})
            elements = int(entry["M"]) * int(entry["N"])
        step["weight"] = f"W_{tag}"
        arrays[step["weight"]] = np.ascontiguousarray(weight).astype(np.int8)
        step["out"] = produce(sink, f"B_{tag}", elements, out_ctype)
        steps.append(step)

    if image is None or not steps:
        raise NotClosed("the capture has no quantized input or no device group")
    final = steps[-1]
    if final.get("dequantize") is None:
        raise NotClosed("the model's last group does not leave as an accumulator to dequantize; nothing to compare")

    # The model's input, quantized the way its one host region does and laid out as the library
    # reads it (positions by channel).
    pixels = np.asarray(json.loads((capture / "inputs.json").read_text(encoding="utf-8"))[0], dtype=np.float32)
    quantized = np.clip(np.rint(pixels / np.float32(image["scale"])), -128, 127).astype(np.int8)
    arrays["IMAGE_DATA"] = np.ascontiguousarray(quantized.transpose(0, 2, 3, 1))
    golden = np.asarray(json.loads((capture / "golden.json").read_text(encoding="utf-8"))[0], dtype=np.float32)
    arrays["GOLDEN"] = golden.reshape(-1)
    return {
        "steps": steps,
        "buffers": [b for b in buffers.values() if b["name"] != "IMAGE"],
        "image_elements": int(arrays["IMAGE_DATA"].size),
        "arrays": arrays,
        "groups": len(groups),
        "device_groups": len(steps),
        "classes": int(golden.size),
    }


def emulate(model: dict[str, Any], *, single_rounding_sums: bool = False) -> dict[str, Any]:
    """The same steps in numpy, with the arithmetic each library call is documented to do.

    It answers one question before any simulator runs: is the INTEGER PROGRAM the model? A wrong
    weight layout, bias rule or multiplier shows here, against the capture's golden, in seconds and
    with every intermediate tensor in hand. Returns the named buffers and the final comparison.

    ``single_rounding_sums`` computes every sum the way the capture does (scale both operands, add,
    round once), so the difference between the two emulations is what the unit's per-operand
    rounding costs THIS model on THIS input, separated from everything else.
    """
    import numpy as np

    def readout(acc, scale, relu):
        out = np.rint(acc.astype(np.float32) * np.float32(scale)) if scale is not None else acc
        out = np.clip(out, -128, 127) if scale is not None else out
        return np.maximum(out, 0) if relu else out

    arrays = model["arrays"]
    values: dict[str, Any] = {"IMAGE": arrays["IMAGE_DATA"].astype(np.int64)}
    for step in model["steps"]:
        if step["kind"] == "conv2d":
            x = values[step["in"]].reshape(step["in_dim"], step["in_dim"], step["ci"])
            pad, k, stride = step["padding"], step["kernel"], step["stride"]
            x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
            windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(0, 1))[::stride, ::stride]
            w = arrays[step["weight"]].astype(np.int64)  # [kh, kw, ci, n]
            acc = np.einsum("hwcij,ijcn->hwn", windows, w)
            if step["bias"]:
                acc = acc + arrays[step["bias"]].astype(np.int64)
            out = readout(acc, step["scale"], step["relu"])
            pool = step["pool"]
            if pool["size"]:
                out = np.pad(out, ((pool["padding"],) * 2, (pool["padding"],) * 2, (0, 0)), constant_values=-128)
                out = np.lib.stride_tricks.sliding_window_view(out, (pool["size"],) * 2, axis=(0, 1))
                out = out[:: pool["stride"], :: pool["stride"]].max(axis=(-1, -2))
            values[step["out"]] = out.reshape(-1).astype(np.int64)
        elif step["kind"] == "matmul":
            a = values[step["in"]].reshape(step["m"], step["k"])
            acc = a @ arrays[step["weight"]].astype(np.int64).reshape(step["k"], step["n"])
            if step["bias"]:
                acc = acc + arrays[step["bias"]].astype(np.int64)
            values[step["out"]] = readout(acc, step["scale"], step["relu"]).reshape(-1).astype(np.int64)
        elif step["kind"] == "sum" and single_rounding_sums:
            total = sum(
                values[step[side]].astype(np.float32) * np.float32(step[f"{side}_load"] * step["readout"])
                for side in ("lhs", "rhs")
            )
            values[step["out"]] = readout(total, 1.0, step["relu"]).astype(np.int64)
        elif step["kind"] == "sum":
            loads = [
                np.clip(np.rint(values[step[side]].astype(np.float32) * np.float32(step[f"{side}_load"])), -128, 127)
                for side in ("lhs", "rhs")
            ]
            values[step["out"]] = readout(loads[0] + loads[1], step["readout"], step["relu"]).astype(np.int64)
        else:
            a = values[step["in"]].reshape(step["window"], step["rows"])
            values[step["out"]] = readout(a.sum(axis=0), step["multiplier"], False).astype(np.int64)
    final = model["steps"][-1]
    logits = values[final["out"]].astype(np.float64) * float(final["dequantize"])
    golden = arrays["GOLDEN"].astype(np.float64)
    cosine = float(logits @ golden / (np.linalg.norm(logits) * np.linalg.norm(golden) + 1e-30))
    return {"values": values, "argmax": int(logits.argmax()), "want": int(golden.argmax()), "cosine": cosine}


#: The digest constants are the hardened per-layer route's own, imported rather than restated: the
#: gemmini layer bench digests every window's output with ``lb_fnv1a_words`` built from these, and a
#: second copy of an FNV offset here is a second place the two routes could quietly disagree about
#: what "the same digest" means.
def _digest_constants():
    from merlin.perf.layer_bench import reference as _ref

    return _ref.FNV_OFFSET, _ref.FNV_PRIME, _ref.DIGEST_MASK


def group_digest(values) -> int:
    """The ORDER-SENSITIVE digest of one group's output, computed the way the device computes it.

    FNV-1a over the output's elements, each sign-extended to a little-endian 64-bit word, top bit
    cleared so it prints and stores as a signed integer.  Widening to 64 bits is what makes this
    reproducible without knowing how wide ``elem_t`` and ``acc_t`` are on the target: the C helper
    widens each element the same way, so one function checks an int8 output and an accumulator
    output without either side holding a width as a constant.

    It replaces an additive sum, which could not see the failure it most needed to: reorder a
    group's output -- a transpose, a stride, a tiling bug -- and the sum is unchanged, as it is for
    any pair of compensating +k/-k errors.  Every whole-model row measured before this change was
    admitted on that sum.
    """
    import numpy as np

    from merlin.perf.layer_bench import reference as _ref

    _offset, _prime, mask = _digest_constants()
    return _ref.fnv1a64_words(np.ascontiguousarray(values, dtype="<i8").tobytes()) & mask


def group_checksums(model: dict[str, Any], emulation: dict[str, Any]) -> dict[str, dict[str, int]]:
    """``{group: {"sum": ..., "fnv1a": ...}}`` for every device group, from the numpy emulation.

    THE ORACLE A v2 POLICY DECLARES.  Both numbers are produced here so a policy generator and this
    program cannot drift into computing them differently; only ``fnv1a`` is order-sensitive, and
    only ``fnv1a`` should be the ``value_key`` of a policy a cycle claim is sealed against.

    The emulated buffer must hold exactly as many elements as the program declares for it, or the
    two are not digesting the same thing and the result would be an incomparable number rather than
    a wrong one.  That is checked, not assumed.
    """
    sizes = {buffer["name"]: int(buffer["elements"]) for buffer in model["buffers"]}
    values = emulation["values"]
    rows: dict[str, dict[str, int]] = {}
    for step in model["steps"]:
        name = step["out"]
        emulated = values[name]
        if int(emulated.size) != sizes[name]:
            raise ValueError(
                f"group {step['group']} emulates {emulated.size} elements of {name!r} and the "
                f"program declares {sizes[name]}; these are not the same buffer"
            )
        rows[str(step["group"])] = {"sum": int(emulated.sum()), "fnv1a": group_digest(emulated)}
    return rows


def _call(step: dict[str, Any]) -> str:
    act = "RELU" if step.get("relu") else "NO_ACTIVATION"
    bias = step.get("bias") or "NULL"
    if step["kind"] == "conv2d":
        pool = step["pool"]
        return (
            f"tiled_conv_auto(1, {step['in_dim']}, {step['in_dim']}, {step['ci']}, {step['n']}, {step['out_dim']}, "
            f"{step['out_dim']}, {step['stride']}, 1, 1, {step['padding']}, {step['kernel']}, false, false, false, "
            f"false, false, {step['in']}, {step['weight']}, {bias}, {step['out']}, {act}, {step['scale']!r}f, "
            f"{pool['size']}, {pool['stride']}, {pool['padding']}, WS);"
        )
    if step["kind"] == "matmul":
        full = step["scale"] is None
        scale = "ACC_SCALE_IDENTITY" if full else f"{step['scale']!r}f"
        return (
            f"tiled_matmul_auto({step['m']}, {step['n']}, {step['k']}, {step['in']}, {step['weight']}, {bias}, "
            f"{step['out']}, {step['k']}, {step['n']}, {step['n']}, {step['n']}, MVIN_SCALE_IDENTITY, "
            f"MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, {act}, {scale}, 0, true, false, false, "
            f"{'true' if full else 'false'}, false, 0, WS);"
        )
    if step["kind"] == "sum":
        return (
            f"tiled_resadd_auto({step['rows']}, {step['cols']}, {step['lhs_load']!r}f, {step['rhs_load']!r}f, "
            f"{step['readout']!r}f, {step['lhs']}, {step['rhs']}, {step['out']}, "
            f"{'true' if step['relu'] else 'false'}, WS);"
        )
    # A mean over a trailing window: the buffer holds [window positions] by [rows channels], so the
    # contraction reads it transposed, against a constant one.
    return (
        f"tiled_matmul_auto({step['rows']}, 1, {step['window']}, {step['in']}, ONES, NULL, {step['out']}, "
        f"{step['rows']}, 1, 1, 1, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, "
        f"NO_ACTIVATION, {step['multiplier']!r}f, 0, false, true, false, false, false, 0, WS);"
    )


#: The label of the one measured window this program publishes.  It is a DEFAULT, not a constant:
#: the label is what ties a cycle number to the policy entry that declares what the window must
#: print, so a build measuring something else passes its own.
DEFAULT_WINDOW_LABEL = "group_model"

#: EVERY LINE THIS PROGRAM PUBLISHES, ONCE.  The C ``printf`` formats below are built from these,
#: and so is :func:`uart_lines`, which renders the UART a run would produce.  One spelling with two
#: renderers is the point: a test that hand-wrote the expected UART would be grading the harness
#: against the same guess that wrote it, which is how this file came to print a metric line nothing
#: could parse and no test noticed for as long as it did.
UART = {
    "invocations": "MERLIN_INVOCATIONS warmup=1 measured=1",
    "window_begin": "MERLIN_WINDOW begin label={label}",
    "warm_begin": "MERLIN_PROFILE warmup begin",
    "warm_end": "MERLIN_PROFILE warmup end rc=0",
    "measured_begin": "MERLIN_PROFILE measured begin",
    # TWO NUMBERS, AND ONLY ONE OF THEM IS A CHECK.  ``sum`` is the additive total this program has
    # always printed; it is kept because it is the only quantity the seven FPGA runs on record also
    # published, so a new build can still be shown to agree with them.  It is NOT what a cycle claim
    # is admitted against: an additive sum is permutation-blind, so a transpose, a stride or a
    # layout bug that REORDERS a group's output leaves it byte-identical, as does any pair of
    # compensating +k/-k errors -- and reordering is a first-class failure mode of a tensor compiler,
    # which is exactly what this program measures.  ``fnv1a`` is the order-sensitive digest a v2
    # policy declares (``value_key: "fnv1a"``); it is the same FNV-1a the hardened per-layer route
    # digests its outputs with (``merlin.perf.layer_bench.reference``), so both routes are checked by
    # one function rather than by two that happen to agree.
    "group": "GM_GROUP {group} {kind} {cycles} sum={sum} fnv1a={checksum}",
    "argmax": "GM_ARGMAX got={got} want={want} agrees={agrees}",
    "cosine": "GM_COSINE_PPM {ppm}",
    "metric": "METRIC cycles {cycles}",
    "measured_end": "MERLIN_PROFILE measured end rc=0",
    "window_end": "MERLIN_WINDOW end label={label}",
}


def _printf(key: str, **conversions: str) -> str:
    """One C ``printf`` of a protocol line, with each field replaced by its conversion."""
    return UART[key].format(**conversions) if conversions else UART[key]


def program_steps(model: dict[str, Any]) -> list[tuple[str, str]]:
    """``(group, kind)`` per device group, in the order the program calls them."""
    return [(str(step["group"]), str(step["kind"])) for step in model["steps"]]


def uart_lines(
    steps: Sequence[tuple[str, str]],
    *,
    cycles: int,
    checksums: Mapping[str, int],
    argmax: int,
    want: int,
    cosine_ppm: int,
    digests: Mapping[str, int] | None = None,
    group_cycles: Mapping[str, int] | None = None,
    window_label: str = DEFAULT_WINDOW_LABEL,
) -> list[str]:
    """The UART this program prints, given what the device computed.  No hardware, no C, no guess.

    ``checksums`` is what each group's output SUMMED TO on the device -- the quantity a v2 policy
    declares an oracle for.  Passing an oracle set renders a correct run's UART; changing one entry
    renders the job-730 shape, a run whose argmax marker is byte-identical to a correct one's and
    whose arithmetic was wrong.  Both are UARTs this program could really emit, which is what makes
    the pair a test of the admission rule rather than of the fixture.

    ``steps`` is ``(group, kind)`` pairs -- :func:`program_steps` derives them from a model, and a
    validation policy's declared groups are the other source, so the UART a policy expects can be
    rendered without the capture that produced the policy.

    ``checksums`` is the ADDITIVE sum; ``digests`` the order-sensitive FNV-1a a cycle claim is
    admitted against (:func:`group_checksums` computes both).  A group with no digest offered
    renders ``fnv1a=UNKNOWN`` rather than being left off the line: the runs on record predate the
    digest and can only be re-framed with the numbers they published, and a rendering that quietly
    dropped the field would let a policy declaring ``fnv1a`` read a missing check as a satisfied
    one.  ``UNKNOWN`` is refused by the admission rule as a value that is not an integer, which is
    the fail-closed reading of "this run cannot answer that question".
    """
    order = [group for group, _kind in steps]
    missing = sorted(set(order) - set(checksums))
    if missing:
        raise ValueError(f"no checksum offered for group(s) {missing}; a partial UART is not one this program prints")
    timings = group_cycles or {}
    lines = [
        UART["invocations"],
        UART["window_begin"].format(label=window_label),
        UART["warm_begin"],
        UART["warm_end"],
        UART["measured_begin"],
    ]
    offered = dict(digests or {})
    lines.extend(
        UART["group"].format(
            group=group,
            kind=kind,
            cycles=int(timings.get(group, 1)),
            sum=int(checksums[group]),
            checksum=int(offered[group]) if group in offered else "UNKNOWN",
        )
        for group, kind in steps
    )
    lines.extend(
        (
            UART["argmax"].format(got=int(argmax), want=int(want), agrees=int(argmax == want)),
            UART["cosine"].format(ppm=int(cosine_ppm)),
            UART["metric"].format(cycles=int(cycles)),
            UART["measured_end"],
            UART["window_end"].format(label=window_label),
        )
    )
    return lines


def render(
    model: dict[str, Any],
    *,
    sched_kernels: dict[str, Any] | None = None,
    window_label: str = DEFAULT_WINDOW_LABEL,
) -> str:
    """The C program: one call per group, each timed, then the comparison with the golden.

    The call is the vendor library's unless ``sched_kernels`` supplies one of our own for that group
    (see :mod:`group_model_sched_kernels`). Groups we cannot express keep the library call, so a run
    is never quietly a mixture reported as one schedule -- the census says which is which.

    WHY THIS PRINTS ``METRIC cycles N`` AND NOT ITS OWN SPELLING.  Until 2026-09-19 the measured
    total left here as ``GROUP_MODEL_TOTAL cycles: %llu`` and no ``MERLIN_INVOCATIONS`` line was
    printed at all, so ``merlin.perf.firesim_receipt._verify_uart`` rejected this program's UART
    outright -- no run of this shape could ever have been sealed, whatever the queue did.  The fix
    is here rather than in the parser: ``METRIC cycles N`` and the byte-exact
    ``MERLIN_INVOCATIONS warmup=1 measured=1`` are not arbitrary constants, they are the lines a
    real FPGA run printed (pinned in ``merlin/tests/data/firesim_queue/job610_uart_marker_skeleton.log``)
    and the ones :mod:`merlin.perf.warm_profile_harness` generates for every other workload.  A
    parser taught to accept a second spelling is a parser with two places a *different* number can
    be read as the measurement -- and the byte-exactness has already earned its keep, catching the
    trailing ``batch=1`` that ``experiments/voyager_h2h/scripts/emit_gemmini_c.py`` appends.  This
    program was the thing that was wrong: it hand-rolled a ``main`` instead of printing the one
    protocol the repository has.

    The window frame (``MERLIN_WINDOW begin/end label=``) is printed even though this program
    measures exactly one window, so a solo UART is a strict prefix of the batched shape
    :mod:`merlin.perf.firesim_batch` reads and one admission rule serves both.
    """
    if not isinstance(window_label, str) or not window_label.strip() or window_label != window_label.strip():
        raise ValueError("the measured window's label must be a nonempty, unpadded token")
    if any(character.isspace() for character in window_label) or "=" in window_label:
        raise ValueError("the measured window's label is one whitespace-free token without '='")
    blobs = "\n".join(
        f'BLOB({name}, "{name}.bin", {"acc_t" if name.startswith("BIAS_") else "float" if name == "GOLDEN" else "elem_t"})'
        for name in model["arrays"]
    )
    buffers = "\n".join(f"static {b['ctype']} {b['name']}[{b['elements']}] row_align(1);" for b in model["buffers"])
    window = max([s["window"] for s in model["steps"] if s["kind"] == "mean"] or [1])
    sizes = {b["name"]: b["elements"] for b in model["buffers"]}
    ours = (sched_kernels or {}).get("calls") or {}
    definitions = (sched_kernels or {}).get("definitions") or ""
    calls = "\n".join(
        f"    t0 = read_cycles(); {ours.get(step['group']) or _call(step)} dt = read_cycles() - t0; total += dt;\n"
        f'    if (measured) printf("'
        + _printf("group", group=step["group"], kind=step["kind"], cycles="%llu", sum="%lld", checksum="%lld")
        + '\\n", (unsigned long long)dt,\n'
        + f"                         sum_{'acc' if step.get('scale', 0) is None else 'elem'}"
        + f"({step['out']}, {sizes[step['out']]}),\n"
        + f"                         checksum_{'acc' if step.get('scale', 0) is None else 'elem'}"
        + f"({step['out']}, {sizes[step['out']]}));"
        for step in model["steps"]
    )
    final = model["steps"][-1]
    _fnv_offset, _fnv_prime, _digest_mask = _digest_constants()
    return f"""/* GENERATED by group_model_program.py -- one library call per compute group of a captured model. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include "include/gemmini_testutils.h"

#define BLOB(sym, file, type) \\
    __asm__(".section .rodata\\n.balign 64\\n.global " #sym "\\n" #sym ":\\n.incbin \\"" file "\\"\\n.previous\\n"); \\
    extern const type sym[];
{blobs}

#define IMAGE ((const elem_t *)IMAGE_DATA)
static elem_t ONES[{window}] row_align(1);
{buffers}

/* Outside the timed window: two numbers per group, which the numpy emulation of the same steps
   reproduces, so the first group that differs is named and not searched for.

   `sum_*` is the additive total this program has always printed. It is kept ONLY so a new build can
   be compared with the FPGA runs already on record, and it is not what a cycle claim is admitted
   against: reorder a group's output and the sum does not move, which makes it blind to exactly the
   transpose / stride / layout failures a tensor compiler produces.

   `checksum_*` is FNV-1a over the output, each element sign-extended to a 64-bit word -- the same
   digest, from the same constants, the hardened per-layer route uses (`lb_fnv1a_words`). Widening
   to 64 bits keeps it independent of how wide elem_t and acc_t are, so one host function
   (`group_digest`) reproduces both. Same linear cost as the sum, and outside the timed window in
   any case. */
static long long sum_elem(const elem_t *v, size_t n) {{ long long s = 0; for (size_t i = 0; i < n; i++) s += v[i]; return s; }}
static long long sum_acc(const acc_t *v, size_t n) {{ long long s = 0; for (size_t i = 0; i < n; i++) s += v[i]; return s; }}
static long long checksum_elem(const elem_t *v, size_t n) {{
    uint64_t h = {_fnv_offset}ULL;
    for (size_t i = 0; i < n; i++) {{ h ^= (uint64_t)(int64_t)v[i]; h *= {_fnv_prime}ULL; }}
    return (long long)(h & {_digest_mask}ULL);
}}
static long long checksum_acc(const acc_t *v, size_t n) {{
    uint64_t h = {_fnv_offset}ULL;
    for (size_t i = 0; i < n; i++) {{ h ^= (uint64_t)(int64_t)v[i]; h *= {_fnv_prime}ULL; }}
    return (long long)(h & {_digest_mask}ULL);
}}

/* Our own schedules for the groups we can express; the rest keep their library call above. */
{definitions}

static uint64_t run(int measured) {{
    uint64_t total = 0, t0, dt;
{calls}
    return total;
}}

int main(void) {{
    for (size_t i = 0; i < {window}; i++) ONES[i] = 1;
    gemmini_flush(0);
    printf("{_printf("invocations")}\\n");
    printf("{_printf("window_begin", label=window_label)}\\n");
    printf("{_printf("warm_begin")}\\n");
    run(0);
    printf("{_printf("warm_end")}\\n");
    printf("{_printf("measured_begin")}\\n");
    uint64_t total = run(1);
    /* The model's output against the capture's own golden. Not byte equality: a residual sum on
       this unit rounds each operand, within the bound its group declares. */
    int got = 0, want = 0;
    double dot = 0, mine = 0, theirs = 0;
    for (int i = 0; i < {model["classes"]}; i++) {{
        double y = (double){final["out"]}[i] * {final["dequantize"]!r};
        if ({final["out"]}[i] > {final["out"]}[got]) got = i;
        if (GOLDEN[i] > GOLDEN[want]) want = i;
        dot += y * GOLDEN[i]; mine += y * y; theirs += (double)GOLDEN[i] * GOLDEN[i];
    }}
    double cosine = dot / (sqrt(mine) * sqrt(theirs) + 1e-30);
    printf("{_printf("argmax", got="%d", want="%d", agrees="%d")}\\n", got, want, got == want);
    printf("{_printf("cosine", ppm="%d")}\\n", (int)(cosine * 1000000.0));
    printf("{_printf("metric", cycles="%llu")}\\n", (unsigned long long)total);
    printf("{_printf("measured_end")}\\n");
    printf("{_printf("window_end", label=window_label)}\\n");
    return 0;
}}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiler_provenance() -> dict[str, Any]:
    """Which lowering and schedule sources this build actually IMPORTED, by content.

    A result recorded without this cannot say what compiled it. Measured 2026-09-18: tracing the
    25.4M-cycle ResNet-50 result back to its compiler took an exhaustive scan of every file in the
    repo, and it succeeded only because one emitted recipe name happened to be unique to one
    worktree -- nothing anywhere recorded the ``PYTHONPATH``, the package path or a source digest.
    The same model compiled from two checkouts is two different programs, and the manifest said
    nothing that would tell them apart.

    Recorded as the FILES PYTHON RESOLVED, not as a path we expect it to have resolved, so a
    shadowing checkout or a dirty tree shows up as a different digest rather than as the same one.
    """
    import merlin
    from merlin.common import provenance as PROV

    resolved: dict[str, Path] = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("merlin.") or module is None:
            continue
        origin = getattr(module, "__file__", None)
        if origin and Path(origin).is_file():
            resolved[name] = Path(origin).resolve()
    try:
        digest = PROV.source_digest(sorted(resolved.values()))
    except Exception as exc:  # noqa: BLE001 -- an undeterminable digest is recorded, never omitted
        digest = f"UNKNOWN: {type(exc).__name__}: {exc}"
    return {
        "merlin_package": str(Path(merlin.__file__).resolve().parent),
        "source_digest": digest,
        "modules": {name: _sha256(path) for name, path in resolved.items()},
        "python": sys.executable,
    }


def build(
    model: dict[str, Any],
    vendor: Path,
    compiler: Path,
    out: Path,
    *,
    sched_kernels: dict[str, Any] | None = None,
    window_label: str = DEFAULT_WINDOW_LABEL,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    for name, array in model["arrays"].items():
        (out / f"{name}.bin").write_bytes(array.tobytes())
    program, elf = out / "group_model_program.c", out / "group_model_program.elf"
    program.write_text(render(model, sched_kernels=sched_kernels, window_label=window_label), encoding="utf-8")
    common = vendor / "riscv-tests" / "benchmarks" / "common"
    includes = [f"-I{vendor / 'riscv-tests'}", f"-I{vendor / 'riscv-tests' / 'env'}", f"-I{vendor}", f"-I{common}"]
    # Two phases with NAMED objects. A one-step compile-and-link records the driver's temporary
    # object names in the executable, so two builds of identical sources differ, and a measurement
    # could not be tied to a rebuild of its own program.
    log, objects = [], []
    for source in (program, common / "syscalls.c", common / "crt.S"):
        unit = out / f"{source.stem}.o"
        done = subprocess.run(
            [str(compiler), *CFLAGS, *includes, f"-Wa,-I{out}", "-c", str(source), "-o", str(unit)],
            capture_output=True,
            text=True,
            cwd=out,
        )
        log.append(done.stdout + done.stderr)
        if done.returncode != 0:
            (out / "build.log").write_text("".join(log), encoding="utf-8")
            raise SystemExit(f"compile of {source.name} failed (see {out / 'build.log'}):\n{done.stderr[-2000:]}")
        objects.append(str(unit))
    done = subprocess.run(
        [str(compiler), *CFLAGS, "-T", str(common / "test.ld"), *objects, "-o", str(elf)],
        capture_output=True,
        text=True,
        cwd=out,
    )
    (out / "build.log").write_text("".join(log) + done.stdout + done.stderr, encoding="utf-8")
    if done.returncode != 0:
        raise SystemExit(f"link failed (see {out / 'build.log'}):\n{done.stderr[-2000:]}")
    return {
        "elf": str(elf),
        "elf_sha256": _sha256(elf),
        "program_sha256": _sha256(program),
        "parameter_header_sha256": _sha256(vendor / "include" / "gemmini_params.h"),
        "compiler": str(compiler),
        "flags": list(CFLAGS),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", required=True, type=Path, help="a capture directory (linalg.mlir, weights, ...)")
    parser.add_argument("--target", required=True)
    parser.add_argument("--vendor-source", required=True, type=Path, help="a snapshot's source/ directory")
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--schedules",
        choices=["vendor", "ours"],
        default="vendor",
        help="'ours' renders each group our recipes can express as our own schedule, keeping the "
        "library call for the rest; the census records which group went which way",
    )
    parser.add_argument(
        "--window-label",
        default=DEFAULT_WINDOW_LABEL,
        help="the label of the one measured window this program frames; a UART validation policy "
        "declares the same label, which is what binds a cycle number to what it had to print",
    )
    args = parser.parse_args(argv)
    try:
        model = extract(args.capture, args.target)
    except NotClosed as refusal:
        print(f"not built: {refusal}", file=sys.stderr)
        return 2
    sched_kernels = None
    if args.schedules == "ours":
        # A sibling script, not a package module: resolved from this file rather than from whatever
        # sys.path happens to hold, so it works the same when imported as when run directly.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from group_model_sched_kernels import render_kernels, summarize

        sched_kernels = render_kernels(model, target=args.target)
        print(summarize(sched_kernels["census"]))
    receipt = build(
        model,
        args.vendor_source,
        args.compiler,
        args.out,
        sched_kernels=sched_kernels,
        window_label=args.window_label,
    )
    kinds: dict[str, int] = {}
    for step in model["steps"]:
        kinds[step["kind"]] = kinds.get(step["kind"], 0) + 1
    manifest = {
        "schema": SCHEMA,
        "capture": str(args.capture),
        "capture_linalg_sha256": _sha256(args.capture / "linalg.mlir"),
        "target": args.target,
        "groups": model["groups"],
        "device_groups": model["device_groups"],
        "steps_by_kind": kinds,
        "weight_bytes": sum(int(a.nbytes) for n, a in model["arrays"].items() if n.startswith("W_")),
        # Which groups ran OUR schedule and which kept the library call. A cycle total for this
        # program is only readable next to this: a mixture reported as one schedule is the way a
        # measurement gets attributed to work it did not do.
        "schedules": args.schedules,
        "schedule_census": (sched_kernels or {}).get("census"),
        # The window this program frames. A cycle claim is sealed against the policy entry carrying
        # this same label, so the two agreeing is a content fact and not a convention.
        "window_label": args.window_label,
        # WHAT COMPILED THIS. Recorded last, after every import this build needed has happened, so
        # the record is of the modules actually resolved rather than of the ones a path suggests.
        "compiler_provenance": _compiler_provenance(),
        "final_dequantize_f32_bits": struct.unpack("<I", struct.pack("<f", model["steps"][-1]["dequantize"]))[0],
        "sums": [
            {k: s[k] for k in ("group", "rows", "cols", "lhs_load", "rhs_load", "readout", "bound_lsb")}
            for s in model["steps"]
            if s["kind"] == "sum"
        ],
        **receipt,
    }
    (args.out / "group_model_program.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"built {receipt['elf']}: {model['device_groups']} device group(s) of {model['groups']}, {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
