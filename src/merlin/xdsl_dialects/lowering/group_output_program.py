"""State a source-proved output cut without re-inferring its gains or integer offsets.

Root geometry and weight orientation use the ordinary contraction statement. The
epilogue instead consumes the exact output-fusion theorem and its declared numeric
formats. This supplies no target implementation or schedule.
"""

from __future__ import annotations

import math
import struct
from dataclasses import replace

from . import compute_groups as CG
from . import group_command as GC


def _refuse(reason):
    raise CG.NoCapsuleForm(f"the selected output-fusion statement: {reason}")


def _integer_offsets(proof, columns):
    offsets = proof.get("offsets")
    if not isinstance(offsets, list) or len(offsets) != columns:
        _refuse("its proved offsets do not match the device output columns")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in offsets):
        _refuse("its proved offsets are not integers")
    if CG._dtype_token((proof.get("accumulator") or {}).get("dtype")) != CG._dtype_token("i32"):
        _refuse("the command-buffer accumulator format is i32; the proof declares another format")
    if any(v < -(1 << 31) or v >= (1 << 31) for v in offsets):
        _refuse("a proved offset is outside the declared i32 accumulator")
    preload = proof.get("preload") or {}
    if any(offsets):
        span = preload.get("range")
        if (CG._dtype_token(preload.get("dtype")) != CG._dtype_token("i32")
                or not isinstance(span, list) or len(span) != 2):
            _refuse("nonzero offsets have no declared integer preload range")
        if any(v < span[0] or v > span[1] for v in offsets):
            _refuse("a proved offset is outside the declared preload range")
    return offsets


def _readout(proof, stages, columns):
    readout = proof.get("readout_stages")
    if not isinstance(readout, list):
        _refuse("the exact readout formats are absent")
    by_kind = {s.get("kind"): s for s in readout}
    if len(by_kind) != len(readout):
        _refuse("repeated readout stages have no command-buffer statement")
    if set(by_kind) - {"activation", "convert", "scale", "requant", "pool"}:
        _refuse("an unknown readout stage has no command-buffer statement")
    for kind, field in (("convert", "to"), ("scale", "dtype")):
        stage = by_kind.get(kind) or {}
        if CG._dtype_token(stage.get(field)) != CG._dtype_token("f32") or stage.get("rounding") != "half_even":
            _refuse(f"{kind} does not declare the command-buffer f32 half-even arithmetic")
    requant = by_kind.get("requant") or {}
    if (CG._dtype_token(requant.get("to")) != CG._dtype_token("i8") or requant.get("rounding") != "half_even"
            or requant.get("clamp") != [-128, 127]):
        _refuse("the readout is not the command-buffer full-range i8 half-even narrowing")
    kinds = [s["kind"] for s in readout]
    if not kinds.index("convert") < kinds.index("scale") < kinds.index("requant"):
        _refuse("the declared conversion, scale and narrowing are in another order")
    if "activation" in by_kind and "relu" in stages:
        activation = by_kind["activation"]
        if activation.get("function") != "relu" or kinds.index("activation") > kinds.index("requant"):
            _refuse("the declared activation is not a rectifier before narrowing")
    gains = proof.get("device_scale")
    if not isinstance(gains, list) or len(gains) != columns:
        _refuse("its proved gains do not match the device output columns")
    for gain in gains:
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            _refuse("a proved gain is not positive and finite")
        try:
            rounded = struct.unpack("<f", struct.pack("<f", gain))[0]
        except (OverflowError, struct.error):
            _refuse("a proved gain is not representable as f32")
        if rounded != gain:
            _refuse("a proved gain would change under command-buffer f32 conversion")
    # These are legality facts from the selected proof, never a guessed mesh width.
    scale = by_kind["scale"]
    granularity = scale.get("granularity")
    if granularity == "per_tensor":
        width = columns
    elif granularity == "per_channel":
        width = 1
    elif granularity == "per_command":
        width = scale.get("columns_per_scale")
    else:
        _refuse("the gain granularity is unknown")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        _refuse("the gain-block width is not a positive integer")
    if any(len(set(gains[start:start + width])) != 1 for start in range(0, columns, width)):
        _refuse("proved gains vary inside a declared gain block")
    return {"acc_scale": gains[0]} if len(set(gains)) == 1 else {"acc_scale_by_column": list(gains)}


def program(group, *, weight_args=None, name=None, extents=None):
    """The same device geometry, with the theorem's exact output arithmetic."""
    fusion = group.fusion or {}
    cut, proof, stages = fusion.get("cut"), fusion.get("proof") or {}, fusion.get("stages") or []
    if cut not in ("accumulator", "requant", "pool"):
        _refuse(f"unknown cut {cut!r}")
    if "residual" in stages:
        _refuse("a fused residual needs an explicit accumulator operand interface")
    prefix = group.members[:group.members.index(group.root) + 1]
    root_group = replace(group, members=prefix, stages=[CG.classify(op).kind for op in prefix], fusion=None)
    root = GC.program(root_group, weight_args=weight_args, name=name, extents=extents)
    if root.stored_operand is not None and root.stored_arg is None:
        stored, argument = GC.stored_operand(root_group, weight_args)
        if stored != root.stored_operand:
            _refuse("the stored argument does not match the native root's operand orientation")
        root = replace(root, stored_arg=argument)
    entry = dict(root.entry)
    columns = int(entry["N"])
    if proof.get("channels") != columns:
        _refuse("the proof's channel axis is not the device's output-column axis")
    offsets = _integer_offsets(proof, columns)
    epilogue = ["bias_add"] if any(offsets) else []
    if cut != "accumulator":
        entry.update(_readout(proof, stages, columns))
        epilogue.append("acc_scale")
        if "relu" in stages:
            epilogue.append("relu")
        if cut == "pool":
            if entry["op"] != "conv2d":
                _refuse("a pooled contraction needs an explicitly stated input plane")
            entry.update(GC._pool(group))
            epilogue.append("maxpool")
    entry["epilogue"] = epilogue
    entry["output_dtype"] = "i32" if cut == "accumulator" else "i8"
    return replace(root, entry=entry, notes=(*root.notes, "output arithmetic uses the source-bound exact fusion proof"))


def prepack(group, stated, manifest, weights, *, device_layout=False):
    """Pack the theorem's exact offsets; never divide the original float bias again."""
    import hashlib

    import numpy as np

    from . import group_numerics as GN
    from . import group_prepack as GP

    proof = group.fusion["proof"]
    offsets = _integer_offsets(proof, int(stated.entry["N"]))
    row = {
        "group": group.index,
        "stages": list(group.stages),
        "leaves_as": "accumulator" if group.fusion["cut"] == "accumulator" else "requant",
        "output_fusion": group.fusion,
    }
    arrays = {}
    for key in ("acc_scale", "acc_scale_by_column"):
        if key in stated.entry:
            row[key] = stated.entry[key]
    if "bias_add" in stated.entry["epilogue"]:
        folded = np.asarray(offsets, dtype="<i4")
        key = f"group_{group.index}_bias_q"
        arrays[key] = folded
        row["bias"] = {
            "fold_rule": "source_bound_output_fusion_offsets",
            "array": key,
            "elements": int(folded.size),
            "max_abs": max(abs(v) for v in offsets),
            "sha256": hashlib.sha256(folded.tobytes()).hexdigest(),
        }
    if device_layout and stated.stored_operand is not None:
        name, stored, dtype = GP._stored_raw(stated.stored_arg, manifest, weights)
        if dtype not in GP._INTEGER_DTYPE:
            raise GN.GroupNumericsError(f"stored tensor {name!r} is {dtype}, not an integer weight")
        laid_out = GP.device_weight(group, stated, stored)
        key = f"group_{group.index}_weight_device"
        arrays[key] = laid_out
        row["weight"] = {
            "stored_tensor": name,
            "arg_index": stated.stored_arg,
            "array": key,
            "shape": [int(v) for v in laid_out.shape],
            "transposed": stated.transposed,
            "column_order": list(stated.column_order) if stated.column_order else None,
            "sha256": hashlib.sha256(laid_out.tobytes()).hexdigest(),
        }
    return row, arrays
