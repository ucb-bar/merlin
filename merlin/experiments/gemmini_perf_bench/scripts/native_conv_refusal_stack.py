"""P1 -- the WHOLE refusal stack for the device convolution sequencer, not the first clause.

The selector is a guard sequence that returns on its first failing clause, so a census of refusals
counts first refusals and says nothing about the clauses never reached. Removing the top clause does
not admit those sites; it reveals the next one. This evaluates every clause INDEPENDENTLY for every
convolution, so the cascade is measured rather than predicted.

Clause set and field widths are transcribed from the selector that produced the recorded refusals
(`native_conv_selector.select_native_loop_conv`, the 90-line da07f9c8 revision). Nothing here is a
target fact: the geometry comes from the emission's own command buffer.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

CB = sys.argv[1]
OPERAND_DTYPE = "i8"  # F.OPERAND_DTYPE for this design, as the selector reads it
ACC_DTYPE = "i32"
NATIVE_LAYOUT = ("NHWC", "HWIO", "NHWC")
ALLOWED_STAGES = {"bias", "bias_add", "acc_scale", "relu"}
STAGE_RANK = {"bias": 0, "acc_scale": 1, "relu": 2}


def clauses(cmd, tensors):
    """Every clause this site fails, in selector order. Independent evaluation, no short circuit."""
    a = cmd.get("attributes") or {}
    ops = cmd.get("operands") or {}
    act = tensors.get(ops.get("ifm")) or {}
    wt = tensors.get(ops.get("weight")) or {}
    dst = tensors.get(ops.get("dst")) or {}
    out: list[str] = []

    dts = [a.get("output_dtype"), act.get("dtype"), wt.get("dtype"), dst.get("dtype")]
    if any(d is not None and d != OPERAND_DTYPE for d in dts):
        out.append("loop_conv_store_is_narrow_only")

    stages = list(a.get("epilogue") or [])
    if any(s not in ALLOWED_STAGES for s in stages):
        out.append("native_loop_conv_epilogue_not_representable")
    ordered = ["bias" if s == "bias_add" else s for s in stages]
    if ordered and (
        len(set(ordered)) != len(ordered) or ordered != sorted(ordered, key=lambda s: STAGE_RANK.get(s, 99))
    ):
        out.append("native_loop_conv_epilogue_order_must_be_bias_scale_activation")

    layout = str(a.get("layout") or "")
    # The emission records ONE layout string for the operation; the selector requires the triple.
    if layout.upper() != "NHWC":
        out.append("native_layout_contract_not_satisfied")

    kh, kw = (a.get("kernel") or [None, None, None, None])[:2]
    sh, sw = (a.get("stride") or [None, None])[:2]
    dh, dw = (a.get("dilation") or [None, None])[:2]
    pad = a.get("padding") or []
    if None in (kh, kw, sh, sw, dh, dw) or kh != kw or sh != sw or dh != dw:
        out.append("loop_conv_requires_uniform_2d_geometry")
    if len(pad) != 4 or len(set(pad)) != 1:
        out.append("loop_conv_requires_uniform_padding")
    elif not (0 <= pad[0] < (kh or 1)):
        out.append("native_loop_conv_requires_0_le_padding_lt_kernel")

    ci, co = (a.get("kernel") or [None] * 4)[2:4]
    ashape = act.get("shape") or []
    dshape = dst.get("shape") or []
    # NHWC: activation [batch, hi, wi, ci]; the emission's shapes are the ground truth.
    if len(ashape) == 4 and ci is not None and ashape[-1] != ci:
        out.append("buffer_shape_does_not_match_native_layout")
    elif len(dshape) == 4 and co is not None and dshape[-1] != co:
        out.append("buffer_shape_does_not_match_native_layout")

    if len(ashape) == 4 and len(dshape) == 4:
        fields = [*ashape, *dshape, kh or 0, kw or 0]
        if any(isinstance(v, int) and v >= (1 << 16) for v in fields):
            out.append("native_loop_conv_16bit_field_overflow")
    if isinstance(sh, int) and (sh >= (1 << 8) or (pad and pad[0] >= (1 << 8))):
        out.append("native_loop_conv_8bit_field_overflow")
    if isinstance(dh, int) and dh >= (1 << 10):
        out.append("kernel_dilation_not_representable")
    if None not in (kh, kw, ci) and kh * kw * ci * 128 * 128 > (1 << 31) - 1:
        out.append("native_loop_conv_accumulator_may_overflow_i32")
    return out


def main():
    doc = json.load(open(CB))
    tensors = doc.get("tensors") or {}
    convs = [c for c in doc.get("commands") or [] if c.get("opcode") == "CONV2D"]
    per_site, depth, first, every = [], Counter(), Counter(), Counter()
    for i, c in enumerate(convs):
        cl = clauses(c, tensors)
        per_site.append(
            {
                "site": i,
                "kernel": (c["attributes"] or {}).get("kernel"),
                "stride": (c["attributes"] or {}).get("stride"),
                "padding": (c["attributes"] or {}).get("padding"),
                "layout": (c["attributes"] or {}).get("layout"),
                "output_dtype": (c["attributes"] or {}).get("output_dtype"),
                "failing_clauses": cl,
            }
        )
        depth[len(cl)] += 1
        if cl:
            first[cl[0]] += 1
        for name in cl:
            every[name] += 1

    print(f"convolutions: {len(convs)}")
    print(f"sites admitted (no failing clause): {depth.get(0, 0)}")
    print("\nSTACK DEPTH  (how many clauses each site fails)")
    for d in sorted(depth):
        print(f"  {d} clause(s): {depth[d]} site(s)")
    print("\nFIRST refusal -- what a short-circuiting census reports")
    for name, n in first.most_common():
        print(f"  {n:3d}  {name}")
    print("\nEVERY failing clause -- what actually has to change")
    for name, n in every.most_common():
        print(f"  {n:3d}  {name}")
    out = {
        "schema": "native_conv_refusal_stack_v1",
        "command_buffer": CB,
        "convolutions": len(convs),
        "admitted": depth.get(0, 0),
        "stack_depth": {str(k): v for k, v in sorted(depth.items())},
        "first_refusal": dict(first),
        "every_clause": dict(every),
        "per_site": per_site,
    }
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    if dest:
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
