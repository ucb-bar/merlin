"""Build a RoCC replay spec for the arcilator @Gemmini harness from a capsule's decoded trace + golden.

The decoded trace (`rocc_decode`) already carries raw rs1/rs2 per instruction as either a constant or an
argbase+offset (a DRAM pointer the kernel was passed). Since the arc harness controls DRAM placement, we
assign each tensor arg a base address, materialize its deterministic bytes, and emit the exact instruction
stream (funct, resolved rs1/rs2) the kernel would issue — plus the golden output to check. The harness
replays this into the isolated @Gemmini arc model. Part of #143-a.

CLI: gen_rocc_replay.py <capsule.yaml> <instruction_trace.json> --out replay.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from merlin.common.paths import repo_root

import numpy as np
import yaml

REPO = repo_root()
sys.path.insert(0, str(REPO / "merlin" / "python"))
from merlin.targetgen import capsule_golden as CG  # noqa: E402

# DRAM layout: give each arg a generous, 4 KB-aligned slab inside the harness's 64 MB buffer.
ARG_BASE = 0x100000   # 1 MB
ARG_STRIDE = 0x40000  # 256 KB per arg


def _arg_roles(capsule: dict) -> list[str]:
    """Kernel arg order = inputs (weight/input) then outputs, matching the emitted gemmini_kernel(...)."""
    roles = []
    for t in (capsule.get("inputs") or []):
        roles.append(t.get("name") or t.get("role"))
    for t in (capsule.get("outputs") or []):
        roles.append(t.get("name") or t.get("role"))
    return roles


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("capsule")
    ap.add_argument("trace")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    capsule = yaml.safe_load(Path(a.capsule).read_text())
    trace = json.loads(Path(a.trace).read_text())

    # arg_index -> DRAM base address
    n_args = 8
    arg_addr = {i: ARG_BASE + i * ARG_STRIDE for i in range(n_args)}

    # materialize deterministic input tensor bytes; place at their arg base.
    leaves = CG.materialize_capsule_leaves(capsule)        # name -> Tensor
    gold = CG.golden(capsule)                               # name -> nested list (e.g. Y0)
    # map arg_index -> tensor by the kernel arg order (inputs then outputs)
    # (we only need input bytes in DRAM; outputs are written by mvout.)
    placements = []
    inputs = capsule.get("inputs") or []
    op = (capsule.get("operation") or {}).get("op")
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    for i, t in enumerate(inputs):
        name = t.get("name") or t.get("role")
        if name not in leaves:
            continue
        # conv2d: the kernel mvin's the IM2COL'd ifm matrix [P, kh*kw*ci], not the raw NHWC IFM —
        # place that (col-padded below). Mirrors capsule_golden's conv path exactly.
        if op in ("conv2d", "conv") and (name == attrs.get("ifm") or t.get("role") == "input"):
            col = CG.im2col(leaves[name], ci=attrs["ci"], kh=attrs["kh"], kw=attrs["kw"],
                            stride=attrs["stride"], padding=attrs["padding"],
                            dilation=attrs["dilation"], layout=attrs.get("layout", "nhwc"))
            arr = np.asarray(col.data).reshape(col.shape)
        else:
            arr = np.asarray(leaves[name].data if hasattr(leaves[name], "data") else leaves[name])
        # The compiler lays each 2D operand in DRAM with its columns padded to a 16-multiple stride
        # (matches the kernel's CONFIG_LD stride; zeros in the pad don't affect the matmul). Naive
        # contiguous placement only works when cols is already a 16-multiple (e.g. A2 16x16). Pad here.
        # NB: materialized leaves are FLAT — reshape to the declared shape or the 2D padding is skipped.
        decl = t.get("shape")
        if decl and int(np.prod(decl)) == arr.size:
            arr = arr.reshape(decl)
        img = arr.astype(np.int8)
        if img.ndim == 2:
            rows, cols = img.shape
            pstride = ((cols + 15) // 16) * 16
            if pstride != cols:
                padded = np.zeros((rows, pstride), dtype=np.int8)
                padded[:, :cols] = img
                img = padded
            b = img.tobytes()
            placements.append({"arg_index": i, "name": name, "addr": arg_addr[i],
                               "bytes_hex": b.hex(), "shape": [rows, cols],
                               "row_stride": pstride, "dtype": "i8"})
        else:
            b = img.tobytes()  # non-2D (e.g. conv IFM pre-im2col) — placed raw; conv needs im2col image
            placements.append({"arg_index": i, "name": name, "addr": arg_addr[i],
                               "bytes_hex": b.hex(), "shape": list(img.shape), "dtype": "i8"})

    # output arg(s): the kernel args after the inputs are the outputs; their names = golden keys
    # (the capsule may not declare `outputs` explicitly — the command buffer's commit dst is the output).
    out_specs = []
    out_names = list((capsule.get("outputs") or []))
    if out_names:
        out_names = [t.get("name") or t.get("role") for t in out_names]
    else:
        out_names = list(gold.keys())   # e.g. ["Y0"]
    for j, name in enumerate(out_names):
        idx = len(inputs) + j
        g = gold.get(name)
        gflat = np.asarray(g).flatten().astype(np.int64).tolist() if g is not None else None
        out_specs.append({"arg_index": idx, "name": name, "addr": arg_addr[idx],
                          "golden_flat": gflat,
                          "shape": list(np.asarray(g).shape) if g is not None else None})

    # instruction stream: keep funct + rs1/rs2 (const raw, or argbase+offset to resolve at addr)
    insns = []
    for ins in trace["instructions"]:
        if ins.get("funct") is None:   # FENCE (inline-asm "fence") — model as a fence marker
            insns.append({"class": ins["class"], "funct": None})
            continue
        insns.append({"class": ins["class"], "funct": ins["funct"],
                      "rs1": ins.get("rs1"), "rs2": ins.get("rs2")})

    # output layout metadata for a generic readback: rows/cols (from golden), element bytes (i32=4/i8=1
    # from the MVOUT readout), and the DRAM row stride (CONFIG_ST out_stride_bytes; fallback cols*elem).
    elem_bytes, out_stride = 4, None
    for ins in trace["instructions"]:
        dec = ins.get("decoded") or {}
        if ins.get("class") == "CONFIG_ST" and dec.get("out_stride_bytes"):
            out_stride = dec["out_stride_bytes"]
        if ins.get("class") == "MVOUT":
            elem_bytes = 1 if dec.get("readout") == "i8" else 4
    for o in out_specs:
        rows, cols = (o["shape"] + [1, 1])[:2] if o["shape"] else (0, 0)
        o["rows"], o["cols"] = int(rows), int(cols)
        o["elem_bytes"] = elem_bytes
        o["stride_bytes"] = int(out_stride) if out_stride else int(cols) * elem_bytes

    spec = {"capsule": capsule.get("name"), "arg_addr": arg_addr,
            "placements": placements, "outputs": out_specs, "insns": insns,
            "rocc_opcode": 0x0b}
    Path(a.out).write_text(json.dumps(spec, indent=1))
    print(f"wrote {a.out}: {len(insns)} insns, {len(placements)} input placements, "
          f"{len(out_specs)} outputs; arg_addr={arg_addr if len(inputs)+len(capsule.get('outputs') or [])<=3 else '...'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
