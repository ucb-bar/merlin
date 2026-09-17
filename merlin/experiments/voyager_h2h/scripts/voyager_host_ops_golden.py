"""Record Voyager's own semantics for the ops the bridge leaves on the host (concession C6).

Runs inside Voyager's environment (``out/build/voyager-venv``) and imports nothing from merlin. Each op
is called the way Voyager's bufferized graph calls it -- the ``quantized_ops`` library the pinned
compiler registers, or ``aten`` where the graph itself uses aten -- on seeded random inputs chosen to
exercise rounding ties, saturation, integers past bfloat16's precision, and a padded pooling window.
The merlin side checks its numpy semantics against this file (``merlin/tests/ir``).

Writes ``golden.npz`` (bfloat16 values widened to float32, integers as int64) and ``manifest.json``
(compiler revision, torch version, and the exact call behind every entry) into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch


def _np(t: torch.Tensor) -> np.ndarray:
    return (t.float() if t.is_floating_point() else t.to(torch.int64)).numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import voyager_compiler  # noqa: F401 -- registers the quantized_ops library
    from voyager_compiler.quantization.fake_quantize import get_quantization_map

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    bf16 = torch.bfloat16
    ops = torch.ops.quantized_ops
    data, calls = {}, {}

    # quantize: x / scale in bfloat16, then the int8 table (round half to even, saturate).
    scale = torch.tensor(0.01184082, dtype=bf16)
    x = (torch.randn(8192) * 2.0).to(bf16)
    data.update(
        quantize_x=_np(x),
        quantize_scale=_np(scale),
        quantize_y=_np(ops.quantize(x, scale, qmap=get_quantization_map("int8"))),
    )
    calls["quantize"] = "quantized_ops.quantize(x, scale, qmap=get_quantization_map('int8'))"

    # dequantize: int32 input (including magnitudes past 2**24) times a bfloat16 scale.
    xi = torch.randint(-(2**27), 2**27, (8192,), dtype=torch.int32)
    dscale = torch.tensor(0.00024986, dtype=bf16)
    y = ops.dequantize(xi, dscale)
    data.update(dequantize_x=_np(xi), dequantize_scale=_np(dscale), dequantize_y=_np(y))
    calls["dequantize"] = f"quantized_ops.dequantize(int32 x, bf16 scale) -> {y.dtype}"

    # max_pool2d on an NHWC tile padded the way async_copy pads it: the leading pad per dim filled
    # with the pad value, the window then read from the padded tile.
    src = torch.randn(1, 11, 13, 16).to(bf16)
    pad_before, pad_value = (0, 1, 1, 0), float("-inf")
    tile = torch.full((1, 12, 14, 16), pad_value, dtype=bf16)
    tile[:, 1:, 1:, :] = src
    y = ops.max_pool2d(tile, [3, 3], [2, 2], [0, 0], [1, 1], False)
    data.update(max_pool_x=_np(src), max_pool_y=_np(y))
    calls["max_pool2d"] = (
        "quantized_ops.max_pool2d(tile, [3,3], [2,2], [0,0], [1,1], False) on a "
        f"tile padded before by {pad_before} with {pad_value}"
    )

    # adaptive_avg_pool2d, the global (1, 1) case ResNet uses and a general (2, 3) window grid.
    a = torch.randn(1, 7, 7, 64).to(bf16) * 4
    b = torch.randn(1, 6, 5, 16).to(bf16)
    data.update(
        avg_pool_x=_np(a),
        avg_pool_y=_np(ops.adaptive_avg_pool2d(a, [1, 1])),
        avg_pool23_x=_np(b),
        avg_pool23_y=_np(ops.adaptive_avg_pool2d(b, [2, 3])),
    )
    calls["adaptive_avg_pool2d"] = "quantized_ops.adaptive_avg_pool2d(x, [1,1]) and (x, [2,3])"

    # linear as the graph's classifier runs it: int8 activations and weights in bfloat16, an int32
    # bias; outputs pass bfloat16's precision. A bias bfloat16 cannot represent separates the two
    # readings (bias rounded to bfloat16 first, or added exactly before the one output rounding).
    lx = torch.randint(-128, 128, (1, 512)).to(bf16)
    lw = torch.randint(-128, 128, (48, 512), dtype=torch.int8)
    small = torch.randint(-200, 200, (48,), dtype=torch.int32)
    big = torch.randint(-(2**20), 2**20, (48,), dtype=torch.int32)
    data.update(
        linear_x=_np(lx),
        linear_w=_np(lw),
        linear_b_small=_np(small),
        linear_b_big=_np(big),
        linear_y_small=_np(torch.ops.aten.linear(lx, lw.to(bf16), small.to(bf16))),
        linear_y_big_bf16bias=_np(torch.ops.aten.linear(lx, lw.to(bf16), big.to(bf16))),
        linear_y_big_fp32=_np(torch.ops.aten.linear(lx.float(), lw.float(), big.float()).to(bf16)),
    )
    calls["linear"] = (
        "aten.linear(x_bf16, w.to(bf16), b.to(bf16)) [small, big bias] and aten.linear(fp32...).to(bf16) [big bias]"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "golden.npz", **data)
    root = Path(os.environ["MERLIN_EXT_VOYAGER_COMPILER"]).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {"voyager_commit": commit, "torch": torch.__version__, "seed": args.seed, "calls": calls}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "entries": sorted(data)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
