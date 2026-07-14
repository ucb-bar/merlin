"""Kernel-sized workload generator — single compiler-emitted ops at curated-kernel shapes.

These are NOT hand-written kernels: each is a tiny linalg-on-tensors module the COMPILER lowers
through the RVV package (the whole point — we measure compiler codegen, then compare it to the
expert kernel at the same op-shape). A workload bundle matches the capture format the runtime
consumes (model.mlir + weights.safetensors[.manifest.json] + inputs.npz + golden.npy +
input_order.json), but with no weights: every operand is a function input.

Bundle key is ``<op>_<dtype>_<MxNxK>`` so it joins the kernel-ceiling + comparison fingerprint.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# safetensors with an empty header: u64 LE length=2, then b"{}", then empty payload.
_EMPTY_SAFETENSORS = struct.pack("<Q", 2) + b"{}"


def _write_empty_weights(bundle: Path) -> None:
    (bundle / "weights.safetensors").write_bytes(_EMPTY_SAFETENSORS)


def _finish(bundle: Path, mlir: str, manifest: dict, order: dict,
            inputs: dict[str, np.ndarray], golden: np.ndarray) -> Path:
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "model.mlir").write_text(mlir, encoding="utf-8")
    _write_empty_weights(bundle)
    (bundle / "weights.safetensors.manifest.json").write_text(json.dumps(manifest, indent=2))
    (bundle / "input_order.json").write_text(json.dumps(order, indent=2))
    np.savez(bundle / "inputs.npz", **inputs)
    np.save(bundle / "golden.npy", golden.astype(np.float32))
    return bundle


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_matmul_f32(out_root: str | Path, M: int = 64, N: int = 64, K: int = 64,
                   seed: int = 0) -> Path:
    """A single fp32 ``linalg.matmul`` (M,K)x(K,N)->(M,N). Both operands are inputs."""
    bundle = Path(out_root) / f"matmul_f32_{M}x{N}x{K}"
    sf = bundle / "weights.safetensors"
    mlir = (
        f'builtin.module attributes {{prov.weights_file = "{sf}", '
        'prov.level = "linalg-on-tensors"} {\n'
        f"  func.func @forward(%a: tensor<{M}x{K}xf32>, %b: tensor<{K}x{N}xf32>) "
        f"-> tensor<{M}x{N}xf32> {{\n"
        "    %cst = arith.constant 0.000000e+00 : f32\n"
        f"    %0 = tensor.empty() : tensor<{M}x{N}xf32>\n"
        f"    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<{M}x{N}xf32>) "
        f"-> tensor<{M}x{N}xf32>\n"
        f"    %2 = linalg.matmul ins(%a, %b : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) "
        f"outs(%1 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>\n"
        f"    return %2 : tensor<{M}x{N}xf32>\n"
        "  }\n}\n"
    )
    r = _rng(seed)
    a = r.standard_normal((M, K)).astype(np.float32)
    b = r.standard_normal((K, N)).astype(np.float32)
    manifest = {"0": {"kind": "input", "name": "a"}, "1": {"kind": "input", "name": "b"}}
    order = {"a": 0, "b": 1}
    return _finish(bundle, mlir, manifest, order, {"in0": a, "in1": b}, (a @ b))


def gen_softmax_f32(out_root: str | Path, M: int = 64, N: int = 64, seed: int = 0) -> Path:
    """A single softmax over the last dim of an (M,N) input, emitted as linalg generics
    (max-reduce, sub+exp, sum-reduce, div). Exercises the reduction + transcendental path."""
    bundle = Path(out_root) / f"softmax_f32_{M}x{N}"
    sf = bundle / "weights.safetensors"
    mlir = (
        f'builtin.module attributes {{prov.weights_file = "{sf}", '
        'prov.level = "linalg-on-tensors"} {\n'
        f"  func.func @forward(%x: tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32> {{\n"
        "    %cneg = arith.constant 0xFF800000 : f32\n"
        "    %czero = arith.constant 0.000000e+00 : f32\n"
        f"    %mi = tensor.empty() : tensor<{M}xf32>\n"
        f"    %mf = linalg.fill ins(%cneg : f32) outs(%mi : tensor<{M}xf32>) -> tensor<{M}xf32>\n"
        f"    %mx = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
        "affine_map<(d0, d1) -> (d0)>], iterator_types = [\"parallel\", \"reduction\"]} "
        f"ins(%x : tensor<{M}x{N}xf32>) outs(%mf : tensor<{M}xf32>) {{\n"
        "    ^bb0(%in: f32, %o: f32):\n"
        "      %m = arith.maximumf %in, %o : f32\n"
        "      linalg.yield %m : f32\n"
        f"    }} -> tensor<{M}xf32>\n"
        f"    %ei = tensor.empty() : tensor<{M}x{N}xf32>\n"
        f"    %ex = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
        "affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], "
        "iterator_types = [\"parallel\", \"parallel\"]} "
        f"ins(%x, %mx : tensor<{M}x{N}xf32>, tensor<{M}xf32>) outs(%ei : tensor<{M}x{N}xf32>) {{\n"
        "    ^bb0(%in: f32, %m: f32, %o: f32):\n"
        "      %s = arith.subf %in, %m : f32\n"
        "      %e = math.exp %s : f32\n"
        "      linalg.yield %e : f32\n"
        f"    }} -> tensor<{M}x{N}xf32>\n"
        f"    %si = tensor.empty() : tensor<{M}xf32>\n"
        f"    %sf = linalg.fill ins(%czero : f32) outs(%si : tensor<{M}xf32>) -> tensor<{M}xf32>\n"
        f"    %sm = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
        "affine_map<(d0, d1) -> (d0)>], iterator_types = [\"parallel\", \"reduction\"]} "
        f"ins(%ex : tensor<{M}x{N}xf32>) outs(%sf : tensor<{M}xf32>) {{\n"
        "    ^bb0(%in: f32, %o: f32):\n"
        "      %a = arith.addf %in, %o : f32\n"
        "      linalg.yield %a : f32\n"
        f"    }} -> tensor<{M}xf32>\n"
        f"    %oi = tensor.empty() : tensor<{M}x{N}xf32>\n"
        f"    %r = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
        "affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], "
        "iterator_types = [\"parallel\", \"parallel\"]} "
        f"ins(%ex, %sm : tensor<{M}x{N}xf32>, tensor<{M}xf32>) outs(%oi : tensor<{M}x{N}xf32>) {{\n"
        "    ^bb0(%in: f32, %s: f32, %o: f32):\n"
        "      %d = arith.divf %in, %s : f32\n"
        "      linalg.yield %d : f32\n"
        f"    }} -> tensor<{M}x{N}xf32>\n"
        f"    return %r : tensor<{M}x{N}xf32>\n"
        "  }\n}\n"
    )
    r = _rng(seed)
    x = r.standard_normal((M, N)).astype(np.float32)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    golden = e / e.sum(axis=1, keepdims=True)
    manifest = {"0": {"kind": "input", "name": "x"}}
    order = {"x": 0}
    return _finish(bundle, mlir, manifest, order, {"in0": x}, golden)


def gen_gelu_f32(out_root: str | Path, N: int = 16384, seed: int = 0) -> Path:
    """A single f32 GELU activation over an N-vector, emitted as a linalg.generic whose body is
    the exact (erf) GELU: 0.5*x*(1+erf(x/sqrt(2))). This is the elementwise activation our RVV
    pipeline vectorizes (math.erf -> the libm/polynomial path); bandwidth-bound, so we sweep N
    rather than a cube. Compares head-to-head with XNNPACK f32-vgelu (rational-12-10)."""
    bundle = Path(out_root) / f"gelu_f32_{N}"
    sf = bundle / "weights.safetensors"
    mlir = (
        f'builtin.module attributes {{prov.weights_file = "{sf}", '
        'prov.level = "linalg-on-tensors"} {\n'
        f"  func.func @forward(%x: tensor<{N}xf32>) -> tensor<{N}xf32> {{\n"
        "    %c1 = arith.constant 1.000000e+00 : f32\n"
        "    %ch = arith.constant 5.000000e-01 : f32\n"
        "    %cs = arith.constant 0.7071067811865476 : f32\n"
        f"    %oi = tensor.empty() : tensor<{N}xf32>\n"
        f"    %r = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]} "
        f"ins(%x : tensor<{N}xf32>) outs(%oi : tensor<{N}xf32>) {{\n"
        "    ^bb0(%in: f32, %o: f32):\n"
        "      %t = arith.mulf %in, %cs : f32\n"
        "      %e = math.erf %t : f32\n"
        "      %p = arith.addf %e, %c1 : f32\n"
        "      %h = arith.mulf %p, %ch : f32\n"
        "      %y = arith.mulf %h, %in : f32\n"
        "      linalg.yield %y : f32\n"
        f"    }} -> tensor<{N}xf32>\n"
        f"    return %r : tensor<{N}xf32>\n"
        "  }\n}\n"
    )
    r = _rng(seed)
    x = (r.standard_normal((N,)).astype(np.float32) * 3.0)
    from math import erf as _erf  # numpy has no erf; use scipy-free vectorized form below
    ev = np.vectorize(_erf)(x / np.float32(np.sqrt(2.0)))
    golden = (0.5 * x * (1.0 + ev)).astype(np.float32)
    manifest = {"0": {"kind": "input", "name": "x"}}
    order = {"x": 0}
    return _finish(bundle, mlir, manifest, order, {"in0": x}, golden)


def gen_sigmoid_f32(out_root: str | Path, N: int = 16384, seed: int = 0) -> Path:
    """A single f32 sigmoid activation over an N-vector: 1/(1+exp(-x)), emitted as a
    linalg.generic. The elementwise transcendental our RVV pipeline vectorizes (math.exp).
    Bandwidth-bound -> sweep N. Compares head-to-head with XNNPACK f32-vsigmoid (rr2-p5)."""
    bundle = Path(out_root) / f"sigmoid_f32_{N}"
    sf = bundle / "weights.safetensors"
    mlir = (
        f'builtin.module attributes {{prov.weights_file = "{sf}", '
        'prov.level = "linalg-on-tensors"} {\n'
        f"  func.func @forward(%x: tensor<{N}xf32>) -> tensor<{N}xf32> {{\n"
        "    %c1 = arith.constant 1.000000e+00 : f32\n"
        "    %cn1 = arith.constant -1.000000e+00 : f32\n"
        f"    %oi = tensor.empty() : tensor<{N}xf32>\n"
        f"    %r = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]} "
        f"ins(%x : tensor<{N}xf32>) outs(%oi : tensor<{N}xf32>) {{\n"
        "    ^bb0(%in: f32, %o: f32):\n"
        "      %nx = arith.mulf %in, %cn1 : f32\n"
        "      %e = math.exp %nx : f32\n"
        "      %d = arith.addf %e, %c1 : f32\n"
        "      %y = arith.divf %c1, %d : f32\n"
        "      linalg.yield %y : f32\n"
        f"    }} -> tensor<{N}xf32>\n"
        f"    return %r : tensor<{N}xf32>\n"
        "  }\n}\n"
    )
    r = _rng(seed)
    x = (r.standard_normal((N,)).astype(np.float32) * 3.0)
    golden = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    manifest = {"0": {"kind": "input", "name": "x"}}
    order = {"x": 0}
    return _finish(bundle, mlir, manifest, order, {"in0": x}, golden)


def gen_silu_f32(out_root: str | Path, N: int = 16384, seed: int = 0) -> Path:
    """A single f32 SiLU/swish activation over an N-vector: x*sigmoid(x) = x/(1+exp(-x)), emitted
    as a linalg.generic. Like sigmoid it vectorizes ``math.exp``; included so the activation feature
    is exercised on a THIRD activation (general, not gelu/sigmoid-overfit). Bandwidth-bound -> sweep N."""
    bundle = Path(out_root) / f"silu_f32_{N}"
    sf = bundle / "weights.safetensors"
    mlir = (
        f'builtin.module attributes {{prov.weights_file = "{sf}", '
        'prov.level = "linalg-on-tensors"} {\n'
        f"  func.func @forward(%x: tensor<{N}xf32>) -> tensor<{N}xf32> {{\n"
        "    %c1 = arith.constant 1.000000e+00 : f32\n"
        "    %cn1 = arith.constant -1.000000e+00 : f32\n"
        f"    %oi = tensor.empty() : tensor<{N}xf32>\n"
        f"    %r = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]} "
        f"ins(%x : tensor<{N}xf32>) outs(%oi : tensor<{N}xf32>) {{\n"
        "    ^bb0(%in: f32, %o: f32):\n"
        "      %nx = arith.mulf %in, %cn1 : f32\n"
        "      %e = math.exp %nx : f32\n"
        "      %d = arith.addf %e, %c1 : f32\n"
        "      %s = arith.divf %c1, %d : f32\n"
        "      %y = arith.mulf %s, %in : f32\n"
        "      linalg.yield %y : f32\n"
        f"    }} -> tensor<{N}xf32>\n"
        f"    return %r : tensor<{N}xf32>\n"
        "  }\n}\n"
    )
    r = _rng(seed)
    x = (r.standard_normal((N,)).astype(np.float32) * 3.0)
    sig = 1.0 / (1.0 + np.exp(-x))
    golden = (x * sig).astype(np.float32)
    manifest = {"0": {"kind": "input", "name": "x"}}
    order = {"x": 0}
    return _finish(bundle, mlir, manifest, order, {"in0": x}, golden)


def gen_batch_matmul_f32(out_root: str | Path, B: int = 4, M: int = 32, N: int = 8, K: int = 32,
                         seed: int = 0) -> Path:
    """A single fp32 ``linalg.batch_matmul`` (B,M,K)x(B,K,N)->(B,M,N) — the attention shape. N can
    be SMALL (e.g. 8, a llama-style attention batch_matmul) to exercise the NR<=N / N-tail path."""
    bundle = Path(out_root) / f"bmm_f32_{B}x{M}x{N}x{K}"
    sf = bundle / "weights.safetensors"
    # Emit as a linalg.generic with batch-matmul indexing maps (b,m,n,k): (b,m,k)x(b,k,n)->(b,m,n).
    # The xdsl frontend has no custom format for linalg.batch_matmul (only matmul/generic), and this
    # generic IS the form attention batch_matmuls take after linalg generalization — the comparator's
    # reduction_to_contract rebuilds the vector.contract from it just like the named op.
    bm = (f"affine_map<(b, m, n, k) -> (b, m, k)>, "
          f"affine_map<(b, m, n, k) -> (b, k, n)>, "
          f"affine_map<(b, m, n, k) -> (b, m, n)>")
    mlir = (
        f'builtin.module attributes {{prov.weights_file = "{sf}", '
        'prov.level = "linalg-on-tensors"} {\n'
        f"  func.func @forward(%a: tensor<{B}x{M}x{K}xf32>, %b: tensor<{B}x{K}x{N}xf32>) "
        f"-> tensor<{B}x{M}x{N}xf32> {{\n"
        "    %cst = arith.constant 0.000000e+00 : f32\n"
        f"    %0 = tensor.empty() : tensor<{B}x{M}x{N}xf32>\n"
        f"    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<{B}x{M}x{N}xf32>) "
        f"-> tensor<{B}x{M}x{N}xf32>\n"
        f"    %2 = linalg.generic {{indexing_maps = [{bm}], "
        'iterator_types = ["parallel", "parallel", "parallel", "reduction"]} '
        f"ins(%a, %b : tensor<{B}x{M}x{K}xf32>, tensor<{B}x{K}x{N}xf32>) "
        f"outs(%1 : tensor<{B}x{M}x{N}xf32>) {{\n"
        "    ^bb0(%in: f32, %in_0: f32, %out: f32):\n"
        "      %m = arith.mulf %in, %in_0 : f32\n"
        "      %a2 = arith.addf %out, %m : f32\n"
        "      linalg.yield %a2 : f32\n"
        f"    }} -> tensor<{B}x{M}x{N}xf32>\n"
        f"    return %2 : tensor<{B}x{M}x{N}xf32>\n"
        "  }\n}\n"
    )
    r = _rng(seed)
    a = r.standard_normal((B, M, K)).astype(np.float32)
    b = r.standard_normal((B, K, N)).astype(np.float32)
    manifest = {"0": {"kind": "input", "name": "a"}, "1": {"kind": "input", "name": "b"}}
    order = {"a": 0, "b": 1}
    return _finish(bundle, mlir, manifest, order, {"in0": a, "in1": b},
                   np.matmul(a, b))


def gen_conv2d_as_matmul_f32(out_root: str | Path, M: int = 64, N: int = 16, K: int = 27,
                             seed: int = 0) -> Path:
    """A conv2d expressed (already im2col'd) as a single ``linalg.matmul`` whose K is the
    patch-volume (Cin*Kh*Kw) — the conv contraction the RVV path sees after im2col. M = output
    spatial positions, N = output channels, K = Cin*Kh*Kw (e.g. 3*3*3=27). Bundle name marks it as
    a conv-origin contraction so the coverage test knows what it is exercising."""
    bundle = Path(out_root) / f"conv2d_im2col_f32_{M}x{N}x{K}"
    sf = bundle / "weights.safetensors"
    mlir = (
        f'builtin.module attributes {{prov.weights_file = "{sf}", '
        'prov.level = "linalg-on-tensors"} {\n'
        f"  func.func @forward(%a: tensor<{M}x{K}xf32>, %b: tensor<{K}x{N}xf32>) "
        f"-> tensor<{M}x{N}xf32> {{\n"
        "    %cst = arith.constant 0.000000e+00 : f32\n"
        f"    %0 = tensor.empty() : tensor<{M}x{N}xf32>\n"
        f"    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<{M}x{N}xf32>) "
        f"-> tensor<{M}x{N}xf32>\n"
        f"    %2 = linalg.matmul ins(%a, %b : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) "
        f"outs(%1 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>\n"
        f"    return %2 : tensor<{M}x{N}xf32>\n"
        "  }\n}\n"
    )
    r = _rng(seed)
    a = r.standard_normal((M, K)).astype(np.float32)
    b = r.standard_normal((K, N)).astype(np.float32)
    manifest = {"0": {"kind": "input", "name": "a"}, "1": {"kind": "input", "name": "b"}}
    order = {"a": 0, "b": 1}
    return _finish(bundle, mlir, manifest, order, {"in0": a, "in1": b}, (a @ b))


_GENERATORS = {"matmul_f32": gen_matmul_f32, "softmax_f32": gen_softmax_f32,
               "batch_matmul_f32": gen_batch_matmul_f32,
               "conv2d_im2col_f32": gen_conv2d_as_matmul_f32,
               "gelu_f32": gen_gelu_f32, "sigmoid_f32": gen_sigmoid_f32,
               "silu_f32": gen_silu_f32}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Generate kernel-sized RVV workload bundles.")
    ap.add_argument("op", choices=sorted(_GENERATORS), help="workload op")
    ap.add_argument("--out-root", default="out/artifacts/kernel-mining/rvv/workloads")
    ap.add_argument("-M", type=int, default=64)
    ap.add_argument("-N", type=int, default=64)
    ap.add_argument("-K", type=int, default=64)
    ap.add_argument("-B", type=int, default=4, help="batch dim (batch_matmul_f32)")
    a = ap.parse_args(argv)
    fn = _GENERATORS[a.op]
    if a.op == "batch_matmul_f32":
        kw = {"B": a.B, "M": a.M, "N": a.N, "K": a.K}
    elif a.op == "softmax_f32":
        kw = {"M": a.M, "N": a.N}
    elif a.op in ("gelu_f32", "sigmoid_f32", "silu_f32"):  # N-vector elementwise activations
        kw = {"N": a.N}
    else:                                   # matmul_f32 / conv2d_im2col_f32: M,N,K
        kw = {"M": a.M, "N": a.N, "K": a.K}
    b = fn(a.out_root, **kw)
    print(f"wrote {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
