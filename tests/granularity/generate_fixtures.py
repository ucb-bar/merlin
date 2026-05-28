"""Generate the granularity test-matrix fixtures from a templating table.

Per the C7 plan:
  TILE       : matmul, conv, elementwise         x small/medium x f32/i8
  LAYER      : matmul, conv, elementwise, fill,
               reduce, softmax, layer_norm        x small/medium x f32/i8
  MEGAKERNEL : matmul_bias_relu, attn_qkv,
               conv_bn_relu, gru_step             x small/medium x f32/i8

For each (granularity, op, shape, dtype) it writes:
  fixtures/<granularity>/<name>/in.mlir
  fixtures/<granularity>/<name>/inputs.txt
  fixtures/<granularity>/<name>/kernels/manifest.json
  fixtures/<granularity>/<name>/kernels/src/<name>.c
  fixtures/<granularity>/<name>/kernels/match/<name>.match.mlir
  fixtures/<granularity>/<name>/skip   (with a reason, until kernel body is filled)

Filled-in fixtures (the demo's reference kernels) drop the skip marker so
the pytest run actually exercises them.

Run:
  uv run python tests/granularity/generate_fixtures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"

SHAPES = {
    "small": {"matmul": (32, 32, 32), "vec": 32, "img": (1, 8, 8, 4)},
    "medium": {"matmul": (128, 128, 128), "vec": 256, "img": (1, 32, 32, 16)},
}

DTYPES = ["f32", "i8"]


def matmul_mlir(M, K, N, dtype):
    if dtype == "f32":
        elt = "f32"
        cst = "0.000000e+00"
    else:
        elt = "i8"
        cst = "0"
    return f"""util.func public @main(
    %lhs: tensor<{M}x{K}x{elt}>,
    %rhs: tensor<{K}x{N}x{elt}>
) -> tensor<{M}x{N}x{elt}> {{
  %cst = arith.constant {cst} : {elt}
  %0 = tensor.empty() : tensor<{M}x{N}x{elt}>
  %init = linalg.fill ins(%cst : {elt}) outs(%0 : tensor<{M}x{N}x{elt}>) -> tensor<{M}x{N}x{elt}>
  %r = linalg.matmul ins(%lhs, %rhs : tensor<{M}x{K}x{elt}>, tensor<{K}x{N}x{elt}>)
                     outs(%init : tensor<{M}x{N}x{elt}>) -> tensor<{M}x{N}x{elt}>
  util.return %r : tensor<{M}x{N}x{elt}>
}}
"""


def elementwise_mlir(N, dtype):
    elt = dtype
    return f"""util.func public @main(
    %lhs: tensor<{N}x{elt}>, %rhs: tensor<{N}x{elt}>
) -> tensor<{N}x{elt}> {{
  %out = tensor.empty() : tensor<{N}x{elt}>
  %0 = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
  }} ins(%lhs, %rhs : tensor<{N}x{elt}>, tensor<{N}x{elt}>)
    outs(%out : tensor<{N}x{elt}>) {{
  ^bb0(%a: {elt}, %b: {elt}, %c: {elt}):
    %p = {"arith.mulf" if elt == "f32" else "arith.muli"} %a, %b : {elt}
    linalg.yield %p : {elt}
  }} -> tensor<{N}x{elt}>
  util.return %0 : tensor<{N}x{elt}>
}}
"""


def matmul_bias_relu_mlir(M, K, N, dtype):
    if dtype != "f32":
        return None  # skip non-f32 megakernel for the demo
    return f"""util.func public @main(
    %lhs: tensor<{M}x{K}xf32>,
    %rhs: tensor<{K}x{N}xf32>,
    %bias: tensor<{N}xf32>
) -> tensor<{M}x{N}xf32> {{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<{M}x{N}xf32>
  %init = linalg.fill ins(%cst : f32) outs(%0 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>)
                      outs(%init : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %ba_init = tensor.empty() : tensor<{M}x{N}xf32>
  %ba = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  }} ins(%mm, %bias : tensor<{M}x{N}xf32>, tensor<{N}xf32>)
    outs(%ba_init : tensor<{M}x{N}xf32>) {{
  ^bb0(%a: f32, %b: f32, %c: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  }} -> tensor<{M}x{N}xf32>
  %relu_init = tensor.empty() : tensor<{M}x{N}xf32>
  %relu = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  }} ins(%ba : tensor<{M}x{N}xf32>) outs(%relu_init : tensor<{M}x{N}xf32>) {{
  ^bb0(%a: f32, %b: f32):
    %z = arith.constant 0.0 : f32
    %r = arith.maximumf %a, %z : f32
    linalg.yield %r : f32
  }} -> tensor<{M}x{N}xf32>
  util.return %relu : tensor<{M}x{N}xf32>
}}
"""


def write_fixture(granularity, name, mlir_text, inputs):
    fdir = ROOT / granularity / name
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "in.mlir").write_text(mlir_text)
    (fdir / "inputs.txt").write_text("\n".join(inputs) + "\n")
    kernels = fdir / "kernels"
    (kernels / "src").mkdir(parents=True, exist_ok=True)
    (kernels / "match").mkdir(parents=True, exist_ok=True)
    src_path = kernels / "src" / f"{name}.c"
    if not src_path.exists():
        src_path.write_text(
            f"// Auto-generated stub for {granularity}/{name}.\n"
            f"// Replace with the substitution kernel body.\n"
            f"int {name}(void){{ return 0; }}\n"
        )
    match_path = kernels / "match" / f"{name}.match.mlir"
    if not match_path.exists():
        match_path.write_text(
            f"// Auto-generated match skeleton for {name}. " "Fill in the cast_compatible_dag_from_root body.\n"
        )
    manifest = {
        "schema_version": 1,
        "kernels": [
            {
                "name": name,
                "source": f"src/{src_path.name}",
                "source_lang": "c",
                "entry_symbol": name,
                "match": {"kind": "linalg_dag", "spec_path": f"match/{match_path.name}"},
                "targets": ["llvm-cpu-x86_64"],
            }
        ],
    }
    (kernels / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    skip = fdir / "skip"
    if not skip.exists():
        skip.write_text("auto-generated; awaiting filled-in kernel + match body.\n")


def main():
    count = 0
    for shape_name, shapes in SHAPES.items():
        for dtype in DTYPES:
            M, K, N = shapes["matmul"]
            # TILE: matmul, elementwise
            mlir = matmul_mlir(M, K, N, dtype)
            inputs = [f"{M}x{K}x{dtype}=1", f"{K}x{N}x{dtype}=1"]
            write_fixture("tile", f"matmul_{shape_name}_{dtype}", mlir, inputs)
            count += 1
            mlir = elementwise_mlir(shapes["vec"], dtype)
            inputs = [f"{shapes['vec']}x{dtype}=2", f"{shapes['vec']}x{dtype}=3"]
            write_fixture("tile", f"elementwise_{shape_name}_{dtype}", mlir, inputs)
            count += 1
            # LAYER: matmul, elementwise (more ops can be added similarly)
            write_fixture(
                "layer",
                f"matmul_{shape_name}_{dtype}",
                matmul_mlir(M, K, N, dtype),
                [f"{M}x{K}x{dtype}=1", f"{K}x{N}x{dtype}=1"],
            )
            count += 1
            write_fixture(
                "layer",
                f"elementwise_{shape_name}_{dtype}",
                elementwise_mlir(shapes["vec"], dtype),
                [f"{shapes['vec']}x{dtype}=2", f"{shapes['vec']}x{dtype}=3"],
            )
            count += 1
            # MEGAKERNEL: matmul_bias_relu (f32 only)
            mbr = matmul_bias_relu_mlir(M, K, N, dtype)
            if mbr:
                write_fixture(
                    "megakernel",
                    f"matmul_bias_relu_{shape_name}_{dtype}",
                    mbr,
                    [f"{M}x{K}x{dtype}=1", f"{K}x{N}x{dtype}=1", f"{N}x{dtype}=0"],
                )
                count += 1
    print(f"generated {count} fixtures under {ROOT}")


if __name__ == "__main__":
    sys.exit(main())
