"""Source-checked integer observers and finite broadcast-scale preparation.

    python source_observer_recipe.py [--library DIR] [--output DIR]

This neutral example assumes test-only RNE/effect permissions. Actual callers
must supply their source, ownership, runtime guard and ordinary build checks.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


def source():
    return """module {func.func @observe(%x:f32,%up:f32)->i8 {
 %two=arith.constant 2.0:f32
 %act=arith.mulf %x,%two:f32
 %prod=arith.mulf %act,%up:f32
 %q=arith.constant 3.0:f32
 %scaled=arith.mulf %prod,%q:f32
 %lo=arith.constant -128.0:f32
 %a=arith.maximumf %scaled,%lo:f32
 %hi=arith.constant 127.0:f32
 %b=arith.minimumf %a,%hi:f32
 %rounded="llvm.intr.roundeven"(%b):(f32)->f32
 %result=arith.fptosi %rounded:f32 to i8
 return %result:i8
} }
"""


def llvm_source():
    return """define i8 @observe(float %x,float %up) {
 %act = fmul float %x, 2.000000e+00
 %prod = fmul float %act, %up
 %scaled = fmul float %prod, 3.000000e+00
 %lo = call float @llvm.maximum.f32(float %scaled,float -128.0)
 %hi = call float @llvm.minimum.f32(float %lo,float 127.0)
 %rounded = call float @llvm.roundeven.f32(float %hi)
 %result = fptosi float %rounded to i8
 ret i8 %result
}
declare float @llvm.maximum.f32(float,float)
declare float @llvm.minimum.f32(float,float)
declare float @llvm.roundeven.f32(float)
"""


def producer_source(rows, columns):
    return f"""module {{func.func @producer(%input:tensor<{rows}x{columns}xi16>,
 %scales:tensor<{columns}xf32>)->tensor<{rows}x{columns}xf32> {{
 %e=tensor.empty():tensor<{rows}x{columns}xf32>
 %r=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%input,%scales:tensor<{rows}x{columns}xi16>,tensor<{columns}xf32>)
 outs(%e:tensor<{rows}x{columns}xf32>) {{^bb0(%v:i16,%scale:f32,%old:f32):
 %f=arith.sitofp %v:i16 to f32
 %factor=arith.constant 0.25:f32
 %constant=arith.mulf %f,%factor:f32
 %scaled=arith.mulf %constant,%scale:f32
 linalg.yield %scaled:f32
 }}->tensor<{rows}x{columns}xf32>
 return %r:tensor<{rows}x{columns}xf32>
}} }}"""


def recipe(rows=3, columns=5, output=None):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.bounded_rne_word_cells import (
        binary32_word_in_rne_cell,
        prepare_bounded_rne_word_cells,
    )
    from merlin.llvmlower.late_quant_rne import rewrite
    from merlin.llvmlower.scaled_integer_finite import (
        emit_finite_scale_scan,
        prove_broadcast_scaled_integer_finite,
    )
    from merlin.llvmlower.source_expression_interval import (
        IntervalEffectContract,
        build_source_interval_table,
        close_scalar_i8_observer,
        emit_source_interval_i8_lookup,
    )
    from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_i8_lookup

    if min(rows, columns) <= 0 or rows <= 1:
        raise ValueError("nonempty repeated broadcast domain required")
    effects = IntervalEffectContract(True, True, True, True, True)  # Neutral fixture only.
    module = parse_mlir_text(source())
    function = module.body.block.first_op
    endpoint = next(op.results[0] for op in function.body.block.ops if op.name == "arith.mulf")
    proof = close_scalar_i8_observer(function.body.block.args[0], endpoint, effects=effects)
    table = build_source_interval_table(proof.expression, effects=effects, leading_bits=9, max_table_bytes=4096)
    cells = prepare_bounded_rne_word_cells(proof, effects=effects)
    original = llvm_source()
    selected, mapping = rewrite_source_interval_i8_lookup(
        original, proofs=(proof,), table=table, lookup_symbol="observe_lookup", effects=effects
    )
    assert len(mapping["routes"]) == 1 and rewrite(original)[0] == original
    _, portable = rewrite(original, host_isa="portable")
    assert len(portable["routes"]) == 1
    producer = parse_mlir_text(producer_source(rows, columns))
    generic = next(op for op in producer.walk() if op.name == "linalg.generic")
    value = list(generic.body.block.ops)[-2].results[0]
    finite = prove_broadcast_scaled_integer_finite(value, effects=effects)
    boundary_checks = 0
    for integer in range(-127, 127):
        for value in (integer - 0.5, integer, integer + 0.5):
            word = struct.unpack("<I", struct.pack("<f", value))[0]
            assert binary32_word_in_rne_cell(word, round(value), cells)
            boundary_checks += 1
    callback = emit_source_interval_i8_lookup(
        table_name="observer_table",
        activation_name="original_activation",
        quantizer_name="original_quantizer",
        lookup_name="observe_lookup",
        leading_bits=9,
        finite_table=table,
        zero_observer_cells=(cells,),
    )
    if output:
        directory = Path(output)
        directory.mkdir(exist_ok=False)
        for name, content in {
            "source.mlir": source(),
            "original.ll": original,
            "selected.ll": selected,
            "lookup.c": callback,
            "scale_scan.c": emit_finite_scale_scan(finite.domain, symbol="scales_finite"),
        }.items():
            (directory / name).write_text(content)
        (directory / "observer_table.bin").write_bytes(table.data)
    return dict(
        schema="neutral.source_integer_observer_recipe.v1",
        source_expression=proof.expression.canonical_sha256,
        source_checked_routes=len(mapping["routes"]),
        table_bytes=len(table.data),
        exact_cell_boundary_checks=boundary_checks,
        finite_scale_values=columns,
        reused_producer_values=rows * columns,
        repetition=finite.repetition,
        scale_limit_word=finite.domain.scale_limit_word(),
        portable_round_routes=len(portable["routes"]),
        timing="UNKNOWN",
        native_numerical_execution="not_established",
        qualification="test-only source/effect contract; runtime incoming-RNE, original fallback, ownership and final build required",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    library = Path(args.library or "submission/compiler_library")
    if library.is_dir():
        sys.path.insert(0, str(library.resolve()))
    print(json.dumps(recipe(args.rows, args.columns, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
