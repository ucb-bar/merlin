"""Ordinary roundeven representation retains complete source-observer closure."""

import ctypes
import math
import runpy
import struct
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.ordered_fma_groups import _supported
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scaled_integer_finite import prove_broadcast_scaled_integer_finite
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    close_scalar_i8_observer,
    emit_source_interval_i8_lookup,
)
from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_i8_lookup
from merlin.llvmlower.toolchain import clang

RECIPE = runpy.run_path(
    str(
        repo_root()
        / "packages/merlin-experiments/src/merlin_experiments/phase1/author_skills/heavy-pass-recipes/scripts/source_observer_recipe.py"
    )
)
EFFECTS = IntervalEffectContract(True, True, True, True, True)


def proof(text=None):
    module = parse_mlir_text(text or RECIPE["source"]())
    function = module.body.block.first_op
    endpoint = next(op.results[0] for op in function.body.block.ops if op.name == "arith.mulf")
    return close_scalar_i8_observer(function.body.block.args[0], endpoint, effects=EFFECTS)


@pytest.mark.parametrize("change", ["round", "bound", "attribute", "escape"])
def test_typed_intrinsic_changed_semantics_and_live_escapes_refuse(change):
    text = RECIPE["source"]()
    if change == "round":
        text = text.replace("llvm.intr.roundeven", "llvm.intr.round")
    elif change == "bound":
        text = text.replace("-128.0", "-129.0")
    elif change == "attribute":
        text = text.replace("(%b):(f32)->f32", '(%b){unproved="yes"}:(f32)->f32')
    else:
        text = text.replace(" return %result", " %escape=arith.addf %act,%x:f32\n return %result")
    with pytest.raises(ValueError):
        proof(text)


def test_finite_preparation_accepts_only_the_exact_intrinsic_with_original_source_purity():
    text = RECIPE["producer_source"](3, 5).replace(
        " linalg.yield %scaled", ' %rounded="llvm.intr.roundeven"(%scaled):(f32)->f32\n linalg.yield %scaled'
    )
    module = parse_mlir_text(text)
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    value = [op.results[0] for op in generic.body.block.ops if op.name == "arith.mulf"][-1]
    assert _supported(generic)  # Registered intrinsic retains the original source purity gate.
    finite = prove_broadcast_scaled_integer_finite(value, effects=EFFECTS)
    assert finite.repetition == 3 and finite.domain_extents == (3, 5)
    bad = parse_mlir_text(text.replace("llvm.intr.roundeven", "llvm.intr.round"))
    generic = next(op for op in bad.walk() if op.name == "linalg.generic")
    value = [op.results[0] for op in generic.body.block.ops if op.name == "arith.mulf"][-1]
    with pytest.raises(ValueError):
        prove_broadcast_scaled_integer_finite(value, effects=EFFECTS)


def test_actual_normal_lowering_and_native_original_observer(tmp_path):
    witness = proof()
    table = build_source_interval_table(witness.expression, effects=EFFECTS, leading_bits=9, max_table_bytes=4096)
    llvm = lower_to_llvm_ir(RECIPE["source"](), workdir=tmp_path / "normal")
    selected, report = rewrite_source_interval_i8_lookup(
        llvm, proofs=(witness,), table=table, lookup_symbol="lookup", effects=EFFECTS
    )
    assert len(report["routes"]) == 1
    assert report["routes"][0]["source_observer_proof"]["source_form"] == "bounded_roundeven_conversion"
    for before, after in (("llvm.roundeven", "llvm.round"), ("-1.280000e+02", "-1.290000e+02")):
        altered = llvm.replace(before, after)
        if altered != llvm:
            assert not rewrite_source_interval_i8_lookup(
                altered, proofs=(witness,), table=table, lookup_symbol="lookup", effects=EFFECTS
            )[1]["routes"]
    (tmp_path / "selected.ll").write_text(selected)
    (tmp_path / "original.ll").write_text(llvm.replace("@observe(", "@original("))
    constants = []
    for lower, upper in struct.iter_unpack("<ff", table.data):
        constants.append(
            "{"
            + ",".join(
                float(v).hex() + "f" if math.isfinite(v) else "__builtin_inff()" if v > 0 else "-__builtin_inff()"
                for v in (lower, upper)
            )
            + "}"
        )
    callback = emit_source_interval_i8_lookup(
        table_name="table", activation_name="activation", quantizer_name="quant", lookup_name="lookup", leading_bits=9
    )
    (tmp_path / "helper.c").write_text(
        callback
        + """
#include <fenv.h>
float activation(float x){return x*2.0f;}
signed char quant(float x){return (signed char)__builtin_roundevenf(__builtin_fminf(__builtin_fmaxf(x,-128.0f),127.0f));}
extern signed char observe(float,float),original(float,float);
signed char guarded(float x,float up){return fegetround()==FE_TONEAREST?observe(x,up):original(x,up);}
"""
        + "const float table[512][2]={"
        + ",".join(constants)
        + "};\n"
    )
    library = tmp_path / "current.so"
    subprocess.run(
        [
            clang(),
            "-O2",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            str(tmp_path / "selected.ll"),
            str(tmp_path / "original.ll"),
            str(tmp_path / "helper.c"),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    loaded = ctypes.CDLL(str(library))
    for function in (loaded.original, loaded.guarded):
        function.argtypes, function.restype = (ctypes.c_float, ctypes.c_float), ctypes.c_int8
    environment = ctypes.CDLL(None)
    saved = environment.fegetround()
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert environment.fesetround(mode) == 0
            for index in range(-259, 260):
                x, up = index / 12, (-0.5 if index & 1 else 1.0)
                assert loaded.original(x, up) == loaded.guarded(x, up)
    finally:
        environment.fesetround(saved)
