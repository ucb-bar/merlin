"""Scalar reduction loops do not relax the surrounding owned-buffer proof.

Every fixture passes the native MLIR verifier through the ordinary lifetime
checker. These are post-deallocation proof regressions, not model execution or
permission for memory ownership carried by an inner loop.
"""

import json
import re

import pytest
from test_owned_buffer_loop_check import COPY, GUARD, loop_source

from merlin.llvmlower import toolchain
from merlin.llvmlower.pipeline import dealloc_placement_violations

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")


def scalar_loop(*, depth=1, bound="dynamic", dtype="i32", mutation=None):
    limit = {"dynamic": "%bound", "zero": "%c0", "static": "%c3"}[bound]
    lines = []

    def nest(level, initial):
        name, acc, iv = f"%sum{level}", f"%acc{level}", f"%j{level}"
        lines.append(f"{name}=scf.for {iv}=%c0 to {limit} step %c1 iter_args({acc}={initial})->{dtype} {{")
        if level + 1 < depth:
            value = nest(level + 1, acc)
        else:
            if mutation == "memref_carry":
                lines.append(
                    "%alias=scf.for %k=%c0 to %bound step %c1 "
                    "iter_args(%buffer=%out)->memref<4xi8> {scf.yield %buffer:memref<4xi8>}"
                )
            elif mutation == "nested_dealloc":
                lines.append("memref.dealloc %fresh:memref<4xi8>")
            elif mutation == "unknown_call":
                lines.append("func.call @opaque():()->()")
            elif mutation == "extract_address":
                lines.append("%address=memref.extract_aligned_pointer_as_index %fresh:memref<4xi8> -> index")
            elif mutation == "scalar_if_unknown":
                lines.append("scf.if %true {func.call @opaque():()->()}")
            lines.append("%value=memref.load %previous[%c0]:memref<4xi8>")
            operation = "arith.extui" if dtype == "i32" else "arith.uitofp"
            lines.append(f"%wide={operation} %value:i8 to {dtype}")
            lines.append(f"%added=arith.{'addi' if dtype == 'i32' else 'addf'} {acc},%wide:{dtype}")
            value = "%added"
        lines.append(f"scf.yield {value}:{dtype}")
        lines.append('} {test.unknown_effect="opaque"}' if mutation == "unknown_attributes" and level == 0 else "}")
        return name

    value = nest(0, "%scalar_zero")
    if dtype == "i32":
        lines.append(f"%byte=arith.trunci {value}:i32 to i8")
    else:
        lines.extend((f"%positive=arith.cmpf ogt,{value},%scalar_zero:f32", "%byte=arith.extui %positive:i1 to i8"))
    lines.append("memref.store %byte,%fresh[%c0]:memref<4xi8>")
    return "\n".join(lines)


def source(*, depth=1, bound="dynamic", dtype="i32", mutation=None):
    module = loop_source()
    zero = "0" if dtype == "i32" else "0.0"
    module = module.replace(
        "%initial=memref.alloc()", f"%scalar_zero=arith.constant {zero}:{dtype}\n%initial=memref.alloc()"
    )
    body = scalar_loop(depth=depth, bound=bound, dtype=dtype, mutation=mutation)
    anchor = GUARD if mutation == "old_after_release" else COPY
    module = module.replace(anchor, anchor + "\n" + body)
    if mutation in ("unknown_call", "scalar_if_unknown"):
        module = module.replace("module {", "module {func.func private @opaque()")
    return module


def check(tmp_path, text):
    # The production helper parses and verifies the complete module natively
    # before checking placement. A malformed mutation raises instead of being
    # mistaken for an expected ownership refusal.
    (tmp_path / "module.mlir").write_text(text)
    violations = dealloc_placement_violations(text, timeout=60)
    (tmp_path / "violations.json").write_text(json.dumps(violations, indent=2))
    return violations


@pytest.mark.parametrize("dtype", ("i32", "f32"))
@pytest.mark.parametrize("bound,depth", (("dynamic", 1), ("zero", 1), ("static", 2), ("dynamic", 3), ("zero", 3)))
def test_nested_scalar_reductions_preserve_outer_allocation_generations(tmp_path, dtype, bound, depth):
    assert check(tmp_path, source(dtype=dtype, bound=bound, depth=depth)) == []


@pytest.mark.parametrize(
    "mutation,diagnostic",
    (
        ("memref_carry", "loop|memref|nonnumeric"),
        ("old_after_release", "after.*release"),
        ("nested_dealloc", "releasing|effect|release"),
        ("unknown_call", "effect|call"),
        ("extract_address", "escaping|effect|escape"),
        ("scalar_if_unknown", "effect|branch|control|call"),
        ("unknown_attributes", "attributes|effect"),
    ),
)
def test_nested_scalar_loop_cannot_hide_ownership_or_unknown_effects(tmp_path, mutation, diagnostic):
    violations = check(tmp_path, source(depth=2, mutation=mutation))
    assert violations
    assert any(re.search(diagnostic, violation) for violation in violations), violations
