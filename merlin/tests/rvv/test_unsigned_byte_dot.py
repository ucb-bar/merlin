"""Unsigned byte normalization is bit-exact and exposes a signed-dot ABI."""

import os

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import int8_contractions as ic
from merlin.llvmlower.unsigned_byte_dot import byte_dot_kind, normalize_unsigned_byte_dots


def module(m=2, n=4, k=32, *, lhs="extui", rhs="extsi", init=0, wrong_yield=False):
    returned = "%acc" if wrong_yield else "%sum"
    return parse_mlir_text(f"""
    builtin.module {{
      func.func @forward(%a:tensor<{m}x{k}xi8>,%b:tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
        %zero = arith.constant {init} : i32
        %initial = tensor.splat %zero : tensor<{m}x{n}xi32>
        %out = linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2)->(d0,d2)>,
                          affine_map<(d0,d1,d2)->(d2,d1)>,affine_map<(d0,d1,d2)->(d0,d1)>],
                          iterator_types=["parallel","parallel","reduction"]}}
            ins(%a,%b:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%initial:tensor<{m}x{n}xi32>) {{
          ^bb0(%x:i8,%y:i8,%acc:i32):
            %xe = arith.{lhs} %x : i8 to i32
            %ye = arith.{rhs} %y : i8 to i32
            %product = arith.muli %xe,%ye : i32
            %sum = arith.addi %acc,%product : i32
            linalg.yield {returned} : i32
        }} -> tensor<{m}x{n}xi32>
        func.return %out : tensor<{m}x{n}xi32>
      }}
    }}
    """)


def dot(mod):
    return next(op for op in mod.walk() if op.name == "linalg.generic")


def test_normalization_exposes_only_proved_signed_dot_and_is_idempotent():
    mod = module()
    assert byte_dot_kind(dot(mod)) == "u8s8"
    assert normalize_unsigned_byte_dots(mod) == 1
    mod.verify()
    assert normalize_unsigned_byte_dots(mod) == 0
    kinds = [byte_dot_kind(op) for op in mod.walk() if op.name == "linalg.generic"]
    assert kinds.count("s8s8") == 1 and "u8s8" not in kinds
    routed = ic.rewrite_contractions(mod, select=lambda shape: True, symbol_prefix="signed_byte")
    assert routed.count == 1 and routed.signatures == {"signed_byte_0": (2, 4, 32)}
    mod.verify()


@pytest.mark.parametrize("options", [dict(lhs="extsi"), dict(rhs="extui"), dict(init=7), dict(wrong_yield=True)])
def test_other_arithmetic_never_changes(options):
    mod = module(**options)
    before = str(mod)
    assert normalize_unsigned_byte_dots(mod) == 0 and str(mod) == before


def test_unproved_scalar_flags_are_refused():
    mod = module()
    next(op for op in mod.walk() if op.name == "arith.muli").attributes["unexpected.semantic"] = StringAttr("x")
    before = str(mod)
    assert normalize_unsigned_byte_dots(mod) == 0 and str(mod) == before


def execute(mod, a, b, work):
    from merlin.llvmlower.kernel_backend import compile_host

    model = compile_host(mod, work)
    output = np.empty((a.shape[0], b.shape[1]), dtype=np.int32)
    model([(a.ctypes.data, a.shape), (b.ctypes.data, b.shape), (output.ctypes.data, output.shape)])
    return output


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (2, 2, 2),
        (4, 4, 8),
        (8, 8, 16),
        (16, 16, 32),
        (32, 32, 64),
        (3, 5, 7),
        (256, 256, 1),
        (1, 4, 1 << 18),
    ],
)
def test_actual_compiled_full_values_and_int32_wrap(tmp_path, monkeypatch, shape):
    from merlin.llvmlower import toolchain

    if not toolchain.available():
        pytest.skip("MLIR/LLVM host compilation unavailable")
    # Compile the runtime with the selected LLVM compiler too. A system GCC
    # predating its __bf16 ABI cannot compile the existing shared runtime.
    compiler_bin = tmp_path / "compiler-bin"
    compiler_bin.mkdir()
    (compiler_bin / "cc").symlink_to(toolchain.clang())
    monkeypatch.setenv("PATH", str(compiler_bin) + os.pathsep + os.environ.get("PATH", ""))
    m, n, k = shape
    a = np.resize(np.arange(256, dtype=np.uint8), (m, k)).copy()
    b = np.resize(np.asarray([-128, -127, -1, 0, 1, 126, 127], dtype=np.int8), (k, n)).copy()
    if (m, n, k) == (256, 256, 1):
        b = np.arange(-128, 128, dtype=np.int16).astype(np.int8).reshape(1, 256)
    if k > 100000:
        a.fill(255)
        b.fill(127)  # mathematical sum exceeds both signed and unsigned i32.
    expected = (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
    original = module(m, n, k)
    before = execute(original, a, b, tmp_path / "source")
    assert normalize_unsigned_byte_dots(original) == 1
    after = execute(original, a, b, tmp_path / "normalized")
    np.testing.assert_array_equal(before, expected)
    np.testing.assert_array_equal(after, expected)


@pytest.mark.parametrize("unsigned", [False, True])
def test_source_admission_checks_actual_operand_signedness(tmp_path, monkeypatch, unsigned):
    from merlin.compile import source_quantization as sq
    from merlin.targetgen import application_inventory

    mod = module(lhs="extui" if unsigned else "extsi")
    mod.attributes["prov.quantization"] = StringAttr(sq.SOURCE_W8A8_SCHEME)
    op = dot(mod)
    op.attributes.update(
        {
            "prov.aten": StringAttr("aten._int_mm.default"),
            "prov.op": StringAttr("int_matmul"),
            "prov.int_mm_lhs_dtype": StringAttr("uint8" if unsigned else "int8"),
            "prov.int_mm_rhs_dtype": StringAttr("int8"),
        }
    )
    (tmp_path / "model.mlir").write_text(str(mod))
    monkeypatch.setattr(
        application_inventory, "verify_capture_receipt", lambda source: {"status": "verified_materialized"}
    )
    facts = sq.inspect_source_w8a8(tmp_path)
    assert facts["integer_matmuls"] == 1
    assert facts["exact_unsigned_normalization_required"] is unsigned
    assert facts["source_matrix_operand_formats"] == {"u8s8" if unsigned else "s8s8": 1}
    op.attributes["prov.int_mm_lhs_dtype"] = StringAttr("int8" if unsigned else "uint8")
    (tmp_path / "model.mlir").write_text(str(mod))
    with pytest.raises(ValueError, match="signedness is unknown or contradicts"):
        sq.inspect_source_w8a8(tmp_path)


def test_legacy_wrong_capture_is_not_silently_repaired(tmp_path, monkeypatch):
    from merlin.compile import source_quantization as sq
    from merlin.targetgen import application_inventory

    mod = module(lhs="extsi")
    mod.attributes["prov.quantization"] = StringAttr(sq.SOURCE_W8A8_SCHEME)
    op = dot(mod)
    op.attributes.update(
        {
            "prov.aten": StringAttr("aten._int_mm.default"),
            "prov.op": StringAttr("int_matmul"),
            "prov.int_mm_lhs_dtype": StringAttr("uint8"),
            "prov.int_mm_rhs_dtype": StringAttr("int8"),
        }
    )
    before = str(mod)
    assert normalize_unsigned_byte_dots(mod) == 0 and str(mod) == before
    (tmp_path / "model.mlir").write_text(before)
    monkeypatch.setattr(
        application_inventory, "verify_capture_receipt", lambda source: {"status": "verified_materialized"}
    )
    with pytest.raises(ValueError, match="old source bytes are not repaired"):
        sq.inspect_source_w8a8(tmp_path)


def test_missing_operand_witness_refuses_source_admission(tmp_path, monkeypatch):
    from merlin.compile import source_quantization as sq
    from merlin.targetgen import application_inventory

    mod = module()
    mod.attributes["prov.quantization"] = StringAttr(sq.SOURCE_W8A8_SCHEME)
    dot(mod).attributes["prov.aten"] = StringAttr("aten._int_mm.default")
    (tmp_path / "model.mlir").write_text(str(mod))
    monkeypatch.setattr(
        application_inventory, "verify_capture_receipt", lambda source: {"status": "verified_materialized"}
    )
    with pytest.raises(ValueError, match=r"dtype evidence \[\]"):
        sq.inspect_source_w8a8(tmp_path)
