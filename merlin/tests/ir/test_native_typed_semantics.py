"""Full-domain admission rejects incomplete guards and runtime products."""

import hashlib
import json
import math
from pathlib import Path

import pytest
from mlir_fixtures import assert_unchanged, forward, generic, pointwise, returned
from targetgen_fixtures import assert_host_refusal
from xdsl.dialects.math import FmaOp
from xdsl.rewriter import Rewriter

import merlin.targetgen.native_ieee_certificate as checker
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.native_cpu_layout import input_layouts, profile
from merlin.llvmlower.significand_products import rewrite_significand_products
from merlin.llvmlower.uniform_numeric_literals import compile_time_operand, fold_uniform_numeric_literals
from merlin.targetgen.native_typed_semantics import prove
from merlin.xdsl_dialects.lowering.group_numerics import _constant_of


def guarded(*, missing=False, escape=False):
    body = """%q=arith.addf %a,%b:f32
          %n=arith.constant -4194304:i32
          %nan=arith.bitcast %n:i32 to f32
          %mask=arith.constant 4194304:i32
          %an=arith.cmpf uno,%a,%a:f32
          %bn=arith.cmpf uno,%b,%b:f32
          %qn=arith.cmpf uno,%q,%q:f32
          %ab=arith.bitcast %a:f32 to i32
          %bb=arith.bitcast %b:f32 to i32
          %aq=arith.ori %ab,%mask:i32
          %bq=arith.ori %bb,%mask:i32
          %av=arith.bitcast %aq:i32 to f32
          %bv=arith.bitcast %bq:i32 to f32
          %invalid=arith.select %qn,%nan,%q:f32
          %first=arith.select %an,%av,%invalid:f32
          %last=arith.select %bn,%bv,%first:f32"""
    result = "%q" if escape else "%first" if missing else "%last"
    return pointwise(["tensor<4xf32>"] * 2, "tensor<4xf32>", body, yields=result)


@pytest.fixture
def certificate(monkeypatch):
    monkeypatch.setattr(
        "merlin.targetgen.native_ieee_certificate.certificate",
        lambda: {
            "arithmetic_checks": [],
            "content_sha256": "test-structural-proof",
            "conversions": [],
            "loop_profile": {},
        },
    )


def test_complete_guard_composition_is_proven(certificate):
    assert prove(guarded())["status"] == "proven"


@pytest.mark.parametrize("options", [dict(missing=True), dict(escape=True)])
def test_partial_nan_guard_or_unprotected_escape_refuses(certificate, options):
    result = prove(guarded(**options))
    assert result["code"] == "native_competing_nan_priority_unproved"


def test_layout_metadata_does_not_read_payloads():
    class NoPayload(dict):
        def values(self):
            raise AssertionError("tensor payload visited")

    tensor = NoPayload(kind="tensor", shape=[33], stride=[1], values="unread", expected="unread")
    args = [object()]
    report = {"cases": [{"case": {"arguments": [tensor]}, "input_indices": [0]}]}
    assert input_layouts(report, args) == {args[0]: {"shape": [33], "stride": [1]}}


def test_vector_tail_and_unknown_layout_domain():
    assert profile({"shape": [33], "stride": [1]}, block=32, grain=32768)["vector_prefix"] == 32
    assert profile({"shape": [33], "stride": [2]}, block=32, grain=32768)["vector_prefix"] == 0
    padded = profile({"shape": [2, 3], "stride": [10, 2]}, block=32, grain=32768)
    assert padded["row"] == 3 and padded["vector_prefix"] == 0
    assert profile({"shape": [2, 3], "stride": [1, 1]}, block=32, grain=32768) is None
    assert profile({"shape": [32769], "stride": [1]}, block=32, grain=32768) is None


def test_runtime_tensor_product_stays_unlowered():
    module = guarded()
    arithmetic = next(op for op in module.walk() if op.name == "arith.addf")
    assert not any(compile_time_operand(v) for v in arithmetic.operands)
    # A closed, unguarded product is numerically lowerable but ineligible.
    module = pointwise(["tensor<4xf16>"] * 2, "tensor<4xf16>", "%v=arith.mulf %a,%b:f16")
    assert_unchanged(module, rewrite_significand_products, ({}, []))


@pytest.mark.parametrize(
    "operation,left,right,expected",
    [
        ("arith.mulf", "0.5", "2.0", 1.0),
        ("arith.addf", "1.0", "0.00048828125", 1.0),
        ("arith.subf", "-0.0", "0.0", -0.0),
    ],
)
def test_source_literal_folding_preserves_typed_rne_and_zero_sign(operation, left, right, expected):
    prelude = f"""%ca=arith.constant {left}:f16
      %cb=arith.constant {right}:f16
      %x=tensor.splat %ca:tensor<4xf16>
      %y=tensor.splat %cb:tensor<4xf16>
"""
    ins = [("%x", "tensor<4xf16>"), ("%y", "tensor<4xf16>")]
    module = forward("", "tensor<4xf16>", prelude + generic(ins, "tensor<4xf16>", f"%v={operation} %a,%b:f16"))
    assert fold_uniform_numeric_literals(module) == 1
    actual = _constant_of(returned(module))
    assert actual == expected
    assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


def resealed_certificate(tmp_path, monkeypatch, edit):
    """Install a copy of the native IEEE certificate with ``edit`` applied and its digest resealed."""
    document = json.loads(Path(checker.__file__).with_suffix(".json").read_text())
    edit(document)
    document.pop("content_sha256")
    document["content_sha256"] = checker.digest(document)
    fixture = tmp_path / "certificate.json"
    fixture.write_text(json.dumps(document))
    monkeypatch.setattr(checker, "__file__", str(fixture.with_suffix(".py")))
    monkeypatch.setattr(checker, "proof_sources", lambda: document["sources"])
    return checker


def test_corrupted_double_rounding_certificate_refuses(tmp_path, monkeypatch):
    def corrupt(document):
        document["double_rounding"]["f16"]["intermediate_precision"] = 23

    checker = resealed_certificate(tmp_path, monkeypatch, corrupt)
    with pytest.raises(ValueError, match="double rounding precision"):
        checker.certificate()


def test_scalar_literal_nan_guard_folds_without_runtime_observation():
    module = parse_mlir_text("""module {func.func @forward()->tensor<4xbf16>{
      %a=arith.constant 0.5:f32
      %b=arith.truncf %a:f32 to bf16
      %n=arith.constant 32704:i16
      %nan=arith.bitcast %n:i16 to bf16
      %c=arith.cmpf uno,%a,%a:f32
      %s=arith.select %c,%nan,%b:bf16
      %r=tensor.splat %s:tensor<4xbf16>
      return %r:tensor<4xbf16>
    }}""")
    assert fold_uniform_numeric_literals(module) == 3
    assert _constant_of(returned(module)) == 0.5


@pytest.mark.parametrize("divisor,accepted", [("2.0", True), ("3.0", False), ("0x00000001", False)])
def test_reciprocal_requires_an_exact_finite_compute_value(certificate, divisor, accepted):
    prelude = f"%d=arith.constant {divisor}:f32\n%y=tensor.splat %d:tensor<4xf32>\n"
    ins = [("%x", "tensor<4xf32>"), ("%y", "tensor<4xf32>")]
    quotient = generic(ins, "tensor<4xf32>", "%v=arith.divf %a,%b:f32")
    module = forward("%x:tensor<4xf32>", "tensor<4xf32>", prelude + quotient)
    result = prove(module)
    assert (result["status"] == "proven") == accepted
    if not accepted:
        assert result["code"] == "native_reciprocal_opmath_unproved"


def test_fused_runtime_product_cannot_bypass_native_unit_restriction(certificate):
    module = guarded()
    added = next(op for op in module.walk() if op.name == "arith.addf")
    fused = FmaOp(added.operands[0], added.operands[1], added.operands[0])
    Rewriter.replace_op(added, fused)
    assert prove(module)["code"] == "native_parallel_runtime_product_construction"


def test_uncertified_binary64_narrowing_cannot_fold_away_its_obligation(certificate):
    # Above a half midpoint, but binary32 first rounds it onto that midpoint.
    module = parse_mlir_text("""module {func.func @forward()->tensor<1xf16>{
      %a=arith.constant 1.0004882812509095:f64
      %b=arith.truncf %a:f64 to f16
      %r=tensor.splat %b:tensor<1xf16>
      return %r:tensor<1xf16>
    }}""")
    assert fold_uniform_numeric_literals(module) == 0
    assert prove(module)["code"] == "native_conversion_form_unproved"


def test_conversion_prelude_cannot_place_a_numerical_comparison(certificate):
    body = """%af=arith.extf %a:f16 to f32
      %bf=arith.extf %b:f16 to f32
      %c=arith.cmpf olt,%af,%bf:f32
      %v=arith.select %c,%af,%bf:f32"""
    module = pointwise(["tensor<4xf16>"] * 2, "tensor<4xf32>", body)
    assert prove(module)["code"] == "native_float_predicate_family_unproved"


def test_dropped_destination_layout_cannot_claim_a_copy_profile(tmp_path, monkeypatch):
    module = pointwise(["tensor<4xf16>"], "tensor<4xf32>", "%v=arith.extf %a:f16 to f32")
    monkeypatch.setattr(
        "merlin.targetgen.native_dtype_domain.component_semantics",
        lambda _: {
            "status": "proven",
            "method": "closed_typed_pointwise_IEEE_composition_v1",
            "cpu_loop_profile": {"reduced_vector_elements": 4, "float_copy_vector_elements": 2, "parallel_grain": 1024},
        },
    )
    layouts = [{"kind": "tensor", "shape": [4], "stride": [1]}, {"kind": "tensor", "shape": [4], "stride": [2]}]
    cases = [{"status": "bundled", "input_indices": [0, 1], "case": {"arguments": layouts}}]
    assert_host_refusal(tmp_path, module, [], cases, "native_copy_layout_context_unproved")


def test_library_is_hashed_once_per_process_until_it_changes(tmp_path, monkeypatch):
    library = tmp_path / "libtorch_cpu.so"
    library.write_bytes(b"certified library")

    def bind(document):
        document["host"].update(library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())

    checker = resealed_certificate(tmp_path, monkeypatch, bind)
    hashed = []
    file_digest = hashlib.file_digest

    def counted(stream, name):
        hashed.append(stream.name)
        return file_digest(stream, name)

    monkeypatch.setattr(hashlib, "file_digest", counted)
    checker.certificate()
    checker.certificate()
    assert hashed == [str(library)]
    library.write_bytes(b"replaced library bytes")
    with pytest.raises(ValueError, match="reference CPU library changed"):
        checker.certificate()
