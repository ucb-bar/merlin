"""Uniform quantized zero needs a complete integral image and an exact divisor, read from the IR."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.quantized_literal_reference import qualify
from merlin.targetgen.source_integer_remainder import specialize

SPEC = Path(__file__).resolve().parents[3] / "examples/gemmini/target/software-spec.yaml"

# Python-style remainder adjustment: r + d where r != 0 and sign(r) != sign(d).
ADJUST = """
        %zero=arith.constant 0.0 : {floating}
        %nonzero=arith.cmpf une,%r,%zero : {floating}
        %rneg=arith.cmpf olt,%r,%zero : {floating}
        %dneg=arith.cmpf olt,%d,%zero : {floating}
        %differ=arith.xori %rneg,%dneg : i1
        %fix=arith.andi %nonzero,%differ : i1
        %sum=arith.addf %r,%d : {floating}
        %v=arith.select %fix,%sum,%r : {floating}
        linalg.yield %v : {floating}"""


def fixture(tmp_path, divisor, *, integer="i64", floating="f32", foreign=False, adjust=False):
    conversion = "arith.uitofp" if integer == "i1" else "arith.sitofp"
    if integer.startswith("f"):
        convert = f"%n=arith.extf %a : {integer} to {floating}"
    else:
        convert = f"%n={conversion} %a : {integer} to {floating}"
    tail = ADJUST.format(floating=floating) if adjust else f"\n        linalg.yield %r : {floating}"
    tmp_path.joinpath("model.mlir").write_text(f"""module {{func.func @forward(%x:tensor<4x{integer}>)
      -> (tensor<4x{floating}>,tensor<4x{integer}>) {{
      {"\"fixture.effect\"() : () -> ()" if foreign else ""}
      %k=arith.constant {divisor!r} : {floating}
      %ks=tensor.splat %k : tensor<4x{floating}>
      %e=tensor.empty() : tensor<4x{floating}>
      %o=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x,%ks:tensor<4x{integer}>,tensor<4x{floating}>) outs(%e:tensor<4x{floating}>) {{
        ^bb0(%a:{integer},%d:{floating},%u:{floating}):
        {convert}
        %r=arith.remf %n,%d : {floating}{tail}
      }} -> tensor<4x{floating}>
      return %o,%x : tensor<4x{floating}>,tensor<4x{integer}>
    }} }}""".replace("inf : ", "0x7F800000 : "))
    # The case record names a different operator and divisor: it must not be read.
    return dict(
        cases=[
            dict(
                status="bundled",
                case=dict(
                    overload="aten.mul.Tensor",
                    arguments=dict(args=dict(items=[dict(kind="tensor", values="NEVER READ"), 3.0])),
                ),
                output_indices=[0],
                semantic_boundary=dict(post_indices=[1]),
            )
        ]
    )


@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("integer", ["i1", "i8", "i32", "i64"])
@pytest.mark.parametrize("divisor", [1.0, -1.0, 0.5, -0.25])
def test_whole_integer_image_has_a_canonical_zero_after_quantization(tmp_path, integer, divisor, adjust):
    report = fixture(tmp_path, divisor, integer=integer, adjust=adjust)
    proof, original = specialize(tmp_path, report, SPEC)
    assert proof["input_signed_storage_bits"] == int(integer[1:])
    assert proof["divisors"] == [divisor]
    assert proof["quantized_contract"]["schema"] == "merlin.quantized_native.v1"
    module = parse_mlir_file(tmp_path / "model.mlir")
    module.verify()
    assert next(op for op in module.walk() if op.name == "arith.constant").value.value.data == 0
    assert original != tmp_path.joinpath("model.mlir").read_bytes()


@pytest.mark.parametrize(
    "divisor,options",
    [(d, {}) for d in (0.0, 0.3, 0.3333333333333333, 2.0, float("inf"))]
    + [(0.5, dict(integer="f32", floating="f64")), (0.5, dict(floating="f16")), (0.5, dict(foreign=True))],
)
def test_special_divisor_unproved_domain_or_effect_refuses(tmp_path, divisor, options):
    report = fixture(tmp_path, divisor, **options)
    before = tmp_path.joinpath("model.mlir").read_bytes()
    assert specialize(tmp_path, report, SPEC) is None
    assert tmp_path.joinpath("model.mlir").read_bytes() == before


def test_oracle_qualification_failure_does_not_turn_into_a_payload_dependent_host_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "merlin.targetgen.core_aten_numeric._source_reference",
        lambda *a: SimpleNamespace(returncode=1, stderr="oracle unavailable"),
    )
    with pytest.raises(ArithmeticError, match="oracle unavailable"):
        qualify(tmp_path, dict(cases=[], output_count=1), dict(exact_dtype_sites=1), {})
