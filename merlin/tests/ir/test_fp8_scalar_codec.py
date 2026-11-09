"""Structural codec admission; numerical source qualification is independent."""

import json

import pytest

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.fp8_scalar_codec import BoundedE4M3FNCodec, legalize_file
from merlin.llvmlower.host_llvm_helpers import HelperIR


def codec(tmp_path):
    from merlin.llvmlower import toolchain

    if not (toolchain.llvm_install() / "bin" / "mlir-opt").is_file():
        pytest.skip("selected stock MLIR normalizer unavailable")
    helper = tmp_path / "helper.ll"
    helper.write_text("""target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define i64 @codec_encode(float %v) { ret i64 0 }
define float @codec_decode(i64 %v) { ret float 0.0 }
""")
    source = tmp_path / "source.cpp"
    source.write_text("// Interface-only fixture, not a numerical codec proof.\n")
    recipe = tmp_path / "compile.json"
    recipe.write_text('{"scope":"structural unit-test fixture only"}')

    def bind(p):
        return HelperIR(p, sha256_file(p))

    return BoundedE4M3FNCodec(
        bind(helper),
        (bind(source),),
        bind(recipe),
        "codec_encode",
        "codec_decode",
        "rne",
        "nontrapping_unobserved",
        True,
    )


CLAMP = """module {
func.func @forward(%x:f32)->f8E4M3FN {
  %lo=arith.constant -448.0:f32
  %hi=arith.constant 448.0:f32
  %below=arith.cmpf olt,%x,%lo:f32
  %lower=arith.select %below,%lo,%x:f32
  %above=arith.cmpf ogt,%lower,%hi:f32
  %bounded=arith.select %above,%hi,%lower:f32
  %f=arith.truncf %bounded:f32 to f8E4M3FN
  return %f:f8E4M3FN
}}
"""


def test_none_has_no_io_or_source_discovery(tmp_path):
    missing = tmp_path / "absent.mlir"
    assert legalize_file(missing, tmp_path / "no-output", None) == missing
    assert not (tmp_path / "no-output").exists()


def test_clamp_cast_retains_public_byte_storage_and_original_selects(tmp_path):
    policy = codec(tmp_path)
    source = tmp_path / "source.mlir"
    source.write_text(CLAMP)
    original = source.read_bytes()
    output = legalize_file(source, tmp_path / "out", policy)
    module = parse_mlir_text(output.read_text())
    module.verify()
    assert source.read_bytes() == original
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    assert str(function.function_type.outputs.data[0]) == "f8E4M3FN"
    names = [op.name for op in module.walk()]
    assert names.count("arith.select") == 2
    assert "arith.truncf" not in names
    assert names.count("func.call") == 1 and names.count("arith.bitcast") == 1
    report = json.loads((output.parent / "report.json").read_text())
    assert report["casts"] == {"decode": 0, "encode": 1}


@pytest.mark.parametrize(
    "change",
    [
        ("%bounded:f32 to", "%x:f32 to"),
        ("-448.0", "-449.0"),
        ("448.0:f32", "449.0:f32"),
        ("cmpf olt", "cmpf ult"),
        ("%below,%lo,%x", "%below,%x,%lo"),
    ],
)
def test_unknown_or_out_of_domain_truncation_refuses(tmp_path, change):
    policy = codec(tmp_path)
    source = tmp_path / "source.mlir"
    source.write_text(CLAMP.replace(*change))
    with pytest.raises(ValueError, match="SSA bound|unknown generic contract"):
        legalize_file(source, tmp_path / "out", policy)
    assert not (tmp_path / "out/model.mlir").exists()


def test_decode_uses_unsigned_byte_container_with_no_external_widening(tmp_path):
    policy = codec(tmp_path)
    source = tmp_path / "source.mlir"
    source.write_text("""module {
func.func @forward(%a:tensor<7xf8E4M3FN>)->tensor<7xf32> {
 %empty=tensor.empty():tensor<7xf32>
 %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%a:tensor<7xf8E4M3FN>) outs(%empty:tensor<7xf32>) {
 ^bb0(%x:f8E4M3FN,%c:f32):
  %v=arith.extf %x:f8E4M3FN to f32
  linalg.yield %v:f32
 }->tensor<7xf32>
 return %r:tensor<7xf32>
}}""")
    output = legalize_file(source, tmp_path / "out", policy)
    module = parse_mlir_text(output.read_text())
    module.verify()
    assert "tensor<7xf8E4M3FN>" in str(module)
    names = [op.name for op in module.walk()]
    assert "arith.extf" not in names and names.count("arith.extui") == 1


def test_changed_or_wrong_abi_helper_refuses(tmp_path):
    policy = codec(tmp_path)
    source = tmp_path / "source.mlir"
    source.write_text(CLAMP)
    policy.helper_ir.path.write_text(policy.helper_ir.path.read_text().replace("define i64", "define zeroext i64"))
    with pytest.raises(ValueError, match="changed"):
        legalize_file(source, tmp_path / "out", policy)
    from dataclasses import replace

    bad = replace(policy, helper_ir=HelperIR(policy.helper_ir.path, sha256_file(policy.helper_ir.path)))
    with pytest.raises(ValueError, match="ABI"):
        legalize_file(source, tmp_path / "out", bad)


def test_exact_effect_permission_is_required(tmp_path):
    from dataclasses import replace

    policy = codec(tmp_path)
    source = tmp_path / "source.mlir"
    source.write_text(CLAMP)
    for bad in (replace(policy, rounding="rtz"), replace(policy, nan_payload_unobserved=False)):
        with pytest.raises(ValueError, match="effect"):
            legalize_file(source, tmp_path / "out", bad)


def test_bitcast_registration_retains_upstream_equal_width_refusal():
    from xdsl.utils.exceptions import VerifyException

    from merlin.xdsl_dialects.fp8 import register_fp8_bitcast_constraints

    register_fp8_bitcast_constraints()
    module = parse_mlir_text("""module {
func.func @bad(%a:f8E4M3FN)->i16 {
 %b=arith.bitcast %a:f8E4M3FN to i16
 return %b:i16
}}""")
    with pytest.raises(VerifyException, match="equal bitwidths"):
        module.verify()


@pytest.mark.parametrize("rank_zero", [False, True])
def test_current_elementwise_tensor_clamp_proves_following_cast(tmp_path, rank_zero):
    policy = codec(tmp_path)
    source = tmp_path / "source.mlir"
    body = CLAMP[CLAMP.index("  %lo=") : CLAMP.index("  %f=")]
    source.write_text(
        """module {
func.func @forward(%a:tensor<7xf32>)->tensor<7xf8E4M3FN> {
 %empty=tensor.empty():tensor<7xf32>
 %clamped=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%a:tensor<7xf32>) outs(%empty:tensor<7xf32>) {
 ^bb0(%x:f32,%c:f32):
"""
        + body
        + """linalg.yield %bounded:f32
 }->tensor<7xf32>
 %empty8=tensor.empty():tensor<7xf8E4M3FN>
 %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%clamped:tensor<7xf32>) outs(%empty8:tensor<7xf8E4M3FN>) {
 ^bb0(%x:f32,%c:f8E4M3FN):
  %v=arith.truncf %x:f32 to f8E4M3FN
  linalg.yield %v:f8E4M3FN
 }->tensor<7xf8E4M3FN>
 return %r:tensor<7xf8E4M3FN>
}}"""
    )
    if rank_zero:
        text = (
            source.read_text().replace("tensor<7xf32>", "tensor<f32>").replace("tensor<7xf8E4M3FN>", "tensor<f8E4M3FN>")
        )
        text = text.replace("affine_map<(d0)->(d0)>", "affine_map<()->()>").replace(
            'iterator_types=["parallel"]', "iterator_types=[]"
        )
        source.write_text(text)
    result = legalize_file(source, tmp_path / "out", policy)
    parse_mlir_text(result.read_text()).verify()
    assert '"arith.truncf"' not in result.read_text()


@pytest.mark.parametrize("dtype", ["f8E5M2", "vector<4xf8E4M3FN>"])
def test_other_format_and_nonscalar_casts_refuse_before_partial_rewrite(tmp_path, dtype):
    policy = codec(tmp_path)
    source = tmp_path / "source.mlir"
    out_type = "vector<4xf32>" if dtype.startswith("vector") else "f32"
    source.write_text(
        "module {func.func @forward(%x:" + dtype + ")->" + out_type + " {\n"
        "%r=arith.extf %x:" + dtype + " to " + out_type + "\nreturn %r:" + out_type + "\n}}"
    )
    from xdsl.utils.exceptions import VerifyException

    with pytest.raises((ValueError, VerifyException)):
        legalize_file(source, tmp_path / "out", policy)
    assert not (tmp_path / "out/model.mlir").exists()


PARTIAL_CLAMP = """module {
func.func @forward(%input:tensor<1xf32>,%initial:tensor<2xf32>)->tensor<2xf8E4M3FN> {
 %r=linalg.generic {indexing_maps=[affine_map<()->(0)>,affine_map<()->(0)>],iterator_types=[]}
 ins(%input:tensor<1xf32>) outs(%initial:tensor<2xf32>) {
 ^bb0(%x:f32,%old:f32):
  %lo=arith.constant -448.0:f32
  %hi=arith.constant 448.0:f32
  %below=arith.cmpf olt,%x,%lo:f32
  %lower=arith.select %below,%lo,%x:f32
  %above=arith.cmpf ogt,%lower,%hi:f32
  %bounded=arith.select %above,%hi,%lower:f32
  linalg.yield %bounded:f32
 }->tensor<2xf32>
 %empty=tensor.empty():tensor<2xf8E4M3FN>
 %out=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%r:tensor<2xf32>) outs(%empty:tensor<2xf8E4M3FN>) {
 ^bb0(%x:f32,%old:f8E4M3FN):
  %encoded=arith.truncf %x:f32 to f8E4M3FN
  linalg.yield %encoded:f8E4M3FN
 }->tensor<2xf8E4M3FN>
 return %out:tensor<2xf8E4M3FN>
}}"""


def test_partial_output_clamp_does_not_bound_retained_initial_elements(tmp_path):
    policy = codec(tmp_path)
    source = tmp_path / "partial.mlir"
    source.write_text(PARTIAL_CLAMP)
    parse_mlir_text(source.read_text()).verify()
    with pytest.raises(ValueError, match="SSA bound|unknown generic contract"):
        legalize_file(source, tmp_path / "out", policy)


@pytest.mark.parametrize("where", ["producer", "consumer"])
@pytest.mark.parametrize("contract", ["library_call", "unknown.contract"])
def test_clamp_body_cannot_launder_unknown_generic_contracts(tmp_path, where, contract):
    from xdsl.dialects.builtin import StringAttr

    from merlin.llvmlower.deployable_ir import checked_generic_text

    policy = codec(tmp_path)
    # Restore complete equal-shape identity coverage so only the unknown
    # contract, not partial coverage, blocks this attempted proof.
    text = PARTIAL_CLAMP.replace("tensor<1xf32>", "tensor<2xf32>")
    text = text.replace("affine_map<()->(0)>", "affine_map<(d0)->(d0)>")
    text = text.replace("iterator_types=[]", 'iterator_types=["parallel"]')
    module = parse_mlir_text(text)
    generic = [op for op in module.walk() if op.name == "linalg.generic"][0 if where == "producer" else 1]
    table = generic.properties if contract == "library_call" else generic.attributes
    table[contract] = StringAttr("unproved")
    source = tmp_path / "contract.mlir"
    source.write_text(checked_generic_text(module))
    with pytest.raises(ValueError, match="SSA bound|unknown generic contract"):
        legalize_file(source, tmp_path / "out", policy)


def test_stock_normalization_retains_unknown_custom_form_attributes(tmp_path):
    policy = codec(tmp_path)
    text = PARTIAL_CLAMP.replace("tensor<1xf32>", "tensor<2xf32>")
    text = text.replace("affine_map<()->(0)>", "affine_map<(d0)->(d0)>")
    text = text.replace("iterator_types=[]", 'iterator_types=["parallel"],audit.unproved_effects="unknown"')
    source = tmp_path / "custom.mlir"
    source.write_text(text)
    with pytest.raises(ValueError, match="SSA bound|unknown generic contract"):
        legalize_file(source, tmp_path / "out", policy)
    assert "audit.unproved_effects" in (tmp_path / "out/normalized.mlir").read_text()


def _complete_pair():
    text = PARTIAL_CLAMP.replace("tensor<1xf32>", "tensor<2xf32>")
    return text.replace("affine_map<()->(0)>", "affine_map<(d0)->(d0)>").replace(
        "iterator_types=[]", 'iterator_types=["parallel"]'
    )


def test_stock_fusion_keeps_original_clamp_local_and_both_provenances(tmp_path):
    from xdsl.dialects.builtin import StringAttr

    from merlin.llvmlower.deployable_ir import checked_generic_text
    from merlin.llvmlower.fp8_encode_fusion import fuse_clamped_encoders
    from merlin.llvmlower.fp8_scalar_codec import _domain

    policy = codec(tmp_path)
    module = parse_mlir_text(_complete_pair())
    generics = [op for op in module.walk() if op.name == "linalg.generic"]
    generics[0].attributes["prov.fqn"] = StringAttr("producer-source")
    generics[1].attributes["prov.fqn"] = StringAttr("consumer-source")
    source = tmp_path / "source.mlir"
    source.write_text(checked_generic_text(module))
    original = source.read_bytes()
    selected = fuse_clamped_encoders(source, tmp_path / "fusion", policy)
    assert source.read_bytes() == original
    fused = parse_mlir_text(selected.read_text())
    fused.verify()
    generics = [op for op in fused.walk() if op.name == "linalg.generic"]
    assert len(generics) == 1
    records = generics[0].attributes["prov.fused_elementwise_sources"]
    assert [row.data["prov.fqn"].data for row in records] == ["producer-source", "consumer-source"]
    cast = next(op for op in generics[0].walk() if op.name == "arith.truncf")
    assert _domain(cast.operands[0]) == (-448.0, 448.0)
    assert [op.name for op in generics[0].body.block.ops].count("arith.constant") == 2
    legalized = legalize_file(selected, tmp_path / "legalized", policy)
    parse_mlir_text(legalized.read_text()).verify()


def test_source_fusion_never_crosses_an_assertion(tmp_path):
    from merlin.llvmlower.fp8_encode_fusion import fuse_clamped_encoders

    policy = codec(tmp_path)
    source = tmp_path / "source.mlir"
    text = _complete_pair().replace(
        " %empty=tensor.empty()", ' %yes=arith.constant true\n cf.assert %yes,"do not cross"\n %empty=tensor.empty()'
    )
    source.write_text(text)
    selected = fuse_clamped_encoders(source, tmp_path / "fusion", policy)
    module = parse_mlir_text(selected.read_text())
    assert sum(op.name == "linalg.generic" for op in module.walk()) == 2
    assert sum(op.name == "cf.assert" for op in module.walk()) == 1
    assert json.loads((selected.parent / "report.json").read_text())["pairs"] == []


def test_source_fusion_none_performs_no_reads_or_discovery(tmp_path):
    from merlin.llvmlower.fp8_encode_fusion import fuse_clamped_encoders

    source = tmp_path / "missing.mlir"
    assert fuse_clamped_encoders(source, tmp_path / "absent", None) == source
    assert not (tmp_path / "absent").exists()


def test_private_scalar_codec_declarations_keep_original_llvm_abi(tmp_path):
    from xdsl.dialects.func import FuncOp

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import passes_xdsl

    module = parse_mlir_text("""module {
      func.func @entry(%x: f32) -> i64 {
        %v = func.call @private_codec(%x) : (f32) -> i64
        return %v : i64
      }
      func.func private @private_codec(f32) -> i64
      func.func private @internal(%x: f32) -> f32 { return %x : f32 }
      func.func private @external(f32) -> f32
    }""")
    assert passes_xdsl.add_c_interface(module) == 1
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, FuncOp)}
    assert "llvm.emit_c_interface" in functions["entry"].attributes
    assert all(
        "llvm.emit_c_interface" not in functions[name].attributes for name in ("private_codec", "internal", "external")
    )
    import subprocess

    from merlin.llvmlower import toolchain
    from merlin.llvmlower.deployable_ir import checked_generic_text

    source, result = tmp_path / "input.mlir", tmp_path / "llvm.mlir"
    source.write_text(checked_generic_text(module))
    cmd = [
        toolchain.llvm_install() / "bin/mlir-opt",
        source,
        "--pass-pipeline=builtin.module(convert-func-to-llvm,convert-arith-to-llvm)",
        "-o",
        result,
    ]
    proc = subprocess.run([str(v) for v in cmd], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    text = result.read_text()
    assert "_mlir_ciface_entry" in text
    assert "_mlir_ciface_private_codec" not in text
    assert "_mlir_ciface_internal" not in text
    assert "_mlir_ciface_external" not in text


def test_explicit_external_c_interface_contract_is_not_erased():
    from xdsl.dialects.func import FuncOp

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import passes_xdsl

    module = parse_mlir_text("""module {
      func.func private @tensor_callback(memref<4xf32>) attributes {llvm.emit_c_interface}
      func.func private @private_body(%a: memref<4xf32>) attributes {llvm.emit_c_interface} { return }
    }""")
    assert passes_xdsl.add_c_interface(module) == 0
    assert all("llvm.emit_c_interface" in op.attributes for op in module.body.block.ops if isinstance(op, FuncOp))


@pytest.mark.parametrize("attribute", ["audit.unproved_effects", "prov.unproved_contract", "prov.fqn"])
def test_scalar_cast_requires_closed_typed_attribute_contract(tmp_path, attribute):
    from xdsl.dialects.builtin import StringAttr, UnitAttr

    from merlin.llvmlower.deployable_ir import checked_generic_text

    policy = codec(tmp_path)
    module = parse_mlir_text(CLAMP)
    cast = next(op for op in module.walk() if op.name == "arith.truncf")
    cast.attributes[attribute] = UnitAttr() if attribute == "prov.fqn" else StringAttr("unknown")
    source = tmp_path / "unknown-cast.mlir"
    source.write_text(checked_generic_text(module))
    with pytest.raises(ValueError, match="unsupported FP8 scalar conversion"):
        legalize_file(source, tmp_path / "out", policy)
    assert not (tmp_path / "out/model.mlir").exists()
