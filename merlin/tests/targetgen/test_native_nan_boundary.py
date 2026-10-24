"""Storage transport with a proved NaN-only epilogue retains every other bit."""

import pytest
from mlir_fixtures import generic
from targetgen_fixtures import selector_catalog

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_exact_dtype import bitwise_catalog_coverage
from merlin.targetgen.native_nan_boundary import expose_device_transport, prove


def source(dtype="f32", *, zero=False, reversed=False, finite_mask=False, unknown=False):
    constants = {"f16": "0x7e00", "bf16": "0x7fc0", "f32": "0x7fc01234", "f64": "0x7ff8123456789abc"}
    value = "0.0" if zero else constants[dtype]
    extra = '%opaque="fixture.opaque"(%word) : (' + dtype + ")->" + dtype if unknown else ""
    if finite_mask:
        n = 64 if dtype == "f64" else 32 if dtype == "f32" else 16
        extra = f"""%raw=arith.bitcast %word : {dtype} to i{n}
          %mask=arith.constant 1 : i{n}
          %changed=arith.xori %raw,%mask : i{n}
          %bad=arith.bitcast %changed : i{n} to {dtype}"""
    data = "%bad" if finite_mask else "%word"
    yes, no = (data, "%nan") if reversed else ("%nan", data)
    t = f"tensor<4x{dtype}>"
    body = f"""%nan=arith.constant {value} : {dtype}
      {extra}
      %p=arith.cmpf uno,%word,%word : {dtype}
      %v=arith.select %p,{yes},{no} : {dtype}"""
    epilogue = generic([("%x", t)], t, body, name="%out", args=["word"])
    return parse_mlir_text(
        f"module {{func.func @forward(%x:{t}) -> ({t},{t}) {{\n{epilogue}\nreturn %out,%x : {t},{t} }} }}"
    )


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
def test_all_input_words_are_identity_outside_the_explicit_nan_boundary(dtype):
    module = source(dtype)
    op = next(op for op in module.walk() if op.name == "linalg.generic")
    proof = prove(op)
    assert proof and proof["input_domain"] == "all_storage_bit_patterns"
    assert proof["runtime_input_observations"] is False
    assert len(expose_device_transport(module)) == 1
    module.verify()
    assert op.inputs[0].owner.name == "linalg.generic"


@pytest.mark.parametrize("options", [dict(zero=True), dict(reversed=True), dict(finite_mask=True), dict(unknown=True)])
def test_numeric_changes_and_unknown_operations_refuse(options):
    op = next(op for op in source(**options).walk() if op.name == "linalg.generic")
    assert prove(op) is None


def test_word_epilogue_requires_real_device_storage_lineage():
    module = source()
    expose_device_transport(module)
    operations = list(module.walk())
    transport = next(op for op in operations if op.name == "linalg.generic")
    ordinal = operations.index(transport)
    manifest = selector_catalog(module, "transport", (ordinal, dict(x=dict(func_arg=0))))
    assert bitwise_catalog_coverage(module, manifest) == [ordinal]
    manifest["bindings"] = []
    with pytest.raises(ValueError, match="no source-bound"):
        bitwise_catalog_coverage(module, manifest)
