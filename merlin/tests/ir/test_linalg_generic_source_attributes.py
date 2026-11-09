"""A source contract cannot disappear before source matching or cache identity."""

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.utils.exceptions import ParseError

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text

SOURCE = '''module {func.func @forward(%x:tensor<2xf32>)->tensor<2xf32>{
 %e=tensor.empty():tensor<2xf32>
 %r=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]@LEADING@}
 ins(%x:tensor<2xf32>)outs(%e:tensor<2xf32>) @EXTRA@ {
 ^bb0(%v:f32,%old:f32):
 linalg.yield %v:f32
 }->tensor<2xf32>
 return %r:tensor<2xf32>
}}'''


def source(*, leading="", extra=""):
    return SOURCE.replace("@LEADING@", leading).replace("@EXTRA@", extra)


@pytest.mark.parametrize("key", ["unknown.contract", "prov.source_node_ids", "numeric.permission"])
def test_leading_dictionary_cannot_silently_erase_an_attribute(key):
    with pytest.raises(ParseError, match="would be discarded"):
        parse_mlir_text(source(leading=f', {key}="retained-value"'))


def test_explicit_extra_dictionary_is_retained_and_round_trips():
    module = parse_mlir_text(source(extra='attrs = {unknown.contract="retained-value"}'))
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert generic.attributes["unknown.contract"] == StringAttr("retained-value")
    reparsed = parse_mlir_text(text(module, generic=True))
    again = next(op for op in reparsed.walk() if op.name == "linalg.generic")
    assert again.attributes["unknown.contract"] == generic.attributes["unknown.contract"]


def test_declared_properties_remain_properties():
    module = parse_mlir_text(source(leading=', doc="known", library_call="declared_symbol"'))
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert generic.properties["doc"] == StringAttr("known")
    assert generic.properties["library_call"] == StringAttr("declared_symbol")


def test_comments_strings_and_generic_operation_names_do_not_trigger_custom_checks():
    module = parse_mlir_text(source(extra='attrs = {note="linalg.generic {not a dictionary}"}'))
    generic = text(module, generic=True)
    reparsed = parse_mlir_text('// linalg.generic { unknown.contract = "comment" }\n' + generic)
    actual = next(op for op in reparsed.walk() if op.name == "linalg.generic")
    assert actual.attributes["note"] == StringAttr("linalg.generic {not a dictionary}")


def test_file_fallback_preserves_leading_source_attributes(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower import toolchain

    if not toolchain.available():
        pytest.skip("selected upstream MLIR parser required")
    path = tmp_path / "source.mlir"
    path.write_text(source(leading=', numeric.permission="retained-value"'))
    module = parse_mlir_file(path)
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert generic.attributes["numeric.permission"] == StringAttr("retained-value")
