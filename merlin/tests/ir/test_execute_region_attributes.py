"""Exactly-once source boundaries and provenance survive both reader forms."""

import subprocess

import pytest
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from merlin.llvmlower import toolchain
from merlin.xdsl_dialects._common import text


def source(flag="no_inline", attributes='prov.op="atomic-snapshot",prov.region_id="source/region"'):
    return f'''module {{func.func @f(%x:i32)->i32 {{
      %r=scf.execute_region -> i32 {flag} {{scf.yield %x:i32}} {{{attributes}}}
      return %r:i32
    }}}}'''


@pytest.mark.parametrize("flag,attributes", [
    ("no_inline", 'prov.op="atomic-snapshot",prov.region_id="source/region"'),
    ("", 'no_inline,prov.op="atomic-snapshot",prov.region_id="source/region"'),
    ("", 'prov.op="ordinary",prov.region_id="source/region"'),
])
def test_custom_and_generic_roundtrips_preserve_every_attribute(flag, attributes):
    module = parse_mlir_text(source(flag, attributes))
    module.verify()
    original, = [op for op in module.walk() if op.name == "scf.execute_region"]
    for generic in (False, True):
        restored = parse_mlir_text(text(module, generic=generic))
        restored.verify()
        region, = [op for op in restored.walk() if op.name == "scf.execute_region"]
        assert region.attributes == original.attributes
        assert region.attributes["prov.region_id"].data == "source/region"


@pytest.mark.parametrize("marked", [False, True])
def test_python_canonicalization_honors_explicit_no_inline(marked):
    module = parse_mlir_text(source("no_inline" if marked else ""))
    CanonicalizePass().apply(make_context(), module)
    module.verify()
    assert len([op for op in module.walk() if op.name == "scf.execute_region"]) == int(marked)


def test_multiblock_region_retains_cfg_and_source_attributes():
    module = parse_mlir_text('''module {func.func @f(%p:i1,%x:i32,%y:i32)->i32 {
      %r=scf.execute_region -> i32 no_inline {
        cf.cond_br %p,^left,^right
      ^left: cf.br ^end(%x:i32)
      ^right: cf.br ^end(%y:i32)
      ^end(%v:i32): scf.yield %v:i32
      } {prov.op="original-cfg"}
      return %r:i32
    }}''')
    restored = parse_mlir_text(text(module))
    restored.verify()
    execute, = [op for op in restored.walk() if op.name == "scf.execute_region"]
    assert len(execute.region.blocks) == 4
    assert execute.no_inline is not None
    assert execute.attributes["prov.op"].data == "original-cfg"


@pytest.mark.parametrize("body", [
    '%r=scf.execute_region -> i32 {scf.yield %x:i32} {no_inline="false"}',
    '"scf.execute_region"() ({^bb0(%arg:i32): scf.yield}):()->()',
    '"scf.execute_region"() ({}):()->()',
])
def test_invalid_attribute_and_region_shapes_remain_refused(body):
    module = parse_mlir_text('module {func.func @f(%x:i32) {' + body + ' return}}')
    with pytest.raises(VerifyException):
        module.verify()


def test_void_custom_print_retains_terminator_and_native_syntax(tmp_path):
    if not toolchain.mlir_translate().with_name("mlir-opt").is_file():
        pytest.skip("selected native MLIR tools required")
    module = parse_mlir_text('''module {func.func @f() {
      scf.execute_region no_inline {scf.yield} {prov.op="void-event"}
      return
    }}''')
    module.verify()
    printed = text(module)
    assert "scf.yield" in printed
    path = tmp_path / "void.mlir"
    path.write_text(printed)
    result = subprocess.run([str(toolchain.mlir_translate().with_name("mlir-opt")), str(path), "--verify-each"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
