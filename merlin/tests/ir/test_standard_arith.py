"""Registered scalar unsigned index casts preserve the standard native contract."""

import pytest
from xdsl.dialects import arith
from xdsl.parser import Parser
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import make_context, text
from merlin.xdsl_dialects.standard_arith import IndexCastUIOp


@pytest.mark.parametrize("flag", ["", "nneg"])
def test_scalar_unsigned_index_cast_has_matching_frontend_and_common_readers(flag):
    source = f'''module {{func.func @forward(%a:i32)->i64 {{
      %i=arith.index_castui %a {flag} {{prov.op="shape_guard"}} : i32 to index
      %v=arith.index_castui %i : index to i64
      return %v:i64
    }}}}'''
    module = parse_mlir_text(source)
    module.verify()
    casts = [op for op in module.walk() if isinstance(op, IndexCastUIOp)]
    assert len(casts) == 2 and (casts[0].nonNeg is not None) == bool(flag)
    for generic in (False, True):
        restored = Parser(make_context(arith.Arith), text(module, generic=generic)).parse_module()
        restored.verify()
        assert text(restored, generic=True) == text(module, generic=True)


@pytest.mark.parametrize("source,target", [("index", "index"), ("i32", "i64"), ("f32", "index"), ("i0", "index")])
def test_scalar_unsigned_cast_refuses_invalid_native_type_pairs(source, target):
    module = parse_mlir_text(f'''module {{func.func @forward(%a:{source})->{target} {{
      %v=arith.index_castui %a : {source} to {target}
      return %v:{target}
    }}}}''')
    with pytest.raises(VerifyException):
        module.verify()
