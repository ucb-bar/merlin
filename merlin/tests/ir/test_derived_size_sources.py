"""Only dominating, injective source equalities transfer dimension bounds."""

import pytest
from xdsl.dialects.func import CallOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.frontends.model_shape_facts import ShapeFacts

SOURCE = """module {func.func @forward(%x:tensor<?xi32>,%y:tensor<?xi32>)->tensor<?xi32> {
  %zero=arith.constant 0:index
  %nx=tensor.dim %x,%zero:tensor<?xi32>
  %ny=tensor.dim %y,%zero:tensor<?xi32>
  %xw=arith.index_castui %nx:index to i64
  %yw=arith.index_castui %ny:index to i64
  %xb=arith.index_castui %xw:i64 to index
  %yb=arith.index_castui %yw:i64 to index
  %xr=arith.cmpi eq,%nx,%xb:index
  %yr=arith.cmpi eq,%ny,%yb:index
  cf.assert %xr,"x checked transport"
  cf.assert %yr,"y checked transport"
  %low=arith.constant 4:i64
  %high=arith.constant 31:i64
  %max=arith.constant 9223372036854775807:i64
  %lo=arith.cmpi uge,%xw,%low:i64
  %hi=arith.cmpi ule,%xw,%high:i64
  %yhi=arith.cmpi ule,%yw,%max:i64
  cf.assert %lo,"x lower"
  cf.assert %hi,"x upper"
  cf.assert %yhi,"y size domain"
  %same=arith.cmpi eq,%xw,%yw:i64
  cf.assert %same,"actual shared source dimensions"
  %two=arith.constant 2:i64
  %half=arith.floordivsi %yw,%two:i64
  %size=arith.index_castui %half:i64 to index
  %out=func.call @kernel(%y,%size):(tensor<?xi32>,index)->tensor<?xi32>
  return %out:tensor<?xi32>
}
func.func private @kernel(tensor<?xi32>,index)->tensor<?xi32>
}"""


def enclosure(source):
    module = parse_mlir_text(source)
    module.verify()
    entry = next(iter(module.ops))
    facts = ShapeFacts.entry(entry, expressions=True)
    call = next(op for op in entry.walk() if isinstance(op, CallOp))
    return facts.range_at(call.operands[1], call)


def test_shared_source_words_bound_actual_arithmetic_without_shape_tags():
    assert enclosure(SOURCE) == (2, 15)
    assert enclosure(SOURCE.replace('"actual shared source dimensions"', '"unrelated message"')) == (2, 15)


@pytest.mark.parametrize("damage", ("equality", "round-trip", "late", "sibling", "narrow"))
def test_incomplete_source_relationship_never_inherits_another_axis_range(damage):
    if damage == "equality":
        source = SOURCE.replace('cf.assert %same,"actual shared source dimensions"', "")
    elif damage == "round-trip":
        source = SOURCE.replace('cf.assert %yr,"y checked transport"', "")
    elif damage == "late":
        source = SOURCE.replace('cf.assert %same,"actual shared source dimensions"', "")
        source = source.replace("return %out:", 'cf.assert %same,"late equality"\n return %out:')
    elif damage == "sibling":
        source = SOURCE.replace(
            'cf.assert %same,"actual shared source dimensions"', 'scf.if %same {cf.assert %same,"branch-only equality"}'
        )
    else:
        source = SOURCE.replace(
            "%same=arith.cmpi eq,%xw,%yw:i64",
            "%xn=arith.trunci %xw:i64 to i4\n %yn=arith.trunci %yw:i64 to i4\n %same=arith.cmpi eq,%xn,%yn:i4",
        )
    interval = enclosure(source)
    assert interval is None or interval[1] > 31
