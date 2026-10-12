"""Typed mixed host routing preserves source bodies, not numerical certification."""

from dataclasses import asdict

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_mixed_patterns import FORMS, recognize, validate_pattern
from merlin.frontends.linalg_patterns import InvalidLinalgPattern


def _module(name, *, init_read=False, bad_literal=False):
    types, body = FORMS[name]
    inputs, output = types[:-2], types[-1]
    tensors = [f"tensor<2x3x{t}>" for t in inputs]
    ot = f"tensor<2x3x{output}>"
    args = ", ".join(f"%x{i}: {t}" for i, t in enumerate(tensors))
    block = ", ".join(f"%a{i}: {t}" for i, t in enumerate(types[:-1]))
    refs = [f"%a{i}" for i in range(len(inputs))]
    instructions = []
    ops = body[:-1]
    if ops[0] == "arith.constant":
        instructions.append(
            f'%lit = "arith.constant"() <{{value = {2.0 if bad_literal else 1.0} : {output}}}> : () -> {output}'
        )
        instructions.append(f'%v = "{ops[1]}"(%lit, %a0) : ({output}, {inputs[0]}) -> {output}')
    elif ops[0] == "arith.extf" and len(ops) == 2:
        narrow = next(i for i, t in enumerate(inputs) if t != output)
        instructions.append(f'%ext = "arith.extf"(%a{narrow}) : ({inputs[narrow]}) -> {output}')
        refs[narrow] = "%ext"
        instructions.append(f'%v = "{ops[1]}"({", ".join(refs)}) : ({output}, {output}) -> {output}')
    else:
        instructions.append(f'%v = "{ops[0]}"({", ".join(refs)}) : ({", ".join(inputs)}) -> {output}')
    instructions.append(
        f'"linalg.yield"(%a{len(inputs)}' + f") : ({output}) -> ()"
        if init_read
        else f'"linalg.yield"(%v) : ({output}) -> ()'
    )
    maps = ", ".join("affine_map<(d0,d1)->(d0,d1)>" for _ in types[:-1])
    return mq.parse(f"""module {{ func.func @f({args}, %init: {ot}) -> {ot} {{
      %r = "linalg.generic"({", ".join(f"%x{i}" for i in range(len(inputs)))}, %init)
      <{{indexing_maps = [{maps}],
         iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
         operandSegmentSizes = array<i32: {len(inputs)}, 1>}}>
      ({{ ^bb0({block}): {" ".join(instructions)} }}) : ({", ".join(tensors)}, {ot}) -> {ot}
      return %r : {ot}
    }} }}""")


@pytest.mark.parametrize("name", list(FORMS))
def test_mixed_body_exact_source_and_serialized_geometry(name):
    op = tuple(mq.walk(_module(name)))[2]
    pattern = recognize(op)
    assert pattern.operation == name
    assert validate_pattern(asdict(pattern))["ordered_types"] == FORMS[name][0]
    bad = asdict(pattern)
    bad["input_maps"] = ("((d0, d1) -> (d1, d0))",) * len(pattern.input_maps)
    with pytest.raises(InvalidLinalgPattern):
        validate_pattern(bad)
    with pytest.raises(InvalidLinalgPattern):
        recognize(tuple(mq.walk(_module(name, init_read=True)))[2])


def test_mixed_reciprocal_requires_unit_numerator():
    name = next(n for n, (_, body) in FORMS.items() if "arith.divf" in body)
    with pytest.raises(InvalidLinalgPattern):
        recognize(tuple(mq.walk(_module(name, bad_literal=True)))[2])
