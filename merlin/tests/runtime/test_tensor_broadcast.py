"""Broadcast coordinates agree with dense array semantics across chunk tails."""

import shutil
import subprocess
from math import prod

import numpy as np
import pytest
from xdsl.context import Context
from xdsl.parser import Parser

from merlin.llvmlower.tensor_broadcast import dense_broadcast_index, singleton_projection_axes


@pytest.mark.parametrize("shape,operand,axes", [
    ((2, 5, 137), (2, 5, 137), (0, 1, 2)),
    ((2, 5, 137), (2, 5, 1), (0, 1, 2)),
    ((2, 5, 137), (137,), (2,)),
    ((2, 5, 137), (2, 137), (0, 2)),
    ((2, 5, 137), (), ()),
    ((2, 5, 137), (1,), (1,)),
    ((1, 3, 2, 5), (3, 1, 5), (1, 2, 3)),
])
def test_compiled_index_agrees_with_array_broadcast_for_every_lane(tmp_path, shape, operand, axes):
    if not shutil.which("cc"):
        pytest.skip("host C compiler unavailable")
    expanded = [1] * len(shape)
    for axis, extent in zip(axes, operand, strict=True):
        expanded[axis] = extent
    expected = np.broadcast_to(np.arange(prod(operand)).reshape(expanded), shape).reshape(-1)
    expr = dense_broadcast_index(shape, operand, axes)
    source, binary = tmp_path / "index.c", tmp_path / "index"
    source.write_text(f'''#include <stddef.h>
      int main(void) {{
        const size_t expected[]={{{','.join(map(str, expected))}}};
        for(size_t base=0;base<{prod(shape)}u;base+=113u)
          for(size_t index=0;index<113u && base+index<{prod(shape)}u;++index)
            if(({expr})!=expected[base+index])return 1;
        return 0;
      }}''')
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("shape,operand,axes", [
    ((2, 3), (2,), (1,)), ((2, 3), (2,), (True,)),
    ((2, 3), (2, 3), (1, 0)), ((2, 3), (1, 1), (0, 0)),
    ((2, 0), (1,), (0,)), ((True, 3), (1,), (0,)),
    ((2, 3), (1,), (2,)), ((2, 3), (1,), ()),
])
def test_unrepresentable_or_malformed_coordinates_refuse(shape, operand, axes):
    with pytest.raises(ValueError):
        dense_broadcast_index(shape, operand, axes)


def test_actual_maps_bind_axes_and_refuse_permutations_and_extra_singletons():
    def parse(text):
        return Parser(Context(), text).parse_affine_map()
    mapping = parse("(d0, d1, d2) -> (d0, d1, 0)")
    assert singleton_projection_axes(mapping, (2, 5, 137), (2, 5, 1)) == (0, 1, 2)
    assert singleton_projection_axes(parse("(d0, d1, d2) -> (d2)"), (2, 5, 137), (137,)) == (2,)
    with pytest.raises(ValueError):
        singleton_projection_axes(parse("(d0, d1) -> (d1, d0)"), (2, 3), (3, 2))
    with pytest.raises(ValueError, match="embedding"):
        singleton_projection_axes(parse("(d0) -> (0, d0)"), (3,), (1, 3))


@pytest.mark.parametrize("name", ["a; return", "", "x+y", "λ"])
def test_emitter_refuses_non_c_identifiers(name):
    with pytest.raises(ValueError):
        dense_broadcast_index((3,), (3,), (0,), index=name)
