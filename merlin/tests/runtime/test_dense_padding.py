"""Private dense padding materializes only axes not handled by the consumer."""

import shutil
import subprocess

import pytest

from merlin.llvmlower.dense_padding import DenseZeroPadding, emit_dense_zero_padding


@pytest.mark.parametrize("axes,storage", [((), (4, 5)), ((0,), (2, 5)), ((1,), (4, 3)), ((0, 1), (2, 3))])
def test_generated_copy_preserves_live_values_and_only_fills_owned_padding(tmp_path, axes, storage):
    plan = DenseZeroPadding((2, 3), (4, 5), axes)
    assert plan.materialized_shape == storage
    if axes == (0, 1):
        assert plan.elements == 0 and emit_dense_zero_padding(plan, source="x", destination="y") == ""
        return
    assert plan.elements == storage[0] * storage[1]
    if not shutil.which("cc"):
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "padding.c"
    code = emit_dense_zero_padding(plan, source="x", destination="y")
    source.write_text(f'''#include <stddef.h>
      #include <stdint.h>
      #include <string.h>
      int main(void) {{
        const uint32_t bits[]={{0,0x80000000u,0x7fc01234u,0x3f800000u,1u,0xff800000u}};
        float x[6], y[{plan.elements + 2}];
        memcpy(x,bits,sizeof(bits)); memset(y,0xa5,sizeof(y));
        {code}
        for(size_t r=0;r<{storage[0]};++r) for(size_t c=0;c<{storage[1]};++c) {{
          uint32_t actual; memcpy(&actual,y+r*{storage[1]}+c,4);
          uint32_t expected=r<2 && c<3?bits[r*3+c]:0;
          if(actual!=expected)return 1;
        }}
        unsigned char *canary=(unsigned char *)(y+{plan.elements});
        for(size_t i=0;i<8;++i)if(canary[i]!=0xa5)return 2;
        return 0;
      }}''')
    binary = tmp_path / "padding"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
                   check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("logical,physical,axes", [((0, 2), (4, 4), ()), ((2, 5), (4, 4), ()),
    ((2, 2), (4, 4), (2,)), ((2, 2), (4, 4), (0, 0)), ((True, 2), (4, 4), ()),
    ((2, 2), (4, 4), (True,))])
def test_invalid_padding_extents_or_axes_refuse(logical, physical, axes):
    with pytest.raises(ValueError):
        DenseZeroPadding(logical, physical, axes)


@pytest.mark.parametrize("source,destination", [("x; bad", "y"), ("x", "x"), ("", "y")])
def test_codegen_refuses_unsafe_or_identical_symbols(source, destination):
    with pytest.raises(ValueError):
        emit_dense_zero_padding(DenseZeroPadding((1, 1), (2, 2)), source=source, destination=destination)
