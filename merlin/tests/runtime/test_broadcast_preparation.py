"""Call-local broadcast preparation is bounded and never reuses stale values."""

import shutil
import subprocess

import pytest

from merlin.llvmlower.broadcast_preparation import emit_broadcast_preparation, plan_broadcast_preparation


def plan(budget=1000):
    return plan_broadcast_preparation(
        (2, 3, 137), ((2, 3, 137), (2, 3, 1), (137,)), ((0, 1, 2), (0, 1, 2), (2,)), element_bytes=2, byte_budget=budget
    )


def test_budget_selects_real_reuse_and_bounds_all_simultaneous_preparations():
    row, channel = plan()
    assert (row.input_index, row.elements, row.bytes, row.repeated_preparations_removed) == (1, 6, 12, 816)
    assert (channel.input_index, channel.elements, channel.bytes, channel.repeated_preparations_removed) == (
        2,
        137,
        274,
        685,
    )
    assert plan(12) == (row,)
    assert plan(274) == (row,)  # Budget includes all concurrently live spans.
    assert plan(286) == (row, channel)
    assert plan(0) == ()


def test_emitted_conversions_refresh_every_element_and_accumulate_refusal(tmp_path):
    if not shutil.which("cc"):
        pytest.skip("host C compiler unavailable")
    row, _ = plan()
    code = emit_broadcast_preparation(row, source="x", destination="prepared", conversion="convert", status="safe")
    source, binary = tmp_path / "prepare.c", tmp_path / "prepare"
    source.write_text(f"""#include <stdint.h>
      #include <stddef.h>
      static unsigned count;
      static uint16_t convert(int x,unsigned *safe) {{++count;*safe &= x>=0;return (uint16_t)x;}}
      int main(void) {{
        int x[6]={{1,2,3,4,5,6}}; uint16_t prepared[7]={{0,0,0,0,0,0,0xa55a}};
        unsigned safe=1;
        {code}
        if(count!=6 || !safe || prepared[6]!=0xa55a)return 1;
        for(unsigned i=0;i<6;++i)if(prepared[i]!=(uint16_t)x[i])return 2;
        x[0]=42;x[5]=-1;safe=1;
        {code}
        if(count!=12 || safe || prepared[0]!=42 || prepared[5]!=65535 || prepared[6]!=0xa55a)return 3;
        return 0;
      }}""")
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_budget_includes_each_prepared_spans_alignment_padding():
    values = plan_broadcast_preparation(
        (1, 3, 137), ((1, 3, 1), (137,)), ((0, 1, 2), (2,)), element_bytes=2, alignment_bytes=64, byte_budget=384
    )
    assert [value.bytes for value in values] == [64, 320]
    limited = plan_broadcast_preparation(
        (1, 3, 137), ((1, 3, 1), (137,)), ((0, 1, 2), (2,)), element_bytes=2, alignment_bytes=64, byte_budget=383
    )
    assert limited == values[:1]


@pytest.mark.parametrize("element_bytes,budget", [(0, 12), (True, 12), (2, -1), (2, True)])
def test_bad_resource_contract_refuses(element_bytes, budget):
    with pytest.raises(ValueError):
        plan_broadcast_preparation((3,), ((1,),), ((0,),), element_bytes=element_bytes, byte_budget=budget)


def test_invalid_maps_and_unsafe_generated_names_refuse():
    with pytest.raises(ValueError):
        plan_broadcast_preparation((3,), ((2,),), ((0,),), element_bytes=2, byte_budget=12)
    row, _ = plan()
    for source, destination, conversion, status in (
        ("x", "x", "convert", None),
        ("x", "y", "f(x)", None),
        ("x", "y", "convert", "a;b"),
    ):
        with pytest.raises(ValueError):
            emit_broadcast_preparation(
                row, source=source, destination=destination, conversion=conversion, status=status
            )
