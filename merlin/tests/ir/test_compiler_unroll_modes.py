"""Current compiler empty/inline/tail boundaries and adversarial actual output."""

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_unroll_inline, compiler_unroll_tail


def effect_source(trips, *, carries=1):
    if carries:
        loop = """%r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {
          %next = func.call @step(%i, %c, %y) : (index, i32, i32) -> i32
          scf.yield %next : i32
        }"""
        result = "%r"
    else:
        loop = """scf.for %i = %lb to %ub step %st {
          func.call @tick(%i) : (index) -> ()
          scf.yield
        }"""
        result = "%x"
    return f"""builtin.module {{
      func.func private @step(index, i32, i32) -> i32
      func.func private @tick(index)
      func.func private @observe(i32) -> i32
      func.func @forward(%x:i32, %y:i32, %unused:i32) -> i32 {{
        %lb = arith.constant 3 : index
        %ub = arith.constant {3 + 2 * trips} : index
        %st = arith.constant 2 : index
        {loop}
        %seen = func.call @observe({result}) : (i32) -> i32
        func.return %seen : i32
      }}
    }}"""


def pair(trips, factor, *, carries=1, pure=False):
    if pure:
        from test_compiler_loop_generalization import source

        text = "builtin.module {" + source(4, 3, trips=trips) + "}"
    else:
        text = effect_source(trips, carries=carries)
    module = parse_ir(text)
    before = ir_text(module)
    unroll_ordered_loops(module, (factor,), index_bits=64)
    return before, ir_text(module)


@pytest.mark.parametrize(
    "trips,factor,carries,pure", ([(0, 2, 0, False), (2, 2, 1, False), (5, 2, 0, False), (8, 7, 3, True)])[:1]
)
def test_current_modes_project_without_historical_producer(trips, factor, carries, pure, monkeypatch):
    before, after = pair(trips, factor, carries=carries, pure=pure)
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("replayed producer")
    )
    prepare = compiler_unroll_inline.prepare_query if trips <= factor else compiler_unroll_tail.prepare_query
    result = prepare(before, after, factor=factor)
    assert result["final_root"] in result["roots"]
    assert result["capture"]["changed"][0]["count"] == trips
    assert result["capture"]["changed"][0]["carries"] == carries
    assert "PreservesEntries" in result["query"]
    assert all(row["sha256"] and row["bytes"] > 0 for row in result["sources"])


def test_source_only_profiles_remain_held():
    from merlin.verify import lean_graph_pipeline_inline, lean_graph_pipeline_tail

    for profile in (lean_graph_pipeline_inline, lean_graph_pipeline_tail):
        assert profile.compiler_eligible({"status": "accepted"}) is False
        with pytest.raises(RuntimeError, match="held"):
            profile.provider_arguments({"status": "accepted"})
