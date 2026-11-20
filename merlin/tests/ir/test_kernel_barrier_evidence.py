"""Independent source phase checks require real control and memory payload."""

from copy import deepcopy

import pytest

from merlin.frontends.kernel_barrier_evidence import checked_program_barriers


def fixture():
    original = """"builtin.module"() ({
      "tt.func"() <{sym_name="exchange",function_type=(!tt.ptr<i64>)->()}> ({
      ^bb0(%p:!tt.ptr<i64>):
        %x="arith.constant"() <{value=4:i64}> : ()->i64
        "tt.store"(%p,%x):(!tt.ptr<i64>,i64)->()
        "ttg.barrier"() <{addrSpace=31:i32}>:()->()
        %y="tt.load"(%p):(!tt.ptr<i64>)->i64
        "tt.return"():()->()
      }):()->()
    }):()->()"""
    protected = """"builtin.module"() ({
      "tt.func"() <{sym_name="exchange",function_type=(!tt.ptr<i64>)->()}> ({
      ^bb0(%p:!tt.ptr<i64>):
        "scf.execute_region"() <{no_inline}> ({
          %x="arith.constant"() <{value=4:i64}> : ()->i64 loc("s1")
          "tt.store"(%p,%x):(!tt.ptr<i64>,i64)->() loc("s2")
          "scf.yield"():()->() loc("s3")
        }):()->() loc("s3")
        %y="tt.load"(%p):(!tt.ptr<i64>)->i64 loc("s4")
        "tt.return"():()->() loc("s5")
      }):()->() loc("s0")
    }):()->()"""
    scope = {
        "schema": "t2m.synchronous-program-barrier.v1",
        "source_id": "s3",
        "address_space_mask": 31,
        "address_spaces": ["local", "global_read", "global_write", "tensor_read", "tensor_write"],
        "participants": "complete-pre-layout-program; single-CTA",
        "ordering": "complete-synchronous-tile-operations-at-source-control-position",
        "external_participants": "not-admitted",
        "physical_mapping": "undecided",
    }
    records = [
        {"id": f"s{i}", "opcode": name}
        for i, name in enumerate(("tt.func", "arith.constant", "tt.store", "ttg.barrier", "tt.load", "tt.return"))
    ]
    records[3].update(properties={"addrSpace": "31 : i32"}, program_barrier=scope)
    graph = {"operations": records, "blocks": [], "attributes": "complete"}
    reports = [
        {
            "kind": "synchronous-program-memory-phase",
            "source_ids": ["s3"],
            "contract": scope,
            "phase_source_ids": ["s1", "s2"],
            "live_value_types": [],
            "representation": "scf.execute_region; full tensor versions or synchronous standard memory",
            "ownership": "unchanged source bounds/initialization/program-conflict gates",
        }
    ]
    return graph, reports, original.encode(), protected.encode()


def test_retained_phase_has_actual_original_control_and_store():
    checked_program_barriers(*fixture())


def test_deleting_the_inventory_and_profile_cannot_hide_a_real_source_barrier():
    graph, _, generic, protected = deepcopy(fixture())
    graph["operations"] = [row for row in graph["operations"] if row["opcode"] != "ttg.barrier"]
    with pytest.raises(ValueError, match="complete phase declarations"):
        checked_program_barriers(graph, None, generic, protected)


def test_reordering_intact_phases_cannot_change_original_memory_order():
    from io import StringIO

    from xdsl.parser import Parser
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import make_context

    graph, reports, generic, protected = deepcopy(fixture())
    generic = generic.replace(
        b'"tt.return"():()->()', b'"ttg.barrier"() <{addrSpace=31:i32}>:()->() "tt.return"():()->()'
    )
    protected = protected.replace(b'"tt.return"():()->() loc("s5")', b'"tt.return"():()->() loc("s6")')
    protected = protected.replace(
        b'%y="tt.load"(%p):(!tt.ptr<i64>)->i64 loc("s4")',
        b'"scf.execute_region"() <{no_inline}> ({'
        b'%y="tt.load"(%p):(!tt.ptr<i64>)->i64 loc("s4") '
        b'"scf.yield"():()->() loc("s5") }):()->() loc("s5")',
    )
    scope = {**deepcopy(reports[0]["contract"]), "source_id": "s5"}
    graph["operations"][5] = {
        "id": "s5",
        "opcode": "ttg.barrier",
        "properties": {"addrSpace": "31 : i32"},
        "program_barrier": scope,
    }
    graph["operations"].append({"id": "s6", "opcode": "tt.return"})
    reports.append({**deepcopy(reports[0]), "source_ids": ["s5"], "contract": scope, "phase_source_ids": ["s4"]})
    checked_program_barriers(graph, reports, generic, protected)
    module = Parser(make_context(), protected.decode()).parse_module()
    first, second = [op for op in module.walk() if op.name == "scf.execute_region"]
    block = first.parent_block()
    block.detach_op(second)
    block.insert_op_before(second, first)
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(module)
    with pytest.raises(ValueError, match="memory order"):
        checked_program_barriers(graph, reports, generic, stream.getvalue().encode())


@pytest.mark.parametrize(
    "change", ["missing", "mask", "prefix", "live", "scope", "no_inline", "lost_phase", "lost_store", "duplicate"]
)
def test_claims_and_tags_cannot_replace_source_scope_or_executable_payload(change):
    graph, reports, generic, protected = deepcopy(fixture())
    if change == "missing":
        reports = None
    elif change == "mask":
        reports[0]["contract"]["address_space_mask"] = 0
    elif change == "prefix":
        reports[0]["phase_source_ids"] = []
    elif change == "live":
        reports[0]["live_value_types"] = ["i64"]
    elif change == "scope":
        reports[0]["contract"]["participants"] = "all-programs"
    elif change == "no_inline":
        protected = protected.replace(b"<{no_inline}>", b"")
    elif change == "lost_phase":
        protected = protected.replace(b'"scf.execute_region"', b'"test.identity"')
    elif change == "lost_store":
        protected = protected.replace(
            b'"tt.store"(%p,%x):(!tt.ptr<i64>,i64)->() loc("s2")',
            b'%unused="arith.constant"() <{value=4:i64}>:()->i64 loc("s2")',
        )
    else:
        reports.append(deepcopy(reports[0]))
    with pytest.raises(ValueError, match="barrier"):
        checked_program_barriers(graph, reports, generic, protected)
