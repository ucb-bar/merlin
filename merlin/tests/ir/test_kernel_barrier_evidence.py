"""Independent source phase checks require real control and memory payload."""

from copy import deepcopy

import pytest

from merlin.frontends.kernel_barrier_evidence import checked_program_barriers


def cfg_function():
    from merlin.frontends.kernel_barrier_evidence import _name
    from merlin.frontends.linalg_mlir import make_context
    from xdsl.parser import Parser

    module = Parser(make_context(), '''"builtin.module"() ({
      "tt.func"() <{sym_name="f",function_type=(i1,i64)->()}> ({
      ^entry(%choose:i1,%x:i64):
        cf.cond_br %choose, ^body(%x:i64), ^done
      ^body(%carry:i64):
        "ttg.barrier"() <{addrSpace=31:i32}>:()->()
        cf.br ^done
      ^done:
        "tt.return"():()->()
      }):()->()
    }):()->()''').parse_module()
    return next(op for op in module.walk() if _name(op) == "tt.func")


def test_scalar_cfg_phase_participation_is_independently_reconstructed():
    from merlin.frontends.kernel_barrier_evidence import _checked_cfg_participation

    assert _checked_cfg_participation(cfg_function()) == "whole-program-scalar-cfg-v1"


@pytest.mark.parametrize("damage", ["condition", "argument", "missing-argument", "foreign-block", "terminator"])
def test_scalar_cfg_phase_scope_cannot_hide_invalid_control(damage):
    from merlin.frontends.kernel_barrier_evidence import _checked_cfg_participation
    from xdsl.dialects import builtin
    from xdsl.ir import Block

    function = cfg_function()
    blocks = list(function.regions[0].blocks)
    branch = blocks[0].last_op
    if damage == "condition":
        branch.operands[0]._type = builtin.TensorType(builtin.i1, [1])
    elif damage == "argument":
        blocks[1].args[0]._type = builtin.i32
    elif damage == "missing-argument":
        blocks[1].insert_arg(builtin.i64, 1)
    elif damage == "foreign-block":
        branch.successors = (Block(arg_types=[builtin.i64]), branch.successors[1])
    else:
        blocks[1].erase_op(blocks[1].last_op)
    with pytest.raises(ValueError, match="CFG"):
        _checked_cfg_participation(function)


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


def atomic_fixture():
    graph, reports, generic, protected = deepcopy(fixture())
    original = b'"tt.store"(%p,%x):(!tt.ptr<i64>,i64)->()'
    atomic = (b'%old="tt.atomic_rmw"(%p,%x) <{atomic_rmw_op=4:i32,sem=4:i32,scope=1:i32}>'
              b':(!tt.ptr<i64>,i64)->i64')
    generic, protected = generic.replace(original, atomic), protected.replace(original, atomic)
    scope = reports[0]["contract"]
    scope.update(schema="t2m.synchronous-program-barrier.v2", atomic_effects={
        "profile": "private-invocation-integer-atomics-v1", "f32_atomic_add": None})
    graph["operations"][3]["program_barrier"] = scope
    graph["operations"][2].update(opcode="tt.atomic_rmw", atomic={
        "operation": "add", "standard_kind": 1, "word_bits": 64,
        "source_order": "acq_rel", "source_scope": "gpu",
        "logical_scope": "all-logical-programs-in-private-invocation",
        "standard_order": "acq_rel", "external_participants": "excluded-by-private-buffer-ABI"})
    return graph, reports, generic, protected


def test_atomic_source_phases_keep_actual_primitive_fields_and_event_order():
    checked_program_barriers(*atomic_fixture())


def volatile_fixture():
    graph, reports, generic, protected = deepcopy(fixture())
    generic = generic.replace(b'"tt.load"(%p)', b'"tt.load"(%p) <{isVolatile=true}>')
    protected = protected.replace(b'"tt.load"(%p)', b'"tt.load"(%p) <{isVolatile=true}>')
    scope = reports[0]["contract"]
    scope.update(schema="t2m.synchronous-program-barrier.v3", volatile_effects={
        "profile": "private-invocation-volatile-loads-v1"})
    graph["operations"][3]["program_barrier"] = scope
    graph["operations"][4]["volatile_read"] = {
        "schema": "t2m.volatile-read.v1", "source_id": "s4", "result_type": "i64",
        "scope": "private-invocation-volatile-loads-v1",
        "observation": "one-actual-volatile-read-per-active-logical-lane",
        "external_participants": "excluded-by-private-buffer-ABI", "physical_mapping": "undecided"}
    return graph, reports, generic, protected


def test_volatile_source_phases_retain_the_actual_unused_observation():
    checked_program_barriers(*volatile_fixture())


def combined_fixture():
    graph, reports, generic, protected = atomic_fixture()
    generic = generic.replace(b'"tt.load"(%p)', b'"tt.load"(%p) <{isVolatile=true}>')
    protected = protected.replace(b'"tt.load"(%p)', b'"tt.load"(%p) <{isVolatile=true}>')
    scope = reports[0]["contract"]
    scope["schema"] = "t2m.synchronous-program-barrier.v4"
    scope["atomic_effects"]["profile"] = "private-invocation-typed-atomics-v2"
    scope["volatile_effects"] = {"profile": "private-invocation-volatile-loads-v1"}
    graph["operations"][4]["volatile_read"] = volatile_fixture()[0]["operations"][4]["volatile_read"]
    return graph, reports, generic, protected


def test_combined_source_phases_retain_both_primitive_effect_contracts():
    checked_program_barriers(*combined_fixture())


@pytest.mark.parametrize("damage", ["atomic", "volatile", "downgrade", "integer-only", "kind", "flag", "primitive"])
def test_one_effect_contract_cannot_replace_the_other_in_a_combined_phase(damage):
    graph, reports, generic, protected = combined_fixture()
    scope = reports[0]["contract"]
    if damage in ("atomic", "volatile"):
        scope.pop(damage + "_effects")
    elif damage == "downgrade":
        scope["schema"] = "t2m.synchronous-program-barrier.v2"
    elif damage == "integer-only":
        scope["atomic_effects"]["profile"] = "private-invocation-integer-atomics-v1"
    elif damage == "kind":
        protected = protected.replace(b"atomic_rmw_op=4:i32", b"atomic_rmw_op=10:i32")
    elif damage == "flag":
        protected = protected.replace(b"isVolatile=true", b"isVolatile=false")
    else:
        graph["operations"][4]["volatile_read"]["observation"] = "stable-tensor-read"
    with pytest.raises(ValueError, match="barrier|volatile|atomic"):
        checked_program_barriers(graph, reports, generic, protected)


@pytest.mark.parametrize("change", ["missing-selection", "downgrade", "scope", "primitive", "flag", "read"])
def test_volatile_phase_claims_cannot_replace_original_source_observations(change):
    graph, reports, generic, protected = volatile_fixture()
    if change == "missing-selection":
        reports[0]["contract"].pop("volatile_effects")
    elif change == "downgrade":
        reports[0]["contract"]["schema"] = "t2m.synchronous-program-barrier.v1"
    elif change == "scope":
        reports[0]["contract"]["volatile_effects"]["profile"] = "external-volatile-loads"
    elif change == "primitive":
        graph["operations"][4]["volatile_read"]["observation"] = "stable-tensor-read"
    elif change == "flag":
        protected = protected.replace(b"isVolatile=true", b"isVolatile=false")
    else:
        protected = protected.replace(b'"tt.load"', b'"test.identity"')
    with pytest.raises(ValueError, match="barrier|volatile"):
        checked_program_barriers(graph, reports, generic, protected)


def test_ordinary_program_phases_cannot_gain_an_unselected_volatile_read():
    graph, reports, generic, protected = fixture()
    protected = protected.replace(b'"tt.load"(%p)', b'"tt.load"(%p) <{isVolatile=true}>')
    with pytest.raises(ValueError, match="invented a volatile"):
        checked_program_barriers(graph, reports, generic, protected)


@pytest.mark.parametrize("change", ["missing-selection", "downgrade", "primitive", "protected-kind", "lost-atomic"])
def test_atomic_phase_annotations_cannot_replace_actual_atomic_source(change):
    graph, reports, generic, protected = atomic_fixture()
    if change == "missing-selection":
        reports[0]["contract"].pop("atomic_effects")
    elif change == "downgrade":
        reports[0]["contract"]["schema"] = "t2m.synchronous-program-barrier.v1"
    elif change == "primitive":
        graph["operations"][2]["atomic"]["source_order"] = "relaxed"
    elif change == "protected-kind":
        protected = protected.replace(b"atomic_rmw_op=4:i32", b"atomic_rmw_op=10:i32")
    else:
        protected = protected.replace(b'"tt.atomic_rmw"', b'"test.identity"')
    with pytest.raises(ValueError, match="barrier"):
        checked_program_barriers(graph, reports, generic, protected)


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
