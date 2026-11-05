"""Unavoidable selected work includes independently proved positive constant loops."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.parser import Parser

from merlin.targetgen.native_dispatch_accounting import _mandatory_command_blocks
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.targetgen.owned_host_model_execution import _mandatory_source_commands


def _loop(bound="4", *, guarded=False, nested=False):
    constant = "" if bound == "%n" else f"%bound = llvm.mlir.constant({bound} : i64) : i64"
    limit = "%n" if bound == "%n" else "%bound"
    work = """%a = llvm.load %p : !llvm.ptr -> i64
              %b = llvm.load %p : !llvm.ptr -> i64"""
    body = f"{work}\nllvm.br ^latch"
    if guarded:
        body = f"llvm.cond_br %flag, ^work, ^latch\n^work:\n{work}\nllvm.br ^latch"
    if nested:
        body = f"""llvm.br ^inner(%zero : i64)
        ^inner(%j: i64):
          %again = llvm.icmp \"ult\" %j, %three : i64
          llvm.cond_br %again, ^work, ^latch
        ^work:
          {work}
          %next_j = llvm.add %j, %one : i64
          llvm.br ^inner(%next_j : i64)"""
    source = f"""builtin.module {{
      llvm.func @kernel(%p: !llvm.ptr, %n: i64, %flag: i1) {{
        %zero = llvm.mlir.constant(0 : i64) : i64
        %one = llvm.mlir.constant(1 : i64) : i64
        %three = llvm.mlir.constant(3 : i64) : i64
        {constant}
        llvm.br ^header(%zero : i64)
      ^header(%i: i64):
        %test = llvm.icmp \"ult\" %i, {limit} : i64
        llvm.cond_br %test, ^body, ^exit
      ^body:
        {body}
      ^latch:
        %next_i = llvm.add %i, %one : i64
        llvm.br ^header(%next_i : i64)
      ^exit:
        llvm.return
      }}
    }}"""
    module = Parser(make_llvm_context(), source).parse_module()
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    commands = [op for op in function.walk() if op.name == "llvm.load"]
    return function, commands


@pytest.mark.parametrize("bound,nested", [("1", False), ("4", False), ("2", True)])
def test_selected_work_in_proven_positive_loop_is_unavoidable(bound, nested):
    function, commands = _loop(bound, nested=nested)
    assert _mandatory_command_blocks(function, set(commands))


@pytest.mark.parametrize("bound,guarded", [("0", False), ("%n", False), ("4", True)])
def test_zero_dynamic_or_conditionally_bypassed_work_remains_refused(bound, guarded):
    function, commands = _loop(bound, guarded=guarded)
    assert not _mandatory_command_blocks(function, set(commands))


def test_positive_loop_cannot_supply_missing_or_foreign_matched_command():
    function, commands = _loop()
    _, foreign = _loop()
    assert not _mandatory_command_blocks(function, set())
    assert not _mandatory_command_blocks(function, set(foreign))
    assert _mandatory_command_blocks(function, {commands[0]})


def test_positive_inner_loop_cannot_overcome_zero_trip_outer_loop():
    function, commands = _loop("0", nested=True)
    assert not _mandatory_command_blocks(function, set(commands))


def test_unreachable_selected_work_is_not_credited():
    module = Parser(
        make_llvm_context(),
        """builtin.module {
          llvm.func @kernel(%p: !llvm.ptr) {
            llvm.return
          ^dead:
            %a = llvm.load %p : !llvm.ptr -> i64
            llvm.return
          }
        }""",
    ).parse_module()
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    command = next(op for op in function.walk() if op.name == "llvm.load")
    assert not _mandatory_command_blocks(function, {command})


def test_positive_loop_preserves_each_fused_source_attribution():
    function, commands = _loop()
    obligations = [
        {"source_op_index": index, "region_id": f"r{index}", "semantic_family": "contraction"} for index in range(2)
    ]
    commands[0].attributes["prov.region_id"] = StringAttr("r0")
    with pytest.raises(ValueError, match="operation 1"):
        _mandatory_source_commands(function, {"contraction": set(commands)}, obligations)
    commands[1].attributes["prov.region_id"] = StringAttr("r1")
    assert len(_mandatory_source_commands(function, {"contraction": set(commands)}, obligations)) == 2
    obligations[1]["semantic_family"] = "movement"
    with pytest.raises(ValueError, match="movement"):
        _mandatory_source_commands(function, {"contraction": set(commands)}, obligations)


def test_unknown_or_incomplete_cfg_accounting_cannot_authorize_work(monkeypatch):
    from merlin.perf import host_cfg_activity

    function, commands = _loop()
    for result in ({"status": "UNKNOWN", "blocks": []}, {"status": "derived", "blocks": []}):
        monkeypatch.setattr(host_cfg_activity, "analyze_host_cfg_activity", lambda *args, **kwargs: result)
        assert not _mandatory_command_blocks(function, set(commands))
