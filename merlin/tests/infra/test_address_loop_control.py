"""Address-only loop counters may exceed output budgets without computing tensor values."""

import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.parser import Parser

from merlin.perf.host_cfg_activity import analyze_host_cfg_activity
from merlin.verify import host_compute_audit as audit


def emitted_loop(*, trips=7, step=1, post_tested=True, escape=None):
    extra = {
        None: "",
        "scalar_command": (
            '"llvm.inline_asm"(%i) <{asm_string = "", constraints = "r", has_side_effects}> : (i64) -> ()'
        ),
        "stored_counter": "llvm.store %i, %base : i64, !llvm.ptr",
        "stored_address": "llvm.store %address, %base : i64, !llvm.ptr",
        "floating_address_value": '''%scalar = llvm.bitcast %address : i64 to f64
          "llvm.inline_asm"(%scalar) <{asm_string = "", constraints = "f", has_side_effects}> : (f64) -> ()''',
        "tensor_compute": """%value = llvm.load %pointer : !llvm.ptr -> i64
          %computed = llvm.add %value, %one : i64
          llvm.store %computed, %pointer : i64, !llvm.ptr""",
    }[escape]
    commands = f"""%pointer = llvm.getelementptr %base[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %address = llvm.ptrtoint %pointer : !llvm.ptr to i64
        "llvm.inline_asm"(%address) <{{asm_string = "", constraints = "r", has_side_effects}}> : (i64) -> ()
        {extra}
        %next = llvm.add %i, %step : i64"""
    if post_tested:
        loop = f"""^loop(%i: i64):
          {commands}
          %test = llvm.icmp "slt" %next, %bound : i64
          llvm.cond_br %test, ^loop(%next : i64), ^exit"""
    else:
        loop = f"""^loop(%i: i64):
          %test = llvm.icmp "slt" %i, %bound : i64
          llvm.cond_br %test, ^body, ^exit
        ^body:
          {commands}
          llvm.br ^loop(%next : i64)"""
    text = f"""module {{
      llvm.func @entry(%base: !llvm.ptr) {{
        %zero = llvm.mlir.constant(0 : i64) : i64
        %one = llvm.mlir.constant(1 : i64) : i64
        %step = llvm.mlir.constant({step} : i64) : i64
        %bound = llvm.mlir.constant({trips * step} : i64) : i64
        llvm.br ^loop(%zero : i64)
        {loop}
        ^exit:
          llvm.return
      }}
    }}"""
    context = Context()
    for dialect in (builtin.Builtin, llvm.LLVM):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return next(op for op in module.ops if isinstance(op, llvm.FuncOp))


@pytest.mark.parametrize("post_tested", [False, True])
@pytest.mark.parametrize("trips,step", [(3, 1), (7, 16)])
def test_proven_address_only_loop_control_is_separate_from_payload(post_tested, trips, step):
    function = emitted_loop(trips=trips, step=step, post_tested=post_tested)
    activity = analyze_host_cfg_activity(function)
    assert activity["status"] == "derived", activity["problems"]
    assert activity["dynamic_operations"]["integer_arithmetic"] == trips
    assert activity["dynamic_address_loop_control_arithmetic"] == trips
    result = audit.audit_group(audit.GroupSite(0, "accelerator", "entry", 1), function)
    assert result["verdict"] == audit.CLEAN, result
    assert result["host_total_arithmetic"] == result["host_address_loop_control_arithmetic"] == trips
    assert result["host_arithmetic"] == result["host_payload_bytes"] == 0


@pytest.mark.parametrize("escape", ["scalar_command", "stored_counter", "stored_address", "floating_address_value"])
def test_address_or_induction_scalar_escape_does_not_receive_exemption(escape):
    function = emitted_loop(escape=escape)
    activity = analyze_host_cfg_activity(function)
    assert activity["dynamic_address_loop_control_arithmetic"] == 0
    result = audit.audit_group(audit.GroupSite(0, "accelerator", "entry", 1), function)
    assert result["verdict"] == audit.HOST_COMPUTE, result


def test_proving_address_control_does_not_exempt_real_tensor_computation():
    function = emitted_loop(escape="tensor_compute")
    activity = analyze_host_cfg_activity(function)
    assert activity["dynamic_address_loop_control_arithmetic"] == 7
    result = audit.audit_group(audit.GroupSite(0, "accelerator", "entry", 1), function)
    assert result["verdict"] == audit.HOST_COMPUTE, result
    assert result["host_arithmetic"] == 7
    assert result["host_payload_bytes"] == 112
