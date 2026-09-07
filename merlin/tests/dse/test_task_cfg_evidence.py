"""Control-flow task receipts must not hide skipped tasks or invalid SSA."""
from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from merlin.perf.host_cfg_index import prepare_host_cfg
from merlin.perf.task_cfg_evidence import analyze_task_cfg
import pytest
from xdsl.dialects.llvm import LLVM


def _check(body, tasks, *, globals_=""):
    context = make_context()
    context.load_dialect(LLVM)
    module = parse_mlir_text('builtin.module {' + globals_
                             + ' llvm.func @kernel(%p: !llvm.ptr, %c: i1) {'
                             + body + '} }', context)
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    return analyze_task_cfg(function, tasks)


def test_prepared_cfg_is_equivalent_and_function_identity_bound():
    context = make_context()
    context.load_dialect(LLVM)
    module = parse_mlir_text('''builtin.module {
      llvm.func @kernel(%p: !llvm.ptr) {
        %v = llvm.load %p {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
        llvm.store %v, %p {merlin.global_task = 0 : i64} : i64, !llvm.ptr
        llvm.return
      }
    }''', context)
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    prepared = prepare_host_cfg(function)
    assert analyze_task_cfg(function, [0], prepared_cfg=prepared) == analyze_task_cfg(function, [0])

    other = parse_mlir_text(str(module), context)
    other_function = next(op for op in other.body.block.ops if op.name == "llvm.func")
    with pytest.raises(ValueError, match="different function object"):
        analyze_task_cfg(other_function, [0], prepared_cfg=prepared)


def test_unreachable_cfg_keeps_legacy_refusal_evidence():
    context = make_context()
    context.load_dialect(LLVM)
    module = parse_mlir_text('''builtin.module {
      llvm.func @kernel(%p: !llvm.ptr) {
        %v = llvm.load %p {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
        llvm.return
      ^dead:
        llvm.return
      }
    }''', context)
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    prepared = prepare_host_cfg(function)

    assert analyze_task_cfg(
        function, [0], prepared_cfg=prepared) == analyze_task_cfg(function, [0])
    assert "owned kernel contains unreachable blocks" in analyze_task_cfg(
        function, [0], prepared_cfg=prepared)["problems"]


def test_task_present_only_on_one_returning_path_is_not_mandatory():
    result = _check('''
      llvm.cond_br %c, ^work, ^exit {merlin.global_task = 0 : i64}
    ^work:
      %v = llvm.load %p {merlin.global_task = 1 : i64} : !llvm.ptr -> i64
      llvm.br ^exit {merlin.global_task = 1 : i64}
    ^exit:
      llvm.return
    ''', [0, 1])
    assert result["status"] == "refused"
    assert "a returning CFG path bypasses an entire planned task" in result["problems"]


def test_definition_in_optional_branch_does_not_dominate_merge():
    result = _check('''
      llvm.cond_br %c, ^work, ^exit {merlin.global_task = 0 : i64}
    ^work:
      %v = llvm.load %p {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
      llvm.br ^exit {merlin.global_task = 0 : i64}
    ^exit:
      llvm.store %v, %p {merlin.global_task = 0 : i64} : i64, !llvm.ptr
      llvm.return
    ''', [0])
    assert result["status"] == "refused"
    assert "kernel SSA definition does not dominate its use" in result["problems"]


def test_load_is_not_permitted_as_unowned_prologue_plumbing():
    result = _check('''
      %v = llvm.load %p : !llvm.ptr -> i64
      llvm.store %v, %p {merlin.global_task = 0 : i64} : i64, !llvm.ptr
      llvm.return
    ''', [0])
    assert result["status"] == "refused"
    assert "kernel operation lacks task ownership: llvm.load" in result["problems"]


def test_entry_symbol_address_is_shared_prologue_plumbing():
    result = _check('''
      %g = "llvm.mlir.addressof"() <{global_name = @weights}> : () -> !llvm.ptr
      %v = llvm.load %g {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
      llvm.store %v, %p {merlin.global_task = 0 : i64} : i64, !llvm.ptr
      llvm.return
    ''', [0], globals_='''
      "llvm.mlir.global"() <{global_type = i64, sym_name = "weights",
          linkage = #llvm.linkage<"internal">, addr_space = 0 : i32, constant,
          value = 0 : i64}> ({ llvm.return }) : () -> ()
    ''')
    assert result["status"] == "verified"
    assert result["shared_prologue_operations"] == {"llvm.mlir.addressof": 1}


def test_mutable_global_address_is_not_shared_prologue_plumbing():
    result = _check('''
      %g = "llvm.mlir.addressof"() <{global_name = @state}> : () -> !llvm.ptr
      %v = llvm.load %g {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
      llvm.store %v, %p {merlin.global_task = 0 : i64} : i64, !llvm.ptr
      llvm.return
    ''', [0], globals_='''
      "llvm.mlir.global"() <{global_type = i64, sym_name = "state",
          linkage = #llvm.linkage<"internal">, addr_space = 0 : i32,
          value = 0 : i64}> ({ llvm.return }) : () -> ()
    ''')
    assert result["status"] == "refused"
    assert "kernel operation lacks task ownership: llvm.mlir.addressof" in result["problems"]


def test_non_entry_symbol_address_still_requires_task_ownership():
    result = _check('''
      llvm.br ^work {merlin.global_task = 0 : i64}
    ^work:
      %g = "llvm.mlir.addressof"() <{global_name = @weights}> : () -> !llvm.ptr
      %v = llvm.load %g {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
      llvm.store %v, %p {merlin.global_task = 0 : i64} : i64, !llvm.ptr
      llvm.return
    ''', [0], globals_='''
      "llvm.mlir.global"() <{global_type = i64, sym_name = "weights",
          linkage = #llvm.linkage<"internal">, addr_space = 0 : i32, constant,
          value = 0 : i64}> ({ llvm.return }) : () -> ()
    ''')
    assert result["status"] == "refused"
    assert "kernel operation lacks task ownership: llvm.mlir.addressof" in result["problems"]


def test_gep_from_shared_symbol_address_still_requires_task_ownership():
    result = _check('''
      %g = "llvm.mlir.addressof"() <{global_name = @weights}> : () -> !llvm.ptr
      %i = llvm.mlir.constant(0 : i64) : i64
      %slot = llvm.getelementptr %g[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %v = llvm.load %slot {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
      llvm.store %v, %p {merlin.global_task = 0 : i64} : i64, !llvm.ptr
      llvm.return
    ''', [0], globals_='''
      "llvm.mlir.global"() <{global_type = i64, sym_name = "weights",
          linkage = #llvm.linkage<"internal">, addr_space = 0 : i32, constant,
          value = 0 : i64}> ({ llvm.return }) : () -> ()
    ''')
    assert result["status"] == "refused"
    assert "kernel operation lacks task ownership: llvm.getelementptr" in result["problems"]


def test_store_through_shared_symbol_address_still_requires_task_ownership():
    result = _check('''
      %g = "llvm.mlir.addressof"() <{global_name = @weights}> : () -> !llvm.ptr
      %v = llvm.load %g {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
      llvm.store %v, %p : i64, !llvm.ptr
      llvm.return
    ''', [0], globals_='''
      "llvm.mlir.global"() <{global_type = i64, sym_name = "weights",
          linkage = #llvm.linkage<"internal">, addr_space = 0 : i32, constant,
          value = 0 : i64}> ({ llvm.return }) : () -> ()
    ''')
    assert result["status"] == "refused"
    assert "kernel operation lacks task ownership: llvm.store" in result["problems"]
