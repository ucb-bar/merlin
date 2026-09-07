"""Control-flow task receipts must not hide skipped tasks or invalid SSA."""
from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from merlin.perf.task_cfg_evidence import analyze_task_cfg
from xdsl.dialects.llvm import LLVM


def _check(body, tasks):
    context = make_context()
    context.load_dialect(LLVM)
    module = parse_mlir_text('builtin.module { llvm.func @kernel(%p: !llvm.ptr, %c: i1) {'
                             + body + '} }', context)
    return analyze_task_cfg(next(iter(module.body.block.ops)), tasks)


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
