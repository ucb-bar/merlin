"""Neutral actual LLVM effect projection and ordinary native execution."""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import fields, replace

import numpy as np
import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.parser import Parser

from merlin.compile.scheduling import staged_dense as S
from merlin.llvmlower import target_staged_dense as T
from merlin.llvmlower.static_llvm_cfg import StaticInt, StaticPointer, trace_static_function

TYPES = (S.OperandLoad, S.AccumulatorZero, S.DescriptorIssue, S.ComputeConsume, S.AccumulatorReadout, S.CompletionFence)
HEADER = "llvm.func @event(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64)"


def contract():
    return S.StagedDenseContract(
        "neutral-rtl",
        "neutral-effects",
        "exact-i8-i32",
        "neutral-runtime",
        1,
        2,
        1,
        8,
        16,
        32,
        "read_at_compute_consume",
        "data_commands_preserve_pending_descriptor",
        "raw_accumulator_identity",
        True,
        True,
        True,
    )


def parse(text):
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(llvm.LLVM)
    result = Parser(ctx, text).parse_module()
    result.verify()
    return result


def values(op):
    result = []
    for f in fields(op):
        value = getattr(op, f.name)
        if f.name == "role":
            value = {"lhs": 0, "rhs": 1}[value]
        if type(value) is bool:
            value = int(value)
        result.append(value)
    assert len(result) <= 6
    return [TYPES.index(type(op)), *result, *([0] * (6 - len(result)))]


def operations():
    result = []
    for m in range(2):
        result.extend(
            (
                S.AccumulatorZero(0, 1, 1),
                S.OperandLoad("lhs", m, 0, 0, 1, 1),
                S.OperandLoad("rhs", 0, 0, 1, 1, 1),
                S.DescriptorIssue(1, 0, 1, 1, 1, True),
                S.CompletionFence(),
                S.ComputeConsume(0, 1, 1),
                S.CompletionFence(),
                S.AccumulatorReadout(0, m, 0, 1, 1),
                S.CompletionFence(),
            )
        )
    return tuple(result)


def module(ops=None):
    lines = [
        "builtin.module {",
        HEADER,
        "llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) {",
        "llvm.br ^work",
        "^work:",
    ]
    for i, op in enumerate(operations() if ops is None else ops):
        args = []
        for j, v in enumerate(values(op)):
            name = f"%v{i}_{j}"
            lines.append(f"{name} = llvm.mlir.constant({v} : i64) : i64")
            args.append(name)
        lines.append(
            "llvm.call @event(%a, %b, %c, " + ", ".join(args) + ") : "
            "(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> ()"
        )
    return parse("\n".join([*lines, "llvm.return", "}", "}"]))


class Owner:
    def __init__(self, mode=None):
        self.mode = mode
        self.calls = 0

    def project(self, actual):
        self.calls += 1
        functions = list(actual.body.block.ops)
        if actual.attributes or actual.properties or len(functions) != 2:
            return None
        if any(type(f) is not llvm.FuncOp for f in functions):
            return None
        if [f.sym_name.data for f in functions] != ["event", "entry"]:
            return None
        fn = functions[1]
        if len(fn.body.blocks.first.args) != 3:
            return None
        try:
            steps = tuple(
                trace_static_function(
                    fn,
                    tuple(StaticPointer(n, StaticInt(0, 64)) for n in "ABC"),
                    observe=lambda op: type(op) is llvm.CallOp,
                    pointer_index_bits=64,
                    max_steps=4096,
                )
            )
        except ValueError:
            return None
        result = []
        for step in steps:
            if step.operation.callee.root_reference.data != "event" or len(step.inputs) != 10:
                return None
            if tuple(step.inputs[:3]) != tuple(StaticPointer(n, StaticInt(0, 64)) for n in "ABC"):
                return None
            data = [v.signed for v in step.inputs[3:] if type(v) is StaticInt]
            if len(data) != 7 or not 0 <= data[0] < len(TYPES):
                return None
            cls = TYPES[data[0]]
            count = len(fields(cls))
            args = data[1 : 1 + count]
            if any(data[1 + count :]):
                return None
            if cls is S.OperandLoad:
                if args[0] not in (0, 1):
                    return None
                args[0] = ("lhs", "rhs")[args[0]]
            if cls in (S.DescriptorIssue, S.ComputeConsume):
                if args[-1] not in (0, 1):
                    return None
                args[-1] = bool(args[-1])
            result.append(cls(*args))
        return tuple(result)

    def emit(self, source, before, after):
        if self.mode == "ignore":
            return source.clone()
        if self.mode == "mutate_source":
            source.attributes["bad"] = builtin.StringAttr("changed")
        result = module(after)
        if self.mode == "wrong_pointer":
            function = list(result.body.block.ops)[1]
            first = next(op for op in function.walk() if type(op) is llvm.CallOp)
            operands = list(first.operands)
            operands[1] = function.body.blocks.first.args[0]
            first.operands = operands
        if self.mode == "missing_wait":
            result = module(tuple(op for op in after if type(op) is not S.CompletionFence))
        return result


def run(source=None, owner=None, **kw):
    owner = owner or Owner()
    return T.rewrite_target_staged_dense(
        module() if source is None else source,
        problem=S.StagedDenseProblem(2, 1, 1),
        contract=contract(),
        adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
        rewrites=(S.StagedDenseRewrite("live_rhs", S.eliminate_redundant_rhs_loads),),
        **kw,
    )


def test_actual_cfg_projection_rewrite_and_reprojection():
    source = module()
    text = str(source)
    result = run(source)
    assert result.report["status"] == "checked" and str(source) == text
    assert len(result.source_operations) == 18 and len(result.selected_operations) == 17
    assert sum(type(op) is S.OperandLoad and op.role == "rhs" for op in result.selected_operations) == 1
    result.revalidate()
    assert result.report["emission"]["emitted"]["outputs"] == 2


@pytest.mark.parametrize("mode", ["ignore", "wrong_pointer", "missing_wait", "mutate_source"])
def test_actual_ignored_wrong_pointer_wait_or_source_mutation_refuses(mode):
    with pytest.raises(ValueError):
        run(owner=Owner(mode))


@pytest.mark.parametrize("permission", [False, True])
def test_unknown_or_unqualified_source_keeps_exact_original(permission):
    owner = Owner()
    source = module()
    source.attributes["unknown"] = builtin.StringAttr("opaque")
    result = T.rewrite_target_staged_dense(
        source,
        problem=S.StagedDenseProblem(2, 1, 1),
        contract=contract(),
        adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=permission),
    )
    assert result.report["status"] == "fallback"
    assert result.module.is_structurally_equivalent(source)
    assert owner.calls == (1 if permission else 0)


@pytest.mark.parametrize("ceiling", ["max_operations", "max_ir_bytes", "max_commands", "max_products"])
def test_actual_work_bounds_refuse(ceiling):
    with pytest.raises(ValueError):
        run(**{ceiling: 1})


def test_late_source_selected_report_and_guard_mutations_refuse():
    for kind in ("source", "selected", "report", "guard"):
        result = run()
        if kind == "source":
            result.source.attributes["late"] = builtin.StringAttr("x")
        if kind == "selected":
            result.module.attributes["late"] = builtin.StringAttr("x")
        if kind == "report":
            result.report["status"] = "forged"
        if kind == "guard":
            object.__setattr__(result, "_check", lambda: None)
        with pytest.raises(ValueError):
            result.revalidate()


def test_contract_callback_code_and_helpers_frozen_before_first_callback(monkeypatch):
    class Mutator(Owner):
        def project(self, actual):
            monkeypatch.setattr(S, "eliminate_redundant_rhs_loads", lambda *a: a[0])
            return super().project(actual)

    with pytest.raises(ValueError, match="binding"):
        run(owner=Mutator())


def test_emitter_must_not_change_original_contract():
    selected = contract()

    class Mutator(Owner):
        def emit(self, source, before, after):
            object.__setattr__(selected, "rhs_immutable", False)
            return super().emit(source, before, after)

    owner = Mutator()
    with pytest.raises(ValueError, match="contract"):
        T.rewrite_target_staged_dense(
            module(),
            problem=S.StagedDenseProblem(2, 1, 1),
            contract=selected,
            adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
        )


def test_result_clone_unissued_and_independent_emission_check():
    result = run()
    with pytest.raises(ValueError, match="unissued"):
        replace(result).revalidate()
    owner = Owner()
    with pytest.raises(ValueError, match="sequence"):
        T.verify_target_staged_dense_emission(
            result.source,
            result.source.clone(),
            problem=S.StagedDenseProblem(2, 1, 1),
            contract=contract(),
            adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
            selected_operations=result.selected_operations,
        )


@pytest.mark.parametrize("damage", ["same_text_clone", "reader", "checker", "defaults"])
def test_first_project_cannot_replace_source_or_original_code(monkeypatch, damage):
    class Mutator(Owner):
        def project(self, actual):
            if damage == "same_text_clone":
                before = str(actual)
                fn = actual.body.block.last_op
                clone = fn.clone()
                for old, new in zip(fn.body.blocks, clone.body.blocks, strict=True):
                    new.name_hint = old.name_hint
                fn.detach()
                actual.body.block.add_op(clone)
                assert str(actual) == before
            elif damage == "reader":
                monkeypatch.setattr(T, "_module_state", lambda *a: ())
            elif damage == "checker":
                monkeypatch.setattr(T.verifier, "verify_staged_dense", lambda *a, **kw: {})
            elif damage == "defaults":
                monkeypatch.setattr(Owner.emit, "__defaults__", (None,))
            return super().project(actual)

    with pytest.raises(ValueError):
        run(owner=Mutator())


@pytest.mark.parametrize("damage", ["drop_output", "wrong_source"])
def test_independent_semantics_refuses_bad_original_rewrite(damage):
    def bad(ops, _problem, _contract):
        if damage == "drop_output":
            return tuple(op for op in ops if type(op) is not S.AccumulatorReadout)
        return tuple(replace(op, source_row=1) if type(op) is S.OperandLoad and op.role == "rhs" else op for op in ops)

    owner = Owner()
    with pytest.raises(ValueError):
        T.rewrite_target_staged_dense(
            module(),
            problem=S.StagedDenseProblem(2, 1, 1),
            contract=contract(),
            adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
            rewrites=(S.StagedDenseRewrite("bad", bad),),
        )


def test_native_actual_llvm_outputs_state_readonly_and_real_rhs_counts(tmp_path):
    translate = os.environ.get("MLIR_TRANSLATE") or shutil.which("mlir-translate")
    clang = os.environ.get("MERLIN_CLANG") or shutil.which("clang-23")
    if not translate or not clang:
        pytest.skip("explicit LLVM native tools unavailable")
    result = run()
    c = tmp_path / "events.c"
    c.write_text("""#include <stdint.h>
static int32_t cells[2], acc; static int rhs,pending;
long counts[2];
void event(int8_t*a,int8_t*b,int32_t*c,long tag,long x,long y,long z,long u,long v,long w){
 (void)w;
 switch(tag){
 case 0:cells[u]=x?b[y]:a[y];counts[x]++;break;
 case 1:acc=0;break;
 case 2:rhs=x;pending=1;break;
 case 3:if(!pending)__builtin_trap();acc+=cells[x]*cells[rhs];pending=0;break;
 case 4:c[y]=acc;break;
 case 5:break;
 default:__builtin_trap();}
}
""")
    all_counts = []
    arms = []
    for arm, actual in [("original", result.source), ("selected", result.module)]:
        mlir = tmp_path / f"{arm}.mlir"
        ll = tmp_path / f"{arm}.ll"
        so = tmp_path / f"{arm}.so"
        mlir.write_text(str(actual))
        subprocess.run(
            [translate, "--mlir-to-llvmir", str(mlir), "-o", str(ll)], check=True, capture_output=True, timeout=30
        )
        subprocess.run(
            [clang, "-shared", "-fPIC", "-O2", str(ll), str(c), "-o", str(so)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        lib = ctypes.CDLL(str(so))
        entry = lib.entry
        entry.argtypes = [ctypes.c_void_p] * 3
        entry.restype = None
        state = 0
        history = []
        epochs = []
        for epoch in range(3):
            prior_state = state
            a = np.array([19, -2 + state % 3, 1, 23], dtype=np.int8)
            b = np.array([17, 2, 29], dtype=np.int8)
            output = np.array([123456, -99, -99, 654321], dtype=np.int32)
            before = (a.copy(), b.copy())
            expected = a[1:3].astype(np.int64) * int(b[1])
            entry(a.ctypes.data + 1, b.ctypes.data + 1, output.ctypes.data + 4)
            np.testing.assert_array_equal(output[1:3], expected)
            np.testing.assert_array_equal(a, before[0])
            np.testing.assert_array_equal(b, before[1])
            assert list(output[[0, 3]]) == [123456, 654321]
            state = int(output[1]) + int(output[2]) + epoch
            history.append((output.copy(), state))
            arrays = tmp_path / f"{arm}-epoch-{epoch}.npz"
            np.savez(
                arrays,
                A_before=before[0],
                B_before=before[1],
                A_after=a,
                B_after=b,
                output_and_guards=output,
                prior_state=np.array(prior_state),
                state=np.array(state),
            )
            epochs.append(str(arrays))
        all_counts.append(tuple((ctypes.c_long * 2).in_dll(lib, "counts")))
        arms.append(dict(arm=arm, epochs=epochs, actual_load_counts=all_counts[-1]))
        if arm == "original":
            baseline = history
        else:
            for old, new in zip(baseline, history, strict=True):
                np.testing.assert_array_equal(old[0], new[0])
                assert old[1] == new[1]
    assert all_counts == [(6, 6), (6, 3)]
    result.revalidate()
    pins = [
        dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
        for p in sorted(tmp_path.iterdir())
        if p.is_file()
    ]
    (tmp_path / "pair.json").write_text(
        json.dumps(
            dict(
                arms=arms,
                pins=pins,
                scope="Neutral ordinary native actual-operand execution only; no target qualification or timing.",
            ),
            indent=2,
        )
        + "\n"
    )
