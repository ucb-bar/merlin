"""Executable schedule.plan controls using existing independent neutral fixtures.

These exercise ordinary consumers and emitted LLVM control, not target numeric
qualification or the missing historical QK source closure.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path


def _reject(action):
    try:
        action()
    except ValueError as error:
        return {"status": "rejected", "reason": str(error)}
    raise AssertionError("invalid schedule was accepted")


def _llvm(result, output):
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp, i64
    from xdsl.printer import Printer

    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
    from merlin.llvmlower.host_builder import FnBuilder

    builder, names = FnBuilder([]), {}
    control = FnBuilderIndexAdapter(builder)

    def effect(tag, args):
        names[tag] = len(args)
        builder.add(llvm.CallOp(tag, *(control.value(value) for value in args.values())))

    receipt = result.emit_index_program(control, effect, complete_epoch=lambda: effect("complete_epoch", {}))
    module = ModuleOp([
        *(llvm.FuncOp(name, llvm.LLVMFunctionType([i64] * count, llvm.LLVMVoidType()))
          for name, count in names.items()),
        llvm.FuncOp("checked_schedule", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish()),
    ])
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    output.write_text(stream.getvalue() + "\n")
    return receipt


def run_schedule_checks(output, *, repo=None):
    from merlin.common.paths import repo_root
    from merlin.compile.scheduling.compact_dense import emit_compact_dense
    from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_operand_hoists
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.dense_segmented_pipeline import verify_dense_segmented_pipeline
    from merlin.compile.scheduling.pointwise_convolution import pointwise_dense_view
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )
    from merlin.compile.scheduling.window_transfer import Window2D
    from merlin.mining.reconstruction_reference import dense_source, dense_source_region
    from merlin.sched.ir import Const, Loop

    repo, output = Path(repo or repo_root()).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from mechanisms import pin, write

    # Reuse the repository's independent arithmetic/effect fixtures verbatim.
    fixtures = repo / "merlin/tests/infra"
    sys.path.insert(0, str(fixtures))
    compact = importlib.import_module("test_compact_dense")
    retained = importlib.import_module("test_dense_retained_reduction")
    segmented = importlib.import_module("test_dense_segmented_pipeline")
    arithmetic = importlib.import_module("test_dense_ready_readout")
    shared = importlib.import_module("test_dense_shared_rhs")
    rows = []
    for family in ("pointwise", "projection"):
        case = output / family
        case.mkdir()
        p, r, s, b, admission = retained.inputs() if family == "pointwise" else segmented.inputs()
        source = dense_source({"m": p.m, "k": p.k, "n": p.n, "dtypes": ["i8", "i8", "i32"],
                               "numeric": {"accumulator_dtype": "i32", "bias": "none", "scale": 1,
                                           "activation": "none", "rounding": "none", "saturation": "none"}})
        source.verify()
        # This existing structural recognizer independently checks the full SSA.
        dense_source_region(source)
        source_text = str(source)
        (case / "source.mlir").write_text(source_text + "\n")
        identity = pin(case / "source.mlir")["sha256"]
        order, transforms = ("m", "n"), ()
        if family == "pointwise":
            view = pointwise_dense_view(Window2D((1, 1, p.m, p.k), (1, 1), (1, 1)),
                                        p.n, output_shape=(p.m, p.n))
            assert view.dimensions == (p.m, p.k, p.n)
            order = ("n", "m")
            raw = emit_compact_dense(p, b, s, compact.ADAPTER, name=family,
                                     dtypes=("i8", "i8", "i32"), outer_order=order)
            transforms = dense_retained_operand_hoists(raw, compact.ADAPTER, role="weight")
        plan = make_dense_schedule_plan(
            p, r, s, compact.ADAPTER, source_identity=identity, dtypes=("i8", "i8", "i32"),
            static_budget=128, block_tiles=(b.m, b.k, b.n), outer_order=order,
            loop_transforms=transforms, pipeline_kind="retained_full_k" if family == "pointwise"
            else "retained_lhs_segmented_k", pipeline_prefetch=family != "pointwise", name=family)
        text = dense_schedule_text(plan)
        parsed = parse_dense_schedule_plan(text)

        def consume(selected, source_id=identity, storage=s):
            return schedule_dense_pipeline(
                p, r, storage, compact.ADAPTER, name=family, dtypes=("i8", "i8", "i32"),
                source_identity=source_id, external_buffers_disjoint=True, schedule_plan=selected,
                pipeline_admission=admission, static_budget=128, timeout_ms=5000)

        first, replay = consume(plan), consume(parsed)
        assert first.kernel.digest() == replay.kernel.digest()
        arithmetic.execute(replay.kernel, p)
        replay.materialize_scf().verify()
        negatives = {"source_binding": _reject(lambda: consume(parsed, source_id="0" * 64)),
                     "storage_binding": _reject(lambda: consume(parsed, storage=replace(
                         s, rhs=replace(s.rhs, backing=s.lhs.backing, base=s.lhs.base))))}
        if family == "projection":
            def change(body):
                return tuple(replace(item, body=change(item.body)) if type(item) is Loop else
                             replace(item, args=tuple((name, Const(0) if name == "depth" else value)
                                     for name, value in item.args)) if item.instr == "compute" else item
                             for item in body)
            corrupt = replace(replay.kernel, body=change(replay.kernel.body))
            negatives["reduction_coverage"] = _reject(lambda: verify_dense_segmented_pipeline(
                corrupt, p, b, r, s, compact.ADAPTER, admission, static_budget=128, timeout_ms=5000))
        (case / "schedule.mlir").write_text(text)
        emission = _llvm(replay, case / "emitted.llvm.mlir")
        rows.append({"family": family, "source": pin(case / "source.mlir"), "source_identity": identity,
                     "schedule": pin(case / "schedule.mlir"), "kernel_sha256": replay.kernel.digest(),
                     "roundtrip_equal": True, "numeric": "pass", "checked_elements": p.m*p.n,
                     "emission": emission, "llvm": pin(case / "emitted.llvm.mlir"), "counterexamples": negatives})

    for quantized in (False, True):
        case = output / ("shared_rhs_i8" if quantized else "shared_rhs_i32")
        case.mkdir()
        model = shared.fixture(quantized=quantized)
        result = shared.schedule(model)
        shared.run(result, model)
        text = result.plan_text
        replay = shared.schedule(model, schedule_plan=parse_dense_schedule_plan(text))
        assert replay.kernel.digest() == result.kernel.digest()
        (case / "schedule.mlir").write_text(text)
        negatives = {"source_binding": _reject(lambda: shared.schedule(
            model, schedule_plan=parse_dense_schedule_plan(text.replace("a" * 64, "b" * 64)))),
            "complete_liveouts": _reject(lambda: replace(result, kernel=replace(
                result.kernel, args=result.kernel.args[:-1])).revalidate())}
        fallback = shared.schedule(model, asynchronous=False)
        shared.run(fallback, model)
        assert fallback.realization["dynamic_epoch_drains"] > result.realization["dynamic_epoch_drains"]
        emission = _llvm(replay, case / "emitted.llvm.mlir")
        rows.append({"family": case.name, "fixture": pin(fixtures / "test_dense_shared_rhs.py"),
                     "source": "existing explicitly neutral grouped effect/numerical fixture",
                     "schedule": pin(case / "schedule.mlir"), "kernel_sha256": replay.kernel.digest(),
                     "roundtrip_equal": True, "numeric": "pass", "emission": emission,
                     "llvm": pin(case / "emitted.llvm.mlir"), "counterexamples": negatives,
                     "unknown_ordering_preserves_drains": True})
    report = {"schema": "kernel_abstractions.schedule_checks.v1", "status": "pass", "rows": rows,
              "fixture_owners": [pin(fixtures / (name + ".py")) for name in (
                  "test_compact_dense", "test_dense_retained_reduction", "test_dense_segmented_pipeline",
                  "test_dense_ready_readout", "test_dense_shared_rhs")],
              "spatial": {"status": "representation_gap", "existing": "checked typed scheduling and rolling SCF",
                          "missing": "schedule.plan spatial decision consumer"},
              "qualification": "Neutral library/LLVM-control and fixture numerics, no target execution, "
                               "expert semantic equality, target-law admission or timing claim."}
    write(output / "index.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run_schedule_checks(Path(sys.argv[1])), indent=2))
