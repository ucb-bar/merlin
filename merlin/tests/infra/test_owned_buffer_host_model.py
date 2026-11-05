"""Explicit fresh tensor recurrence through the ordinary saved-model entry.

No provider or opaque calls occur in this fixture. Original alloc_tensor copy
semantics force a new buffer; admission must be established from the actual
post-sink result, rather than inferred from the source allocation operation.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import test_host_layout_model as layout
from safetensors.numpy import save_file
from test_dense_source_catalog import compilation_memory
from xdsl.dialects import arith, memref, scf

from merlin.common import compile_trace
from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.compile.baremetal_model import compile_saved_model
from merlin.llvmlower import toolchain
from merlin.llvmlower.pipeline import dealloc_placement_violations
from merlin.model_check import full_build as FB
from merlin.transforms.certificates import ir_text


def model_source(shape, dtype):
    m, n = shape
    dims = f"{m}x{n}"
    zero, add = ("0.0", "addf") if dtype == "f32" else ("0", "addi")
    return f"""module {{
 func.func @forward(%input:tensor<{dims}x{dtype}>,%initial:tensor<{dims}x{dtype}>,%position:tensor<i32>)
 ->(tensor<3x{dims}x{dtype}>,tensor<{dims}x{dtype}>,tensor<i32>,tensor<{dims}x{dtype}>,tensor<{dims}x{dtype}>) {{
 %zero=arith.constant {zero}:{dtype}
 %one=arith.constant 1:i32
 %c0=arith.constant 0:index
 %c1=arith.constant 1:index
 %c3=arith.constant 3:index
 %he=tensor.empty():tensor<3x{dims}x{dtype}>
 %hz=linalg.fill ins(%zero:{dtype}) outs(%he:tensor<3x{dims}x{dtype}>)->tensor<3x{dims}x{dtype}>
 %state,%history,%last=scf.for %step=%c0 to %c3 step %c1
 iter_args(%s=%initial,%h=%hz,%p=%position)->(tensor<{dims}x{dtype}>,tensor<3x{dims}x{dtype}>,tensor<i32>) {{
 %fresh=bufferization.alloc_tensor() copy(%s):tensor<{dims}x{dtype}>
 %next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}}
 ins(%input:tensor<{dims}x{dtype}>) outs(%fresh:tensor<{dims}x{dtype}>) {{
 ^bb0(%x:{dtype},%previous:{dtype}): %sum=arith.{add} %previous,%x:{dtype}
 linalg.yield %sum:{dtype}
 }}->tensor<{dims}x{dtype}>
 %h1="tensor.insert_slice"(%next,%h,%step) <{{operandSegmentSizes=array<i32:1,1,1,0,0>,
 static_offsets=array<i64:-9223372036854775808,0,0>,static_sizes=array<i64:1,{m},{n}>,
 static_strides=array<i64:1,1,1>}}>
 :(tensor<{dims}x{dtype}>,tensor<3x{dims}x{dtype}>,index)->tensor<3x{dims}x{dtype}>
 %pv=tensor.extract %p[]:tensor<i32>
 %pn=arith.addi %pv,%one:i32
 %p1=tensor.insert %pn into %p[]:tensor<i32>
 scf.yield %next,%h1,%p1:tensor<{dims}x{dtype}>,tensor<3x{dims}x{dtype}>,tensor<i32>
 }}
 return %history,%state,%last,%input,%initial:
 tensor<3x{dims}x{dtype}>,tensor<{dims}x{dtype}>,tensor<i32>,tensor<{dims}x{dtype}>,tensor<{dims}x{dtype}>
 }} }}"""


def capture(directory, shape, dtype):
    directory.mkdir()
    if dtype == "f32":
        data = np.resize(np.array([-2.25, -0.0, 0.0, 0.25, 7.5], "<f4"), shape)
        initial = np.resize(np.array([13.0, -0.0, 0.5, -4.25], "<f4"), shape)
    else:
        data = np.resize(np.array([-32768, -20001, -1, 0, 1, 20001, 32767], "<i2"), shape)
        initial = np.resize(np.array([32767, -32768, 655, -191], "<i2"), shape)
    state, history = initial.copy(), []
    for _ in range(3):
        state = (
            np.add(state, data, dtype=np.float32)
            if dtype == "f32"
            else ((state.astype(np.int64) + data.astype(np.int64) + 32768) % 65536 - 32768).astype("<i2")
        )
        history.append(state.copy())
    expected = (np.stack(history), state, np.array(10, "<i4"), data.copy(), initial.copy())
    save_file({}, directory / "weights.safetensors")
    names = ("input", "state", "position")
    (directory / "weights.safetensors.manifest.json").write_text(
        json.dumps({str(i): dict(kind="input", name=name) for i, name in enumerate(names)})
    )
    (directory / "input_order.json").write_text(json.dumps(dict(zip(names, range(3), strict=True))))
    np.savez(directory / "inputs.npz", in0=data, in1=initial, in2=np.array(7, "<i4"))
    np.save(directory / "golden.npy", expected[0])
    outputs = ("history", "state", "position", "input", "initial")
    np.savez(directory / "goldens.npz", **dict(zip(outputs, expected, strict=True)))
    (directory / "output_order.json").write_text(json.dumps(outputs))
    module = mq.parse(model_source(shape, dtype))
    module.verify()
    (directory / "model.mlir").write_text(ir_text(module))
    artifacts = {p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in directory.iterdir()}
    (directory / "capture_receipt.json").write_text(
        json.dumps(
            dict(
                schema="m2m.capture-receipt.v1",
                materialized_abi=dict(complete=True),
                artifacts=artifacts,
            )
        )
    )
    return expected


def observe_owned_loop(text):
    module = mq.parse(text)
    loops = [
        op
        for op in module.walk()
        if isinstance(op, scf.ForOp) and len(op.iter_args) == 2 and str(op.iter_args[-1].type) == "i1"
    ]
    assert len(loops) == 1, "fixture must retain fresh loop-carried ownership after upstream sinking"
    loop = loops[0]
    assert isinstance(loop.iter_args[0].owner, memref.AllocOp)
    assert isinstance(loop.iter_args[-1].owner, arith.ConstantOp) and loop.iter_args[-1].owner.value.value.data == 0
    yielded = loop.body.block.last_op
    assert isinstance(yielded, scf.YieldOp) and len(yielded.operands) == 2
    assert isinstance(yielded.operands[0].owner, memref.AllocOp)
    assert yielded.operands[0].owner.parent is loop.body.block
    assert isinstance(yielded.operands[-1].owner, arith.ConstantOp)
    assert yielded.operands[-1].owner.value.value.data == -1
    guarded = [
        op
        for op in loop.body.block.ops
        if isinstance(op, scf.IfOp) and tuple(op.operands) == (loop.body.block.args[-1],)
    ]
    assert len(guarded) == 1 and sum(isinstance(op, memref.DeallocOp) for op in guarded[0].walk()) == 1
    return module, loop


def generic_snapshot(source, destination):
    """Preserve actual native IR, changing printer syntax only for xDSL reading."""
    program = """import sys
from torch_mlir import ir
with ir.Context() as ctx:
 with open(sys.argv[1]) as f: module=ir.Module.parse(f.read(),ctx)
 module.operation.verify()
 with open(sys.argv[2],'x') as f: f.write(module.operation.get_asm(print_generic_op_form=True))
"""
    command = [str(toolchain.m2m_python()), "-B", "-c", program, str(source), str(destination)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return destination.read_text()


def run_arm(root, options, expected, *, checked):
    from merlin.llvmlower import dealloc_lifetime

    root.mkdir()
    trace_root = root / "lifetime_trace"
    started = time.perf_counter()
    with (
        compilation_memory() as memory,
        compile_trace.session(
            compile_trace.Request(str(trace_root), dump_after=("mlir:optimize-allocation-liveness",))
        ),
    ):
        receipt = compile_saved_model(**options, output=root / "model")
    elapsed = time.perf_counter() - started
    assert receipt["status"] == "verified_complete_output"
    assert [row["name"] for row in receipt["output"]["per_output"]] == [
        "history",
        "state",
        "position",
        "input",
        "initial",
    ]
    assert all(row["passed"] and row["mismatched_elements"] == 0 for row in receipt["output"]["per_output"])
    console = Path(receipt["output"]["console"])
    assert sha256_file(console) == receipt["output"]["console_sha256"]
    actual, metadata = FB.read(console.read_bytes(), options["capture"], receipt["output"]["build_hash"])
    assert len(actual) == len(expected) == 5
    for got, want in zip(actual, expected, strict=True):
        assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()
    trace = json.loads((trace_root / "trace.json").read_text())
    snapshots = []
    for event in trace["stages"]:
        if event["kind"] != "native" or event["stage"] != "mlir:optimize-allocation-liveness":
            continue
        for filename in event["files"]:
            path = trace_root / filename
            text = path.read_text()
            if "iter_args" not in text:
                continue
            generic = root / ("post-sink-" + str(len(snapshots)) + ".generic.mlir")
            module, loop = observe_owned_loop(generic_snapshot(path, generic))
            assert dealloc_placement_violations(text) == []
            entry = loop.iter_args[-1].owner
            entry.properties["value"] = loop.body.block.last_op.operands[-1].owner.properties["value"]
            module.verify()
            broken = root / "wrong-initial-owner.mlir"
            broken.write_text(ir_text(module))
            assert dealloc_placement_violations(broken.read_text())
            snapshots.append(
                dict(
                    path=str(path),
                    sha256=sha256_file(path),
                    generic_path=str(generic),
                    generic_sha256=sha256_file(generic),
                    wrong_initial_owner_sha256=sha256_file(broken),
                )
            )
    assert trace["outcome"] == compile_trace.COMPLETED and len(snapshots) == int(checked)
    scheduled = any("optimize-allocation-liveness" in row["pipeline"] for row in trace["native_segments"])
    assert scheduled == checked
    output = receipt["output"]
    for identity in output["host_artifacts"].values():
        assert sha256_file(Path(identity["path"])) == identity["sha256"]
    elf = Path(output["elf"])
    assert sha256_file(elf) == output["elf_sha256"]
    recipe_path = root / "model" / "build" / "lower" / "lowering_recipe.json"
    recipe = json.loads(recipe_path.read_text())
    checker = recipe["sources"]["loop_ownership_checker"]
    assert Path(checker["path"]) == Path(dealloc_lifetime.__file__).resolve()
    assert checker["sha256"] == sha256_file(Path(checker["path"]))
    for identity in recipe["sources"].values():
        assert sha256_file(Path(identity["path"])) == identity["sha256"]
    report = dict(
        snapshots=snapshots,
        sinking_scheduled=scheduled,
        source_sha256=sha256_file(options["capture"] / "model.mlir"),
        prepared_source=recipe["sources"]["prepared_mlir"],
        implementation_sources={
            name: recipe["sources"][name]
            for name in ("pipeline_driver", "loop_ownership_checker", "translation_transport")
        },
        lowering_recipe_sha256=sha256_file(recipe_path),
        compile_and_execute_seconds=elapsed,
        compilation_memory=memory,
        host_artifacts=output["host_artifacts"],
        elf_sha256=output["elf_sha256"],
        elf_bytes=elf.stat().st_size,
        elf_sections=FB.elf_sections(elf),
        functional_forward_instructions=metadata["metrics"]["cycles"],
        complete_output_judgments=output["per_output"],
        scope="fresh carried-buffer ownership and all five complete outputs; no arena/peak-memory or timing claim",
    )
    (root / "lifetime_model.json").write_text(json.dumps(report, indent=2))
    return report, actual


@pytest.mark.parametrize("shape,dtype", [((2, 19), "f32"), ((3, 7), "i16")])
def test_ordinary_host_model_executes_fresh_carried_storage(tmp_path, monkeypatch, record_property, shape, dtype):
    monkeypatch.setattr(layout, "capture", lambda directory, shape, **_: capture(directory, shape, dtype))
    expected, root, options = layout.setup(tmp_path, monkeypatch, shape)
    capture_files = {path.name: sha256_file(path) for path in options["capture"].iterdir()}
    baseline_controls = dict(
        MERLIN_SINK_DEALLOCS=None, MERLIN_SPLIT_FRESH_DEALLOCS="0", MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT="0"
    )
    controls = (
        ("baseline", baseline_controls),
        ("checked", {**baseline_controls, "MERLIN_SINK_DEALLOCS": "1"}),
        ("default", dict.fromkeys(baseline_controls)),
    )
    arms, outputs = [], []
    for name, settings in controls:
        with monkeypatch.context() as arm:
            for key, value in settings.items():
                if value is None:
                    arm.delenv(key, raising=False)
                else:
                    arm.setenv(key, value)
            report, actual = run_arm(root / name, options, expected, checked=name != "baseline")
        assert {path.name: sha256_file(path) for path in options["capture"].iterdir()} == capture_files
        report.update(arm=name, controls=settings)
        (root / name / "lifetime_model.json").write_text(json.dumps(report, indent=2))
        arms.append(report)
        outputs.append(actual)
    before = arms[0]
    for after, actual in zip(arms[1:], outputs[1:], strict=True):
        assert after["source_sha256"] == before["source_sha256"]
        assert after["prepared_source"]["sha256"] == before["prepared_source"]["sha256"]
        assert after["implementation_sources"] == before["implementation_sources"]
        for got, baseline in zip(actual, outputs[0], strict=True):
            assert got.dtype == baseline.dtype and got.shape == baseline.shape and got.tobytes() == baseline.tobytes()
    report = dict(
        dtype=dtype,
        shape=shape,
        captured_artifacts_sha256=capture_files,
        arms=arms,
        numerical_policy="ordered f32 addition or wrapping i16 addition; all complete outputs exact",
        comparison_scope=(
            "baseline/checked isolate sinking with split/slice disabled in both; default restores all defaults"
        ),
        timing_scope="compile+execute wall and functional Spike instructions; no hardware latency or speedup claim",
    )
    (root / "lifetime_model.json").write_text(json.dumps(report, indent=2))
    record_property("owned_buffer_host_model", json.dumps(report, sort_keys=True))
