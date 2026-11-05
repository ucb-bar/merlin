"""Ordinary recurrent host models prepack exact immutable weight layout chains.

Both arms use one capture, compiler/library, numerical policy and native toolchain.
Actual lowered IR, runtime blob/ABI and complete outputs are reopened. Functional
Spike instruction counts and logical materializations are not accelerator latency.
"""

import importlib
import itertools
import json
import time
from math import prod
from pathlib import Path

import numpy as np
import pytest
import test_host_layout_model as layout
from safetensors.numpy import save_file
from test_dense_source_catalog import compilation_memory
from test_owned_buffer_host_model import generic_snapshot
from xdsl.dialects import func
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir.affine import AffineDimExpr, AffineMap

from merlin.capture.safetensors import load_header
from merlin.common import compile_trace
from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.compile.baremetal_model import compile_saved_model
from merlin.llvmlower.arena_bind import _call_target, _defines, _malloc_size
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike_model
from merlin.transforms.certificates import ir_text

CASES = (((2, 3, 2, 5), (0, 2, 3, 1), "i8"), ((3, 2, 7), (2, 0, 1), "i16"))
NAMES = ("history", "state", "position", "input", "initial", "initial_position")


def model_source(shape, permutation, dtype):
    count, columns = prod(shape), 3
    rows = count // columns
    stored = "x".join(map(str, shape))
    moved = "x".join(str(shape[i]) for i in permutation)
    matrix = f"{rows}x{columns}"
    return f"""module {{
 func.func @forward(%weight:tensor<{stored}x{dtype}>,%input:tensor<{matrix}x{dtype}>,
 %initial:tensor<{matrix}x{dtype}>,%position:tensor<i32>)
 ->(tensor<3x{matrix}x{dtype}>,tensor<{matrix}x{dtype}>,tensor<i32>,tensor<{matrix}x{dtype}>,
 tensor<{matrix}x{dtype}>,tensor<i32>) {{
 %init=tensor.empty():tensor<{moved}x{dtype}>
 %reordered=linalg.transpose ins(%weight:tensor<{stored}x{dtype}>)
 outs(%init:tensor<{moved}x{dtype}>) permutation={list(permutation)}
 %flat=tensor.collapse_shape %reordered {[list(range(len(shape)))]}
 :tensor<{moved}x{dtype}> into tensor<{count}x{dtype}>
 %expanded=tensor.expand_shape %flat [[0,1]] output_shape [{columns},{rows}]
 :tensor<{count}x{dtype}> into tensor<{columns}x{rows}x{dtype}>
 %last_init=tensor.empty():tensor<{matrix}x{dtype}>
 %w=linalg.transpose ins(%expanded:tensor<{columns}x{rows}x{dtype}>)
 outs(%last_init:tensor<{matrix}x{dtype}>) permutation=[1,0]
 %zero=arith.constant 0:{dtype}
 %one=arith.constant 1:i32
 %c0=arith.constant 0:index
 %c1=arith.constant 1:index
 %c3=arith.constant 3:index
 %he=tensor.empty():tensor<3x{matrix}x{dtype}>
 %hz=linalg.fill ins(%zero:{dtype}) outs(%he:tensor<3x{matrix}x{dtype}>)->tensor<3x{matrix}x{dtype}>
 %state,%history,%last=scf.for %step=%c0 to %c3 step %c1
 iter_args(%s=%initial,%h=%hz,%p=%position)
 ->(tensor<{matrix}x{dtype}>,tensor<3x{matrix}x{dtype}>,tensor<i32>) {{
 %fresh=tensor.empty():tensor<{matrix}x{dtype}>
 %next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%w,%input,%s:tensor<{matrix}x{dtype}>,tensor<{matrix}x{dtype}>,tensor<{matrix}x{dtype}>)
 outs(%fresh:tensor<{matrix}x{dtype}>) {{
 ^bb0(%weight_scalar:{dtype},%x:{dtype},%previous:{dtype},%unused:{dtype}):
 %product=arith.muli %weight_scalar,%x:{dtype}
 %sum=arith.addi %previous,%product:{dtype}
 linalg.yield %sum:{dtype}
 }}->tensor<{matrix}x{dtype}>
 %h1="tensor.insert_slice"(%next,%h,%step) <{{operandSegmentSizes=array<i32:1,1,1,0,0>,
 static_offsets=array<i64:-9223372036854775808,0,0>,static_sizes=array<i64:1,{rows},{columns}>,
 static_strides=array<i64:1,1,1>}}>
 :(tensor<{matrix}x{dtype}>,tensor<3x{matrix}x{dtype}>,index)->tensor<3x{matrix}x{dtype}>
 %pv=tensor.extract %p[]:tensor<i32>
 %pn=arith.addi %pv,%one:i32
 %p1=tensor.insert %pn into %p[]:tensor<i32>
 scf.yield %next,%h1,%p1:tensor<{matrix}x{dtype}>,tensor<3x{matrix}x{dtype}>,tensor<i32>
 }}
 return %history,%state,%last,%input,%initial,%position:
 tensor<3x{matrix}x{dtype}>,tensor<{matrix}x{dtype}>,tensor<i32>,tensor<{matrix}x{dtype}>,
 tensor<{matrix}x{dtype}>,tensor<i32>
 }} }}"""


def independent_weight_records(raw, shape, permutation):
    """Map coordinates to original byte records without NumPy transpose/reshape."""
    width = len(raw) // prod(shape)
    moved_shape = [shape[i] for i in permutation]
    records = []
    for output in itertools.product(*(range(d) for d in moved_shape)):
        original = [0] * len(shape)
        for axis, source_axis in enumerate(permutation):
            original[source_axis] = output[axis]
        flat = 0
        for coordinate, dimension in zip(original, shape, strict=True):
            flat = flat * dimension + coordinate
        records.append(raw[flat * width : (flat + 1) * width])
    # The two static reshapes preserve linear element order. The final transpose
    # reads a [3, count/3] matrix by columns.
    rows = len(records) // 3
    return b"".join(records[column * rows + row] for row in range(rows) for column in range(3))


def wrap(value, bits):
    return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def implementation_pins():
    modules = (
        "merlin.capture.rewrite",
        "merlin.xdsl_dialects.lowering.weight_layout",
        "merlin.llvmlower.weight_prepack",
        "merlin.llvmlower.weight_prepack_runtime",
        "merlin.llvmlower.prepared_operand_effects",
        "merlin.llvmlower.pipeline",
        "merlin.llvmlower.c_runtime",
        "merlin.compile.baremetal_model",
    )
    return {name: sha256_file(Path(importlib.import_module(name).__file__)) for name in modules}


def seal_capture(directory):
    artifacts = {
        p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p))
        for p in directory.iterdir()
        if p.name != "capture_receipt.json"
    }
    (directory / "capture_receipt.json").write_text(
        json.dumps(
            dict(
                schema="m2m.capture-receipt.v1",
                materialized_abi=dict(complete=True),
                artifacts=artifacts,
            )
        )
    )


def capture(directory, case):
    directory.mkdir()
    shape, permutation, dtype = case
    bits = int(dtype[1:])
    numeric = "<i1" if bits == 8 else "<i2"
    count, rows = prod(shape), prod(shape) // 3
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    stored = np.resize(np.array([lo, hi, -17, -3, -1, 0, 1, 7, 23], dtype=numeric), count).reshape(shape)
    packed = independent_weight_records(stored.tobytes(), shape, permutation)
    weights = [
        int.from_bytes(packed[i : i + bits // 8], "little", signed=True) for i in range(0, len(packed), bits // 8)
    ]
    data = np.resize(np.array([lo, hi, -5, -1, 0, 1, 3, 13], dtype=numeric), (rows, 3))
    initial = np.resize(np.array([hi, lo, 17, -29, 0], dtype=numeric), (rows, 3))
    state, history = initial.flatten().tolist(), []
    for _ in range(3):
        state = [
            wrap(previous + wrap(weight * int(x), bits), bits)
            for previous, weight, x in zip(state, weights, data.flat, strict=True)
        ]
        history.append(np.array(state, dtype=numeric).reshape(rows, 3))
    expected = (
        np.stack(history),
        history[-1].copy(),
        np.array(10, "<i4"),
        data.copy(),
        initial.copy(),
        np.array(7, "<i4"),
    )
    save_file(dict(weight=stored), directory / "weights.safetensors")
    (directory / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": dict(kind="buffer", weight="weight", dtype="int" + str(bits), shape=list(shape)),
                **{str(i): dict(kind="input", name=name) for i, name in enumerate(("input", "state", "position"), 1)},
            }
        )
    )
    np.savez(directory / "inputs.npz", in0=data, in1=initial, in2=np.array(7, "<i4"))
    (directory / "input_order.json").write_text('{"input":0,"state":1,"position":2}')
    np.save(directory / "golden.npy", expected[0])
    np.savez(directory / "goldens.npz", **dict(zip(NAMES, expected, strict=True)))
    (directory / "output_order.json").write_text(json.dumps(NAMES))
    module = mq.parse(model_source(shape, permutation, dtype))
    module.attributes["prov.weights_file"] = StringAttr(str((directory / "weights.safetensors").resolve()))
    module.verify()
    (directory / "model.mlir").write_text(ir_text(module))
    seal_capture(directory)
    return expected


def setup(tmp_path, monkeypatch, case):
    monkeypatch.setattr(layout, "capture", lambda directory, _shape, **_: capture(directory, case))
    return layout.setup(tmp_path, monkeypatch, case[0])


def movement_regions(module):
    """Record actual scalar copies, separating permutations from fills/projections."""
    result = []
    for index, op in enumerate(module.walk()):
        if op.name == "memref.copy":
            output = op.operands[1].type
            result.append(
                dict(
                    operation_index=index,
                    category="memref_copy",
                    input_type=str(op.operands[0].type),
                    output_type=str(output),
                    indexing_maps=[],
                    logical_static_output_bytes=prod(output.get_shape()) * output.element_type.bitwidth // 8,
                    count_scope="one actual static IR operation; no dynamic or bus-traffic claim",
                )
            )
            continue
        if not isinstance(op, GenericOp) or len(op.inputs) != 1 or len(op.outputs) != 1:
            continue
        body = list(op.body.block.ops)
        maps = op.indexing_maps.data
        if len(body) == 1 and body[0].name == "linalg.yield" and tuple(body[0].operands) == (op.body.block.args[0],):
            full_rank = all(
                not item.data.num_symbols
                and len(item.data.results) == item.data.num_dims
                and all(isinstance(expression, AffineDimExpr) for expression in item.data.results)
                and sorted(expression.position for expression in item.data.results) == list(range(item.data.num_dims))
                for item in maps
            )
            kind = (
                "permutation"
                if full_rank and any(item.data != AffineMap.identity(item.data.num_dims) for item in maps)
                else "identity_copy"
                if full_rank
                else "scalar_broadcast"
                if len(maps[0].data.results) == 0
                else "other_indexed_copy"
            )
            output = op.outputs[0].type
            result.append(
                dict(
                    operation_index=index,
                    category=kind,
                    input_type=str(op.inputs[0].type),
                    output_type=str(output),
                    indexing_maps=[str(item) for item in maps],
                    logical_static_output_bytes=prod(output.get_shape()) * output.element_type.bitwidth // 8,
                    count_scope="one actual static IR operation; no dynamic or bus-traffic claim",
                )
            )
    return result


def materializations(trace_root, root):
    trace = json.loads((trace_root / "trace.json").read_text())
    assert trace["outcome"] == compile_trace.COMPLETED
    upstream, buffered, sunk = [], [], []
    for event in trace["stages"]:
        files = event.get("files", []) or ([event["file"]] if "file" in event else [])
        for filename in files:
            path = trace_root / filename
            if event["kind"] == "ir" and event["stage"] == "upstream":
                module = mq.parse(path.read_text())
                module.verify()
                upstream.append(
                    dict(
                        path=str(path),
                        sha256=sha256_file(path),
                        transposes=sum(op.name == "linalg.transpose" for op in module.walk()),
                    )
                )
            elif event["kind"] == "native" and event["stage"] in (
                "mlir:one-shot-bufferize",
                "mlir:optimize-allocation-liveness",
            ):
                generic = root / (event["stage"].split(":")[1] + "-" + str(len(buffered) + len(sunk)) + ".mlir")
                module = mq.parse(generic_snapshot(path, generic))
                movement = movement_regions(module)
                regions = [region for region in movement if region["category"] == "permutation"]
                record = dict(
                    path=str(path),
                    sha256=sha256_file(path),
                    generic_sha256=sha256_file(generic),
                    permutation_regions=len(regions),
                    logical_materialized_output_bytes=sum(region["logical_static_output_bytes"] for region in regions),
                    scalar_movement_regions=movement,
                    allocations=sum(op.name == "memref.alloc" for op in module.walk()),
                )
                (buffered if event["stage"] == "mlir:one-shot-bufferize" else sunk).append(record)
    assert len(upstream) == len(buffered) == len(sunk) == 1
    return dict(
        upstream=upstream[0],
        bufferized=buffered[0],
        post_sink=sunk[0],
        trace_sha256=sha256_file(trace_root / "trace.json"),
    )


def runtime_weight_identity(build, capture_path, prepared, expected_bytes):
    generated = build / "cgen" / "model_gen.h"
    assert generated.is_file()
    # Generated ABI rows are original compiler output, independently reopened.
    lines = generated.read_text().splitlines()
    rows = [line.strip() for line in lines if line.strip().startswith("{MERLIN_WEIGHT,")]
    assert len(rows) == 1
    prefix, _, tail = rows[0].removeprefix("{").partition("{")
    dimensions, _, tail = tail.partition("}")
    fields = prefix.rstrip(", ").split(",")
    offset, rank = int(fields[1].strip().removesuffix("L")), int(fields[2])
    dimensions = [int(value) for value in dimensions.split(",")]
    blob = build / "cgen" / "weights.bin"
    assert blob.read_bytes()[offset : offset + len(expected_bytes)] == expected_bytes
    header, payload = load_header(prepared / "weights.safetensors")
    module = mq.parse((prepared / "model.mlir").read_text())
    assert module.attributes["prov.weights_file"].data == str((prepared / "weights.safetensors").resolve())
    begin, end = header["weight"]["data_offsets"]
    assert (prepared / "weights.safetensors").read_bytes()[payload + begin : payload + end] == expected_bytes
    assert rank == len(header["weight"]["shape"]) and dimensions == header["weight"]["shape"]
    return dict(
        header_sha256=sha256_file(generated),
        runtime_blob_sha256=sha256_file(blob),
        offset=offset,
        rank=rank,
        prepared_weight_header=header["weight"],
        original_capture_sha256=sha256_file(capture_path / "weights.safetensors"),
    )


def run_arm(root, options, expected, *, case, prepack, monkeypatch):
    from merlin.llvmlower import weight_prepack

    root.mkdir()
    pins = implementation_pins()
    observed = {}
    original_build, original_run = spike_model.build, spike_model.run_raw
    original_prepare = weight_prepack.prepare_build_bundle

    def build(*args, **kwargs):
        started = time.perf_counter()
        with compilation_memory() as memory:
            result = original_build(*args, **kwargs)
        observed.update(compile_seconds=time.perf_counter() - started, compilation_memory=memory)
        return result

    def run(*args, **kwargs):
        started = time.perf_counter()
        result = original_run(*args, **kwargs)
        observed["simulator_process_wall_seconds"] = time.perf_counter() - started
        return result

    def prepare(*args, **kwargs):
        started = time.perf_counter()
        result = original_prepare(*args, **kwargs)
        observed["weight_preparation_seconds"] = time.perf_counter() - started
        return result

    trace_root = root / "trace"
    with (
        monkeypatch.context() as arm,
        compile_trace.session(
            compile_trace.Request(
                str(trace_root),
                dump_after=("upstream", "mlir:one-shot-bufferize", "mlir:optimize-allocation-liveness"),
            )
        ),
    ):
        arm.setattr(spike_model, "build", build)
        arm.setattr(spike_model, "run_raw", run)
        arm.setattr(weight_prepack, "prepare_build_bundle", prepare)
        receipt = compile_saved_model(**options, output=root / "model", prepack_weights=prepack)
    assert receipt["status"] == "verified_complete_output"
    assert [item["name"] for item in receipt["output"]["per_output"]] == list(NAMES)
    assert all(item["passed"] and item["mismatched_elements"] == 0 for item in receipt["output"]["per_output"])
    console = Path(receipt["output"]["console"])
    assert sha256_file(console) == receipt["output"]["console_sha256"]
    actual, metadata = FB.read(console.read_bytes(), options["capture"], receipt["output"]["build_hash"])
    assert len(actual) == len(expected) == 6
    for got, want in zip(actual, expected, strict=True):
        assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()
    build_root = root / "model" / "build"
    preparation_path = build_root / "bundle_preparation.json"
    if prepack:
        preparation = json.loads(preparation_path.read_text())
        assert preparation["schema"] == "merlin.bundle-preparation.v2"
        assert preparation["source_bundle"] == str(options["capture"].resolve())
        assert preparation["immutable_arguments"] == [0]
        assert preparation["consumers"] == ["model_lowering", "runtime_abi_and_weights"]
        prepared = Path(preparation["prepared_bundle"])
        assert preparation["identity_sha256"] == sha256_file(prepared / "weight-prepack-identity.json")
        identity = json.loads((prepared / "weight-prepack-identity.json").read_text())
        for item in (
            *identity["inputs"]["files"].values(),
            *identity["inputs"]["implementations"],
            *identity["outputs"].values(),
        ):
            assert sha256_file(Path(item["path"])) == item["sha256"]
    else:
        assert not preparation_path.exists()
        preparation, prepared = None, options["capture"]
    weight_header, offset = load_header(options["capture"] / "weights.safetensors")
    begin, end = weight_header["weight"]["data_offsets"]
    original = (options["capture"] / "weights.safetensors").read_bytes()[offset + begin : offset + end]
    selected_bytes = independent_weight_records(original, case[0], case[1]) if prepack else original
    weights = runtime_weight_identity(build_root, options["capture"], prepared, selected_bytes)
    artifacts = receipt["output"]["host_artifacts"]
    for item in artifacts.values():
        assert sha256_file(Path(item["path"])) == item["sha256"]
    llvm = Path(artifacts["llvm_ir"]["path"])
    lines = llvm.read_text().splitlines()
    allocations = [
        dict(function=name, bytes=_malloc_size(line.strip()))
        for name, start, end in _defines(lines)
        for line in lines[start : end + 1]
        if _call_target(line.strip()) == "malloc"
    ]
    elf = Path(receipt["output"]["elf"])
    assert sha256_file(elf) == receipt["output"]["elf_sha256"]
    recipe_path = build_root / "lower" / "lowering_recipe.json"
    recipe = json.loads(recipe_path.read_text())
    assert recipe["sources"]["prepared_mlir"]["sha256"] == sha256_file(Path(recipe["sources"]["prepared_mlir"]["path"]))
    compilation_path = build_root / "compilation_recipe.json"
    compilation = json.loads(compilation_path.read_text())
    assert compilation["status"] == "completed"
    if prepack:
        assert compilation["preparation"]["weight_prepack"]["sha256"] == sha256_file(preparation_path)
        for identity in compilation["preparation"].values():
            assert sha256_file(Path(identity["path"])) == identity["sha256"]
    assert pins == implementation_pins()
    report = dict(
        **observed,
        implementation_sources_sha256=pins,
        prepack_weights=prepack,
        capture_source_sha256=sha256_file(options["capture"] / "model.mlir"),
        preparation=preparation,
        runtime_weight_identity=weights,
        offline_relayout_output_bytes=len(original) if prepack else 0,
        materializations=materializations(trace_root, root),
        llvm_allocations=allocations,
        host_artifacts=artifacts,
        lowering_recipe_sha256=sha256_file(recipe_path),
        compilation_recipe_sha256=sha256_file(compilation_path),
        elf_bytes=elf.stat().st_size,
        elf_sections=FB.elf_sections(elf),
        elf_sha256=sha256_file(elf),
        functional_forward_instructions=metadata["metrics"]["cycles"],
        complete_outputs=receipt["output"]["per_output"],
        movement_scope="actual lowered transpose output bytes, not measured bus traffic or latency",
    )
    (root / "weight_model.json").write_text(json.dumps(report, indent=2))
    return report, actual


@pytest.mark.parametrize("case", CASES)
def test_ordinary_saved_model_preapplies_immutable_weight_chain(tmp_path, monkeypatch, record_property, case):
    expected, root, options = setup(tmp_path, monkeypatch, case)
    capture_hashes = {path.name: sha256_file(path) for path in options["capture"].iterdir()}
    # Keep fixture-only shape metadata out of the real API call.
    reports, outputs = [], []
    for enabled in (False, True):
        arm = root / ("prepacked" if enabled else "baseline")
        report, actual = run_arm(arm, options, expected, case=case, prepack=enabled, monkeypatch=monkeypatch)
        reports.append(report)
        outputs.append(actual)
        assert capture_hashes == {path.name: sha256_file(path) for path in options["capture"].iterdir()}
    before, after = reports
    assert before["capture_source_sha256"] == after["capture_source_sha256"]
    assert before["implementation_sources_sha256"] == after["implementation_sources_sha256"]
    assert before["materializations"]["upstream"]["transposes"] == 2
    assert after["materializations"]["upstream"]["transposes"] == 0
    for stage in ("bufferized", "post_sink"):
        assert before["materializations"][stage]["permutation_regions"] == 2
        assert after["materializations"][stage]["permutation_regions"] == 0
        assert (
            before["materializations"][stage]["logical_materialized_output_bytes"]
            == 2 * prod(case[0]) * int(case[2][1:]) // 8
        )
        assert after["materializations"][stage]["logical_materialized_output_bytes"] == 0
    assert len(after["llvm_allocations"]) < len(before["llvm_allocations"])
    for baseline, optimized in zip(*outputs, strict=True):
        assert (
            baseline.dtype == optimized.dtype
            and baseline.shape == optimized.shape
            and baseline.tobytes() == optimized.tobytes()
        )
    result = dict(
        case=case,
        capture_artifacts_sha256=capture_hashes,
        arms=reports,
        numerical_policy="source-ordered wrapping integer multiplication and addition; all6completeoutputs exact",
        timing_scope=(
            "compile RSS/wall, simulator process wall and functional instruction count; no latency/speedup claim"
        ),
    )
    (root / "weight_model.json").write_text(json.dumps(result, indent=2))
    record_property("weight_prepack_model", json.dumps(result, sort_keys=True))


@pytest.mark.parametrize("case", CASES)
def test_weight_model_fixture_is_verified_and_complete(tmp_path, case):
    expected = capture(tmp_path / "capture", case)
    module = mq.parse((tmp_path / "capture" / "model.mlir").read_text())
    module.verify()
    assert sum(op.name == "linalg.transpose" for op in module.walk()) == 2
    assert len(expected) == 6 and expected[0].shape == (3, prod(case[0]) // 3, 3)
    assert expected[1].tobytes() == expected[0][-1].tobytes()


def no_execution(*args, **kwargs):
    pytest.fail("unsupported or corrupted weight prepacking reached model simulation")


@pytest.mark.parametrize("mutation", ["opaque", "mutable", "state", "stream"])
def test_ordinary_weight_prepacking_refuses_missing_ownership(tmp_path, monkeypatch, mutation):
    _expected, root, options = setup(tmp_path, monkeypatch, CASES[0])
    directory = options["capture"]
    if mutation == "opaque":
        module = mq.parse((directory / "model.mlir").read_text())
        fn = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
        opaque = func.FuncOp.external("unqualified_borrower", [fn.args[1].type], [])
        module.body.block.add_op(opaque)
        fn.body.block.insert_op_before(func.CallOp("unqualified_borrower", [fn.args[1]], []), fn.body.block.last_op)
        module.verify()
        (directory / "model.mlir").write_text(ir_text(module))
    elif mutation == "mutable":
        path = directory / "weights.safetensors.manifest.json"
        manifest = json.loads(path.read_text())
        manifest["0"]["mutable"] = True
        path.write_text(json.dumps(manifest))
    else:
        family = "states" if mutation == "state" else "streams"
        (directory / "session_contract.yaml").write_text(
            f"version: 1\nsteps: 3\n{family}:\n- input_arg: 0\n  output_index: 1\n  key: changing_weight\n"
        )
    seal_capture(directory)
    assert sum(op.name == "linalg.transpose" for op in mq.parse((directory / "model.mlir").read_text()).walk()) == 2
    monkeypatch.setattr(spike_model, "run_raw", no_execution)
    with pytest.raises(Exception, match="immutable|unknown|pure|hoistable|tensor function"):
        compile_saved_model(**options, output=root / "refused", prepack_weights=True)
    receipt = json.loads((root / "refused" / "baremetal_model.json").read_text())
    assert receipt["status"] == "failed"
    assert not (root / "refused" / "build" / "model.elf").exists()


def test_ordinary_weight_prepacking_checks_reendorsed_cache_payload(tmp_path, monkeypatch):
    from merlin.llvmlower import weight_prepack
    from merlin.llvmlower.weight_prepack_runtime import fixed_arguments

    _expected, root, options = setup(tmp_path, monkeypatch, CASES[1])
    directory = options["capture"]
    manifest = json.loads((directory / "weights.safetensors.manifest.json").read_text())
    immutable = fixed_arguments(directory, manifest)
    assert immutable == (0,)
    prepared, _ = weight_prepack.prepacked_bundle(directory, immutable_arguments=immutable)
    header, offset = load_header(prepared / "weights.safetensors")
    path = prepared / "weights.safetensors"
    damaged = bytearray(path.read_bytes())
    damaged[offset + header["weight"]["data_offsets"][0]] ^= 1
    path.write_bytes(damaged)
    identity_path = prepared / "weight-prepack-identity.json"
    identity = json.loads(identity_path.read_text())
    for filename, observed in identity["outputs"].items():
        changed = prepared / filename
        observed.update(bytes=changed.stat().st_size, sha256=sha256_file(changed))
    identity_path.write_text(json.dumps(identity))
    monkeypatch.setattr(spike_model, "run_raw", no_execution)
    with pytest.raises(Exception, match="inverse layout|stored element bytes"):
        compile_saved_model(**options, output=root / "refused", prepack_weights=True)
    receipt = json.loads((root / "refused" / "baremetal_model.json").read_text())
    assert receipt["status"] == "failed"
    assert not (root / "refused" / "build" / "model.elf").exists()
