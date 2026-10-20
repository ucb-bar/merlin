"""Two shared source families in one ordinary recurrent model transaction.

Neutral C primitives establish shared discovery, composition and artifact
custody. Complete independent outputs and actual model counters are checked;
there is no accelerator qualification or timing claim.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_dense_source_catalog import compilation_memory
from test_pointwise_source_provider import Provider
from test_pointwise_source_whole_model import epilogue
from test_row_softmax import exp_object as exp_object  # noqa: F401
from test_source_row_model import independent_softmax
from test_source_row_provider import RowProvider
from xdsl.dialects import func
from xdsl.rewriter import Rewriter

from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.llvmlower.cpu_softmax import CPU_SOFTMAX_F32_AVX2_POLICY as POLICY
from merlin.llvmlower.cpu_softmax import build_cpu_softmax_lastdim
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike, spike_model
from merlin.targetgen.elf_lanes import read_program_section
from merlin.transforms.certificates import ir_text


def model_text():
    return f"""module {{
 func.func private @row_placeholder(tensor<2x19xf32>)->tensor<2x19xf32>
 func.func @forward(%scores:tensor<2x19xf32>,%bias:tensor<2x19xi32>)
 ->(tensor<3x2x19xf32>,tensor<2x19xf32>,tensor<3x2x19xi8>,tensor<2x19xi8>,
 tensor<2x19xf32>,tensor<2x19xi32>) {{
 %zero=arith.constant 0.0:f32
 %izero=arith.constant 0:i8
 %c0=arith.constant 0:index
 %c1=arith.constant 1:index
 %c3=arith.constant 3:index
 %fe=tensor.empty():tensor<3x2x19xf32>
 %fz=linalg.fill ins(%zero:f32) outs(%fe:tensor<3x2x19xf32>)->tensor<3x2x19xf32>
 %he=tensor.empty():tensor<3x2x19xi8>
 %hz=linalg.fill ins(%izero:i8) outs(%he:tensor<3x2x19xi8>)->tensor<3x2x19xi8>
 %history,%integer_history=scf.for %step=%c0 to %c3 step %c1
 iter_args(%fh=%fz,%ih=%hz)->(tensor<3x2x19xf32>,tensor<3x2x19xi8>) {{
 %first=arith.cmpi eq,%step,%c0:index
 %current=scf.if %first ->(tensor<2x19xf32>) {{
 scf.yield %scores:tensor<2x19xf32>
 }} else {{
 %previous=arith.subi %step,%c1:index
 %prior="tensor.extract_slice"(%fh,%previous) <{{
 operandSegmentSizes=array<i32:1,1,0,0>,static_offsets=array<i64:-9223372036854775808,0,0>,
 static_sizes=array<i64:1,2,19>,static_strides=array<i64:1,1,1>}}>
 :(tensor<3x2x19xf32>,index)->tensor<2x19xf32>
 scf.yield %prior:tensor<2x19xf32>
 }}
 %prob=func.call @row_placeholder(%current):(tensor<2x19xf32>)->tensor<2x19xf32>
 %qe=tensor.empty():tensor<2x19xi32>
 %quantized=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}}
 ins(%prob:tensor<2x19xf32>) outs(%qe:tensor<2x19xi32>) {{
 ^bb0(%p:f32,%unused:i32):
 %scale=arith.constant 128.0:f32
 %scaled=arith.mulf %p,%scale:f32
 %quantized_scalar=arith.fptosi %scaled:f32 to i32
 linalg.yield %quantized_scalar:i32
 }}->tensor<2x19xi32>
 {epilogue("integer", "quantized", "bias", 2, 19)}
 %fh1="tensor.insert_slice"(%prob,%fh,%step) <{{
 operandSegmentSizes=array<i32:1,1,1,0,0>, static_offsets=array<i64:-9223372036854775808,0,0>,
 static_sizes=array<i64:1,2,19>, static_strides=array<i64:1,1,1>}}>
 :(tensor<2x19xf32>,tensor<3x2x19xf32>,index)->tensor<3x2x19xf32>
 %ih1="tensor.insert_slice"(%integer,%ih,%step) <{{
 operandSegmentSizes=array<i32:1,1,1,0,0>, static_offsets=array<i64:-9223372036854775808,0,0>,
 static_sizes=array<i64:1,2,19>, static_strides=array<i64:1,1,1>}}>
 :(tensor<2x19xi8>,tensor<3x2x19xi8>,index)->tensor<3x2x19xi8>
 scf.yield %fh1,%ih1:tensor<3x2x19xf32>,tensor<3x2x19xi8>
 }}
 %last="tensor.extract_slice"(%history) <{{
 operandSegmentSizes=array<i32:1,0,0,0>,static_offsets=array<i64:2,0,0>,
 static_sizes=array<i64:1,2,19>,static_strides=array<i64:1,1,1>}}>
 :(tensor<3x2x19xf32>)->tensor<2x19xf32>
 %integer_last="tensor.extract_slice"(%integer_history) <{{
 operandSegmentSizes=array<i32:1,0,0,0>,static_offsets=array<i64:2,0,0>,
 static_sizes=array<i64:1,2,19>,static_strides=array<i64:1,1,1>}}>
 :(tensor<3x2x19xi8>)->tensor<2x19xi8>
 return %history,%last,%integer_history,%integer_last,%scores,%bias:
 tensor<3x2x19xf32>,tensor<2x19xf32>,tensor<3x2x19xi8>,tensor<2x19xi8>,tensor<2x19xf32>,tensor<2x19xi32>
 }} }}"""


def capture(directory):
    directory.mkdir()
    rng = np.random.default_rng(871)
    scores = (rng.integers(-32, 33, (2, 19)) / 8).astype("<f4")
    bias = rng.integers(-160, 161, (2, 19)).astype("<i4")
    bias.flat[::5], bias.flat[1::5] = (1 << 31) - 2, -(1 << 31)
    row_trace, pointwise_trace, probability_history, integer_history = [], [], [], []
    current = scores.copy()
    for step in range(3):
        probability = independent_softmax(current, directory, "step" + str(step))
        row_trace.append((current.copy(), probability.copy()))
        scaled = np.multiply(probability, np.float32(128), dtype="<f4")
        assert np.all(np.isfinite(scaled)) and np.all((scaled >= 0) & (scaled <= 128))
        quantized = np.trunc(scaled).astype("<i4")
        pointwise_trace.append((0, quantized.copy(), bias.copy()))
        wide = quantized.astype(np.int64) + bias.astype(np.int64)
        wrapped = (wide + (1 << 31)) % (1 << 32) - (1 << 31)
        integer = np.clip(wrapped, -128, 127).astype("i1")
        probability_history.append(probability)
        integer_history.append(integer)
        current = probability
    expected = (np.stack(probability_history), current, np.stack(integer_history), integer, scores.copy(), bias.copy())
    save_file({}, directory / "weights.safetensors")
    manifest = {str(i): dict(kind="input", name=name) for i, name in enumerate(("scores", "bias"))}
    (directory / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    (directory / "input_order.json").write_text('{"scores":0,"bias":1}')
    np.savez(directory / "inputs.npz", in0=scores, in1=bias)
    np.save(directory / "golden.npy", expected[0])
    source = directory / "model.mlir"
    source.write_text(model_text())
    module = mq.parse(source.read_text()).clone()
    placeholder = next(
        op for op in module.walk() if isinstance(op, func.CallOp) and op.callee.root_reference.data == "row_placeholder"
    )
    operations, result = build_cpu_softmax_lastdim(
        placeholder.arguments[0], backend_policy=POLICY, assume_finite_inputs=True
    )
    Rewriter.replace_op(placeholder, operations, [result])
    declaration = next(
        op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "row_placeholder"
    )
    Rewriter.erase_op(declaration)
    module.verify()
    source.write_text(ir_text(module))
    return expected, row_trace, pointwise_trace


def console_counts(console, marker, expected):
    rows = [
        [int(value) for value in line.split()[1:]] for line in console.splitlines() if line.startswith(marker + b" ")
    ]
    assert len(rows) == 3
    assert [row[5] for row in rows] == [1, 2, 3]
    assert rows[-1] == expected
    for invocation, row in enumerate(rows, 1):
        assert row == [value * invocation // 3 for value in expected]
    return rows


def linked_evidence(elf, row, pointwise):
    kernel, section = read_program_section(elf, ".merlin_row.text", max_bytes=64 << 10)
    assert not section["writable"]
    programs = []
    from merlin.targetgen.isa_dialect_gen import decode_instruction

    for record, result in zip(pointwise.records, pointwise.results, strict=True):
        expected = b"".join(result.images())
        actual, header = read_program_section(elf, record["program_section"], max_bytes=len(expected))
        assert actual == expected and not header["writable"]
        width, byteorder = result.selected_contracts.isa.word_bits // 8, result.selected_contracts.instruction_byteorder
        decoded = [
            decode_instruction(result.selected_contracts.isa, int.from_bytes(actual[i : i + width], byteorder))[0]
            for i in range(0, len(actual), width)
        ]
        assert decoded == ["fused"]
        programs.append(dict(**header, sha256=hashlib.sha256(actual).hexdigest(), instructions=decoded))
    row.result.revalidate()
    return dict(row_kernel={**section, "sha256": hashlib.sha256(kernel).hexdigest()}, pointwise_programs=programs)


def test_ordinary_mixed_source_families_share_one_model_transaction(tmp_path, record_property, exp_object):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle = tmp_path / "capture"
    expected, row_trace, pointwise_trace = capture(bundle)
    observations, outputs = [], []
    for order in (("row", "pointwise"), ("pointwise", "row")):
        name = "_".join(order)
        work = tmp_path / name
        work.mkdir()
        row = RowProvider(tmp_path / (name + "_row"), columns=19, chunk=None, exp_object=exp_object)
        pointwise = Provider(tmp_path / (name + "_pointwise"), enabled=True, separate=True)
        configurations = dict(row=row.selection(row_trace), pointwise=pointwise.selection(pointwise_trace))
        started = time.perf_counter()
        with compilation_memory() as memory:
            built = spike_model.build(
                bundle,
                work,
                arena_mb=1,
                dram_bytes=64 << 20,
                code_reserve=16 << 20,
                backend="scalar",
                full_readback=True,
                shared_source_lowerings=tuple(configurations[family] for family in order),
                cflags_override=[
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-O2",
                    "-ffreestanding",
                    "-fno-builtin",
                    "-ffp-contract=off",
                ],
                host_llvm_bitcode=True,
            )
        elapsed = time.perf_counter() - started
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        (work / "console.bin").write_bytes(console)
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        assert len(actual) == len(expected) == 6
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(
                got.view("<u4") if want.dtype.kind == "f" else got, want.view("<u4") if want.dtype.kind == "f" else want
            )
        outputs.append(actual)
        native = {
            family: json.loads((provider.work / "native_qualification.json").read_text())
            for family, provider in (("row", row), ("pointwise", pointwise))
        }
        counters = dict(
            row=console_counts(console, b"MERLIN_ROW_COUNTS", native["row"]["counters"]),
            pointwise=console_counts(console, b"MERLIN_POINTWISE_COUNTS", native["pointwise"]["counters"]),
        )
        labels = [
            line.split()[0]
            for line in console.splitlines()
            if line.startswith((b"MERLIN_ROW_COUNTS ", b"MERLIN_POINTWISE_COUNTS "))
        ]
        assert labels == [b"MERLIN_ROW_COUNTS", b"MERLIN_POINTWISE_COUNTS"] * 3
        receipts = list(work.rglob("source_schedules.json"))
        assert len(receipts) == 1
        source = json.loads(receipts[0].read_text())
        assert source["schema"] == "merlin.source_model_lowering.v1"
        assert source["selected_sha256"] == sha256_file(receipts[0].with_suffix(".mlir"))
        assert len(source["bindings"]) == 2
        assert {b["family"] for b in source["bindings"]} == {"row", "pointwise"}
        aggregate_path = work / "host_provider" / "source_lowerings.json"
        aggregate = json.loads(aggregate_path.read_text())
        assert aggregate["schema"] == "merlin.source_provider_bundle.v1"
        assert (
            aggregate["scope"] == "individual witness custody only; no aggregate execution or numerical qualification"
        )
        assert {item["family"] for item in aggregate["providers"]} == {"row", "pointwise"}
        providers = {family: provider for family, provider in (("row", row), ("pointwise", pointwise))}
        for record in aggregate["providers"]:
            witness = sha256_file(providers[record["family"]].work / "native_qualification.json")
            assert record["numerical"] == record["effects"] == witness
        linked = json.loads((work / "host_provider" / "host_provider.json").read_text())
        assert linked["status"] == "completed"
        assert linked["numerical_witness_sha256"] == linked["effect_witness_sha256"] == sha256_file(aggregate_path)
        assert linked["resolved_required_symbols"]["row_exp"] == "row_exp"
        assert len(linked["objects"]) == 3
        assert {str(Path(item["object"]["path"]).parent.name) for item in linked["objects"]} == {"row", "pointwise"}
        observations.append(
            dict(
                order=order,
                source=source,
                aggregate_provider=aggregate,
                linked=linked_evidence(built["elf"], row, pointwise),
                native_qualification=native,
                actual_model_counter_traces=counters,
                compile_seconds=elapsed,
                compilation_memory=memory,
                elf_bytes=built["elf"].stat().st_size,
                elf_sha256=sha256_file(built["elf"]),
                model_object_sha256=sha256_file(work / "model.o"),
                functional_forward_instructions=metadata["metrics"]["cycles"],
                simulator_process_wall_seconds=wall,
                counter_scope="functional host execution includes counter logging, not accelerator latency",
            )
        )
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    before, after = observations
    for key in ("source_sha256", "selected_sha256"):
        assert before["source"][key] == after["source"][key]
    assert before["model_object_sha256"] == after["model_object_sha256"]
    assert before["actual_model_counter_traces"] == after["actual_model_counter_traces"]
    (tmp_path / "mixed.json").write_text(json.dumps(observations, indent=2))
    record_property("mixed_shared_source_model", json.dumps(observations))
