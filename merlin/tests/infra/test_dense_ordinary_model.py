"""Ordinary selected source-dense scheduling, actual primitives and full state."""

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_dense_source_catalog import compilation_memory
from test_dense_source_provider import DenseProvider
from test_dense_source_trace import check_trace
from test_dense_source_whole_model import model_source
from test_selected_dense_contracts import inputs as dense_inputs

from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike, spike_model
from merlin.targetgen.elf_lanes import read_program_section
from merlin.transforms.certificates import ir_text


def product(left, right):
    output = np.empty((left.shape[0], right.shape[1]), dtype="<i4")
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            value = sum(int(left[i, k]) * int(right[k, j]) for k in range(left.shape[1]))
            output[i, j] = (value + (1 << 31)) % (1 << 32) - (1 << 31)
    return output


def capture(directory, shape, width):
    directory.mkdir()
    m, k, n = shape
    dtype = "i8" if width == 1 else "i16"
    npdtype = "i1" if width == 1 else "<i2"
    rng = np.random.default_rng(921)
    bound = 1 << (8 * width - 1)
    w0 = rng.integers(-bound, bound, (k, n)).astype(npdtype)
    w1 = rng.integers(-bound, bound, (n, k)).astype(npdtype)
    data = rng.integers(-bound, bound, (m, k)).astype(npdtype)
    initial = rng.integers(-bound, bound, (m, k)).astype(npdtype)
    state, position = initial.copy(), 7
    history, trace = [], []
    for _ in range(3):
        a = np.add(data, state, dtype=state.dtype)
        first = product(a, w0)
        trace.append((0, a.copy(), w0.copy(), first.copy()))
        activation = np.maximum(first, 0).astype(state.dtype)
        second = product(activation, w1)
        trace.append((1, activation.copy(), w1.copy(), second.copy()))
        state = np.add(second, np.int32(position), dtype=np.int32).astype(state.dtype)
        history.append(state.copy())
        position += 1
    expected = (np.stack(history), state, np.array(position, "<i4"), data.copy(), w0.copy(), w1.copy())
    save_file(dict(w0=w0, w1=w1), directory / "weights.safetensors")
    manifest = {
        "0": dict(kind="param", weight="w0"),
        "1": dict(kind="param", weight="w1"),
        "2": dict(kind="input", name="input"),
        "3": dict(kind="input", name="state"),
        "4": dict(kind="input", name="position"),
    }
    (directory / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    np.savez(directory / "inputs.npz", in0=data, in1=initial, in2=np.array(7, "<i4"))
    (directory / "input_order.json").write_text('{"input":0,"state":1,"position":2}')
    np.save(directory / "golden.npy", expected[0])
    text = model_source(m, k, n, dtype)
    # Echo both original weight tensors too; no captured region labels are used.
    old = (
        f"return %history,%state,%lastpos,%input:tensor<3x{m}x{k}x{dtype}>,\n"
        f"   tensor<{m}x{k}x{dtype}>,tensor<i32>,tensor<{m}x{k}x{dtype}>"
    )
    new = (
        old.replace("%lastpos,%input:", "%lastpos,%input,%w0,%w1:")
        + f",tensor<{k}x{n}x{dtype}>,tensor<{n}x{k}x{dtype}>"
    )
    assert text.count(old) == 1
    text = text.replace(old, new)
    old_type = f"-> (tensor<3x{m}x{k}x{dtype}>,tensor<{m}x{k}x{dtype}>,tensor<i32>,tensor<{m}x{k}x{dtype}>)"
    text = text.replace(old_type, old_type[:-1] + f",tensor<{k}x{n}x{dtype}>,tensor<{n}x{k}x{dtype}>)", 1)
    old_slice = (
        f"%history1=tensor.insert_slice %next into %h[%step,0,0] [1,{m},{k}] [1,1,1]\n"
        f"    :tensor<{m}x{k}x{dtype}> into tensor<3x{m}x{k}x{dtype}>"
    )
    new_slice = f"""%history1="tensor.insert_slice"(%next,%h,%step) <{{
 operandSegmentSizes=array<i32:1,1,1,0,0>,static_offsets=array<i64:-9223372036854775808,0,0>,
 static_sizes=array<i64:1,{m},{k}>,static_strides=array<i64:1,1,1>}}>
 :(tensor<{m}x{k}x{dtype}>,tensor<3x{m}x{k}x{dtype}>,index)->tensor<3x{m}x{k}x{dtype}>"""
    assert text.count(old_slice) == 1
    text = text.replace(old_slice, new_slice)
    source = directory / "model.mlir"
    module = mq.parse(text).clone()
    for operation in module.walk():
        operation.attributes.pop("prov.region_id", None)
    module.verify()
    source.write_text(ir_text(module))
    return expected, trace


def observe_console(console, provider, native):
    calls = []
    pending = {i: [] for i in range(len(provider.records))}
    for line in console.splitlines():
        if line.startswith(b"MERLIN_DENSE_TRACE "):
            index, *record = map(int, line.split()[1:])
            pending[index].append(record)
        elif line.startswith(b"MERLIN_DENSE_COUNTS "):
            index, *counters = map(int, line.split()[1:])
            item = provider.records[index]
            effects = check_trace(
                pending[index], item["problem"], provider.storage, item["result"].compiled.kernel.args
            )
            calls.append(dict(index=index, trace=pending[index], effects=effects, counters=counters))
            pending[index] = []
    assert not any(pending.values())
    assert calls == native["calls"]
    assert [row["index"] for row in calls] == [0, 1] * 3
    return calls


def build_model(bundle, work, selection):
    return spike_model.build(
        bundle,
        work,
        arena_mb=1,
        dram_bytes=64 << 20,
        code_reserve=16 << 20,
        backend="scalar",
        full_readback=True,
        shared_source_lowerings=(selection,),
        host_llvm_bitcode=True,
        cflags_override=[
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
        ],
    )


@pytest.mark.parametrize("shape,tile,width,shared", [((3, 7, 5), (2, 3, 4), 1, False), ((4, 9, 7), (3, 2, 5), 2, True)])
def test_ordinary_dense_selection_runs_full_model_and_observed_effects(
    tmp_path, record_property, shape, tile, width, shared
):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("native lowering and functional simulator unavailable")
    facts = tmp_path / "capabilities"
    facts.mkdir()
    model = dense_inputs(facts, tile=tile, widths=(width, width, 4), shared=shared, shape=shape)
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, shape, width)
    observations = []
    outputs = []
    for name, streaming in (("baseline", True), ("optimized", False)):
        provider = DenseProvider(tmp_path / (name + "_provider"), model, trace, streaming=streaming)
        work = tmp_path / name
        work.mkdir()
        started = time.perf_counter()
        with compilation_memory() as memory:
            built = build_model(bundle, work, provider.selection())
        elapsed = time.perf_counter() - started
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        (work / "console.bin").write_bytes(console)
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        assert len(actual) == len(expected) == 6
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)
        outputs.append(actual)
        native = json.loads((provider.work / "native_qualification.json").read_text())
        actual_calls = observe_console(console, provider, native)
        sections = []
        for record in provider.records:
            code, header = read_program_section(built["elf"], record["section"], max_bytes=64 << 10)
            assert not header["writable"]
            sections.append(dict(**header, sha256=hashlib.sha256(code).hexdigest()))
            record["result"].revalidate()
        receipts = list(work.rglob("source_dense.json"))
        assert len(receipts) == 1
        source = json.loads(receipts[0].read_text())
        assert len(source["bindings"]) == 2 and all(r["family"] == "dense" for r in source["bindings"])
        assert source["selected_sha256"] == sha256_file(receipts[0].with_suffix(".mlir"))
        host = json.loads((work / "host_provider/host_provider.json").read_text())
        assert host["status"] == "completed" and len(host["objects"]) == 2
        observations.append(
            dict(
                arm=name,
                compile_seconds=elapsed,
                compilation_memory=memory,
                actual_calls=actual_calls,
                native_qualification=native,
                source_selection=source,
                choices=[asdict(r["result"].compiled.choice) for r in provider.records],
                model_object_sha256=sha256_file(work / "model.o"),
                elf_sha256=sha256_file(built["elf"]),
                elf_bytes=built["elf"].stat().st_size,
                executable_sections=sections,
                host_provider=host,
                functional_forward_instructions=metadata["metrics"]["cycles"],
                simulator_process_wall_seconds=wall,
                scope=(
                    "neutral host primitives include trace logging; "
                    "not accelerator latency or arbitrary C/ELF conformance"
                ),
            )
        )
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    before, after = observations
    assert before["model_object_sha256"] == after["model_object_sha256"]
    assert before["source_selection"]["source_sha256"] == after["source_selection"]["source_sha256"]
    assert before["source_selection"]["selected_sha256"] == after["source_selection"]["selected_sha256"]
    for b, a in zip(before["source_selection"]["bindings"], after["source_selection"]["bindings"], strict=True):
        bp, ap = b["schedule"]["pipeline"], a["schedule"]["pipeline"]
        assert bp["library_sources_sha256"] == ap["library_sources_sha256"]
        assert bp["selected_contracts"] == ap["selected_contracts"]
        for key in ("hardware", "execution", "library", "numerical_policy"):
            assert bp["context"][key] == ap["context"][key]
    for key in ("requested_read_bytes",):
        assert sum(c["effects"][key] for c in after["actual_calls"]) < sum(
            c["effects"][key] for c in before["actual_calls"]
        )
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))
    record_property("ordinary_dense_pair", json.dumps(observations))


@pytest.mark.parametrize(
    "mutation,diagnostic", [("address", "wrong tile address"), ("continuation", "Arrays are not equal")]
)
def test_wrong_actual_primitive_refuses_ordinary_model_before_emission(tmp_path, mutation, diagnostic):
    from merlin.llvmlower import toolchain

    if not toolchain.available():
        pytest.skip("native lowering unavailable")
    facts = tmp_path / "capabilities"
    facts.mkdir()
    model = dense_inputs(facts, shape=(3, 7, 5))
    bundle = tmp_path / "capture"
    _, trace = capture(bundle, (3, 7, 5), 1)
    provider = DenseProvider(tmp_path / "provider", model, trace, mutation=mutation)
    work = tmp_path / "model"
    work.mkdir()
    with pytest.raises(AssertionError, match=diagnostic):
        build_model(bundle, work, provider.selection())
    assert not (provider.work / "native_qualification.json").exists()
    assert not list(work.rglob("*.elf")) and not list(work.rglob("host_provider.json"))


def test_unavailable_reduction_tails_keep_complete_host_model(tmp_path):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("native lowering and functional simulator unavailable")
    facts = tmp_path / "capabilities"
    facts.mkdir()

    def unavailable(facts, software, dense):
        dense["support"]["partial_reduction_tiles"] = False

    model = dense_inputs(facts, shape=(3, 7, 5), mutate=unavailable)
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, (3, 7, 5), 1)
    provider = DenseProvider(tmp_path / "provider", model, trace)
    work = tmp_path / "model"
    work.mkdir()
    built = build_model(bundle, work, provider.selection())
    console, _ = FB.run_host(built["elf"], timeout_s=30)
    (work / "console.bin").write_bytes(console)
    actual, _ = FB.read(console, bundle, built["build_hash"])
    assert len(actual) == len(expected) == 6
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)
    assert not provider.records and b"MERLIN_DENSE_" not in console
    assert not (provider.work / "native_qualification.json").exists()
    receipts = list(work.rglob("source_dense.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["bindings"] == [] and len(receipt["declined"]) == 2
    assert all("reduction" in reason for reason in receipt["declined"])
    assert receipt["source_sha256"] == receipt["selected_sha256"] == sha256_file(receipts[0].parent / "input.mlir")
