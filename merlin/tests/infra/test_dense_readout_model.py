"""Ordinary recurrent dense readout: full i32 reduction, exact i8 destination.

Observed neutral C primitive requests are not accelerator traffic or latency.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_dense_integer_readout import readout_inputs
from test_dense_ordinary_model import build_model, product
from test_dense_readout_provider import ReadoutProvider
from test_dense_source import contraction
from test_dense_source_catalog import compilation_memory
from test_dense_source_trace import KINDS, check_trace
from test_owned_buffer_host_model import generic_snapshot

from merlin.common import compile_trace
from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike
from merlin.targetgen.elf_lanes import read_program_section
from merlin.transforms.certificates import ir_text


def model_source(shape, width):
    m, k, n = shape
    dtype = "i8" if width == 1 else "i16"
    operand, weight, output = f"tensor<{m}x{k}x{dtype}>", f"tensor<{k}x{n}x{dtype}>", f"tensor<{m}x{n}xi8>"
    history = f"tensor<3x{m}x{n}xi8>"
    dense = contraction(shape, (dtype, dtype, "i32"), lhs="mixed", rhs="weight", output="sum")
    return f"""module {{
func.func @forward(%weight:{weight},%input:{operand},%initial:tensor<i32>,%position:tensor<i32>)
 ->({history},tensor<i32>,tensor<i32>,{operand},{weight}) {{
%zero8=arith.constant 0:i8
%zero32=arith.constant 0:i32
%high=arith.constant 127:i32
%one=arith.constant 1:i32
%c0=arith.constant 0:index
%c1=arith.constant 1:index
%c3=arith.constant 3:index
%he=tensor.empty():{history}
%hz=linalg.fill ins(%zero8:i8) outs(%he:{history})->{history}
%state,%history,%last=scf.for %step=%c0 to %c3 step %c1
 iter_args(%s=%initial,%h=%hz,%p=%position)->(tensor<i32>,{history},tensor<i32>) {{
%me=tensor.empty():{operand}
%mixed=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->()>,
 affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%input,%s:{operand},tensor<i32>) outs(%me:{operand}) {{
 ^bb0(%x:{dtype},%sv:i32,%old:{dtype}):
 %small=arith.trunci %sv:i32 to {dtype}
 %added=arith.addi %x,%small:{dtype}
 linalg.yield %added:{dtype}
}}->{operand}
{dense}
%oe=tensor.empty():{output}
%next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}} ins(%sum:tensor<{m}x{n}xi32>) outs(%oe:{output}) {{
 ^bb0(%v:i32,%unused:i8):
 %positive=arith.maxsi %v,%zero32:i32
 %bounded=arith.minsi %positive,%high:i32
 %narrow=arith.trunci %bounded:i32 to i8
 linalg.yield %narrow:i8
}}->{output}
%h1="tensor.insert_slice"(%next,%h,%step) <{{operandSegmentSizes=array<i32:1,1,1,0,0>,
 static_offsets=array<i64:-9223372036854775808,0,0>,static_sizes=array<i64:1,{m},{n}>,
 static_strides=array<i64:1,1,1>}}> :({output},{history},index)->{history}
%chosen=tensor.extract %next[%c0,%c0]:{output}
%chosen32=arith.extsi %chosen:i8 to i32
%oldstate=tensor.extract %s[]:tensor<i32>
%pv=tensor.extract %p[]:tensor<i32>
%increased=arith.addi %oldstate,%chosen32:i32
%sv1=arith.addi %increased,%pv:i32
%s1=tensor.insert %sv1 into %s[]:tensor<i32>
%pn=arith.addi %pv,%one:i32
%p1=tensor.insert %pn into %p[]:tensor<i32>
scf.yield %s1,%h1,%p1:tensor<i32>,{history},tensor<i32>
}}
func.return %history,%state,%last,%input,%weight:{history},tensor<i32>,tensor<i32>,{operand},{weight}
}}
}}"""


def wrap(values, width):
    bits = width * 8
    return ((np.asarray(values, dtype=np.int64) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))).astype(
        {1: "i1", 2: "<i2", 4: "<i4"}[width]
    )


def capture(directory, shape, width):
    directory.mkdir()
    m, k, n = shape
    dtype = "i1" if width == 1 else "<i2"
    seed = [64, 64, 1, 0, -1, 127, -128] if width == 1 else [32767, 32767, 32767, 64, 64, 1, -1, 127, -128]
    data = np.resize(np.array(seed, dtype=dtype), (m, k))
    weight = np.zeros((k, n), dtype=dtype)
    if width == 1:
        weight[:2, 0], weight[:2, 1], weight[2, 2], weight[5, 3] = 1, -1, 1, 1
        weight[:2, 4], weight[2, 4] = 1, -2
    else:
        weight[:3, 0], weight[:3, 1], weight[3:5, 2], weight[5, 3], weight[7, 4] = 32767, -32768, 1, 1, 1
        weight[7, 5], weight[6, 5], weight[8, 6] = 1, 1, -1
        assert sum(int(data[0, t]) * int(weight[t, 0]) for t in range(k)) > (1 << 31) - 1
    state, position, history, trace = 0, 7, [], []
    for _ in range(3):
        mixed = wrap(data.astype(np.int64) + state, width)
        raw = product(mixed, weight)
        trace.append((0, mixed.copy(), weight.copy(), raw.copy()))
        result = np.clip(raw.astype(np.int64), 0, 127).astype("i1")
        history.append(result.copy())
        state = int(wrap([state + int(result[0, 0]) + position], 4)[0])
        position += 1
    # Deliberately distinguish clamp-before-truncation and the 126/127 bound.
    raw_first = trace[0][-1]
    assert np.any(raw_first == 128) and np.any(raw_first == 127) and np.any(raw_first < 0)
    expected = (np.stack(history), np.array(state, "<i4"), np.array(position, "<i4"), data.copy(), weight.copy())
    save_file({"weight": weight}, directory / "weights.safetensors")
    (directory / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": dict(kind="param", weight="weight"),
                "1": dict(kind="input", name="input"),
                "2": dict(kind="input", name="state"),
                "3": dict(kind="input", name="position"),
            }
        )
    )
    np.savez(directory / "inputs.npz", in0=data, in1=np.array(0, "<i4"), in2=np.array(7, "<i4"))
    (directory / "input_order.json").write_text(json.dumps(dict(input=0, state=1, position=2)))
    np.save(directory / "golden.npy", expected[0])
    names = ("history", "state", "position", "input", "weight")
    np.savez(directory / "goldens.npz", **dict(zip(names, expected, strict=True)))
    (directory / "output_order.json").write_text(json.dumps(names))
    module = parse_mlir_text(model_source(shape, width))
    for op in module.walk():
        op.attributes.pop("prov.region_id", None)
    module.verify()
    (directory / "model.mlir").write_text(ir_text(module))
    (directory / "capture_receipt.json").write_text(
        json.dumps(
            dict(
                schema="m2m.capture-receipt.v1",
                materialized_abi=dict(complete=True),
                artifacts={p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in directory.iterdir()},
            )
        )
    )
    return expected, trace


@pytest.mark.parametrize("shape,width", [((3, 7, 5), 1), ((4, 9, 7), 2)])
def test_readout_capture_has_typed_recurrence_and_full_domain_boundaries(tmp_path, shape, width):
    expected, trace = capture(tmp_path / "capture", shape, width)
    module = parse_mlir_text((tmp_path / "capture/model.mlir").read_text())
    module.verify()
    assert sum(op.name == "scf.for" for op in module.walk()) == 1
    assert len(expected) == 5 and len(trace) == 3
    assert expected[0].dtype == np.dtype("i1") and expected[1].dtype == np.dtype("<i4")
    assert not np.array_equal(trace[0][1], trace[1][1])


def actual_calls(console, provider, native):
    pending, calls = [], []
    record = provider.records[0]
    output_dtype = record["result"].compiled.kernel.args[-1].dtype
    for line in console.splitlines():
        if line.startswith(b"MERLIN_DENSE_TRACE "):
            index, *row = map(int, line.split()[1:])
            assert index == 0
            pending.append(row)
        elif line.startswith(b"MERLIN_DENSE_COUNTS "):
            index, *counters = map(int, line.split()[1:])
            assert index == 0
            effects = check_trace(
                pending,
                record["problem"],
                provider.storage,
                record["result"].compiled.kernel.args,
                output_dtype=output_dtype,
            )
            calls.append(dict(index=index, trace=pending, effects=effects, counters=counters))
            pending = []
    assert not pending and len(calls) == 3 and calls == native["calls"]
    return calls


def realized_intermediate(trace_root, work, shape):
    m, _, n = shape
    doc = json.loads((trace_root / "trace.json").read_text())
    assert doc["outcome"] == compile_trace.COMPLETED
    records = []
    for stage in doc["stages"]:
        if stage["kind"] != "native" or stage["stage"] != "mlir:one-shot-bufferize":
            continue
        for filename in stage["files"]:
            path = trace_root / filename
            generic = work / ("bufferized-" + str(len(records)) + ".mlir")
            module = mq.parse(generic_snapshot(path, generic))
            # The native compiler verified this retained upstream product.
            # Its arithmetic dialect has properties absent in this xDSL
            # version. Observe and verify only the typed allocations here;
            # preserve every original arithmetic property without interpreting it.
            allocations = [
                op
                for op in module.walk()
                if op.name in ("memref.alloc", "memref.alloca")
                and op.results[0].type.get_shape() == (m, n)
                and str(op.results[0].type.element_type) == "i32"
            ]
            for allocation in allocations:
                allocation.verify()
            records.append(
                dict(
                    path=str(path),
                    sha256=sha256_file(path),
                    generic_sha256=sha256_file(generic),
                    raw_i32_output_allocations=len(allocations),
                )
            )
    assert len(records) == 1
    return records[0]


@pytest.mark.parametrize("shape,tile,width,shared", [((3, 7, 5), (2, 3, 4), 1, False), ((4, 9, 7), (3, 2, 5), 2, True)])
def test_ordinary_recurrent_dense_readout_removes_dump_and_conversion(
    tmp_path, record_property, shape, tile, width, shared
):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    facts = tmp_path / "capabilities"
    facts.mkdir()
    model = readout_inputs(facts, tile=tile, widths=(width, width, 4), shape=shape, shared=shared)
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, shape, width)
    observations, outputs = [], []
    for arm, enabled in (("baseline", False), ("optimized", True)):
        provider = ReadoutProvider(tmp_path / (arm + "_provider"), model, trace, enabled=enabled)
        work = tmp_path / arm
        work.mkdir()
        trace_root = work / "lowering_trace"
        started = time.perf_counter()
        with (
            compilation_memory() as memory,
            compile_trace.session(compile_trace.Request(str(trace_root), dump_after=("mlir:one-shot-bufferize",))),
        ):
            built = build_model(bundle, work, provider.selection())
        seconds = time.perf_counter() - started
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        (work / "console.bin").write_bytes(console)
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        assert len(actual) == len(expected) == 5
        for got, want in zip(actual, expected, strict=True):
            assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()
        outputs.append(actual)
        assert len(provider.records) == 1
        record = provider.records[0]
        result = record["result"]
        assert result.compiled.kernel.args[-1].dtype == ("i8" if enabled else "i32")
        assert model[1].accumulator_bytes == 4
        native = json.loads((provider.work / "native_qualification.json").read_text())
        calls = actual_calls(console, provider, native)
        for call in calls:
            stores = [row for row in call["trace"] if KINDS[row[0]] in ("store", "store_i8")]
            assert stores and all(KINDS[row[0]] == ("store_i8" if enabled else "store") for row in stores)
            bad = [row.copy() for row in call["trace"]]
            bad[next(i for i, row in enumerate(bad) if KINDS[row[0]] in ("store", "store_i8"))][0] = KINDS.index(
                "store" if enabled else "store_i8"
            )
            with pytest.raises(AssertionError, match="readout mode"):
                check_trace(
                    bad,
                    record["problem"],
                    provider.storage,
                    result.compiled.kernel.args,
                    output_dtype=result.output_dtype,
                )
        code, section = read_program_section(built["elf"], record["section"], max_bytes=64 << 10)
        assert code and not section["writable"]
        result.revalidate()
        (source_path,) = work.rglob("source_dense.json")
        source = json.loads(source_path.read_text())
        selected = source_path.with_suffix(".mlir")
        assert source["selected_sha256"] == sha256_file(selected)
        selected_module = parse_mlir_text(selected.read_text())
        readout_scalars = sum(op.name in ("arith.maxsi", "arith.minsi") for op in selected_module.walk())
        assert readout_scalars == (0 if enabled else 2)
        materialization = realized_intermediate(trace_root, work, shape)
        assert materialization["raw_i32_output_allocations"] == (0 if enabled else 1)
        host = built["host_provider"]
        assert host["status"] == "completed" and len(host["objects"]) == 1
        for pin in (*host["source_model"], *(row[key] for row in host["objects"] for key in ("object", "llvm"))):
            assert pin["sha256"] == sha256_file(Path(pin["path"]))
        recipe = json.loads((work / "compilation_recipe.json").read_text())
        observations.append(
            dict(
                arm=arm,
                compile_seconds=seconds,
                compilation_memory=memory,
                actual_calls=calls,
                native_qualification=native,
                original_qualification=provider.qualification,
                original_qualification_sha256=provider.qualification_sha256,
                source_receipt=source,
                realized_materialization=materialization,
                elf_sha256=sha256_file(built["elf"]),
                elf_bytes=built["elf"].stat().st_size,
                elf_sections=FB.elf_sections(built["elf"]),
                actual_kernel_section=dict(**section, sha256=hashlib.sha256(code).hexdigest()),
                host_model_object_sha256=sha256_file(work / "model.o"),
                compiler_commands=len(recipe["commands"]),
                provider_object_bytes=Path(host["objects"][0]["object"]["path"]).stat().st_size,
                functional_forward_instructions=metadata["metrics"]["cycles"],
                simulator_process_wall_seconds=wall,
                physical_traffic_bytes=None,
                host_accelerator_component_seconds=None,
                scope=(
                    "neutral instrumented host primitives; requested logical payload/packing only; "
                    "no latency or speedup claim"
                ),
            )
        )
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    baseline, optimized = observations
    for key in ("original_qualification", "original_qualification_sha256"):
        assert baseline[key] == optimized[key]
    b = baseline["source_receipt"]["bindings"][0]["schedule"]["pipeline"]
    a = optimized["source_receipt"]["bindings"][0]["schedule"]["pipeline"]
    assert b["library_sources_sha256"] == a["library_sources_sha256"]
    assert {k: v for k, v in b["selected_contracts"].items() if k not in ("consumed_scopes", "output_readout")} == {
        k: v for k, v in a["selected_contracts"].items() if k not in ("consumed_scopes", "output_readout")
    }
    assert "output_readout" not in b["selected_contracts"]
    readout = a["selected_contracts"]["output_readout"]
    assert readout["identity"] == model[0].dense.output_readout.identity
    assert readout["qualification_identity"] == optimized["original_qualification"]["identity"]
    assert readout["output_dtype"] == "i8"
    for key in ("hardware", "library", "numerical_policy"):
        assert b["context"][key] == a["context"][key]
    assert baseline["source_receipt"]["source_sha256"] == optimized["source_receipt"]["source_sha256"]
    before, after = (row["actual_calls"][-1]["counters"] for row in observations)
    assert before[:2] == after[:2]  # Same actual MAC/load/store command count and operand loads.
    assert before[2] == after[2] * 4 == 12 * shape[0] * shape[2]
    assert before[3] - after[3] == before[4] - after[4] == 9 * shape[0] * shape[2]
    assert before[5] == after[5] == 3
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))
    record_property("exact_dense_readout_pair", json.dumps(observations))


@pytest.mark.parametrize("mutation", ["readout_order", "readout_bound"])
def test_wrong_actual_readout_fails_before_provider_link(tmp_path, mutation):
    from merlin.llvmlower import toolchain

    if not toolchain.available():
        pytest.skip("native ABI qualification unavailable")
    facts = tmp_path / "capabilities"
    facts.mkdir()
    model = readout_inputs(facts, shape=(3, 7, 5))
    bundle = tmp_path / "capture"
    _, trace = capture(bundle, (3, 7, 5), 1)
    provider = ReadoutProvider(tmp_path / "provider", model, trace, enabled=True, mutation=mutation)
    work = tmp_path / "model"
    work.mkdir()
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        build_model(bundle, work, provider.selection())
    assert not (provider.work / "native_qualification.json").exists()
    assert not list(work.rglob("*.elf"))


@pytest.mark.parametrize("unavailable", [False, True])
def test_unqualified_or_unavailable_readout_keeps_dense_and_host_conversion(tmp_path, unavailable):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    facts = tmp_path / "capabilities"
    facts.mkdir()
    model = readout_inputs(
        facts,
        shape=(3, 7, 5),
        mutate=(lambda f, s, d: d.pop("output_readout")) if unavailable else None,
    )
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, (3, 7, 5), 1)
    provider = ReadoutProvider(tmp_path / "provider", model, trace, enabled=True, qualified=unavailable)
    work = tmp_path / "model"
    work.mkdir()
    built = build_model(bundle, work, provider.selection())
    console, _ = FB.run_host(built["elf"], timeout_s=30)
    (work / "console.bin").write_bytes(console)
    actual, _ = FB.read(console, bundle, built["build_hash"])
    assert len(actual) == 5
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)
    assert len(provider.records) == 1
    assert provider.records[0]["result"].output_dtype == "i32"
    assert len(provider.records[0]["result"].operations) == 1
    native = json.loads((provider.work / "native_qualification.json").read_text())
    calls = actual_calls(console, provider, native)
    assert all(KINDS[row[0]] != "store_i8" for call in calls for row in call["trace"])
    (source_path,) = work.rglob("source_dense.json")
    selected = parse_mlir_text(source_path.with_suffix(".mlir").read_text())
    assert sum(op.name in ("arith.maxsi", "arith.minsi") for op in selected.walk()) == 2
