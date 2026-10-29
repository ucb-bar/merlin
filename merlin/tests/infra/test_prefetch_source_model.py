"""Ordinary recurrent model prefetch with observed issue/completion behavior.

The neutral byte machine runs on the host CPU. Requested payload counts and
functional simulator instructions are not physical traffic or device latency.
"""

import hashlib
import json
import struct
import time
from dataclasses import asdict, replace

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_dense_source_catalog import compilation_memory
from test_prefetch_source_provider import PrefetchProvider

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike, spike_model
from merlin.targetgen.elf_lanes import read_program_section
from merlin.transforms.certificates import ir_text


def model_source(shape):
    m, n = shape
    tensor = f"tensor<{m}x{n}xi8>"
    history = f"tensor<3x{m}x{n}xi8>"

    def add(name, left, right):
        return f"""
%{name}_init=tensor.empty():{tensor}
%{name}=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%{left},%{right}:{tensor},{tensor}) outs(%{name}_init:{tensor}) {{
 ^bb0(%{name}_x:i8,%{name}_y:i8,%{name}_old:i8):
 %{name}_sum=arith.addi %{name}_x,%{name}_y:i8
 linalg.yield %{name}_sum:i8
}}->{tensor}
"""

    return f"""module {{
func.func @forward(%weight:{tensor},%input:{tensor},%initial:{tensor},%position:tensor<i32>)
 ->({history},{tensor},tensor<i32>,{tensor},{tensor}) {{
%zero=arith.constant 0:i8
%one=arith.constant 1:i32
%c0=arith.constant 0:index
%c1=arith.constant 1:index
%c3=arith.constant 3:index
%he=tensor.empty():{history}
%hz=linalg.fill ins(%zero:i8) outs(%he:{history})->{history}
%state,%history,%last=scf.for %step=%c0 to %c3 step %c1
 iter_args(%s=%initial,%h=%hz,%p=%position)->({tensor},{history},tensor<i32>) {{
{add("mixed", "input", "s")}
{add("weighted", "mixed", "weight")}
%next_init=tensor.empty():{tensor}
%next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}} ins(%weighted:{tensor}) outs(%next_init:{tensor}) {{
 ^bb0(%v:i8,%unused:i8):
 %increment=arith.constant 1:i8
 %result=arith.addi %v,%increment:i8
 linalg.yield %result:i8
}}->{tensor}
%h1="tensor.insert_slice"(%next,%h,%step) <{{operandSegmentSizes=array<i32:1,1,1,0,0>,
 static_offsets=array<i64:-9223372036854775808,0,0>,static_sizes=array<i64:1,{m},{n}>,
 static_strides=array<i64:1,1,1>}}> :({tensor},{history},index)->{history}
%pv=tensor.extract %p[]:tensor<i32>
%pn=arith.addi %pv,%one:i32
%p1=tensor.insert %pn into %p[]:tensor<i32>
scf.yield %next,%h1,%p1:{tensor},{history},tensor<i32>
}}
func.return %history,%state,%last,%input,%weight:{history},{tensor},tensor<i32>,{tensor},{tensor}
}}
}}"""


def wrap(values):
    return ((np.asarray(values, dtype=np.int64) + 128) % 256 - 128).astype("i1")


def capture(directory, shape):
    directory.mkdir()
    data = np.resize(np.array([-128, 127, -1, 0, 1, 91], dtype="i1"), shape)
    initial = np.resize(np.array([127, -128, 17, -53], dtype="i1"), shape)
    weight = np.resize(np.array([-128, -1, 0, 127, 33], dtype="i1"), shape)
    state, history, trace = initial.copy(), [], []
    for _ in range(3):
        mixed = wrap(data.astype(np.int64) + state.astype(np.int64))
        weighted = wrap(mixed.astype(np.int64) + weight.astype(np.int64))
        trace.append((0, weighted.copy()))
        state = wrap(weighted.astype(np.int64) + 1)
        history.append(state.copy())
    expected = (np.stack(history), state, np.array(10, "<i4"), data.copy(), weight.copy())
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
    np.savez(directory / "inputs.npz", in0=data, in1=initial, in2=np.array(7, "<i4"))
    (directory / "input_order.json").write_text(json.dumps({"input": 0, "state": 1, "position": 2}))
    np.save(directory / "golden.npy", expected[0])
    names = ("history", "state", "position", "input", "weight")
    np.savez(directory / "goldens.npz", **dict(zip(names, expected, strict=True)))
    (directory / "output_order.json").write_text(json.dumps(names))
    module = parse_mlir_text(model_source(shape))
    module.verify()
    (directory / "model.mlir").write_text(ir_text(module))
    (directory / "capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "materialized_abi": {"complete": True},
                "artifacts": {
                    p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in directory.iterdir()
                },
            }
        )
    )
    return expected, trace


def observe_program(image, data):
    """Independent fixed-word inverse/replay, not the selected effect decoder."""
    word = struct.Struct("<BBBQQQ")
    assert len(image) % word.size == 0
    count = data.size
    first, second = bytearray([99] * 200), bytearray([119] * 200)
    raw, output = bytearray([71] * 86), bytearray([91] * 86)
    raw[:count] = data.tobytes()
    memory = [first, second, raw, output]
    pending, seen, high, events = None, set(), [0, 0], []
    reads, writes, copies, overlap = 0, 0, 0, 0
    for offset in range(0, len(image), word.size):
        opcode, packed_read, write, source, destination, size = word.unpack_from(image, offset)
        read, token = packed_read & 3, packed_read >> 2
        assert opcode in (0, 1, 2, 3) and write < 4
        events.append(
            dict(opcode=opcode, read=read, write=write, source=source, destination=destination, size=size, token=token)
        )
        if opcode == 3:
            assert pending is not None and token == pending[-1]
            r, w, s, d, n, _ = pending
            memory[w][d : d + n] = memory[r][s : s + n]
            pending = None
            continue
        assert size > 0
        for bank, address in ((read, source), (write, destination)):
            assert address + size <= (192 if bank < 2 else count)
            if bank < 2:
                assert address >= (8 if bank == 0 else 16)
                high[bank] = max(high[bank], address + size)
        if pending is not None:
            r, w, s, d, n, _ = pending
            for bank, address, writing in ((read, source, False), (write, destination, True)):
                for previous_bank, previous_address, previous_write in ((r, s, False), (w, d, True)):
                    assert not (
                        bank == previous_bank
                        and (writing or previous_write)
                        and address < previous_address + n
                        and previous_address < address + size
                    )
            assert opcode in (0, 1)
            overlap += 1
        reads += size
        writes += size
        copies += opcode in (0, 2)
        if opcode == 2:
            assert pending is None and token not in seen and read == 2 and write < 2
            seen.add(token)
            pending = read, write, source, destination, size, token
        elif opcode == 0:
            assert read < 2 and write == 3
            memory[write][destination : destination + size] = memory[read][source : source + size]
        else:
            assert read == write and source == destination
            memory[write][destination : destination + size] = bytes(
                (v + 1) % 256 for v in memory[read][source : source + size]
            )
    assert pending is None and seen == {1, 2, 3}
    assert raw[:count] == data.tobytes() and raw[count:] == bytes([71] * (86 - count))
    assert output[:count] == wrap(data.astype(np.int64) + 1).tobytes()
    assert output[count:] == bytes([91] * (86 - count))
    assert first[:8] == bytes([99] * 8) and second[:16] == bytes([119] * 16)
    assert first[192:] == bytes([99] * 8) and second[192:] == bytes([119] * 8)
    return dict(
        commands=events,
        counters=[len(events), reads, writes],
        copies=copies,
        high_water=high,
        synchronous_while_pending=overlap,
    )


@pytest.mark.parametrize("shape", [(1, 47), (2, 23)])
def test_ordinary_recurrent_model_prefetches_before_work_with_complete_state(tmp_path, record_property, shape):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, shape)
    observations, outputs = [], []
    for arm, enabled in (("baseline", False), ("optimized", True)):
        provider = PrefetchProvider(tmp_path / (arm + "_provider"), enabled=enabled, separate=True)
        selection = provider.selection(trace)
        qualification_path = provider.work / "original_concurrency_qualification.json"
        qualification = json.loads(qualification_path.read_text())
        qualification_sha256 = sha256_file(qualification_path)
        assert qualification["identity"] == selection.prefetch_qualification_identity
        assert qualification["executor_source_sha256"] == hashlib.sha256(provider.executor_source.encode()).hexdigest()
        assert qualification["facts_sha256"] == sha256_file(provider.work / "capabilities" / "facts.json")
        assert qualification["software_sha256"] == sha256_file(provider.work / "capabilities" / "software.yaml")
        # Original caller review of this ranked wrapper: no retained pointers,
        # no caller releases, and completion before return. This is separate
        # from the numerical and concurrent-byte-machine qualification.
        selection = replace(
            selection,
            borrowed_effects=BorrowedBufferEffects("reviewed-neutral-prefetch-model-ranked-ABI-v1", True, True, True),
        )
        work = tmp_path / arm
        work.mkdir()
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
        compile_seconds = time.perf_counter() - started
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        assert sha256_file(qualification_path) == qualification_sha256
        (work / "console.bin").write_bytes(console)
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        assert len(actual) == len(expected) == 5
        for got, wanted in zip(actual, expected, strict=True):
            assert got.dtype == wanted.dtype and got.shape == wanted.shape and got.tobytes() == wanted.tobytes()
        outputs.append(actual)
        assert len(provider.records) == len(provider.results) == 1
        result, record = provider.results[0], provider.records[0]
        image, section = read_program_section(built["elf"], record["program_section"], max_bytes=27 * 32)
        assert not section["writable"] and image == b"".join(result.images())
        assert record["program_sha256"] == sha256_file(provider.work / "0" / "program.bin")
        observed = observe_program(image, trace[0][1])
        assert observed["counters"] == [12, 3 * np.prod(shape), 3 * np.prod(shape)]
        assert observed["copies"] == 6 and observed["synchronous_while_pending"] == (4 if enabled else 0)
        assert [row["opcode"] for row in observed["commands"]] == (
            [2, 3, 2, 1, 0, 3, 2, 1, 0, 3, 1, 0] if enabled else [2, 3, 1, 0] * 3
        )
        for i, row in enumerate(observed["commands"]):
            if row["opcode"] == 3:
                bad = bytearray(image)
                bad[i * 27 + 1] ^= 4
                with pytest.raises(AssertionError):
                    observe_program(bytes(bad), trace[0][1])
                break
        storage = result.compiled.report["storage"]
        assert storage["prefetch"]["status"] == ("applied" if enabled else "unchanged")
        assert len(storage["prefetch"]["moves"]) == (2 if enabled else 0)
        assert all(move["distance"] == 2 for move in storage["prefetch"]["moves"])
        native = json.loads((provider.work / "native_qualification.json").read_text())
        counters = [
            [int(x) for x in line.split()[1:]]
            for line in console.splitlines()
            if line.startswith(b"MERLIN_POINTWISE_COUNTS ")
        ]
        assert len(counters) == 3 and counters[-1] == native["counters"]
        cumulative = [0] * 10
        for row, (_, data) in zip(counters, trace, strict=True):
            reconstructed = observe_program(image, data)
            for i in range(3):
                cumulative[i] += reconstructed["counters"][i]
            cumulative[3] += 2 * data.size
            cumulative[4] += 2 * data.size
            cumulative[5] += 1
            cumulative[6:8] = [max(a, b) for a, b in zip(cumulative[6:8], reconstructed["high_water"], strict=True)]
            cumulative[8] += reconstructed["copies"]
            cumulative[9] += reconstructed["synchronous_while_pending"]
            assert row == cumulative
        receipts = list(work.rglob("source_pointwise.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert len(receipt["bindings"]) == 1 and receipt["selected_sha256"] == sha256_file(
            receipts[0].with_suffix(".mlir")
        )
        recipe = json.loads((work / "compilation_recipe.json").read_text())
        observations.append(
            dict(
                arm=arm,
                budgets=asdict(selection.budgets),
                compile_seconds=compile_seconds,
                compilation_memory=memory,
                elf_sha256=sha256_file(built["elf"]),
                elf_bytes=built["elf"].stat().st_size,
                elf_sections=FB.elf_sections(built["elf"]),
                compiler_commands=len(recipe["commands"]),
                host_object_sha256=sha256_file(work / "model.o"),
                provider_object_bytes=(work / "host_provider" / "provider.o").stat().st_size,
                image_bytes=len(image),
                linked_section=section,
                actual_program=observed,
                actual_counter_trace=counters,
                native_qualification=native,
                source_receipt=receipt,
                facts_sha256=sha256_file(provider.work / "capabilities" / "facts.json"),
                software_sha256=sha256_file(provider.work / "capabilities" / "software.yaml"),
                library_sources=dict(result.library_sources),
                original_qualification=qualification,
                original_qualification_sha256=qualification_sha256,
                functional_forward_instructions=metadata["metrics"]["cycles"],
                simulator_process_wall_seconds=wall,
                complete_output_agreement="all five outputs and recurrent state match independent modular reference",
                physical_traffic_bytes=None,
                host_accelerator_component_seconds=None,
                scope=(
                    "neutral host byte interpreter and actual pending-window counters; "
                    "no accelerator latency or speedup claim"
                ),
            )
        )
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    baseline, optimized = observations
    assert {key for key, value in baseline["budgets"].items() if optimized["budgets"][key] != value} == {
        "prefetch_distance"
    }
    for key in (
        "facts_sha256",
        "software_sha256",
        "library_sources",
        "host_object_sha256",
        "image_bytes",
        "original_qualification_sha256",
    ):
        assert baseline[key] == optimized[key]
    for key in ("source_sha256", "selected_sha256"):
        assert baseline["source_receipt"][key] == optimized["source_receipt"][key]
    assert baseline["actual_counter_trace"][-1][:6] == optimized["actual_counter_trace"][-1][:6]
    assert baseline["actual_counter_trace"][-1][8] == optimized["actual_counter_trace"][-1][8] == 18
    assert baseline["actual_counter_trace"][-1][9] == 0 and optimized["actual_counter_trace"][-1][9] == 12
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))
    record_property("prefetch_complete_model_pair", json.dumps(observations))
