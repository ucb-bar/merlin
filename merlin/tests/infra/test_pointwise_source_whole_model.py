"""Ordinary saved-model preparation/link/execution with checked source fusion.

The accelerator here is explicitly a neutral interpreted byte machine, running
on the host CPU. Functional counters are not accelerator latency. Complete
model arithmetic, packing, allocation and recurrent state stay in the test.
"""

import json
import struct
import time

import numpy as np
import pytest
from test_dense_source import contraction
from test_dense_source_catalog import compilation_memory
from test_pointwise_source_provider import Provider, qualify

from merlin.common.digest import sha256_file
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike, spike_model
from merlin.targetgen.elf_lanes import read_program_section


def epilogue(name, left, bias, m, n):
    return f"""
 %{name}_init=tensor.empty():tensor<{m}x{n}xi8>
 %{name}=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%{left},%{bias}:tensor<{m}x{n}xi32>,tensor<{m}x{n}xi32>)
 outs(%{name}_init:tensor<{m}x{n}xi8>) {{^bb0(%x:i32,%b:i32,%unused:i8):
 %sum=arith.addi %x,%b:i32
 %low=arith.constant -128:i32
 %high=arith.constant 127:i32
 %lower=arith.maxsi %sum,%low:i32
 %bounded=arith.minsi %lower,%high:i32
 %result=arith.trunci %bounded:i32 to i8
 linalg.yield %result:i8
 }}->tensor<{m}x{n}xi8>
 """


def model_source(m, k, n):
    first = contraction((m, k, n), ("i8", "i8", "i32"), lhs="a", rhs="w0", output="first")
    second = contraction((m, n, k), ("i8", "i8", "i32"), lhs="activation", rhs="w1", output="second")
    return f"""module {{
 func.func @forward(%w0:tensor<{k}x{n}xi8>,%w1:tensor<{n}x{k}xi8>,
 %b0:tensor<{m}x{n}xi32>,%b1:tensor<{m}x{k}xi32>,%input:tensor<{m}x{k}xi8>,
 %initial:tensor<{m}x{k}xi8>,%position:tensor<i32>)
 ->(tensor<3x{m}x{k}xi8>,tensor<{m}x{k}xi8>,tensor<i32>,tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) {{
 %zero=arith.constant 0:i8
 %one=arith.constant 1:i32
 %c0=arith.constant 0:index
 %c1=arith.constant 1:index
 %c3=arith.constant 3:index
 %he=tensor.empty():tensor<3x{m}x{k}xi8>
 %hz=linalg.fill ins(%zero:i8) outs(%he:tensor<3x{m}x{k}xi8>)->tensor<3x{m}x{k}xi8>
 %state,%history,%last=scf.for %step=%c0 to %c3 step %c1
 iter_args(%s=%initial,%h=%hz,%p=%position)->(tensor<{m}x{k}xi8>,tensor<3x{m}x{k}xi8>,tensor<i32>) {{
 %ae=tensor.empty():tensor<{m}x{k}xi8>
 %a=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%input,%s:tensor<{m}x{k}xi8>,tensor<{m}x{k}xi8>) outs(%ae:tensor<{m}x{k}xi8>) {{
 ^bb0(%x:i8,%y:i8,%unused:i8): %v=arith.addi %x,%y:i8
 linalg.yield %v:i8
 }}->tensor<{m}x{k}xi8>
 {first}
 {epilogue("activation", "first", "b0", m, n)}
 {second}
 {epilogue("next", "second", "b1", m, k)}
 %h1=tensor.insert_slice %next into %h[%step,0,0] [1,{m},{k}] [1,1,1]
 :tensor<{m}x{k}xi8> into tensor<3x{m}x{k}xi8>
 %pv=tensor.extract %p[]:tensor<i32>
 %pn=arith.addi %pv,%one:i32
 %p1=tensor.insert %pn into %p[]:tensor<i32>
 scf.yield %next,%h1,%p1:tensor<{m}x{k}xi8>,tensor<3x{m}x{k}xi8>,tensor<i32>
 }}
 return %history,%state,%last,%input,%w0:tensor<3x{m}x{k}xi8>,tensor<{m}x{k}xi8>,tensor<i32>,
 tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>
 }} }}"""


def reference(w0, w1, b0, b1, data, initial, position):
    """Independent wide arithmetic, wrapping before signed clamp and narrowing."""
    state, history, calls = initial.copy(), [], []
    for _ in range(3):
        a = ((data.astype(np.int64) + state.astype(np.int64) + 128) % 256 - 128).astype("i1")
        first = (a.astype(np.int64) @ w0.astype(np.int64)).astype("<i4")
        calls.append((0, first, b0))
        wide = first.astype(np.int64) + b0.astype(np.int64)
        activation = np.clip((wide + (1 << 31)) % (1 << 32) - (1 << 31), -128, 127).astype("i1")
        second = (activation.astype(np.int64) @ w1.astype(np.int64)).astype("<i4")
        calls.append((1, second, b1))
        wide = second.astype(np.int64) + b1.astype(np.int64)
        state = np.clip((wide + (1 << 31)) % (1 << 32) - (1 << 31), -128, 127).astype("i1")
        history.append(state.copy())
    return (np.stack(history), state, np.array(position + 3, "<i4"), data.copy(), w0.copy()), calls


def capture(directory, shape):
    directory.mkdir()
    m, k, n = shape
    rng = np.random.default_rng(941)
    weights = {
        "w0": rng.integers(-128, 128, (k, n)).astype("i1"),
        "w1": rng.integers(-128, 128, (n, k)).astype("i1"),
        "b0": rng.integers(-128, 128, (m, n)).astype("<i4"),
        "b1": rng.integers(-128, 128, (m, k)).astype("<i4"),
    }
    weights["b0"].flat[::3] = (1 << 31) - 2
    weights["b1"].flat[1::3] = -(1 << 31) + 2
    data = rng.integers(-128, 128, (m, k)).astype("i1")
    initial = rng.integers(-128, 128, (m, k)).astype("i1")
    expected, trace = reference(**weights, data=data, initial=initial, position=7)
    header, chunks, offset = {}, [], 0
    for name, array in weights.items():
        raw = array.tobytes()
        header[name] = dict(
            dtype="I8" if array.dtype.itemsize == 1 else "I32",
            shape=list(array.shape),
            data_offsets=[offset, offset + len(raw)],
        )
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    (directory / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))
    manifest = {str(i): dict(kind="param", weight=name) for i, name in enumerate(weights)}
    manifest.update(
        {
            "4": dict(kind="input", name="input"),
            "5": dict(kind="input", name="state"),
            "6": dict(kind="input", name="position"),
        }
    )
    (directory / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    np.savez(directory / "inputs.npz", in0=data, in1=initial, in2=np.array(7, "<i4"))
    np.save(directory / "golden.npy", expected[0])
    (directory / "input_order.json").write_text('{"input":0,"state":1,"position":2}')
    (directory / "model.mlir").write_text(model_source(m, k, n))
    return expected, trace


@pytest.mark.parametrize("shape,separate", [((3, 7, 5), False), ((2, 5, 11), True)])
def test_ordinary_recurrent_model_runs_source_bound_fusion(
    tmp_path, record_property, shape, separate, provider_class=Provider, optimization="fusion"
):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, shape)
    observations, outputs = [], []
    for name, enabled in (("baseline", False), ("optimized", True)):
        provider = provider_class(tmp_path / (name + "_provider"), enabled=enabled, separate=separate)

        def prepare(source, work):
            selected = provider.prepare(source, work)
            qualified, _ = qualify(provider, trace)
            assert len(qualified) == 6
            return selected

        work = tmp_path / name
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
                prepared_model_transform=prepare,
                host_provider_builder=provider.build,
                cflags_override=[
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-O2",
                    "-ffreestanding",
                    "-fno-builtin",
                ],
                host_llvm_bitcode=True,
            )
        elapsed = time.perf_counter() - started
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        (work / "console.bin").write_bytes(console)
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)
        outputs.append(actual)
        assert len(provider.results) == 2
        native = json.loads((provider.work / "native_qualification.json").read_text())
        expected_commands_and_traffic = [
            3 * sum(result.conformance[key] for result in provider.results)
            for key in ("actual_commands", "requested_read_bytes", "requested_write_bytes")
        ]
        assert native["counters"][:3] == expected_commands_and_traffic
        # Two four-byte inputs and one one-byte output, copied both ways.
        host_pack_bytes = 3 * 9 * shape[0] * (shape[1] + shape[2])
        assert native["counters"][3:6] == [host_pack_bytes, host_pack_bytes, 6]
        if provider.observation_count > 6:
            state = provider.results[-1].conformance["configuration"]["known_final_state"]
            assert native["counters"][6:] == [state[key] for key in sorted(state)]
        recipe = json.loads((work / "compilation_recipe.json").read_text())
        linked_programs = []
        for record, result in zip(provider.records, provider.results, strict=True):
            image, section = read_program_section(
                built["elf"], record["program_section"], max_bytes=len(provider.expected_instructions()) * 41
            )
            assert not section["writable"]
            assert image == b"".join(result.images())
            selected = result.selected_contracts
            from merlin.targetgen.isa_dialect_gen import decode_instruction

            decoded = [
                decode_instruction(selected.isa, int.from_bytes(image[i : i + 41], "little"))[0]
                for i in range(0, len(image), 41)
            ]
            assert decoded == provider.expected_instructions()
            linked_programs.append(dict(**section, sha256=record["program_sha256"], instructions=decoded))
        observations.append(
            dict(
                arm=name,
                optimization=optimization,
                compile_seconds=elapsed,
                compilation_memory=memory,
                elf_sha256=sha256_file(built["elf"]),
                elf_bytes=built["elf"].stat().st_size,
                host_object_sha256=sha256_file(work / "model.o"),
                sections=FB.elf_sections(built["elf"]),
                compiler_commands=len(recipe["commands"]),
                native_qualification=native,
                functional_forward_count=metadata["metrics"]["cycles"],
                counter_unit="functional Spike instructions; not accelerator latency",
                simulator_process_wall_seconds=wall,
                encoded_program_bytes=sum(sum(map(len, result.images())) for result in provider.results),
                source_conformance=[result.conformance for result in provider.results],
                library_sources=dict(provider.results[0].library_sources),
                linked_programs=linked_programs,
                host_accelerator_timing="no accelerator; byte interpreter executes on host CPU",
            )
        )
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    before, after = observations
    assert before["host_object_sha256"] == after["host_object_sha256"]
    assert before["native_qualification"]["counters"][0] > after["native_qualification"]["counters"][0]
    eliminated = 12 * shape[0] * (shape[1] + shape[2]) if optimization == "fusion" else 0
    for counter in (1, 2):
        assert (
            before["native_qualification"]["counters"][counter] - after["native_qualification"]["counters"][counter]
            == eliminated
        )
    assert before["native_qualification"]["counters"][3:] == after["native_qualification"]["counters"][3:]
    assert before["library_sources"] == after["library_sources"]
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))
    record_property("source_pointwise_model_pair", json.dumps(observations))
