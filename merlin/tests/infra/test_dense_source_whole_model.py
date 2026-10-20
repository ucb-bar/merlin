"""Source-derived dense scheduling through real ordinary catalog compilation."""

import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_dense_source import contraction
from test_dense_source_catalog import Catalog, compilation_memory, digest, qualify_native
from test_selected_dense_contracts import inputs

from merlin.compile.baremetal_model import compile_saved_model
from merlin.llvmlower.device_build import DeviceRouting
from merlin.model_check import full_build as FB
from merlin.model_check.admission import service


def model_source(m, k, n, dtype):
    first = contraction((m, k, n), (dtype, dtype, "i32"), lhs="a", rhs="w0", output="first", tag="first_source")
    second = contraction(
        (m, n, k), (dtype, dtype, "i32"), lhs="activation", rhs="w1", output="second", tag="second_source"
    )
    return f"""module {{
func.func @forward(%w0:tensor<{k}x{n}x{dtype}>, %w1:tensor<{n}x{k}x{dtype}>,
 %input:tensor<{m}x{k}x{dtype}>, %initial:tensor<{m}x{k}x{dtype}>, %position:tensor<i32>)
 -> (tensor<3x{m}x{k}x{dtype}>,tensor<{m}x{k}x{dtype}>,tensor<i32>,tensor<{m}x{k}x{dtype}>) {{
 %zero=arith.constant 0 : {dtype}
 %zero32=arith.constant 0 : i32
 %one32=arith.constant 1 : i32
 %c0=arith.constant 0:index
 %c1=arith.constant 1:index
 %c3=arith.constant 3:index
 %he=tensor.empty():tensor<3x{m}x{k}x{dtype}>
 %hz=linalg.fill ins(%zero:{dtype}) outs(%he:tensor<3x{m}x{k}x{dtype}>)->tensor<3x{m}x{k}x{dtype}>
 %state,%history,%lastpos=scf.for %step=%c0 to %c3 step %c1
    iter_args(%s=%initial,%h=%hz,%p=%position) ->
    (tensor<{m}x{k}x{dtype}>,tensor<3x{m}x{k}x{dtype}>,tensor<i32>) {{
  %ae=tensor.empty():tensor<{m}x{k}x{dtype}>
  %a=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
    affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
    ins(%input,%s:tensor<{m}x{k}x{dtype}>,tensor<{m}x{k}x{dtype}>)
    outs(%ae:tensor<{m}x{k}x{dtype}>) {{^bb0(%x:{dtype},%y:{dtype},%unused:{dtype}):
      %v=arith.addi %x,%y:{dtype}
      linalg.yield %v:{dtype}
    }}->tensor<{m}x{k}x{dtype}>
  {first}
  %acte=tensor.empty():tensor<{m}x{n}x{dtype}>
  %activation=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
    iterator_types=["parallel","parallel"]}} ins(%first:tensor<{m}x{n}xi32>)
    outs(%acte:tensor<{m}x{n}x{dtype}>) {{^bb0(%v:i32,%unused:{dtype}):
      %positive=arith.cmpi sgt,%v,%zero32:i32
      %clamp=arith.select %positive,%v,%zero32:i32
      %narrow=arith.trunci %clamp:i32 to {dtype}
      linalg.yield %narrow:{dtype}
    }}->tensor<{m}x{n}x{dtype}>
  {second}
  %se=tensor.empty():tensor<{m}x{k}x{dtype}>
  %next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->()>,
    affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
    ins(%second,%p:tensor<{m}x{k}xi32>,tensor<i32>) outs(%s:tensor<{m}x{k}x{dtype}>) {{
    ^bb0(%v:i32,%pos:i32,%unused:{dtype}):
      %biased=arith.addi %v,%pos:i32
      %narrow=arith.trunci %biased:i32 to {dtype}
      linalg.yield %narrow:{dtype}
    }}->tensor<{m}x{k}x{dtype}>
  %history1=tensor.insert_slice %next into %h[%step,0,0] [1,{m},{k}] [1,1,1]
    :tensor<{m}x{k}x{dtype}> into tensor<3x{m}x{k}x{dtype}>
  %pv=tensor.extract %p[]:tensor<i32>
  %pn=arith.addi %pv,%one32:i32
  %p1=tensor.insert %pn into %p[]:tensor<i32>
  scf.yield %next,%history1,%p1:tensor<{m}x{k}x{dtype}>,tensor<3x{m}x{k}x{dtype}>,tensor<i32>
 }}
 return %history,%state,%lastpos,%input:tensor<3x{m}x{k}x{dtype}>,
   tensor<{m}x{k}x{dtype}>,tensor<i32>,tensor<{m}x{k}x{dtype}>
}} }}"""


def reference(w0, w1, data, state, position):
    history = []
    for _ in range(3):
        a = np.add(data, state, dtype=state.dtype)
        first = (a.astype(np.int64) @ w0.astype(np.int64)).astype(np.int32)
        activation = np.maximum(first, 0).astype(state.dtype)
        second = (activation.astype(np.int64) @ w1.astype(np.int64)).astype(np.int32)
        state = np.add(second, np.int32(position), dtype=np.int32).astype(state.dtype)
        history.append(state.copy())
        position += 1
    return np.stack(history), state, np.array(position, np.int32), data.copy()


@pytest.mark.parametrize("tile,width,shared", [((2, 3, 4), 1, False), ((3, 2, 5), 2, True)])
def test_source_scheduling_runs_through_complete_ordinary_model(
    tmp_path, monkeypatch, record_property, tile, width, shared, group_profile=False
):
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike, spike_model
    from merlin.targetgen.rtl.facts import observed_facts

    if not toolchain.available() or not spike.available():
        pytest.skip("selected native lowering and functional simulator required")
    dtype = "i8" if width == 1 else "i16"
    npdtype = np.int8 if width == 1 else np.int16

    def declare(facts, software, dense):
        facts["datapaths"] = [
            dict(name="input", dtype="int8" if width == 1 else "int16"),
            dict(name="accumulator", dtype="int32"),
        ]

    model = inputs(tmp_path, tile=tile, widths=(width, width, 4), shared=shared, shape=(5, 14, 13), mutate=declare)
    selected, _, _, _ = model
    root, capture, host, package = (tmp_path / name for name in ("out", "capture", "host", "device"))
    for p in (root, capture, host, package):
        p.mkdir()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    (package / "README").write_text("Neutral interpreted byte machine for a shared compiler integration test.\n")
    (host / "manifest.yaml").write_text(
        yaml.safe_dump(
            dict(
                target="host",
                run_id="dense-source-test",
                family="scalar_linalg",
                authoring=dict(mode="deterministic_generated_from_spec", generated_by_agent=False),
                outputs=dict(knobs="knobs.yaml"),
            )
        )
    )
    (host / "knobs.yaml").write_text(
        yaml.safe_dump(
            dict(
                backend="scalar",
                dtype_strategy="fp32",
                cflags=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
            )
        )
    )
    dts, catalog = tmp_path / "cpu.dts", tmp_path / "boards.yaml"
    dts.write_text('cpu@0 { device_type = "cpu"; riscv,isa = "rv64imafdc_zicsr_zifencei"; };\n')
    catalog.write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                boards=dict(
                    selected=dict(
                        target="neutral",
                        dram_bytes=64 << 20,
                        dram_base=spike_model.DRAM_BASE,
                        harts=1,
                        console="htif",
                        flow="baremetal",
                        code_reserve=16 << 20,
                        host_dts_sha256=digest(dts),
                    )
                ),
            )
        )
    )
    m, k, n = 5, 14, 13
    rng = np.random.default_rng(94)
    bound = 128 if width == 1 else 32768
    w0 = rng.integers(-bound, bound, size=(k, n), dtype=npdtype)
    w1 = rng.integers(-bound, bound, size=(n, k), dtype=npdtype)
    data = rng.integers(-bound, bound, size=(m, k), dtype=npdtype)
    state = rng.integers(-bound, bound, size=(m, k), dtype=npdtype)
    expected = reference(w0, w1, data, state, 7)
    raw0, raw1 = w0.tobytes(), w1.tobytes()
    fields = {
        name: dict(dtype="I8" if width == 1 else "I16", shape=list(array.shape), data_offsets=offsets)
        for name, array, offsets in [("w0", w0, [0, len(raw0)]), ("w1", w1, [len(raw0), len(raw0) + len(raw1)])]
    }
    header = json.dumps(fields).encode()
    (capture / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw0 + raw1)
    (capture / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": dict(kind="param", weight="w0"),
                "1": dict(kind="param", weight="w1"),
                "2": dict(kind="input", name="input"),
                "3": dict(kind="input", name="state"),
                "4": dict(kind="input", name="position"),
            }
        )
    )
    np.savez(capture / "inputs.npz", in0=data, in1=state, in2=np.array(7, np.int32))
    np.save(capture / "golden.npy", expected[0])
    (capture / "input_order.json").write_text('{"input":0,"state":1,"position":2}')
    (capture / "model.mlir").write_text(model_source(m, k, n, dtype))
    artifacts = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in capture.iterdir()}
    (capture / "capture_receipt.json").write_text(
        json.dumps(dict(schema="m2m.capture-receipt.v1", materialized_abi=dict(complete=True), artifacts=artifacts))
    )
    observations = []
    outputs = []
    with observed_facts(selected.target, json.loads(selected.facts_bytes)):
        for name, streaming in [("baseline", True), ("optimized", False)]:
            provider = Catalog(model, streaming=streaming)
            routing = DeviceRouting(
                selected.target,
                package,
                dtype,
                "i32",
                capture=capture,
                catalog_builder=provider,
                post_offload_transform=provider.fresh_writers,
                linked_elf_admission=service(selected.target),
            )
            started = time.perf_counter()
            with compilation_memory() as memory:
                receipt = compile_saved_model(
                    capture=capture,
                    package=host,
                    board_catalog=catalog,
                    board="selected",
                    dts=dts,
                    output=root / name,
                    target=selected.target,
                    run="none",
                    device=routing,
                    arena_mb=1,
                    readback="full",
                    group_profile=group_profile,
                    respect_captured_quantization_scope=True,
                    host_llvm_bitcode=True,
                )
            elapsed = time.perf_counter() - started
            elf = Path(receipt["output"]["elf"])
            console, wall = FB.run_host(elf, timeout_s=30)
            (root / name / "console.bin").write_bytes(console)
            actual, metadata = FB.read(console, capture, receipt["output"]["build_hash"])
            for found, want in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(found, want)
            if group_profile:
                from merlin.runtime.out_bin import binary_console_diagnostics
                from merlin.runtime.whole_model_readback import group_names, parse_group_profile

                profile = parse_group_profile(binary_console_diagnostics(console).decode())
                sidecar = json.loads((elf.parent / "device_signatures.json").read_text())
                rows = sidecar["routed"]
                assert profile["calls"] == 6 and profile["dropped"] == 0
                expected_names = [name for name, _ in group_names(rows)] + [r["symbol"] for r in rows] * 2
                assert [r["name"] for r in profile["groups"]] == expected_names
                assert all(r["cycles"] > 0 for r in profile["groups"])
                (root / name / "group_profile_observation.json").write_text(json.dumps(profile, indent=2))
            outputs.append(actual)
            assert len(provider.results) == 2
            qualification, history, carried, lastpos = qualify_native(provider, w0, w1, data, state, 7, root / name)
            for got, want in zip((history, carried, np.array(lastpos, np.int32)), expected[:3], strict=True):
                np.testing.assert_array_equal(got, want)
            observations.append(
                dict(
                    arm=name,
                    native_qualification=qualification,
                    compile_seconds=elapsed,
                    compilation_memory=memory,
                    elf_sha256=digest(elf),
                    elf_bytes=elf.stat().st_size,
                    host_object_sha256=digest(elf.parent / "model.o"),
                    functional_forward_count=metadata["metrics"]["cycles"],
                    counter_unit="functional Spike counter; not accelerator latency",
                    whole_model_requested_transfer_bytes="not measured",
                    per_group_counters=metadata.get("group_profile", "not selected"),
                    sections=FB.elf_sections(elf),
                    provider_compiler_commands=provider.commands,
                    simulator_process_wall_seconds=wall,
                    source_scheduling=[
                        dict(choice=asdict_result(r.compiled.choice), realization=r.compiled.realization)
                        for r, *_ in provider.results
                    ],
                )
            )
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    assert observations[0]["host_object_sha256"] == observations[1]["host_object_sha256"]
    before, after = [
        sum(row["requested_load_bytes"] for row in arm["native_qualification"]["counters"]) for arm in observations
    ]
    assert after < before
    (root / "pair.json").write_text(json.dumps(observations, indent=2))
    record_property("source_dense_model_pair", json.dumps(observations))


def asdict_result(value):
    from dataclasses import asdict

    return asdict(value)
