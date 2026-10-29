"""Whole-row source scheduling in an ordinary recurrent attention model.

Neutral C primitives execute on the host and functional Spike. They establish
source/model mechanism scope, never accelerator support or latency cycles.
"""

import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, replace

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_dense_source_catalog import compilation_memory
from test_row_softmax import exp_object as exp_object  # noqa: F401
from xdsl.dialects import func
from xdsl.rewriter import Rewriter

from merlin.common import compile_trace
from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.cpu_softmax import CPU_SOFTMAX_F32_AVX2_POLICY as POLICY
from merlin.llvmlower.cpu_softmax import build_cpu_softmax_lastdim
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike, spike_model
from merlin.targetgen.elf_lanes import read_program_section
from merlin.transforms.certificates import ir_text


def contraction(name, left, right, m, n, k, *, transposed=False):
    rhs = "(j,k)" if transposed else "(k,j)"
    rshape = f"{n}x{k}" if transposed else f"{k}x{n}"
    return f"""
 %{name}_empty=tensor.empty():tensor<{m}x{n}xf32>
 %{name}_zero=linalg.fill ins(%zero:f32) outs(%{name}_empty:tensor<{m}x{n}xf32>)->tensor<{m}x{n}xf32>
 %{name}=linalg.generic {{indexing_maps=[affine_map<(i,j,k)->(i,k)>,affine_map<(i,j,k)->{rhs}>,
 affine_map<(i,j,k)->(i,j)>],iterator_types=["parallel","parallel","reduction"]}}
 ins(%{left},%{right}:tensor<{m}x{k}xf32>,tensor<{rshape}xf32>)
 outs(%{name}_zero:tensor<{m}x{n}xf32>) {{^bb0(%x:f32,%y:f32,%acc:f32):
 %product=arith.mulf %x,%y:f32
 %sum=arith.addf %acc,%product:f32
 linalg.yield %sum:f32
 }}->tensor<{m}x{n}xf32>
 """


def model_text(columns, depth=4):
    return f"""module {{
 func.func private @row_placeholder(tensor<2x{columns}xf32>)->tensor<2x{columns}xf32>
 func.func @forward(%new_key:tensor<1x{depth}xf32>,%new_value:tensor<1x{depth}xf32>,
 %query:tensor<2x{depth}xf32>,%keys:tensor<{columns}x{depth}xf32>,%values:tensor<{columns}x{depth}xf32>,
 %initial:tensor<2x{depth}xf32>,%position:tensor<i32>)
 ->(tensor<3x2x{depth}xf32>,tensor<2x{depth}xf32>,tensor<3x2x{columns}xf32>,tensor<3x2x{columns}xf32>,
 tensor<{columns}x{depth}xf32>,tensor<{columns}x{depth}xf32>,tensor<i32>,tensor<2x{depth}xf32>,tensor<1x{depth}xf32>) {{
 %zero=arith.constant 0.0:f32
 %one=arith.constant 1:i32
 %n=arith.constant {columns}:i32
 %c0=arith.constant 0:index
 %c1=arith.constant 1:index
 %c3=arith.constant 3:index
 %he=tensor.empty():tensor<3x2x{depth}xf32>
 %hz=linalg.fill ins(%zero:f32) outs(%he:tensor<3x2x{depth}xf32>)->tensor<3x2x{depth}xf32>
 %pe=tensor.empty():tensor<3x2x{columns}xf32>
 %pz=linalg.fill ins(%zero:f32) outs(%pe:tensor<3x2x{columns}xf32>)->tensor<3x2x{columns}xf32>
 %se=tensor.empty():tensor<3x2x{columns}xf32>
 %sz=linalg.fill ins(%zero:f32) outs(%se:tensor<3x2x{columns}xf32>)->tensor<3x2x{columns}xf32>
 %state,%kh,%vh,%last,%history,%probs,%scores_history=scf.for %step=%c0 to %c3 step %c1
 iter_args(%s=%initial,%kc=%keys,%vc=%values,%p=%position,%h=%hz,%ph=%pz,%sh=%sz)
 ->(tensor<2x{depth}xf32>,tensor<{columns}x{depth}xf32>,tensor<{columns}x{depth}xf32>,tensor<i32>,
 tensor<3x2x{depth}xf32>,tensor<3x2x{columns}xf32>,tensor<3x2x{columns}xf32>) {{
 %pv=tensor.extract %p[]:tensor<i32>
 %slot32=arith.remui %pv,%n:i32
 %slot=arith.index_cast %slot32:i32 to index
 %ku=tensor.insert_slice %new_key into %kc[%slot,0] [1,{depth}] [1,1]
 :tensor<1x{depth}xf32> into tensor<{columns}x{depth}xf32>
 %vu=tensor.insert_slice %new_value into %vc[%slot,0] [1,{depth}] [1,1]
 :tensor<1x{depth}xf32> into tensor<{columns}x{depth}xf32>
 %qe=tensor.empty():tensor<2x{depth}xf32>
 %q=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%query,%s:tensor<2x{depth}xf32>,tensor<2x{depth}xf32>) outs(%qe:tensor<2x{depth}xf32>) {{
 ^bb0(%x:f32,%y:f32,%unused:f32): %sum=arith.addf %x,%y:f32
 linalg.yield %sum:f32
 }}->tensor<2x{depth}xf32>
 {contraction("scores", "q", "ku", 2, columns, depth, transposed=True)}
 %prob=func.call @row_placeholder(%scores):(tensor<2x{columns}xf32>)->tensor<2x{columns}xf32>
 {contraction("context", "prob", "vu", 2, depth, columns)}
 %h1=tensor.insert_slice %context into %h[%step,0,0] [1,2,{depth}] [1,1,1]
 :tensor<2x{depth}xf32> into tensor<3x2x{depth}xf32>
 %ph1=tensor.insert_slice %prob into %ph[%step,0,0] [1,2,{columns}] [1,1,1]
 :tensor<2x{columns}xf32> into tensor<3x2x{columns}xf32>
 %sh1=tensor.insert_slice %scores into %sh[%step,0,0] [1,2,{columns}] [1,1,1]
 :tensor<2x{columns}xf32> into tensor<3x2x{columns}xf32>
 %pn=arith.addi %pv,%one:i32
 %p1=tensor.insert %pn into %p[]:tensor<i32>
 scf.yield %context,%ku,%vu,%p1,%h1,%ph1,%sh1:tensor<2x{depth}xf32>,tensor<{columns}x{depth}xf32>,
 tensor<{columns}x{depth}xf32>,tensor<i32>,tensor<3x2x{depth}xf32>,tensor<3x2x{columns}xf32>,tensor<3x2x{columns}xf32>
 }}
 return %history,%state,%probs,%scores_history,%kh,%vh,%last,%query,%new_key:
 tensor<3x2x{depth}xf32>,tensor<2x{depth}xf32>,tensor<3x2x{columns}xf32>,tensor<3x2x{columns}xf32>,
 tensor<{columns}x{depth}xf32>,tensor<{columns}x{depth}xf32>,tensor<i32>,tensor<2x{depth}xf32>,tensor<1x{depth}xf32>
 }} }}"""


def ordered_product(left, right):
    out = np.empty((left.shape[0], right.shape[1]), dtype="<f4")
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            value = np.float32(0)
            for k in range(left.shape[1]):
                product = np.float32(left[i, k] * right[k, j])
                value = np.float32(value + product)
            out[i, j] = value
    return out


def independent_softmax(values, root, name):
    from merlin.llvmlower import toolchain

    source, output = root / (name + ".npy"), root / (name + "-softmax.npy")
    np.save(source, values)
    result = subprocess.run(
        [
            str(toolchain.compiler_python()),
            "-c",
            """
import sys,torch,numpy as np
if not torch.__version__.startswith('2.10.') or torch.backends.cpu.get_cpu_capability()!='AVX2':sys.exit(77)
x=np.load(sys.argv[1]);np.save(sys.argv[2],torch.softmax(torch.from_numpy(x),-1).numpy())
""",
            str(source),
            str(output),
        ],
        env={**os.environ, "ATEN_CPU_CAPABILITY": "avx2"},
        capture_output=True,
        text=True,
    )
    if result.returncode == 77:
        pytest.skip("independent pinned Torch2.10 AVX2 unavailable")
    assert result.returncode == 0, result.stderr
    return np.load(output)


def reference(weights, inputs, root):
    query, keys, values, state, position = (np.array(v, copy=True) for v in inputs)
    history, norms, scores, trace = [], [], [], []
    for step in range(3):
        slot = int(position) % keys.shape[0]
        keys[slot] = weights["new_key"][0]
        values[slot] = weights["new_value"][0]
        current = np.asarray(query + state, dtype="<f4")
        score = ordered_product(current, keys.T)
        assert np.all(np.isfinite(score))
        prob = independent_softmax(score, root, "step" + str(step))
        trace.append((score.copy(), prob.copy()))
        state = ordered_product(prob, values)
        position = np.asarray(int(position) + 1, dtype="<i4")
        history.append(state.copy())
        norms.append(prob)
        scores.append(score)
    return (
        np.stack(history),
        state,
        np.stack(norms),
        np.stack(scores),
        keys,
        values,
        position,
        query,
        weights["new_key"].copy(),
    ), trace


def capture(directory, columns):
    directory.mkdir()
    rng = np.random.default_rng(672)
    weights = {
        "new_key": (rng.integers(-8, 9, (1, 4)) / 16).astype("<f4"),
        "new_value": (rng.integers(-8, 9, (1, 4)) / 16).astype("<f4"),
    }
    inputs = [(rng.integers(-8, 9, shape) / 16).astype("<f4") for shape in ((2, 4), (columns, 4), (columns, 4), (2, 4))]
    inputs.append(np.asarray(columns - 2, dtype="<i4"))
    expected, trace = reference(weights, inputs, directory)
    save_file(weights, directory / "weights.safetensors")
    manifest = {"0": dict(kind="param", weight="new_key"), "1": dict(kind="param", weight="new_value")}
    manifest.update(
        {
            str(i + 2): dict(kind="input", name=name)
            for i, name in enumerate(("query", "keys", "values", "state", "position"))
        }
    )
    (directory / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    np.savez(directory / "inputs.npz", **{f"in{i}": value for i, value in enumerate(inputs)})
    (directory / "input_order.json").write_text(
        json.dumps({name: i for i, name in enumerate(("query", "keys", "values", "state", "position"))})
    )
    np.save(directory / "golden.npy", expected[0])
    source = directory / "model.mlir"
    source.write_text(model_text(columns))
    module = mq.parse(source).clone()
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
    return expected, trace


def reviewed_row_selection(provider, trace):
    config = provider.selection(trace)
    assert config.borrowed_effects is None
    # Original caller review: the private ranked row wrapper copies into
    # local float arrays, calls a synchronous scalar executor/exponential,
    # retains no descriptor pointers and frees no caller storage. Numerical
    # qualification alone does not supply these ABI lifetime premises.
    return replace(
        config,
        borrowed_effects=BorrowedBufferEffects(
            "reviewed-neutral-row-model-ABI-lifetime-v1",
            True,
            True,
            True,
        ),
    )


def build_model(bundle, work, provider, trace, *, config=None, device=None):
    if config is None:
        config = reviewed_row_selection(provider, trace)
    return spike_model.build(
        bundle,
        work,
        arena_mb=1,
        dram_bytes=64 << 20,
        code_reserve=16 << 20,
        backend="scalar",
        full_readback=True,
        shared_row_lowering=config,
        device=device,
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


@pytest.mark.parametrize("columns", [19, 33])
def test_ordinary_recurrent_attention_schedules_canonical_rows(tmp_path, record_property, exp_object, columns):
    from test_source_row_provider import RowProvider

    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, columns)
    observations, all_outputs = [], []
    for name, chunk in (("baseline", 1), ("optimized", None)):
        provider = RowProvider(tmp_path / (name + "_provider"), columns=columns, chunk=chunk, exp_object=exp_object)
        work = tmp_path / name
        work.mkdir()

        started = time.perf_counter()
        with compilation_memory() as memory:
            built = build_model(bundle, work, provider, trace)
        elapsed = time.perf_counter() - started
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        (work / "console.bin").write_bytes(console)
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        assert len(actual) == len(expected) == 9
        for got, want in zip(actual, expected, strict=True):
            if want.dtype.kind == "f":
                np.testing.assert_array_equal(got.view("<u4"), want.view("<u4"))
            else:
                np.testing.assert_array_equal(got, want)
        all_outputs.append(actual)
        native = json.loads((provider.work / "native_qualification.json").read_text())
        counts = [
            [int(word) for word in line.split()[1:]]
            for line in console.splitlines()
            if line.startswith(b"MERLIN_ROW_COUNTS ")
        ]
        assert len(counts) == 3
        for invocation, actual_counts in enumerate(counts, 1):
            assert actual_counts == [value * invocation // 3 for value in native["counters"]]
        report = provider.result.compiled.report
        # This is actual retained executable C kernel text, not a made-up
        # instruction buffer or an encoding conformance claim for a device.
        kernel, section = read_program_section(built["elf"], ".merlin_row.text", max_bytes=64 << 10)
        assert not section["writable"]
        (work / "linked-row-kernel.bin").write_bytes(kernel)
        provider.result.revalidate()
        receipt = json.loads((work / "host_provider" / "host_provider.json").read_text())
        observations.append(
            dict(
                arm=name,
                columns=columns,
                chunk=chunk,
                compile_seconds=elapsed,
                compilation_memory=memory,
                elf_bytes=built["elf"].stat().st_size,
                elf_sha256=sha256_file(built["elf"]),
                host_object_sha256=sha256_file(work / "model.o"),
                sections=FB.elf_sections(built["elf"]),
                kernel_section={**section, "sha256": hashlib.sha256(kernel).hexdigest()},
                native_qualification=native,
                actual_model_counters=counts[-1],
                row_schedule=report,
                source_passes=[asdict(p) for p in provider.result.source_passes],
                functional_forward_count=metadata["metrics"]["cycles"],
                counter_unit="functional Spike instructions; not accelerator latency",
                simulator_process_wall_seconds=wall,
                provider_receipt=receipt,
                timing_scope="all primitives execute on host; no accelerator or paired hardware timing",
            )
        )
    for before, after in zip(*all_outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    before, after = observations
    assert before["host_object_sha256"] == after["host_object_sha256"]
    assert before["native_qualification"]["counters"][0] > after["native_qualification"]["counters"][0]
    assert before["native_qualification"]["counters"][3:] == after["native_qualification"]["counters"][3:]
    assert before["row_schedule"]["numerical_identity"] == after["row_schedule"]["numerical_identity"]
    for key in ("library_sources_sha256", "index_bits"):
        assert before["row_schedule"]["pipeline"][key] == after["row_schedule"]["pipeline"][key]
    for key in ("rtl_facts", "software_spec"):
        assert (
            before["row_schedule"]["pipeline"]["selected_contracts"][key]["sha256"]
            == (after["row_schedule"]["pipeline"]["selected_contracts"][key]["sha256"])
        )
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))
    record_property("source_row_whole_model_pair", json.dumps(observations))


def test_wrong_native_row_continuation_fails_before_model_link(tmp_path, exp_object):
    from test_source_row_provider import RowProvider

    bundle = tmp_path / "capture"
    _, trace = capture(bundle, 19)
    provider = RowProvider(tmp_path / "provider", columns=19, chunk=None, exp_object=exp_object, faulty=True)
    work = tmp_path / "model"
    work.mkdir()
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        build_model(bundle, work, provider, trace)
    assert not (provider.work / "native_qualification.json").exists()
    assert not (work / "model.elf").exists()


def test_unavailable_finite_premise_executes_complete_original_model_without_provider(
    tmp_path, exp_object, monkeypatch, record_property
):
    from test_source_row_provider import RowProvider

    from merlin.llvmlower import dealloc_lifetime, toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, 19)
    provider = RowProvider(tmp_path / "provider", columns=19, chunk=None, exp_object=exp_object)
    config = replace(provider.selection(trace), assume_finite_inputs=False)
    captured = {path.name: sha256_file(path) for path in bundle.iterdir()}
    observations, outputs = [], []
    for arm in ("baseline", "checked"):
        work = tmp_path / arm
        trace_root = tmp_path / (arm + "_trace")
        started = time.perf_counter()
        with (
            monkeypatch.context() as controls,
            compilation_memory() as memory,
            compile_trace.session(
                compile_trace.Request(str(trace_root), dump_after=("mlir:optimize-allocation-liveness",))
            ),
        ):
            controls.delenv("MERLIN_SINK_DEALLOCS", raising=False)
            for switch in ("MERLIN_SPLIT_FRESH_DEALLOCS", "MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT"):
                if arm == "baseline":
                    controls.setenv(switch, "0")
                else:
                    controls.delenv(switch, raising=False)
            built = build_model(bundle, work, provider, trace, config=config)
        elapsed = time.perf_counter() - started
        assert provider.result is None and not provider.c_source.exists()
        assert "host_provider" not in built and not (work / "host_provider").exists()
        recipe_path = work / "lower" / "lowering_recipe.json"
        recipe = json.loads(recipe_path.read_text())
        assert recipe["sources"]["loop_ownership_checker"]["sha256"] == sha256_file(dealloc_lifetime.__file__)
        for identity in recipe["sources"].values():
            assert sha256_file(identity["path"]) == identity["sha256"]
        transform_path = work / "prepared_model_transform" / "prepared_model_transform.json"
        transform = json.loads(transform_path.read_text())
        assert transform["source_sha256"] == transform["selected_sha256"]
        assert sha256_file(transform["source_path"]) == transform["source_sha256"]
        assert sha256_file(transform["selected_path"]) == transform["selected_sha256"]
        selection = json.loads((work / "prepared_model_transform" / "source_rows.json").read_text())
        assert selection["bindings"] == [] and selection["premises"]["assume_finite_inputs"] is False
        assert {path.name: sha256_file(path) for path in bundle.iterdir()} == captured
        stages = json.loads((trace_root / "trace.json").read_text())
        assert stages["outcome"] == compile_trace.COMPLETED
        snapshots = [
            dict(path=str(trace_root / filename), sha256=sha256_file(trace_root / filename))
            for stage in stages["stages"]
            if stage["kind"] == "native" and stage["stage"] == "mlir:optimize-allocation-liveness"
            for filename in stage["files"]
        ]
        assert bool(snapshots) == (arm == "checked")
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        console_path = work / "console.bin"
        console_path.write_bytes(console)
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        assert len(actual) == len(expected) == 9
        for got, want in zip(actual, expected, strict=True):
            assert got.shape == want.shape and got.dtype == want.dtype
            assert got.tobytes() == want.tobytes()
        outputs.append(actual)
        observations.append(
            dict(
                arm=arm,
                controls=("legacy allocation pipeline" if arm == "baseline" else "default allocation pipeline"),
                capture_sha256=captured,
                build_hash=built["build_hash"],
                compile_seconds=elapsed,
                compilation_memory=memory,
                console=dict(path=str(console_path), sha256=sha256_file(console_path)),
                elf=dict(
                    path=str(built["elf"]),
                    sha256=sha256_file(built["elf"]),
                    bytes=built["elf"].stat().st_size,
                    sections=FB.elf_sections(built["elf"]),
                ),
                recipe=dict(path=str(recipe_path), sha256=sha256_file(recipe_path)),
                source_selection=dict(path=str(transform_path), sha256=sha256_file(transform_path)),
                prepared_sha256=recipe["sources"]["prepared_mlir"]["sha256"],
                snapshots=snapshots,
                simulator_process_wall_seconds=wall,
                functional_forward_instructions=metadata["metrics"]["cycles"],
                output_sha256=[hashlib.sha256(value.tobytes()).hexdigest() for value in actual],
                scope="original CPU fallback and complete state; no provider, traffic or accelerator latency claim",
            )
        )
    for before, after in zip(*outputs, strict=True):
        assert before.tobytes() == after.tobytes()
    assert observations[0]["prepared_sha256"] == observations[1]["prepared_sha256"]
    (tmp_path / "fallback_pair.json").write_text(json.dumps(observations, indent=2))
    record_property("original_row_fallback_pair", json.dumps(observations))


@pytest.mark.parametrize("bad_value", ["inf", "nan"])
def test_actual_row_input_finite_guard_refuses_nonfinite_scores(tmp_path, exp_object, bad_value):
    """The island checks produced scores, not merely finite Q/K/V tensors."""
    import sys

    from test_source_row_provider import RowProvider, qualify

    bundle = tmp_path / "capture"
    _, trace = capture(bundle, 19)
    provider = RowProvider(tmp_path / "provider", columns=19, chunk=None, exp_object=exp_object)
    provider.prepare(bundle / "model.mlir", tmp_path)
    witness = qualify(provider, trace)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            """
import ctypes,sys
class Descriptor(ctypes.Structure):
 _fields_=[('allocated',ctypes.c_void_p),('aligned',ctypes.c_void_p),('offset',ctypes.c_int64),
           ('sizes',ctypes.c_int64*2),('strides',ctypes.c_int64*2)]
a=(ctypes.c_float*38)(*[0]*38);a[17]=float(sys.argv[2]);b=(ctypes.c_float*38)()
x=Descriptor(ctypes.addressof(a),ctypes.addressof(a),0,(ctypes.c_int64*2)(2,19),(ctypes.c_int64*2)(19,1))
y=Descriptor(ctypes.addressof(b),ctypes.addressof(b),0,(ctypes.c_int64*2)(2,19),(ctypes.c_int64*2)(19,1))
f=getattr(ctypes.CDLL(sys.argv[1]),sys.argv[3])
f.argtypes=[ctypes.POINTER(Descriptor)]*2;f(ctypes.byref(x),ctypes.byref(y))
""",
            str(provider.work / witness["library_path"]),
            bad_value,
            provider.symbol,
        ],
        capture_output=True,
    )
    # A native synchronous trap in a separate process, never a pytest crash.
    assert result.returncode == -signal.SIGILL
    (tmp_path / "nonfinite-refusal.json").write_text(
        json.dumps(dict(value=bad_value, returncode=result.returncode, source_sha256=witness["source_sha256"]))
    )
