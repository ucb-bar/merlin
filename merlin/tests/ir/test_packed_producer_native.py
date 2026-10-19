"""Ordinary native execution of exact split/permuted integer producers."""

from math import prod
from pathlib import Path

import pytest
from test_packed_producer_destinations import program


def compilation_memory():
    """Sample current process and descendants, not cumulative child high water."""
    import os
    import threading
    from contextlib import contextmanager

    import psutil

    @contextmanager
    def measured():
        stop = threading.Event()
        result = dict(
            peak_tree_rss_bytes_sampled_100ms=0,
            samples=0,
            scope="process-tree RSS; shared pages counted per process; short peaks may be missed",
        )

        def sample():
            root = psutil.Process(os.getpid())
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.Error:
                processes = [root]
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except psutil.Error:
                    pass
            result["peak_tree_rss_bytes_sampled_100ms"] = max(result["peak_tree_rss_bytes_sampled_100ms"], rss)
            result["samples"] += 1

        def run():
            while not stop.is_set():
                sample()
                stop.wait(0.1)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield result
        finally:
            stop.set()
            thread.join()
            sample()

    return measured()


def text_section(path):
    """Read actual ELF64 LE .text bytes without a compiler/planner decoder."""
    import struct

    raw = path.read_bytes()
    assert raw[:6] == b"\x7fELF\x02\x01"
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    offset, width, count, strings_index = header[6], header[11], header[12], header[13]
    sections = [struct.unpack_from("<IIQQQQIIQQ", raw, offset + width * i) for i in range(count)]
    names = sections[strings_index]
    table = raw[names[4] : names[4] + names[5]]
    for section in sections:
        if table[section[0] :].split(b"\0", 1)[0] == b".text":
            return raw[section[4] : section[4] + section[5]]
    raise AssertionError("linked native image lacks .text")


@pytest.mark.parametrize("shape,permutation,bits", [((2, 3, 4), (1, 0, 2), 32), ((3, 2, 5), (2, 0, 1), 16)])
def test_ordinary_native_packed_producer_removes_actual_scan_and_keeps_complete_state(
    tmp_path, shape, permutation, bits
):
    import ctypes
    import hashlib
    import json
    import shutil
    import subprocess
    import time

    import numpy as np

    from merlin.common.digest import sha256_file
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.optional_passes import Selection, applied
    from merlin.llvmlower.toolchain import clang

    source = program(*shape, permutation, bits)
    original_source = tmp_path / "original.mlir"
    original_source.write_text(source)
    rows, outer, inner = shape
    logical = (rows, outer * inner)
    packed = tuple(shape[i] for i in permutation)
    dtype = np.dtype(f"<i{bits // 8}")
    observer = tmp_path / "observer.c"
    observer.write_text("""#include <stdint.h>
uint64_t read_bytes, write_bytes;
#define TRACE(N) \\
void __sanitizer_cov_load##N(void *p) { read_bytes += N; } \\
void __sanitizer_cov_store##N(void *p) { write_bytes += N; }
TRACE(1) TRACE(2) TRACE(4) TRACE(8) TRACE(16)
void __sanitizer_cov_trace_pc_guard(uint32_t *p) {}
void __sanitizer_cov_trace_pc_guard_init(uint32_t *a, uint32_t *b) {}
""")
    observed_compiler = Path(shutil.which(str(clang())) or clang()).resolve()
    observed_inputs = (observed_compiler, mlir_runtime_c().resolve(), observer.resolve())
    observed_pins = {str(path): sha256_file(path) for path in observed_inputs}
    results = {}
    for arm, selector in (("baseline", "-automatic-host-permutations"), ("optimized", "automatic-host-permutations")):
        started = time.perf_counter()
        with applied(Selection.parse(selector)), compilation_memory() as memory:
            compiled = lower_model(source, tmp_path / arm, targets=("host",), vectorize=False, parallel=False)
        seconds = time.perf_counter() - started
        assert compiled.host_so is not None
        assert compiled.stats["automatic_host_permutations"]["rewrites"] == int(arm == "optimized")
        # Normal ordinary -O2 linked image plus separately observed exact LLVM.
        # Instrumented counters are requested scalar accesses, not bus/timing.
        observed = tmp_path / (arm + "-observed.so")
        observed_argv = [
            str(clang()),
            "-O0",
            "-shared",
            "-fPIC",
            "-fsanitize-coverage=trace-loads,trace-stores,trace-pc-guard",
            str(compiled.ll_path),
            str(mlir_runtime_c()),
            str(observer),
            "-lm",
            "-o",
            str(observed),
        ]
        observer_started = time.perf_counter()
        subprocess.run(
            observed_argv,
            check=True,
            capture_output=True,
        )
        observer_compile_seconds = time.perf_counter() - observer_started
        image_results = []
        for image, instrumented in ((compiled.host_so, False), (observed, True)):
            model = HostModel.load(str(image), image_policy=PrivateHostImagePolicy(image.parent.resolve()))

            def counters():
                return tuple(ctypes.c_uint64.in_dll(model.lib, name).value for name in ("read_bytes", "write_bytes"))

            state = np.zeros(logical, dtype=dtype)
            complete, invocations = [], []
            for epoch in range(3):
                values = np.asarray([(-200 + 59 * i + epoch) for i in range(prod(logical))], dtype=dtype).reshape(
                    logical
                )
                values.flat[0] = np.iinfo(dtype).max
                values.flat[1] = np.iinfo(dtype).min
                # Exercise input/state aliases while keeping the original inputs
                # unchanged; a later epoch must not reuse earlier prepared data.
                if epoch == 1:
                    state = values
                old_values, old_state = values.copy(), state.copy()
                expected_values = []
                next_values = []
                for x, y in zip(old_values.flat, old_state.flat, strict=True):
                    total = (int(x) + int(y) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
                    expected_values.append(min(127, max(0, total)))
                    next_values.append((int(y) + 1 + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1)))
                expected = np.transpose(np.asarray(expected_values, dtype=np.int8).reshape(shape), permutation).copy()
                expected_state = np.asarray(next_values, dtype=dtype).reshape(logical)
                buffers = [
                    np.full(prod(dims) + 32, 83, dtype=typ) for dims, typ in ((packed, np.int8), (logical, dtype))
                ]
                outputs = [buf[16:-16].reshape(dims) for buf, dims in zip(buffers, (packed, logical), strict=True)]
                call_directory = tmp_path / arm / ("observed" if instrumented else "normal")
                call_directory.mkdir(exist_ok=True)
                inputs = call_directory / f"inputs-{epoch}.npz"
                np.savez(inputs, values=old_values, state=old_state)
                before_counts = counters() if instrumented else None
                model([(a.ctypes.data, a.shape) for a in (values, state, *outputs)])
                after_counts = counters() if instrumented else None
                for actual, reference in zip(outputs, (expected, expected_state), strict=True):
                    assert actual.tobytes() == reference.tobytes()
                assert values.tobytes() == old_values.tobytes() and state.tobytes() == old_state.tobytes()
                assert all(np.all(b[:16] == 83) and np.all(b[-16:] == 83) for b in buffers)
                returned = call_directory / f"outputs-{epoch}.npz"
                np.savez(returned, packed=outputs[0], state=outputs[1])
                invocations.append(
                    dict(
                        inputs=str(inputs),
                        inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                        outputs=str(returned),
                        outputs_sha256=hashlib.sha256(returned.read_bytes()).hexdigest(),
                        readonly_alias=epoch == 1,
                        actual_input_sha256_after_call={
                            name: hashlib.sha256(array.tobytes()).hexdigest()
                            for name, array in (("values", values), ("state", state))
                        },
                        observed_read_write_bytes=(
                            [after - before for before, after in zip(before_counts, after_counts, strict=True)]
                            if instrumented
                            else None
                        ),
                    )
                )
                complete.append([hashlib.sha256(a.tobytes()).hexdigest() for a in outputs])
                state = outputs[1].copy()
            image_results.append(
                dict(
                    image=str(image),
                    sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    bytes=image.stat().st_size,
                    text_bytes=len(text_section(image)),
                    complete_outputs_sha256=complete,
                    instrumented=instrumented,
                    invocations=invocations,
                    read_bytes=ctypes.c_uint64.in_dll(model.lib, "read_bytes").value if instrumented else None,
                    write_bytes=ctypes.c_uint64.in_dll(model.lib, "write_bytes").value if instrumented else None,
                )
            )
        assert image_results[0]["complete_outputs_sha256"] == image_results[1]["complete_outputs_sha256"]
        llvm = compiled.ll_path.read_text()
        results[arm] = dict(
            compile_seconds=seconds,
            sampled_peak_memory=memory,
            llvm_path=str(compiled.ll_path),
            llvm_sha256=hashlib.sha256(llvm.encode()).hexdigest(),
            llvm_malloc_sites=sum("call ptr @malloc(" in line for line in llvm.splitlines()),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            source_path=str(original_source),
            object_path=str(compiled.host_so.with_suffix(".o")),
            object_sha256=hashlib.sha256(compiled.host_so.with_suffix(".o").read_bytes()).hexdigest(),
            observed_image_compile_seconds=observer_compile_seconds,
            observed_build_argv=observed_argv,
            observed_build_input_sha256={
                str(compiled.ll_path.resolve()): sha256_file(compiled.ll_path),
                **observed_pins,
            },
            observed_compile_scope=(
                "separate instrumented -O0 LLVM/runtime/observer build; excluded from compile_seconds"
            ),
            images=image_results,
            pipeline=compiled.stats["automatic_host_permutations"],
            timing_ns=None,
            physical_traffic_bytes=None,
            device_commands=None,
            scope=(
                "serial host integer layout realization; no target pack format, "
                "accelerator qualification or speedup claim"
            ),
        )
    baseline, optimized = results["baseline"], results["optimized"]
    assert baseline["source_sha256"] == optimized["source_sha256"]
    assert baseline["pipeline"]["library_sources_sha256"] == optimized["pipeline"]["library_sources_sha256"]
    assert baseline["llvm_malloc_sites"] == optimized["llvm_malloc_sites"] + 1
    assert baseline["images"][0]["complete_outputs_sha256"] == optimized["images"][0]["complete_outputs_sha256"]
    for field in ("read_bytes", "write_bytes"):
        assert baseline["images"][1][field] - optimized["images"][1][field] == 3 * prod(shape)
    assert observed_pins == {str(path): sha256_file(path) for path in observed_inputs}
    (tmp_path / "pair.json").write_text(json.dumps(results, indent=2))
