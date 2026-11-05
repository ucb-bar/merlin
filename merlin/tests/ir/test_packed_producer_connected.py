"""Connected channel shuffle preserves both contractions and invocation-owned state.

This public synthetic source is independent of supplied target fixtures. Native
host execution establishes no target compiler, runtime or numerical qualification.
"""

from pathlib import Path

import numpy as np
import pytest
from test_packed_producer_native import compilation_memory, text_section


def source():
    from merlin.common.paths import merlin_dir

    return (merlin_dir() / "tests/ir/producer_packing_connected.mlir").read_text()


def reference(x, w1, w2, state):
    """Ordered signed integer operations, with an explicit shuffle coordinate map."""
    wide = np.empty((2, 12), dtype=np.int32)
    packed = np.empty((2, 4, 3), dtype=np.int8)
    next_state = np.empty((2, 12), dtype=np.int32)
    for m in range(2):
        for n in range(12):
            total = 0
            for k in range(4):
                total += int(x[m, k]) * int(w1[k, n])
            assert -(1 << 31) <= total < (1 << 31)
            wide[m, n] = total
            total += int(state[m, n])
            assert -(1 << 31) <= total < (1 << 31)
            # n=4*a+b; the final shuffle writes [m,b,a], not [m,a,b].
            a, b = divmod(n, 4)
            packed[m, b, a] = min(127, max(-128, total))
        for n in range(12):
            total = 0
            for k in range(12):
                b, a = divmod(k, 3)
                total += int(packed[m, b, a]) * int(w2[k, n])
            assert -(1 << 31) <= total < (1 << 31)
            next_state[m, n] = total
    return next_state, packed, wide


def inputs(epoch, state):
    x = np.asarray([((i * 5 + epoch * 3) % 17) - 8 for i in range(8)], dtype=np.int8).reshape(2, 4)
    w1 = np.asarray([((i * 3 + 2) % 9) - 4 for i in range(48)], dtype=np.int8).reshape(4, 12)
    w2 = np.asarray([((i * 7 + 1) % 9) - 4 for i in range(144)], dtype=np.int8).reshape(12, 12)
    return x, w1, w2, state


def test_connected_source_preserves_both_real_contractions_and_output_abi():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
    from merlin.llvmlower.destination_check import verify_permutation_rewrite

    original = parse_mlir_text(source())
    original.verify()
    selected, report = prepare_automatic_destinations(source(), serial=True, permuted_outputs=True)
    actual = parse_mlir_text(selected)
    actual.verify()
    assert report["rewrites"] == 1
    assert len(verify_permutation_rewrite(original, actual, index_bits=64)) == 1
    for module in (original, actual):
        contractions = [
            op
            for op in module.walk()
            if op.name == "linalg.generic" and any(iterator.data.value == "reduction" for iterator in op.iterator_types)
        ]
        assert len(contractions) == 2
        assert [str(v.type) for v in contractions[1].inputs] == ["tensor<2x12xi8>", "tensor<12x12xi8>"]
    assert original.body.block.first_op.function_type == actual.body.block.first_op.function_type


def test_wrong_same_shape_shuffle_is_not_a_valid_connected_rewrite():
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
    from merlin.llvmlower.destination_check import verify_permutation_rewrite

    selected, _ = prepare_automatic_destinations(source(), serial=True, permuted_outputs=True)
    actual = parse_mlir_text(selected)
    producer = next(
        op
        for op in actual.walk()
        if op.name == "linalg.generic"
        and len(op.iterator_types) == 3
        and all(iterator.data.value == "parallel" for iterator in op.iterator_types)
    )
    maps = list(producer.indexing_maps)
    maps[0] = AffineMapAttr(AffineMap(3, 0, (AffineDimExpr(0), AffineDimExpr(2) * 3 + AffineDimExpr(1))))
    producer.properties["indexing_maps"] = ArrayAttr(maps)
    actual.verify()
    with pytest.raises(ValueError, match="coordinates"):
        verify_permutation_rewrite(parse_mlir_text(source()), actual, index_bits=64)


def test_connected_native_pair_preserves_complete_outputs_and_state(tmp_path):
    import ctypes
    import hashlib
    import json
    import shutil
    import subprocess
    import time

    from merlin.common.digest import sha256_file
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.optional_passes import Selection, applied
    from merlin.llvmlower.toolchain import clang

    original_source = tmp_path / "original.mlir"
    original_source.write_text(source())
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
    compiler = Path(shutil.which(str(clang())) or clang()).resolve()
    observed_inputs = (compiler, mlir_runtime_c().resolve(), observer.resolve())
    pins = {str(path): sha256_file(path) for path in observed_inputs}
    pair = {}
    for arm, option in (("baseline", "-automatic-host-permutations"), ("optimized", "automatic-host-permutations")):
        started = time.perf_counter()
        with applied(Selection.parse(option)), compilation_memory() as memory:
            compiled = lower_model(source(), tmp_path / arm, targets=("host",), vectorize=False, parallel=False)
        seconds = time.perf_counter() - started
        observed = tmp_path / f"{arm}-observed.so"
        argv = [
            str(compiler),
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
        observed_started = time.perf_counter()
        subprocess.run(argv, check=True, capture_output=True)
        observed_seconds = time.perf_counter() - observed_started
        images = []
        for image, instrumented in ((compiled.host_so, False), (observed, True)):
            model = HostModel.load(str(image), image_policy=PrivateHostImagePolicy(image.parent.resolve()))
            state = np.asarray(([-129, -128, -1, 0, 1, 126, 127, 128] * 3), dtype=np.int32).reshape(2, 12)
            records = []
            for epoch in range(3):
                arguments = inputs(epoch, state)
                originals = [value.copy() for value in arguments]
                expected = reference(*originals)
                shapes = ((2, 12), (2, 4, 3), (2, 12))
                types = (np.int32, np.int8, np.int32)
                guards = [
                    np.full(int(np.prod(shape)) + 32, 83, dtype=dtype)
                    for shape, dtype in zip(shapes, types, strict=True)
                ]
                outputs = [array[16:-16].reshape(shape) for array, shape in zip(guards, shapes, strict=True)]
                call_dir = tmp_path / arm / ("observed" if instrumented else "normal")
                call_dir.mkdir(exist_ok=True)
                before = call_dir / f"inputs-{epoch}.npz"
                np.savez(before, **dict(zip(("x", "w1", "w2", "state"), originals, strict=True)))

                def counts():
                    return [ctypes.c_uint64.in_dll(model.lib, key).value for key in ("read_bytes", "write_bytes")]

                old_counts = counts() if instrumented else None
                model([(value.ctypes.data, value.shape) for value in (*arguments, *outputs)])
                delta = [a - b for a, b in zip(counts(), old_counts, strict=True)] if instrumented else None
                # Retain actual complete buffers, including guards and inputs,
                # before assertions so independent reopening can inspect failures.
                after = call_dir / f"outputs-{epoch}.npz"
                np.savez(
                    after,
                    next_state=outputs[0],
                    packed=outputs[1],
                    wide=outputs[2],
                    **{f"guard_{i}": array for i, array in enumerate(guards)},
                    **{
                        f"after_{name}": array
                        for name, array in zip(("x", "w1", "w2", "state"), arguments, strict=True)
                    },
                )
                for actual, golden in zip(outputs, expected, strict=True):
                    assert actual.tobytes() == golden.tobytes()
                assert all(
                    actual.tobytes() == original.tobytes()
                    for actual, original in zip(arguments, originals, strict=True)
                )
                assert all(np.all(array[:16] == 83) and np.all(array[-16:] == 83) for array in guards)
                records.append(
                    dict(
                        inputs=str(before),
                        inputs_sha256=sha256_file(before),
                        outputs=str(after),
                        outputs_sha256=sha256_file(after),
                        observed_read_write_bytes=delta,
                        complete_sha256=[hashlib.sha256(value.tobytes()).hexdigest() for value in outputs],
                    )
                )
                state = outputs[0].copy()
            images.append(
                dict(
                    path=str(image),
                    sha256=sha256_file(image),
                    bytes=image.stat().st_size,
                    text_bytes=len(text_section(image)),
                    instrumented=instrumented,
                    calls=records,
                )
            )
        pair[arm] = dict(
            compile_seconds=seconds,
            sampled_peak_memory=memory,
            images=images,
            llvm=dict(path=str(compiled.ll_path), sha256=sha256_file(compiled.ll_path)),
            object=dict(
                path=str(compiled.host_so.with_suffix(".o")), sha256=sha256_file(compiled.host_so.with_suffix(".o"))
            ),
            llvm_malloc_sites=sum("call ptr @malloc(" in line for line in compiled.ll_path.read_text().splitlines()),
            source=dict(path=str(original_source), sha256=sha256_file(original_source)),
            observed_build_argv=argv,
            observed_image_compile_seconds=observed_seconds,
            observed_input_sha256={str(compiled.ll_path.resolve()): sha256_file(compiled.ll_path), **pins},
            pipeline=compiled.stats["automatic_host_permutations"],
            timing_ns=None,
            physical_traffic_bytes=None,
            device_commands=None,
            scope="ordinary native host proof only; no target/capture/admission qualification",
        )
        (tmp_path / "pair.json").write_text(json.dumps(pair, indent=2))
    baseline, optimized = pair["baseline"], pair["optimized"]
    assert baseline["pipeline"]["library_sources_sha256"] == optimized["pipeline"]["library_sources_sha256"]
    assert [baseline["pipeline"]["rewrites"], optimized["pipeline"]["rewrites"]] == [0, 1]
    assert baseline["llvm_malloc_sites"] == optimized["llvm_malloc_sites"] + 1
    for arm in pair.values():
        normal, observed = arm["images"]
        assert [row["complete_sha256"] for row in normal["calls"]] == [
            row["complete_sha256"] for row in observed["calls"]
        ]
    assert [row["complete_sha256"] for row in baseline["images"][0]["calls"]] == [
        row["complete_sha256"] for row in optimized["images"][0]["calls"]
    ]
    for component in range(2):
        b = sum(row["observed_read_write_bytes"][component] for row in baseline["images"][1]["calls"])
        o = sum(row["observed_read_write_bytes"][component] for row in optimized["images"][1]["calls"])
        assert b - o == 72
    assert pins == {str(path): sha256_file(path) for path in observed_inputs}
