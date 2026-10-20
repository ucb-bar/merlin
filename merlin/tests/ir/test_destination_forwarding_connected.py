"""Connected integer destination writes and independently ordered full outputs."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from merlin.common.paths import merlin_dir

OUTPUT_NAMES = ("next_state", "padded", "next", "original_pad", "wide")


def source():
    return (merlin_dir() / "tests/ir/destination_forwarding_connected.mlir").read_text()


def inputs(epoch, state):
    def signed(count, multiplier):
        values = np.asarray([((i * multiplier + epoch * 11) % 256) - 128 for i in range(count)], dtype=np.int8)
        values[0], values[1] = -128, 127
        return values

    return signed(8, 37).reshape(2, 4), signed(24, 59).reshape(4, 6), signed(40, 71).reshape(8, 5), state.copy()


def wrap32(value):
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def reference(x, w1, w2, state):
    """Original scalar order; wrap before clamp and retain the immutable pad."""
    assert [(v.shape, v.dtype) for v in (x, w1, w2, state)] == [
        ((2, 4), np.dtype("int8")),
        ((4, 6), np.dtype("int8")),
        ((8, 5), np.dtype("int8")),
        ((2, 6), np.dtype("int32")),
    ]
    wide = np.zeros((2, 6), dtype=np.int32)
    next_state = np.zeros_like(wide)
    padded = np.zeros((4, 8), dtype=np.int8)
    for m in range(2):
        for n in range(6):
            accumulator = 0
            for k in range(4):
                accumulator = wrap32(accumulator + int(x[m, k]) * int(w1[k, n]))
            wide[m, n] = accumulator
            value = wrap32(accumulator + int(state[m, n]))
            next_state[m, n] = value
            padded[m + 1, n + 1] = max(-128, min(127, value))
    following = np.zeros((4, 5), dtype=np.int32)
    for m in range(4):
        for n in range(5):
            accumulator = 0
            for k in range(8):
                # Read the original insertion coordinates, not an optimized view.
                value = 0 if not (1 <= m < 3 and 1 <= k < 7) else int(padded[m, k])
                accumulator = wrap32(accumulator + value * int(w2[k, n]))
            following[m, n] = accumulator
    return next_state, padded, following, np.zeros((4, 8), dtype=np.int8), wide


def test_public_source_has_two_complete_small_signed_integer_contractions():
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry

    module = parse_mlir_text(source())
    module.verify()
    contractions = [
        op
        for op in module.walk()
        if type(op) is GenericOp and any(it.data is IteratorType.REDUCTION for it in op.get_iterator_types())
    ]
    assert [static_zero_integer_matmul_geometry(op) for op in contractions] == [(2, 4, 6), (4, 8, 5)]


def test_reference_retains_zero_exterior_and_wrap_before_clamp():
    x = np.ones((2, 4), dtype=np.int8)
    w1 = np.ones((4, 6), dtype=np.int8)
    w2 = np.ones((8, 5), dtype=np.int8)
    state = np.zeros((2, 6), dtype=np.int32)
    state[0, 0] = np.iinfo(np.int32).max
    updated, padded, following, original, wide = reference(x, w1, w2, state)
    assert updated[0, 0] == np.iinfo(np.int32).min + 3 and padded[1, 1] == -128
    assert np.all(wide == 4) and np.all(original == 0)
    assert np.all(padded[0] == 0) and np.all(padded[-1] == 0)
    assert np.all(padded[:, 0] == 0) and np.all(padded[:, -1] == 0)
    assert np.all(following[0] == 0) and np.all(following[-1] == 0)


def test_native_destination_pair_preserves_full_connected_state_and_actual_copies(tmp_path):
    import ctypes
    import subprocess

    from test_packed_producer_native import text_section

    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.optional_passes import Selection, applied
    from merlin.llvmlower.toolchain import clang

    support = tmp_path / "copy_counter.c"
    support.write_text(
        """#include <stdint.h>
#include <stddef.h>
struct unranked { int64_t rank; void *descriptor; };
struct ranked { void *allocated, *aligned; int64_t offset; int64_t sizes[]; };
uint64_t copy_calls, copy_bytes, malloc_calls, malloc_bytes;
extern void *__real_malloc(size_t);
void *__wrap_malloc(size_t bytes) { malloc_calls++; malloc_bytes+=bytes; return __real_malloc(bytes); }
extern void __real_memrefCopy(int64_t, struct unranked *, struct unranked *);
void __wrap_memrefCopy(int64_t width, struct unranked *src, struct unranked *dst) {
  struct ranked *desc=src->descriptor; uint64_t count=1;
  for (int64_t i=0; i<src->rank; i++) count*=desc->sizes[i];
  copy_calls++; copy_bytes+=count*width; __real_memrefCopy(width,src,dst);
}
"""
    )
    source_pins = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (
            merlin_dir() / "tests/ir/destination_forwarding_connected.mlir",
            merlin_dir() / "tests/ir/test_destination_forwarding_connected.py",
            merlin_dir() / "tests/ir/test_packed_producer_native.py",
        )
    ]
    arms = {}
    for arm, enabled in (("baseline", False), ("candidate", True)):
        controls = Selection.of(
            enable=("automatic-host-destinations",) if enabled else (),
            disable=("automatic-host-permutations", "automatic-host-gathers")
            + (() if enabled else ("automatic-host-destinations",)),
        )
        with applied(controls):
            compiled = lower_model(source(), tmp_path / arm, targets=(), vectorize=False, parallel=False)
        report = compiled.stats["automatic_host_destinations"]
        assert report["rewrites"] == int(enabled)
        assert compiled.stats["automatic_host_permutations"]["rewrites"] == 0
        assert compiled.stats["automatic_host_gathers"]["rewrites"] == 0
        image = tmp_path / (arm + ".so")
        command = [
            str(clang()),
            "-O2",
            "-fPIC",
            "-shared",
            str(compiled.ll_path),
            str(mlir_runtime_c()),
            str(support),
            "-Wl,--wrap=memrefCopy",
            "-Wl,--wrap=malloc",
            "-o",
            str(image),
        ]
        subprocess.run(command, check=True, capture_output=True)
        model = HostModel.load(str(image), image_policy=PrivateHostImagePolicy(tmp_path.resolve()))
        state = np.arange(-6, 6, dtype=np.int32).reshape(2, 6)
        state[0, 0], state[0, 1] = np.iinfo(np.int32).max, np.iinfo(np.int32).min
        calls = []
        for epoch in range(3):
            originals = inputs(epoch, state)
            expected = reference(*originals)
            input_guards = [np.full(v.size + 32, 67, dtype=v.dtype) for v in originals]
            arguments = [b[16:-16].reshape(v.shape) for b, v in zip(input_guards, originals, strict=True)]
            for argument, original in zip(arguments, originals, strict=True):
                argument[:] = original
            before = [b.copy() for b in input_guards]
            output_guards = [np.full(v.size + 32, 83, dtype=v.dtype) for v in expected]
            outputs = [b[16:-16].reshape(v.shape) for b, v in zip(output_guards, expected, strict=True)]
            model([(v.ctypes.data, v.shape) for v in (*arguments, *outputs)])
            assert all(a.tobytes() == b.tobytes() for a, b in zip(outputs, expected, strict=True))
            assert all(a.tobytes() == b.tobytes() for a, b in zip(input_guards, before, strict=True))
            assert all(np.all(b[:16] == 83) and np.all(b[-16:] == 83) for b in output_guards)
            overflow_count = sum(
                not -(1 << 31) <= int(old) + int(wide) < (1 << 31)
                for old, wide in zip(originals[-1].flat, outputs[-1].flat, strict=True)
            )
            if epoch == 0:
                assert overflow_count > 0
            actual = tmp_path / f"{arm}-actual-{epoch}.npz"
            np.savez(
                actual,
                **dict(zip(OUTPUT_NAMES, outputs, strict=True)),
                **{f"input{i}": v for i, v in enumerate(arguments)},
                **{f"input_guard{i}": v for i, v in enumerate(input_guards)},
                **{f"output_guard{i}": v for i, v in enumerate(output_guards)},
            )
            calls.append(
                {
                    "path": str(actual),
                    "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
                    "inputs_sha256": [hashlib.sha256(v.tobytes()).hexdigest() for v in originals],
                    "outputs_sha256": [hashlib.sha256(v.tobytes()).hexdigest() for v in outputs],
                    "overflow_before_clamp": overflow_count,
                }
            )
            state = outputs[0].copy()
        arms[arm] = {
            "report": report,
            "calls": calls,
            "native_link_command": command,
            "artifacts": [
                {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in (compiled.ll_path, image, support, mlir_runtime_c(), clang())
            ],
            "counters": {
                name: ctypes.c_uint64.in_dll(model.lib, name).value
                for name in ("copy_calls", "copy_bytes", "malloc_calls", "malloc_bytes")
            },
            "text_bytes": len(text_section(image)),
            "loaded_image_sha256": model.image_sha256,
        }
        assert model.image_sha256 == hashlib.sha256(image.read_bytes()).hexdigest()
    baseline, candidate = arms["baseline"], arms["candidate"]
    for a, b in zip(baseline["calls"], candidate["calls"], strict=True):
        assert a["inputs_sha256"] == b["inputs_sha256"] and a["outputs_sha256"] == b["outputs_sha256"]
    assert candidate["counters"]["copy_calls"] < baseline["counters"]["copy_calls"]
    assert candidate["counters"]["copy_bytes"] < baseline["counters"]["copy_bytes"]
    from pathlib import Path

    for pin in source_pins + [row for arm in arms.values() for group in ("artifacts", "calls") for row in arm[group]]:
        assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest() == pin["sha256"]
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "arms": arms,
                "source_and_reference": source_pins,
                "scope": "instrumented native host copies; no device or latency claim",
            },
            indent=2,
        )
        + "\n"
    )
