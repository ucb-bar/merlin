"""Native source-consumer proof, separate from original target qualification."""

from __future__ import annotations

import hashlib
import json
from math import prod

import numpy as np
import pytest
from test_packed_producer_destinations import program
from test_packed_producer_native import compilation_memory, text_section

from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.lower import lower_model_file
from merlin.llvmlower.source_optimization_policy import SourceOptimizationPolicy
from merlin.llvmlower.source_optimization_request import PACKING_PASS
from merlin.runtime.source_optimization_build import SourceOptimizationBuild


@pytest.mark.parametrize("shape,permutation,bits", [((2, 3, 4), (1, 0, 2), 32), ((3, 2, 5), (2, 0, 1), 16)])
def test_policy_precedes_source_and_preserves_full_native_recurrence(tmp_path, monkeypatch, shape, permutation, bits):
    import time

    # Both choices and their original permission/ceiling exist before source
    # construction. Neither selection uses a baseline output or prepared SHA.
    policies = {
        arm: SourceOptimizationPolicy(enabled, False, 4096, (PACKING_PASS,), 4096)
        for arm, enabled in (("baseline", False), ("candidate", True))
    }
    builds = {arm: SourceOptimizationBuild(value) for arm, value in policies.items()}
    source = program(*shape, permutation, bits)
    path = tmp_path / "model.prepared.mlir"
    path.write_text(source)
    monkeypatch.setenv("MERLIN_PASSES", "automatic-host-gathers,-automatic-host-permutations")
    results = {}
    dtype = np.dtype(f"<i{bits // 8}")
    logical = (shape[0], shape[1] * shape[2])
    packed = tuple(shape[index] for index in permutation)
    for arm, guard in builds.items():
        guard.bind(path)
        selected = guard.options()["source_optimizations"]
        assert selected.request.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        started = time.perf_counter()
        with compilation_memory() as memory:
            compiled = lower_model_file(path, tmp_path / arm, targets=("host",), **guard.options())
        compile_seconds = time.perf_counter() - started
        guard.revalidate()
        report = compiled.stats["source_optimizations"]
        assert report["selection"] == selected.record()
        assert report["consumers"][PACKING_PASS]["rewrites"] == int(arm == "candidate")
        assert report["consumers"]["automatic-host-gathers"]["rewrites"] == 0
        model = HostModel.load(
            str(compiled.host_so), image_policy=PrivateHostImagePolicy(compiled.host_so.parent.resolve())
        )
        state = np.zeros(logical, dtype=dtype)
        calls = []
        for epoch in range(3):
            values = np.asarray([(-200 + index * 59 + epoch) for index in range(prod(logical))], dtype=dtype).reshape(
                logical
            )
            values.flat[0], values.flat[1] = np.iinfo(dtype).min, np.iinfo(dtype).max
            if epoch == 0:
                state.flat[0], state.flat[1] = -1, 1  # Addition wraps BEFORE clamp.
            originals = values.copy(), state.copy()
            narrow, next_values = [], []
            for x, y in zip(originals[0].flat, originals[1].flat, strict=True):
                total = (int(x) + int(y) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
                narrow.append(max(0, min(127, total)))
                next_values.append((int(y) + 1 + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1)))
            expected = (
                np.transpose(np.asarray(narrow, dtype=np.int8).reshape(shape), permutation).copy(),
                np.asarray(next_values, dtype=dtype).reshape(logical),
            )
            canaries = [
                np.full(prod(dims) + 32, 83, dtype=kind) for dims, kind in ((packed, np.int8), (logical, dtype))
            ]
            outputs = [raw[16:-16].reshape(dims) for raw, dims in zip(canaries, (packed, logical), strict=True)]
            model([(value.ctypes.data, value.shape) for value in (values, state, *outputs)])
            actual = tmp_path / arm / f"actual-{epoch}.npz"
            np.savez(
                actual,
                before_input=originals[0],
                before_state=originals[1],
                after_input=values,
                after_state=state,
                packed=outputs[0],
                next_state=outputs[1],
                guard0=canaries[0],
                guard1=canaries[1],
            )
            assert all(a.tobytes() == b.tobytes() for a, b in zip(outputs, expected, strict=True))
            assert all(a.tobytes() == b.tobytes() for a, b in zip((values, state), originals, strict=True))
            assert all(np.all(raw[:16] == 83) and np.all(raw[-16:] == 83) for raw in canaries)
            calls.append({"path": str(actual), "sha256": hashlib.sha256(actual.read_bytes()).hexdigest()})
            state = outputs[1].copy()
            guard.revalidate()
        llvm = compiled.ll_path.read_text()
        results[arm] = {
            "original_policy": policies[arm].record(),
            "bound_selection": selected.record(),
            "prepared_input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_report": report,
            "calls": calls,
            "compile_seconds": compile_seconds,
            "sampled_peak_memory": memory,
            "elf_bytes": compiled.host_so.stat().st_size,
            "text_bytes": len(text_section(compiled.host_so)),
            "llvm_malloc_sites": sum("call " in line and "@malloc(" in line for line in llvm.splitlines()),
            "artifacts": [
                {"path": str(product), "sha256": hashlib.sha256(product.read_bytes()).hexdigest()}
                for product in (path, compiled.ll_path, compiled.host_so.with_suffix(".o"), compiled.host_so)
            ],
        }
    assert results["baseline"]["llvm_malloc_sites"] > results["candidate"]["llvm_malloc_sites"]
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "arms": results,
                "scope": "ordinary native source consumer; complete recurrence; engineering policy transport only",
                "target_qualification": None,
                "timing_ns": None,
            },
            indent=2,
        )
        + "\n"
    )
