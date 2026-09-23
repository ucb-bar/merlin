"""Production wiring for the explicit cycle-only warm whole-program harness."""
from __future__ import annotations

import pytest

from merlin.perf.execution_policy import WarmProfileContract
from merlin.runtime.backends import base as backends
from merlin.targetgen.contract import compile as compiler


def whole_program():
    return {
        "abi_version": "0.1", "target": "gemmini", "commands": [],
        "tensors": {
            "A": {"shape": [2, 2], "dtype": "i8", "role": "input"},
            "Y": {"shape": [2, 2], "dtype": "i8", "role": "output"},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": "A", "access": "read"},
                {"tensor": "Y", "access": "write"},
            ],
            "outputs": ["Y"],
        },
    }


def test_explicit_whole_program_profile_has_exact_completed_invocation_order(monkeypatch):
    monkeypatch.setenv("MERLIN_CACHE_STATE", "cold")
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    source = backends.get_backend("gemmini").render_harness(
        whole_program(), target="gemmini", inputs={"A": [[1, 2], [3, 4]]},
        warm_profile=WarmProfileContract())
    call = "gemmini_kernel((void*)T_A, (void*)T_Y);"
    assert source.count(call) == source.count("gemmini_fence();") == 2
    warm_call = source.index(call)
    warm_done = source.index("gemmini_fence();", warm_call)
    start = source.index("merlin_profile_cycle_start", warm_done)
    measured_call = source.index(call, start)
    measured_done = source.index("gemmini_fence();", measured_call)
    end = source.index("merlin_profile_cycle_end", measured_done)
    metric = source.index("METRIC cycles ", end)
    readback = source.index('printf("OUT Y', metric)
    done = source.index('printf("DONE', readback)
    assert warm_call < warm_done < start < measured_call < measured_done < end < metric < readback < done
    assert source.count("METRIC ") == 1
    assert "cycle_window_gemmini_region" not in source
    assert "counter_configure" not in source


def test_profile_is_whole_program_only_and_strict():
    backend = backends.get_backend("gemmini")
    with pytest.raises(Exception, match="whole-program"):
        backend.render_harness(
            {"tensors": {}, "commands": []}, target="gemmini",
            warm_profile=WarmProfileContract())
    with pytest.raises(Exception, match="exactly one warm"):
        backend.render_harness(
            whole_program(), target="gemmini", inputs={"A": [[1, 2], [3, 4]]},
            warm_profile=WarmProfileContract(warmup_runs=2))


def test_generic_compile_refuses_non_whole_profile_before_compilation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        compiler, "llvm_mlir_to_object",
        lambda *args, **kwargs: pytest.fail("invalid profile reached compilation"))
    with pytest.raises(ValueError, match="whole-program"):
        compiler.compile_lowered_to_elf(
            {"kernel_abi": {"kind": "per_operation"}}, "unused", tmp_path,
            target="gemmini", warm_profile=WarmProfileContract())


def test_profiled_compile_bypasses_legacy_build_cache(monkeypatch, tmp_path):
    from merlin.targetgen import build_cache

    for name in ("build_identity", "reuse", "store"):
        monkeypatch.setattr(
            build_cache, name,
            lambda *args, **kwargs: pytest.fail("profiled build reached legacy cache"))
    monkeypatch.setattr(
        compiler, "llvm_mlir_to_object",
        lambda *args, **kwargs: tmp_path / "kernel.o")
    seen = []

    def link(cb, obj, workdir, **kwargs):
        seen.append(kwargs)
        return tmp_path / "package.elf"

    monkeypatch.setattr(compiler, "link_elf", link)
    profile = WarmProfileContract()
    result = compiler.compile_lowered_to_elf(
        whole_program(), "unused", tmp_path, target="gemmini",
        inputs={"A": [[1, 2], [3, 4]]}, warm_profile=profile)
    assert result == tmp_path / "package.elf"
    assert seen == [{
        "target": "gemmini", "inputs": {"A": [[1, 2], [3, 4]]},
        "warm_profile": profile,
    }]


def test_profile_output_remains_compatible_with_existing_parser():
    console = """MERLIN_INVOCATIONS warmup=1 measured=1
MERLIN_PROFILE warmup begin
MERLIN_PROFILE warmup end rc=0
MERLIN_PROFILE measured begin
METRIC cycles 37
MERLIN_PROFILE measured end rc=0
OUT Y 1 2 4 5
DONE
"""
    outputs, metrics = backends.get_backend("gemmini").parse_output(console)
    assert outputs == {"Y": [[4, 5]]}
    assert metrics == {"cycles": 37}
