"""The paper compile seam must preserve exact K1 execution choices and measurements."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from merlin import compile_cli
from merlin.mining import k1
from merlin.mining import registry


def test_compile_rvv_forwards_strict_backend_harts_and_surfaces_metrics(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}\n")
    monkeypatch.setattr(compile_cli, "_ensure_bundle", lambda *_a, **_kw: bundle)
    monkeypatch.setattr(registry, "load_rvv_package", lambda _path: SimpleNamespace(
        is_int8=False, run_id="test", schedule_text="", cflags=[], compiler_features=()))
    monkeypatch.setattr(k1, "available", lambda: True)
    seen = {}

    def fake_run(model_dir, work, pkg, **kwargs):
        seen.update(kwargs)
        return {
            "local_binary": str(tmp_path / "model"),
            "metrics": {"cycles": 123, "wall_ns": 456, "peak_rss_kb": 7},
            "vlen": 256, "vlen_source": "csr", "memory_policy": "mmap",
            "execution_mode": "rvv_openmp", "requested_execution_mode": "rvv_openmp",
            "fallback_used": False, "core_count": 8, "requested_core_count": 8,
            "affinity_source": "sched_getaffinity", "n_xnn_routed": 4,
            "n_xnn_eligible": 4, "n_xnn_candidates": 5,
            "board_conditions": {"before": {"governor": "performance"},
                                 "after": {"governor": "performance"}},
            "sustained_wall_ns": {"median": 40}, "iter_wall_ns": [39, 40, 41],
            "sustained": {"median": 120}, "iter_cycles": [119, 120, 121],
            "trajectory_quality": {"scope": "trajectory", "steps": 3, "min_cosine": 0.999},
            "prefix": [0.0],
        }

    monkeypatch.setattr(k1, "run_on_k1", fake_run)
    result = compile_cli.compile_rvv(
        "fixture", "fp32", run="k1", verify=False, package="package", auto_capture=False,
        timeout=30, harts=8, iters=3, warmup=2, kernel_backend="xnnpack",
        fallback_policy="forbid", bundle_path=bundle)

    assert seen == {
        "timeout": 30, "iters": 3, "warmup": 2, "session_repeats": None,
        "deadline_ns": None,
        "kernel_backend": "xnnpack",
        "parallel_harts": 8, "fallback_policy": "forbid", "require_csr_vlen": True,
    }
    assert result["cycles"] == 123 and result["wall_ns"] == 456
    assert result["peak_rss_bytes"] == 7 * 1024
    assert result["execution"] == {
        "mode": "rvv_openmp", "requested_mode": "rvv_openmp", "fallback_used": False,
        "core_count": 8, "requested_core_count": 8,
        "affinity_source": "sched_getaffinity",
        "semantic_session": False, "same_input_repetition": True,
        "kernel_backend": "xnnpack", "n_routed": 4, "n_eligible": 4,
        "n_candidates": 5,
    }
    assert result["vlen_source"] == "csr"
    assert result["board_conditions"]["before"]["governor"] == "performance"
    assert result["iter_wall_ns"] == [39, 40, 41]
    assert result["trajectory_quality"]["scope"] == "trajectory"


def test_compile_rvv_separates_session_repeats_from_observations(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}\n")
    monkeypatch.setattr(registry, "load_rvv_package", lambda _path: SimpleNamespace(
        is_int8=False, run_id="test", schedule_text="", cflags=[], compiler_features=()))
    monkeypatch.setattr(k1, "available", lambda: True)
    seen = {}

    def fake_run(_model_dir, _work, _pkg, **kwargs):
        seen.update(kwargs)
        return {
            "local_binary": str(tmp_path / "model"), "metrics": {}, "vlen": 256,
            "vlen_source": "csr", "memory_policy": "resident", "execution_mode": "rvv",
            "requested_execution_mode": "rvv", "fallback_used": False, "core_count": 1,
            "requested_core_count": 1, "affinity_source": "sched_getaffinity",
            "sustained_wall_ns": {"median": 100}, "iter_wall_ns": [99, 100, 101],
            "prefix": [0.0],
        }

    monkeypatch.setattr(k1, "run_on_k1", fake_run)
    compile_cli.compile_rvv(
        "fixture", "fp32", run="k1", verify=False, package="package", auto_capture=False,
        timeout=10, iters=32, warmup=1, session_repeats=3, bundle_path=bundle)
    assert seen["iters"] == 32
    assert seen["session_repeats"] == 3
    assert seen["warmup"] == 1
    assert seen["timeout"] == 10


def test_compile_rvv_rejects_a_nonpositive_whole_session_budget(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}\n")

    with pytest.raises(ValueError, match="positive whole-session budget"):
        compile_cli.compile_rvv(
            "fixture", "fp32", run="none", verify=False, package="package",
            auto_capture=False, timeout=0, bundle_path=bundle)


def test_multi_program_verify_uses_same_precision_trajectory_not_root_golden(
        tmp_path, monkeypatch):
    bundle = tmp_path / "session"
    bundle.mkdir()
    (bundle / "session_contract.yaml").write_text(yaml.safe_dump({"version": 2}))
    monkeypatch.setattr(registry, "load_rvv_package", lambda _path: SimpleNamespace(
        is_int8=False, run_id="test", schedule_text="", cflags=[], compiler_features=()))
    monkeypatch.setattr(k1, "available", lambda: True)
    monkeypatch.setattr(k1, "run_on_k1", lambda *_args, **_kwargs: {
        "local_binary": str(tmp_path / "model"), "metrics": {}, "vlen": 256,
        "vlen_source": "csr", "memory_policy": "mmap", "execution_mode": "rvv",
        "requested_execution_mode": "rvv", "fallback_used": False, "core_count": 1,
        "requested_core_count": 1, "affinity_source": "sched_getaffinity",
        "trajectory_correctness": {
            "scope": "trajectory", "steps": 3, "min_cosine": 0.9999,
            "max_relative_error": 0.001, "top1_matches": 3, "top1_agreement": 1.0,
        },
        "trajectory_quality": {
            "scope": "trajectory", "steps": 3, "min_cosine": 0.99,
            "max_relative_error": 0.1, "top1_matches": 3, "top1_agreement": 1.0,
        },
        "stage_wall_ns": {"prefill": [10, 11, 10], "decode": [80, 81, 79]},
        "prefix": [0.0],
    })
    result = compile_cli.compile_rvv(
        "fixture", "fp32", run="k1", verify=True, package="package", auto_capture=False,
        timeout=10, iters=3, warmup=0, session_repeats=3, bundle_path=bundle)
    assert result["status"] == "verified"
    assert result["verify"]["reference"] == "eager_same_precision"
    assert result["trajectory_quality"]["min_cosine"] == 0.99
    assert result["stage_wall_ns"]["decode"] == [80, 81, 79]
    assert result["execution"]["semantic_session"] is True
    assert result["execution"]["same_input_repetition"] is False


def test_forged_session_version_is_not_reported_as_semantic(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}\n")
    (bundle / "session_contract.yaml").write_text(yaml.safe_dump({"version": 999}))
    monkeypatch.setattr(registry, "load_rvv_package", lambda _path: SimpleNamespace(
        is_int8=False, run_id="test", schedule_text="", cflags=[], compiler_features=()))
    monkeypatch.setattr(k1, "available", lambda: True)
    monkeypatch.setattr(k1, "run_on_k1", lambda *_args, **_kwargs: {
        "local_binary": str(tmp_path / "model"), "metrics": {}, "vlen": 256,
        "vlen_source": "csr", "memory_policy": "resident", "execution_mode": "rvv",
        "requested_execution_mode": "rvv", "fallback_used": False, "core_count": 1,
        "requested_core_count": 1, "affinity_source": "sched_getaffinity", "prefix": [0.0],
    })

    result = compile_cli.compile_rvv(
        "fixture", "fp32", run="k1", verify=False, package="package", auto_capture=False,
        timeout=10, iters=3, session_repeats=2, bundle_path=bundle)

    assert result["execution"]["semantic_session"] is False
    assert result["execution"]["same_input_repetition"] is True


def test_k1_subprocesses_share_one_monotonic_deadline(monkeypatch):
    observed = []
    now = 1_000_000_000
    monkeypatch.setattr(k1.time, "monotonic_ns", lambda: now)
    monkeypatch.setattr(k1.subprocess, "run", lambda *_args, **kwargs: (
        observed.append(kwargs["timeout"]) or SimpleNamespace(returncode=0, stdout="", stderr="")))

    @k1._whole_cell_deadline
    def two_phases():
        k1._ssh("perf", timeout=100)
        k1._ssh("validation", timeout=100)

    two_phases(deadline_ns=now + 5_000_000_000)

    assert observed == [5, 5]
