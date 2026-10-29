"""Whole-model builds refuse an unbound cross-compiler index width before lowering."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "observation",
    [
        None,
        {},
        {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": "fake-clang",
            "compiler_resolved": "/absent/fake-clang",
            "compiler_sha256": "0" * 64,
            "cross_flags": ["--target=riscv64-unknown-elf", "-march=rv64gc"],
            "data_layout": "e-p:64:64:64:32",
            "index_bits": 64,  # Refuses a width that disagrees with the reported GEP index width.
        },
        {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": "fake-clang",
            "compiler_resolved": "/absent/fake-clang",
            "compiler_sha256": "0" * 64,
            "cross_flags": ["--target=riscv64-unknown-elf", "-march=rv64gc"],
            "data_layout": "e-p:32:32",
            "index_bits": 32,  # Valid generic lowering, but not the selected int64_t C descriptor ABI.
        },
    ],
)
def test_build_refuses_missing_or_mismatched_index_observation_before_lowering(tmp_path, monkeypatch, observation):
    from merlin.llvmlower import qinner, target_data_layout, toolchain, weight_prepack
    from merlin.runtime.backends import spike, spike_model, zephyr_model

    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text("module {}")
    monkeypatch.setattr(zephyr_model, "load_matrix_signatures", lambda *_: None)
    monkeypatch.setattr(weight_prepack, "prepare_build_bundle", lambda model_dir, *_: model_dir)
    monkeypatch.setattr(spike, "gcc_path", lambda: Path("/absent/gcc"))
    monkeypatch.setattr(toolchain, "clang", lambda: "fake-clang")
    monkeypatch.setattr(qinner, "plan_for_bundle", lambda *_: False)
    monkeypatch.setattr(target_data_layout, "observe_index_width", lambda *_: observation)
    monkeypatch.setattr(spike_model, "lower_model_file", lambda *_a, **_k: pytest.fail("lowering must not run"))

    with pytest.raises((spike_model.SpikeModelError, ValueError, TypeError)):
        spike_model.build(
            model,
            tmp_path / "build",
            backend="scalar",
            cflags_override=["-march=rv64gc"],
        )
    assert not (tmp_path / "build" / "model.o").exists()
    assert not list((tmp_path / "build").glob("*.elf"))


@pytest.mark.parametrize("backend,vlen", [("scalar", None), ("rvv", None), ("rvv", 256)])
def test_compiler_plan_reexport_preserves_flags_and_original_spike_globals(monkeypatch, backend, vlen):
    from merlin.llvmlower import impr_features
    from merlin.runtime.backends import host_compiler_policy, spike_model

    assert spike_model.selected_model_compiler_plan is host_compiler_policy.selected_model_compiler_plan
    defaults = ["-march=rv64gcv", "-mabi=lp64d", "-O2"]
    monkeypatch.setattr(spike_model, "RVV_CFLAGS", defaults)
    monkeypatch.setattr(spike_model.toolchain, "clang", lambda: Path("/selected/clang"))
    harness_flags = []

    def harness(flags):
        harness_flags.append(list(flags))
        return [*flags, "-g"]

    def features(flags, selected):
        assert selected == frozenset({"test_feature"})
        return [*flags, "-fno-vectorize"]

    observed = []

    def observe(compiler, flags):
        observed.append((compiler, list(flags)))
        return {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": str(compiler),
            "compiler_resolved": "/selected/clang",
            "compiler_sha256": "1" * 64,
            "cross_flags": list(flags),
            "index_bits": 64,
            "data_layout": "e-p:64:64",
        }

    monkeypatch.setattr(spike_model, "_harness_cflags", harness)
    monkeypatch.setattr(impr_features, "apply_cflags", features)
    monkeypatch.setattr(spike_model.target_data_layout, "observe_index_width", observe)
    plan = spike_model.selected_model_compiler_plan(
        backend=backend, cflags_override=None, vlen=vlen, features=frozenset({"test_feature"})
    )
    march = "-march=rv64gc" if backend == "scalar" else "-march=rv64gcv"
    assert harness_flags == [[march, "-mabi=lp64d", "-O2"]]
    if vlen is not None:
        march += "_zvl256b"
    assert plan["clang_cflags"] == [march, "-mabi=lp64d", "-O2"]
    assert plan["gcc_cflags"] == [*plan["clang_cflags"], "-g"]
    assert plan["model_cflags"] == [*plan["clang_cflags"], "-fno-vectorize"]
    assert observed == [(Path("/selected/clang"), [spike_model.CLANG_TARGET, *plan["model_cflags"]])]
    assert defaults == ["-march=rv64gcv", "-mabi=lp64d", "-O2"]
