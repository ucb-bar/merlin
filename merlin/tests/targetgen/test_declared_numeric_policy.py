"""Numerical assumptions are explicit inputs, not ambient target-profile discovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0.declarations import for_target

from merlin.common.yaml import load_yaml
from merlin.compile import mesh
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import oracle_policy

pytestmark = pytest.mark.target("atlas", "gemmini", "mx_gemmini", "radiance", "saturn_opu", "saturn_opu_rvv")


@pytest.mark.parametrize(
    "target,expected",
    [
        ("atlas", {"compare": "tolerance_float", "atol": 0.25, "rtol": 0.02, "subnormal_operand_flush": True}),
        ("gemmini", {"compare": "exact_int"}),
        (
            "mx_gemmini",
            {
                "operand_dtype": "mxfp8",
                "accum_dtype": "bf16",
                "scaling": "block_e8m0",
                "compare": "tolerance_float",
                "atol": 0.03125,
                "rtol": 0.03125,
            },
        ),
        (
            "radiance",
            {
                "operand_dtype": "f32",
                "accum_dtype": "f32",
                "compare": "tolerance_float",
                "atol": 0.03125,
                "rtol": 0.015625,
            },
        ),
        ("saturn_opu", {"compare": "exact_int"}),
        ("saturn_opu_rvv", {"compare": "exact_int"}),
    ],
)
def test_all_public_recipe_projections_are_unchanged(target, expected, monkeypatch):
    profile = load_yaml(for_target(target).recipe)

    def no_read(*args, **kwargs):
        raise AssertionError("projection must not read a profile or any sidecar")

    monkeypatch.setattr(Path, "read_text", no_read)
    monkeypatch.setattr(Path, "read_bytes", no_read)
    assert CS.profile_datapath(profile, numeric_only=True) == expected


def test_target_name_is_not_a_profile():
    with pytest.raises(TypeError, match="explicitly loaded"):
        CS.profile_datapath("synthetic")


def test_mesh_binding_preserves_complete_policy_and_operation_overlay(monkeypatch):
    policy = {
        "operand_dtype": "fp8_e4m3",
        "accum_dtype": "bf16",
        "compare": "tolerance_float",
        "atol": 0.25,
        "rtol": 0.02,
        "subnormal_operand_flush": True,
        "future_numeric_choice": "retained",
        "required_oracle_tiers": ["L3"],
    }
    original = dict(policy)
    observed = []
    monkeypatch.setattr(oracle_policy, "selected_sim_via", lambda target: "fixture")

    def derive(te, datapath):
        observed.append(datapath)
        return SimpleNamespace(integer=False, **datapath)

    monkeypatch.setattr(CS, "derive_binding", derive)
    binding = mesh._mesh_tile_binding("synthetic", "f32", "f32", numeric_policy=policy)
    assert binding.subnormal_operand_flush is True
    assert observed == [
        {
            **{key: value for key, value in policy.items() if key != "required_oracle_tiers"},
            "operand_dtype": "f32",
            "accum_dtype": "f32",
        }
    ]
    assert policy == original


def test_missing_floating_policy_refuses(monkeypatch):
    monkeypatch.setattr(oracle_policy, "selected_sim_via", lambda target: "fixture")
    monkeypatch.setattr(CS, "derive_binding", lambda *args: SimpleNamespace(integer=False))
    with pytest.raises(ValueError, match="explicit declared numeric_policy"):
        mesh._mesh_tile_binding("synthetic", "f32", "f32")


def test_integer_binding_remains_omission_compatible(monkeypatch):
    monkeypatch.setattr(oracle_policy, "selected_sim_via", lambda target: "fixture")
    binding = SimpleNamespace(integer=True)
    monkeypatch.setattr(CS, "derive_binding", lambda *args: binding)
    assert mesh._mesh_tile_binding("synthetic", "i8", "i32") is binding


def test_row_recursion_threads_identical_policy(monkeypatch):
    policy = {"compare": "tolerance_float", "subnormal_operand_flush": True, "atol": 0.25, "rtol": 0.02}
    seen = []

    def run(target, a, w, **kwargs):
        seen.append(kwargs["numeric_policy"])
        return [[1] for _ in a]

    monkeypatch.setattr(mesh, "run_matmul_on_mesh", run)
    result = mesh._mesh_rows(
        "synthetic",
        [[1]] * 4,
        [[1]],
        operand_dtype="f32",
        accum_dtype="f32",
        numeric_policy=policy,
        simulator=None,
        package=None,
        timeout=1,
        epilogue=None,
        acc_scale=None,
        M=4,
    )
    assert result == [[1]] * 4
    assert len(seen) == 3 and all(item is policy for item in seen)


@pytest.mark.parametrize("native", [False, True])
def test_device_build_passes_full_policy_to_binding(monkeypatch, tmp_path, native):
    from merlin.llvmlower import device_build, device_native, device_shim
    from merlin.system import derive
    from merlin.targetgen import oot_runner, target_experiment

    policy = {"compare": "tolerance_float", "subnormal_operand_flush": True, "atol": 0.25, "rtol": 0.02}

    class BindingReached(Exception):
        pass

    def binding(target, operand, accum, **kwargs):
        assert kwargs["numeric_policy"] is policy
        raise BindingReached

    monkeypatch.setattr(mesh, "_mesh_tile_binding", binding)
    monkeypatch.setattr(oot_runner, "load_package", lambda *args: object())
    monkeypatch.setattr(device_build, "objects_buildable", lambda target: None)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda target: object())
    monkeypatch.setattr(device_native, "seam_emittable", lambda target: None)
    monkeypatch.setattr(
        target_experiment, "load_capability_manifest", lambda target: SimpleNamespace(endpoint_kind="fixture")
    )
    monkeypatch.setattr(derive, "link_for", lambda *args: SimpleNamespace(device_dram_base=0))
    kwargs = dict(
        package_dir=tmp_path, workdir=tmp_path / "build", operand_dtype="f32", accum_dtype="f32", numeric_policy=policy
    )
    with pytest.raises(BindingReached):
        if native:
            device_native.build_device_native_seam("synthetic", {}, **kwargs)
        else:
            device_build.build_device_objects("synthetic", {}, {}, **kwargs)
