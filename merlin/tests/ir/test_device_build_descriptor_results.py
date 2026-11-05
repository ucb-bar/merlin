"""Retain actual selected shim transport; mocked tools establish no execution."""

import json
from pathlib import Path

import pytest
from test_device_build_parallel import synthetic_build  # noqa: F401

from merlin.common import invocation_record
from merlin.llvmlower import device_build as B
from merlin.llvmlower import device_shim as D
from merlin.llvmlower.device_shim import emit_translation_unit


def test_actual_emitted_shim_identity_is_joined_to_observed_compilation(synthetic_build, monkeypatch):  # noqa: F811
    build, _, _, _, _, _ = synthetic_build
    monkeypatch.setattr(D, "emit_translation_unit", emit_translation_unit)
    monkeypatch.setattr(D, "kernel_abi_for", lambda _: D.KernelAbi("kernel", version=1))
    result = build(names=("first",), descriptor_results=True, tile_edge=2)
    assert result.ok
    unit = result.shim_unit
    assert type(unit) is D.ShimUnit and len(unit.result_identities) == 1
    assert unit.result_identities[0].symbol == "first"
    actual_c = result.shim_object.with_suffix(".c")
    assert actual_c.read_text() == unit.text
    observed = invocation_record.verify(result.shim_invocation["path"])
    assert observed["stage"] == "device_shim_compilation"
    assert observed["inputs"][0]["path"] == str(actual_c)
    assert observed["outputs"][0]["path"] == str(result.shim_object)
    owners = {row["path"] for row in observed["dependencies"]}
    assert {str(path) for path in D.descriptor_result_dependencies(unit)} <= owners
    assert json.loads(Path(result.shim_invocation["path"]).read_text()) == observed


def test_old_positional_device_build_constructor_keeps_kernel_mapping(tmp_path):
    object_path, shim = tmp_path / "kernel.o", tmp_path / "shim.o"
    result = B.DeviceBuild("neutral", (object_path, shim), shim, {"call": "kernel"})
    assert result.kernels == {"call": "kernel"} and result.shim_unit is None


@pytest.mark.parametrize("selected", [1, None, "true"])
def test_invalid_build_selection_refuses_before_package_callbacks(tmp_path, monkeypatch, selected):
    def forbidden(_):
        raise AssertionError("invalid descriptor selection reached package admission")

    monkeypatch.setattr(B, "objects_buildable", forbidden)
    with pytest.raises(ValueError, match="original boolean"):
        B.build_device_objects(
            "neutral",
            {},
            {},
            package_dir=tmp_path,
            workdir=tmp_path,
            operand_dtype="i8",
            accum_dtype="i32",
            descriptor_results=selected,
        )
