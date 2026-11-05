"""Pure original routing/proof transport only; no native or target execution."""

import builtins
import json
from dataclasses import replace

import pytest
from test_device_routing_llvm_transform import source
from test_selected_fusion_contracts import inputs

from merlin.llvmlower import device_build as B
from merlin.llvmlower import device_preparation as P
from merlin.llvmlower.target_schedule import TargetScheduleSelection


def selected(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    contracts, kernel, objects, adapter, cap, encoding, limits, options = inputs(tmp_path)
    return TargetScheduleSelection(
        contracts,
        objects,
        adapter,
        encoding,
        limits,
        parameters={
            **options,
            "fusion_emitters": {cap.readout.selector: cap.emit},
            "fusion_qualification_identity": "original-neutral-only",
        },
    )


def route(tmp_path):
    return B.DeviceRouting("neutral", tmp_path, "i8", "i32", target_schedule=selected(tmp_path / "selection"))


def prepare(device, frozen, tmp_path, flags=lambda: []):
    return P.build_prepared_device_objects(
        device,
        tmp_path,
        offload_selection=frozen,
        device_admission_selection=None,
        target_configuration_selection=None,
        device_ir_budget=None,
        shim_cflags=flags,
    )


def test_default_none_has_no_selected_import_or_callbacks(tmp_path, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in ("target_schedule", "device_target_passes")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    device = B.DeviceRouting("neutral", tmp_path, "i8", "i32")
    assert P.freeze_device_offload_selection(device) is None
    assert P.require_device_offload_selection(device, None) is None
    assert B.freeze_device_target_configuration(device) is None
    assert P.build_prepared_device_objects(
        None,
        tmp_path,
        offload_selection=None,
        device_admission_selection=None,
        target_configuration_selection=None,
        device_ir_budget=None,
        shim_cflags=lambda: pytest.fail("queriedtools"),
    )[-2:] == (None, None)


@pytest.mark.parametrize("bad", [{}, "candidate", object()])
def test_untyped_selection_refuses(tmp_path, bad):
    with pytest.raises(TypeError, match="original TargetScheduleSelection"):
        B.DeviceRouting("neutral", tmp_path, "i8", "i32", target_schedule=bad)


def test_schedule_and_configuration_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(route(tmp_path), target_configuration=object())


@pytest.mark.parametrize("damage", ["drop", "replace", "attach", "context"])
def test_original_owner_frozen_before_preparation(tmp_path, damage):
    device = route(tmp_path)
    if damage == "attach":
        owner = device.target_schedule
        device = replace(device, target_schedule=None)
    frozen = P.freeze_device_offload_selection(device)
    if damage == "context":
        device.target_schedule.selected_contracts.facts_path.write_text("{}")
    else:
        object.__setattr__(
            device,
            "target_schedule",
            None if damage == "drop" else owner if damage == "attach" else replace(device.target_schedule),
        )
    with pytest.raises(ValueError, match="changed|reselect"):
        P.require_device_offload_selection(device, frozen)


def test_wrong_target_refuses(tmp_path):
    device = replace(route(tmp_path), device="foreign")
    with pytest.raises(ValueError, match="different target"):
        P.freeze_device_offload_selection(device)


def test_shadowed_method_refuses_before_callback(tmp_path):
    device = route(tmp_path)
    object.__setattr__(device.target_schedule, "record", lambda: pytest.fail("shadow executed"))
    with pytest.raises(ValueError, match="original bound"):
        P.freeze_device_offload_selection(device)


def test_empty_or_catalog_route_cannot_ignore_selection(tmp_path, monkeypatch):
    device = route(tmp_path)
    frozen = P.freeze_device_offload_selection(device)
    with pytest.raises(ValueError, match="routed package-emitted"):
        prepare(device, frozen, tmp_path)
    source(tmp_path)
    side = tmp_path / "device_signatures.json"
    d = json.loads(side.read_text())
    d["catalog_manifest"] = "original-catalog"
    side.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="routed package-emitted"):
        prepare(device, frozen, tmp_path)


def test_late_flag_callback_cannot_change_original_selection(tmp_path, monkeypatch):
    device = route(tmp_path)
    frozen = P.freeze_device_offload_selection(device)
    source(tmp_path)
    monkeypatch.setattr(B, "require_device_elf_admission", lambda *a: None)
    monkeypatch.setattr(B, "build_device_objects", lambda *a, **k: pytest.fail("changed reached builder"))

    def flags():
        object.__setattr__(device, "target_schedule", None)
        return []

    # The fixture sidecar is intentionally synthetic; align only target identity.
    side = tmp_path / "device_signatures.json"
    d = json.loads(side.read_text())
    d["device"] = "neutral"
    side.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="changed"):
        prepare(device, frozen, tmp_path, flags)


def test_builder_cannot_omit_consumed_schedule_receipt(tmp_path, monkeypatch):
    device = route(tmp_path)
    frozen = P.freeze_device_offload_selection(device)
    source(tmp_path)
    side = tmp_path / "device_signatures.json"
    d = json.loads(side.read_text())
    d["device"] = "neutral"
    side.write_text(json.dumps(d))
    monkeypatch.setattr(B, "require_device_elf_admission", lambda *a: None)
    seen = []

    def build(*args, **kw):
        seen.append(kw["target_schedule"])
        return B.DeviceBuild("neutral", objects=(tmp_path / "x.o",), kernels={"a": "actual_kernel"}, custody={"a": {}})

    monkeypatch.setattr(B, "build_device_objects", build)
    with pytest.raises(ValueError, match="not consumed"):
        prepare(device, frozen, tmp_path)
    assert seen == [device.target_schedule]


def test_none_keeps_old_economic_routing_record_exact(tmp_path, monkeypatch):
    from test_device_offload_selection import fixture

    from merlin.llvmlower import device_offload_selection as S

    bound, *_ = fixture(tmp_path, monkeypatch)
    actual = S._route(bound.routing)
    original_fields = S.fields
    monkeypatch.setattr(S, "fields", lambda obj: tuple(f for f in original_fields(obj) if f.name != "target_schedule"))
    expected = S._route(bound.routing)
    assert actual == expected and "target_schedule" not in actual[0]


def test_economic_source_choice_needs_separate_qualification(tmp_path, monkeypatch):
    from test_device_offload_selection import fixture

    from merlin.llvmlower import device_offload_selection as S

    bound, *_ = fixture(tmp_path, monkeypatch)
    owner = selected(tmp_path / "new-selection")
    with pytest.raises(ValueError, match="separate economic cost qualification"):
        S._route(replace(bound.routing, target_schedule=owner))
