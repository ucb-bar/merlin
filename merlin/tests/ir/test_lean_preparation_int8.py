"""Int8 capture has an independent roster and grants no policy proof."""

import json

import pytest
from test_lean_boundary_plans import SOURCE
from test_lean_preparation_capture import _options, _ordinary_selection  # noqa: F401

from merlin.llvmlower.quant_scope import FEATURE
from merlin.runtime.backends import zephyr_model
from merlin.verify.lean import plans, preparation, stages

CONFIG = dict(int8_compute=True, quant_passes=["contraction_int8"], features=frozenset({FEATURE}))


def test_int8_recording_is_byte_neutral_and_keeps_unchanged_boundary(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    plain = tmp_path / "plain"
    plain.mkdir()
    output, features = zephyr_model.prepare_for_lowering(source, plain, **CONFIG)
    captured = preparation.capture(source, tmp_path / "recorded", _options(), **CONFIG)
    assert captured.target == output.read_bytes()
    assert captured.features == features == CONFIG["features"]
    published, _ = preparation.validate(captured, _options(), captured.target)
    names = [json.loads((p / "meta.json").read_text())["boundary"] for p in published]
    assert names[names.index("prepare-qinner-lift") + 1] == "prepare-quantize-int8"
    assert names[names.index("prepare-quantize-int8") + 1] == "prepare-lower-quant-ext"
    quant = published[names.index("prepare-quantize-int8")]
    assert (quant / "before.mlir").read_bytes() == (quant / "after.mlir").read_bytes()


@pytest.mark.parametrize(
    "overrides",
    [
        {"features": frozenset()},
        {"quant_passes": None},
        {"quant_passes": []},
        {"quant_passes": ["conv_int8"]},
        {"quant_passes": ["contraction_int8", "conv_int8"]},
        {"quant_passes": ["contraction_int8", "contraction_int8"]},
    ],
)
def test_other_int8_configurations_refuse_before_execution(tmp_path, monkeypatch, overrides):
    monkeypatch.setattr(zephyr_model, "prepare_for_lowering", lambda *a, **k: pytest.fail("entered preparation"))
    with pytest.raises(ValueError):
        preparation.capture(tmp_path / "absent", tmp_path / "work", _options(), **(CONFIG | overrides))
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("fault", ["omitted", "reordered"])
def test_int8_roster_refuses_missing_or_moved_slot(tmp_path, monkeypatch, fault):
    original = stages.record_boundary
    deferred = []

    def record(name, entry, before, after, **kwargs):
        if name == "prepare-quantize-int8":
            deferred.append((name, entry, before, after))
            return None
        result = original(name, entry, before, after, **kwargs)
        if name == "prepare-lower-quant-ext" and deferred and fault == "reordered":
            original(*deferred.pop())
        return result

    monkeypatch.setattr(stages, "record_boundary", record)
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    with pytest.raises(plans.PlanMismatch):
        preparation.capture(source, tmp_path / "work", _options(), **CONFIG)


def test_group_routing_is_explicit_assumption_and_not_refinement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.llvmlower import device_offload
    from merlin.llvmlower.device_build import DeviceRouting

    device = DeviceRouting(
        "test-device", tmp_path, "i8", "i32", select=lambda _: True, granularity=device_offload.BY_GROUP
    )

    def rewrite(path, *args, **kwargs):
        path.write_text(path.read_text() + "\n")
        return SimpleNamespace(moved=0, signatures=[], skipped=[])

    monkeypatch.setattr(device_offload, "rewrite_prepared_file", rewrite)
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    plain_work = tmp_path / "plain"
    plain_work.mkdir()
    plain, _ = zephyr_model.prepare_for_lowering(source, plain_work, device=device)
    captured = preparation.capture(source, tmp_path / "recorded", _options(), device=device)
    assert captured.target == plain.read_bytes()
    paths = stages.completed(captured.root)
    seam = next(p for p in paths if p.name.endswith("prepare-device-routing"))
    meta = json.loads((seam / "meta.json").read_text())
    assert meta["extra"]["trust_class"] == "assumed"
    assert meta["extra"]["claim_type"] == "contractSeam"
    for before, after in zip(paths, paths[1:]):
        assert (before / "after.mlir").read_bytes() == (after / "before.mlir").read_bytes()
    with pytest.raises(ValueError, match="segmented composition"):
        preparation.validate(captured, _options(), captured.target)


@pytest.mark.parametrize(
    "field", ["prepared_transform", "catalog_builder", "post_offload_transform", "final_elf_audit"]
)
def test_group_routing_refuses_provider_callbacks(tmp_path, field):
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.llvmlower.device_offload import BY_GROUP

    device = DeviceRouting("test-device", tmp_path, "i8", "i32", granularity=BY_GROUP, **{field: lambda *a: None})
    with pytest.raises(ValueError, match="provider callbacks"):
        preparation.capture(tmp_path / "absent", tmp_path / "work", _options(), device=device)
    assert not (tmp_path / "work").exists()


def test_firing_int8_contraction_is_byte_neutral(tmp_path):
    from test_quant_transposed_weight import IR

    source = tmp_path / "source.mlir"
    source.write_text(IR)
    plain_work = tmp_path / "plain"
    plain_work.mkdir()
    plain, _ = zephyr_model.prepare_for_lowering(source, plain_work, **CONFIG)
    captured = preparation.capture(source, tmp_path / "recorded", _options(), **CONFIG)
    assert captured.target == plain.read_bytes()
    paths = stages.completed(captured.root)
    quant = next(p for p in paths if p.name.endswith("prepare-quantize-int8"))
    assert json.loads((quant / "meta.json").read_text())["changed"]
    assert '"arith.extsi"' in (quant / "after.mlir").read_text()
    for before, after in zip(paths, paths[1:]):
        assert (before / "after.mlir").read_bytes() == (after / "before.mlir").read_bytes()


def test_composite_exception_restores_child_recording(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(SOURCE)

    def fail(module):
        assert not stages.enabled()
        raise RuntimeError("pass failed")

    with stages.capture(tmp_path / "stages"):
        with pytest.raises(RuntimeError, match="pass failed"):
            stages.run("failing", fail, module, _record_nested=False)
        assert stages.enabled()
        stages.run("restored", lambda module: None, module)
    assert len(stages.completed(tmp_path / "stages")) == 1


def test_latest_output_ignores_other_recording_runs(tmp_path, monkeypatch):
    root = tmp_path / "stages"
    with stages.capture(root):
        stages.record_boundary("own", "test", "before", "own-output")
        foreign = stages.record_boundary("foreign", "test", "before", "foreign-output")
        meta = json.loads((foreign / "meta.json").read_text())
        meta["run"] = "another-run"
        (foreign / "meta.json").write_text(json.dumps(meta))
        assert stages.latest_after() == "own-output"
