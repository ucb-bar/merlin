"""Exercise ordinary sidecar/build transport, without issuing device authority."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower import device_build as DB
from merlin.llvmlower import device_catalog as DC
from merlin.llvmlower import device_preparation as DP
from merlin.llvmlower.device_offload import BY_GROUP, DeviceRewrite, Routed


def write_source(tmp_path, *, grouped=False):
    source = DeviceRewrite(
        "neutral",
        signatures={"entry": (2, 3, 4)},
        routed=(Routed("entry", (2, 3), (4,), ("i8", "i8", "i32"), operation_id="source-op"),),
        granularity=BY_GROUP if grouped else "contraction",
        entries={"entry": {"op": "matmul", "M": 2, "N": 3, "K": 4}} if grouped else {},
    )
    source.write_sidecar(tmp_path)
    return source


def controls(monkeypatch):
    events = []
    target = object()
    monkeypatch.setattr(DP, "require_device_offload_selection", lambda *args: events.append("source"))
    monkeypatch.setattr(DB, "require_device_elf_admission", lambda *args: events.append("elf"))

    def require_target(*args):
        events.append("target")
        return target

    monkeypatch.setattr(DB, "require_device_target_configuration", require_target)
    return events, target


def run(tmp_path, device, events, **options):
    def flags():
        events.append("flags")
        return ["--target=neutral-unknown-elf", "-O2"]

    return DP.build_prepared_device_objects(
        device,
        tmp_path,
        offload_selection="original-source",
        device_admission_selection="original-elf",
        target_configuration_selection="original-target",
        device_ir_budget="original-budget",
        shim_cflags=flags,
        **options,
    )


def test_absent_route_retains_empty_result_and_never_queries_tools(tmp_path, monkeypatch):
    events, _ = controls(monkeypatch)
    sidecar, arguments, built, artifacts = run(tmp_path, None, events)
    assert sidecar == {}
    assert arguments == {"signatures": {}, "dtypes": {}, "entries": None, "call_buffers": None}
    assert built is artifacts is None
    assert events == ["source"]


@pytest.mark.parametrize("grouped", [False, True])
def test_actual_sidecar_arguments_flags_order_and_artifact_bytes_survive(tmp_path, monkeypatch, grouped, capsys):
    events, target = controls(monkeypatch)
    source = write_source(tmp_path, grouped=grouped)
    source = replace(
        source,
        expected_interfaces={"entry": {"mlir": "original interface", "sha256": "a" * 64}},
        package_sha256="b" * 64,
        call_buffers={"entry": [{"role": "output", "shape": [2, 3], "dtype": "i32"}]},
    )
    source.write_sidecar(tmp_path)
    device = DB.DeviceRouting("neutral", tmp_path / "package", "i8", "i32", granularity=source.granularity)
    observed = []
    build = DB.DeviceBuild(
        "neutral",
        objects=(tmp_path / "device/entry.o", tmp_path / "device/device_shim.o"),
        shim_object=tmp_path / "device/device_shim.o",
        kernels={"entry": "actual_kernel_7"},
        custody={"entry": {"translation": {"retained": True}}},
        built_from={"entry": "group_program" if grouped else "bare_extents"},
        object_dedup={"unique_artifacts": 1},
        build_metrics={"entry": {"retained": "actual metric"}},
    )

    def compile_objects(*args, **kwargs):
        events.append("build")
        observed.append((args, kwargs))
        for name in (
            "entry.device.mlir",
            "entry.ll",
            "entry.o",
            "device_shim.o",
            "entry.optimized.ll",
            "entry.optimized.mlir",
        ):
            path = tmp_path / "device" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
        for step, name in (("optimization", "entry.optimized.ll"), ("optimized_import", "entry.optimized.mlir")):
            output = tmp_path / "device" / name
            receipt = output.with_suffix(output.suffix + ".invocation.json")
            receipt.write_text(json.dumps({"outputs": [{"path": str(output), "sha256": sha256_file(output)}]}))
            build.custody["entry"][step] = {"path": str(receipt), "sha256": sha256_file(receipt)}
        return build

    monkeypatch.setattr(DB, "build_device_objects", compile_objects)
    sidecar, arguments, actual, artifacts = run(tmp_path, device, events)
    assert actual is build
    assert sidecar == source.to_sidecar()
    assert arguments["signatures"] == source.signatures
    assert arguments["entries"] == (source.entries or None)
    assert arguments["call_buffers"] == source.call_buffers
    args, kwargs = observed.pop()
    assert args == ("neutral", {"entry": (2, 3, 4)}, {"entry": ("i8", "i8", "i32")})
    assert kwargs == dict(
        package_dir=device.package_dir,
        workdir=tmp_path / "device",
        operand_dtype="i8",
        accum_dtype="i32",
        codegen_target="riscv",
        numeric_policy=None,
        entries=source.entries or None,
        stop_on_first_failure=True,
        cflags=["--target=neutral-unknown-elf", "-O2"],
        expected_interfaces=source.expected_interfaces,
        package_sha256=source.package_sha256,
        call_buffers=source.call_buffers,
        device_ir_budget="original-budget",
        target_configuration=target,
    )
    assert events == ["source", "elf", "source", "target", "flags", "build", "target"]
    row = artifacts["kernels"]["entry"]
    assert row["kernel_symbol"] == "actual_kernel_7"
    assert row["custody"] is build.custody["entry"]
    assert row["build_metrics"] is build.build_metrics["entry"]
    for record in row["artifacts"].values():
        path = Path(record["path"])
        assert record["sha256"] == sha256_file(path)
        assert record["size_bytes"] == path.stat().st_size
    assert "1 kernel(s) + shim" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["no_device", "wrong_device", "missing_group", "incomplete_build"])
def test_original_route_refusals_precede_artifact_publication(tmp_path, monkeypatch, damage):
    events, _ = controls(monkeypatch)
    source = write_source(tmp_path, grouped=damage == "missing_group")
    device = DB.DeviceRouting("neutral", tmp_path, "i8", "i32", granularity=source.granularity)
    if damage == "no_device":
        device = None
    elif damage == "wrong_device":
        device = replace(device, device="other")
    elif damage == "missing_group":
        path = tmp_path / "device_signatures.json"
        record = json.loads(path.read_text())
        record["entries"] = {}
        path.write_text(json.dumps(record))

    def compile_objects(*args, **kwargs):
        events.append("build")
        return DB.DeviceBuild("neutral", skipped=(("entry", "original decline"),))

    monkeypatch.setattr(DB, "build_device_objects", compile_objects)
    diagnostic = {
        "no_device": "no `device=` routing",
        "wrong_device": "does not match selected device",
        "missing_group": "complete stated group program",
        "incomplete_build": "did not build every routed kernel",
    }[damage]
    with pytest.raises((ValueError, RuntimeError), match=diagnostic):
        run(tmp_path, device, events)
    assert ("build" in events) == (damage == "incomplete_build")
    assert not (tmp_path / "device/entry.o").exists()


def test_exact_selection_checks_run_before_and_after_actual_builder(tmp_path, monkeypatch):
    events, _ = controls(monkeypatch)
    source = replace(
        write_source(tmp_path),
        package_sha256="a" * 64,
        transport="llvm",
        abi_sha256="b" * 64,
        certification_sha256=("c" * 64,),
    )
    source.write_sidecar(tmp_path)
    exact = SimpleNamespace(
        by_operation_id={"source-op": object()},
        package_sha256="a" * 64,
        transport="llvm",
        abi_sha256="b" * 64,
        certification_sha256=("c" * 64,),
        check_package=lambda package: events.append("package"),
        check_backend_contract=lambda: (_ for _ in ()).throw(ValueError("original contract changed")),
    )
    device = DB.DeviceRouting("neutral", tmp_path, "i8", "i32", exact_selection=exact)

    def compile_objects(*args, **kwargs):
        events.append("build")
        return DB.DeviceBuild("neutral")

    monkeypatch.setattr(DB, "build_device_objects", compile_objects)
    with pytest.raises(ValueError, match="original contract changed"):
        run(tmp_path, device, events)
    assert events == ["source", "elf", "source", "target", "flags", "build", "target", "package"]
    exact.by_operation_id = {"foreign-op": object()}
    events.clear()
    with pytest.raises(RuntimeError, match="lost exact operation or package identity"):
        run(tmp_path, device, events)
    assert "flags" not in events and "build" not in events


def test_catalog_route_keeps_original_arguments_and_has_no_llvm_inventory(tmp_path, monkeypatch):
    events, _ = controls(monkeypatch)
    monkeypatch.setattr(DB, "require_device_target_configuration", lambda *args: None)
    record = write_source(tmp_path).to_sidecar()
    record.update(catalog_manifest="selected-manifest", catalog_object="selected-object", model_sha256="a" * 64)
    (tmp_path / "device_signatures.json").write_text(json.dumps(record))
    device = DB.DeviceRouting("neutral", tmp_path, "i8", "i32", catalog_manifest="selected-manifest")
    observed = []
    built = DB.DeviceBuild(
        "neutral",
        objects=(tmp_path / "catalog.o",),
        shim_object=tmp_path / "catalog.o",
        kernels={"entry": "kernel"},
    )
    monkeypatch.setattr(DC, "build_catalog_objects", lambda *args, **kwargs: observed.append((args, kwargs)) or built)
    _, _, actual, artifacts = run(tmp_path, device, events)
    assert actual is built and artifacts is None
    args, kwargs = observed.pop()
    assert args == ("neutral", {"entry": (2, 3, 4)}, {"entry": ("i8", "i8", "i32")})
    assert kwargs["manifest_path"] == "selected-manifest"
    assert kwargs["object_path"] == "selected-object"
    assert kwargs["source_sha256"] == "a" * 64
    assert kwargs["routed"] == record["routed"]
    assert kwargs["cflags"] == ["--target=neutral-unknown-elf", "-O2"]


@pytest.mark.parametrize("drift", [False, True])
def test_ordinary_build_pins_moved_owner_before_preparation_and_completion(tmp_path, monkeypatch, drift):
    from merlin.llvmlower import weight_prepack
    from merlin.runtime.backends import spike_model

    owner = tmp_path / "device_preparation.py"
    owner.write_bytes(Path(DP.__file__).read_bytes())
    digest = sha256_file(owner)
    monkeypatch.setattr(DP, "__file__", str(owner))
    copy_tool = shutil.which("cp")
    assert copy_tool is not None

    class ReachedPreparation(Exception):
        pass

    def prepare(bundle, work, features, *, compilation):
        # This callback is reached by the ordinary build before any compiler.
        recorded = compilation.record["preparation"]["device_preparation"]
        assert recorded == {"path": str(owner), "sha256": digest, "bytes": owner.stat().st_size}
        assert compilation.record["status"] == "prepared"
        source, output = work / "transport-input", work / "transport-output"
        source.write_bytes(b"recipe transport test, not a compiler product")
        compilation.run(
            [copy_tool, source, output],
            runner=lambda argv: subprocess.run(argv, capture_output=True),
            inputs=[source],
            output=output,
        )
        if drift:
            content = owner.read_bytes()
            owner.write_bytes(b"X" + content[1:])
            with pytest.raises(ValueError, match="preparation changed"):
                compilation.completed_product(output, kind="llvm_ir")
        else:
            compilation.completed_product(output, kind="llvm_ir")
            assert compilation.record["status"] == "completed"
        raise ReachedPreparation

    monkeypatch.setattr(weight_prepack, "prepare_build_bundle", prepare)
    with pytest.raises(ReachedPreparation):
        spike_model.build(tmp_path / "model", tmp_path / "work", backend="scalar")
