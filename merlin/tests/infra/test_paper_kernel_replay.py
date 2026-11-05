"""Byte binding and refusal controls for saved-image replay, with no hardware calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

MODULE = repo_root() / "build_tools/scripts/replay_paper_kernels.py"
spec = importlib.util.spec_from_file_location("paper_kernel_replay", MODULE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value).encode() if isinstance(value, dict) else value)
    return {"path": name, "sha256": replay.sha(path)}


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    source = root / "product/sources/compiler"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("entrypoints: {}\n")
    (source / "compiler.py").write_text("def lower(): return 1\n")
    source_hash = replay.source_sha(source)
    interface = write(root, "product/fixtures/case/input.mlir", b"module {}")
    tensor = write(root, "product/fixtures/case/A.bin", b"\x01\x02")
    case = {
        "case_id": "case",
        "interface_sha256": interface["sha256"],
        "tensors": {"A": {"file": "/original/A.bin", "sha256": tensor["sha256"]}},
    }
    catalog = write(root, "product/cases.json", {"cases": [case]})
    study = write(root, "product/study.json", {"sources": [{"label": "compiler", "expected": source_hash}]})
    obj = write(root, "objects/compiler/case/kernel.o", b"object")
    elf = write(root, "product/measurements/image/uploaded.elf", b"\x7fELFpreserved")
    harness = write(root, "product/measurements/image/image-harness.c", b"harness")
    linker = write(root, "product/measurements/image/image-link.ld", b"linker")
    runtime = write(root, "product/measurements/image/image-rt.c", b"runtime")
    manifest = {
        "schema": "firesim-controlled-batch.v1",
        "elf_sha256": elf["sha256"],
        "harness_sha256": harness["sha256"],
        "linker_sha256": linker["sha256"],
        "runtime_sources": {"rt.c": runtime["sha256"]},
        "slots": [
            {
                "slot": 0,
                "system": "compiler",
                "case_id": "case",
                "kernel_o_sha256": obj["sha256"],
                "compiler_result": {
                    "status": "compiled",
                    "kernel_sha256": obj["sha256"],
                    "source_sha256": source_hash,
                    "interface_sha256": interface["sha256"],
                },
            }
        ],
    }
    manifest_pin = write(root, "product/measurements/image/image-manifest.json", manifest)
    hardware = write(
        root,
        "product/measurements/image/hardware.json",
        {"hardware": {"agfi": "selected-image", "runtime_hw_config": "selected-config"}},
    )
    hw_summary = write(
        root, "product/measurements/image/HW_CFG_SUMMARY", b"AGFI: selected-image\nRuntimeHWConfig: selected-config\n"
    )
    execution = write(root, "product/measurements/image/summary.json", {"elf_sha256": elf["sha256"]})
    image = {
        "directory": "product/measurements/image",
        "elf": elf,
        "manifest": manifest_pin,
        "hardware": hardware,
        "hardware_summary": hw_summary,
        "historical_execution": execution,
    }
    config = {
        "schema": replay.SCHEMA,
        "artifact_root": "product",
        "cases": catalog,
        "study": study,
        "object_roots": ["objects"],
        "images": {"image": image},
    }
    return root, config


def test_preserved_image_binds_objects_source_fixtures_and_hardware(inputs):
    root, config = inputs
    result = replay.validate(root, config, "image")
    assert result["inputs_verified"] is True
    assert result["bound_objects"] == result["bound_cases"] == 1
    assert result["bound_compiler_sources"] == ["compiler"]
    assert result["new_execution"] is False
    assert result["new_numerical_verdict"] == "not_executed"


@pytest.mark.parametrize(
    "path",
    [
        "objects/compiler/case/kernel.o",
        "product/sources/compiler/compiler.py",
        "product/fixtures/case/A.bin",
        "product/fixtures/case/input.mlir",
        "product/measurements/image/image-rt.c",
        "product/measurements/image/image-harness.c",
        "product/measurements/image/uploaded.elf",
    ],
)
def test_altered_replay_input_refuses_before_any_wrapper(inputs, path):
    root, config = inputs
    (root / path).write_bytes(b"changed")
    with pytest.raises(ValueError):
        replay.validate(root, config, "image")


def test_conflicting_duplicate_object_is_not_silently_selected(inputs):
    root, config = inputs
    config["object_roots"].append("other")
    write(root, "other/compiler/case/kernel.o", b"different-object")
    with pytest.raises(ValueError, match="conflicting"):
        replay.validate(root, config, "image")


def test_image_manifest_and_selected_elf_must_agree(inputs):
    root, config = inputs
    image = config["images"]["image"]
    path = root / image["manifest"]["path"]
    manifest = json.loads(path.read_text())
    manifest["elf_sha256"] = "0" * 64
    image["manifest"] = write(root, image["manifest"]["path"], manifest)
    with pytest.raises(ValueError, match="ELF digest"):
        replay.validate(root, config, "image")


def test_hardware_summary_must_match_selected_hardware(inputs):
    root, config = inputs
    image = config["images"]["image"]
    image["hardware_summary"] = write(root, image["hardware_summary"]["path"], b"AGFI: another\n")
    with pytest.raises(ValueError, match="hardware summary"):
        replay.validate(root, config, "image")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a//b", "a/./b", "a/../b", "a\\b"])
def test_unsafe_artifact_paths_refuse(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe artifact"):
        replay.relative(tmp_path, name)


def test_symlink_artifact_refuses(tmp_path):
    (tmp_path / "real").write_bytes(b"data")
    (tmp_path / "indirect").symlink_to("real")
    with pytest.raises(ValueError, match="symlink"):
        replay.relative(tmp_path, "indirect")


def test_fake_wrapper_exit_does_not_become_a_numerical_verdict(inputs, tmp_path):
    root, config = inputs
    wrapper = tmp_path / "local-test-wrapper"
    wrapper.write_text("#!/bin/sh\nprintf 'local mock transport only\\n'\n")
    wrapper.chmod(0o755)
    platform = tmp_path / "platform.json"
    platform.write_text('{"agfi":"selected-image","runtime_hw_config":"selected-config"}')
    output = tmp_path / "out/artifacts/mock-replay"
    result = replay.replay(
        root, config, "image", wrapper=wrapper, wrapper_sha256=replay.sha(wrapper), platform=platform, output=output
    )
    assert result["status"] == "wrapper_completed"
    assert result["new_numerical_verdict"] == "awaiting_full_uart_and_actual_hardware_checks"
    assert result["hardware_execution_verified"] is False
    assert (output / "image.elf").read_bytes() == b"\x7fELFpreserved"


def test_wrong_explicit_platform_refuses_before_launch(inputs, tmp_path):
    root, config = inputs
    wrapper = tmp_path / "local-test-wrapper"
    wrapper.write_text("#!/bin/sh\nexit 99\n")
    wrapper.chmod(0o755)
    platform = tmp_path / "platform.json"
    platform.write_text('{"agfi":"wrong","runtime_hw_config":"selected-config"}')
    output = tmp_path / "out/artifacts/must-not-exist"
    with pytest.raises(ValueError, match="platform selection"):
        replay.replay(
            root, config, "image", wrapper=wrapper, wrapper_sha256=replay.sha(wrapper), platform=platform, output=output
        )
    assert not output.exists()


def test_rebuild_plan_preserves_scripts_and_reports_missing_closure(inputs, tmp_path):
    root, config = inputs
    compile_pin = write(
        root,
        "product/scripts/compile_one.py",
        b'from pathlib import Path\nLLVM=Path("/recorded/llvm")\nRUNTIME=Path("/recorded/python")\n',
    )
    batch_pin = write(root, "product/scripts/build_batch_v2.py", b"# exact archived harness script\n")
    old_study = write(root, "product/scripts/study.json", {"sources": []})
    path = root / config["study"]["path"]
    study = json.loads(path.read_text())
    study.update(artifact_dir="/recorded/artifact", build_dir="/recorded/build")
    config["study"] = write(root, config["study"]["path"], study)
    config["rebuild"] = {
        "compile_script": compile_pin,
        "batch_script": batch_pin,
        "script_study": old_study,
        "dependency_mounts": [
            {
                "source": "dependencies/python",
                "destination_from": {"script": "compile_script", "name": "RUNTIME"},
                "inventory": None,
            }
        ],
    }
    workspace = tmp_path / "out/build/not-created-by-plan"
    result = replay.rebuild_plan(root, config, "compiler", "case", workspace)
    assert result["ready_for_source_rebuild"] is False
    assert result["study_overlay_explicit"] is True
    assert {row["reason"] for row in result["missing"]} == {
        "dependency_tree_not_restored",
        "complete_verified_inventory_not_supplied",
        "compiler_library_not_selected",
    }
    assert result["argv"][-3:] == ["/recorded/artifact/scripts/compile_one.py", "compiler", "case"]
    assert not workspace.exists()
    assert (
        root / compile_pin["path"]
    ).read_bytes() == b'from pathlib import Path\nLLVM=Path("/recorded/llvm")\nRUNTIME=Path("/recorded/python")\n'


def inventory(root, alias, destination):
    tree = root / "dependencies" / alias
    entries = []
    for path in sorted(tree.rglob("*")):
        row = {"path": path.relative_to(tree).as_posix(), "mode": oct(path.lstat().st_mode & 0o7777)}
        if path.is_symlink():
            row.update(kind="symlink", target=path.readlink().as_posix())
        elif path.is_dir():
            row.update(kind="directory")
        else:
            row.update(kind="file", sha256=replay.sha(path), bytes=path.stat().st_size)
        entries.append(row)
    import hashlib

    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    pin = write(
        root,
        "dependencies/" + alias + ".inventory.json",
        {
            "schema": "merlin.paper_dependency_inventory.v1",
            "members": entries,
            "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        },
    )
    selected = {"source": "dependencies/" + alias, "inventory": pin}
    return {
        "source": str(tree),
        "destination": destination,
        "inventory_data": replay.dependency_inventory(root, selected),
    }


def test_dependency_links_resolve_only_through_selected_inventory_roots(tmp_path):
    write(tmp_path, "dependencies/interpreter/bin/python", b"interpreter")
    (tmp_path / "dependencies/environment/bin").mkdir(parents=True)
    (tmp_path / "dependencies/environment/bin/python").symlink_to("/selected/interpreter/bin/python")
    interpreter = inventory(tmp_path, "interpreter", "/selected/interpreter")
    environment = inventory(tmp_path, "environment", "/selected/environment")
    selected = [interpreter, environment]
    replay.check_dependency_links(selected)
    owner, name = replay.resolve_dependency(Path("/selected/environment/bin/python"), selected)
    assert owner is interpreter and name == "bin/python"
    with pytest.raises(ValueError, match="outside selected"):
        replay.check_dependency_links([environment])


def test_extra_dependency_file_cannot_hide_outside_the_inventory(tmp_path):
    write(tmp_path, "dependencies/runtime/module.py", b"original")
    selected = inventory(tmp_path, "runtime", "/selected/runtime")
    write(tmp_path, "dependencies/runtime/unrecorded.py", b"extra")
    pin = {
        "source": "dependencies/runtime",
        "inventory": {
            "path": "dependencies/runtime.inventory.json",
            "sha256": replay.sha(tmp_path / "dependencies/runtime.inventory.json"),
        },
    }
    with pytest.raises(ValueError, match="unrecorded"):
        replay.dependency_inventory(tmp_path, pin)
    assert selected["inventory_data"]["entries"]["module.py"]["kind"] == "file"


def test_historical_editable_paths_stay_inaccessible_without_mounting_the_framework(tmp_path):
    write(tmp_path, "dependencies/runtime/site-packages/framework.pth", b"/historical/framework/src\n")
    write(tmp_path, "dependencies/runtime/site-packages/hook.pth", b"import retained_hook; retained_hook.install()\n")
    runtime = inventory(tmp_path, "runtime", "/selected/runtime")
    runtime["recorded_name"] = "RUNTIME"
    observed = replay.check_python_search_paths([runtime])
    assert {row["kind"] for row in observed} == {"inaccessible_in_original_namespace", "preserved_tool_startup_hook"}
    assert any(row.get("path") == "/historical/framework/src" for row in observed)


def test_rebuild_refuses_incomplete_closure_without_running_a_process(inputs, tmp_path, monkeypatch):
    root, config = inputs
    monkeypatch.setattr(replay, "rebuild_plan", lambda *a: {"ready_for_source_rebuild": False})
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(ValueError, match="dependency trees or inventories"):
        replay.execute_rebuild(
            root,
            config,
            "compiler",
            "case",
            image="image",
            workspace=tmp_path / "new",
            bwrap=tmp_path / "not-launched",
            bwrap_sha256="0" * 64,
        )


def test_separately_restored_inventory_keeps_its_exact_pin(tmp_path):
    write(tmp_path, "dependencies/runtime/module.py", b"original")
    inventory(tmp_path, "runtime", "/selected/runtime")
    original = tmp_path / "dependencies/runtime.inventory.json"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    original.rename(metadata / original.name)
    selected = {
        "source": "dependencies/runtime",
        "inventory": {"path": "dependencies/runtime.inventory.json", "sha256": replay.sha(metadata / original.name)},
    }
    assert replay.dependency_inventory(tmp_path, selected, metadata)["entries"]["module.py"]["kind"] == "file"
    (metadata / original.name).write_text("{}")
    with pytest.raises(ValueError, match="missing or changed artifact"):
        replay.dependency_inventory(tmp_path, selected, metadata)


def test_successful_mock_rebuild_records_object_identity_without_numerical_claim(inputs, tmp_path, monkeypatch):
    from types import SimpleNamespace

    root, config = inputs
    source_hash = replay.source_sha(root / "product/sources/compiler")
    manifest = replay.read(root / config["images"]["image"]["manifest"]["path"])
    slot = manifest["slots"][0]
    tool = tmp_path / "bwrap"
    tool.write_bytes(b"selected executable")
    tool.chmod(0o755)
    workspace = tmp_path / "new-build"
    plan = {
        "ready_for_source_rebuild": True,
        "source_sha256": source_hash,
        "argv": ["bwrap", "--proc", "/proc", "python", "script", "compiler", "case"],
        "verified_tools": [],
    }
    monkeypatch.setattr(replay, "rebuild_plan", lambda *a: plan)

    def compile_fixture(argv, **kwargs):
        assert argv.index("--setenv") < argv.index("python")
        write(workspace, "kernels/compiler/case/kernel.o", b"object")
        write(
            workspace,
            "kernels/compiler/case/result.json",
            {
                "status": "compiled",
                "source_unchanged": True,
                "source_sha256": source_hash,
                "interface_sha256": slot["compiler_result"]["interface_sha256"],
                "kernel_sha256": slot["kernel_o_sha256"],
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(replay.subprocess, "run", compile_fixture)
    result = replay.execute_rebuild(
        root, config, "compiler", "case", image="image", workspace=workspace, bwrap=tool, bwrap_sha256=replay.sha(tool)
    )
    assert result["status"] == "compiled" and result["matches_preserved_object"] is True
    assert result["numerical_verdict"] == "not_executed" and result["hardware_execution"] is False


def test_optional_dangling_link_is_reported_but_selected_tool_cannot_use_it(tmp_path):
    write(tmp_path, "dependencies/runtime/bin/other", b"usable")
    (tmp_path / "dependencies/runtime/bin/optional").symlink_to("missing")
    selected = inventory(tmp_path, "runtime", "/selected/runtime")
    observed = replay.check_dependency_links([selected])
    assert len(observed) == 1 and observed[0]["path"] == "/selected/runtime/bin/optional"
    tools = write(tmp_path, "tools.json", {"compiler": {"path": "/selected/runtime/bin/optional", "sha256": "0" * 64}})
    with pytest.raises(replay.MissingDependencyTarget):
        replay.verify_recorded_tools(tmp_path, {"rebuild": {"toolchains": tools}}, [selected])
