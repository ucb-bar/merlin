"""Public composition commands retain source pins and default allocation bounds."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

import composition
import cache_composition


@pytest.mark.parametrize("merge_adjacent", [False, True])
def test_compile_only_commands_keep_default_wall_and_source_pins(tmp_path, monkeypatch, merge_adjacent):
    driver = composition
    regions, compiled, execution = (tmp_path / name for name in ("regions", "compiled", "execution"))
    order = [f"layer{i}" for i in range(22)] + ["norm", "head"]
    for kind in order:
        folder = regions / kind
        folder.mkdir(parents=True)
        (folder / "source.json").write_text(json.dumps({
            "model_sha256": sha256(kind.encode()).hexdigest(), "source_boundaries": ["initial"]}))
    (regions / "manifest.json").write_text(json.dumps({
        "source_model_sha256": "a" * 64,
        "whole_model": {"regions": order, "links": [{"kind": kind, "inputs": []} for kind in order],
                        "embedding_execution": "framework_source_capture"}}))
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_bytes(b"source contract")
    (export / "policy.json").write_bytes(b"source policy")
    contract_sha = sha256((export / "manifest.json").read_bytes()).hexdigest()
    policy_sha = sha256((export / "policy.json").read_bytes()).hexdigest()
    commands = []

    def compile_stub(command, **kwargs):
        commands.append(command)
        # This is a command-wiring test; generated numerical evidence is tested separately.
        assert Path(command[1]).name == "compile_capsule.py"
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        source = json.loads((regions / "layer0/source.json").read_text())
        (destination / "compiled.json").write_text(json.dumps({
            "status": "COMPILED", "kind": "layer0", "source_model_sha256": source["model_sha256"],
            "constant_bytes_identical": True,
            "merge_adjacent_device_regions": "--merge-adjacent-device-regions" in command,
            "reduction_policy": command[command.index("--reduction-policy") + 1]}))
        (destination / "precision.json").write_text(json.dumps({
            "already_quantized_mask": command[command.index("--softmax-mask-policy") + 1],
            "reduction": command[command.index("--reduction-policy") + 1]}))

    argv = [driver.__file__, "--regions", str(regions), "--compiled", str(compiled),
            "--execution", str(execution), "--atlas-root", str(tmp_path),
            "--model-python", sys.executable, "--model-root", str(tmp_path),
            "--compile-only", "--stop-after", "layer0"]
    if merge_adjacent:
        argv.append("--merge-adjacent-device-regions")
    argv += ["--adapter-export", str(export), "--expected-contract-sha256", contract_sha,
                 "--expected-policy-sha256", policy_sha]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(driver.subprocess, "run", compile_stub)
    driver.main()
    command, = commands
    assert command[command.index("--allocation-timeout-ms") + 1] == "1000"
    assert ("--merge-adjacent-device-regions" in command) == merge_adjacent
    assert command[command.index("--adapter-export") + 1] == str(export)
    assert command[command.index("--expected-contract-sha256") + 1] == contract_sha
    assert command[command.index("--expected-policy-sha256") + 1] == policy_sha
    receipt = json.loads((execution / "receipt.json").read_text())
    assert receipt["status"] == "COMPILED"
    assert receipt["requested_allocation_timeout_ms"] == 1000
    assert receipt["requested_merge_adjacent_device_regions"] == merge_adjacent
    argv[argv.index("--execution") + 1] = str(tmp_path / "opposite-execution")
    if merge_adjacent:
        argv.remove("--merge-adjacent-device-regions")
    else:
        argv.append("--merge-adjacent-device-regions")
    with pytest.raises(ValueError, match="different adjacent-device selection"):
        driver.main()


@pytest.mark.parametrize("selection", [
    ["--adapter-export", "selected"],
    ["--expected-contract-sha256", "a" * 64],
    ["--adapter-export", "selected", "--expected-contract-sha256", "a" * 64],
])
@pytest.mark.parametrize("driver", [composition, cache_composition])
def test_partial_adapter_selection_refuses_before_reading_sources(monkeypatch, selection, driver):
    monkeypatch.setattr(sys, "argv", [driver.__file__, "--regions", "absent-regions",
        "--compiled", "absent-compiled", "--execution", "absent-execution", "--atlas-root", "absent-atlas",
        "--model-python", sys.executable, "--model-root", "absent-model", "--compile-only", *selection])
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2


@pytest.mark.parametrize("merge_adjacent", [False, True])
def test_cached_driver_forwards_adjacent_selection_and_refuses_reuse(tmp_path, monkeypatch, merge_adjacent):
    import cache_composition
    from atlas_mlir_native.reduction import REDUCTION_POLICY

    regions, compiled, execution = (tmp_path / name for name in ("regions", "compiled", "execution"))
    source = regions / "layer0"
    source.mkdir(parents=True)
    (source / "model.pt2").write_bytes(b"source archive")
    source_hash = sha256((source / "model.pt2").read_bytes()).hexdigest()
    (source / "source.json").write_text(json.dumps({"model_sha256": source_hash}))
    (regions / "manifest.json").write_text(json.dumps({
        "source_model_sha256": "source", "cache_phase": {"cache_capture_sha256": "capture", "phase": "prefill"}}))
    commands = []

    def compile_stub(command, **kwargs):
        commands.append(command)
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        (destination / "compiled.json").write_text(json.dumps({
            "status": "COMPILED", "kind": "layer0", "source_model_sha256": source_hash, "constant_bytes_identical": True,
            "merge_adjacent_device_regions": "--merge-adjacent-device-regions" in command,
            "io_memory_bytes": int(command[command.index("--io-memory-bytes") + 1]),
            "application_memory_bytes": int(command[command.index("--application-memory-bytes") + 1])}))
        (destination / "precision.json").write_text(json.dumps({"reduction": REDUCTION_POLICY}))

    argv = [cache_composition.__file__, "--regions", str(regions), "--compiled", str(compiled),
            "--execution", str(execution), "--atlas-root", str(tmp_path), "--model-root", str(tmp_path),
            "--model-python", sys.executable, "--compile-only"]
    if merge_adjacent:
        argv.append("--merge-adjacent-device-regions")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cache_composition, "ORDER", ["layer0"])
    monkeypatch.setattr(cache_composition, "phase_inputs", lambda *_: ({"past_length": 0}, {"layer0": []}))
    monkeypatch.setattr(cache_composition.subprocess, "run", compile_stub)
    cache_composition.main()
    command, = commands
    assert ("--merge-adjacent-device-regions" in command) == merge_adjacent
    assert command[command.index("--allocation-timeout-ms") + 1] == "1000"
    receipt = json.loads((execution / "receipt.json").read_text())
    assert receipt["requested_merge_adjacent_device_regions"] == merge_adjacent
    argv[argv.index("--execution") + 1] = str(tmp_path / "opposite-execution")
    if merge_adjacent:
        argv.remove("--merge-adjacent-device-regions")
    else:
        argv.append("--merge-adjacent-device-regions")
    with pytest.raises(ValueError, match="different adjacent-device selection"):
        cache_composition.main()
