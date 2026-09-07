"""Exact host runtime grants preserve the existing answer masks."""
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from merlin.targetgen.sandbox.executable_dependencies import HostExecutableDependencies


@pytest.fixture
def setup(tmp_path):
    engine, elf, receipt = (tmp_path / name for name in ("engine", "program.elf", "build_receipt.json"))
    engine.write_bytes(b"\x7fELFengine")
    engine.chmod(0o700)
    elf.write_bytes(b"\x7fELFworkload")
    receipt.write_text('{"complete": true}')
    interpreter = Path("/usr/bin/python3").resolve()
    argv = (str(interpreter), "-c", "trusted wrapper", str(engine), str(elf))
    calls = []
    cap = HostExecutableDependencies(argv, str(engine), str(elf), str(receipt),
        tuple((str(p), hashlib.sha256(p.read_bytes()).hexdigest())
              for p in (interpreter, engine, elf, receipt)), lambda: calls.append(True))
    oracle = tmp_path / "reference.py"
    oracle.write_text("secret")
    policy = {"command_prefix": ["bwrap", "--clearenv", "--ro-bind", str(tmp_path), str(tmp_path),
        "--ro-bind", "/dev/null", str(oracle), "--chdir", str(tmp_path)],
        "answer_surfaces": [{"path": str(oracle), "kind": "file", "origin": "oracle"}]}
    return cap, policy, calls


def test_only_identity_engine_leaf_before_original_masks(setup):
    cap, policy, calls = setup
    original = list(policy["command_prefix"])
    actual = cap.extend(policy, cap.argv)
    at = original.index("/dev/null") - 1
    assert actual == original[:at] + ["--ro-bind", cap.executable_path, cap.executable_path] + original[at:]
    assert policy["command_prefix"] == original
    assert cap.extend(policy, cap.argv) == actual
    assert len(calls) == 4


@pytest.mark.parametrize("name", ["executable_path", "artifact_path", "engine_receipt_path"])
def test_changed_runtime_bytes_refused(setup, name):
    cap, policy, _ = setup
    Path(getattr(cap, name)).write_bytes(b"changed")
    with pytest.raises(ValueError, match="pin changed"):
        cap.extend(policy, cap.argv)


def test_other_command_and_missing_receipt_refused(setup):
    cap, policy, _ = setup
    with pytest.raises(ValueError, match="another trusted command"):
        cap.extend(policy, (*cap.argv, "+different-config"))
    with pytest.raises(ValueError, match="explicit engine"):
        replace(cap, file_pins=cap.file_pins[:-1]).extend(policy, cap.argv)


def test_engine_cannot_overlap_answer_surface(setup):
    cap, policy, _ = setup
    policy["command_prefix"] += ["--ro-bind", "/dev/null", cap.executable_path]
    policy["answer_surfaces"].append({"path": cap.executable_path, "kind": "file"})
    with pytest.raises(ValueError, match="overlaps"):
        cap.extend(policy, cap.argv)


def test_missing_original_mask_and_target_revalidation_failure(setup):
    cap, policy, _ = setup
    with pytest.raises(RuntimeError, match="stale target"):
        def stale():
            raise RuntimeError("stale target")
        replace(cap, command_revalidator=stale).extend(policy, cap.argv)
    p = policy["command_prefix"]
    at = p.index("/dev/null") - 1
    del p[at:at+3]
    with pytest.raises(ValueError, match="answer masks"):
        cap.extend(policy, cap.argv)
