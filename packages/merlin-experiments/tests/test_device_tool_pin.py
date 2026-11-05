"""Real tool observations conform to the unchanged strict native file contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from merlin.llvmlower.device_build import _observed_build_tool
from merlin.targetgen.native_model_execution import NativeModelExecutionError, _pinned_native_file


def _produce(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_bytes(b"actual input")
    command = [sys.executable, "-c", "from pathlib import Path; import sys; "
               "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())", str(source), str(output)]
    result, pin, elapsed = _observed_build_tool(
        command, timeout=3, directory=tmp_path, stage="device_llvm_translation", inputs=(source,), output=output,
    )
    return result, pin, elapsed


def test_real_producer_pin_passes_strict_validator_and_preserves_separate_timing(tmp_path):
    result, pin, elapsed = _produce(tmp_path)
    assert result.returncode == 0
    assert set(pin) == {"path", "sha256", "size_bytes"}
    path = _pinned_native_file({"translation": pin}, "translation")
    document = json.loads(path.read_bytes())
    assert document["status"] == "completed" and document["inputs_unchanged"] is True
    assert document["tool_elapsed_s"] == elapsed and elapsed >= 0
    assert Path(document["outputs"][0]["path"]).read_bytes() == b"actual input"


@pytest.mark.parametrize("tamper", ["content", "size", "extra_timing"])
def test_strict_validator_still_refuses_changed_or_open_pin(tmp_path, tamper):
    _result, pin, elapsed = _produce(tmp_path)
    if tamper == "content":
        path = Path(pin["path"])
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"completed", b"tampered!"))  # same byte count
    elif tamper == "size":
        pin["size_bytes"] += 1
    else:
        pin["elapsed_s"] = elapsed
    with pytest.raises(NativeModelExecutionError):
        _pinned_native_file({"translation": pin}, "translation")
