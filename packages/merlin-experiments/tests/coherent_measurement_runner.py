"""Selected stock CPU diagnostic process; no loader or measurement authority."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def main():
    config_path, elf, request_path, output = map(Path, sys.argv[1:])
    config = json.loads(config_path.read_text())
    recorder = Path(config["recorder"])
    if hashlib.sha256(recorder.read_bytes()).hexdigest() != config["recorder_sha256"]:
        raise ValueError("diagnostic recorder changed")
    spec = importlib.util.spec_from_file_location("selected_diagnostic_recorder", recorder)
    recorder_owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder_owner)
    request = json.loads(request_path.read_text())
    digest = hashlib.sha256(elf.read_bytes()).hexdigest()
    if digest != request["elf_sha256"]:
        raise ValueError("diagnostic request selected a different ELF")
    tool = Path(config["tool"])
    if hashlib.sha256(tool.read_bytes()).hexdigest() != config["tool_sha256"]:
        raise ValueError("diagnostic tool changed")
    result = recorder_owner.run(
        [str(tool), "--isa=rv64gc", str(elf)],
        directory=output.parent / "engine",
        stage="coherent_stock_cpu",
        inputs=(elf, request_path, config_path),
        dependencies=(Path(__file__).resolve(), tool, Path(recorder_owner.__file__).resolve()),
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        cwd=output.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=True,
    )
    if len(result.stdout) > 65536:
        raise ValueError("diagnostic console exceeds its closed byte budget")
    rows = result.stdout.decode("ascii").splitlines()
    frames = [row for row in rows if row.startswith("COHERENT_HEX ")]
    if len(frames) != 1 or [row for row in rows if row not in frames] != ["DONE"]:
        raise ValueError("diagnostic full-byte publication is missing or ambiguous")
    raw = bytes.fromhex(frames[0][len("COHERENT_HEX ") :])
    if len(raw) != request["product_bytes"]:
        raise ValueError("diagnostic publication omitted original objects")
    with output.open("xb") as stream:
        stream.write(raw)
    print("DONE")


if __name__ == "__main__":
    main()
