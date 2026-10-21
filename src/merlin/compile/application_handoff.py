"""Render a C/H consumer view of an existing application compilation."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path


def application_handoff(source, interface, regions, files, native_root):
    """Copy final images and constants; keep platform linking with the consumer."""
    output = {}
    for key in ("source", "header"):
        member = interface[key]
        output["handoff/" + Path(member).name] = files["c-package/" + member]
    images, calls = {}, []
    for region in regions:
        bindings = []
        for member in region["native_programs"]:
            path = Path(member)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe native program in C handoff")
            root = native_root / path
            if root.is_symlink() or (root / "program.bin").is_symlink():
                raise ValueError("linked native program in C handoff")
            manifest = json.loads((root / "manifest.json").read_text())
            binary = (root / "program.bin").read_bytes()
            digest = sha256(binary).hexdigest()
            if digest != manifest["binary_sha256"]:
                raise ValueError("native program changed during handoff")
            filename = f"programs/{digest}.bin"
            output["handoff/" + filename] = binary
            images[digest] = {
                "path": filename,
                "sha256": digest,
                "bytes": len(binary),
                "execution": json.loads((root / "execution_plan.json").read_text())
                if (root / "execution_plan.json").is_file()
                else None,
            }
            bindings.append(digest)
        calls.append({"region": region["id"], "placement": region["placement"], "images": bindings})
    constants = []
    for constant in source.constants:
        payload = bytes.fromhex(constant.data_hex)
        digest = sha256(payload).hexdigest()
        filename = f"constants/{digest}.bin"
        output["handoff/" + filename] = payload
        constants.append(
            {
                "id": constant.node_id,
                "path": filename,
                "sha256": digest,
                "encoding": constant.encoding,
                "type": source.node(constant.node_id).type.record(),
            }
        )
    abi = {
        "schema": "merlin.c_handoff.v1",
        "target_identity": source.target_identity,
        "request_digest": source.digest(),
        "platform_status": "unbound",
        "source": Path(interface["source"]).name,
        "header": Path(interface["header"]).name,
        "entry": interface["entry"],
        "binding": interface.get("platform_binding"),
        "inputs": [{"id": key, "type": source.node(key).type.record()} for key, _ in source.input_storages],
        "outputs": [{"id": key, "type": source.node(key).type.record()} for key in source.outputs],
        "constants": constants,
        "programs": list(images.values()),
        "calls": calls,
        "embedded_programs_and_constants": True,
        "files": {key.removeprefix("handoff/"): sha256(value).hexdigest() for key, value in output.items()},
    }
    output["handoff/abi.json"] = (json.dumps(abi, indent=2, sort_keys=True) + "\n").encode()
    output["handoff/README.md"] = f"""# Generated application

Compile `{abi["source"]}` as C11 and include `{abi["header"]}`. The source embeds
the program and constants; `programs/` and `constants/` provide matching binary
copies for inspection. `abi.json` describes tensors, calls and device addresses.
Keep the platform's startup, linker script, drivers and host executable build.
No host main function or ELF is included. Match the application's floating-point
ABI required by the selected platform.

Use the generated binding functions listed in abi.json when present. Keep one
device context per physical accelerator. Supply the actual memory map, memory
callbacks and cache maintenance. Do not reuse a timed-out call's buffers until
completion or reset. This package is unbound to a bitstream; compilation and
software checks do not establish FPGA execution.
""".encode()
    return output
