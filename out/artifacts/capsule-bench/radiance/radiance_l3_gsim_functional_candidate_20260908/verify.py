#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    receipt = json.loads((ROOT / "receipt.json").read_text())
    emulator = ROOT / receipt["emulator"]["path"]
    console = ROOT / receipt["smoke"]["console_path"]
    assert emulator.stat().st_size == receipt["emulator"]["bytes"]
    assert sha256(emulator) == receipt["emulator"]["sha256"]
    assert sha256(console) == receipt["smoke"]["console_sha256"]
    text = console.read_text(errors="replace")
    assert text.count(receipt["smoke"]["completion_witness"]) == 1
    assert "Exit status: 0" in text
    assert "writes_resultpage=0" in text
    print(json.dumps({"ok": True, "status": receipt["status"]}, indent=2))


if __name__ == "__main__":
    main()
