"""Snapshot the bytes a reproduced RTL elaboration receipt binds into Phase 0 evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def snapshot_elaboration(
    production: Mapping[str, Any],
    consistency: Mapping[str, Any],
    observe: Callable[..., bytes | None],
) -> None:
    """Observe the receipt and every member it hashes; refuse a byte that changed since selection.

    Only a selection whose consistency view reproduced the exact FIRRTL carries a receipt. Each
    member -- the configuration source, the elaborator tool, both runs' FIRRTL and their logs -- is
    read through ``observe`` so it enters the evidence bundle, and is compared with the digest the
    receipt recorded; a mismatch means the evidence would describe different bytes than were checked.
    """
    if (consistency.get("elaboration") or {}).get("status") != "reproduced_exact_firrtl":
        return

    def member(path: str | Path, expected: str, role: str, *, label: str | None = None) -> bytes:
        captured = observe(path, role, required=True)
        if hashlib.sha256(captured).hexdigest() != expected:
            raise ValueError(f"selected elaboration {label or role} changed during evidence snapshot")
        return captured

    selected = production["production"]["elaboration"]
    receipt_bytes = member(selected["path"], selected["sha256"], "rtl-elaboration-receipt", label="receipt")
    receipt = json.loads(receipt_bytes)
    source = receipt["source"]
    member(Path(source["root"]) / source["config_file"], source["config_sha256"], "rtl-elaboration-config")
    member(receipt["tool"]["path"], receipt["tool"]["sha256"], "rtl-elaboration-tool")
    for run in receipt["runs"]:
        member(run["firrtl"], run["firrtl_sha256"], "rtl-elaboration-output")
        parent = Path(run["firrtl"]).parent
        member(parent / "stdout.log", run["stdout_sha256"], "rtl-elaboration-stdout")
        member(parent / "stderr.log", run["stderr_sha256"], "rtl-elaboration-stderr")
