#!/usr/bin/env python3
"""Offline integrity and falsifiability checks for the LayerNorm L3 pair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _read(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    positive = _read("cases/positive/receipt.json")
    negative = _read("cases/negative/receipt.json")
    assert positive["capsule"] == negative["capsule"] == "RP5_layernorm_fp32_pt"
    assert positive["numeric_verdict"]["status"] == "pass"
    assert positive["numeric_verdict"]["elements_checked"] == 256
    assert negative["numeric_verdict"]["status"] == "fail"
    assert positive["submitted_kernel_elf_sha256"] == negative["submitted_kernel_elf_sha256"]
    for name in ("command_buffer.json", "lowered.llvm.mlir", "selection_report.json"):
        assert _sha(f"cases/positive/{name}") == _sha(f"cases/negative/{name}")
    cb = _read("cases/positive/command_buffer.json")
    assert not any(key.startswith("_oracle") for key in cb)
    report = cb["params"]["kernel_family_selection"]
    assert report["request"] == {
        "op": "layernorm", "dtype": "fp32", "shape": {"cols": 16, "rows": 16}}
    assert report["selected_family"] == "kernels/layernorm"
    assert len(report["decisions"]) == 23
    assert {d["family"] for d in report["decisions"] if d["status"] == "disabled"} == {
        "kernels/flash_attention_mx_gemma", "kernels/flash_attention_mx_fp6"}
    print("ok: identical LayerNorm compiler artifact passes and rejects perturbed-golden control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
