#!/usr/bin/env python3
"""Verify that the frozen four-norm evidence is explicitly a failed candidate."""
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / "cases/positive/receipt.json").read_text(encoding="utf-8"))
    report = json.loads((root / "cases/positive/selection_report.json").read_text(encoding="utf-8"))
    assert receipt["capsule"] == "RP13_gemma_4norm_bf16_pt"
    assert receipt["expected_control"] == "pass"
    assert receipt["numeric_verdict"]["status"] == "fail"
    assert receipt["numeric_verdict"]["elements_checked"] == 256
    assert report["selected_family"] == "kernels/gemma_4norm"
    print("ok: four-norm candidate is recorded as failed and must remain unregistered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
