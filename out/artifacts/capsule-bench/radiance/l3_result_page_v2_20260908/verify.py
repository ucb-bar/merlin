#!/usr/bin/env python3
"""Offline integrity and semantic-witness verification for this artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = {
    "rp10_pass": ("pass", "0x80000086", 120_000, 32),
    "rp10_negative_control": ("fail", "0x800000c6", 120_000, 32),
    "r4_rmsnorm_observed_fail": ("fail", "0x800000c6", 360_000, 256),
    "rp12_embed_scale": ("fail", "0x800000c6", 360_000, 256),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_pc(console: str) -> str | None:
    marker = "[gsim-probe final] rocket_pc="
    for line in reversed(console.splitlines()):
        if marker in line:
            return line.split(marker, 1)[1].split(maxsplit=1)[0].lower()
    return None


def main() -> int:
    failures: list[str] = []
    checksum_lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    forbidden = ("golden.yaml", "result_carrier.c", "kernel.soc.elf")
    for line in checksum_lines:
        expected, rel = line.split("  ", 1)
        if any(rel.endswith(name) for name in forbidden):
            failures.append(f"private golden-derived artifact listed for publication: {rel}")
        path = ROOT / rel
        if not path.is_file() or digest(path) != expected:
            failures.append(f"hash mismatch: {rel}")

    for name, (status, pc, cap, elements) in CASES.items():
        case = ROOT / "cases" / name
        console = (case / "gsim_console.log").read_text(encoding="utf-8")
        manifest = json.loads((case / "result_page.json").read_text(encoding="utf-8"))
        harness = (case / "main.c").read_text(encoding="utf-8")
        if final_pc(console) != pc:
            failures.append(f"{name}: final PC is not declared {status} witness {pc}")
        if f"FINISHED: cycles={cap}" not in console:
            failures.append(f"{name}: missing bounded {cap}-cycle observation")
        if manifest.get("schema") != "merlin.muon-result-page.v1":
            failures.append(f"{name}: wrong result-page schema")
        outputs = manifest.get("outputs") or []
        if len(outputs) != 1 or outputs[0].get("elements") != elements:
            failures.append(f"{name}: wrong declared output extent")
        if "merlin_result_status" not in harness or "merlin_result_0" not in harness:
            failures.append(f"{name}: harness lacks declared linker-visible results")
        if "while(merlin_result_status[2]" not in harness:
            failures.append(f"{name}: harness lacks READY/ACK protocol")
        result = case / "adapter_result.json"
        if result.is_file():
            verdict = json.loads(result.read_text(encoding="utf-8")).get("numeric_verdict", {})
            if verdict.get("status") != status or verdict.get("elements_checked") != elements:
                failures.append(f"{name}: adapter verdict disagrees with sealed witness")

    positive = ROOT / "cases/rp10_pass"
    negative = ROOT / "cases/rp10_negative_control"
    if digest(positive / "kernel.radiance.elf") != digest(negative / "kernel.radiance.elf"):
        failures.append("RP10 positive/negative submitted kernel ELFs are not identical")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS: public-safe hashes, bounds, manifests, and all four final-PC witnesses verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
