#!/usr/bin/env python3
"""Offline integrity and semantic-witness verification for this artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = {
    "rp10_pass": ("pass", "0x80000086", 120_000, 32,
                  "merlin.muon-result-page.v1", "legacy"),
    "rp10_negative_control": ("fail", "0x800000c6", 120_000, 32,
                              "merlin.muon-result-page.v1", "legacy"),
    "r4_rmsnorm_observed_fail": ("pass", "0x80000186", 360_000, 256,
                                 "merlin.muon-result-mailbox.v2", "initial_mailbox"),
    "rp12_embed_scale": ("pass", "0x80000186", 360_000, 256,
                         "merlin.muon-result-mailbox.v2", "sequence_token"),
    "rp12_negative_control": ("fail", "0x800001c6", 360_000, 256,
                              "merlin.muon-result-mailbox.v2", "sequence_token"),
}
FORBIDDEN_NAMES = {"golden.yaml", "result_carrier.c", "kernel.soc.elf", ".merlin_build_key"}
UNINDEXED_ALLOWLIST = {"SHA256SUMS"}  # a checksum file cannot include its own digest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_pc(console: str) -> str | None:
    marker = "[gsim-probe final] rocket_pc="
    for line in reversed(console.splitlines()):
        if marker in line:
            return line.split(marker, 1)[1].split(maxsplit=1)[0].lower()
    return None


def publication_failures(root: Path) -> list[str]:
    """Verify that the checksum index exactly covers the recursive publication."""
    failures: list[str] = []
    checksum = root / "SHA256SUMS"
    try:
        lines = checksum.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"cannot read SHA256SUMS: {exc}"]

    listed: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        fields = line.split("  ", 1)
        if len(fields) != 2 or not fields[0] or not fields[1]:
            failures.append(f"malformed SHA256SUMS line {number}")
            continue
        expected, rel = fields
        relpath = Path(rel)
        if relpath.is_absolute() or ".." in relpath.parts:
            failures.append(f"unsafe SHA256SUMS path: {rel}")
            continue
        if rel in listed:
            failures.append(f"duplicate SHA256SUMS path: {rel}")
            continue
        listed[rel] = expected

    actual: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in UNINDEXED_ALLOWLIST:
            continue
        actual.add(rel)
        if path.name in FORBIDDEN_NAMES:
            failures.append(f"forbidden published file: {rel}")

    for rel in sorted(actual - set(listed)):
        failures.append(f"unlisted published file: {rel}")
    for rel in sorted(set(listed) - actual):
        failures.append(f"listed file is absent: {rel}")
    for rel in sorted(actual & set(listed)):
        if digest(root / rel) != listed[rel]:
            failures.append(f"hash mismatch: {rel}")
    return failures


def main() -> int:
    failures = publication_failures(ROOT)

    for name, (status, pc, cap, elements, schema, protocol) in CASES.items():
        case = ROOT / "cases" / name
        console = (case / "gsim_console.log").read_text(encoding="utf-8")
        manifest = json.loads((case / "result_page.json").read_text(encoding="utf-8"))
        harness_case = ROOT / "cases" / ("rp12_embed_scale" if name == "rp12_negative_control" else name)
        harness = (harness_case / "main.c").read_text(encoding="utf-8")
        if final_pc(console) != pc:
            failures.append(f"{name}: final PC is not declared {status} witness {pc}")
        if f"FINISHED: cycles={cap}" not in console:
            failures.append(f"{name}: missing bounded {cap}-cycle observation")
        if manifest.get("schema") != schema:
            failures.append(f"{name}: wrong result-page schema")
        outputs = manifest.get("outputs") or []
        if len(outputs) != 1 or outputs[0].get("elements") != elements:
            failures.append(f"{name}: wrong declared output extent")
        if schema.endswith("mailbox.v2"):
            mailbox = manifest.get("mailbox") or {}
            if mailbox.get("words") != 32 or "soc_address" in outputs[0]:
                failures.append(f"{name}: manifest exposes more than the fixed 32-word mailbox")
            common = ("merlin_result_mailbox[32]", "_base+=32u")
            if protocol == "sequence_token":
                required = common + ("merlin_result_status[1]=_count",
                                     "merlin_result_status[0]=(0x4d525231u^_merlin_sequence)",
                                     "merlin_result_status[2]!=(0x4d524131u^_merlin_sequence)",
                                     'fence rw,rw', 'fence r,rw')
            else:
                required = common + ("merlin_result_status[1]=_merlin_sequence",
                                     "merlin_result_status[2]=_count",
                                     "merlin_result_status[4]!=_merlin_sequence")
            if any(token not in harness for token in required):
                failures.append(f"{name}: harness lacks streaming READY(sequence,count)/ACK protocol")
        else:
            if "merlin_result_status" not in harness or "merlin_result_0" not in harness:
                failures.append(f"{name}: legacy harness lacks declared linker-visible results")
            if "while(merlin_result_status[2]" not in harness:
                failures.append(f"{name}: legacy harness lacks READY/ACK protocol")
        result = case / "adapter_result.json"
        if result.is_file():
            verdict = json.loads(result.read_text(encoding="utf-8")).get("numeric_verdict", {})
            if verdict.get("status") != status or verdict.get("elements_checked") != elements:
                failures.append(f"{name}: adapter verdict disagrees with sealed witness")

    positive = ROOT / "cases/rp10_pass"
    negative = ROOT / "cases/rp10_negative_control"
    if digest(positive / "kernel.radiance.elf") != digest(negative / "kernel.radiance.elf"):
        failures.append("RP10 positive/negative submitted kernel ELFs are not identical")

    identity = json.loads((ROOT / "submission_identity.json").read_text(encoding="utf-8"))
    hashes = identity.get("submitted_muon_elf_sha256") or {}
    rp12_hash = digest(ROOT / "cases/rp12_embed_scale/kernel.radiance.elf")
    if not hashes.get("byte_identical") or hashes.get("positive") != hashes.get("negative"):
        failures.append("RP12 positive/negative receipt does not declare identical submitted ELFs")
    if hashes.get("positive") != rp12_hash:
        failures.append("RP12 submitted ELF disagrees with its negative-control identity receipt")
    carriers = identity.get("private_carrier_sha256") or {}
    if carriers.get("byte_identical") or carriers.get("positive") == carriers.get("negative"):
        failures.append("RP12 trusted carrier did not change for the perturbed golden")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS: public-safe hashes, mailbox protocol, bounds, and five final-PC witnesses verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
