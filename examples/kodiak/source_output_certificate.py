#!/usr/bin/env python3
"""Create and recheck a full-output Spike certificate for a TorchAO FireSim leg.

This is an independent evidence tool. It never changes a FireSim plan or its
acceptance status. Stage the same-run debugger files below the job directory
before invoking ``create``; ``verify`` reopens all original bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from examples.lib import firesim_flow as flow  # noqa: E402
from merlin.runtime.backends import output_identity, zephyr_model  # noqa: E402

SCHEMA = "merlin.source_full_output_certificate.v1"
FILES = ("raw", "console", "debugger_commands", "debugger_log", "disassembly", "spike_version")


class CertificateError(ValueError):
    """A required byte, identity, or full numerical verdict is absent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _evidence_file(job_dir: Path, path: Path) -> dict[str, str]:
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked evidence file: {path}")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(job_dir.resolve())
    except ValueError as exc:
        raise CertificateError(f"evidence must be staged inside the job directory: {path}") from exc
    return {"path": str(relative), "sha256": sha256(resolved)}


def _plan(job_dir: Path, leg: str) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = job_dir / flow.MANIFEST
    require(manifest_path.is_file(), f"missing FireSim plan: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == flow.SCHEMA, "wrong FireSim plan schema")
    require(leg in flow.LEGS and leg in manifest.get("legs", {}), f"unknown FireSim leg: {leg}")
    source = manifest.get("source") or {}
    require(source.get("dtype") == "int8_w8a8", "plan is not an exact TorchAO source W8A8 plan")
    reference = source.get("reference") or {}
    require(
        reference.get("kind") == "torchao_w8a8_framework" and reference.get("file") == "golden.npy",
        "reference must be the independent TorchAO golden.npy",
    )
    bundle = Path(manifest["bundle"])
    golden_path = bundle / "golden.npy"
    framework_path = Path(source["framework_result"])
    row = manifest["legs"][leg]
    elf = Path(row["elf"])
    require(
        golden_path.is_file() and framework_path.is_file() and elf.is_file(),
        "missing golden, framework result, or staged ELF",
    )
    require(sha256(golden_path) == reference.get("sha256"), "TorchAO golden changed")
    require(sha256(framework_path) == source.get("framework_result_sha256"), "framework result changed")
    capture_sha = flow._capture_sha256(bundle)
    require(
        capture_sha == source.get("capture_sha256") == manifest.get("capture_sha256"),
        "source capture changed",
    )
    require(sha256(elf) == row.get("elf_sha256"), "staged ELF changed")
    require(elf.resolve().is_relative_to(job_dir.resolve()), "staged ELF is outside the job directory")
    require(bool(row.get("build_hash")), "missing image build hash")
    identity = row.get("spike_output_identity") or {}
    whole_hash = identity.get("out_hash") or {}
    require(whole_hash.get("algo") == "fnv1a32", "Spike plan lacks a supported whole-output hash")
    golden = np.load(golden_path, allow_pickle=False)
    require(
        golden.dtype == np.float32 and golden.size > 0 and np.isfinite(golden).all(), "TorchAO golden is not finite f32"
    )
    require(
        whole_hash.get("elems") == int(golden.size)
        and 0 < identity.get("prefix_elems", 0) <= int(golden.size)
        and bool(identity.get("prefix_sha256")),
        "Spike plan has incomplete output identity",
    )
    plan = {
        "job_id": job_dir.name,
        "workload": manifest["name"],
        "leg": leg,
        "board": manifest["board"],
        "hw_config": manifest["hw_config"],
        "harts": manifest["harts"],
        "capture_sha256": capture_sha,
        "framework_result_sha256": sha256(framework_path),
        "reference_kind": reference["kind"],
        "golden_sha256": sha256(golden_path),
        "elf_sha256": sha256(elf),
        "build_hash": str(row["build_hash"]),
        "spike_output_identity": identity,
    }
    return plan, golden


def _hex_in(text: str, value: int) -> bool:
    return f"0x{value:x}" in text.lower() or f"{value:x}:" in text.lower()


def _verification(job_dir: Path, plan: dict[str, Any], evidence: dict[str, Any], golden: np.ndarray) -> dict[str, Any]:
    files = evidence.get("files") or {}
    require(set(files) == set(FILES), "required debugger or raw evidence is missing")
    paths: dict[str, Path] = {}
    for name in FILES:
        item = files[name]
        path = job_dir / item["path"]
        actual = _evidence_file(job_dir, path)
        require(actual == item, f"{name} evidence changed")
        paths[name] = path
    spike = Path(evidence["spike_executable"])
    require(spike.is_file() and sha256(spike) == evidence.get("spike_sha256"), "Spike executable changed")
    version = paths["spike_version"].read_text().strip()
    require(bool(version), "Spike version evidence is empty")
    commands = paths["debugger_commands"].read_text()
    debug_log = paths["debugger_log"].read_text()
    disassembly = paths["disassembly"].read_text()
    address = int(evidence["output_address"])
    halt_pc = int(evidence["halt_pc"])
    require(address > 0 and halt_pc > 0, "output address and halt PC must be positive")
    require("MERLIN_OUTPUT_PTR[0]" in commands, "debugger commands did not resolve MERLIN_OUTPUT_PTR[0]")
    require(_hex_in(debug_log, address) and _hex_in(debug_log, halt_pc), "debugger log omits the pointer or halt PC")
    require(
        _hex_in(disassembly, halt_pc) and "MERLIN_OUTPUT_PTR" in disassembly,
        "halt PC or output symbol is absent from exact ELF disassembly",
    )

    console = paths["console"].read_text()
    lines = console.splitlines()
    hash_lines = [i for i, line in enumerate(lines) if line.startswith("HASH ")]
    out_lines = [i for i, line in enumerate(lines) if line.startswith("OUT ")]
    done_lines = [i for i, line in enumerate(lines) if line.strip() == "DONE"]
    require(len(hash_lines) == len(out_lines) == len(done_lines) == 1, "console needs one HASH, OUT, and DONE")
    require(hash_lines[0] < out_lines[0] < done_lines[0], "console output order is not HASH, OUT, DONE")
    out_parts = lines[out_lines[0]].split()
    require(len(out_parts) == 2 + int(out_parts[1]), "console OUT count does not match its printed bits")
    try:
        parsed = zephyr_model._parse_console(console, 0)
    except (ValueError, zephyr_model.ZephyrModelError) as exc:
        raise CertificateError(f"same-run Spike console could not be parsed: {exc}") from exc
    identity = output_identity.output_identity(parsed)
    require(str(parsed["metrics"].get("build_hash")) == plan["build_hash"], "console image build hash differs")
    require(identity == plan["spike_output_identity"], "same-run console differs from planned Spike identity")
    whole_hash = identity["out_hash"]
    raw = paths["raw"].read_bytes()
    require(len(raw) == 4 * int(golden.size), "raw dump is not the complete f32 output")
    full = np.frombuffer(raw, dtype="<f4").reshape(golden.shape)
    require(np.isfinite(full).all(), "raw output contains non-finite values")
    prefix_bytes = raw[: 4 * identity["prefix_elems"]]
    require(hashlib.sha256(prefix_bytes).hexdigest() == identity["prefix_sha256"], "raw prefix differs from console")
    raw_fnv = zephyr_model.out_hash_fnv1a32(full)
    require(raw_fnv == whole_hash["value"], "raw whole-output FNV differs from console")
    grade = zephyr_model._gate(full, {"w8a8": golden}, min_coverage=1)
    require(
        grade.get("ok") is True
        and grade.get("tier_ok") == "w8a8"
        and grade.get("comparison_complete") is True
        and grade.get("n_compared") == grade.get("n_reference") == int(golden.size),
        "complete TorchAO W8A8 numerical gate failed",
    )
    actual_argmax = np.argmax(full, axis=-1).tolist()
    golden_argmax = np.argmax(golden, axis=-1).tolist()
    require(actual_argmax == golden_argmax, "per-token argmax differs from TorchAO")
    return {
        "full_shape": list(golden.shape),
        "dtype": "<f4",
        "raw_bytes": len(raw),
        "raw_sha256": sha256(paths["raw"]),
        "raw_fnv1a32": raw_fnv,
        "prefix_elems": identity["prefix_elems"],
        "prefix_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
        "full_gate": grade,
        "per_token_argmax": actual_argmax,
        "spike_version": version,
        "scope": "complete_spike_output_vs_torchao_golden",
    }


def create(
    job_dir: Path, leg: str, *, evidence_files: dict[str, Path], spike: Path, output_address: int, halt_pc: int
) -> Path:
    job_dir = job_dir.resolve()
    plan, golden = _plan(job_dir, leg)
    evidence = {
        "files": {name: _evidence_file(job_dir, evidence_files[name]) for name in FILES},
        "spike_executable": str(spike.resolve()),
        "spike_sha256": sha256(spike),
        "output_address": output_address,
        "halt_pc": halt_pc,
    }
    verification = _verification(job_dir, plan, evidence, golden)
    result = {
        "schema": SCHEMA,
        "job_dir": str(job_dir),
        "plan": plan,
        "plan_sha256": canonical_sha256(plan),
        "evidence": evidence,
        "verification": verification,
    }
    destination = job_dir / "full_output" / leg / "certificate.json"
    require(not destination.exists(), f"certificate already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return destination


def verify(certificate: Path) -> dict[str, Any]:
    saved = json.loads(certificate.read_text())
    require(saved.get("schema") == SCHEMA, "wrong certificate schema")
    job_dir = Path(saved["job_dir"]).resolve()
    require(
        certificate.resolve() == job_dir / "full_output" / saved["plan"]["leg"] / "certificate.json",
        "certificate moved",
    )
    plan, golden = _plan(job_dir, saved["plan"]["leg"])
    require(plan == saved["plan"] and canonical_sha256(plan) == saved["plan_sha256"], "immutable plan identity changed")
    result = _verification(job_dir, plan, saved["evidence"], golden)
    require(result == saved["verification"], "certificate verification changed")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create", help="grade an exact leg and seal its staged evidence")
    make.add_argument("job_dir", type=Path)
    make.add_argument("--leg", choices=flow.LEGS, required=True)
    for name in FILES:
        make.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    make.add_argument("--spike", type=Path, required=True)
    make.add_argument("--output-address", type=lambda x: int(x, 0), required=True)
    make.add_argument("--halt-pc", type=lambda x: int(x, 0), required=True)
    check = sub.add_parser("verify", help="reopen the plan and every sealed evidence file")
    check.add_argument("certificate", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            path = create(
                args.job_dir,
                args.leg,
                evidence_files={name: getattr(args, name) for name in FILES},
                spike=args.spike,
                output_address=args.output_address,
                halt_pc=args.halt_pc,
            )
            print(json.dumps({"status": "verified_full_output_on_spike", "certificate": str(path)}))
        else:
            result = verify(args.certificate)
            print(json.dumps({"status": "verified_full_output_on_spike", "plan_sha256": result["plan_sha256"]}))
    except (CertificateError, KeyError, OSError, ValueError) as exc:
        parser.exit(1, f"source output certificate refused: {exc}\n")


if __name__ == "__main__":
    main()
