#!/usr/bin/env python3
"""Execute one host-analyzed whole-program candidate on GSIM with its frozen model oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import perf_agent_stage as PAS
import produce_gsim_certificate as PGC


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-worker-result", type=Path, required=True)
    parser.add_argument("--capsule-manifest", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.output.exists() or args.output.is_symlink():
        parser.error("timeout must be positive and output must be fresh")
    source = PAS._mapping_file(args.analysis_worker_result.resolve(strict=True))
    analysis, artifacts = source.get("analysis"), source.get("artifacts")
    if not isinstance(analysis, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("analysis worker result lacks analysis/artifacts mappings")
    if (analysis.get("candidate_sha256") != args.candidate_sha256
            or artifacts.get("candidate_sha256") != args.candidate_sha256):
        raise ValueError("analysis worker result is not bound to the requested candidate")
    command_buffer, lowered = artifacts.get("command_buffer"), artifacts.get("lowered_text")
    if not isinstance(command_buffer, Mapping) or not isinstance(lowered, str):
        raise ValueError("analysis worker result lacks retained command buffer/lowered text")
    command_buffer_path = (args.analysis_worker_result.resolve().parent
                           / "compiler_scratch/candidate/command_buffer.json")
    command_buffer_text = command_buffer_path.read_text(encoding="utf-8")
    cb_sha256 = hashlib.sha256(command_buffer_text.encode()).hexdigest()
    lowered_sha256 = hashlib.sha256(lowered.encode()).hexdigest()
    if (cb_sha256 != artifacts.get("candidate_command_buffer_sha256")
            or lowered_sha256 != artifacts.get("candidate_lowered_sha256")):
        raise ValueError("retained execution artifacts do not match their analysis digests")
    if json.loads(command_buffer_text) != command_buffer:
        raise ValueError("retained command-buffer file and worker result differ")

    command_buffer, expected, matches_expected, semantic_reference = PGC._semantic_oracle(
        args.capsule_manifest.resolve(strict=True), command_buffer)
    workdir = args.workdir.resolve()
    if workdir.exists() or workdir.is_symlink():
        raise ValueError(f"workdir must be fresh: {workdir}")
    workdir.mkdir(parents=True)
    from merlin.targetgen.contract.compile import compile_lowered_to_elf
    started = time.monotonic()
    elf = Path(compile_lowered_to_elf(
        command_buffer, lowered, workdir, target=args.target)).resolve(strict=True)
    build_seconds = time.monotonic() - started
    elf_sha256 = PGC._sha_file(elf)

    from merlin.runtime.backends import base
    from merlin.runtime.commandbuffer import declared_output_dtypes
    backend = base.get_backend(args.target)
    if not backend.available("gsim"):
        raise ValueError(f"{args.target} GSIM backend is unavailable")
    started = time.monotonic()
    console = backend.run_elf(elf, simulator="gsim", timeout=args.timeout)
    run_seconds = time.monotonic() - started
    if PGC._sha_file(elf) != elf_sha256:
        raise ValueError("ELF changed during GSIM execution")
    outputs, metrics = backend.parse_output(console)
    outputs = base.decode_float_readback(outputs, declared_output_dtypes(command_buffer))
    if not matches_expected(outputs):
        raise ValueError("GSIM whole-program output does not match the frozen model golden")
    console_path = workdir / "gsim_console.txt"
    console_path.write_text(console, encoding="utf-8")
    result = {
        "schema": "whole_program_gsim_smoke_v1", "status": "passed",
        "candidate_sha256": args.candidate_sha256,
        "analysis_worker_result": str(args.analysis_worker_result.resolve()),
        "analysis_worker_result_sha256": PGC._sha_file(args.analysis_worker_result),
        "capsule_manifest_sha256": PGC._sha_file(args.capsule_manifest),
        "command_buffer_sha256": cb_sha256, "lowered_sha256": lowered_sha256,
        "elf": str(elf), "elf_sha256": elf_sha256,
        "build_wall_seconds": build_seconds, "run_wall_seconds": run_seconds,
        "metrics": _jsonable(metrics), "outputs": _jsonable(outputs),
        "expected": _jsonable(expected), "semantic_reference": _jsonable(semantic_reference),
        "console_sha256": PGC._sha_file(console_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(PAS._canonical_json(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
