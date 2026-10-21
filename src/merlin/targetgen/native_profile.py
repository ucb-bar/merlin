"""Compile bounded variants and measure them through an explicit external runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from time import monotonic

from merlin.semantic_compiler.codegen import FACTORS, CodegenOptions, check_profile, codegen_options
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import open_native_snapshot
from merlin.semantic_compiler.target_binding import load_native_target_binding, verify_native_publication


def run_profile(args):
    """Runner data is evaluation-only; ordinary compilation never invokes it."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = open_native_snapshot(Path(args.snapshot))
    binding = load_native_target_binding(args.support)
    if snapshot.profile.digest() != binding.profile().digest():
        raise ValueError("native snapshot differs from selected support profile")
    request = KernelRequest.from_record(json.loads(Path(args.request).read_text()))
    abi = binding.plan_boundary_abi(request) if args.abi == "auto" else json.loads(Path(args.abi).read_text())
    limits = (
        SearchLimits.from_record(json.loads(Path(args.search_limits).read_text()))
        if args.search_limits
        else SearchLimits()
    )
    runner = json.loads(Path(args.runner_command).read_text())
    if not isinstance(runner, list) or not runner or any(not isinstance(x, str) or not x for x in runner):
        raise ValueError("runner command must be a nonempty JSON argv array")
    key_factory = getattr(binding, "tuning_key", None)
    if key_factory is None:
        raise ValueError("selected support does not expose reduction-unroll profiling")
    with codegen_options(CodegenOptions(program_size_limit_bytes=args.program_size_limit_bytes)):
        key = key_factory(snapshot, request, abi["fixed_inputs"], tuple(abi["fixed_outputs"]), limits)
    rows = []
    runner_identity = {
        "argv": runner,
        "files": {
            str(Path(item).resolve()): hashlib.sha256(Path(item).read_bytes()).hexdigest()
            for item in runner
            if Path(item).is_file()
        },
    }
    for factor in FACTORS:
        destination = out / f"unroll-{factor}"
        row = {"factor": factor, "status": "failed"}
        try:
            compile_start = monotonic()
            with codegen_options(CodegenOptions(str(factor), args.program_size_limit_bytes)):
                manifest = binding.compile(
                    snapshot,
                    request,
                    fixed_inputs=abi["fixed_inputs"],
                    fixed_outputs=tuple(abi["fixed_outputs"]),
                    target_source=Path(args.target_source),
                    destination=destination,
                    limits=limits,
                )
            row["compile_seconds"] = monotonic() - compile_start
            verify_native_publication(
                destination,
                manifest,
                engine="merlin_native",
                request_digest=request.digest(),
                target_identity=request.target_identity,
            )
            binary = (destination / "program.bin").read_bytes()
            identity = hashlib.sha256(binary).hexdigest()
            result_path = (out / f"unroll-{factor}-measurements.json").resolve()
            runner_start = monotonic()
            result = subprocess.run(
                [
                    *runner,
                    "--compiled",
                    str(destination.resolve()),
                    "--out",
                    str(result_path),
                    "--warmup",
                    "1",
                    "--repetitions",
                    "5",
                ],
                capture_output=True,
                text=True,
                timeout=args.runner_timeout,
            )
            row["runner_seconds"] = monotonic() - runner_start
            (out / f"unroll-{factor}-runner.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"runner exited with status {result.returncode}")
            measured = json.loads(result_path.read_text())
            if (
                not isinstance(measured, dict)
                or measured.get("program_sha256") != identity
                or measured.get("correctness") != "passed"
            ):
                raise ValueError("runner did not check the selected binary")
            if hashlib.sha256((destination / "program.bin").read_bytes()).hexdigest() != identity:
                raise ValueError("runner changed the compiled binary")
            verify_native_publication(
                destination,
                manifest,
                engine="merlin_native",
                request_digest=request.digest(),
                target_identity=request.target_identity,
            )
            samples = measured.get(args.objective)
            available = (
                isinstance(samples, list)
                and len(samples) == 5
                and all(type(value) is int and value >= 0 for value in samples)
            )
            row.update(
                measured, factor=factor, bytes=len(binary), status="passed" if available else "measurement_unavailable"
            )
            if not available:
                row["reason"] = "runner did not measure the selected objective"
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            row.setdefault("compile_seconds", monotonic() - compile_start)
            row.update(status=getattr(error, "status", "failed"), reason=str(error))
        rows.append(row)
    profile = {
        "schema": "merlin.native_tuning_profile.v1",
        "engine": "merlin_native",
        "key": key,
        "objective": args.objective,
        "runner": runner_identity,
        "variants": rows,
    }
    check_profile(profile)
    (out / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    passed = sum(row["status"] == "passed" for row in rows)
    return {
        "status": "measured" if passed else "failed",
        "passed": passed,
        "failed": len(rows) - passed,
        "profile": str(out / "profile.json"),
    }


def command(args):
    try:
        report = run_profile(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}))
        return 1
    print(json.dumps(report))
    return 0 if report["status"] == "measured" else 1
