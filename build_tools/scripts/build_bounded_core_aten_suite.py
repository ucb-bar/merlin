#!/usr/bin/env python3
"""Generate and exactly reduce the bounded Core ATen semantic/pairwise suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.targetgen import core_aten_bounded, core_aten_cover, core_aten_eqsat
from merlin.targetgen import core_aten_cases as case_registry
from merlin.targetgen._aten_opset_worker import core_opset
from merlin.targetgen.core_aten_bounded import (
    BoundedCoverageProfile,
    _one_overload_pool,
    assemble_bounded_core_aten_suite,
    bounded_summary,
)
from merlin.targetgen.core_aten_cases import case_document, core_aten_cases
from merlin.targetgen.core_aten_cover import json_bytes

_HERE = Path(__file__).resolve()


def _generator_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        _HERE,
        *(Path(module.__file__) for module in (core_aten_bounded, case_registry, core_aten_cover, core_aten_eqsat)),
    ):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _worker(args: argparse.Namespace) -> int:
    profile = BoundedCoverageProfile.from_dict(json.loads(args.worker_profile_json))
    registry = core_aten_cases()
    if args.worker_overload not in registry:
        raise RuntimeError(f"worker overload is not in the canonical registry: {args.worker_overload}")
    pool = _one_overload_pool(
        args.worker_overload,
        case_document(registry[args.worker_overload]),
        profile,
    )
    record = {
        "schema_version": 1,
        "overload": args.worker_overload,
        "pytorch_version": args.worker_pytorch_version,
        "profile_sha256": profile.sha256,
        "generator_sha256": args.worker_generator_sha256,
        "pool": pool,
    }
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.worker_output.with_name(f".{args.worker_output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json_bytes(record))
    temporary.replace(args.worker_output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory (defaults beneath the configured generated-artifact root)",
    )
    parser.add_argument(
        "--maximum-candidates-per-overload",
        type=int,
        default=BoundedCoverageProfile.maximum_candidates_per_overload,
    )
    parser.add_argument("--jobs", type=int, default=4, help="isolated overload workers to run concurrently")
    parser.add_argument("--worker-timeout", type=int, default=900, help="seconds allowed per overload worker")
    parser.add_argument("--no-resume", action="store_true", help="regenerate matching per-overload pool shards")
    parser.add_argument("--worker-overload", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-profile-json", help=argparse.SUPPRESS)
    parser.add_argument("--worker-pytorch-version", help=argparse.SUPPRESS)
    parser.add_argument("--worker-generator-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_overload:
        required = (
            args.worker_output,
            args.worker_profile_json,
            args.worker_pytorch_version,
            args.worker_generator_sha256,
        )
        if not all(required):
            parser.error("bounded-suite worker invocation is incomplete")
        return _worker(args)
    if args.maximum_candidates_per_overload < 1:
        parser.error("--maximum-candidates-per-overload must be positive")
    if args.jobs < 1 or args.worker_timeout < 1:
        parser.error("--jobs and --worker-timeout must be positive")
    destination = args.output_dir or artifacts_dir() / "verification" / "core-aten" / "bounded"
    opset = core_opset()
    profile = BoundedCoverageProfile(maximum_candidates_per_overload=args.maximum_candidates_per_overload)
    destination.mkdir(parents=True, exist_ok=True)
    pool_root = destination / "pools"
    pool_root.mkdir(parents=True, exist_ok=True)
    generator_sha256 = _generator_digest()
    profile_json = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def pool_path(overload: str) -> Path:
        return pool_root / f"{overload.replace('.', '__')}.json"

    def load_matching(path: Path, overload: str):
        if args.no_resume or not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        expected = {
            "overload": overload,
            "pytorch_version": opset["torch"],
            "profile_sha256": profile.sha256,
            "generator_sha256": generator_sha256,
        }
        return record.get("pool") if all(record.get(key) == value for key, value in expected.items()) else None

    pools = {}
    pending = []
    for overload in opset["ops"]:
        cached = load_matching(pool_path(overload), overload)
        if cached is None:
            pending.append(overload)
        else:
            pools[overload] = cached
    print(f"bounded pools: {len(pools)} reused, {len(pending)} isolated workers pending", flush=True)

    def run_worker(overload: str):
        command = [
            sys.executable,
            str(_HERE),
            "--worker-overload",
            overload,
            "--worker-output",
            str(pool_path(overload)),
            "--worker-profile-json",
            profile_json,
            "--worker-pytorch-version",
            opset["torch"],
            "--worker-generator-sha256",
            generator_sha256,
        ]
        environment = dict(os.environ)
        environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        process = subprocess.run(
            command,
            cwd=str(repo_root()),
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.worker_timeout,
        )
        if process.returncode != 0:
            diagnostic = (process.stderr or process.stdout or "worker wrote no diagnostic")[-4000:]
            raise RuntimeError(f"{overload} worker exited {process.returncode}: {diagnostic}")
        record = json.loads(pool_path(overload).read_text(encoding="utf-8"))
        return overload, record["pool"]

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_worker, overload): overload for overload in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            overload = futures[future]
            try:
                name, pool = future.result()
                pools[name] = pool
            except Exception as exc:  # noqa: BLE001 -- retain every isolated crash/timeout diagnostic
                failures.append(f"{overload}: {type(exc).__name__}: {exc}")
            if completed % 10 == 0 or completed == len(pending):
                print(f"bounded pools: completed {completed}/{len(pending)}", flush=True)
    if failures:
        raise RuntimeError("bounded overload workers failed:\n" + "\n".join(failures))
    document = assemble_bounded_core_aten_suite(
        opset["ops"],
        pytorch_version=opset["torch"],
        profile=profile,
        pools=pools,
        generator_sha256=generator_sha256,
    )
    (destination / "bounded_core_aten_suite.json").write_bytes(json_bytes(document))
    (destination / "bounded_core_aten_coverage.md").write_text(bounded_summary(document), encoding="utf-8")
    print(
        f"{document['selected_count']}/{document['candidate_count']} cases cover "
        f"{document['witnessed_obligation_count']} bounded obligations for "
        f"{document['overload_count']} Core ATen overloads"
    )
    print(f"status: {'complete' if document['complete'] else 'incomplete'}")
    print(f"artifacts: {destination}")
    return 0 if document["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
