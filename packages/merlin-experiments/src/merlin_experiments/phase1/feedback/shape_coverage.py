"""Lowering-coverage sweeps for one immutable grading candidate: reuse, concurrency, child processes.

``loop_grading`` owns the shape GATE (whether a grade may pass); this module owns how the emit-only
sweep behind it is measured. A sweep is reused only for byte-identical inputs under the original
compiler profile, and corners run concurrently either on threads or, under that profile, in fresh
grader processes. Every fault here is a miss or an ``error`` outcome, never coverage.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from merlin.targetgen.target_experiment import load_target_experiment

SHAPE_CACHE_SCHEMA = "merlin.phase1.shape_coverage_cache.v1"
#: Deterministic emit outcomes of the submitted bytes. An ``error`` may be a harness or resource fault,
#: so a sweep containing one is never cached: the next grade of the same bytes probes again.
_CACHEABLE_OUTCOMES = frozenset({"lowered", "declined", "empty"})
#: Grader code that stages a probe, routes its execution or classifies what came back.
_SHAPE_HARNESS_MODULES = (
    "merlin.targetgen.lowering_coverage",
    "merlin.targetgen.capability_probes",
    "merlin.targetgen.capsule_common",
    "merlin.targetgen.package_runtime",
    "merlin.targetgen.contract.schemas",
    "merlin.targetgen.artifact_scale",
    "merlin.verify.lean_emission_context",
    "merlin_experiments.phase1.feedback.compiler_profile_runtime",
    "merlin_experiments.phase1.brokers.compiler_call_channel",
    "merlin_experiments.phase2.campaign",
    "merlin_experiments.phase1.feedback.loop_grading",
    __name__,
)


def selected_workers() -> int:
    """``MERLIN_SHAPE_COVERAGE_WORKERS`` (1-16); unselected keeps the original serial sweep."""
    if "MERLIN_SHAPE_COVERAGE_WORKERS" not in os.environ:
        return 1
    raw = os.environ["MERLIN_SHAPE_COVERAGE_WORKERS"].strip()
    if not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= 16:
        raise ValueError("MERLIN_SHAPE_COVERAGE_WORKERS must be an integer between 1 and 16")
    return int(raw)


def cache_selected() -> bool:
    """``MERLIN_SHAPE_COVERAGE_CACHE=1`` selects reuse; unselected probes every grade."""
    if "MERLIN_SHAPE_COVERAGE_CACHE" not in os.environ:
        return False
    raw = os.environ["MERLIN_SHAPE_COVERAGE_CACHE"]
    if raw not in {"0", "1"}:
        raise ValueError("MERLIN_SHAPE_COVERAGE_CACHE must be 0 or 1")
    return raw == "1"


def _file_sha256(path) -> str | None:
    try:
        with open(path, "rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except (OSError, TypeError):
        return None


def _tree_identity(root: Path) -> str | None:
    """Exact identity of everything a probe could read: kind, relative path, mode, bytes or link text.

    Unlike the authoring ``hash_tree`` this skips nothing (build products, caches and permission bits
    can all change what an entrypoint does), so two trees share an identity only when byte-identical.
    """
    root = Path(root)
    if root.is_file():
        return _file_sha256(root)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            row = ["link", relative, os.readlink(path)]
        elif stat.S_ISREG(info.st_mode):
            row = ["file", relative, info.st_mode & 0o7777, _file_sha256(path)]
        elif stat.S_ISDIR(info.st_mode):
            row = ["dir", relative, info.st_mode & 0o7777]
        else:
            row = ["other", relative, info.st_mode]
        digest.update(json.dumps(row).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _shape_cache_material(cand, *, plan_id, context, contract, timeout, additional_forbidden, service) -> dict:
    """Every input a sweep reads besides the probe plan: bytes, contract, facts, tools and grader code."""
    import importlib.util

    harness = {}
    for name in _SHAPE_HARNESS_MODULES:
        spec = importlib.util.find_spec(name)
        harness[name] = _file_sha256(spec.origin) if spec is not None and spec.origin else None
    material = {
        "schema": SHAPE_CACHE_SCHEMA,
        "candidate": _tree_identity(Path(cand)),
        "plan": plan_id,
        "target": context.target,
        "contract": _tree_identity(Path(contract)),
        "timeout_s": timeout,
        "additional_forbidden": sorted(additional_forbidden),
        "descriptor_sha256": _file_sha256(context.descriptor),
        "selected_inputs": {
            name: _file_sha256(os.environ[name]) if os.environ.get(name) else None
            for name in ("MERLIN_TARGET_EXPERIMENT", "MERLIN_TARGET_CONTRACT", "MERLIN_RTL_FACTS")
        },
        # The original profile pins the frozen tool closure every entrypoint runs in.
        "executor": _executor_identity(service),
        "python": [sys.executable, sys.version],
        "harness": harness,
    }
    if material["candidate"] is None or material["contract"] is None:
        raise ValueError("shape-coverage cache needs an existing candidate and contract")
    return json.loads(json.dumps(material, sort_keys=True))


def _cacheable_sweep(cov, plan) -> bool:
    corners = cov.get("corners") if isinstance(cov, dict) else None
    return (
        isinstance(corners, list)
        and [c.get("corner") if isinstance(c, dict) else None for c in corners] == [name for name, _ in plan["probes"]]
        and all(c.get("outcome") in _CACHEABLE_OUTCOMES for c in corners)
        and type(cov.get("all_covered")) is bool
        and type(cov.get("baseline_tile_lowered")) is bool
    )


def _shape_cache_lookup(path: Path, key: str, material: dict, plan: dict) -> dict | None:
    """A stored sweep for exactly this key material, or None. Anything malformed is a miss."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(document, dict)
        or document.get("schema") != SHAPE_CACHE_SCHEMA
        or document.get("key") != key
        or document.get("material") != material
        or hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest() != key
        or not _cacheable_sweep(document.get("result"), plan)
    ):
        return None
    return document


def _shape_cache_store(path: Path, key: str, material: dict, plan: dict, cov: dict, cand) -> bool:
    if not _cacheable_sweep(cov, plan) or _tree_identity(Path(cand)) != material["candidate"]:
        return False  # an emit error, or bytes that moved while probing, never become reusable
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    document = {
        "schema": SHAPE_CACHE_SCHEMA,
        "key": key,
        "material": material,
        "result": cov,
        "stored_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{id(document)}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return True


SHAPE_PROBE_CHILD_SCHEMA = "merlin.phase1.shape_probe_child.v1"


def _executor_identity(service) -> dict:
    return {
        "bundle_sha256": hashlib.sha256(service.bundle_bytes).hexdigest(),
        "snapshot_sha256": service.snapshot_digest,
        "profile_sha256": hashlib.sha256(
            json.dumps(service.profile, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    }


def _process_probe_runner(cand, run_dir, *, context, contract, timeout, additional_forbidden, service, workers):
    """Probe corner groups in fresh grader processes, each under its own original-profile scope.

    Measured on a retained candidate, one emit probe under the original profile spent ~97% of its wall
    time in grader-process Python (per-entrypoint sandbox policy and snapshot checks), so corners on
    threads share one interpreter lock and barely overlap. Each child re-opens the same archived
    owner (refusing a different bundle, snapshot or profile), holds one shared compute slot for its
    probes, and returns only per-corner outcomes. The parent holds no slot while it waits. A child that
    fails, times out or returns a malformed result yields ``error`` outcomes, which are never covered.
    """
    import subprocess
    import tempfile
    import time

    from merlin.common import cpu_pool
    from merlin.targetgen.capsule_runner import _kill_tree

    def run(plan):
        probes = [[name, list(shape)] for name, shape in plan["probes"]]
        groups = [probes[i::workers] for i in range(min(workers, len(probes)))]
        common = {
            "schema": SHAPE_PROBE_CHILD_SCHEMA,
            "candidate": str(Path(cand).resolve()),
            "target": context.target,
            "descriptor": str(context.descriptor),
            "contract": str(contract),
            "timeout": timeout,
            "additional_forbidden": list(additional_forbidden),
            "operand_mlir": plan["operand_mlir"],
            "accum_mlir": plan["accum_mlir"],
            "workspace": str(service.workspace),
            "bundle_manifest": str(service.bundle_manifest),
            "repo": str(service.repo),
            "executor": _executor_identity(service),
            "work_root": str(Path(run_dir).resolve() / "_qa_work" / "profile_package_execution"),
        }
        results, walls = {}, {}
        root = Path(run_dir).resolve() / "_qa_work" / "shape_probe_children"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="shape-probes-", dir=root) as private:
            launched = []
            try:
                for index, group in enumerate(groups):
                    spec, out = Path(private) / f"spec-{index}.json", Path(private) / f"out-{index}.json"
                    spec.write_text(json.dumps({**common, "probes": group}), encoding="utf-8")
                    log = (Path(private) / f"child-{index}.log").open("wb")
                    command = _child_command(spec, out)
                    environment = dict(os.environ)
                    process = subprocess.Popen(
                        cpu_pool.worker_pool_command(command, environment),
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    # Setup plus build and three entrypoints per corner, each bounded by the probe timeout.
                    deadline = time.monotonic() + len(group) * (4 * timeout + 60) + 300
                    launched.append((group, out, log, process, deadline))
                for group, out, log, process, deadline in launched:
                    detail = None
                    try:
                        process.wait(timeout=max(1, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        _kill_tree(process.pid)
                        process.wait(timeout=10)
                        detail = "shape probe child exceeded its deadline"
                    log.close()
                    document = None
                    if detail is None and process.returncode == 0 and out.is_file():
                        try:
                            document = json.loads(out.read_text(encoding="utf-8"))
                        except ValueError:
                            document = None
                    rows = document.get("outcomes") if isinstance(document, dict) else None
                    if (
                        not isinstance(document, dict)
                        or document.get("schema") != SHAPE_PROBE_CHILD_SCHEMA
                        or document.get("probes") != group
                        or not isinstance(rows, list)
                        or len(rows) != len(group)
                    ):
                        tail = (
                            Path(log.name).read_bytes()[-300:].decode("utf-8", "replace")
                            if Path(log.name).is_file()
                            else ""
                        )
                        rows = [
                            [
                                "error",
                                detail
                                or f"shape probe child exited {process.returncode} without a complete result: {tail}",
                                0,
                            ]
                        ] * len(group)
                    for (name, _), row in zip(group, rows, strict=True):
                        results[name] = row
                    observed = document.get("corner_wall_s") if isinstance(document, dict) else None
                    walls.update(observed if isinstance(observed, dict) else {})
            finally:
                for *_, log, process, _deadline in launched:
                    if process.poll() is None:
                        _kill_tree(process.pid)
                        process.wait(timeout=10)
                    log.close()
        unprobed = ["error", "corner was not probed", 0]  # recorded as a failure, never coverage
        return [results[name] if name in results else unprobed for name, _ in probes], walls

    return run


def _child_command(spec: Path, out: Path) -> list[str]:
    from merlin.targetgen.capsule_runner import _CHILD_GUARD

    return [
        sys.executable,
        "-c",
        _CHILD_GUARD
        + (
            "from merlin_experiments.phase1.feedback.shape_coverage import probe_child;"
            "sys.exit(probe_child(sys.argv[1:]))"
        ),
        str(spec),
        str(out),
    ]


def probe_child(argv) -> int:
    """Child entry: probe a corner group under a freshly re-opened original profile scope."""
    from merlin.common.cpu_pool import shared_compute_pool
    from merlin.targetgen import lowering_coverage as LC

    from .compiler_profile_runtime import CommonTransformService, original_compiler_execution

    spec_path, out_path = map(Path, argv)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != SHAPE_PROBE_CHILD_SCHEMA:
        raise ValueError("shape probe child received an unknown specification")
    service = CommonTransformService(
        workspace=Path(spec["workspace"]), bundle_manifest=Path(spec["bundle_manifest"]), repo=Path(spec["repo"])
    )
    if _executor_identity(service) != spec["executor"]:
        raise ValueError("shape probe child re-opened a different original profile owner")
    probes = [(name, tuple(shape)) for name, shape in spec["probes"]]
    with original_compiler_execution(
        service,
        target_experiment=load_target_experiment(Path(spec["descriptor"])),
        candidate=Path(spec["candidate"]),
        work_root=Path(spec["work_root"]),
    ):
        with shared_compute_pool():
            outcomes, walls = LC.probe_corners(
                spec["candidate"],
                probes,
                target=spec["target"],
                operand_mlir=spec["operand_mlir"],
                accum_mlir=spec["accum_mlir"],
                contract=spec["contract"],
                timeout=spec["timeout"],
                additional_forbidden=tuple(spec["additional_forbidden"]),
            )
    out_path.write_text(
        json.dumps(
            {
                "schema": SHAPE_PROBE_CHILD_SCHEMA,
                "probes": spec["probes"],
                "outcomes": [list(row) for row in outcomes],
                "corner_wall_s": walls,
            }
        ),
        encoding="utf-8",
    )
    return 0


def measure(
    cand,
    run_dir,
    *,
    context,
    contract,
    timeout,
    additional_forbidden,
    service,
    before_probe,
    workers: int,
    use_cache: bool,
) -> dict:
    """Run, or reuse for byte-identical inputs, the lowering-coverage sweep of one candidate.

    Reuse requires the original compiler profile (it pins the frozen tool closure), an exact key over
    every input in :func:`_shape_cache_material` plus the probe plan, and a stored sweep with no emit
    error. Any cache fault is a miss: the probes run, and a sweep that did not run is still recorded
    as not covered by the caller.
    """
    from merlin.targetgen import lowering_coverage as LC

    cache = None
    if use_cache and service is not None:
        try:
            plan = LC.sweep_plan(context.target)
            material = _shape_cache_material(
                cand,
                plan_id=LC.plan_identity(plan),
                context=context,
                contract=contract,
                timeout=timeout,
                additional_forbidden=additional_forbidden,
                service=service,
            )
            key = hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()
            path = Path(run_dir) / "_qa_work" / "shape_coverage_cache" / f"{key}.json"
            cache = (plan, material, key, path)
            stored = _shape_cache_lookup(path, key, material, plan)
            if stored is not None:
                result = dict(stored["result"])
                result["cache"] = {
                    "hit": True,
                    "key": key,
                    "stored_at": stored.get("stored_at"),
                    "probed_package": result.get("package"),
                }
                result["package"] = str(cand)
                result["timing"] = {"reused": True, "probed_timing": result.get("timing")}
                return result
        except Exception as exc:  # noqa: BLE001 -- a cache fault only costs a fresh sweep
            print(f"  [shape-coverage] cache unavailable: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            cache = None
    if before_probe is not None:
        try:
            before_probe()
        except Exception as exc:  # noqa: BLE001 -- early feedback is optional; the gate is not
            print(f"  [shape-coverage] pending verdict not published: {type(exc).__name__}: {exc}", flush=True)
    concurrency = {}
    if workers > 1:
        concurrency["workers"] = workers
        if service is not None:
            concurrency["runner"] = _process_probe_runner(
                cand,
                run_dir,
                context=context,
                contract=contract,
                timeout=timeout,
                additional_forbidden=additional_forbidden,
                service=service,
                workers=workers,
            )
    cov = LC.sweep(
        cand,
        target=context.target,
        contract=contract,
        timeout=timeout,
        additional_forbidden=additional_forbidden,
        **concurrency,
        **({"plan": cache[0]} if cache is not None else {}),
    )
    if cache is not None:
        plan, material, key, path = cache
        try:
            stored = _shape_cache_store(path, key, material, plan, cov, cand)
        except Exception as exc:  # noqa: BLE001 -- an unwritable cache never changes this sweep
            print(f"  [shape-coverage] cache not stored: {type(exc).__name__}: {exc}", flush=True)
            stored = False
        cov = {**cov, "cache": {"hit": False, "key": key, "stored": stored}}
    return cov
