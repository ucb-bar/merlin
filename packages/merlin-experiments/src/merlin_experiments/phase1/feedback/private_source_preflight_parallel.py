"""Bounded source-only workers with strict complete-roster aggregation.

Each ordinary CLI worker checks the entire private declaration/freeze, then one
complete model. No compiler capability, candidate or numerical authority is
transported. A successful partition is explicitly not a full-roster pass.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from merlin.common import invocation_record as IR
from merlin.common import owned_process
from merlin.common.digest import sha256_file

from . import private_full_models as models
from . import private_source_freeze as freeze
from . import private_source_preflight as single

REPORT, PROGRESS = single.REPORT, single.PROGRESS


def _command(name, spec, descriptor, directory, target, native, cpu):
    argv = [
        sys.executable,
        "-u",
        "-m",
        single.__name__,
        "--spec",
        str(spec),
        "--descriptor",
        str(descriptor),
        "--out",
        str(directory),
        "--target",
        target,
        "--model",
        name,
    ]
    if native:
        argv += ["--native-result-format-selection", str(native)]
    return [shutil.which("taskset"), "-c", str(cpu), *argv] if cpu is not None else argv


def _child_binding(spec, parent, child, target):
    record = json.loads(json.dumps(parent))
    original = Path(record["root"])
    record["root"] = str(child)
    for row in (*record["archives"], *record["sources"]):
        path = Path(row["frozen_path"])
        if path.parent != original:
            raise ValueError("source partition frozen member escaped its original root")
        row["frozen_path"] = str(child / path.name)
    return freeze.binding(spec, record, root=child, target=target)


def _worker(name, spec, descriptor, out, target, native, deadline, cpu):
    directory = out / "partitions" / name
    argv = _command(name, spec, descriptor, directory, target, native, cpu)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        result = IR.run(
            argv,
            directory=out / "workers" / name,
            stage="private_source_preflight_partition",
            inputs=(spec, descriptor, *((Path(native),) if native else ())),
            outputs=(directory / single.REPORT,),
            dependencies=(Path(__file__), Path(single.__file__), Path(models.__file__), Path(freeze.__file__)),
            owned_process_tree=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=max(0.001, deadline - time.monotonic()),
        )
        (invocation,) = (out / "workers" / name / "invocations").glob("*/invocation.json")
        return {
            "model": name,
            "returncode": result.returncode,
            "invocation": str(invocation),
            "cleanup": result.owned_process_cleanup,
            "report": str(directory / single.REPORT),
        }
    except owned_process.CleanupUnknown:
        raise  # The surrounding owner must remain responsible; never claim a clean source check.
    except Exception as error:
        return {"model": name, "error": type(error).__name__, "cleanup": getattr(error, "owned_process_cleanup", None)}


def preflight(
    spec,
    *,
    descriptor,
    out,
    target,
    native_result_format_selection=None,
    cpus=(),
    workers=3,
    timeout_s=1200,
    progress_path=None,
):
    """Return one complete source-only roster; each child has an owned process tree."""
    spec, descriptor, out = map(lambda path: Path(path).absolute(), (spec, descriptor, out))
    if out.exists() or out.is_symlink():
        raise ValueError("parallel source preflight needs a fresh private output directory")
    if (
        type(workers) is not int
        or workers < 1
        or type(timeout_s) not in (int, float)
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("parallel source preflight needs finite positive resource bounds")
    deadline = time.monotonic() + timeout_s
    cpus = tuple(cpus)
    if cpus and (
        len(set(cpus)) != len(cpus)
        or any(type(cpu) is not int or cpu not in os.sched_getaffinity(0) for cpu in cpus)
        or shutil.which("taskset") is None
    ):
        raise ValueError("source preflight CPUs must be distinct and available")
    roster, programs = models.requirements_for(descriptor), models.program_requirements_for(descriptor)
    if (
        not roster
        or set(programs) != set(roster)
        or any(
            not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") for name in roster
        )
    ):
        raise ValueError("parallel source preflight needs a complete model/program roster")
    source_pins = {
        str(path): sha256_file(path)
        for path in (
            spec,
            descriptor,
            Path(__file__),
            Path(single.__file__),
            Path(models.__file__),
            Path(freeze.__file__),
        )
    }
    out.mkdir(mode=0o700, parents=True)
    observe = single._observer(None, Path(progress_path) if progress_path else None)
    observe({"stage": "parallel_source_preflight", "total": len(roster)})
    parent = freeze.stage(spec, out / "sources", target=target)
    capacity = min(workers, len(roster), len(cpus) if cpus else workers)
    with ThreadPoolExecutor(max_workers=capacity) as pool:
        futures = [
            pool.submit(
                _worker,
                name,
                spec,
                descriptor,
                out,
                target,
                native_result_format_selection,
                deadline,
                cpus[index % len(cpus)] if cpus else None,
            )
            for index, name in enumerate(roster)
        ]
        observed = [future.result() for future in futures]
    rows, reports, passed = [], {}, True
    for index, (name, worker) in enumerate(zip(roster, observed, strict=True)):
        if worker.get("returncode") != 0:
            passed = False
            rows.append({"model": name, "status": "fail", "reason": worker.get("error", "source worker refused")})
            continue
        call = IR.verify(Path(worker["invocation"]))
        path = Path(worker["report"])
        report = json.loads(path.read_bytes())
        if (
            call.get("stage") != "private_source_preflight_partition"
            or call.get("argv")
            != _command(
                name,
                spec,
                descriptor,
                out / "partitions" / name,
                target,
                native_result_format_selection,
                cpus[index % len(cpus)] if cpus else None,
            )
            or not any(
                pin.get("path") == str(path) and pin.get("sha256") == sha256_file(path) for pin in call["outputs"]
            )
            or report.get("mode") != "source_preflight_partition"
            or report.get("schema") != models.RESULT_SCHEMA
            or report.get("scope") != models.SOURCE_PREFLIGHT_SCOPE
            or report.get("partition_passed") is not False
            or report.get("selected_model") != name
            or report.get("passed") is not False
            or report.get("source_preflight_passed") is not False
            or report.get("candidate_tree_sha256") is not None
            or report.get("source_partition_passed") is not True
            or report.get("target") != target
            or report.get("private_spec_sha256") != source_pins[str(spec)]
            or report.get("required_models") != list(roster)
            or report.get("required_programs") != {model: list(programs[model]) for model in roster}
            or len(report.get("models", [])) != 1
            or report["models"][0].get("model") != name
            or report["models"][0].get("status") != "source_inputs_verified"
            or set(report["models"][0].get("checks", {}).get("source", {})) != set(programs[name])
            or report.get("authored_source_freeze") != _child_binding(spec, parent, path.parent / "sources", target)
        ):
            raise ValueError("source partition omitted or changed selected roster/source ownership")
        rows.extend(report["models"])
        reports[name] = {"path": str(path), "sha256": sha256_file(path), "invocation": worker["invocation"]}
    if any(sha256_file(path) != digest for path, digest in source_pins.items()):
        raise ValueError("parallel source preflight inputs changed")
    report = {
        "schema": models.RESULT_SCHEMA,
        "target": target,
        "mode": "source_preflight",
        "source_preflight_passed": passed,
        "passed": False,
        "candidate_tree_sha256": None,
        "private_spec_sha256": source_pins[str(spec)],
        "required_models": list(roster),
        "required_programs": {name: list(programs[name]) for name in roster},
        "models": rows,
        "scope": models.SOURCE_PREFLIGHT_SCOPE,
        "full_model_numerical_equivalence": "not_run",
        "paper_accuracy": "not_claimed",
        "authored_source_freeze": freeze.binding(spec, parent, root=out / "sources", target=target),
        "source_partitions": reports,
        "workers": observed,
        "source_workers": capacity,
        "timeout_s": timeout_s,
        "source_inputs": source_pins,
    }
    path = out / single.REPORT
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.chmod(0o600)
    observe({"stage": "complete", "status": "source_inputs_verified" if passed else "fail"})
    return report
