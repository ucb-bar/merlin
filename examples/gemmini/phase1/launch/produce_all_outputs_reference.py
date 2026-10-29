"""Produce private all-declared-output oracles under an explicit launch selection.

Run this operator outside the candidate workspace. It uses existing captures and
inputs, never modifies the selected bundle, and requires every judged output to
bitmatch its sealed reference before publishing the supplementary manifest. The
manifest is compatible with ``merlin.model_check.full_output_audit``. The caller
must admit and protect its SHA before using it as an offline comparison oracle.

This is the reusable form of the operator that produced the v1 supplements;
historical receipts retain their original producer digest, not this source's.
"""

import argparse
import ctypes
import datetime
import hashlib
import importlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def pin(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def save(path, value):
    temporary = Path(path).with_suffix(".next")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def selected(args):
    from merlin_experiments.phase1 import launch_config as L

    config = Path(args.config).resolve()
    launch = L.build(L.load(config), task_root=args.task_root, prep=args.prep, run_id=args.run_id)
    spec = importlib.util.spec_from_file_location("all_outputs_launcher", Path(args.launcher).resolve())
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    return launch, operator


def worker(args):
    import numpy as np

    from merlin.common.cpu_pool import AUXILIARY, cpu_priority, shared_compute_pool
    from merlin.model_check import bundle as B
    from merlin.model_check import chain as CH
    from merlin.model_check import full as F
    from merlin.model_check import models as M
    from merlin.model_check import references as REF
    from merlin.runtime import linalg_numpy as LN

    launch, operator = selected(args)
    if args.worker_model not in args.model:
        raise ValueError("worker model is outside the explicitly selected roster")
    out = Path(args.out) / args.worker_model
    out.mkdir()
    started = time.monotonic()
    with operator.configured_environment(launch), cpu_priority(AUXILIARY), shared_compute_pool(timeout_s=args.timeout):
        bundle = B.Bundle(Path(os.environ["MERLIN_MODEL_CHECK_BUNDLE"]))
        models = M.load(bundle.root / "configuration/models.yaml")
        model = models.model(args.worker_model)
        if any(not path_component(program.name) or program.steps < 1 for program in model.programs):
            raise ValueError("programs need safe names and a positive complete step roster")
        input_id = args.input
        if Path(CH.__file__).resolve() != Path(args.repo).resolve() / "src/merlin/model_check/chain.py":
            raise ValueError("imported reference engine is outside the selected source checkout")
        input_path = F.derived_input(bundle, model.name, input_id)
        with np.load(input_path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        catalogs = {program.name: bundle.catalog(model.name, program.name) for program in model.programs}
        sources = [
            Path(__file__),
            Path(args.config),
            Path(args.launcher),
            models.path,
            bundle.root / "MANIFEST.json",
            bundle.root / "SHA256SUMS",
            input_path,
            bundle.root / model.name / input_id / "int8.npz",
        ]
        for name in (
            "merlin.model_check.chain",
            "merlin.model_check.catalog",
            "merlin.model_check.references",
            "merlin.model_check.full",
            "merlin.model_check.models",
            "merlin.runtime.linalg_numpy",
            "merlin.runtime.dispatch_runtime",
            "merlin.runtime.host_math",
            "merlin.runtime.host_math_reference",
        ):
            sources.append(Path(importlib.import_module(name).__file__))
        for program in model.programs:
            catalog = catalogs[program.name]
            sources.extend(
                [
                    bundle.program_dir(model.name, program.name) / "catalog.json",
                    catalog.ref_prepared,
                    catalog.stage / "model.mlir",
                    catalog.stage / "inputs.npz",
                ]
            )
            from merlin.runtime.dispatch_runtime import ForwardArgsCache

            for name in (
                *ForwardArgsCache.FILES,
                "weights.npz",
                "weights.bin",
                "weights.json",
                "session_contract.yaml",
            ):
                if (catalog.stage / name).is_file():
                    sources.append(catalog.stage / name)
        before = [pin(path) for path in dict.fromkeys(sources)]
        with REF.reference_math_context(model, bundle, out / "scratch"):
            numeric = LN.numeric_contract()
            cache = CH.ChainCache()

            def progress(program, step, catalog, arguments):
                save(
                    out / "progress.json",
                    {
                        "model": model.name,
                        "program": program,
                        "step": step,
                        "status": "reference_step_started",
                        "elapsed_s": time.monotonic() - started,
                    },
                )
                print(f"[reference] {model.name} {program}#{step}", flush=True)

            run = CH.run_reference(model, catalogs, arrays, out / "scratch", cache=cache, record=progress)
            if LN.numeric_contract() != numeric:
                raise ValueError("selected numerical identity changed during reference chain")
            judged = CH.judged_outputs(model, run)
            frozen = F._ordered(bundle.root / model.name / input_id / "int8.npz")
            if len(judged) != len(frozen) or any(
                a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes()
                for a, b in zip(judged, frozen, strict=True)
            ):
                raise ValueError("supplemental reference does not bitmatch the selected sealed judged outputs")
            programs = []
            for program in model.programs:
                catalog = catalogs[program.name]
                elements = CH.result_elements(cache.module(catalog.ref_prepared))
                steps, declaration = [], None
                if len(run.steps[program.name]) != program.steps:
                    raise ValueError("reference step roster differs")
                for step, values in enumerate(run.steps[program.name]):
                    if len(values) != len(elements):
                        raise ValueError("reference declared output arity differs")
                    stored = [LN._store(value, element) for value, element in zip(values, elements, strict=True)]
                    descriptions = [
                        {"output": index, "dtype": value.dtype.str, "shape": list(value.shape), "element_type": element}
                        for index, (value, element) in enumerate(zip(stored, elements, strict=True))
                    ]
                    if any(not np.isfinite(value).all() for value in stored):
                        raise ValueError("supplemental reference has nonfinite output")
                    if declaration is None:
                        declaration = descriptions
                    elif descriptions != declaration:
                        raise ValueError("recurrent output shape/dtype differs from static source")
                    path = out / f"{program.name}.{step}.npz"
                    np.savez(path, **{f"out{index}": value for index, value in enumerate(stored)})
                    steps.append(
                        {
                            "step": step,
                            "outputs_file": pin(path),
                            "outputs": [
                                {
                                    **description,
                                    "sha256_of_array_bytes": hashlib.sha256(
                                        np.ascontiguousarray(value).tobytes()
                                    ).hexdigest(),
                                }
                                for description, value in zip(descriptions, stored, strict=True)
                            ],
                        }
                    )
                programs.append(
                    {
                        "program": program.name,
                        "steps_declared": program.steps,
                        "declared_outputs": declaration,
                        "steps": steps,
                        "stage": str(catalog.stage),
                        "catalog": pin(bundle.program_dir(model.name, program.name) / "catalog.json"),
                        "reference_program": pin(catalog.ref_prepared),
                        "capture_program": pin(catalog.stage / "model.mlir"),
                    }
                )
        after = [pin(item["path"]) for item in before]
        if after != before:
            raise ValueError("reference source/input bytes changed")
        result = {
            "schema": "merlin.operator.full_declared_output_reference.v1",
            "model": model.name,
            "input_id": input_id,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "selected_identity": {
                "bundle_manifest_sha256": pin(bundle.root / "MANIFEST.json")["sha256"],
                "bundle_sums_sha256": pin(bundle.root / "SHA256SUMS")["sha256"],
                "models_config_sha256": pin(models.path)["sha256"],
            },
            "input": pin(input_path),
            "implementation": {"source_pins": before, "numeric_contract": numeric},
            "programs": programs,
            "representation": "declared_storage",
            "judged_crossjoin": {
                "sealed_npz": pin(bundle.root / model.name / input_id / "int8.npz"),
                "bit_exact": True,
                "outputs": len(judged),
            },
            "seconds": time.monotonic() - started,
            "status": "complete",
            "scope": "supplemental original quantized reference only; no compiled execution or qualification",
        }
        save(out / "reference.json", result)
        sums = [(pin(path)["sha256"], path.name) for path in sorted(out.glob("*.npz"))]
        sums.append((pin(out / "reference.json")["sha256"], "reference.json"))
        (out / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for digest, name in sums))
        for path in [out / "reference.json", out / "SHA256SUMS", *out.glob("*.npz")]:
            path.chmod(0o444)
        save(
            out / "status.json",
            {
                "status": "complete",
                "reference": pin(out / "reference.json"),
                "sums": pin(out / "SHA256SUMS"),
                "seconds": time.monotonic() - started,
            },
        )
        print(
            json.dumps({"model": model.name, "status": "complete", "seconds": time.monotonic() - started}), flush=True
        )


def supervise(args):
    from merlin.common.owned_children import become_child_subreaper, reap_owned_children

    become_child_subreaper()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: (_ for _ in ()).throw(InterruptedError("operator stop")))
    root = Path(args.out)
    root.mkdir(exist_ok=False)
    started, children = time.monotonic(), {}
    try:
        for model in args.model:
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                *args.operator_argv,
                "--worker",
                "--worker-model",
                model,
            ]
            env = {
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            with (root / f"{model}.log").open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            children[model] = child
        save(
            root / "jobs.json",
            {
                "script": pin(__file__),
                "deadline_utc": args.deadline,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "controller": process_identity(os.getpid()),
                "children": {
                    name: {**process_identity(child.pid), "argv": child.args} for name, child in children.items()
                },
            },
        )
        deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
        while any(child.poll() is None for child in children.values()):
            if time.time() >= deadline or time.monotonic() - started >= args.timeout:
                raise TimeoutError("supplemental derivation deadline")
            available = next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            if available < args.min_headroom_gib * 1024**3:
                raise MemoryError("host headroom below selected guard")
            pids, pss = [child.pid for child in children.values() if child.poll() is None], 0
            for pid in pids:
                try:
                    pids.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
                    pss += next(
                        int(line.split()[1]) * 1024
                        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
                        if line.startswith("Pss:")
                    )
                except (OSError, StopIteration):
                    pass
            if pss > args.max_pss_gib * 1024**3:
                raise MemoryError("owned PSS exceeds selected guard")
            save(
                root / "status.json",
                {
                    "status": "running",
                    "elapsed_s": time.monotonic() - started,
                    "models": {name: child.poll() for name, child in children.items()},
                    "pss_bytes": pss,
                    "host_available_bytes": available,
                },
            )
            time.sleep(2)
        if any(child.returncode for child in children.values()):
            raise RuntimeError("supplemental reference worker failed")
        save(
            root / "status.json",
            {
                "status": "complete",
                "elapsed_s": time.monotonic() - started,
                "models": {name: child.returncode for name, child in children.items()},
            },
        )
    except BaseException as error:
        save(
            root / "status.json",
            {
                "status": "failed",
                "reason": str(error),
                "models": {name: child.poll() for name, child in children.items()},
            },
        )
        raise
    finally:
        cleanup = reap_owned_children(None, 5)
        save(root / "cleanup.json", cleanup)
        if not cleanup["complete"]:
            raise RuntimeError("supplemental reference children were not completely reaped")


def process_identity(pid):
    """Record a Linux birth identity while it is observable; never invent one."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(fields[19])
    except (OSError, IndexError, ValueError):
        ticks = None
    return {"pid": pid, "start_ticks": ticks, "owner_uid": os.getuid()}


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    for field in ("repo", "config", "launcher", "task-root", "prep", "run-id", "out", "deadline", "input"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument(
        "--model", action="append", required=True, help="explicit model roster; repeat for parallel workers"
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-pss-gib", type=float, default=192)
    parser.add_argument("--min-headroom-gib", type=float, default=24)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.operator_argv = list(sys.argv[1:] if argv is None else argv)
    if len(set(args.model)) != len(args.model) or any(not path_component(name) for name in args.model):
        parser.error("model roster needs distinct single path components")
    if args.input.count("/") != 1 or not all(path_component(part) for part in args.input.split("/")):
        parser.error("input must be an explicit split/case identifier")
    if args.worker != bool(args.worker_model) or args.worker_model and args.worker_model not in args.model:
        parser.error("worker must belong to the selected model roster")
    deadline = datetime.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline.timestamp() <= time.time():
        parser.error("deadline must include a timezone and be in the future")
    bounds = (args.timeout, args.max_pss_gib, args.min_headroom_gib)
    if not all(math.isfinite(value) for value in bounds) or min(bounds[:2]) <= 0 or args.min_headroom_gib < 0:
        parser.error("resource bounds must be positive (headroom may be zero)")
    return args


def path_component(value):
    return bool(value) and value not in {".", ".."} and Path(value).name == value


def main(argv=None):
    args = arguments(argv)
    if args.worker:
        from merlin.common.owned_children import become_child_subreaper, reap_owned_children

        become_child_subreaper()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, lambda number, frame: (_ for _ in ()).throw(InterruptedError("operator stop")))
        try:
            worker(args)
        finally:
            reap_owned_children(None, 5)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
