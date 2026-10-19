"""Per-input references of a bundle: FP outputs, the INT8 reference chain, its metrics, and device calls.

For every prepared evaluation input (``<private>/derived/<model>/<split>/<id>/``) this writes into the
bundle, under ``<model>/<split>/<id>/``:

- ``fp.npz``            the FP reference's judged outputs (from :mod:`reference_worker`, pinned env);
- ``int8.npz``          the INT8 reference chain's judged outputs (:mod:`chain`, ``linalg_numpy``);
- ``metrics_int8.json`` the INT8 reference's metrics against FP (reference admission, contract §4.5);
- ``fp_baseline.npz`` / ``metrics_baseline.json`` when the models file names a ``reference.baseline``: the
  float model at that precision and its own error against the FP reference;
- ``<program>/calls.json`` the device calls of each program's first step (the replayed boundaries).

When the models file names ``reference.seeds`` (an input array and a count), every input is also judged
under ``count - 1`` further draws of that standard-normal array, each in ``<split>/<id>/seeds/<k>/`` with its
own ``inputs.npz`` and the same reference files; draw 0 is the stored input. Draw ``k`` is seeded by the
sha256 of ``<split>/<id>/<array>/<k>``, so a rebuild reproduces it. Device calls are recorded for draw 0
only; admission is judged over every (input, draw). :func:`rejudge` re-evaluates a built bundle's admission
from the stored metrics alone, when a bar changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from merlin.model_check import catalog as CAT
from merlin.model_check import chain as CH
from merlin.model_check import inputs as INP
from merlin.model_check import metrics as M
from merlin.model_check.bundle import Bundle
from merlin.model_check.models import Model, selected_host_math_policy
from merlin.runtime import linalg_numpy as LN

REFERENCE_WORKER = Path(__file__).with_name("reference_worker.py")
FP_TASK = {"logits": "fp_resnet", "next_token_distribution": "fp_tiny_llama", "per_channel": "fp_smolvla"}
#: The FP reference runs with the numerics a sealed capture has: ATen's portable kernels (the capture
#: sandbox exposes no CPU capabilities, so ATen falls back to them) on one thread. A bf16 model's FP
#: output otherwise moves by ~1% between two valid hosts, and the reference would depend on the machine.
FP_NUMERICS_ENV = {"ATEN_CPU_CAPABILITY": "default", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def input_ids(private: Path, model: str, splits: tuple[str, ...]) -> list[tuple[str, str]]:
    out = []
    for split in splits:
        root = Path(private) / "derived" / model / split
        if root.is_dir():
            out += [(split, p.name) for p in sorted(root.iterdir()) if p.is_dir()]
    return out


def _derived(private: Path, model: str, split: str, ident: str) -> Path:
    directory = Path(private) / "derived" / model / split / ident
    for name in ("inputs.npz", "observation.npz"):
        if (directory / name).is_file():
            return directory / name
    raise INP.InputError(f"no derived inputs at {directory}")


def _draw_seed(split: str, ident: str, key: str, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{split}/{ident}/{key}/{k}".encode()).digest()[:8], "little")


def _cases(bundle: Bundle, model: Model, private: Path, ids: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Every (input, draw) to judge: draw 0 is the stored input, further draws re-sample ``reference.seeds``."""
    seeds = (model.judged.get("reference") or {}).get("seeds") or {}
    count = int(seeds.get("count", 1))
    cases = []
    for split, ident in ids:
        source = _derived(private, model.name, split, ident)
        home = bundle.root / model.name / split / ident
        home.mkdir(parents=True, exist_ok=True)
        cases.append(
            {"split": split, "ident": ident, "draw": 0, "inputs": source, "out": home, "name": f"{split}/{ident}"}
        )
        for k in range(1, count):
            out = home / "seeds" / str(k)
            out.mkdir(parents=True, exist_ok=True)
            with np.load(source) as data:
                arrays = {n: np.asarray(data[n]) for n in data.files}
            key = str(seeds["input"])
            rng = np.random.default_rng(_draw_seed(split, ident, key, k))
            arrays[key] = rng.standard_normal(arrays[key].shape).astype(arrays[key].dtype)
            INP._save_npz(out / "inputs.npz", **arrays)  # noqa: SLF001
            cases.append(
                {
                    "split": split,
                    "ident": ident,
                    "draw": k,
                    "inputs": out / "inputs.npz",
                    "out": out,
                    "name": f"{split}/{ident}#s{k}",
                }
            )
    return cases


def build(
    bundle: Bundle,
    model: Model,
    *,
    private: Path,
    fp_python: Path,
    fp_request: dict[str, Any],
    fp_env: dict[str, str],
    splits: tuple[str, ...] = ("visible", "sealed"),
    keep_call: Callable[[str, int], bool] | None = None,
    log: Callable[[str], None] = print,
    workers: int = 1,
    worker_memory: int | None = None,
) -> dict[str, Any]:
    """Every case's references (see the module docstring); returns the ``reference_quality.json`` summary.

    ``workers`` > 1 runs the INT8 cases in that many single-threaded processes and splits the FP
    reference the same way. Each worker has its own chain scratch and its own cache; files, rows and log
    lines come out in case order, with the bytes of a serial build. The count is bounded by the cases and
    by the host's available memory over ``worker_memory`` bytes per worker (default:
    :func:`estimate_worker_memory`). With workers, ``keep_call`` must be picklable (a module-level
    function, not a lambda).
    """
    math_configuration = _configuration_binding(bundle)
    math_policy = bundle_math_policy(model, bundle)
    _recheck_configuration(math_configuration)
    ids = input_ids(private, model.name, splits)
    if not ids:
        raise INP.InputError(f"no prepared inputs for {model.name} under {private}")
    kind = model.judged["kind"]
    cases = _cases(bundle, model, private, ids)
    catalogs = {p.name: bundle.catalog(model.name, p.name) for p in model.programs}
    count = worker_count(workers, len(cases), worker_memory or estimate_worker_memory(catalogs))
    if count < workers:
        log(f"[references] {model.name}: {count} of {workers} requested worker(s) (bounded by cases and memory)")
    started = time.monotonic()
    items = [{"id": c["name"], "inputs": str(c["inputs"]), "out": str(c["out"] / "fp.npz")} for c in cases]
    reference = dict(model.judged.get("reference") or {})
    request = {**fp_request, "items": items, "input_worker": str(INP.WORKER)}
    if kind == "next_token_distribution":
        request.update(prefill_tokens=int(model.chain["prefill_tokens"]), decode_steps=int(model.chain["decode_steps"]))
    runs = {"fp.npz": reference.get("precision", "native")}
    if reference.get("baseline"):
        runs["fp_baseline.npz"] = reference["baseline"]
    for name, precision in runs.items():
        _fp_reference(
            fp_python,
            FP_TASK[kind],
            {
                **request,
                "precision": precision,
                "items": [{**i, "out": str(Path(i["out"]).with_name(name))} for i in items],
            },
            bundle.root / model.name / ".fp_work",
            env={**fp_env, **FP_NUMERICS_ENV},
            shards=count,
        )
    log(f"[references] {model.name}: FP reference for {len(cases)} cases in {time.monotonic() - started:.0f} s")
    job = CaseJob(
        model=model,
        catalogs=catalogs,
        blobs=bundle.root / model.name / "blobs",
        baseline="fp_baseline.npz" in runs,
        keep_call=keep_call,
        host_math_policy=math_policy,
        host_math_configuration=math_configuration,
    )
    scratch = bundle.root / model.name / ".chain_work"
    if count == 1:
        cache = CH.ChainCache()
        results = (reference_case(job, case, cache=cache, scratch=scratch) for case in cases)
    else:
        results = iter(_pooled_cases(job, cases, count, scratch, log))
    summary: dict[str, Any] = {
        "model": model.name,
        "fp_numerics_env": FP_NUMERICS_ENV,
        "fp_runs": runs,
        "int8_numeric_contract": LN.numeric_contract(),
        "inputs": {},
    }
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for case, result in zip(cases, results, strict=True):
        split, ident, here = case["split"], case["ident"], case["out"]
        rows.append({"split": split, "draw": case["draw"], **result["metrics"]})
        if result["baseline"] is not None:
            baseline_rows.append({"split": split, "draw": case["draw"], **result["baseline"]})
        for program, calls in result["calls"].items():
            CAT.write_calls(
                here / program / "calls.json",
                calls,
                model=model.name,
                program=program,
                input=f"{split}/{ident}",
                step=0,
            )
        headline = {k: v for k, v in result["metrics"].items() if k != "per_channel"}
        summary["inputs"][case["name"]] = {
            **headline,
            "steps": result["steps"],
            "seconds": result["seconds"],
            "int8_numeric_contract": result["int8_numeric_contract"],
            "evaluator_sha256": result["evaluator_sha256"],
        }
        log(f"[references] {model.name} {case['name']}: {json.dumps(headline)}")
    summary["admission"] = _admission(model, rows, baseline_rows)
    (bundle.root / model.name / "reference_quality.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


@dataclass
class CaseJob:
    """What every INT8 reference case of one model needs, in whichever process it runs."""

    model: Model
    catalogs: dict[str, CAT.ProgramCatalog]
    blobs: Path
    baseline: bool
    keep_call: Callable[[str, int], bool] | None = None
    host_math_policy: str | None = None
    host_math_configuration: tuple[Path, str] | None = None


def _configuration_binding(bundle: Bundle) -> tuple[Path, str] | None:
    path = bundle.root / "configuration" / "models.yaml"
    return (path, hashlib.sha256(path.read_bytes()).hexdigest()) if path.is_file() else None


def _recheck_configuration(binding: tuple[Path, str] | None) -> None:
    if binding is not None and hashlib.sha256(binding[0].read_bytes()).hexdigest() != binding[1]:
        raise INP.InputError("reference host math configuration changed during derivation")


def bundle_math_policy(model: Model, bundle: Bundle) -> str:
    """Resolve the same common/model declaration used by whole-model execution."""
    from merlin.model_check.models import load

    configuration = bundle.root / "configuration" / "models.yaml"
    common = load(configuration).full if configuration.is_file() else {}
    return selected_host_math_policy(model, common)


@contextmanager
def reference_math_context(model: Model, bundle: Bundle, work: Path):
    """Select primitives from declared data, without invoking host lowering or a device package."""
    from merlin.runtime.host_math_reference import host_math_context, reference_math

    binding = _configuration_binding(bundle)
    evaluator = reference_math(bundle_math_policy(model, bundle), work / "host_math")
    _recheck_configuration(binding)
    with host_math_context(evaluator):
        yield
    _recheck_configuration(binding)


def reference_case(job: CaseJob, case: dict[str, Any], *, cache: CH.ChainCache | None, scratch: Path) -> dict[str, Any]:
    """Derive a case under its explicitly selected host arithmetic implementation."""
    from merlin.runtime.host_math_reference import host_math_context, reference_math

    _recheck_configuration(job.host_math_configuration)
    policy = (job.host_math_policy if job.host_math_policy is not None
              else selected_host_math_policy(job.model))
    evaluator = reference_math(policy, scratch / "host_math")
    with host_math_context(evaluator):
        return _reference_case(job, case, cache=cache, scratch=scratch)


def _reference_case(
    job: CaseJob, case: dict[str, Any], *, cache: CH.ChainCache | None, scratch: Path
) -> dict[str, Any]:
    """One case's INT8 reference chain, its metrics against FP, and (draw 0) its device calls.

    Writes the case's own files (``int8.npz``, ``metrics_int8.json``, ``metrics_baseline.json``) and
    returns its rows; the caller writes ``calls.json``. ``scratch`` must be this process's own."""
    model, here = job.model, case["out"]
    numeric_contract = LN.numeric_contract()
    evaluator_sha256 = hashlib.sha256(Path(LN.__file__).read_bytes()).hexdigest()
    kind = model.judged["kind"]
    blobs = CAT.BlobStore(job.blobs)
    t0 = time.monotonic()
    with np.load(case["inputs"]) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    calls_by_program: dict[str, list[dict[str, Any]]] = {}

    def record(program: str, step: int, catalog: CAT.ProgramCatalog, args: list[np.ndarray]) -> None:
        if step != 0 or case["draw"] != 0 or not catalog.interfaces:
            return
        calls_by_program[program] = CAT.record_calls(
            catalog,
            args,
            blobs=blobs,
            keep_call=job.keep_call,
            module=cache.module(catalog.dev_prepared) if cache is not None else None,
            constants=cache.constants if cache is not None else None,
        )

    run = CH.run_reference(model, job.catalogs, arrays, scratch, record=record, cache=cache)
    _recheck_configuration(job.host_math_configuration)
    if (
        LN.numeric_contract() != numeric_contract
        or hashlib.sha256(Path(LN.__file__).read_bytes()).hexdigest() != evaluator_sha256
    ):
        raise INP.InputError("reference evaluator source/policy changed during the chain")
    judged = CH.judged_outputs(model, run)
    with np.load(here / "fp.npz") as data:
        fp = [np.asarray(data[k]) for k in data.files]
    metrics = M.model_metrics(kind, judged, fp, model.judged)
    _save(here / "int8.npz", judged)
    numeric_receipt = {
        "schema": "merlin.model_check.reference_numeric_contract.v1",
        "numeric_contract": numeric_contract,
        "evaluator_sha256": evaluator_sha256,
        "outputs_sha256": hashlib.sha256((here / "int8.npz").read_bytes()).hexdigest(),
    }
    (here / "reference_numeric_contract.json").write_text(json.dumps(numeric_receipt, indent=2) + "\n")
    (here / "metrics_int8.json").write_text(json.dumps(metrics, indent=1) + "\n")
    base = None
    if job.baseline:
        with np.load(here / "fp_baseline.npz") as data:
            base = M.model_metrics(kind, [np.asarray(data[k]) for k in data.files], fp, model.judged)
        (here / "metrics_baseline.json").write_text(json.dumps(base, indent=1) + "\n")
    return {
        "int8_numeric_contract": numeric_contract,
        "evaluator_sha256": evaluator_sha256,
        "metrics": metrics,
        "baseline": base,
        "calls": calls_by_program,
        "steps": CH.summary(run),
        "seconds": round(time.monotonic() - t0, 1),
    }


# ------------------------------------------------------------------------------------------ the pool

#: One worker's peak memory with every cache full is estimated as a base plus a multiple of the bytes
#: of the files its caches hold (the stages' weights and lifted constants, the reference and device
#: programs). Measured peaks on 2026-10-18: 31.6 GiB over 8.4 GiB of files, 23.7 GiB over 1.4 GiB;
#: the estimate is above both by at least 10%.
WORKER_MEMORY_BASE = 24 << 30
WORKER_MEMORY_PER_BYTE = 1.5
#: A pool worker runs on one thread: BLAS threads would only oversubscribe the workers' CPUs.
WORKER_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def estimate_worker_memory(catalogs: dict[str, CAT.ProgramCatalog]) -> int:
    """An upper estimate of one worker's peak memory, from the sizes of the files its caches hold."""
    sources: set[Path] = set()
    for catalog in catalogs.values():
        sources.update(Path(p).resolve() for p in (catalog.ref_prepared, catalog.dev_prepared) if Path(p).is_file())
        for name in ("weights.safetensors", "extra.npz"):
            if (Path(catalog.stage) / name).is_file():
                sources.add((Path(catalog.stage) / name).resolve())
    return WORKER_MEMORY_BASE + int(WORKER_MEMORY_PER_BYTE * sum(p.stat().st_size for p in sources))


def available_memory() -> int | None:
    """The host's ``MemAvailable`` in bytes, or None where the host does not state it."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key == "MemAvailable":
                return int(value.split()[0]) * 1024
    except OSError:
        return None
    return None


def worker_count(requested: int, cases: int, per_worker: int, available: int | None = None) -> int:
    """``requested`` bounded by the number of cases and by how many workers of ``per_worker`` bytes the
    host's available memory holds; at least 1."""
    count = max(1, min(int(requested), int(cases)))
    available = available_memory() if available is None else available
    if count > 1 and per_worker > 0 and available is not None:
        count = max(1, min(count, available // per_worker))
    return count


def _shards(items: list[Any], count: int) -> list[list[Any]]:
    """Round-robin: item ``i`` goes to shard ``i % count``."""
    return [items[k::count] for k in range(count) if items[k::count]]


def _fp_reference(
    python: Path, task: str, request: dict[str, Any], scratch: Path, *, env: dict[str, str], shards: int
) -> None:
    """The FP worker over ``request["items"]``, split round-robin over ``shards`` concurrent processes
    (each item's output depends on that item alone)."""
    if shards <= 1:
        INP.run_worker(python, task, request, scratch, worker=REFERENCE_WORKER, env=env)
        return
    parts = _shards(request["items"], shards)
    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        futures = [
            pool.submit(
                INP.run_worker,
                python,
                task,
                {**request, "items": part},
                scratch / "shards" / str(k),
                worker=REFERENCE_WORKER,
                env=env,
            )
            for k, part in enumerate(parts)
        ]
        for future in futures:
            future.result()


def _pooled_cases(
    job: CaseJob, cases: list[dict[str, Any]], count: int, scratch: Path, log: Callable[[str], None]
) -> list[dict[str, Any]]:
    """Every case's :func:`reference_case` result, in case order, from ``count`` worker processes.

    Each worker walks the cases in order and runs every case it is first to claim (an exclusive
    create of ``claims/<i>``), so a slow case (a first draw, which records device calls) does not hold
    the others back. A worker keeps one cache and its own scratch (``<scratch>/workers/<k>/``) and
    stores each result as its case finishes; the results are read back in case order once every
    worker is done. Which worker ran a case does not change its bytes."""
    root = scratch / "workers"
    try:
        payload = pickle.dumps({"job": job, "cases": list(enumerate(cases)), "root": root})
    except (pickle.PicklingError, AttributeError, TypeError) as error:
        raise INP.InputError(f"reference workers need a picklable job (a lambda keep_call?): {error}") from error
    for name in ("claims", "results"):
        (root / name).mkdir(parents=True, exist_ok=True)
        for stale in (root / name).iterdir():
            stale.unlink()
    (root / "request.pkl").write_bytes(payload)
    procs = []
    for k in range(count):
        (root / str(k)).mkdir(parents=True, exist_ok=True)
        with (root / str(k) / "worker.log").open("w") as out:
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "merlin.model_check.chain_worker", str(root / "request.pkl"), str(k)],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **WORKER_ENV},
                )
            )
    log(f"[references] {job.model.name}: {len(cases)} INT8 cases over {count} worker processes")
    failed = [k for k, proc in enumerate(procs) if proc.wait() != 0]
    if failed:
        tails = "; ".join(f"worker {k}: {(root / str(k) / 'worker.log').read_text()[-1500:]}" for k in failed)
        raise INP.InputError(f"reference worker(s) {failed} failed: {tails}")
    return [pickle.loads((root / "results" / f"{i}.pkl").read_bytes()) for i in range(len(cases))]


def run_worker(request_path: Path, worker: int) -> None:
    """Worker ``worker`` of :func:`_pooled_cases`: every case it claims, in order, with one cache and
    its own scratch."""
    request = pickle.loads(Path(request_path).read_bytes())
    root = Path(request["root"])
    cache = CH.ChainCache()
    for index, case in request["cases"]:
        try:
            claim = os.open(root / "claims" / str(index), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(claim, str(worker).encode())
        os.close(claim)
        result = reference_case(request["job"], case, cache=cache, scratch=root / str(worker) / "chain")
        out = root / "results" / f"{index}.pkl"
        tmp = out.with_name(f"{index}.tmp")
        tmp.write_bytes(pickle.dumps(result))
        tmp.replace(out)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)  # KiB on Linux
        print(f"[references] {case['name']}: {result['seconds']} s, peak {peak:.1f} GiB", flush=True)


def _admission(model: Model, rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rules = model.judged.get("admission") or ()
    return {
        split: M.admission(
            rules, [r for r in rows if r["split"] == split], [r for r in baseline_rows if r["split"] == split]
        )
        for split in sorted({r["split"] for r in rows})
    }


def rejudge(bundle: Bundle, model: Model) -> dict[str, Any]:
    """``model``'s admission rules over a built bundle's stored per-case metrics (``metrics_int8.json``,
    ``metrics_baseline.json``), nothing recomputed; ``reference_quality.json``'s ``admission`` is replaced."""
    path = bundle.root / model.name / "reference_quality.json"
    summary = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for name in summary["inputs"]:
        where, _, draw = name.partition("#s")
        split, _, ident = where.partition("/")
        here = bundle.root / model.name / split / ident
        if draw:
            here = here / "seeds" / draw
        row = {"split": split, "draw": int(draw or 0)}
        rows.append({**row, **json.loads((here / "metrics_int8.json").read_text())})
        if (here / "metrics_baseline.json").is_file():
            baseline_rows.append({**row, **json.loads((here / "metrics_baseline.json").read_text())})
    summary["admission"] = _admission(model, rows, baseline_rows)
    path.write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _save(path: Path, arrays: list[np.ndarray]) -> None:
    INP._save_npz(path, **{f"out{i}": np.asarray(a) for i, a in enumerate(arrays)})  # noqa: SLF001
