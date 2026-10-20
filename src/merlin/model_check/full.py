"""``check --mode full``: every program's linked image, each input's whole chain on Spike, judged.

Contract: ``docs/reference/model_check.md`` §2 (full mode) and §4. Per submission each program's image is
built once (:mod:`merlin.model_check.full_build`). Per input the programs run as the models file chains
them (:func:`merlin.model_check.chain.bind`); every step is one run of its program's image relinked with
that step's bound inputs -- a recurrent program's state is an input like any other, carried by the chain
from the previous step's read-back outputs, so every step's outputs are observed whole.

Judged (the fast kernel replay runs first and contributes ``build`` and ``kernel_exact``):

- Default ``host-twin`` selection: ``differential`` is the integration gate. Each program with device
  groups also has a host twin: the
  same recipe and host code with every group on the host and no submission object. Every step's exact
  bound inputs run on both, and every output (integer and float) must be bit-identical. What the twin
  and the submission share -- Merlin's host code -- cannot be charged to the submission. A model whose
  ``full.differential`` is ``cached`` compares only with the bundle's twin cache: a step without a cached
  twin is ``pending`` (not run, not failed), and so is the gate unless a step differs.
- Explicit ``quantized-reference`` selection: complete chains are compared directly with every sealed
  judged INT8 output, under the recorded float policy and exact integer equality; no CPU twin or full
  NumPy reference recomputation is selected. The gate is ``quantized_reference``.
- Reported, not gated: ``model_metrics.numerics`` (contract §4.4 on the measured judged outputs, ratio
  rule vs FP) and ``model_metrics.host_drift`` with each output's ``host_drift`` (Merlin's compiled host
  code against the numpy INT8 reference chain). ``--decode-steps K`` below a recurrent program's declared
  steps judges the first K steps only and marks the gates partial.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from merlin.model_check import bundle as B
from merlin.model_check import chain as CH
from merlin.model_check import full_build as FB
from merlin.model_check import metrics as M
from merlin.model_check.models import Model, ModelsFileError

INTEGER = ("i1", "i8", "i16", "i32", "i64")
DERIVED = ("inputs.npz", "observation.npz")


class FullCheckError(RuntimeError):
    """Full mode cannot judge: an input, a reference or a build is missing."""


def step_counts(model: Model, decode_steps: int | None) -> dict[str, int]:
    """Steps each program runs; ``decode_steps`` caps a causal chain's recurrent programs."""
    counts = {p.name: p.steps for p in model.programs}
    if decode_steps is None:
        return counts
    if model.chain.get("kind") != "causal":
        raise ModelsFileError(f"--decode-steps applies to a causal chain; {model.name} is {model.chain.get('kind')}")
    if decode_steps < 1:
        raise ModelsFileError("--decode-steps must be at least 1")
    return {name: min(n, decode_steps) if n > 1 else n for name, n in counts.items()}


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _ordered(path: Path) -> list[np.ndarray]:
    """An npz's arrays in numeric key order (``out0, out1, ..., out10``), or file order."""
    with np.load(path) as data:
        keys = data.files
        digits = [k.lstrip("abcdefghijklmnopqrstuvwxyz_") for k in keys]
        if all(d.isdigit() for d in digits):
            keys = [k for _, k in sorted(zip((int(d) for d in digits), keys, strict=True))]
        return [np.asarray(data[k]) for k in keys]


def derived_input(bundle: B.Bundle, model: str, input_id: str) -> Path:
    """The model-ready arrays of one bundle input (``<private>/derived/<model>/<split>/<id>/``)."""
    directory = bundle.root.resolve().parent / "derived" / model / input_id
    for name in DERIVED:
        if (directory / name).is_file():
            return directory / name
    raise FullCheckError(f"no derived inputs for {model} {input_id} under {directory}")


def judged_inputs(
    bundle: B.Bundle, model: str, split: str, only: Sequence[str] | None, *, reference_only: bool = False
) -> list[str]:
    """Inputs of ``split`` that carry FP and INT8 references (``visible/v00``...), optionally filtered."""
    filename = "int8.npz" if reference_only else "fp.npz"
    found = [i for i in bundle.inputs(model, split) if (bundle.root / model / i / filename).is_file()]
    if only:
        found = [i for i in found if i in only or i.partition("/")[2] in only]
    if not found:
        raise FullCheckError(
            f"bundle has no judged {split} inputs for {model}" + (f" among {list(only)}" if only else "")
        )
    return found


# ------------------------------------------------------------------------------------ INT8 reference


def reference_chain(model: Model, bundle: B.Bundle, arrays, counts: dict[str, int], cache: Path) -> CH.ChainRun:
    """Recompute under the bundle's declared host primitive policy, binding its actual source bytes."""
    from merlin.model_check.references import reference_math_context

    with reference_math_context(model, bundle, cache):
        return _reference_chain(model, bundle, arrays, counts, cache)


def _reference_chain(model: Model, bundle: B.Bundle, arrays, counts: dict[str, int], cache: Path) -> CH.ChainRun:
    """Every INT8 reference step, with byte-bound disk reuse and one constant cache for the chain."""
    from merlin.model_check import catalog as CAT
    from merlin.model_check import full_reference_cache as RC
    from merlin.runtime import linalg_numpy as LN

    catalogs = {p.name: bundle.catalog(model.name, p.name) for p in model.programs}
    identity = RC.material(model, bundle, arrays, counts, catalogs)
    stored = RC.StepCache(cache, identity)
    evaluator = CH.ChainCache()
    run = CH.ChainRun()
    pending = []
    for program in model.programs:
        run.steps[program.name] = []
        catalog = catalogs[program.name]
        module = evaluator.module(catalog.ref_prepared)
        elements = CH.result_elements(module)
        arity = CAT._function_arity(module)  # noqa: SLF001
        for step in range(counts[program.name]):
            inputs = CH.bind(program, model, arrays, step, run)
            values = stored.get(program.name, step, inputs, len(elements))
            if values is None:
                args = evaluator.arguments(catalog, inputs, cache / "shadow" / program.name)
                values = CH.to_compute(
                    [np.asarray(v) for v in LN.evaluate(module, args[:arity], constants=evaluator.constants)], elements
                )
                pending.append((program.name, step, inputs, values))
            run.steps[program.name].append(values)
    # A source/input mutation during evaluation must never publish mixed answers under the old key.
    if RC.material(model, bundle, arrays, counts, catalogs) != identity:
        raise FullCheckError("INT8 reference inputs, capture or evaluator changed during the chain")
    for program, step, inputs, values in pending:
        stored.put(program, step, inputs, values)
    return run


# ------------------------------------------------------------------------------------------- judging


def _compare(value: np.ndarray, reference: np.ndarray, dtype: str) -> dict[str, Any]:
    value, reference = np.asarray(value), np.asarray(reference)
    row: dict[str, Any] = {"dtype": dtype, "shape": list(reference.shape)}
    if value.shape != reference.shape:
        return {**row, "exact": False, "detail": f"shape {list(value.shape)} != reference"}
    if dtype in INTEGER:
        mismatched = int(np.count_nonzero(value.astype(np.int64) != reference.astype(np.int64)))
        return {**row, "exact": mismatched == 0, "mismatched": mismatched}
    a, b = value.astype(np.float32), reference.astype(np.float32)
    exact = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    return {
        **row,
        "exact": exact,
        "mismatched": int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
        "rel_rmse_vs_int8": M.rel_rmse(a, b),
        "worst_vs_int8": M.worst(a, b),
    }


def model_gate(kind: str, ours: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, Any]:
    """§4.4 for one input: every check of the model's gate, and whether all hold."""
    checks: dict[str, Any] = {}
    if kind == "logits":
        checks["rel_rmse"] = M.ratio_rule(ours["rel_rmse"], ref["rel_rmse"], M.FLOOR_REL)
        checks["worst"] = M.ratio_rule(ours["worst"], ref["worst"], M.FLOOR_REL)
        for key in (k for k in ref if k.startswith("top") and k.endswith("_agree")):
            checks[key] = {"value": ours[key], "reference": ref[key], "pass": bool(ours[key] >= ref[key])}
    elif kind == "next_token_distribution":
        checks["kl_worst"] = M.ratio_rule(ours["kl_worst"], ref["kl_worst"], M.FLOOR_KL)
        checks["top1_agree"] = {
            "value": ours["top1_agree"],
            "reference": ref["top1_agree"],
            "pass": bool(ours["top1_agree"] >= ref["top1_agree"]),
        }
    elif kind == "per_channel":
        for channel, row in ours["per_channel"].items():
            base = ref["per_channel"][channel]
            checks[f"{channel}.rel_rmse"] = M.ratio_rule(row["rel_rmse"], base["rel_rmse"], M.FLOOR_REL)
            checks[f"{channel}.worst"] = M.ratio_rule(row["worst"], base["worst"], M.FLOOR_REL)
    else:
        raise ValueError(f"unknown judged kind {kind!r}")
    return {"checks": checks, "pass": all(c["pass"] for c in checks.values())}


def judged_metrics(model: Model, run: CH.ChainRun, fp: list[np.ndarray]) -> dict[str, Any]:
    """The model's judged outputs of ``run`` against FP, on as many distributions as ``run`` has."""
    judged = CH.judged_outputs(model, run)
    kind = model.judged["kind"]
    if kind == "next_token_distribution":
        rows = sum(int(np.prod(np.shape(a)[:-1])) for a in judged)
        fp = [np.asarray(fp[0]).reshape(-1, np.shape(fp[0])[-1])[:rows]]
    return M.model_metrics(kind, judged, fp, model.judged)


# ---------------------------------------------------------------------------------------- one input


def execute(
    target: str, record: Mapping[str, Any], stage: Path, inputs, step_dir: Path, deadline: float, *, twin: bool = False,
    admission=None,
) -> dict:
    """One program step: ``record``'s image relinked with ``inputs``, run on Spike, every output read back."""
    started = time.monotonic()
    if not twin and admission is None:
        from merlin.model_check.admission import service

        admission = service(target)
    elf = FB.relink(record, stage, inputs, step_dir, admission=admission)
    out: dict[str, Any] = {"relink_seconds": round(time.monotonic() - started, 1), "elf_bytes": elf.stat().st_size}
    admission = step_dir / "native_linked_elf.json"
    if admission.is_file():
        out["instruction_admission"] = {"path": str(admission), "sha256": FB.sha256(admission)}
    try:
        remaining = max(1.0, deadline - time.monotonic())
        console, wall = FB.run_host(elf, timeout_s=remaining) if twin else FB.run(target, elf, timeout_s=remaining)
    finally:
        FB.discard(step_dir)  # the relinked image and its generated inputs; the bound inputs stay
    (step_dir / "console.bin").write_bytes(console)  # kept only when it cannot be read back
    values, info = FB.read(console, stage, record["build_hash"])
    (step_dir / "console.bin").unlink()
    np.savez(step_dir / "outputs.npz", **{f"out{i}": v for i, v in enumerate(values)})
    cycles = info["metrics"].get("cycles")
    out.update(
        seconds=round(wall, 1),
        cycles_functional=int(cycles) if str(cycles).isdigit() else None,
        group_profile=info.get("group_profile"),
    )
    return {"values": values, "row": out}


def _same(values: Sequence[np.ndarray], twin: Sequence[np.ndarray]) -> list[int]:
    """Indices of the outputs whose bytes (dtype, shape, every bit) differ."""
    return [
        i
        for i, (a, b) in enumerate(zip(values, twin, strict=True))
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes()
    ]


def carried(values: Sequence[np.ndarray], specs: Sequence[tuple[list[int], str]]) -> list[np.ndarray]:
    """A step's read-back outputs as the chain carries them on (float results as float32)."""
    return [v if dt in INTEGER else np.asarray(v, np.float32) for v, (_shape, dt) in zip(values, specs, strict=True)]


def run_input(
    model: Model,
    *,
    target: str,
    builds: Mapping[str, dict[str, Any]],
    twins: Mapping[str, dict[str, Any] | None],
    caches: Mapping[str, Any],
    stages: Mapping[str, Path],
    specs: Mapping[str, list[tuple[list[int], str]]],
    arrays: Mapping[str, np.ndarray],
    counts: Mapping[str, int],
    work: Path,
    deadline: float,
    rows: list[dict[str, Any]],
    twin_pool,
    policy: str = "required",
    deferred: frozenset[str] = frozenset(),
    admission=None,
    comparison: str = "host-twin",
    completed_step: Callable[[str, int, Sequence[np.ndarray]], bool] | None = None,
    log=print,
) -> CH.ChainRun:
    """The submission's chain on one input: every step relinked, run on Spike and read back (each
    completed step appended to ``rows``). Every step must finish before ``deadline`` (monotonic).

    The differential: each step's exact bound inputs are looked up in the program's twin cache
    (:mod:`merlin.model_check.twins`); on a miss they run on the host twin (submitted to ``twin_pool``
    before the step itself, so the two overlap). The step's row records whether every output is
    bit-identical and whether the twin was ``cached`` or ``computed``. Under the ``cached`` policy a miss
    runs nothing and the step is ``pending``, as is every step of a ``deferred`` program (one with no twin
    cache in the bundle, whose twin was not built). Direct quantized-reference callers may compare
    each completed step immediately; a false callback result stops before the next execution."""
    from merlin.model_check import twins as TW

    if completed_step is not None and comparison != "quantized-reference":
        raise ValueError("completed-step comparison requires quantized-reference selection")
    run = CH.ChainRun(steps={program.name: [] for program in model.programs})
    pending = []
    for program in model.programs:
        stage, twin, cache = stages[program.name], twins.get(program.name), caches.get(program.name)
        for step in range(counts[program.name]):
            if deadline - time.monotonic() < 1:
                raise FullCheckError(f"the time budget ran out before {program.name}#{step}")
            inputs = CH.bind(program, model, arrays, step, run)
            future = cached = None
            if comparison == "host-twin" and twin is not None:
                cached = cache.get(TW.inputs_sha256(stage, inputs)) if cache is not None else None
                if cached is None and policy == "required":
                    future = twin_pool.submit(
                        execute, target, twin, stage, inputs, work / f"{program.name}.{step}.host", deadline, twin=True
                    )
            done = execute(target, builds[program.name], stage, inputs, work / f"{program.name}.{step}", deadline,
                           admission=admission)
            row = {"program": program.name, "step": step, "engine": "spike", "status": "completed", **done["row"]}
            values = done["values"]
            run.steps[program.name].append(carried(values, specs[program.name]))
            rows.append(row)
            log(f"[full] {model.name} {program.name}#{step}: {row['cycles_functional']} cycles, {row['seconds']} s")
            if comparison == "quantized-reference":
                row["comparison"] = {"selected": comparison, "host_twin": "not_selected"}
                if completed_step is not None and not completed_step(program.name, step, values):
                    log(f"[full] {model.name} {program.name}#{step}: selected output differs; stopping this input")
                    return run
            else:
                pending.append((row, values, twin is not None or program.name in deferred, future, cached))
    for row, values, has_twin, future, cached in pending:
        if not has_twin:
            row["differential"] = {"status": "identical", "detail": "a host-only program is its own host twin"}
            continue
        if cached is not None:
            twin_values, twin_row, source = cached[0], cached[1], "cached"
        elif future is None:
            row["differential"] = {
                "status": "pending",
                "twin": "uncached",
                "detail": "no cached host twin for these inputs (full.differential: cached): not run, not failed",
            }
            continue
        else:
            try:
                done = future.result()
            except Exception as error:  # noqa: BLE001 -- the step is reported, never silently skipped
                row["differential"] = {
                    "status": "error",
                    "twin": "computed",
                    "detail": f"{type(error).__name__}: {error}"[:1000],
                }
                continue
            twin_values, twin_row, source = done["values"], done["row"], "computed"
        differing = _same(values, twin_values)
        row["differential"] = {
            "status": "differs" if differing else "identical",
            "differing_outputs": differing,
            "twin": source,
            "twin_run": {k: twin_row.get(k) for k in ("seconds", "cycles_functional")},
        }
    return run


# ------------------------------------------------------------------------------------------ command


def _outputs_rows(model: Model, run: CH.ChainRun, ref: CH.ChainRun, specs) -> dict[tuple[str, int], list[dict]]:
    out = {}
    for program in model.programs:
        dtypes = [dt for _shape, dt in specs[program.name]]
        for step, (values, expected) in enumerate(zip(run.steps[program.name], ref.steps[program.name], strict=True)):
            out[(program.name, step)] = [
                {"output": i, **_compare(v, e, dtypes[i])}
                for i, (v, e) in enumerate(zip(values, expected, strict=True))
            ]
    return out


def build_root(work: Path, model: str) -> Path:
    """Where the whole-model images are built: ``work`` when it lies below the out root (the only place a
    whole-model build may write), else a fresh directory of this check's own under it. Runs, references and
    every other product stay in ``work``."""
    from merlin.common.paths import build_dir, out_dir

    if Path(work).resolve().is_relative_to(out_dir().resolve()):
        return Path(work)
    root = build_dir() / "model_check" / "full"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(tempfile.mkdtemp(prefix=f"{model}-{stamp}-", dir=root))


def default_jobs(inputs: int) -> int:
    """Inputs run concurrently when ``--jobs`` is not given: one per input, up to a quarter of the CPUs
    (each chain is one single-threaded Spike run at a time; host twins, run only on a cache miss, take at
    most half as many more -- 8 sealed inputs on 32 CPUs: 8 chains and at most 4 twins)."""
    return max(1, min(inputs, (os.cpu_count() or 4) // 4))


def _kernels(bundle: B.Bundle, model: str, program: str, build: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The program's device kernels (the bundle's grouping, which the build routed) with their identities."""
    from merlin.model_check.commands import _kernel_identity

    catalog = bundle.catalog(model, program)
    status = "linked" if build.get("device_routed") else "host"
    return [{"symbol": symbol, **_kernel_identity(catalog, symbol), "status": status} for symbol in catalog.interfaces]


def record_programs(report, bundle, model, builds, results):
    """Attach identical linked-image, kernel, census and step attribution for either full comparison."""
    from merlin.model_check.commands import _program_census

    replayed = {row.get("program"): row for row in report["programs"]}
    report["programs"] = [row for row in report["programs"] if row.get("program") not in builds]
    for program in model.programs:
        build = builds[program.name]
        fast = replayed.get(program.name) or {}
        report["programs"].append(
            {
                **{k: v for k, v in fast.items() if k not in ("build", "census", "kernels")},
                "program": program.name,
                "kernel_build": fast.get("build"),
                "build": {
                    k: build[k]
                    for k in ("status", "elf_sha256", "elf_bytes", "sections", "build_hash", "seconds", "reused")
                }
                | {"dedup_weights": build["dedup_weights"], "device_routed": build["device_routed"]}
                | ({"native_qualification": build["native_qualification"]} if "native_qualification" in build else {}),
                "census": fast.get("census") or _program_census(bundle, model.name, program.name, {}),
                "kernels": fast.get("kernels") or _kernels(bundle, model.name, program.name, build),
                "runs": [
                    {"input": r["input"], "split": r["input"].partition("/")[0], **row}
                    for r in results
                    for row in r["rows"]
                    if row["program"] == program.name
                ],
            }
        )


def validate(args) -> None:
    """Refuse the options full mode does not take (a usage error, before anything runs)."""
    if getattr(args, "program", None):
        raise ModelsFileError("check --mode full runs a model's whole chain; --program selects nothing here")
    from merlin.model_check import full_quantized as FQ

    if getattr(args, "full_comparison", "host-twin") not in ("host-twin", "quantized-reference"):
        raise ModelsFileError("unknown full-mode comparison selection")
    FQ.policy(args)


def check(args, report: dict, *, models, bundle: B.Bundle, admission=None, build_provider=None, log=print) -> dict:
    """Fill ``report`` with the full-mode judgement of ``args.model`` on ``args.split``.

    ``args.timeout_s`` bounds the whole check (builds, references and every Spike run); a step that would
    start past it is not run and the report says so. ``admission`` replaces model_check's structural
    linked-image admission (:mod:`merlin.model_check.admission`) with the caller's."""
    from merlin.common.paths import build_dir
    from merlin.llvmlower.c_runtime import _out_specs
    from merlin.model_check import full_reference_cache as RC

    started_check = time.monotonic()
    budget = float(args.timeout_s)
    deadline = started_check + budget - min(120.0, 0.05 * budget)  # leave time to judge and write the report
    model = models.model(args.model)
    validate(args)
    manifest = bundle.manifest()
    captures = Path(manifest.get("captures") or "")
    counts = step_counts(model, args.decode_steps)
    partial = any(counts[p.name] < p.steps for p in model.programs)
    direct = getattr(args, "full_comparison", "host-twin") == "quantized-reference"
    inputs = judged_inputs(bundle, model.name, args.split, getattr(args, "inputs", None), reference_only=direct)
    from merlin.model_check.full_reference_cache import cached_twin_readiness

    references = {}
    if direct:
        from merlin.model_check import full_quantized as FQ

        report["comparison"] = {"selected": "quantized-reference", "float_policy": FQ.policy(args),
                                "host_twin": {"selected": False, "status": "not_run"}}
        references = {input_id: FQ.sealed_reference(bundle, model, models.path, input_id) for input_id in inputs}
        exact = report["gates"].get("kernel_exact") or {}
        if exact.get("status") != "pass" or exact.get("partial"):
            report["gates"]["quantized_reference"] = {
                "status": "fail", "partial": partial,
                "detail": "full execution requires the complete exact accelerator-kernel gate",
            }
            return report
    else:
        report["comparison"] = {"selected": "host-twin"}
        readiness = cached_twin_readiness(bundle, model, counts, inputs)
        if readiness["status"] == "pending":
            log(f"[full] {model.name}: {readiness['detail']}: {', '.join(readiness['pending'])}")
    submission_sha = report["submission"]["tree_sha256"]
    work = Path(args.work or (build_dir() / "model_check" / "full")) / submission_sha[:16] / model.name
    images = build_root(work, model.name)
    settings = FB.BuildSettings.of(models, model, args.host_package)
    stages = {p.name: B.stage_dir(captures, model, p.name) for p in model.programs}
    specs = {name: _out_specs(stage / "model.mlir") for name, stage in stages.items()}
    report["bundle"] = {"path": str(bundle.root), "merlin_commit": manifest.get("merlin_commit")}
    timing: dict[str, Any] = {"build_seconds": 0.0, "reference_seconds": 0.0, "chain_seconds": {}}

    # This check's own runs (two checks of one model may run at once); builds and references are shared.
    (work / "runs").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    runs = Path(tempfile.mkdtemp(prefix=f"{args.split}-{stamp}-", dir=work / "runs"))
    builds: dict[str, dict[str, Any]] = {}
    for program in model.programs:
        log(f"[full] {model.name}/{program.name}: building")
        build = build_provider if build_provider is not None else FB.build
        builds[program.name] = build(
            stages[program.name],
            target=models.target,
            label=f"{model.name}:{program.name}",
            submission=args.submission,
            submission_sha256=submission_sha,
            settings=settings,
            directory=images / "programs" / program.name,
            admission=admission,
        )
        timing["build_seconds"] += 0 if builds[program.name]["reused"] else builds[program.name]["seconds"]
        log(f"[full] {model.name}/{program.name}: {builds[program.name]['elf_bytes']} B image")
    if direct:
        return FQ.check(
            args, report, models=models, model=model, bundle=bundle, references=references, builds=builds,
            stages=stages, specs=specs, counts=counts, inputs=inputs, runs=runs, deadline=deadline,
            partial=partial, timing=timing, log=log,
        )
    # The differential's host twins: the same recipe with every group on the host, no submission object.
    policy = (model.extra.get("full") or {}).get("differential", "required")
    cached_twins = (manifest.get("models") or {}).get(model.name, {}).get("twins") or {}
    twins: dict[str, dict[str, Any] | None] = {}
    deferred = set()  # cached policy and no twin cache in the bundle: every step pending, no twin built
    for program in model.programs:
        if not builds[program.name]["device_routed"]:
            twins[program.name] = None  # already host only: the image is its own twin
            continue
        if policy == "cached" and program.name not in cached_twins:
            twins[program.name] = None
            deferred.add(program.name)
            continue
        twins[program.name] = FB.build(
            stages[program.name],
            target=models.target,
            label=f"{model.name}:{program.name}",
            submission=args.submission,
            submission_sha256=submission_sha,
            settings=settings,
            directory=images / "host" / program.name,
            offload=False,
        )
        timing["build_seconds"] += 0 if twins[program.name]["reused"] else twins[program.name]["seconds"]
    from merlin.model_check import twins as TW

    caches = {
        name: TW.TwinCache(bundle, model.name, name, TW.identity(record))
        for name, record in twins.items()
        if record is not None
    }
    kernels_built = report["gates"].get("build")  # the kernel replay's, when it ran first
    report["gates"]["build"] = {
        "status": kernels_built["status"] if kernels_built else "pass",
        "detail": f"{len(builds)} program image(s) and {sum(t is not None for t in twins.values())} host twin(s) "
        "linked" + (f"; kernel replay: {kernels_built['detail']}" if kernels_built else ""),
    }

    def one(input_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {"input": input_id, "rows": []}
        try:
            arrays = _arrays(derived_input(bundle, model.name, input_id))
            started = time.monotonic()
            ref = reference_chain(model, bundle, arrays, counts, work / "references" / input_id)
            result["reference_seconds"] = round(time.monotonic() - started, 1)
            started = time.monotonic()
            run = run_input(
                model,
                target=models.target,
                builds=builds,
                twins=twins,
                caches=caches,
                stages=stages,
                specs=specs,
                arrays=arrays,
                counts=counts,
                work=runs / input_id,
                deadline=deadline,
                rows=result["rows"],
                twin_pool=twin_pool,
                policy=policy,
                admission=admission,
                deferred=frozenset(deferred),
                log=log,
            )
            result["chain_seconds"] = round(time.monotonic() - started, 1)
            fp = _ordered(bundle.root / model.name / input_id / "fp.npz")
            frozen = _ordered(bundle.root / model.name / input_id / "int8.npz")
            ours, base = judged_metrics(model, run, fp), judged_metrics(model, ref, fp)
            recomputed = CH.judged_outputs(model, ref)
            result.update(
                outputs=_outputs_rows(model, run, ref, specs),
                metrics=ours,
                reference=base,
                gate=model_gate(model.judged["kind"], ours, base),
                reference_consistent=RC.same_outputs(recomputed, RC.selected_outputs(model, counts, frozen)),
            )
        except Exception as error:  # noqa: BLE001 -- one input's failure is reported, the others still judged
            result["error"] = f"{type(error).__name__}: {error}"[:2000]
            log(f"[full] {model.name} {input_id}: {result['error']}")
        return result

    jobs = int(args.jobs) if getattr(args, "jobs", None) else default_jobs(len(inputs))
    # Twins run only on a cache miss; half a worker per chain keeps the machine's Spike count bounded.
    with ThreadPoolExecutor(max_workers=max(1, jobs // 2)) as twin_pool, ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, inputs))
    timing["jobs"] = jobs

    exact_rows = integer_bad = float_exact = float_total = 0
    steps = identical = pending = 0
    drift: list[float] = []
    per_input = []
    for result in results:
        split = result["input"].partition("/")[0]
        if "error" in result:
            report["failures"].append(
                {
                    "kind": "infrastructure",
                    "gate": None,
                    "input": result["input"],
                    "split": split,
                    "detail": result["error"],
                }
            )
            continue
        for row in result["rows"]:
            # Host drift: how far Merlin's compiled host code is from the numpy INT8 reference. Information
            # only -- the same drift is in the host twin, so the differential never charges it to a submission.
            row["outputs"] = [
                {"output": o["output"], "dtype": o["dtype"], "shape": o["shape"], "host_drift": o}
                for o in result["outputs"][(row["program"], row["step"])]
            ]
            for out in row["outputs"]:
                drifted = out["host_drift"]
                if out["dtype"] in INTEGER:
                    exact_rows += 1
                    integer_bad += int(not drifted["exact"])
                else:
                    float_total += 1
                    float_exact += int(drifted["exact"])
                    drift.append(float(drifted.get("rel_rmse_vs_int8") or 0.0))
            differential = row.get("differential") or {"status": "error", "detail": "no host twin ran"}
            steps += 1
            identical += int(differential["status"] == "identical")
            pending += int(differential["status"] == "pending")
            if differential["status"] not in ("identical", "pending"):
                report["failures"].append(
                    {
                        "kind": "gate" if differential["status"] == "differs" else "infrastructure",
                        "gate": "differential",
                        "input": result["input"],
                        "split": split,
                        "program": row["program"],
                        "step": row["step"],
                        "detail": f"outputs {differential.get('differing_outputs')} differ from the host twin"
                        if differential["status"] == "differs"
                        else differential.get("detail"),
                    }
                )
        per_input.append(
            {
                "split": split,
                "id": result["input"],
                **{k: v for k, v in result["metrics"].items() if k != "per_channel"},
                **({"per_channel": result["metrics"]["per_channel"]} if "per_channel" in result["metrics"] else {}),
                "reference": result["reference"],
                "checks": result["gate"]["checks"],
                "pass": result["gate"]["pass"],
                "reference_consistent": result["reference_consistent"],
            }
        )
        if not result["reference_consistent"]:
            report["failures"].append(
                {
                    "kind": "infrastructure",
                    "gate": None,
                    "input": result["input"],
                    "split": split,
                    "detail": "the recomputed INT8 reference chain differs from the bundle's",
                }
            )
        timing["reference_seconds"] += result["reference_seconds"]
        timing["chain_seconds"][result["input"]] = result["chain_seconds"]

    record_programs(report, bundle, model, builds, results)
    judged = len(per_input) == len(results)
    complete = judged and steps and identical + pending == steps
    report["gates"]["differential"] = {
        "status": ("pending" if pending else "pass") if complete else "fail",
        "partial": partial,
        "policy": policy,
        "pending": pending,
        "detail": f"{identical}/{steps} program steps bit-identical (every output) to the host twin on the same "
        "inputs"
        + (f", {pending} pending (no cached twin; full.differential: cached)" if pending else "")
        + f"; {len(per_input)}/{len(results)} input(s) completed",
    }
    report["model_metrics"] = {
        "source": "measured",
        "inputs": len(per_input),
        "per_input": per_input,
        "worst": M.worst_of(per_input),
        "mean": M.mean_of(per_input),
        # Reported, not a gate: on the measured outputs, which the differential ties to the host twin.
        "numerics": {
            "status": "pass" if judged and all(row["pass"] for row in per_input) else "fail",
            "partial": partial,
            "detail": f"contract §4.4 ({model.judged['kind']}) on {len(per_input)}/{len(results)} input(s)"
            + (f", first {counts} steps only" if partial else ""),
        },
        "host_drift": {
            "integer_outputs_exact": f"{exact_rows - integer_bad}/{exact_rows}",
            "float_outputs_exact": f"{float_exact}/{float_total}",
            "float_rel_rmse_worst": max(drift) if drift else 0.0,
            "detail": "Merlin's compiled host code against the numpy INT8 reference chain; information only",
        },
    }
    report["timing"].update({k: (round(v, 1) if isinstance(v, float) else v) for k, v in timing.items()})
    return report
