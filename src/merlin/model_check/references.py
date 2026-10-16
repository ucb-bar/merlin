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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from merlin.model_check import catalog as CAT
from merlin.model_check import chain as CH
from merlin.model_check import inputs as INP
from merlin.model_check import metrics as M
from merlin.model_check.bundle import Bundle
from merlin.model_check.models import Model

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
) -> dict[str, Any]:
    ids = input_ids(private, model.name, splits)
    if not ids:
        raise INP.InputError(f"no prepared inputs for {model.name} under {private}")
    kind = model.judged["kind"]
    cases = _cases(bundle, model, private, ids)
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
        INP.run_worker(
            fp_python,
            FP_TASK[kind],
            {
                **request,
                "precision": precision,
                "items": [{**i, "out": str(Path(i["out"]).with_name(name))} for i in items],
            },
            bundle.root / model.name / ".fp_work",
            worker=REFERENCE_WORKER,
            env={**fp_env, **FP_NUMERICS_ENV},
        )
    log(f"[references] {model.name}: FP reference for {len(cases)} cases in {time.monotonic() - started:.0f} s")
    catalogs = {p.name: bundle.catalog(model.name, p.name) for p in model.programs}
    blobs = bundle.blobs(model.name)
    summary: dict[str, Any] = {"model": model.name, "fp_numerics_env": FP_NUMERICS_ENV, "fp_runs": runs, "inputs": {}}
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for case in cases:
        split, ident, here = case["split"], case["ident"], case["out"]
        t0 = time.monotonic()
        with np.load(case["inputs"]) as data:
            arrays = {k: np.asarray(data[k]) for k in data.files}
        calls_by_program: dict[str, list[dict[str, Any]]] = {}

        def record(
            program: str, step: int, catalog: CAT.ProgramCatalog, args: list[np.ndarray], draw: int = case["draw"]
        ) -> None:
            if step != 0 or draw != 0 or not catalog.interfaces:
                return
            calls_by_program[program] = CAT.record_calls(catalog, args, blobs=blobs, keep_call=keep_call)

        run = CH.run_reference(model, catalogs, arrays, bundle.root / model.name / ".chain_work", record=record)
        judged = CH.judged_outputs(model, run)
        with np.load(here / "fp.npz") as data:
            fp = [np.asarray(data[k]) for k in data.files]
        metrics = M.model_metrics(kind, judged, fp, model.judged)
        _save(here / "int8.npz", judged)
        (here / "metrics_int8.json").write_text(json.dumps(metrics, indent=1) + "\n")
        rows.append({"split": split, "draw": case["draw"], **metrics})
        if "fp_baseline.npz" in runs:
            with np.load(here / "fp_baseline.npz") as data:
                base = M.model_metrics(kind, [np.asarray(data[k]) for k in data.files], fp, model.judged)
            (here / "metrics_baseline.json").write_text(json.dumps(base, indent=1) + "\n")
            baseline_rows.append({"split": split, "draw": case["draw"], **base})
        for program, calls in calls_by_program.items():
            CAT.write_calls(
                here / program / "calls.json",
                calls,
                model=model.name,
                program=program,
                input=f"{split}/{ident}",
                step=0,
            )
        headline = {k: v for k, v in metrics.items() if k != "per_channel"}
        summary["inputs"][case["name"]] = {
            **headline,
            "steps": CH.summary(run),
            "seconds": round(time.monotonic() - t0, 1),
        }
        log(f"[references] {model.name} {case['name']}: {json.dumps(headline)}")
    summary["admission"] = _admission(model, rows, baseline_rows)
    (bundle.root / model.name / "reference_quality.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


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
