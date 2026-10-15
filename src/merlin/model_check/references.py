"""Per-input references of a bundle: FP outputs, the INT8 reference chain, its metrics, and device calls.

For every prepared evaluation input (``<private>/derived/<model>/<split>/<id>/``) this writes into the
bundle, under ``<model>/<split>/<id>/``:

- ``fp.npz``            the FP reference's judged outputs (from :mod:`reference_worker`, pinned env);
- ``int8.npz``          the INT8 reference chain's judged outputs (:mod:`chain`, ``linalg_numpy``);
- ``metrics_int8.json`` the INT8 reference's metrics against FP (reference admission, contract §4.5);
- ``<program>/calls.json`` the device calls of each program's first step (the replayed boundaries).
"""

from __future__ import annotations

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
    out_dir = {key: bundle.root / model.name / key[0] / key[1] for key in ids}
    for directory in out_dir.values():
        directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    items = [
        {"id": f"{s}/{i}", "inputs": str(_derived(private, model.name, s, i)), "out": str(out_dir[(s, i)] / "fp.npz")}
        for s, i in ids
    ]
    request = {**fp_request, "items": items, "input_worker": str(INP.WORKER)}
    if kind == "next_token_distribution":
        request.update(prefill_tokens=int(model.chain["prefill_tokens"]), decode_steps=int(model.chain["decode_steps"]))
    INP.run_worker(
        fp_python, FP_TASK[kind], request, bundle.root / model.name / ".fp_work", worker=REFERENCE_WORKER, env=fp_env
    )
    log(f"[references] {model.name}: FP reference for {len(ids)} inputs in {time.monotonic() - started:.0f} s")
    catalogs = {p.name: bundle.catalog(model.name, p.name) for p in model.programs}
    blobs = bundle.blobs(model.name)
    summary: dict[str, Any] = {"model": model.name, "inputs": {}}
    for split, ident in ids:
        t0 = time.monotonic()
        with np.load(_derived(private, model.name, split, ident)) as data:
            arrays = {k: np.asarray(data[k]) for k in data.files}
        calls_by_program: dict[str, list[dict[str, Any]]] = {}

        def record(program: str, step: int, catalog: CAT.ProgramCatalog, args: list[np.ndarray]) -> None:
            if step != 0 or not catalog.interfaces:
                return
            calls_by_program[program] = CAT.record_calls(catalog, args, blobs=blobs, keep_call=keep_call)

        run = CH.run_reference(model, catalogs, arrays, bundle.root / model.name / ".chain_work", record=record)
        judged = CH.judged_outputs(model, run)
        with np.load(out_dir[(split, ident)] / "fp.npz") as data:
            fp = [np.asarray(data[k]) for k in data.files]
        metrics = M.model_metrics(kind, judged, fp, model.judged)
        _save(out_dir[(split, ident)] / "int8.npz", judged)
        (out_dir[(split, ident)] / "metrics_int8.json").write_text(json.dumps(metrics, indent=1) + "\n")
        for program, calls in calls_by_program.items():
            CAT.write_calls(
                out_dir[(split, ident)] / program / "calls.json",
                calls,
                model=model.name,
                program=program,
                input=f"{split}/{ident}",
                step=0,
            )
        headline = {k: v for k, v in metrics.items() if k != "per_channel"}
        summary["inputs"][f"{split}/{ident}"] = {
            **headline,
            "steps": CH.summary(run),
            "seconds": round(time.monotonic() - t0, 1),
        }
        log(f"[references] {model.name} {split}/{ident}: {json.dumps(headline)}")
    (bundle.root / model.name / "reference_quality.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _save(path: Path, arrays: list[np.ndarray]) -> None:
    INP._save_npz(path, **{f"out{i}": np.asarray(a) for i, a in enumerate(arrays)})  # noqa: SLF001
