"""Reproduce Voyager's Table 3 accuracy cells (arXiv 2509.15205) with Voyager's own quantization code.

The merlin-side driver, run with ``.venv/bin/python``. It resolves the dataset caches, runs the
Voyager-env worker ``voyager_accuracy_eval.py`` once per model, records provenance, and compares every
cell against the paper (the plan's +-0.3 top-1 parity band) and against the accelerator regression's
own gold table (its +-1.0 tolerance). The protocol is in ../AGENT.md.

  # SST-2 (public): BERT-base and MobileBERT-TINY on the full 872-sentence validation split
  .venv/bin/python merlin/experiments/dataset_accuracy/scripts/run_table3.py --task sst2
  # ImageNet (gated): all 50k images plus Voyager's own test_codegen.py --evaluate path on its 1000
  .venv/bin/python merlin/experiments/dataset_accuracy/scripts/run_table3.py --task imagenet \\
      --fetch --literal
  # ImageNet pipeline smoke on synthetic JPEGs: no dataset access, never compared to the paper
  .venv/bin/python merlin/experiments/dataset_accuracy/scripts/run_table3.py --task imagenet \\
      --synthetic 200 --smoke-out /scratch/.../smoke
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import yaml

from merlin.common import provenance
from merlin.common.artifacts import cache_dir, new_product
from merlin.common.paths import build_dir, env

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
WORKER = HERE / "voyager_accuracy_eval.py"
PAPER = HERE.parent / "paper_table3.yaml"

IMAGENET_REPO = "timm/imagenet-1k-wds"  # what Voyager's retrieve_dataset streams
IMAGENET_REVISION = "cdf403ce12f01022a0c36e584e588c0b9cebc4af"
N_VAL_SHARDS = 64
VAL_BYTES = 7.0e9  # hub file metadata at IMAGENET_REVISION (validation split only; never train)
DISK_MARGIN_BYTES = 10e9  # refuse a download that would leave less than this free (shared filesystem)
PAPER_SUBSET_N = 1000  # test_codegen.py: imagenet.retrieve_dataset(1000, "resnet") under --dump_dataset
PARITY_BAND = 0.3  # plan G3b: match the paper within +-0.3 top-1 points before any claim
Z95 = 1.959963984540054
QUANTIZED = ("E4M3", "Posit8", "INT8", "MXINT8")
#: our model name -> the row in paper_table3.yaml
PAPER_ROW = {"resnet18": "resnet18", "resnet50": "resnet50", "bert": "bert_base", "mobilebert": "mobilebert_tiny"}
TASKS = {
    "imagenet": {
        "models": "resnet18,resnet50",
        "target": "imagenet-1k",
        "sources": (
            "test/test_codegen.py",
            "test/run_ci.py",
            "test/utils/models/torchvision_models.py",
            "test/utils/dataset/imagenet.py",
            "src/voyager_compiler/quantization/quantize_pt2e.py",
        ),
    },
    "sst2": {
        "models": "bert,mobilebert",
        "target": "glue-sst2",
        "sources": (
            "test/test_codegen.py",
            "test/run_ci.py",
            "test/utils/models/bert.py",
            "test/utils/models/mobilebert.py",
            "test/utils/dataset/glue.py",
            "src/voyager_compiler/quantization/quantize_pt2e.py",
        ),
    },
}


# ---------------------------------------------------------------------------------------- helpers
def wilson(k: int, n: int) -> list[float]:
    p = k / n
    d = 1 + Z95**2 / n
    c = (p + Z95**2 / (2 * n)) / d
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / d
    return [round(100 * (c - h), 3), round(100 * (c + h), 3)]


def acc_block(k: int, n: int, paper: float | None, gold: float | None = None, gold_tol: float | None = None) -> dict:
    acc = 100 * k / n
    lo, hi = wilson(k, n)
    b = {"correct": k, "n": n, "top1": round(acc, 3), "wilson95": [lo, hi]}
    if paper is not None:
        b["delta_vs_paper"] = round(acc - paper, 3)
        b["within_band"] = abs(acc - paper) <= PARITY_BAND + 1e-9
        b["paper_in_wilson95"] = lo <= paper <= hi
    if gold is not None:
        b["delta_vs_accel_gold"] = round(acc - gold, 3)
        b["within_accel_gold_tolerance"] = abs(acc - gold) < gold_tol
    return b


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def voyager_env(threads: int, tmp: Path) -> tuple[Path, Path, Path, dict]:
    root = Path(env("MERLIN_EXT_VOYAGER_COMPILER") or build_dir() / "external" / "voyager-compiler").resolve()
    accel = Path(env("MERLIN_EXT_VOYAGER_ACCELERATOR") or build_dir() / "external" / "voyager-accelerator").resolve()
    for key, path in (("MERLIN_EXT_VOYAGER_COMPILER", root), ("MERLIN_EXT_VOYAGER_ACCELERATOR", accel)):
        os.environ.setdefault(key, str(path))  # so provenance.verify finds the checkouts
    vpy = build_dir() / "voyager-venv" / "bin" / "python"
    tmp.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ)
    e.update(
        MERLIN_EXT_VOYAGER_COMPILER=str(root),
        MERLIN_EXT_VOYAGER_ACCELERATOR=str(accel),
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        TMPDIR=str(tmp),
        HF_HOME=str(cache_dir("huggingface")),
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    e.pop("HF_TOKEN", None)  # workers read cached/local files only
    return root, accel, vpy, e


def run_worker(vpy: Path, venv: dict, args: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        return subprocess.run(
            [str(vpy), str(WORKER), *args], env=venv, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        ).returncode


def _blocked_line(log: Path) -> str | None:
    if not log.is_file():
        return None
    return next(
        (line.partition("BLOCKED:")[2].strip() for line in log.read_text().splitlines() if line.startswith("BLOCKED:")),
        None,
    )


# ---------------------------------------------------------------------------------------- ImageNet data
def fetch_shards(dest: Path, vpy: Path, venv: dict) -> dict:
    """Download the ImageNet validation shards (only) into the purgeable cache, or say why not."""
    if len(list(dest.glob("imagenet1k-validation-*.tar"))) == N_VAL_SHARDS:
        return {"status": "present"}
    free = shutil.disk_usage(dest).free
    if free - VAL_BYTES < DISK_MARGIN_BYTES:
        return {
            "status": "blocked",
            "reason": f"disk: {free / 1e9:.1f} GB free on the cache "
            f"filesystem; the {VAL_BYTES / 1e9:.1f} GB validation split would leave less than "
            f"{DISK_MARGIN_BYTES / 1e9:.0f} GB",
        }
    token = env("HF_TOKEN")
    if not token:
        return {"status": "blocked", "reason": "no HF_TOKEN in the environment or .env"}
    code = (
        "import os, sys\nfrom huggingface_hub import snapshot_download\n"
        f"snapshot_download({IMAGENET_REPO!r}, repo_type='dataset', "
        f"revision={IMAGENET_REVISION!r}, allow_patterns=['imagenet1k-validation-*.tar'], "
        "local_dir=sys.argv[1], max_workers=4, token=os.environ['HF_TOKEN'])\n"
    )
    e = {**venv, "HF_TOKEN": token, "HF_HUB_OFFLINE": "0", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    proc = subprocess.run(
        [str(vpy), "-c", code, str(dest)], env=e, capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    if proc.returncode == 0:
        return {"status": "downloaded"}
    tail = [line for line in proc.stderr.splitlines() if line.strip()]
    reason = next((line for line in reversed(tail) if "Error" in line), tail[-1] if tail else "")
    return {"status": "blocked", "reason": f"{IMAGENET_REPO} download failed: {reason.strip()[:300]}"}


def make_synthetic(dest: Path, n: int, n_shards: int = 4, seed: int = 0) -> None:
    """Noise JPEGs in webdataset layout: exercises decode/preprocess/quantize/eval, measures speed."""
    from PIL import Image

    rng = random.Random(seed)
    dest.mkdir(parents=True, exist_ok=True)
    per = math.ceil(n / n_shards)
    for s in range(n_shards):
        with tarfile.open(dest / f"imagenet1k-validation-{s:02d}.tar", "w") as tf:
            for i in range(s * per, min(n, (s + 1) * per)):
                w, h = rng.randint(200, 500), rng.randint(200, 500)
                img = Image.frombytes("RGB", (w, h), rng.randbytes(w * h * 3))
                if i % 50 == 7:
                    img = img.convert("L")  # ImageNet val holds grayscale JPEGs too
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=90)
                for ext, data in (("jpg", buf.getvalue()), ("cls", str(rng.randrange(1000)).encode())):
                    info = tarfile.TarInfo(f"synthetic_{i:08d}.{ext}")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))


def select(index: dict, spec: str) -> list[str]:
    keys, labels = index["keys"], index["labels"]
    kind, _, rest = spec.partition(":")
    if kind == "full":
        return list(keys)
    if kind == "first":
        return keys[: int(rest)]
    if kind == "stratified":  # stratified:<N>:<seed>, N/1000 images per class
        n_text, _, seed_text = rest.partition(":")
        by_class: dict[int, list[str]] = {}
        for k, lab in zip(keys, labels):
            by_class.setdefault(lab, []).append(k)
        per = int(n_text) // len(by_class)
        rng = random.Random(int(seed_text or 0))
        return [k for lab in sorted(by_class) for k in rng.sample(sorted(by_class[lab]), per)]
    raise SystemExit(f"unknown --eval {spec!r}")


# ---------------------------------------------------------------------------------------- comparison
def _hypotheses(col: str, cell: dict, unquantized_ok: bool | None, subset_n: int) -> list[str]:
    """Candidate explanations for a gap, from what this run measured -- to be tested, not asserted."""
    out = []
    parity = cell.get("parity")
    if parity and parity.get("within_band") is False:
        if col in ("FP32", "BF16"):
            out.append(
                "unquantized cell: weights, preprocessing and argmax are deterministic, so a "
                "gap here means different evaluation inputs (subset/order, decoder, resize or "
                "tokenizer versions) or checkpoint than the paper's run"
            )
        elif unquantized_ok:
            out.append(
                "FP32/BF16 match on the same inputs, so the gap is in quantization: the "
                "pinned compiler (2026) is not the revision behind the paper (v1, Sep 2025), "
                "and per-tensor scales calibrated on a handful of samples move with any "
                "quantizer change"
            )
        else:
            out.append("the unquantized cells miss too; resolve those first")
        lit, sub = cell.get("literal_lowered"), cell.get("quantized_graph")
        if lit and sub and lit["top1"] != sub["top1"]:
            out.append(
                "the lowered graph and the pre-transform quantized graph disagree: part of "
                "the gap is Voyager's lowering (bf16 re-association), not the quantizer"
            )
        gold = parity.get("delta_vs_accel_gold")
        if gold is not None and parity.get("within_accel_gold_tolerance"):
            out.append(
                "inside the accelerator regression's own +-1.0 gold tolerance: the paper "
                "and that gold table disagree with each other by up to 2 points, so a "
                "sub-point gap is within the variation Voyager itself accepts"
            )
    ev = cell.get("evalset")
    if ev and ev.get("paper_in_wilson95") is False and ev["n"] != subset_n:
        out.append(
            f"evaluation-set cell (n={ev['n']}) excludes the paper value from its 95% "
            f"interval: the paper evaluated {subset_n} samples, so its cell carries "
            f"~{100 * math.sqrt(0.25 / subset_n):.1f} pt sampling error on its own"
        )
    return out


def _refs(paper_doc: dict, model: str) -> tuple[dict, dict, float]:
    row = PAPER_ROW.get(model)
    paper = paper_doc["rows"].get(row, {}).get("top1", {}) if row else {}
    gold_doc = paper_doc.get("accelerator_regression_gold", {})
    gold = gold_doc.get("top1", {}).get(row, {}) if row else {}
    return paper, gold, gold_doc.get("source", {}).get("tolerance_points", 1.0)


def _finish_cells(cells: dict, paper: dict, subset_n: int) -> None:
    unq = [cells[c]["parity"].get("within_band") for c in ("FP32", "BF16") if c in cells]
    unquantized_ok = all(unq) if unq and None not in unq else None
    for col, c in cells.items():
        c["hypotheses"] = _hypotheses(col, c, unquantized_ok, subset_n) if paper else []


def _literal_cell(c: dict, lit: dict | None, p, g, tol) -> None:
    if lit is None:
        return
    c["literal_error"] = lit.get("error")
    stage = next((s for s in lit.get("stages", []) if s["stage"] == "lowered_graph"), None)
    if stage:
        c["literal_lowered"] = acc_block(stage["top1_correct"], stage["n"], p, g, tol)
        c["literal_lowered_preds"] = stage["preds"]


def _parity(c: dict, basis: str) -> None:
    keys = ("top1", "n", "delta_vs_paper", "within_band", "delta_vs_accel_gold", "within_accel_gold_tolerance")
    c["parity"] = {"basis": basis, **{k: c[basis][k] for k in keys if k in c[basis]}}


def compare_imagenet(model, columns, index, eval_keys, qdir, literal, paper_doc, synthetic) -> dict:
    res = json.loads((qdir / "results.json").read_text())
    preds = json.loads((qdir / "preds.json").read_text())
    paper, gold, tol = ({}, {}, 1.0) if synthetic else _refs(paper_doc, model)
    pos = {k: i for i, k in enumerate(preds["keys"])}
    labels = preds["labels"]
    subset = [pos[k] for k in index["keys"][:PAPER_SUBSET_N] if k in pos]
    evalset = [pos[k] for k in eval_keys if k in pos]
    cells = {}
    for col in columns:
        p, g = paper.get(col), gold.get(col)
        top1 = preds["top1"][col]
        c = {
            "paper_top1": p,
            "accel_gold_top1": g,
            "flags": res["columns"][col]["flags"],
            "images_per_second": res["columns"][col]["images_per_second"],
        }
        name = "quantized_graph" if col in QUANTIZED else "eager_model"
        c[name] = acc_block(sum(top1[i] == labels[i] for i in subset), len(subset), p, g, tol)
        c["evalset"] = acc_block(sum(top1[i] == labels[i] for i in evalset), len(evalset), p, g, tol)
        if "static_check" in res["columns"][col]:
            c["static_check"] = res["columns"][col]["static_check"]
        _literal_cell(c, literal.get(col), p, g, tol)
        if "literal_lowered_preds" in c:
            lp = c.pop("literal_lowered_preds")
            n = min(len(lp), len(subset))
            c["literal_vs_quantized_graph_top1_agree"] = (
                sum(lp[j] == top1[subset[j]] for j in range(n)) / n if n else None
            )
        first = next((r for r in literal.values() if r.get("stages")), None)
        if col == "BF16" and first:
            s = first["stages"][0]
            c["literal_unquantized_bf16"] = acc_block(s["top1_correct"], s["n"], p, g, tol)
        _parity(c, "literal_lowered" if "literal_lowered" in c else name)
        cells[col] = c
    _finish_cells(cells, paper, PAPER_SUBSET_N)
    return {
        "model": model,
        "task": "imagenet",
        "recipe": res["recipe"],
        "array": res["array"],
        "weights": res["columns"][columns[0]]["weights"],
        "versions": res["versions"],
        "preprocessing": res["preprocessing"],
        "calibration_keys": {c: res["columns"][c]["calibration_keys"] for c in columns},
        "wall_seconds": res["wall_seconds"],
        "cells": cells,
    }


def compare_glue(model, gdir, literal, paper_doc) -> dict:
    res = json.loads((gdir / "results.json").read_text())
    preds = json.loads((gdir / "preds.json").read_text())
    paper, gold, tol = _refs(paper_doc, model)
    labels = preds["labels"]
    cells = {}
    for col, meta in res["columns"].items():
        p, g = paper.get(col), gold.get(col)
        if meta.get("error"):
            cells[col] = {
                "paper_top1": p,
                "accel_gold_top1": g,
                "flags": meta["flags"],
                "status": f"ERROR: {meta['error'][:160]}",
            }
            continue
        top1 = preds["top1"][col]
        c = {
            "paper_top1": p,
            "accel_gold_top1": g,
            "flags": meta["flags"],
            "calibration_steps": meta["calibration_steps"],
            "evaluated": meta["evaluated"],
            "seconds": round(meta["seconds"]),
        }
        name = "quantized_graph" if col in QUANTIZED else "eager_model"
        c[name] = acc_block(sum(a == b for a, b in zip(top1, labels)), len(labels), p, g, tol)
        c["evalset"] = c[name]  # the full validation split is also the paper's protocol
        _literal_cell(c, literal.get(col), p, g, tol)
        if "literal_lowered_preds" in c:
            lp = c.pop("literal_lowered_preds")
            n = min(len(lp), len(top1))
            c["literal_vs_quantized_graph_top1_agree"] = sum(lp[j] == top1[j] for j in range(n)) / n if n else None
        _parity(c, "literal_lowered" if "literal_lowered" in c else name)
        cells[col] = c
    _finish_cells(cells, paper, len(labels))
    return {
        "model": model,
        "task": "sst2",
        "recipe": res["recipe"],
        "array": res["array"],
        "checkpoint": res["checkpoint"],
        "checkpoint_revision": res["checkpoint_revision"],
        "dataset": res["dataset"],
        "versions": res["versions"],
        "cells": cells,
    }


def table_md(rows: list[dict], eval_label: str) -> str:
    def num(v, fmt="{:.2f}"):
        return "-" if v is None else fmt.format(v)

    def yn(v):
        return "-" if v is None else ("yes" if v else "NO")

    lines = [
        f"| Model | Column | Paper | Accel gold | Ours, paper protocol (n, basis) | Δ paper | "
        f"±{PARITY_BAND} | Δ gold (±1.0) | Ours, {eval_label} [Wilson 95%] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("blocked"):
            lines.append(f"| {r['model']} | all | - | - | BLOCKED: {r['blocked']} | | | | |")
            continue
        for col, c in r["cells"].items():
            par, ev = c.get("parity") or {}, c.get("evalset")
            ev_txt = (
                "-"
                if not ev
                else (f"{ev['top1']:.2f} (n={ev['n']}) [{ev['wilson95'][0]:.2f}, {ev['wilson95'][1]:.2f}]")
            )
            ours = (
                f"{num(par.get('top1'))} ({par.get('n', '-')}, {par.get('basis', '-')})"
                if par
                else c.get("status", "-")
            )
            gold_txt = (
                f"{num(par.get('delta_vs_accel_gold'), '{:+.2f}')} {yn(par.get('within_accel_gold_tolerance'))}"
                if par.get("delta_vs_accel_gold") is not None
                else "-"
            )
            lines.append(
                f"| {r['model']} | {col} | {num(c.get('paper_top1'), '{}')} | "
                f"{num(c.get('accel_gold_top1'), '{}')} | {ours} | "
                f"{num(par.get('delta_vs_paper'), '{:+.2f}')} | {yn(par.get('within_band'))} | "
                f"{gold_txt} | {ev_txt} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------- main
def _run_literal(vpy, venv, out, tmp, model, columns, a, shard_dir=None) -> dict:
    literal = {}
    for col in columns:
        if col not in QUANTIZED:
            continue
        ldir = out / "runs" / model / f"test_codegen_{col}"
        t0 = time.time()
        args = [
            "test_codegen",
            "--model",
            model,
            "--column",
            col,
            "--out",
            str(ldir),
            "--work",
            str(tmp / f"work_{model}_{col}"),
            "--threads",
            str(a.threads),
            "--recipe",
            a.recipe,
            "--array",
            a.array,
        ]
        if shard_dir is not None:
            args += ["--shards", str(shard_dir)]
        run_worker(vpy, venv, args, ldir / "worker.log")
        rpath = ldir / "results.json"
        literal[col] = (
            json.loads(rpath.read_text()) if rpath.is_file() else {"error": f"no results; see {ldir / 'worker.log'}"}
        )
        print(
            json.dumps(
                {"model": model, "literal": col, "error": literal[col].get("error"), "seconds": round(time.time() - t0)}
            ),
            flush=True,
        )
    return literal


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="imagenet", choices=sorted(TASKS))
    ap.add_argument("--models", default=None, help="default: every model of the task")
    ap.add_argument("--columns", default="FP32,BF16,INT8,MXINT8")
    ap.add_argument(
        "--recipe",
        default="accelerator",
        choices=("accelerator", "ci"),
        help="accelerator: run_regression.py's flags (the paper's accuracy flow); ci: test/run_ci.py SCHEME_ARGS",
    )
    ap.add_argument("--array", default="16,16", help="IC,OC: MXINT8 block size and the compile")
    ap.add_argument("--eval", default="full", help="ImageNet: full | stratified:<N>:<seed> | first:<N>")
    ap.add_argument(
        "--literal",
        action="store_true",
        help="also run Voyager's test_codegen.py --evaluate as-is for quantized columns",
    )
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--check-static", type=int, default=8)
    ap.add_argument("--fetch", action="store_true", help="download ImageNet validation if missing")
    ap.add_argument("--shards", type=Path, default=None)
    ap.add_argument("--synthetic", type=int, default=0, help="ImageNet smoke on N synthetic JPEGs")
    ap.add_argument("--smoke-out", type=Path, default=None)
    ap.add_argument(
        "--attach",
        type=Path,
        action="append",
        default=[],
        help="JSON evidence to embed (e.g. a smoke summary); repeatable",
    )
    a = ap.parse_args(argv)

    task = TASKS[a.task]
    tmp = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / "dataset_accuracy"
    root, accel, vpy, venv = voyager_env(a.threads, tmp)
    models, columns = (a.models or task["models"]).split(","), a.columns.split(",")
    paper_doc = yaml.safe_load(PAPER.read_text())
    worker_common = ["--threads", str(a.threads), "--recipe", a.recipe, "--array", a.array]

    if a.synthetic:
        if a.task != "imagenet" or a.smoke_out is None:
            raise SystemExit("--synthetic is an ImageNet smoke and needs --smoke-out")
        out, prod = a.smoke_out, None
    else:
        prod = new_product(
            "dataset-accuracy",
            version=1,
            target=task["target"],
            sources=[
                str(WORKER.relative_to(REPO)),
                str(Path(__file__).resolve().relative_to(REPO)),
                str(PAPER.relative_to(REPO)),
            ],
            notes=f"Voyager Table 3 (arXiv 2509.15205) {a.task} reproduction with Voyager's own quantization code",
        )
        out = prod.path
    out.mkdir(parents=True, exist_ok=True)

    # Verify the bytes actually read (content check), not just the commit.
    pins = {"voyager_compiler": provenance.verify("voyager_compiler", reads=list(task["sources"]))}
    if a.task == "sst2" and "mobilebert" in models:
        pins["voyager_accelerator"] = provenance.verify("voyager_accelerator", reads=["run_regression.py"])
    record = {
        "experiment": "voyager_table3",
        "task": a.task,
        "paper": paper_doc["source"],
        "accel_gold": paper_doc.get("accelerator_regression_gold", {}).get("source"),
        "parity_band_top1_points": PARITY_BAND,
        "models": models,
        "columns": columns,
        "recipe": a.recipe,
        "array": a.array,
        "threads": a.threads,
    }
    rows, weights = [], {}

    if a.task == "sst2":
        record["data"] = {
            "repo": "nyu-mll/glue",
            "config": "sst2",
            "split": "validation",
            "revisions": sorted(
                p.name for p in (cache_dir("huggingface") / "hub" / "datasets--nyu-mll--glue" / "snapshots").glob("*")
            ),
        }
        record["paper_protocol"] = "the full SST-2 validation split (what test_codegen evaluates)"
        for model in models:
            gdir = out / "runs" / model / "glue"
            t0 = time.time()
            rc = run_worker(
                vpy,
                venv,
                [
                    "glue",
                    "--model",
                    model,
                    "--columns",
                    ",".join(columns),
                    "--out",
                    str(gdir),
                    "--work",
                    str(tmp / f"glue_{model}"),
                    *worker_common,
                ],
                gdir / "worker.log",
            )
            print(json.dumps({"model": model, "glue_rc": rc, "seconds": round(time.time() - t0)}), flush=True)
            if rc:
                rows.append(
                    {
                        "model": model,
                        "cells": {},
                        "blocked": _blocked_line(gdir / "worker.log") or f"worker failed; see {gdir / 'worker.log'}",
                    }
                )
                continue
            literal = _run_literal(vpy, venv, out, tmp, model, columns, a) if a.literal else {}
            rows.append(compare_glue(model, gdir, literal, paper_doc))
        record["status"] = "measured"
    else:
        if a.synthetic:
            shard_dir = a.smoke_out / "shards"
            make_synthetic(shard_dir, a.synthetic)
            data = {"status": "synthetic", "n": a.synthetic}
            index_path = a.smoke_out / "index.json"
        else:
            shard_dir = a.shards or cache_dir("imagenet-1k-wds")
            present = len(list(shard_dir.glob("imagenet1k-validation-*.tar"))) == N_VAL_SHARDS
            data = (
                fetch_shards(shard_dir, vpy, venv)
                if a.fetch
                else (
                    {"status": "present"}
                    if present
                    else {"status": "blocked", "reason": "validation shards missing; pass --fetch"}
                )
            )
            index_path = shard_dir / "index.json"
        data.update(repo=IMAGENET_REPO, revision=IMAGENET_REVISION, split="validation", shard_dir=str(shard_dir))
        record.update(
            data=data,
            eval=a.eval,
            workers=a.workers,
            batch=a.batch,
            paper_protocol=f"the first {PAPER_SUBSET_N} images of the {IMAGENET_REPO} validation stream",
        )
        if data["status"] in ("present", "downloaded", "synthetic"):
            if not index_path.is_file():
                if run_worker(
                    vpy, venv, ["index", "--shards", str(shard_dir), "--out", str(index_path)], tmp / "index.log"
                ):
                    raise SystemExit(f"index failed; see {tmp / 'index.log'}")
            index = json.loads(index_path.read_text())
            if not a.synthetic and "shard_sha256" not in index:
                index["shard_sha256"] = {s: sha256(shard_dir / s) for s in index["shards"]}
                index_path.write_text(json.dumps(index))
            data.update(n_stream=len(index["keys"]), shard_sha256=index.get("shard_sha256"))
            eval_keys = select(index, a.eval)
            wanted = sorted(set(eval_keys) | set(index["keys"][:PAPER_SUBSET_N]))
            select_path = out / "selection.json"
            select_path.write_text(json.dumps(wanted))
            record["n_eval"] = len(eval_keys)
            for model in models:
                qdir = out / "runs" / model / "quantized"
                t0 = time.time()
                rc = run_worker(
                    vpy,
                    venv,
                    [
                        "quantized",
                        "--model",
                        model,
                        "--columns",
                        ",".join(columns),
                        "--shards",
                        str(shard_dir),
                        "--index",
                        str(index_path),
                        "--select",
                        str(select_path),
                        "--out",
                        str(qdir),
                        "--workers",
                        str(a.workers),
                        "--batch",
                        str(a.batch),
                        "--check-static",
                        str(a.check_static),
                        *worker_common,
                    ],
                    qdir / "worker.log",
                )
                print(json.dumps({"model": model, "quantized_rc": rc, "seconds": round(time.time() - t0)}), flush=True)
                if rc:
                    rows.append({"model": model, "cells": {}, "blocked": f"worker failed; see {qdir / 'worker.log'}"})
                    continue
                literal = _run_literal(vpy, venv, out, tmp, model, columns, a, shard_dir) if a.literal else {}
                rows.append(
                    compare_imagenet(model, columns, index, eval_keys, qdir, literal, paper_doc, bool(a.synthetic))
                )
            record["status"] = "smoke" if a.synthetic else "measured"
        else:
            record["status"] = "blocked"
            record["blocked_reason"] = data.get("reason")
            for model in models:
                paper, gold, _ = _refs(paper_doc, model)
                rows.append(
                    {
                        "model": model,
                        "cells": {
                            col: {"paper_top1": paper.get(col), "accel_gold_top1": gold.get(col), "status": "BLOCKED"}
                            for col in columns
                        },
                    }
                )

    record["rows"] = rows
    for r in rows:
        w = r.get("weights")
        if w and w.get("file") and Path(w["file"]).is_file():
            weights[f"{r['model']}_weights"] = w["file"]
    sources = [WORKER, Path(__file__).resolve(), PAPER] + [root / s for s in task["sources"]]
    if "voyager_accelerator" in pins:
        sources.append(accel / "run_regression.py")
    record["provenance"] = provenance.record(
        pins=pins,
        sources=sources,
        artifacts=weights,
        extra={"voyager_checkout": str(root), "voyager_venv_python": str(vpy)},
    )
    record["attached"] = {str(p): json.loads(p.read_text()) for p in a.attach}

    label = "synthetic" if a.synthetic else (a.eval if a.task == "imagenet" else "full validation")
    (out / "results.json").write_text(json.dumps(record, indent=1, default=str))
    header = f"# Voyager Table 3 reproduction: {a.task} ({record['status']})\n\n"
    if record["status"] == "blocked":
        header += f"Blocked: {record.get('blocked_reason')}\n\n"
    (out / "table.md").write_text(header + table_md(rows, label))
    if prod is not None:
        shutil.copy(PAPER, out / "paper_table3.yaml")
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "manifest.yaml":
                prod.add_artifact(str(p.relative_to(out)))
        prod.write_manifest()
    print(json.dumps({"out": str(out), "status": record["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
