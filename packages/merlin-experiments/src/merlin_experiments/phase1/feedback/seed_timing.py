"""What a seeded run already knows about performance: the seed's model cycles and per-case baseline.

A seeded launch stages two operator files in the run directory, beside the joint campaign state
(``joint_seed.stage_run_evidence``); nothing here acts without them, so a from-scratch run is unchanged:

* ``seed_timing/<elf sha256>.json``: the seed run's complete timing-model projections, keyed by the
  bytes of the executable they observed. :func:`carried` returns one for an executable with exactly
  those bytes, the same test-case declaration, the same timing selection, model, facts and epoch, so
  :func:`.joint_performance.attach` attaches it instead of observing the program again. The record
  says it was carried and from which run; the act of measurement is not repeated, nothing else
  changes meaning. ``MERLIN_SEED_TIMING=0`` turns the carry off.
* ``seed_baseline.json``: per case, the seed's numbers from its last completed grade.
  :func:`annotate` adds ``seed_cycles`` and ``vs_seed`` (this run's cycles / the seed's) to ranked
  performance entries, comparing only like with like (model cycles under the same model, estimates
  with estimates), so the author sees from the first grade what each change did against the seed.

The files live in the run directory, which the author cannot write; the author's own copy of the
baseline is ``SEED_PERFORMANCE/baseline.json`` at the workspace root.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

TIMING_DIR = "seed_timing"
BASELINE = "seed_baseline.json"
ENV = "MERLIN_SEED_TIMING"
KEYS = ("capsule_sha256", "selection_sha256", "model_sha256", "facts_sha256", "timing_epoch")
_BASELINE_MEMO: dict = {}


def _run_dir() -> Path | None:
    from .. import joint_control

    state = os.environ.get(joint_control.ENV, "").strip()
    return Path(state).parent if state else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def carried(elf: Path, *, capsule: str, key: dict, root: Path | None = None) -> dict | None:
    """The seed's ``{performance, compiler_diagnostics, seed_run}`` for exactly these bytes, else ``None``."""
    from ..author_profiles import current

    profile = current()
    if profile is not None and profile.fresh10 and not profile.seeded:
        return None
    if (os.environ.get(ENV, "") or "").strip() == "0":
        return None
    root = _run_dir() if root is None else Path(root)
    if root is None or not (root / TIMING_DIR).is_dir():
        return None
    try:
        digest = _sha256(Path(elf))
        document = json.loads((root / TIMING_DIR / f"{digest}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for record in (document or {}).get("records") or ():
        stored = (record or {}).get("key") or {}
        if (record.get("capsule") == capsule and stored.get("elf_sha256") == digest
                and all(stored.get(k) == key.get(k) for k in KEYS)):
            performance = copy.deepcopy(record.get("performance") or {})
            run = record.get("seed_run") or document.get("seed_run")
            performance["carried_from"] = {
                "run": run, "elf_sha256": digest,
                "basis": "same executable bytes, declaration, timing selection, model, facts and epoch",
            }
            performance["reason"] = (f"carried from {run}: the same executable was timed by the same model in the "
                                     "seed run; not observed again")
            return {"performance": performance, "compiler_diagnostics": copy.deepcopy(
                record.get("compiler_diagnostics")), "seed_run": run}
    return None


def baseline(root: Path | None = None) -> dict:
    """``{case: seed row}`` from the run's seed baseline, or ``{}`` (not seeded)."""
    from ..author_profiles import current

    profile = current()
    if profile is not None and profile.fresh10 and not profile.seeded:
        return {}
    root = _run_dir() if root is None else Path(root)
    if root is None:
        return {}
    path = root / BASELINE
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _BASELINE_MEMO:
        try:
            cases = json.loads(path.read_text(encoding="utf-8")).get("cases")
        except (OSError, ValueError, AttributeError):
            cases = None
        _BASELINE_MEMO.clear()
        _BASELINE_MEMO[key] = cases if isinstance(cases, dict) else {}
    return _BASELINE_MEMO[key]


def _kind(evidence) -> str | None:
    text = str(evidence or "")
    return "modeled" if text.startswith("modeled") else "estimated" if text.startswith("estimated") else None


def annotate(entries, rows=None, root: Path | None = None) -> dict | None:
    """Add ``seed_cycles``/``vs_seed`` to ranked entries; a summary for the brief, or ``None`` when not seeded.

    ``rows`` (``{case: grade row}``) supplies each entry's timing model, so model cycles are compared
    only under the model the seed's were taken with."""
    cases = baseline(root)
    if not cases:
        return None
    compared, faster, slower = [], 0, 0
    for entry in entries or ():
        if not isinstance(entry, dict):
            continue
        seed = cases.get(entry.get("test_case"))
        cycles = entry.get("cycles")
        row = (rows or {}).get(entry.get("test_case")) or {}
        performance = (row.get("performance") or {}) if isinstance(row, dict) else {}
        if isinstance(performance.get("carried_from"), dict) and type(cycles) in (int, float) and cycles > 0:
            # The very executable the seed timed: its own cycles are the seed's, exactly.
            entry.update(seed_cycles=cycles, seed_basis="the same executable the seed timed")
            seed = {"cycles": cycles}
        elif not isinstance(seed, dict) or type(seed.get("cycles")) not in (int, float) or not seed["cycles"]:
            continue
        else:
            entry["seed_cycles"] = seed["cycles"]
            entry["seed_basis"] = "the seed's last completed grade" + (
                " (an earlier seed version)" if seed.get("currency") == "older_version" else "")
        same_kind = entry["seed_basis"].startswith("the same") or (
            _kind(entry.get("evidence")) == _kind(seed.get("evidence")))
        model = performance.get("model_sha256")
        same_model = not seed.get("model_sha256") or not model or seed["model_sha256"] == model
        if type(cycles) in (int, float) and cycles > 0 and same_kind and same_model:
            ratio = round(cycles / seed["cycles"], 4)
            entry["vs_seed"] = ratio
            compared.append(ratio)
            faster += ratio < 0.995
            slower += ratio > 1.005
    if not compared:
        return {"cases_in_seed_baseline": len(cases), "compared": 0}
    from statistics import geometric_mean

    return {"cases_in_seed_baseline": len(cases), "compared": len(compared), "faster_than_seed": faster,
            "slower_than_seed": slower, "geomean_vs_seed": round(geometric_mean(compared), 4),
            "reading": "vs_seed = this run's cycles / the seed's for the same case (below 1 is faster)"}
