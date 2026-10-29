"""Seeded grades: carry what the seed run proved, check what differs, time what still matters.

A seeded campaign starts from a finished run's compiler. Most of its test cases emit exactly the
executables the seed run already proved, and re-proving them is not the focus of the run. The
seeded plan (``examples/<target>/phase1/launch/fresh_seed.py``) re-keys the seed run's passing tier
verdicts onto this export's judge (:mod:`merlin_experiments.phase1.seed_certificates`) and sets
:data:`ENV` to the seed run's id. Nothing here acts without it, so a from-scratch run grades exactly
as before.

With it set, and only while the portable tier-certificate identity is selected:

* **The screen is carried too.** The first tier the ladder executes is otherwise always paid for,
  because its adapter builds the executable the certificate key is taken from. Here the build still
  runs -- it is how the grade learns the bytes -- and the screen's verdict is looked up after the
  build and before dispatch, keyed on the executable's bytes and the screen's instrument (the
  grading path's bytes, which ``seed_certificates`` only re-keys when they decide verdicts the same
  way in both exports, or an operator review names the exact difference). A hit is carried and
  labelled; a miss executes as before. A case whose screen execution is custody for a live
  whole-model gate is never carried (:func:`screen_carry`).
* **Order follows need** (:func:`rank`): cases without proof first (their correctness is unknown),
  then cases proven by execution in this run, by largest cycle gap (they need timing), and cases
  whose verdict is carried last. The order never skips, reuses or credits a case.
* **Carried cases read as passing** with ``carried from <run>`` (:func:`label`), so the first grade
  reports the seed's proven cases as passed instead of starting from zero.

The final grade is unchanged in kind: every executable without a carried verdict for the exact
same bytes and judge runs every tier, hidden checks included, and a carried verdict stays labelled.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

#: Set by the seeded plan to the seed run's id; absent (every from-scratch run) disables everything here.
ENV = "MERLIN_SEEDED_GRADE"
#: Store subdirectory holding, per seed run, which test cases carry a verdict into seeded runs.
INDEX = "seeded-carry"
SCHEMA = "merlin.seeded_carry_index.v1"

#: :func:`rank` values, in grading order.
NEEDS_CORRECTNESS, NEEDS_TIMING, PROVEN = 0, 1, 2

_PROVEN_MEMO: dict = {}


def seed_run() -> str | None:
    """The seed run whose proof this grade carries, or ``None`` (not seeded, cache off, or not portable)."""
    from . import tier_cache

    raw = (os.environ.get(ENV, "") or "").strip()
    if not raw or tier_cache.disabled() or not tier_cache.portable():
        return None
    return raw


def screen_carry(capsule_name: str, tier: str) -> bool:
    """Whether the first executed tier of ``capsule_name`` may carry a verdict after its build."""
    if seed_run() is None:
        return False
    try:
        from merlin_experiments.phase1.feedback.rolling_model_gate import needs_original_execution
    except ImportError:
        return True
    return not needs_original_execution(capsule_name, tier)


# --------------------------------------------------------------------------- which cases the seed proved


def index_path(root: Path, run_id: str) -> Path:
    name = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:32]
    return Path(root) / INDEX / f"{name}.json"


def write_index(root: Path, run_id: str, carried: dict) -> Path:
    """Record ``{test case: [tiers]}`` carried from ``run_id`` (written by ``seed_certificates``)."""
    path = index_path(root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": SCHEMA,
        "seed_run": str(run_id),
        "carried": {name: sorted(set(tiers)) for name, tiers in sorted(carried.items())},
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def proven(run_id: str | None = None, *, root: Path | None = None) -> frozenset:
    """Test cases holding a carried verdict from the seed run. Ordering input only: never a pass."""
    from . import tier_cache

    run_id = run_id if run_id is not None else seed_run()
    root = root if root is not None else tier_cache.cache_root()
    if not run_id or root is None:
        return frozenset()
    path = index_path(root, run_id)
    try:
        stat = path.stat()
    except OSError:
        return frozenset()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _PROVEN_MEMO:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return frozenset()
        ok = isinstance(document, dict) and document.get("seed_run") == run_id
        carried = document.get("carried") if ok else None
        _PROVEN_MEMO[key] = frozenset(carried) if isinstance(carried, dict) else frozenset()
    return _PROVEN_MEMO[key]


# --------------------------------------------------------------------------- order


def fully_carried(document: dict, oracle_tiers=None) -> bool:
    """Whether every passing (oracle) tier of a capsule result was carried: no oracle proved it now.

    ``oracle_tiers`` restricts the question to the ladder's oracle tiers; in-process reference checks
    that run on every grade are then not counted. ``None`` counts every tier.
    """
    tiers = document.get("tiers") if isinstance(document, dict) else None
    passing = [
        r
        for t, r in (tiers or {}).items()
        if isinstance(r, dict) and r.get("status") == "pass" and (oracle_tiers is None or t in oracle_tiers)
    ]
    return bool(passing) and all(isinstance(r.get("carried"), dict) and r["carried"].get("carried") for r in passing)


def carried_in_full(document: dict) -> bool:
    """Whether the runner labelled a capsule result ``carried from <run>`` (see :func:`label`)."""
    reuse = document.get("tier_reuse") if isinstance(document, dict) else None
    return isinstance(reuse, dict) and isinstance(reuse.get("carried_from"), str)


def rank(outcome: dict | None, *, proven_by_seed: bool) -> int:
    """Seeded grading order for one case from its newest result in this run.

    No result yet: a case the seed proved is expected to be carried (:data:`PROVEN`), any other needs
    correctness. A failing case needs correctness. A pass proved by execution needs timing; a pass
    whose every tier was carried is proven and goes last.
    """
    if not outcome:
        return PROVEN if proven_by_seed else NEEDS_CORRECTNESS
    if outcome.get("passed") is not True:
        return NEEDS_CORRECTNESS
    return PROVEN if outcome.get("carried") is True else NEEDS_TIMING


def headroom(verdict: dict | None) -> dict:
    """``{test case: headroom cycles}`` from a verdict's development headroom (largest gap first)."""
    brief = (verdict or {}).get("feedback_brief") if isinstance(verdict, dict) else None
    priority = (brief or {}).get("performance_priority") if isinstance(brief, dict) else None
    block = (priority or {}).get("development_headroom") if isinstance(priority, dict) else None
    out = {}
    for entry in (block or {}).get("largest_headroom") or () if isinstance(block, dict) else ():
        if not isinstance(entry, dict) or not isinstance(entry.get("test_case"), str):
            continue
        value = entry.get("headroom_cycles", entry.get("cycles"))
        if type(value) in (int, float) and value >= 0:
            out[entry["test_case"]] = float(value)
    return out


def workspace_gap(workspace):
    """``gap(member)`` from the workspace's current verdict headroom, or ``None`` when not seeded/absent."""
    if seed_run() is None:
        return None
    try:
        verdict = json.loads((Path(workspace) / "qa" / "verdict.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    gaps = headroom(verdict)
    return (lambda member: gaps.get(str(member.get("name")))) if gaps else None


# --------------------------------------------------------------------------- labels


def _passing(document: dict) -> bool:
    """A pass, or a result whose only shortfall is a required tier this grade cannot reach.

    A development grade runs the functional screen only: every case that clears it reads ``incomplete``
    (its elaborated-RTL tier is unreachable here), never ``pass``. Such a case carried in full is exactly
    what the seed proved, so it is labelled too; any failed or errored tier is not."""
    if document.get("status") == "pass":
        return True
    failure = document.get("failure") if isinstance(document.get("failure"), dict) else {}
    tiers = document.get("tiers") or {}
    return (document.get("status") == "incomplete" and failure.get("plane") == "oracle_unavailable"
            and not any(isinstance(r, dict) and r.get("status") in ("fail", "error") for r in tiers.values()))


def label(document: dict, oracle_tiers=None) -> str | None:
    """``carried from <run>`` for a passing result whose oracle verdicts were all carried, else ``None``."""
    if not isinstance(document, dict) or not _passing(document) or not fully_carried(document, oracle_tiers):
        return None
    runs = []
    for record in (document.get("tiers") or {}).values():
        block = record.get("carried") if isinstance(record, dict) else None
        if not isinstance(block, dict) or not block.get("carried"):
            continue
        source = str(block.get("source") or "")
        run = source.removeprefix("carried from ").strip() if source.startswith("carried from ") else None
        run = run or block.get("earned_by_run")
        if isinstance(run, str) and run and run not in runs:
            runs.append(run)
    return "carried from " + (", ".join(runs) if runs else "an earlier certification")


def summary(results, oracle_tiers=None) -> dict:
    """Per tier, how many cases executed it and how many carried it; and how many passed fully carried."""
    executed: dict = {}
    carried: dict = {}
    fully = 0
    for document in results or ():
        if not isinstance(document, dict):
            continue
        for tier, record in (document.get("tiers") or {}).items():
            if not isinstance(record, dict) or record.get("status") in ("skipped", "unavailable", None):
                continue
            block = record.get("carried")
            bucket = carried if isinstance(block, dict) and block.get("carried") else executed
            bucket[tier] = bucket.get(tier, 0) + 1
        fully += label(document, oracle_tiers) is not None
    return {
        "executed": dict(sorted(executed.items())),
        "carried": dict(sorted(carried.items())),
        "fully_carried_cases": fully,
        "cases": sum(1 for d in results or () if isinstance(d, dict)),
    }
