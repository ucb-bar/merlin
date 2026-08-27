"""Ingest Autocomp-generated kernels (Gemmini target).

Autocomp writes a flat directory of hash-named C files plus a ``manifest.jsonl`` with one
JSON object per kernel ``{source_path, experiment, score, code_hash, dest_path}``. The
manifest carries no shape/dtype, so we parse the C entry signature
``void test(<dtype> A[..][..], <dtype> B[..][..], <dtype> C[..][..])`` to recover op/shape/
dtype. The Autocomp ``score`` is recorded in ``meta`` only (provenance/tie-break) and is NOT
treated as a correctness signal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel, normalize_dtype

# Matches the test() entry point and captures its parameter list.
_SIG_RE = re.compile(r"void\s+test\s*\(([^)]*)\)", re.DOTALL)
# One parameter: <type> <name> <dims like [3][3][128]>
_PARAM_RE = re.compile(r"(\w+)\s+(\w+)\s*((?:\[\s*\d+\s*\])+)")
_DIM_RE = re.compile(r"\[\s*(\d+)\s*\]")


def _parse_signature(text: str) -> tuple[str, str, dict[str, object]]:
    """Return (op, dtype, shape) parsed from the ``void test(...)`` signature."""
    sig = _SIG_RE.search(text)
    if not sig:
        return "unknown", "unknown", {}
    params = _PARAM_RE.findall(sig.group(1))
    if not params:
        return "unknown", "unknown", {}
    dtype = normalize_dtype(params[0][0])
    names = {name.lower() for _t, name, _d in params}
    dims = {name: [int(d) for d in _DIM_RE.findall(d)] for _t, name, d in params}
    # Convolution: 4-D tensors or conv-flavored names.
    is_conv = any(len(d) >= 4 for d in dims.values()) or bool(
        names & {"inp", "input", "weights", "weight", "output"}
    )
    if is_conv:
        return "conv", dtype, {k: v for k, v in dims.items()}
    # Matmul: three 2-D operands A[M][K], B[K][N], C[M][N].
    twod = [(n, d) for _t, n, _ in params for n, d in [(n, dims[n])] if len(d) == 2]
    shape: dict[str, object] = {}
    if len(twod) >= 3:
        (a_n, a), (b_n, b), (c_n, c) = twod[0], twod[1], twod[2]
        shape = {"M": a[0], "K": a[1], "N": b[1]}
    else:
        shape = {k: v for k, v in dims.items()}
    return "matmul", dtype, shape


_HASH_RE = re.compile(r"kernel_([0-9a-f]+)\.c$")


def _manifest_index(manifest: Path) -> dict[str, dict]:
    """Index ``manifest.jsonl`` by the 12-char hash prefix used in kernel filenames."""
    index: dict[str, dict] = {}
    if not manifest.is_file():
        return index
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = entry.get("code_hash") or ""
            key = h[:12] or _HASH_RE.search(entry.get("dest_path", "") or "")
            if isinstance(key, re.Match):
                key = key.group(1)[:12]
            if key:
                index[key] = {
                    "score": entry.get("score"),
                    "experiment": entry.get("experiment"),
                    "code_hash": entry.get("code_hash"),
                }
    return index


#: Ledger ``outcome`` -> whether the attempt was kept. Values not listed here are recorded verbatim and
#: counted as attempted-but-unclassified, never quietly folded into "failed": an outcome vocabulary that
#: grows without this map noticing would silently shrink the denominator of the base rate below.
_LEDGER_KEPT = {"improved": True, "regressed": False, "compile_error": False,
                "incorrect": False, "no_change": False, "correct_no_gain": False}


def ledger_rows(repo: str) -> "Iterator[dict]":
    """Every recorded transform ATTEMPT, from ``output/transform_ledger.jsonl``.

    The ingester above reads the kernels a search KEPT. This reads what it TRIED, which is the only
    record of what does not work — and a mining loop that sees only winners over-proposes. Measured on
    this ledger: 1509 attempts, of which a large majority never compiled, a further group regressed, and
    a minority improved. That base rate is the prior a proposal loop needs; without it the loop keeps
    re-proposing transforms somebody already refuted.

    Yields raw rows; :func:`ledger_search_steps` maps them onto the search record.
    """
    import json

    path = Path(repo) / "output" / "transform_ledger.jsonl"
    if not path.is_file():
        path = Path(repo) / "transform_ledger.jsonl"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue                 # a malformed line is skipped, and counted by ledger_summary
        if isinstance(row, dict):
            yield row


def ledger_summary(repo: str) -> dict:
    """Outcome counts + the base rate a proposal loop should be calibrated against.

    Reports ``unclassified`` separately rather than lumping unknown outcomes into failures, so a new
    outcome value shows up as a gap in this map instead of quietly changing the rate.
    """
    counts: dict[str, int] = {}
    for row in ledger_rows(repo):
        counts[str(row.get("outcome") or "unknown")] = counts.get(
            str(row.get("outcome") or "unknown"), 0) + 1
    total = sum(counts.values())
    kept = sum(n for o, n in counts.items() if _LEDGER_KEPT.get(o) is True)
    unclassified = sum(n for o, n in counts.items() if o not in _LEDGER_KEPT)
    return {"total": total, "outcomes": dict(sorted(counts.items())),
            "improved": kept, "unclassified": unclassified,
            "improvement_rate": (kept / total) if total else None}


def ledger_search_steps(repo: str, *, target: str) -> "Iterator[object]":
    """Ledger rows as :class:`kernels.search_step.SearchStep` records.

    The mapping is close to 1:1 and the differences are the honest part. ``achieved`` is False for every
    row: the ledger records an outcome, not an intended-vs-emitted audit, so claiming the promise was
    kept would fabricate the one check the search discipline rests on. ``speedup`` is credited only when
    the attempt was CORRECT, matching the fail-closed rule that no speedup counts for a fork that broke
    numerics.
    """
    from ..search_step import SearchStep

    for row in ledger_rows(repo):
        outcome = str(row.get("outcome") or "unknown")
        correct = bool(row.get("correct"))
        yield SearchStep(
            axis=f"autocomp:{row.get('strategy_num') or 'S?'}",
            category=None,               # the ledger has no CCA axis, so it has no category
            action_class="HEURISTIC",    # an LLM-proposed source transform, not a typed compiler lever
            target_seam=f"autocomp:{target}:{row.get('run') or '?'}",
            intended_facet={},
            achieved=False,              # no intended-vs-emitted audit exists for these rows
            residual=[],
            correctness_ok=correct,
            speedup=(row.get("speedup") if correct else None),
            rationale=f"[{outcome}] {str(row.get('strategy') or '')[:200]}",
        )


def ingest_autocomp(repo: str, target: str, limit: int | None = None) -> Iterator[NormalizedKernel]:
    """Yield NormalizedKernels for Autocomp kernels under ``repo/kernels/``.

    Globs the kernel directory directly (the manifest's ``dest_path`` values are stale
    absolute paths), skipping the ~1700 0-byte dedup placeholders, and joins manifest
    metadata (score/experiment) by the hash embedded in each filename.
    """
    root = Path(repo)
    index = _manifest_index(root / "manifest.jsonl")
    count = 0
    for path in sorted((root / "kernels").glob("kernel_*.c")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip() or "void test" not in text:
            continue  # empty placeholder or no entry point
        op, dtype, shape = _parse_signature(text)
        m = _HASH_RE.search(path.name)
        meta = dict(index.get(m.group(1)[:12], {})) if m else {}
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        yield NormalizedKernel(
            source="autocomp", target=target, path=rel, op=op, dtype=dtype,
            shape=shape, raw_text=text, meta=meta,
        )
        count += 1
        if limit is not None and count >= limit:
            return
