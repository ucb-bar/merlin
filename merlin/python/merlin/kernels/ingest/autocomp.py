"""Ingest Autocomp-generated kernels (Gemmini target).

Autocomp writes a flat directory of hash-named C files plus a ``manifest.jsonl`` with one
JSON object per kernel ``{source_path, experiment, score, code_hash, dest_path}``. The
manifest carries no shape/dtype, so we parse the C entry signature
``void test(<dtype> A[..][..], <dtype> B[..][..], <dtype> C[..][..])`` to recover op/shape/
dtype. The Autocomp ``score`` is recorded in ``meta`` only (provenance/tie-break) and is NOT
treated as a correctness signal.

The signature is read with a small C declarator scanner rather than a pattern: the entry point is
located by token, its parameter list split, and each ``<type> <name><dims>`` parameter walked
field by field. A parameter the scanner cannot read is not skipped silently — it is reported by
:func:`parse_signature`, which returns the unreadable text alongside the shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel, normalize_dtype

_ENTRY_RETURN, _ENTRY_NAME = "void", "test"

#: Parameter names that make a signature a convolution regardless of rank.
_CONV_NAMES = frozenset({"inp", "input", "weights", "weight", "output"})


def _word_run(text: str, start: int) -> int:
    """End of the identifier run at ``start``. ``str.isalnum()`` plus ``_`` is the old ``\\w``:
    Unicode-aware, so a non-ASCII type or parameter name still reads as one word."""
    i = start
    while i < len(text) and (text[i] == "_" or text[i].isalnum()):
        i += 1
    return i


def _space_run(text: str, start: int) -> int:
    """End of the whitespace run at ``start`` (the old ``\\s*``; ``\\s`` spans newlines too)."""
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _entry_params(text: str) -> str | None:
    """The parameter-list text of the ``void test(...)`` entry point, else ``None``.

    Deliberately keeps the old accept set, which has NO word boundary before ``void``: ``avoid
    test(...)`` is read as an entry point, as it always was. The list ends at the first ``)``, so a
    function-pointer parameter still truncates it — same as before, and the truncated remainder
    surfaces as an unreadable parameter rather than as a missing one.
    """
    at = text.find(_ENTRY_RETURN)
    while at != -1:
        i = at + len(_ENTRY_RETURN)
        j = _space_run(text, i)
        if j > i and text.startswith(_ENTRY_NAME, j):
            k = _space_run(text, j + len(_ENTRY_NAME))
            close = text.find(")", k + 1) if text.startswith("(", k) else -1
            if close != -1:
                return text[k + 1:close]
        at = text.find(_ENTRY_RETURN, at + 1)  # an unclosed list is not the entry point; keep looking
    return None


def _param_at(params: str, start: int) -> tuple[tuple[str, str, list[int]], int] | None:
    """One ``<type> <name><dims>`` parameter starting at ``start`` -> ``((type, name, dims), end)``.

    Every field is required: a word, whitespace, a word, optional whitespace, then one or more
    ``[<decimal>]`` dimensions. Qualifiers fall out for free — ``const int8_t A[8][8]`` starts a
    failed read at ``const`` and a successful one at ``int8_t``, which is what the old scan did.
    """
    tname = _word_run(params, start)
    if tname == start:
        return None
    i = _space_run(params, tname)
    if i == tname:                                  # the separator between type and name
        return None
    pname = _word_run(params, i)
    if pname == i:
        return None
    j = _space_run(params, pname)
    dims: list[int] = []
    # DELIBERATELY WIDER than the pattern this replaces: that one allowed whitespace inside a
    # dimension (`A[ 8 ]`) but not BETWEEN two (`A[8] [4]`), so a legal C declarator read as rank 1
    # and the operand was mis-measured with no sign that anything was missed. Both are read here.
    while params.startswith("[", _space_run(params, j)):
        j = _space_run(params, j)
        a = _space_run(params, j + 1)
        b = a
        while b < len(params) and params[b].isdecimal():
            b += 1
        c = _space_run(params, b)
        if b == a or not params.startswith("]", c):
            break                                   # not a constant dimension: stop the run
        dims.append(int(params[a:b]))
        j = c + 1
    if not dims:
        return None
    return (params[start:tname], params[i:pname], dims), j


def parse_signature(text: str) -> tuple[str, str, dict[str, object], tuple[str, ...]]:
    """``(op, dtype, shape, unreadable)`` for the ``void test(...)`` entry point.

    ``unreadable`` holds the parameter-list fragments the scanner could not read as a dimensioned
    parameter (a pointer parameter, a macro'd type, a truncated list). They are REPORTED rather than
    dropped: a signature this scanner half-understands must not be mistaken for one it read whole.
    """
    params = _entry_params(text)
    if params is None:
        return "unknown", "unknown", {}, ()
    parsed: list[tuple[str, str, list[int]]] = []
    covered: list[tuple[int, int]] = []
    i = 0
    while i < len(params):
        got = _param_at(params, i)
        if got is None:                             # not a parameter here: step one char and retry
            i += 1
            continue
        param, end = got
        parsed.append(param)
        covered.append((i, end))
        i = end
    # Report by comma-separated element, so a leading qualifier (`const`) is not mistaken for an
    # unread parameter: an element is unreadable only when NO parameter was read inside it.
    skipped: list[str] = []
    at = 0
    for element in params.split(","):
        lo, hi = at, at + len(element)
        at = hi + 1
        if element.strip() and not any(lo <= a and b <= hi for a, b in covered):
            skipped.append(element.strip())
    if not parsed:
        return "unknown", "unknown", {}, tuple(skipped)
    dtype = normalize_dtype(parsed[0][0])
    names = {name.lower() for _t, name, _d in parsed}
    dims = {name: list(d) for _t, name, d in parsed}
    # Convolution: 4-D tensors or conv-flavored names.
    is_conv = any(len(d) >= 4 for d in dims.values()) or bool(names & _CONV_NAMES)
    if is_conv:
        return "conv", dtype, {k: v for k, v in dims.items()}, tuple(skipped)
    # Matmul: three 2-D operands A[M][K], B[K][N], C[M][N].
    twod = [(n, dims[n]) for _t, n, _d in parsed if len(dims[n]) == 2]
    if len(twod) >= 3:
        (_a_n, a), (_b_n, b), (_c_n, _c) = twod[0], twod[1], twod[2]
        shape: dict[str, object] = {"M": a[0], "K": a[1], "N": b[1]}
    else:
        shape = {k: v for k, v in dims.items()}
    return "matmul", dtype, shape, tuple(skipped)


def _parse_signature(text: str) -> tuple[str, str, dict[str, object]]:
    """Return (op, dtype, shape) parsed from the ``void test(...)`` signature."""
    op, dtype, shape, _unreadable = parse_signature(text)
    return op, dtype, shape


_KERNEL_STEM, _KERNEL_EXT = "kernel_", ".c"
_HEX = frozenset("0123456789abcdef")


def kernel_hash(name: str) -> str | None:
    """The hex hash of a ``kernel_<hash>.c`` path, else ``None``.

    Accepts the same set the old pattern did: the name must END in ``.c`` and the hash must be a
    non-empty run of LOWERCASE hex (``kernel_ABC.c`` is not one). The leftmost ``kernel_`` whose
    remainder is all hex wins, so ``kernel_kernel_abc.c`` yields ``abc`` — as before.
    """
    if not name.endswith(_KERNEL_EXT):
        return None
    stem = name[:-len(_KERNEL_EXT)]
    at = stem.find(_KERNEL_STEM)
    while at != -1:
        digits = stem[at + len(_KERNEL_STEM):]
        if digits and all(ch in _HEX for ch in digits):
            return digits
        at = stem.find(_KERNEL_STEM, at + 1)
    return None


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
            dest = kernel_hash(entry.get("dest_path", "") or "")
            key = h[:12] or (dest[:12] if dest else "")
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
        op, dtype, shape, unreadable = parse_signature(text)
        h = kernel_hash(path.name)
        meta: dict = dict(index.get(h[:12], {})) if h else {}
        if unreadable:
            # A parameter the scanner could not read leaves the shape PARTIAL. Recording it keeps
            # the record honest instead of letting a half-read signature pass as a whole one.
            meta["unreadable_params"] = list(unreadable)
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
