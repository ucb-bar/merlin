"""Per-case cycle targets from reference compilers the operator measured, next to the author's own cycles.

The headroom ranking compares a case's cycles with its structural roofline, a bound no real program
reaches; a run whose author closed most of the gap on small cases still did not know whether its large
cases were near what a human-built compiler or a vendor software library achieves on the same shape.
The operator measures those references once, on the same engine and harness, and stages the numbers
read-only beside the corpus: a ``merlin.performance_baselines.v1`` file holding, per corpus case, the
reference's cycles with the engine, measurement window, source label, compiler revision and
provenance. It holds only numbers and labels -- no code, schedules or command streams -- and
:func:`validate` refuses anything else.

For each case with a baseline this module adds ``reference_baselines`` (cycles per source label and,
when the author's evidence was measured on the SAME engine, the ratio author ÷ baseline; never a
measured cycle against a model estimate, never gSIM against FireSim) and a short summary: per label
the geometric mean of those ratios over the compared cases and the worst case. Advisory only: it
changes no grade, gate, ranking or threshold. No target name, no ``re``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = "merlin.performance_baselines.v1"
#: Readonly public-reference directory in the author workspace, and the file's path inside it.
REFERENCE_ROOT = "reference-kernels"
#: A positively reviewed (v2) reference manifest stages each file at ``<origin name>/<destination>``,
#: and an origin name is an identifier: the baselines origin is ``performance_baselines``.
STAGED_ORIGIN, STAGED_DESTINATION = "performance_baselines", "baselines.json"
STAGED = STAGED_ORIGIN + "/" + STAGED_DESTINATION
#: Where a historical (v1) reference manifest staged the file; still read in older workspaces.
LEGACY_STAGED = "performance-baselines/baselines.json"
STAGED_PATHS = (STAGED, LEGACY_STAGED)
REFERENCE_MANIFEST = "reference-manifest.json"
#: Engines a baseline may name, with their display names.
ENGINES = {"gsim": "gSIM", "firesim": "FireSim"}
#: Ratios kept per label in a summary's ``worst``.
WORST = 1
_MAX_BYTES = 4 << 20
_MAX_TEXT = 400
_TOP_KEYS = {
    "schema",
    "advisory_only",
    "generated_at",
    "note",
    "corpus",
    "sources",
    "excluded_sources",
    "instruments",
    "baselines",
    "unmapped",
}
_SOURCE_KEYS = {"display", "kind", "revision", "description"}
_INSTRUMENT_KEYS = {"engine", "emulator_sha256", "window", "harness", "memory_model"}
_ENTRY_KEYS = {
    "case",
    "interface_sha256",
    "shape_key",
    "source",
    "engine",
    "cycles",
    "instrument",
    "window",
    "revision",
    "useful_macs",
    "provenance",
}
_PROVENANCE_KEYS = {"study", "study_sha256", "row", "elf_sha256", "kernel_o_sha256", "checksum", "measured_at"}
_UNMAPPED_KEYS = {"row", "source", "reason"}
NOTE = (
    "Cycles of reference compilers the operator measured on the same engine and harness: targets to "
    "reach with general compiler mechanisms, not code to copy. ratio = your cycles / baseline cycles "
    "(above 1 is slower), only for the same engine; measurement windows may differ slightly."
)


def _text(value, where: str) -> str:
    if type(value) is not str or len(value) > _MAX_TEXT or "\n" in value or "\0" in value:
        raise ValueError(f"{where}: a bounded one-line label is required")
    return value


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _keys(document, allowed: set, where: str) -> dict:
    if not isinstance(document, dict) or not set(document) <= allowed:
        extra = sorted(set(document) - allowed) if isinstance(document, dict) else type(document).__name__
        raise ValueError(f"{where}: unexpected fields {extra}")
    return document


def _labels(document, allowed: set, where: str) -> None:
    for key, value in _keys(document, allowed, where).items():
        if value is not None and not (type(value) in (int, float) and math.isfinite(value)):
            _text(value, f"{where}.{key}")


def validate(document) -> dict:
    """``document`` when it is a well-formed baselines file of numbers and one-line labels; raises otherwise."""
    _keys(document, _TOP_KEYS, "baselines")
    if document.get("schema") != SCHEMA or document.get("advisory_only") is not True:
        raise ValueError(f"baselines: schema {SCHEMA} with advisory_only true is required")
    for key in ("generated_at", "note"):
        if key in document:
            _text(document[key], key)
    _labels(document.get("corpus") or {}, {"manifest_sha256", "release"}, "corpus")
    sources = document.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("baselines: at least one declared source label is required")
    for label, source in sources.items():
        _text(label, "source label")
        _labels(source, _SOURCE_KEYS, f"sources.{label}")
    for item in document.get("excluded_sources") or ():
        _text(item, "excluded_sources")
    instruments = document.get("instruments") or {}
    for name, instrument in instruments.items() if isinstance(instruments, dict) else ():
        _labels(instrument, _INSTRUMENT_KEYS, f"instruments.{name}")
        if instrument.get("engine") not in ENGINES:
            raise ValueError(f"instruments.{name}: engine must be one of {sorted(ENGINES)}")
    rows = document.get("baselines")
    if not isinstance(rows, list):
        raise ValueError("baselines: a list of baseline rows is required")
    for index, row in enumerate(rows):
        where = f"baselines[{index}]"
        _keys(row, _ENTRY_KEYS, where)
        _labels({k: v for k, v in row.items() if k != "provenance"}, _ENTRY_KEYS, where)
        _labels(row.get("provenance") or {}, _PROVENANCE_KEYS, f"{where}.provenance")
        if not isinstance(row.get("case"), str) or row.get("source") not in sources:
            raise ValueError(f"{where}: a case id and a declared source label are required")
        if row.get("engine") not in ENGINES or not _positive(row.get("cycles")):
            raise ValueError(f"{where}: an engine in {sorted(ENGINES)} and positive cycles are required")
        if row.get("instrument") is not None and row["instrument"] not in instruments:
            raise ValueError(f"{where}: undeclared instrument {row['instrument']!r}")
    for index, row in enumerate(document.get("unmapped") or ()):
        _labels(row, _UNMAPPED_KEYS, f"unmapped[{index}]")
    return document


def staged_file(root: Path) -> Path | None:
    """The staged baselines file below a reference root (current path first), else None."""
    return next((Path(root) / name for name in STAGED_PATHS if (Path(root) / name).is_file()), None)


def _pinned(root: Path, payload: bytes, relative: str = STAGED) -> bool:
    """False only when the staged reference manifest pins this file to other bytes."""
    try:
        manifest = json.loads((root / REFERENCE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    members = manifest.get("members") if isinstance(manifest, dict) else None
    for member in members if isinstance(members, list) else ():
        if isinstance(member, dict) and member.get("path") == relative:
            return member.get("sha256") == hashlib.sha256(payload).hexdigest()
    # A positively reviewed (v2) manifest stages each file at <origin name>/<destination>.
    origins = manifest.get("origins") if isinstance(manifest, dict) else None
    for origin in origins if isinstance(origins, list) else ():
        files = origin.get("files") if isinstance(origin, dict) else None
        for member in files if isinstance(files, list) else ():
            if isinstance(member, dict) and f"{origin.get('name')}/{member.get('destination')}" == relative:
                return member.get("sha256") == hashlib.sha256(payload).hexdigest()
    return True


def load(workspace) -> dict | None:
    """The staged baselines document of ``workspace`` (validated, digest-checked), else None; never raises."""
    from ..author_profiles import current

    profile = current()
    if profile is not None and profile.fresh10:
        # The removal study supplies no reference performance targets in any
        # arm, independently of whether raw source/ELF references are selected.
        return None
    root = Path(workspace if workspace is not None else Path.cwd()) / REFERENCE_ROOT
    try:
        path = staged_file(root)
        if path is None or path.is_symlink() or path.stat().st_size > _MAX_BYTES:
            return None
        payload = path.read_bytes()
        if not _pinned(root, payload, path.relative_to(root).as_posix()):
            return None
        return validate(json.loads(payload))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def index(document) -> dict:
    """``{case: [baseline row]}`` of a validated document (empty for None)."""
    found: dict = {}
    for row in (document or {}).get("baselines") or ():
        found.setdefault(row["case"], []).append(row)
    return found


def evidence_engine(evidence) -> str | None:
    """The engine a headroom/brief evidence label was measured on; None for a model estimate."""
    text = str(evidence or "")
    if not text.startswith("measured"):
        return None
    lowered = text.lower()
    return next((engine for engine in ENGINES if engine in lowered), None)


def compare(cycles, evidence, rows) -> list[dict]:
    """One comparison per baseline row: its cycles and, on the same engine only, author ÷ baseline."""
    engine = evidence_engine(evidence)
    out = []
    for row in rows or ():
        item = {"source": row["source"], "engine": row["engine"], "baseline_cycles": row["cycles"]}
        if engine is None:
            item["ratio"] = None
            item["not_compared"] = "your cycles are a model estimate; baselines are measured"
        elif engine != row["engine"]:
            item["ratio"] = None
            item["not_compared"] = f"your cycles are {ENGINES[engine]}, this baseline is {ENGINES[row['engine']]}"
        elif _positive(cycles):
            item["ratio"] = round(cycles / row["cycles"], 3)
        out.append(item)
    return out


def annotate(entries, baselines: dict) -> int:
    """Add ``reference_baselines`` to each entry that has one; returns how many entries got one."""
    count = 0
    for entry in entries or ():
        rows = baselines.get(entry.get("test_case")) if baselines else None
        if rows:
            entry["reference_baselines"] = compare(entry.get("cycles"), entry.get("evidence"), rows)
            count += 1
    return count


def _display(document, label: str) -> str:
    source = ((document or {}).get("sources") or {}).get(label) or {}
    return source.get("display") or label


def summary(entries, document) -> dict | None:
    """Per source label: the geometric-mean ratio over compared cases, the worst case and the counts."""
    with_baseline = [e for e in entries or () if e.get("reference_baselines")]
    if not with_baseline:
        return None
    labels: dict = {}
    for entry in with_baseline:
        for item in entry["reference_baselines"]:
            lane = labels.setdefault(
                item["source"],
                {"display": _display(document, item["source"]), "ratios": [], "not_compared": 0, "engines": set()},
            )
            if item.get("ratio"):
                lane["ratios"].append((item["ratio"], str(entry.get("test_case"))))
                lane["engines"].add(item["engine"])
            else:
                lane["not_compared"] += 1
    by_label = {}
    for label, lane in sorted(labels.items()):
        ratios = lane["ratios"]
        row = {"display": lane["display"], "cases_compared": len(ratios), "not_compared": lane["not_compared"]}
        if ratios:
            row["engine"] = "/".join(ENGINES[e] for e in sorted(lane["engines"]))
            row["geomean_ratio"] = round(math.exp(sum(math.log(r) for r, _ in ratios) / len(ratios)), 3)
            worst = max(ratios)
            row["worst"] = {"test_case": worst[1], "ratio": worst[0]}
        by_label[label] = row
    return {
        "schema": SCHEMA,
        "advisory_only": True,
        "cases_with_baseline": len(with_baseline),
        "by_label": by_label,
        "note": NOTE,
    }


def times(value) -> str:
    return f"×{value:.2f}" if value < 10 else f"×{value:.1f}"


def headline_line(block) -> str | None:
    """``vs reference compilers on N large cases: geomean ×1.27 slower than ...; worst: <case> ×4.1``."""
    if not isinstance(block, dict) or not block.get("by_label"):
        return None
    parts, worst = [], None
    for row in block["by_label"].values():
        if not row.get("cases_compared"):
            continue
        ratio = row["geomean_ratio"]
        relation = "slower than" if ratio > 1 else "on par with" if ratio == 1 else "faster than"
        shown = ratio if ratio >= 1 else 1 / ratio
        compared = f"{row['engine']}, {row['cases_compared']} case{'s' if row['cases_compared'] != 1 else ''}"
        parts.append(f"geomean {times(shown)} {relation} {row['display']} ({compared})")
        candidate = (row["worst"]["ratio"], row["worst"]["test_case"], row["display"])
        worst = candidate if worst is None or candidate > worst else worst
    count = block.get("cases_with_baseline")
    if not parts:
        return (
            f"vs reference compilers on {count} large case(s): no same-engine comparison yet "
            "(baselines are measured; these cases have only model estimates or another engine)"
        )
    text = f"vs reference compilers on {count} large case(s): " + "; ".join(parts)
    if worst is not None:
        text += f"; worst: `{worst[1]}` {times(worst[0])} vs {worst[2]}"
    return text


def compact(items) -> dict:
    """``{label: ratio, else baseline cycles}`` of one entry's comparisons (a same-engine ratio wins)."""
    out: dict = {}
    for item in items or ():
        if item.get("ratio"):
            out[item["source"]] = item["ratio"]
        elif item["source"] not in out:
            out[item["source"]] = f"{item['baseline_cycles']} {ENGINES[item['engine']]} cycles (not compared)"
    return out


GUIDANCE = (
    "Reference compiler baselines. `{root}/{path}` (read-only, pinned in `{root}/{manifest}`) lists, for "
    "{cases} large test case(s), the cycles that {sources} took on the same shapes, measured by the operator "
    "on {engines} with the same harness. Each verdict's `performance_headline`, the `development_headroom` "
    "entries and every self-check `performance_brief` repeat them as `reference_baselines` with the ratio "
    "of your cycles to theirs (same engine only; above 1 is slower). Treat them as targets your compiler "
    "should reach or beat on every shape through general mechanisms -- tiling, data movement, overlap, "
    "configuration hoisting -- that derive their choices from the input IR and the target's capabilities. "
    "The file holds only cycle counts and labels; it contains no code or schedules, and no per-shape "
    "special case is expected or rewarded. The numbers are advisory and change no grade."
)


def guidance(document) -> str:
    """The operator's author-guidance paragraph for ``document`` (a template filled from the file itself)."""
    rows = (document or {}).get("baselines") or ()
    names = [_display(document, label) for label in sorted({row["source"] for row in rows})]
    sources = " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names) or "none"
    engines = "/".join(ENGINES[e] for e in sorted({row["engine"] for row in rows})) or "no engine"
    return GUIDANCE.format(
        root=REFERENCE_ROOT,
        path=STAGED,
        manifest=REFERENCE_MANIFEST,
        cases=len({row["case"] for row in rows}),
        sources=sources,
        engines=engines,
    )
