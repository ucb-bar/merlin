"""Bounded, cached, read-only record access shared by every console source.

A :class:`Reader` belongs to one experiment. It caches each parsed document by its stat identity, so a
stored run is parsed once and a live run only re-reads what changed. Every refresh has a byte budget;
exhausting it yields ``None`` plus a notice rather than a partial parse. Files are opened read-only and
never locked, renamed or written.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from merlin.experiments.tracking.console import SNAPSHOT_SCHEMA
from merlin.experiments.tracking.readers.records import JsonlTail

# ---- io ----------------------------------------------------------------------------------------

_HEX = frozenset("0123456789abcdef")


def obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def rows(value: Any, limit: int = 4096) -> list:
    if isinstance(value, dict):
        value = list(value.values())
    return [row for row in value[:limit] if isinstance(row, dict)] if isinstance(value, list) else []


def num(value: Any) -> float | int | None:
    """A finite recorded number, else None (booleans are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) else None


def positive(value: Any) -> float | int | None:
    value = num(value)
    return value if value is not None and value > 0 else None


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def clean(value: Any, limit: int = 200) -> str:
    """Record text that cannot carry terminal controls, newlines or unbounded length."""
    if value is None:
        return ""
    return "".join(c if c.isprintable() else " " for c in str(value))[:limit]


def epoch(value: Any) -> float | None:
    """A recorded time as epoch seconds.

    Accepts seconds, milliseconds and nanoseconds (told apart by magnitude), ``YYYYmmddTHHMMSSZ`` and
    ISO-8601 strings; anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        if not math.isfinite(value) or value <= 0:
            return None
        if value > 1e17:
            return value / 1e9
        if value > 1e14:
            return value / 1e6
        if value > 1e11:
            return value / 1e3
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def mtime(path: Path | str | None) -> float | None:
    if path is None:
        return None
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def is_file(path: Path | str) -> bool:
    """Like ``Path.is_file`` but false, never raising, for unreadable paths."""
    try:
        return os.path.isfile(path)
    except (OSError, ValueError):
        return False


def is_dir(path: Path | str) -> bool:
    try:
        return os.path.isdir(path)
    except (OSError, ValueError):
        return False


def bounded_glob(root: Path, pattern: str, limit: int) -> list[Path]:
    """At most ``limit`` matches, without sorting a whole directory first."""
    out: list[Path] = []
    try:
        for path in Path(root).glob(pattern):
            out.append(path)
            if len(out) >= limit:
                break
    except OSError:
        pass
    return out


def scan_dir(root: Path, limit: int) -> Iterator[os.DirEntry]:
    """Entries of one directory, bounded, silently empty when it cannot be listed."""
    try:
        with os.scandir(root) as entries:
            for count, entry in enumerate(entries):
                if count >= limit:
                    return
                yield entry
    except OSError:
        return


class Reader:
    """One experiment's document cache with a per-refresh byte budget.

    ``budget=None`` is unbounded (a one-shot dump reads everything once). ``reserve`` bytes of each
    refresh are held back from ordinary reads and spent only while :attr:`priority` is set, so the
    panels read last are not starved by the bulk scans read before them. Unchanged documents (same
    stat identity) come from the cache and cost nothing.
    """

    #: What :meth:`extract` returns for a record this refresh could not read (budget, racing write).
    WAIT = object()

    def __init__(self, *, budget: int | None = 128 << 20, reserve: int = 0) -> None:
        self.budget = math.inf if budget is None else budget
        self.reserve = min(reserve, self.budget)
        self.remaining = self.budget
        self.priority = False
        self.notices: set[str] = set()
        self._documents: dict[str, tuple[tuple, Any]] = {}
        self._detail_digests: dict[str, tuple[tuple, str]] = {}
        self._result_indexes: dict[str, tuple[dict, dict]] = {}
        self._extracts: dict[tuple[str, str], tuple[tuple, Any]] = {}
        self._tails: dict[str, JsonlTail] = {}

    def begin(self) -> None:
        """Start a refresh: restore the byte budget and clear its notices."""
        self.remaining = self.budget
        self.notices = set()

    def available(self) -> float:
        """Bytes this read may still spend: all that remains for a priority read, less the reserve otherwise."""
        return self.remaining if self.priority else max(0, self.remaining - self.reserve)

    @staticmethod
    def identity(path: Path) -> tuple | None:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)

    def _load(self, path: Path, limit: int, parse, stale_ok: bool = False) -> Any:
        path = Path(path)
        key = str(path)
        ident = self.identity(path)
        if ident is None:
            self._documents.pop(key, None)
            return None
        cached = self._documents.get(key)
        if cached and cached[0] == ident:
            return cached[1]
        size = ident[2]
        if size > limit:
            self.notices.add(f"{path.name} exceeds the {limit >> 20} MiB per-file read bound")
            return None
        if size > self.available():
            self.notices.add("refresh byte budget reached; some records wait for the next refresh")
            return self._previous(cached) if stale_ok else None
        try:
            with open(path, "rb") as stream:
                data = stream.read(limit + 1)
        except OSError:
            return None
        self.remaining -= len(data)
        try:
            document = parse(data)
        except (ValueError, UnicodeError):
            self.notices.add(f"{path.name} is unreadable")
            document = None
        if self.identity(path) != ident:
            # Changed while read: the next refresh reads it whole.
            return self._previous(cached) if stale_ok else None
        self._documents[key] = (ident, document)
        if path.parent.parent.name == "cases" and path.parent.parent.parent.name == "qa":
            from hashlib import sha256

            self._detail_digests[key] = (ident, sha256(data).hexdigest())
        return document

    def _previous(self, cached: tuple[tuple, Any] | None) -> Any:
        """The last whole read of a record this refresh could not re-read, rather than nothing.

        Only for callers passing ``stale_ok``: a record rewritten often (the tier ledger, the agent's
        verdict) would otherwise read as absent whenever the budget ran out or a write raced the
        read, and its panel would claim nothing was recorded. Callers that key their own caches by
        stat identity must not take it, or they would file an old parse under the new identity.
        """
        if cached is None or cached[1] is None:
            return None
        self.notices.add("some records are shown as last read; the next refresh re-reads them")
        return cached[1]

    def extract(self, path: Path, digest, *, key: str, limit: int = 8 << 20) -> Any:
        """``digest(document)`` of one JSON record, cached by the file's stat identity.

        For scans over many large records of which a panel keeps only a few fields (every response of
        a run, every archived verdict): the parsed document is dropped once digested, so the cache
        holds the digest alone. A document the reader already holds for the same identity is digested
        without reading it again. ``key`` names the digest (one record may serve several). Returns
        ``digest(None)`` for an unreadable or oversized record, and :attr:`WAIT` when this refresh
        cannot read it (over the budget, or rewritten while read): the caller retries next refresh.
        """
        path = Path(path)
        slot = (key, str(path))
        ident = self.identity(path)
        if ident is None:
            self._extracts.pop(slot, None)
            return digest(None)
        cached = self._extracts.get(slot)
        if cached and cached[0] == ident:
            return cached[1]
        held = self._documents.get(str(path))
        if held and held[0] == ident:
            if isinstance(held[1], dict) and held[1].get("result_index"):
                return digest(self._details(held[1], path))
            value = digest(held[1])
            self._extracts[slot] = (ident, value)
            return value
        size = ident[2]
        if size > limit:
            self.notices.add(f"{path.name} exceeds the {limit >> 20} MiB per-file read bound")
            value = digest(None)
            self._extracts[slot] = (ident, value)
            return value
        if size > self.available():
            self.notices.add("refresh byte budget reached; some records wait for the next refresh")
            return self.WAIT
        try:
            with open(path, "rb") as stream:
                data = stream.read(limit + 1)
        except OSError:
            return self.WAIT
        self.remaining -= len(data)
        try:
            document = json.loads(data)
        except (ValueError, UnicodeError):
            self.notices.add(f"{path.name} is unreadable")
            document = None
        if self.identity(path) != ident:
            return self.WAIT
        value = digest(self._details(document, path))
        if not (isinstance(document, dict) and document.get("result_index")):
            self._extracts[slot] = (ident, value)
        return value

    def json(self, path: Path, *, limit: int = 8 << 20, stale_ok: bool = False) -> Any:
        return self._load(path, limit, json.loads, stale_ok)

    def _details(self, document, path):
        """Expand public result indexes using this reader's ordinary IO budget."""
        if isinstance(document, dict) and document.get("result_index"):
            from merlin.development.backend.feedback.selfcheck.results import hydrate

            cached = self._result_indexes.get(str(path))
            if cached and cached[0] is document:
                return cached[1]
            expanded = hydrate(document, document_path=path, reader=self.result_detail)
            pending = any(
                isinstance(row, dict) and row.get("detail") and row is opened
                for key in ("per_capsule", "cases")
                for row, opened in zip(document.get(key) or (), expanded.get(key) or ())
            )
            if not pending:
                self._result_indexes[str(path)] = (document, expanded)
            return expanded
        return document

    def result_detail(self, path: Path, *, limit: int):
        """Public JSON plus its original raw-byte digest, charged to this reader."""
        value = self.document(path, limit=limit)
        pinned = self._detail_digests.get(str(path))
        digest = pinned[1] if pinned and pinned[0] == self.identity(path) else None
        return value, digest

    def document(self, path: Path, *, limit: int = 8 << 20, stale_ok: bool = False) -> dict | None:
        """A JSON object, or None for anything else (``stale_ok``: see :meth:`_previous`)."""
        value = self.json(path, limit=limit, stale_ok=stale_ok)
        return self._details(value, path) if isinstance(value, dict) else None

    def parsed(self, path: Path, parse, *, limit: int = 8 << 20) -> Any:
        """``parse(bytes)`` of one file, cached by stat identity like :meth:`json` (one ``parse`` per path:
        a second caller with another ``parse`` gets the first one's cached value)."""
        return self._load(path, limit, parse)

    def yaml(self, path: Path, *, limit: int = 8 << 20) -> Any:
        def parse(data: bytes) -> Any:
            import yaml

            try:
                return yaml.load(data, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
            except yaml.YAMLError as exc:
                raise ValueError("unreadable YAML") from exc

        return self._load(path, limit, parse)

    def text_tail(self, path: Path, nbytes: int = 64 << 10) -> list[str]:
        """The complete lines among the last ``nbytes`` of a text file (never cached)."""
        try:
            with open(path, "rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                start = max(0, size - nbytes)
                stream.seek(start)
                data = stream.read(int(min(nbytes, self.available())))
        except OSError:
            return []
        self.remaining -= len(data)
        lines = data.decode(errors="replace").splitlines()
        return lines[1:] if start > 0 and lines else lines

    def tail(self, path: Path) -> JsonlTail:
        key = str(path)
        if key not in self._tails:
            self._tails[key] = JsonlTail(Path(path))
        return self._tails[key]

    def poll(self, path: Path, consume, on_reset=None) -> int:
        """Rows completed since the previous poll of ``path`` (see :class:`~..tail.JsonlTail`)."""
        tail = self.tail(path)
        before = tail.bytes_read
        count = tail.poll(consume, on_reset)
        self.remaining -= tail.bytes_read - before
        return count


def first_existing(paths: Iterable[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


# ---- model -------------------------------------------------------------------------------------
# Assemble one experiment's snapshot from the console sources, on a fast and a slow cadence.
#
# Each source module in :mod:`.sources` exposes ``read(ctx) -> dict`` and, optionally,
# ``inspect(ctx, key) -> dict`` for the drill-down drawer. A source sees the outputs of the sources
# before it in :data:`SOURCES` through ``ctx.shared`` (its latest refresh, even when that refresh was on
# the slow cadence), so dependencies are explicit in the order. A source that raises becomes an
# ``{"error": ...}`` section plus a notice; it never stops the console.
#
# Snapshot (``merlin.console.snapshot.v1``)::
#
#     {"schema", "generated_at", "experiment": {...discovery row...}, "operator_private",
#      "sections": {name: section}, "changed": [names refreshed this time],
#      "timings": {name: seconds}, "diagnostics": [...], "notices": [...]}
#
# Section shapes are documented in each source module's docstring and in
# ``docs/guides/experiment_dashboard.md``.

#: Bytes one interactive refresh may read, and the share of it held back for the insight panels.
REFRESH_BUDGET = 128 << 20
INSIGHT_RESERVE = REFRESH_BUDGET // 4
#: The small authoritative records several panels judge by, read before any bulk scan of a refresh
#: that includes slow sources: ``(experiment field, relative path, per-file bound)``.
AUTHORITATIVE = (
    ("workspace", "qa/tier_state.json", 64 << 20),
    ("workspace", "qa/verdict.json", 64 << 20),
    ("workspace", "qa/progress.json", 64 << 20),
    ("workspace", "qa/latest_completed_grade.json", 64 << 20),
    ("workspace", "qa/joint_physical_correctness.json", 16 << 20),
)

#: ``(name, tier)`` in dependency order. ``fast`` refreshes every interval; ``slow`` on the slow interval.
SOURCES: tuple[tuple[str, str], ...] = (
    ("resources", "fast"),
    ("machine", "fast"),
    ("lifecycle", "fast"),
    ("agents", "fast"),
    ("tokens", "fast"),
    ("jobs", "fast"),
    ("activity", "fast"),
    ("events", "fast"),
    ("timeline", "fast"),
    ("capsules", "slow"),
    ("progress", "slow"),
    ("agent_view", "slow"),
    ("perf", "slow"),
    ("golden", "slow"),
    ("passes", "slow"),
    ("stages", "slow"),
    ("performance", "slow"),
    ("tools", "slow"),
    ("library", "slow"),
    ("grader", "slow"),
    ("integrity", "slow"),
    ("attempts", "slow"),
    # CASES first: the CERTIFICATION headline repeats its cases-passing totals (ctx.shared).
    ("cases", "slow"),
    ("certification", "slow"),
    ("focus", "slow"),
    ("cost", "slow"),
)
#: The insight panels: every one returns the generic panel shape of :mod:`.sources.insight_common`.
INSIGHTS = (
    "performance",
    "cases",
    "tools",
    "grader",
    "integrity",
    "attempts",
    "certification",
    "focus",
    "cost",
    "library",
)
#: Sources whose content exists only for an operator (never shown in public mode).
OPERATOR_ONLY = frozenset({"golden"})


@dataclass
class Context:
    """What a source may use: the experiment row, the shared reader and its own persistent state."""

    experiment: dict[str, Any]
    reader: Reader
    state: dict[str, Any]
    operator_private: bool
    now: float
    shared: dict[str, Any] = field(default_factory=dict)
    live: bool = False


def _module(name: str):
    return importlib.import_module(f"{__package__}.sources.{name}")


class Collector:
    """One experiment's sources, reader and per-source state between refreshes."""

    def __init__(
        self,
        experiment: dict[str, Any],
        *,
        operator_private: bool = False,
        clock=time.time,
        sources: tuple[tuple[str, str], ...] = SOURCES,
        budget: int | None = REFRESH_BUDGET,
    ) -> None:
        """``budget=None`` reads without a byte budget: for a one-shot dump, never an interactive loop."""
        self.experiment = experiment
        self.operator_private = operator_private
        self.clock = clock
        self.sources = tuple((n, t) for n, t in sources if operator_private or n not in OPERATOR_ONLY)
        reserve = 0 if budget is None else min(INSIGHT_RESERVE, budget // 4)
        self.reader = Reader(budget=budget, reserve=reserve)
        self.state: dict[str, dict[str, Any]] = {name: {} for name, _ in self.sources}
        self.sections: dict[str, Any] = {}
        self.timings: dict[str, float] = {}

    def update_experiment(self, experiment: dict[str, Any]) -> None:
        """Adopt a fresher discovery row (state, last record time) for the same run."""
        if experiment.get("id") == self.experiment.get("id"):
            self.experiment = experiment

    def context(self, name: str, now: float) -> Context:
        return Context(
            experiment=self.experiment,
            reader=self.reader,
            state=self.state[name],
            operator_private=self.operator_private,
            now=now,
            shared=self.sections,
            live=self.experiment.get("state") in ("LIVE", "STALLED", "ACTIVE"),
        )

    def refresh(self, tier: str = "all") -> dict[str, Any]:
        """Re-read the sources of ``tier`` (``fast``, ``slow`` or ``all``); return the whole snapshot."""
        now = self.clock()
        self.reader.begin()
        if tier != "fast":
            self.prefetch()
        changed = []
        for name, source_tier in self.sources:
            if tier != "all" and source_tier != tier and name in self.sections:
                continue
            started = time.monotonic()
            # Insight panels come last in the order; the reserve keeps the bulk scans before them
            # from spending their share of the budget.
            self.reader.priority = name in INSIGHTS
            try:
                section = _module(name).read(self.context(name, now))
            except Exception as exc:  # noqa: BLE001 -- one broken reader never stops the console
                section = {"error": f"{type(exc).__name__}: {clean(exc, 160)}"}
            finally:
                self.reader.priority = False
            self.timings[name] = round(time.monotonic() - started, 4)
            self.sections[name] = section
            changed.append(name)
        from merlin.experiments.tracking.console.diagnose import diagnose

        snapshot = {
            "schema": SNAPSHOT_SCHEMA,
            "generated_at": now,
            "experiment": self.experiment,
            "operator_private": self.operator_private,
            "sections": self.sections,
            "changed": changed,
            "timings": dict(self.timings),
            "notices": sorted(self.reader.notices),
        }
        snapshot["diagnostics"] = diagnose(snapshot)
        return snapshot

    def prefetch(self) -> None:
        """Read the authoritative records into the cache first, from the whole budget.

        Unchanged ones are cache hits and cost nothing; a changed one is read before any bulk scan
        could exhaust the refresh, so the panels judging by it never show "not recorded" for lack of
        bytes. The newest fast verdict is one of them.
        """
        paths = []
        for field_name, relative, limit in AUTHORITATIVE:
            root = self.experiment.get(field_name)
            if root:
                paths.append((Path(root) / relative, limit))
        run_dir = self.experiment.get("run_dir")
        if run_dir:
            history = Path(run_dir) / "qa_history"
            newest = None
            for entry in scan_dir(history, 8192):
                name = entry.name
                token = name[len("verdict_fast_") : -len(".json")]
                if name.startswith("verdict_fast_") and name.endswith(".json") and token.isdigit():
                    if newest is None or int(token) > newest[0]:
                        newest = (int(token), name)
            if newest is not None:
                paths.append((history / newest[1], 32 << 20))
        self.reader.priority = True
        try:
            for path, limit in paths:
                self.reader.json(path, limit=limit)
        finally:
            self.reader.priority = False

    def inspect(self, kind: str, key: str) -> dict[str, Any]:
        """The drill-down record for one row: ``kind`` names the source, ``key`` the row."""
        names = {name for name, _ in self.sources}
        if kind not in names:
            return {"title": f"{kind}: not a console source", "sections": [], "paths": []}
        module = _module(kind)
        if not hasattr(module, "inspect"):
            return {"title": f"{kind} {key}", "sections": [], "paths": [], "notice": "this panel has no deeper record"}
        self.reader.begin()
        self.reader.priority = True  # one drawer, asked for now: the whole budget is its own
        try:
            return module.inspect(self.context(kind, self.clock()), key)
        except Exception as exc:  # noqa: BLE001
            return {
                "title": f"{kind} {key}",
                "sections": [],
                "paths": [],
                "error": f"{type(exc).__name__}: {clean(exc, 160)}",
            }
        finally:
            self.reader.priority = False


__all__ = [
    "Collector",
    "Context",
    "INSIGHTS",
    "OPERATOR_ONLY",
    "Reader",
    "SOURCES",
    "bounded_glob",
    "clean",
    "epoch",
    "first_existing",
    "is_dir",
    "is_file",
    "is_sha256",
    "mtime",
    "num",
    "obj",
    "positive",
    "rows",
    "scan_dir",
]
