"""Per-test-case performance history: every passing observation of each case, by evidence kind, in time order.

Not a console source itself; :mod:`.cases` draws it. Read-only, from the run's own records, each read
once and kept only as a small digest cached by the file's stat identity (:meth:`..io.Reader.extract`):

* **measured** -- gSIM ``barrier_cycles`` of passing L3 rows: ``.qa_channel/simresp_*.json`` (L3
  certification after promotion) and ``resp_*.json`` with ``sim: gsim`` (the author's gSIM self-checks);
* **modeled** -- the Spike timing model's ``estimated_whole_soc_cycles`` with complete event accounting,
  of rows passing their numerical check: ``resp_*.json`` self-checks (the author's and the operator's
  ``op_idle_`` idle timing) and the grade verdicts ``qa_history/verdict_fast_*.json``;
* **first pass** -- the first verdict in which the case passed L2.

The definitions (roofline bound and ``bound_by``, array utilization, the busiest modeled unit, which
rows count as measured or modeled, which responses describe ``submission/``) are the author skill's own
``perf_summary.py`` (``roofline-and-utilization``), loaded by path, so the console and the author read
the same evidence the same way. Variant and knob-copy responses never describe ``submission/``; their
rows count only toward the same-executable modeled/measured pairs, as in the skill's calibration.

When the grader keeps the per-case ledger ``qa/perf_history.jsonl`` (``merlin.phase1.perf_history.v1``,
one row per case and evidence kind whenever a completed grade sees new cycles), it is tailed instead
of deriving: far fewer bytes, and its times are recorded rather than file modification times.

Each refresh reads new records newest first, at most :data:`STEP` bytes (a one-shot dump reads all),
so a first look at a long run catches up over a few refreshes; :attr:`History.pending` says how many
records still wait. Measured and modeled observations are never mixed in one series.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from ..io import Reader, epoch, is_dir, obj, positive, scan_dir

#: Bytes of new records one refresh may digest (the reader's budget still applies on top).
STEP = 48 << 20
#: Largest single record read (whole-corpus self-check responses reach ~11 MB).
RECORD_LIMIT = 48 << 20
MAX_CHANNEL_FILES = 65536
MAX_VERDICTS = 8192
LEDGER = Path("qa") / "perf_history.jsonl"
LEDGER_SCHEMA = "merlin.phase1.perf_history.v1"
KINDS = ("measured", "modeled")
_SKILL_SCRIPT = ("phase1", "author_skills", "roofline-and-utilization", "scripts", "perf_summary.py")
_SHARED: Any = None


def definitions():
    """The author skill's ``perf_summary.py`` module, loaded once by path (no bytecode is written)."""
    global _SHARED
    if _SHARED is None:
        import merlin_experiments

        _SHARED = load_script(Path(merlin_experiments.__file__).parent.joinpath(*_SKILL_SCRIPT))
    return _SHARED


def load_script(path: Path):
    """A stdlib-only script loaded as a module by path, writing no bytecode beside it."""
    spec = importlib.util.spec_from_file_location("merlin_experiments._perf_summary_definitions", path)
    module = importlib.util.module_from_spec(spec)
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def observation(kind: str, cycles: float, performance: Any, *, t: float, digest: Any, origin: str, path: str) -> dict:
    """One observation of one case, with the roofline terms of the record that produced it."""
    ps = definitions()
    performance = obj(performance)
    roofline = obj(performance.get("structural_roofline"))
    limit, limiter = ps.bound(roofline)
    return {
        "t": t,
        "kind": kind,
        "cycles": cycles,
        "digest": digest if isinstance(digest, str) and digest else None,
        "origin": origin,
        "path": path,
        "bound": limit,
        "by": limiter,
        "util": ps.utilization_pct(roofline, cycles),
        "macs": positive(roofline.get("useful_macs")),
        "peak": positive(roofline.get("structural_peak_macs_per_cycle")),
        "busiest": ps.busiest(performance),
        "traffic": ps.traffic(performance, roofline) or None,
    }


def _origin(name: str, document: dict) -> str:
    gsim = document.get("sim") == "gsim"
    if name.startswith("simresp_"):
        return "gSIM L3 certification" if gsim else "simulation"
    if gsim:
        return "idle gSIM measurement" if name.startswith("resp_op_idle_") else "gSIM self-check"
    return "idle timing" if name.startswith("resp_op_idle_") else "self-check"


def response_digest(path: str, mtime: float | None):
    """The digest of one self-check or simulation response: its observations and its pair halves."""
    name = os.path.basename(path)

    def digest(document: Any) -> dict | None:
        if not isinstance(document, dict):
            return None
        ps = definitions()
        t = epoch(document.get("graded_at_unix_ns")) or mtime
        origin = _origin(name, document)
        submission = ps.is_submission(document)
        found: list[dict] = []
        pairs: list[tuple] = []
        for row in ps.case_rows(document):
            case = ps.case_name(row)
            performance = obj(row.get("performance"))
            digest_ = row.get("execution_digest")
            measured = ps.measured_cycles(document, row)
            modeled = ps.modeled_cycles(performance)
            if isinstance(digest_, str) and digest_ and t:
                if measured is not None:
                    pairs.append((case, digest_, "measured", measured, t))
                if modeled is not None:
                    pairs.append((case, digest_, "modeled", modeled, t))
            if not submission or not t:
                continue
            if measured is not None:
                found.append(
                    {
                        "case": case,
                        **observation("measured", measured, performance, t=t, digest=digest_, origin=origin, path=path),
                    }
                )
            elif modeled is not None and ps.passes(row):
                found.append(
                    {
                        "case": case,
                        **observation("modeled", modeled, performance, t=t, digest=digest_, origin=origin, path=path),
                    }
                )
        return {"t": t, "observations": found, "pairs": pairs}

    return digest


def verdict_digest(path: str, token: int):
    """The digest of one archived grade verdict: who passed L2 then, and its timing-model estimates."""

    def digest(document: Any) -> dict | None:
        if not isinstance(document, dict):
            return None
        ps = definitions()
        t = epoch(document.get("graded_at")) or epoch(token)
        status: dict[str, tuple] = {}
        found: list[dict] = []
        for row in ps.case_rows(document):
            case = ps.case_name(row)
            tiers = obj(row.get("tiers"))
            if not tiers and row.get("status") in (None, "gated", "deferred", "declined", "incomplete", "carried"):
                continue  # the case did not run in this grade
            passed = ps.passes(row)
            status[case] = (passed, row.get("status"), row.get("failure_category"))
            modeled = ps.modeled_cycles(row.get("performance"))
            if passed and modeled is not None and t:
                found.append(
                    {
                        "case": case,
                        **observation(
                            "modeled",
                            modeled,
                            row.get("performance"),
                            t=t,
                            digest=row.get("execution_digest"),
                            origin=f"grade {token}",
                            path=path,
                        ),
                    }
                )
        return {"t": t, "token": token, "status": status, "observations": found, "pairs": []}

    return digest


def ledger_observation(row: dict, path: str) -> dict | None:
    """One ``merlin.phase1.perf_history.v1`` ledger row as an observation (``None`` for other rows)."""
    if row.get("schema") != LEDGER_SCHEMA or row.get("kind") not in KINDS or not isinstance(row.get("test_case"), str):
        return None
    cycles, t = positive(row.get("cycles")), epoch(row.get("t"))
    if cycles is None or t is None:
        return None
    busiest = obj(row.get("dominant_modeled_resource")) or None
    return {
        "case": row["test_case"],
        "t": t,
        "kind": row["kind"],
        "cycles": cycles,
        "digest": row.get("execution_digest") if isinstance(row.get("execution_digest"), str) else None,
        "origin": f"grade {row.get('grade_token')}" if row.get("grade_token") else "grade",
        "path": path,
        "bound": positive(row.get("roofline_bound_cycles")),
        "by": row.get("bound_by") if isinstance(row.get("bound_by"), str) else None,
        "util": positive(row.get("array_utilization_pct")),
        "macs": positive(row.get("useful_macs")),
        "peak": positive(row.get("structural_peak_macs_per_cycle")),
        "busiest": busiest if busiest and busiest.get("name") else None,
        "traffic": obj(row.get("traffic")) or None,
        "evidence": row.get("evidence") if isinstance(row.get("evidence"), str) else None,
    }


class History:
    """One experiment's per-case observations, extended incrementally from new records."""

    def __init__(self) -> None:
        self.digests: dict[str, tuple[tuple, Any]] = {}  # path -> (stat identity, digest)
        self.pending = 0
        self.records = 0
        self.version = 0
        self.basis = "derived"
        self.ledger: list[dict] = []
        self.ledger_passes: dict[str, float] = {}
        self.ledger_pairs: dict[str, dict] = {}
        self.ledger_status: dict[str, list[tuple]] = {}
        self._cases: dict[str, dict] | None = None
        self._read_any = False

    # -- reading ------------------------------------------------------------------------------------
    def refresh(self, ctx) -> None:
        workspace = ctx.experiment.get("workspace")
        ledger = Path(workspace) / LEDGER if workspace else None
        if ledger is not None and os.path.isfile(ledger):
            self._tail_ledger(ctx, ledger)
            return
        if self.basis != "derived":
            self.__init__()
        self._derive(ctx)

    def _tail_ledger(self, ctx, path: Path) -> None:
        if self.basis != "ledger":
            self.__init__()
            self.basis = "ledger"

        def reset() -> None:
            self.ledger.clear()
            self.ledger_passes.clear()
            self.ledger_pairs.clear()
            self.ledger_status.clear()

        def consume(row: dict) -> None:
            if row.get("schema") != LEDGER_SCHEMA or not isinstance(row.get("test_case"), str):
                return
            t = epoch(row.get("t"))
            if row.get("kind") == "status":
                if t is not None:
                    passed = row.get("l2") == "pass"
                    if passed:
                        self.ledger_passes.setdefault(row["test_case"], t)
                    self.ledger_status.setdefault(row["test_case"], []).append((t, passed, row.get("l2"), None))
                return
            if row.get("kind") == "pair":
                modeled, measured = positive(row.get("modeled")), positive(row.get("measured"))
                if modeled and measured and isinstance(row.get("execution_digest"), str):
                    self.ledger_pairs[row["test_case"]] = {
                        "digest": row["execution_digest"],
                        "modeled": modeled,
                        "measured": measured,
                        "t": t,
                    }
                return
            found = ledger_observation(row, str(path))
            if found is not None:
                self.ledger.append(found)

        if ctx.reader.poll(path, consume, reset):
            self.version += 1
            self._cases = None
        self.records = ctx.reader.tail(path).rows_read
        self.pending = 0

    def _inventory(self, ctx) -> list[tuple[float, str, tuple, Any]]:
        """``(mtime, path, identity, digest factory)`` of every record, newest first."""
        found: dict[str, tuple[float, str, tuple, Any]] = {}
        channels = []
        if ctx.experiment.get("workspace"):
            channels.append(Path(ctx.experiment["workspace"]) / ".qa_channel")
        run_dir = ctx.experiment.get("run_dir")
        if run_dir:
            channels.append(Path(run_dir) / "agent_evidence_snapshot" / ".qa_channel")
        for channel in channels:
            if not is_dir(channel):
                continue
            for entry in scan_dir(channel, MAX_CHANNEL_FILES):
                name = entry.name
                if not name.endswith(".json") or not name.startswith(("resp_", "simresp_")) or name in found:
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                ident = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                found[name] = (st.st_mtime, entry.path, ident, response_digest(entry.path, st.st_mtime))
        if run_dir:
            history = Path(run_dir) / "qa_history"
            for entry in scan_dir(history, MAX_VERDICTS):
                name = entry.name
                token = name[len("verdict_fast_") : -len(".json")]
                if not (name.startswith("verdict_fast_") and name.endswith(".json") and token.isdigit()):
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                ident = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                found[f"qa_history/{name}"] = (st.st_mtime, entry.path, ident, verdict_digest(entry.path, int(token)))
        return sorted(found.values(), key=lambda item: -item[0])

    def _derive(self, ctx) -> None:
        reader: Reader = ctx.reader
        inventory = self._inventory(ctx)
        present = {path for _, path, _, _ in inventory}
        changed = False
        for path in [p for p in self.digests if p not in present]:
            del self.digests[path]
            changed = True
        start = reader.remaining
        unbounded = reader.budget == float("inf")
        # A bulk scan: it spends the ordinary budget, never the reserve the insight panels judge by.
        priority, reader.priority = reader.priority, False
        try:
            self.pending = self._digest_new(reader, inventory, start, unbounded)
        finally:
            reader.priority = priority
        self.records = len(self.digests)
        if changed or self._read_any:
            self.version += 1
            self._cases = None

    def _digest_new(self, reader: Reader, inventory: list, start: float, unbounded: bool) -> int:
        """Digest records not read yet (or changed); returns how many still wait."""
        self._read_any = False
        waiting = 0
        for _, path, ident, factory in inventory:
            held = self.digests.get(path)
            if held is not None and held[0] == ident:
                continue
            if not unbounded and start - reader.remaining >= STEP:
                waiting += 1
                continue
            value = reader.extract(Path(path), factory, key="case_history", limit=RECORD_LIMIT)
            if value is Reader.WAIT:
                waiting += 1
                continue
            self.digests[path] = (ident, value)
            self._read_any = True
        return waiting

    # -- aggregation --------------------------------------------------------------------------------
    def cases(self) -> dict[str, dict]:
        """``{case: {"measured": [obs], "modeled": [obs], "first_pass_at", "pair"}}``, observations by time."""
        if self._cases is not None:
            return self._cases
        out: dict[str, dict] = {}

        def case(name: str) -> dict:
            return out.setdefault(name, {"measured": [], "modeled": [], "first_pass_at": None, "pair": None})

        pairs: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
        if self.basis == "ledger":
            for found in self.ledger:
                case(found["case"])[found["kind"]].append(found)
            for name, t in self.ledger_passes.items():
                case(name)["first_pass_at"] = t
            for name, pair in self.ledger_pairs.items():
                case(name)["pair"] = dict(pair)
        for _, digest in self.digests.values():
            if not isinstance(digest, dict):
                continue
            for found in digest["observations"]:
                case(found["case"])[found["kind"]].append(found)
            for name, (passed, _, _) in obj(digest.get("status")).items():
                if passed and digest.get("t"):
                    entry = case(name)
                    if entry["first_pass_at"] is None or digest["t"] < entry["first_pass_at"]:
                        entry["first_pass_at"] = digest["t"]
            for name, executable, kind, cycles, t in digest["pairs"]:
                halves = pairs.setdefault((name, executable), {})
                if kind not in halves or t > halves[kind][1]:
                    halves[kind] = (cycles, t)
        for (name, executable), halves in pairs.items():
            if len(halves) == 2:
                when = max(halves["measured"][1], halves["modeled"][1])
                entry = case(name)
                if entry["pair"] is None or when > entry["pair"]["t"]:
                    entry["pair"] = {
                        "digest": executable,
                        "measured": halves["measured"][0],
                        "modeled": halves["modeled"][0],
                        "t": when,
                    }
        for entry in out.values():
            for kind in KINDS:
                entry[kind].sort(key=lambda o: (o["t"], o["cycles"]))
                if entry["first_pass_at"] is None and entry[kind]:
                    entry["first_pass_at"] = entry[kind][0]["t"]
                elif entry[kind]:
                    entry["first_pass_at"] = min(entry["first_pass_at"], entry[kind][0]["t"])
        self._cases = out
        return out

    def statuses(self) -> dict[str, list[tuple[float, bool, Any, Any]]]:
        """``{case: [(t, passed L2, status, failure category)]}`` per archived verdict (or ledger status
        row), oldest first."""
        out: dict[str, list] = {name: list(values) for name, values in self.ledger_status.items()}
        for _, digest in self.digests.values():
            if isinstance(digest, dict) and digest.get("status") and digest.get("t"):
                for name, (passed, status, category) in digest["status"].items():
                    out.setdefault(name, []).append((digest["t"], passed, status, category))
        for values in out.values():
            values.sort(key=lambda v: v[0])
        return out


def summarize(observations: list[dict]) -> dict | None:
    """First, best and latest of one evidence kind, with the speedups since the first observation."""
    if not observations:
        return None
    first, latest = observations[0], observations[-1]
    best = min(observations, key=lambda o: (o["cycles"], o["t"]))
    return {
        "first": first,
        "best": best,
        "latest": latest,
        "count": len(observations),
        "distinct": len({(o["cycles"], o["digest"]) for o in observations}),
        "speedup_best": first["cycles"] / best["cycles"],
        "speedup_latest": first["cycles"] / latest["cycles"],
    }


def history_for(ctx) -> History:
    """This experiment's history, kept in the source's own state between refreshes."""
    history = ctx.state.get("history")
    if not isinstance(history, History):
        history = ctx.state["history"] = History()
    history.refresh(ctx)
    return history


__all__ = [
    "History",
    "KINDS",
    "LEDGER",
    "LEDGER_SCHEMA",
    "STEP",
    "definitions",
    "history_for",
    "ledger_observation",
    "observation",
    "response_digest",
    "summarize",
    "verdict_digest",
]
