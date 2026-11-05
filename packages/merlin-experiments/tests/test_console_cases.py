"""The CASES insight panel: one row per test case, its history per evidence kind, read-only and cheap."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from merlin_experiments.tracking.console import model, plain
from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import INSIGHTS, SOURCES, Context
from merlin_experiments.tracking.console.sources import case_history, cases, insight_common

NOW = time.time()
START = NOW - 6 * 3600
#: Sized like a live joint run: 252 test cases in three cohorts, a few hundred responses, many verdicts.
N_CASES = 252
COHORTS = ("functional", "scalability", "calibration")
PEAK = 256
TOKEN0 = int(START * 1e9)


def _write(path: Path, document, at: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document if isinstance(document, str) else json.dumps(document))
    if at is not None:
        os.utime(path, (at, at))
    return path


def _name(i: int) -> str:
    return f"T{i:03d}_case"


def _roofline(macs: int, read_cycles: float = 0.0) -> dict:
    return {
        "useful_macs": macs,
        "structural_peak_macs_per_cycle": PEAK,
        "structural_compute_floor_cycles": -(-macs // PEAK),
        "logical_request_ideal_service": {
            "read": {"ideal_requested_service_cycles": read_cycles, "requested_bytes": 4096}
        },
    }


def _modeled(cycles: float, macs: int, busiest: str = "accelerator.storage.accumulator.all") -> dict:
    return {
        "status": "PROVISIONAL",
        "complete_event_accounting": True,
        "estimated_whole_soc_cycles": cycles,
        "structural_roofline": _roofline(macs),
        "scoped_resource_feedback": {
            "resources": [
                {"name": busiest, "capacity_normalized_activity": 0.6},
                {"name": "cpu.issue", "capacity_normalized_activity": 0.2},
            ]
        },
        "feature_counts": {"target.requested_read_bytes": 8192, "target.commands": 40},
    }


def _gsim(ws: Path, tag: str, rows: list[tuple[str, float, str]], at: float, macs: int = 1 << 20) -> None:
    """One L3 certification response: ``rows`` of (case, barrier cycles, execution digest)."""
    _write(
        ws / ".qa_channel" / f"simresp_promo0_{tag}.json",
        {
            "sim": "gsim",
            "per_capsule": [
                {
                    "capsule": c,
                    "barrier_cycles": cyc,
                    "barrier_tier": "L3",
                    "barrier_status": "pass",
                    "pass": True,
                    "execution_digest": d,
                    "performance": {"status": "UNKNOWN", "structural_roofline": _roofline(macs)},
                }
                for c, cyc, d in rows
            ],
        },
        at,
    )


def _selfcheck(
    ws: Path,
    rid: str,
    rows: list[tuple[str, float, str | None]],
    at: float,
    *,
    variant: bool = False,
    macs: int = 1 << 20,
) -> None:
    """A Spike self-check (``op_idle_`` for idle timing): ``rows`` of (case, modeled cycles, digest)."""
    document = {
        "selfcheck_submission": "variants/v1" if variant else "submission",
        "graded_at_unix_ns": int(at * 1e9),
        "submission_sha256": "a" * 64,
        "cases": [
            {
                "capsule": c,
                "tiers": {"L2": "pass"},
                "status": "pass",
                "execution_digest": d,
                "performance": _modeled(cyc, macs),
            }
            for c, cyc, d in rows
        ],
    }
    if variant:
        document["variant"] = {"name": "v1"}
    _write(ws / ".qa_channel" / f"resp_{rid}.json", document, at)


def _verdict(run: Path, k: int, passing: set[str], modeled: dict[str, float], at: float) -> None:
    rows = []
    for i in range(N_CASES):
        name = _name(i)
        ok = name in passing
        row = {
            "capsule": name,
            "status": "pass" if ok else "fail",
            "tiers": {"L2": "pass" if ok else "fail"},
            "failure_category": None if ok else "numeric_mismatch",
        }
        if name in modeled:
            row["performance"] = _modeled(modeled[name], 1 << 20)
        rows.append(row)
    token = TOKEN0 + k * 10**9
    _write(
        run / "qa_history" / f"verdict_fast_{token}.json",
        {"per_capsule": rows, "graded_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(at))},
        at,
    )


def _tier(l2: str, l3: str | None, digest: str, cert_digest: str | None = None) -> dict:
    state = {"L2": {"status": l2, "execution_digest": digest}}
    if l3:
        state["L3"] = {"status": l3, "execution_digest": cert_digest or digest}
        slots = {"old": {"status": "pass", "execution_digest": "old" + digest}}
        if l3 == "pass":
            slots[cert_digest or digest] = {"status": "pass", "execution_digest": cert_digest or digest}
        state["<certs>"] = {"L3": slots}
    return state


@pytest.fixture
def run(tmp_path):
    """A live-sized run: gSIM certificates improving over time, idle timing, verdicts, a variant."""
    run = tmp_path / "out" / "runs" / "tgt" / "phase1" / "run-v1"
    ws = tmp_path / "out" / "build" / "ws" / "run-v1" / "workspace"
    (run / "qa_history").mkdir(parents=True)
    (ws / ".qa_channel").mkdir(parents=True)
    tier_state, rows, lanes = {}, [], {"primary": [], "correctness_regression": []}
    for i in range(N_CASES):
        name = _name(i)
        if i < 120:  # measured: three certificates, each faster; the newest certifies the current bytes
            for k, cycles in enumerate((40_000, 30_000, 20_000)):
                _gsim(ws, f"{i:03d}_{k}", [(name, cycles + i, f"d{i}-{k}")], START + 600 * (k + 1) + i)
            tier_state[name] = _tier("pass", "pass", f"d{i}-2")
        elif i < 150:  # L3 invalidated by a newer executable: re-proving, measured on earlier bytes
            _gsim(ws, f"{i:03d}_0", [(name, 50_000, f"d{i}-0")], START + 900)
            tier_state[name] = _tier("pass", "pending", f"d{i}-new", cert_digest=f"d{i}-0")
        elif i < 248:  # modeled only: L2 certified
            tier_state[name] = _tier("pass", None, f"d{i}")
        else:  # failing
            tier_state[name] = _tier("fail", None, f"d{i}")
        status = "fail" if i >= 248 else ("incomplete" if i >= 150 else "pass")
        rows.append(
            {
                "capsule": name,
                "status": status,
                "tiers": {"L2": "fail" if i >= 248 else "pass"},
                "failure_category": "numeric_mismatch" if i >= 248 else None,
                "graded": {"token": str(TOKEN0), "graded_at": "x"},
            }
        )
        (lanes["primary"] if 150 <= i < 248 else lanes["correctness_regression"]).append(name)
    # Idle timing of the modeled cases, three rounds, the second and third faster.
    for k, cycles in enumerate((90_000, 60_000, 45_000)):
        for j in range(150, 248, 4):
            batch = [(_name(n), cycles + n, None) for n in range(j, min(j + 4, 248))]
            _selfcheck(ws, f"op_idle_{k}_{j}", batch, START + 3000 + 900 * k + j)
    # The author's self-check timed case 0's current executable: a same-executable pair (modeled 19,000).
    _selfcheck(ws, "3_0001", [(_name(0), 19_000, "d0-2")], START + 4000)
    # A variant times case 1 much faster: it never describes submission/.
    _selfcheck(ws, "4_0002", [(_name(1), 1_000, "variant-digest")], START + 4100, variant=True)
    passing = {_name(i) for i in range(248)}
    for k in range(40):
        modeled = {_name(i): 100_000 - 1000 * k for i in range(150, 160)}
        _verdict(run, k, passing if k else set(), modeled, START + 300 + 400 * k)
    _write(ws / "qa" / "tier_state.json", tier_state)
    _write(
        ws / "qa" / "latest_completed_grade.json",
        {"per_capsule": rows, "joint_test_selection": {"performance_cohorts": lanes}},
    )
    experiment = {
        "id": str(run),
        "run_id": "run-v1",
        "run_dir": str(run),
        "workspace": str(ws),
        "state": "LIVE",
        "task": str(tmp_path),
        "runner_pid": None,
    }
    # The capsule catalog (the CAPSULES source shares its rows with the insight panels); the last two
    # cases are not in it.
    catalog = [
        {"name": _name(i), "cohort": COHORTS[i % 3], "op": "conv2d" if i % 2 else "matmul", "family": "contraction"}
        for i in range(N_CASES - 2)
    ]
    insight_common.share_rows(_ctx(experiment), "capsules", catalog)
    return experiment


def _ctx(
    experiment: dict,
    *,
    private: bool = True,
    reader: Reader | None = None,
    state: dict | None = None,
    shared: dict | None = None,
) -> Context:
    return Context(
        experiment=experiment,
        reader=reader or Reader(budget=None),
        state=state if state is not None else {},
        operator_private=private,
        now=NOW,
        shared=shared or {},
        live=True,
    )


def _rows(section: dict) -> dict[str, list[str]]:
    columns = section["table"]["columns"]
    return {r["key"]: dict(zip(columns, r["cells"], strict=True)) for r in section["table"]["rows"]}


def _summary(section: dict) -> dict[str, str]:
    return {line["label"]: line["value"] for line in section["summary"]}


# -- contract -------------------------------------------------------------------------------------
def test_cases_is_an_insight_after_what_it_reads():
    names = [n for n, _ in SOURCES]
    assert "cases" in INSIGHTS
    for needs in ("capsules", "perf"):
        assert names.index(needs) < names.index("cases")
    # CERTIFICATION repeats this panel's cases-passing totals, so it reads after it.
    assert names.index("cases") < names.index("certification")


def test_definitions_are_the_author_skills_own_and_write_no_bytecode(monkeypatch, tmp_path):
    import shutil
    import sys

    monkeypatch.setattr(case_history, "_SHARED", None)
    ps = case_history.definitions()
    script = Path(ps.__file__)
    assert script.name == "perf_summary.py" and script.parent.parent.name == "roofline-and-utilization"
    assert ps.bound(_roofline(1 << 20, read_cycles=9000.0)) == (9000.0, "read requests")
    # Loading writes no bytecode beside the script (checked on a copy: other tests may load the original).
    copy = tmp_path / "perf_summary.py"
    shutil.copy(script, copy)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    assert case_history.load_script(copy).bound is not None
    assert not (tmp_path / "__pycache__").exists() and sys.dont_write_bytecode is False


# -- the panel ------------------------------------------------------------------------------------
def test_every_case_has_a_row_with_status_cycles_roofline_and_gain(run):
    section = cases.read(_ctx(run))
    rows = _rows(section)
    assert len(rows) == N_CASES and section["title"] == "CASES"
    certified = rows[_name(0)]
    assert certified["status"] == "L3 ✓" and certified["cohort"] == "functional" and certified["family"] == "dense"
    assert rows[_name(251)]["cohort"] == "correctness_regression"  # not in the catalog: its performance lane
    assert certified["gSIM cyc"] == "20.0k" and certified["model cyc"] == "19.0k"
    assert certified["mod÷meas"] == "×0.95"  # same executable d0-2: 19,000 / 20,000
    assert certified["bound"] == "4,096" and certified["by"] == "compute"
    assert certified["gap"] == "×4.88" and certified["util"] == "20%"
    assert certified["gSIM best/now"] == "×2.00/×2.00"  # 40,000 first -> 20,000
    assert certified["model best/now"] == "first"  # one estimate: nothing to compare
    reproving = rows[_name(130)]
    assert reproving["status"] == "L3 ↻" and reproving["gap"].endswith("*")  # measured on earlier bytes
    modeled = rows[_name(200)]
    assert modeled["status"] == "L2 ✓" and modeled["gSIM cyc"] == "—" and modeled["model cyc"] == "45.2k"
    assert modeled["model best/now"] == "×2.00/×2.00"  # idle timing: 90,200 -> 45,200
    graded = rows[_name(152)]  # estimates of the grades, the newest last: 100,000 -> 61,000
    assert graded["model cyc"] == "61.0k"
    assert modeled["busiest"] == "accumulator.all"
    failing = rows[_name(250)]
    assert failing["status"] == "fail numeric_mismatch" and failing["gap"] == "—"
    # A variant timed case 1 at 1,000 cycles: never the submission's estimate.
    assert rows[_name(1)]["model cyc"] == "—"


def test_summary_counts_cohorts_and_keeps_measured_and_modeled_apart(run):
    section = cases.read(_ctx(run))
    values = _summary(section)
    assert values["cases passing"] == (
        "248/252 · calibration 82/83 · correctness_regression 0/2 · functional 83/84 · scalability 83/83"
    )
    assert "gSIM 150 · model 98 · none 0" == values["evidence of passing cases"]
    since = values["since first pass"]
    assert since.startswith("gSIM geomean ×") and "120/120 improved" in since
    assert "model geomean ×" in since and "98/98 improved" in since
    totals = section["totals"]
    assert totals["compared"] == {"measured": 120, "modeled": 98}
    # Measured speedup is from measured cycles only (40k -> 20k); modeled from estimates only.
    assert 1.9 < totals["speedup"]["measured"] < 2.01
    assert 1.4 < totals["speedup"]["modeled"] < 2.01
    assert totals["utilization_median_pct"] is not None and totals["gap_median"] is not None
    assert [s["label"] for s in section["series"]] == ["gSIM speedup vs first", "model speedup vs first"]


def test_sorts_order_every_row(run):
    section = cases.read(_ctx(run))
    labels = [s["label"] for s in section["sorts"]]
    assert labels == ["biggest headroom", "worst gap", "lowest utilization", "biggest gain", "cohort", "status", "name"]
    n = len(section["table"]["rows"])
    for sort in section["sorts"]:
        assert sorted(sort["order"]) == list(range(n))
    rows = section["table"]["rows"]
    worst = [rows[i] for i in section["sorts"][1]["order"]]
    gaps = [float(r["cells"][9].strip("×*")) for r in worst if r["cells"][9] != "—"]
    assert gaps == sorted(gaps, reverse=True)
    by_status = [rows[i]["cells"][3] for i in section["sorts"][5]["order"]]
    assert by_status[0].startswith("fail")


def test_history_updates_when_a_case_improves_and_is_cheap_when_nothing_changed(run):
    state: dict = {}
    reader = Reader(budget=None)
    first = cases.read(_ctx(run, reader=reader, state=state))
    assert cases.read(_ctx(run, reader=reader, state=state)) is first  # memoized: nothing re-encoded
    ws = Path(run["workspace"])
    _gsim(ws, "000_3", [(_name(0), 10_000, "d0-3")], NOW - 60)
    second = cases.read(_ctx(run, reader=reader, state=state))
    assert second is not first
    row = _rows(second)[_name(0)]
    assert row["gSIM cyc"] == "10.0k" and row["gSIM best/now"] == "×4.00/×4.00"
    # A slower measurement afterwards: best stays, latest moves.
    _gsim(ws, "000_4", [(_name(0), 16_000, "d0-4")], NOW - 30)
    row = _rows(cases.read(_ctx(run, reader=reader, state=state)))[_name(0)]
    assert row["gSIM best/now"] == "×4.00/×2.50"


def test_a_budgeted_refresh_catches_up_without_rereading(run, monkeypatch):
    monkeypatch.setattr(case_history, "STEP", 64 << 10)
    reader, state = Reader(budget=8 << 20), {}
    pending = []
    for _ in range(80):
        reader.begin()
        section = cases.read(_ctx(run, reader=reader, state=state))
        pending.append(section["history"]["pending"])
        if not pending[-1]:
            break
    assert pending[0] > 0 and pending[-1] == 0 and pending == sorted(pending, reverse=True)
    assert "still to read" not in _summary(section)["history"]
    reader.begin()
    cases.read(_ctx(run, reader=reader, state=state))
    assert reader.remaining == reader.budget  # every record digested once, none read again


def test_public_mode_names_no_case_and_inspect_still_works(run):
    state: dict = {}
    section = cases.read(_ctx(run, private=False, state=state))
    text = json.dumps(section)
    assert "T000_case" not in text and "case 001" in text
    drawer = cases.inspect(_ctx(run, private=False, state=state), "case 001")
    assert drawer["paths"] == [] and drawer["title"] == "test case case 001"
    lines = "\n".join(line for s in drawer["sections"] for line in s["lines"])
    assert "T000_case" not in lines and "exe " not in lines and "40,000 cyc" in lines


def test_inspect_shows_the_full_history_with_first_best_latest(run):
    state: dict = {}
    cases.read(_ctx(run, state=state))
    drawer = cases.inspect(_ctx(run, state=state), _name(0))
    sections = {s["title"]: s["lines"] for s in drawer["sections"]}
    measured = sections["gSIM measured"]
    assert len(measured) == 3 and "[first]" in measured[0] and "[best, latest]" in measured[-1]
    assert "gSIM L3 certification" in measured[0] and "exe d0-0" in measured[0]
    assert sections["Modeled ÷ measured"][0].startswith("modeled 19,000 ÷ measured 20,000 = ×0.95")
    assert any("L2 pass" in line for line in sections["Grades (changes only)"])
    assert sections["Status"][1].startswith("status L3 ✓")
    assert any(p.endswith("tier_state.json") for p in drawer["paths"])
    assert cases.inspect(_ctx(run, state=state), "nope")["notice"] == "no such test case"


def test_first_pass_is_the_first_verdict_passing_l2(run):
    history = case_history.History()
    history.refresh(_ctx(run))
    entry = history.cases()[_name(200)]
    assert entry["first_pass_at"] == pytest.approx(START + 300 + 400, abs=1)


def test_an_empty_run_shows_no_invented_numbers(tmp_path):
    run = tmp_path / "out" / "runs" / "tgt" / "phase1" / "r"
    run.mkdir(parents=True)
    experiment = {
        "id": str(run),
        "run_id": "r",
        "run_dir": str(run),
        "workspace": str(tmp_path / "ws"),
        "state": "LIVE",
        "task": None,
        "runner_pid": None,
    }
    section = cases.read(_ctx(experiment))
    assert section["table"]["rows"] == []
    assert all(line["value"] == "—" for line in section["summary"] if line["label"] != "history")


def test_collector_and_plain_text_show_the_panel(run):
    collector = model.Collector(run, operator_private=True, clock=lambda: NOW, budget=None)
    snapshot = collector.refresh()
    assert snapshot["sections"]["cases"]["title"] == "CASES"
    text = plain.render(snapshot, width=220)
    assert " CASES" in text and "cases passing" in text and "more test cases" in text
    drawer = collector.inspect("cases", _name(0))
    assert drawer["title"] == f"test case {_name(0)}"


# -- the reader's digest cache --------------------------------------------------------------------
def test_reader_extract_keeps_only_the_digest_and_reuses_held_documents(tmp_path):
    path = _write(tmp_path / "a.json", {"x": 1, "bulk": "y" * 4096})
    reader = Reader(budget=1 << 20)
    calls = []

    def digest(document):
        calls.append(1)
        return document and document["x"]

    assert reader.extract(path, digest, key="k") == 1
    spent = reader.budget - reader.remaining
    assert spent > 4096 and str(path) not in reader._documents
    assert reader.extract(path, digest, key="k") == 1 and len(calls) == 1  # cached by identity
    other = _write(tmp_path / "b.json", {"x": 2})
    reader.document(other)
    before = reader.remaining
    assert reader.extract(other, digest, key="k") == 2 and reader.remaining == before  # held: no re-read
    tight = Reader(budget=100)
    assert tight.extract(path, digest, key="k") is Reader.WAIT  # over budget: retried next refresh
    assert reader.extract(tmp_path / "missing.json", digest, key="k") is None


def test_performance_totals_leave_out_excluded_cases_but_count_them_for_correctness():
    def case(name, util, scope):
        return {
            "name": name,
            "cohort": "functional",
            "passing": True,
            "util": util,
            "gap": util / 10,
            "cycles": 100,
            "current": {"kind": "modeled", "macs": util, "peak": 1},
            "measured": None,
            "modeled": None,
            "performance_scope": scope,
        }

    grade = {"joint_test_selection": {"performance_scope": {"excluded": ["copy"]}}}
    assert cases._excluded(grade) == {"copy"} and cases._excluded({}) == set()
    totals = cases._totals([case("scorer", 40.0, "scored"), case("copy", 90.0, "excluded")])
    assert totals["passing"] == 2 and totals["by_cohort"] == {"functional": [2, 2]}
    assert totals["utilization_median_pct"] == 40.0 and totals["performance_excluded"] == 1


# -- reference-compiler baselines -----------------------------------------------------------------
def _baselines(ws: Path, rows: list[tuple[str, str, str, int]]) -> None:
    """A staged ``merlin.performance_baselines.v1`` file: ``rows`` of (case, source, engine, cycles)."""
    from merlin_experiments.phase1.feedback import reference_baselines as RB

    _write(
        ws / RB.REFERENCE_ROOT / RB.STAGED,
        {
            "schema": RB.SCHEMA,
            "advisory_only": True,
            "sources": {"fsm-library": {"display": "the FSM software library"}, "merlin-reference": {}},
            "baselines": [{"case": c, "source": s, "engine": e, "cycles": n} for c, s, e, n in rows],
        },
    )


def test_baseline_columns_compare_the_newest_gsim_cycles_on_the_same_engine_only(run):
    ws = Path(run["workspace"])
    # T000 latest gSIM = 20,000; T150 has only model estimates; T001 has a FireSim-only baseline.
    _baselines(
        ws,
        [
            (_name(0), "fsm-library", "gsim", 10_000),
            (_name(0), "merlin-reference", "gsim", 25_000),
            (_name(150), "fsm-library", "gsim", 30_000),
            (_name(1), "fsm-library", "firesim", 15_000),
        ],
    )
    state: dict = {}
    section = cases.read(_ctx(run, state=state))
    columns = section["table"]["columns"]
    at = columns.index("mod÷meas")
    assert columns[at + 1 : at + 5] == ["fsm-library cyc", "÷fsm-library", "merlin-reference cyc", "÷merlin-reference"]
    rows = _rows(section)
    assert rows[_name(0)]["fsm-library cyc"] == "10.0k" and rows[_name(0)]["÷fsm-library"] == "×2.00"
    assert rows[_name(0)]["÷merlin-reference"] == "×0.80"
    assert rows[_name(150)]["fsm-library cyc"] == "30.0k" and rows[_name(150)]["÷fsm-library"] == "—"
    assert rows[_name(1)]["fsm-library cyc"] == "15.0k FS" and rows[_name(1)]["÷fsm-library"] == "—"
    assert rows[_name(2)]["fsm-library cyc"] == "—"
    line = _summary(section)["vs reference compilers"]
    assert line.startswith("geomean ×2.00 slower than the FSM software library (gSIM, 1 case)")
    assert "faster than merlin-reference" in line and "worst: `T000_case` ×2.00" in line
    worst = next(s for s in section["sorts"] if s["label"] == "slowest vs reference")
    assert section["table"]["rows"][worst["order"][0]]["key"] == _name(0)
    drawer = cases.inspect(_ctx(run, state=state), _name(0))
    lines = {s["title"]: s["lines"] for s in drawer["sections"]}["Reference compilers (operator-measured, advisory)"]
    assert lines[0] == "fsm-library: 10,000 cycles on gsim; newest gSIM ÷ baseline ×2.00"
    public = cases.read(_ctx(run, private=False))
    assert "T000_case" not in json.dumps(public) and "worst: `case 001`" in _summary(public)["vs reference compilers"]


def test_without_baselines_the_columns_are_unchanged(run):
    section = cases.read(_ctx(run))
    assert section["table"]["columns"] == cases.COLUMNS
    assert "vs reference compilers" not in _summary(section)
