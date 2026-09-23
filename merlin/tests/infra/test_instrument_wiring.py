"""An instrument nothing calls is not an instrument.

Four modules that compute a quality signal had tests and no production caller: what fraction of a
model's calls reached the accelerator and which fell back unexplained, which of the target's
declared instructions the program never emits, how many cycles the array is occupied issuing a
program's tiles, and what a candidate TRADED when it spent one resource to buy another. Each was
complete, each was covered, and none of them reached a reader -- which reads exactly like a check
that passes.

This file holds two kinds of test, and both are needed. The first asks the structural question the
wiring gate asks -- does production code import this module -- because a caller is what makes the
measurement reachable at all. The second asks the question that one cannot: does the caller put the
measurement somewhere a human or an agent reads, and does the number change when the thing it
measures changes. A module imported and then discarded would pass the first and fail the second.
"""

from __future__ import annotations

import ast
import importlib.util
import sys

import pytest
from phase1_feedback import feedback_source

from build_tools.scripts._source_layout import SOURCE_SCAN_ROOTS, module_name, python_files
from merlin.common.paths import merlin_dir, repo_root

ROOT = repo_root()

#: The four instruments this change wired, as repo-relative module paths.
INSTRUMENTS = (
    "merlin.perf.lowering_coverage",
    "merlin.perf.isa_utilization",
    "merlin.perf.mesh_occupancy",
    "merlin.perf.cost_terms",
)

#: Where a production importer may live. Mirrors the wiring gate's own roster: the library, the
#: experiments, the targets and the build tools. The test suite is deliberately absent -- a test
#: proves a module works, not that anything uses it.
PRODUCTION = (*SOURCE_SCAN_ROOTS, "merlin/experiments", "merlin/targets", "build_tools")


def _module_name(path):
    return module_name(path, ROOT)


def _imports(path) -> set[str]:
    """Every dotted module name ``path`` imports, with relative imports resolved.

    Structural (``ast``) rather than a word search: a module's name appears in the comments and
    docstrings of code that never imports it, and counting those would report a dead instrument as
    wired -- which is the failure this test exists to catch.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    own = _module_name(path)
    package = None
    if own is not None:
        package = own if path.name == "__init__.py" else own.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                if package is None:
                    continue
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            if base:
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _production_importers(module: str) -> list[str]:
    out: list[str] = []
    for relative in python_files(ROOT, PRODUCTION):
        if "tests" in relative.parts:
            continue
        path = ROOT / relative
        if _module_name(path) == module:
            continue
        if module in _imports(path):
            out.append(str(relative))
    return out


@pytest.mark.parametrize("source", ["src/merlin", "packages/merlin-analysis/src/merlin"])
def test_wiring_follows_relocated_relative_imports(tmp_path, monkeypatch, source):
    """A caller in either namespace owner counts; a package's tests and generated files do not."""
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    package = tmp_path / source
    instrument = package / "perf/instrument.py"
    instrument.parent.mkdir(parents=True)
    instrument.write_text("from . import instrument\n")
    for relative in ("perf/caller.py", "perf/tests/test_caller.py", "perf/build/generated.py"):
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "from . import instrument\n" if relative == "perf/caller.py" else "import merlin.perf.instrument\n"
        )
    if source == "src/merlin":
        alias = tmp_path / "merlin/python/merlin"
        alias.parent.mkdir(parents=True)
        alias.symlink_to(package, target_is_directory=True)
    assert _production_importers("merlin.perf.instrument") == [f"{source}/perf/caller.py"]


@pytest.mark.parametrize("module", INSTRUMENTS)
def test_every_instrument_has_a_production_caller(module: str) -> None:
    assert _production_importers(module), (
        f"{module} has tests at most -- no production code imports it. Call it from the path it "
        f"was built for, or delete it."
    )


# --------------------------------------------------------------------------------------------
# lowering_coverage -- the grader's whole-model placement census
# --------------------------------------------------------------------------------------------


def _ledger_result(*entries):
    return {"mesh_execution": {"dispatch_ledger": [dict(e, ordinal=i, status="pass") for i, e in enumerate(entries)]}}


def test_the_grader_names_the_host_call_that_stated_no_reason() -> None:
    """The finding: a call that went elsewhere and said nothing.

    The counter rule beside it answers "did anything fall back" with one violation string, which
    cannot separate a shape the hardware genuinely cannot take from a lowering nobody finished.
    """
    from merlin.targetgen import capsule_grade as CG

    census = CG.model_execution_check(
        _ledger_result(
            {"symbol": "k0", "lane": "on_mesh"},
            {"symbol": "k1", "lane": "host_fallback", "mesh_decline": "reduction depth exceeds the tile edge"},
            {"symbol": "k2", "lane": "scalar_rvv_lane", "executor": "a host library"},
        )
    )["lowering_coverage"]

    assert census["operations"] == 3
    assert census["on_accelerator"] == 1 and census["on_host"] == 2
    assert census["unjustified_host_operations"] == ["k2"]
    assert [row["reason"] for row in census["host_reasons"]] == ["reduction depth exceeds the tile edge"]


def test_a_justified_fallback_is_not_reported_as_unexplained() -> None:
    """The mutation that must move the verdict: give the same call a reason and it clears.

    A census that reported every fallback would be unsatisfiable -- some are forced by the ISA --
    and a rule nobody can satisfy is a rule everybody routes around.
    """
    from merlin.targetgen import capsule_grade as CG

    census = CG.model_execution_check(
        _ledger_result(
            {"symbol": "k2", "lane": "scalar_rvv_lane", "reason": "no vector-map opcode for this dtype"},
        )
    )["lowering_coverage"]
    assert census["unjustified_host_operations"] == []
    assert census["admitted"] is True


def test_a_blank_reason_does_not_count_as_a_justification() -> None:
    from merlin.targetgen import capsule_grade as CG

    census = CG.model_execution_check(
        _ledger_result(
            {"symbol": "k3", "lane": "host_fallback", "mesh_decline": "   "},
        )
    )["lowering_coverage"]
    assert census["unjustified_host_operations"] == ["k3"]


def test_the_placement_census_reaches_the_row_the_agent_reads() -> None:
    """Computed-and-discarded is the state this wiring ends; assert the redaction keeps it."""
    qa = _load_harness("qa_check")
    row = qa._placement_coverage(
        {
            "model_execution_check": {
                "lowering_coverage": {
                    "operations": 3,
                    "on_accelerator": 1,
                    "on_host": 2,
                    "coverage": 0.333333,
                    "host_reasons": [{"reason": "no vector-map opcode", "operations": 1}],
                    "unjustified_host_operations": ["k2"],
                    "by_family": {"on_mesh": {"accelerator": 1, "host": 0}},
                    "licence": "...",
                    "blocking_operations": ["k2"],
                }
            }
        }
    )
    assert row["unjustified_host_operations"] == ["k2"]
    assert row["on_host"] == 2


def test_an_operator_capsule_reports_no_census_rather_than_an_empty_one() -> None:
    """Absent and zero are different facts: one row had nothing to look at, the other looked."""
    qa = _load_harness("qa_check")
    assert qa._placement_coverage({}) is None
    assert qa._placement_coverage({"model_execution_check": {"lowering_coverage": {"operations": 0}}}) is None


# --------------------------------------------------------------------------------------------
# cost_terms -- the resource trade in the agent's own brief
# --------------------------------------------------------------------------------------------


def _brief(analysis):
    from merlin.perf.agent_guidance import PackageOptimizationInventory, guidance_for_emission_analysis

    return guidance_for_emission_analysis(analysis, PackageOptimizationInventory(symbols=(), surfaces=()))


def _arms(before, after):
    return {
        "arms": {
            "baseline": {"macs": before[0], "movement": {"known_bytes": before[1]}},
            "candidate": {"macs": after[0], "movement": {"known_bytes": after[1]}},
        }
    }


def test_the_brief_calls_a_trade_a_trade_and_not_a_regression() -> None:
    """An edit that spends one resource to buy another is a BET, and every other finding in the
    brief reads it as a loss on whichever term it spent."""
    brief = _brief(_arms((2_000_000, 1_000), (1_000_000, 4_000)))
    trade = [row for row in brief["ranked_actions"] if row["kind"] == "resource_trade"]
    assert trade, "a candidate that cut MACs and raised movement produced no trade finding"
    assert trade[0]["evidence"]["terms_increased"] == ["declared_movement_bytes"]


def test_the_composite_is_withheld_while_a_moved_term_has_no_rate() -> None:
    """Weighting an unpriced term as zero hides a resource the edit actually spent."""
    cost = _brief(_arms((2_000_000, 1_000), (1_000_000, 4_000)))["resource_cost"]
    assert cost["composite_status"] == "unpriced"
    assert cost["composite_cycle_delta"] is None
    assert set(cost["unpriced_moved_terms"]) == {"contraction_macs", "declared_movement_bytes"}


def test_a_composite_appears_once_every_moved_term_carries_a_measured_rate() -> None:
    analysis = _arms((2_000_000, 1_000), (1_000_000, 4_000))
    analysis["resource_rates"] = {
        "contraction_macs": {"cycles_per_unit": 0.01, "status": "measured", "provenance": "run r0"},
        "declared_movement_bytes": {
            "cycles_per_unit": 0.5,
            "status": "derived",
            "provenance": "bus width from the target's own facts",
        },
    }
    cost = _brief(analysis)["resource_cost"]
    assert cost["composite_status"] == "priced"
    assert cost["composite_cycle_delta"] == pytest.approx(-1_000_000 * 0.01 + 3_000 * 0.5)


def test_a_rate_with_no_provenance_is_rejected_by_name_not_used_and_not_dropped() -> None:
    """A rate without evidence silently decides which optimizations the loop pursues; a rate
    silently dropped is indistinguishable from one nobody declared."""
    analysis = _arms((2_000_000, 1_000), (1_000_000, 4_000))
    analysis["resource_rates"] = {
        "contraction_macs": {"cycles_per_unit": 0.01, "status": "measured", "provenance": "  "}
    }
    cost = _brief(analysis)["resource_cost"]
    assert cost["rates_rejected"] == ["contraction_macs"]
    assert cost["composite_status"] == "unpriced"


# --------------------------------------------------------------------------------------------
# mesh_occupancy -- the array's issue time, in the envelope's own vocabulary
# --------------------------------------------------------------------------------------------


def test_array_issue_time_charges_the_partial_block_the_sequencer_pays() -> None:
    """A tile narrower than the array wastes array columns for the whole tile; that asymmetry is
    the entire reason operand orientation matters and it does not survive a division."""
    from merlin.perf.envelope import array_issue_time

    time = array_issue_time(
        [{"rows": 64, "depth": 16, "cols": 4}],
        array_rows=16,
        array_cols=16,
        resource="array",
        provenance="a declared tiling",
    )
    assert time.known and time.cycles == 64.0
    assert time.unit == "issue_cycles"


@pytest.mark.parametrize("rows,cols", [(None, 16), (16, None), (0, 16)])
def test_array_issue_time_is_unknown_when_the_geometry_was_not_derived(rows, cols) -> None:
    """The array is a PARAMETER: an unresolved dimension is UNKNOWN with a reason, never a default
    array that would produce a plausible, precise, wrong number."""
    from merlin.perf.decompose import is_unknown
    from merlin.perf.envelope import array_issue_time

    time = array_issue_time(
        [{"rows": 8, "depth": 8, "cols": 8}], array_rows=rows, array_cols=cols, resource="array", provenance="p"
    )
    assert is_unknown(time.cycles)
    assert "array geometry" in time.reason and "UNKNOWN" in time.reason


def test_an_unreadable_tile_makes_the_time_unknown_rather_than_smaller() -> None:
    """Dropping it would make the metric improve as the caller understood less of its own program."""
    from merlin.perf.decompose import is_unknown
    from merlin.perf.envelope import array_issue_time

    time = array_issue_time(
        [{"rows": 16, "depth": 16, "cols": 16}, {"rows": 1}],
        array_rows=16,
        array_cols=16,
        resource="array",
        provenance="p",
    )
    assert is_unknown(time.cycles)
    assert "could not be read" in time.reason


def _load_harness(name: str):
    """Load a capsule-bench harness script by path -- they are scripts, not installed modules."""
    path = feedback_source(name, merlin_dir() / "experiments" / "capsule_bench" / "harness" / f"{name}.py")
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))
