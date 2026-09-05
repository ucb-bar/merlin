"""A per-cycle trace folds into the SAME timing block an in-sim harness emits -- or into nothing.

The defect these cover: an engine that DUMPS the design's activity ports (one row per cycle) instead
of accumulating them inside the simulator returned no ``timing_observations``, so every consumer that
reads occupancy telemetry saw it as an instrument with no timing capability. Adopting a faster engine
then silently cost a perf campaign its per-capsule decomposition -- while the measurement was sitting
in a file the engine had already written.

Each test here is a way the fold could produce a number that is wrong in the flattering direction.
"""
from __future__ import annotations

import json

import pytest

from merlin.perf import cycle_trace as CT
from merlin.perf import observations as OBS

DECL = {
    "schema_version": CT.DECLARATION_SCHEMA,
    "busy_columns": {
        "aComp": {"unit": "a", "kind": "compute"},
        "bMove": {"unit": "b", "kind": "movement"},
    },
    OBS.UNMEASURED_UNITS_KEY: [],
}


def _rows(**cols):
    n = len(next(iter(cols.values())))
    return [{k: str(v[i]) for k, v in cols.items()} for i in range(n)]


def _write(tmp_path, decl, name=CT.DECLARATION_NAME):
    (tmp_path / name).write_text(json.dumps(decl), encoding="utf-8")
    return tmp_path


def test_busy_idle_and_overlap_are_counted_off_the_rows():
    rows = _rows(cycle=[0, 1, 2, 3], aComp=[1, 1, 0, 0], bMove=[0, 1, 0, 0])
    block = CT.block_from_rows(rows, DECL)
    got = {o["quantity"]: o["value"] for o in block[OBS.TIMING_OBSERVATIONS_KEY]}
    assert got["busy_cycles.a.in_program"] == 2
    assert got["busy_cycles.b.in_program"] == 1
    assert got[OBS.IDLE_QUANTITY] == 2
    assert got[OBS.OVERLAP_OBSERVED] == 1            # cycle 1 only
    assert got[OBS.OVERLAP_ACROSS_KINDS] == 1        # and the two are different kinds
    assert got[OBS.SAMPLED_QUANTITY] == 4
    # The buckets reconcile: busy-union + idle == sampled.
    assert (got["busy_cycles.a.in_program"] + got["busy_cycles.b.in_program"]
            - got[OBS.OVERLAP_OBSERVED] + got[OBS.IDLE_QUANTITY]) == got[OBS.SAMPLED_QUANTITY]


def test_the_block_is_a_joint_reading_not_a_partition():
    """``partitioned: false`` is what licenses an overlap reading at all. A partition reports zero
    overlap by construction, so a fold that quietly became marginal must fail loudly here."""
    block = CT.block_from_rows(_rows(aComp=[1], bMove=[1]), DECL)
    assert block[OBS.PARTITIONED_KEY] is False
    validated = OBS.validate_block(dict(block, alias_collisions=0))
    assert validated is not None and validated.overlap_cycles() == 1


def test_columns_naming_one_unit_fold_instead_of_overlapping_with_themselves():
    """A unit's halves nest inside it. Counting them as separate units charges the unit two cycles
    for one and reports it overlapping with itself -- measured on a real engine as 136 cycles of
    fabricated overlap, exactly the LSU's own busy count."""
    decl = dict(DECL, busy_columns={
        "lsuBusy": {"unit": "lsu", "kind": "movement"},
        "vloadBusy": {"unit": "lsu", "kind": "movement"},
        "vstoreBusy": {"unit": "lsu", "kind": "movement"}})
    rows = _rows(lsuBusy=[1, 1], vloadBusy=[1, 0], vstoreBusy=[0, 1])
    got = {o["quantity"]: o["value"] for o in CT.block_from_rows(rows, decl)[OBS.TIMING_OBSERVATIONS_KEY]}
    assert got["busy_cycles.lsu.in_program"] == 2     # not 4
    assert got[OBS.OVERLAP_OBSERVED] == 0             # a unit does not overlap itself
    assert got[OBS.IDLE_QUANTITY] == 0


def test_a_non_boolean_column_is_unmeasured_never_folded_into_idle():
    """State registers are the trap: state 0 is a state, not idleness, and 'nonzero means busy' on
    one is simply false. A column this fold cannot read must leave the counts and be NAMED."""
    decl = dict(DECL, busy_columns=dict(DECL["busy_columns"], st={"unit": "s", "kind": "compute"}))
    rows = _rows(aComp=[0, 0], bMove=[0, 0], st=[2, 3])
    block = CT.block_from_rows(rows, decl)
    got = {o["quantity"]: o["value"] for o in block[OBS.TIMING_OBSERVATIONS_KEY]}
    assert got["busy_cycles.s.in_program"] == 0
    assert "st" in block[OBS.UNMEASURED_UNITS_KEY]
    assert "not boolean" in block["unmeasured_note"]


def test_idle_is_declared_an_upper_bound_when_anything_is_unread():
    """A unit with no port read is a unit that inflates idle. Cross-checked against a second engine
    over the same design, the two agreed on every shared unit and their idle figures differed by
    precisely the unread units' cycles -- so idle must SAY it is a bound."""
    decl = dict(DECL, **{OBS.UNMEASURED_UNITS_KEY: ["dma0"]})
    idle = next(o for o in CT.block_from_rows(_rows(aComp=[0], bMove=[0]), decl)[
        OBS.TIMING_OBSERVATIONS_KEY] if o["quantity"] == OBS.IDLE_QUANTITY)
    assert "UPPER BOUND" in idle["note"]
    exact = next(o for o in CT.block_from_rows(_rows(aComp=[0], bMove=[0]), DECL)[
        OBS.TIMING_OBSERVATIONS_KEY] if o["quantity"] == OBS.IDLE_QUANTITY)
    assert "UPPER BOUND" not in exact["note"]


def test_a_declared_column_absent_from_the_trace_is_unmeasured_not_idle():
    decl = dict(DECL, busy_columns=dict(DECL["busy_columns"], gone={"unit": "g", "kind": "compute"}))
    block = CT.block_from_rows(_rows(aComp=[1], bMove=[0]), decl)
    assert "gone" in block[OBS.UNMEASURED_UNITS_KEY]
    assert "absent from the trace" in block["unmeasured_note"]


@pytest.mark.parametrize("bad", [
    pytest.param({"busy_columns": {"a": {}}, OBS.UNMEASURED_UNITS_KEY: []}, id="no-schema-tag"),
    pytest.param({"schema_version": CT.DECLARATION_SCHEMA, "busy_columns": {},
                  OBS.UNMEASURED_UNITS_KEY: []}, id="no-columns"),
    pytest.param({"schema_version": CT.DECLARATION_SCHEMA, "busy_columns": {"a": {}}},
                 id="no-unmeasured-units"),
    pytest.param({"schema_version": "some.other.v9", "busy_columns": {"a": {}},
                  OBS.UNMEASURED_UNITS_KEY: []}, id="wrong-schema-tag"),
])
def test_an_unusable_declaration_yields_no_capability_rather_than_a_block(tmp_path, bad):
    """None means 'this engine reports no timing', which is the honest answer. `unmeasured_units`
    may not be defaulted: a declaration that does not say what it failed to read is claiming a
    completeness it has not earned."""
    assert CT.load_declaration(_write(tmp_path, bad)) is None


def test_no_declaration_and_no_trace_both_mean_absent_never_zeros(tmp_path):
    assert CT.load_declaration(tmp_path) is None
    assert CT.block_from_trace(tmp_path / "nope.csv", tmp_path) is None
    _write(tmp_path, DECL)
    assert CT.block_from_trace(tmp_path / "nope.csv", tmp_path) is None


def test_a_written_trace_round_trips_through_the_validator(tmp_path):
    _write(tmp_path, DECL)
    (tmp_path / "t.csv").write_text("cycle,aComp,bMove\n0,1,0\n1,1,1\n2,0,0\n", encoding="utf-8")
    block = CT.block_from_trace(tmp_path / "t.csv", tmp_path)
    validated = OBS.validate_block(dict(block, alias_collisions=0))
    assert validated is not None and validated.usable and validated.refusals == ()
    assert validated.busy_by_unit() == {"a": 2, "b": 1}
    assert validated.kinds() == {"a": "compute", "b": "movement"}


def test_the_trace_is_read_by_header_not_by_position(tmp_path):
    """A positional read silently re-binds every unit the day a column is inserted."""
    _write(tmp_path, DECL)
    (tmp_path / "t.csv").write_text("cycle,bMove,aComp\n0,1,0\n", encoding="utf-8")
    got = {o["quantity"]: o["value"]
           for o in CT.block_from_trace(tmp_path / "t.csv", tmp_path)[OBS.TIMING_OBSERVATIONS_KEY]}
    assert got["busy_cycles.b.in_program"] == 1 and got["busy_cycles.a.in_program"] == 0


# ---------------------------------------------------------------------------------------------
# The wiring: an engine that only DUMPS its ports must end up carrying the same block as one that
# accumulates them, and an engine that declares nothing must stay byte-identical to before.
# ---------------------------------------------------------------------------------------------

_RUNNER = '''
import csv
def run_program(words, preload=None, reads=None, max_cycles=20000, timeout=600,
                per_cycle_csv=None):
    if per_cycle_csv:
        with open(per_cycle_csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["cycle", "aComp", "bMove"])
            w.writerow([0, 1, 0]); w.writerow([1, 1, 1]); w.writerow([2, 0, 0])
    return {"halted": True, "cycles": 3, "outputs": [b"\\x00" * 4], "alias_collisions": 0}
'''


def _engine(tmp_path, name, *, declare=True):
    d = tmp_path / name
    d.mkdir()
    (d / "verilator_run.py").write_text(_RUNNER, encoding="utf-8")
    if declare:
        (d / CT.DECLARATION_NAME).write_text(json.dumps(DECL), encoding="utf-8")
    return d


def _run(monkeypatch, tmp_path, engine_dir):
    from merlin.targetgen import program_oracle as PO
    monkeypatch.setattr(PO, "emit_bundle", lambda **kw: {"words": [0]})
    monkeypatch.setattr(PO, "_bundle_preload", lambda bundle, cb: [])
    monkeypatch.setattr(PO, "_resolve_out_specs",
                        lambda t, cb, b: {"Y": {"base": 0, "shape": [1], "dtype": "i32",
                                                "physical": "i32"}})
    monkeypatch.setattr(PO, "_out_nbytes", lambda s: 4)
    monkeypatch.setattr(PO, "_decode_output",
                        lambda raw, shape, dtype, phys: __import__("numpy").zeros(1, dtype="int32"))
    monkeypatch.setattr(PO, "_engine_provenance", lambda *a, **k: {})
    return PO.run_program_verilator_oracle(
        "t", model_ext="", vsim_dir=engine_dir, cb={}, program="p", workdir=tmp_path / "wd")


def test_a_dumping_engine_gets_the_block_folded_from_its_own_trace(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, _engine(tmp_path, "eng"))
    got = {o["quantity"]: o["value"] for o in out["timing_observations"]}
    assert got["busy_cycles.a.in_program"] == 2 and got["busy_cycles.b.in_program"] == 1
    assert out["timing_capability"][OBS.PARTITIONED_KEY] is False
    # The alias accounting the RUNNER reported travels with the folded block, not the fold's silence.
    assert out["timing_capability"][OBS.ALIAS_COLLISIONS_KEY] == 0


def test_an_engine_that_declares_nothing_reports_no_timing_capability(monkeypatch, tmp_path):
    """Absent is not zero: no declaration means the engine is unchanged, not measured as idle."""
    out = _run(monkeypatch, tmp_path, _engine(tmp_path, "eng2", declare=False))
    assert "timing_observations" not in out and "timing_capability" not in out


def test_the_derived_trace_is_cleaned_up_after_it_is_folded(monkeypatch, tmp_path):
    """The fold's temporary costs one row per cycle; leaving it behind would grow without bound."""
    out = _run(monkeypatch, tmp_path, _engine(tmp_path, "eng3"))
    assert out["timing_observations"]
    assert not (tmp_path / "wd" / "per_cycle_trace.csv").exists()
