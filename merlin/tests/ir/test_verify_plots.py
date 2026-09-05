"""The verification figures must draw, and must draw what the RECORD says -- not what the code says.

Two separate properties are checked here, because they fail independently:

* the figures generate at all (a plotting module that raises is a figure nobody regenerates); and
* every number and every target name on the canvas came out of the record at plot time. That is
  tested by feeding the plotters a record full of values that appear nowhere in the repo and looking
  for those values in the rendered text. A figure with a measurement typed into it would still draw
  perfectly, and would silently report last month's numbers forever -- which is exactly the failure
  this suite exists to prevent, so it is asserted rather than reviewed.

A companion source check keeps a target name or a measured constant from being pasted back into the
plotting module later, where the synthetic-record test alone might not notice it.
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import merlin_dir

pytest.importorskip("matplotlib")

from merlin.verify import plots  # noqa: E402


# --- synthetic records: every value below is deliberately absurd, so finding it on the canvas ------
# proves it was read, and NOT recognisable as any measurement this repo has ever produced.

_MAGIC_STATIC_SECONDS = 0.7654321
_MAGIC_FORMAL_SECONDS = 111.222333
_MAGIC_TARGET = "zz_synthetic_device"


def _fake_detection() -> dict:
    faults = [
        {"name": "zz_fault_alpha", "summary": "synthetic", "expected": ["static"]},
        {"name": "zz_fault_beta", "summary": "synthetic", "expected": ["formal"]},
    ]
    detections = []
    for fault in faults:
        for layer, seconds in (("static", _MAGIC_STATIC_SECONDS), ("formal", _MAGIC_FORMAL_SECONDS),
                               ("dynamic", 0.5)):
            detections.append({
                "fault": fault["name"], "layer": layer,
                "detected": layer in fault["expected"], "seconds": seconds, "diagnostic": "",
            })
    return {
        "schema": plots.DETECTION_SCHEMA,
        "shape": {"m": 3, "k": 5, "n": 7, "reuse": 9},
        "layers": ["static", "formal", "dynamic"],
        "layers_not_measured": {"rtl": "synthetic reason"},
        "false_positives": [{"fault": "<none: unmutated>", "layer": l, "detected": False,
                             "seconds": 0.1, "diagnostic": ""}
                            for l in ("static", "formal", "dynamic")],
        "detections": detections,
        "faults": faults,
    }


def _fake_coverage() -> dict:
    return {
        "schema": plots.COVERAGE_SCHEMA,
        "targets": [{
            "target": _MAGIC_TARGET, "obligations_declared": 7, "emitted": 3, "omitted": 4,
            "omission_reasons": [{"obligation": "zz_obligation", "reason": "zz synthetic reason"}],
            "checks": [], "mesh_edge": {"value": 64, "derived": True},
        }],
        "declared_total": 7, "emitted_total": 3, "omitted_total": 4,
        "baseline": {"emitted": 0, "source": "zz synthetic baseline"},
    }


def _fake_scaling() -> dict:
    return {
        "schema": plots.SCALING_SCHEMA, "reuse": 9, "timeout_ms": 4000,
        "derived_mesh_edges": {_MAGIC_TARGET: 64},
        "points": [
            {"m": 2, "k": 2, "n": 2, "product": 8, "seconds": 0.25, "status": "unsat",
             "verified": True, "n_outputs": 2, "mesh_tile_for": []},
            {"m": 64, "k": 64, "n": 64, "product": 262144, "seconds": _MAGIC_FORMAL_SECONDS,
             "status": "unsat", "verified": True, "n_outputs": 2,
             "mesh_tile_for": [_MAGIC_TARGET]},
        ],
    }


def _canvas_text(fig) -> str:
    """Every string the figure would print: axes text, tick labels, titles, legend and captions.

    Whitespace is collapsed, because captions are line-wrapped for layout and a phrase that happens
    to straddle a wrap point is still a phrase the reader sees.
    """
    parts = (t.get_text() for t in fig.findobj(match=lambda o: hasattr(o, "get_text")))
    return " ".join(" ".join(parts).split())


# --- the figures draw --------------------------------------------------------------------------

def test_all_four_figures_generate(tmp_path):
    records = {plots.DETECTION_FILE: _fake_detection(),
               plots.COVERAGE_FILE: _fake_coverage(),
               plots.SCALING_FILE: _fake_scaling()}
    written = plots.draw_all(records, tmp_path, formats=("png",))
    stems = {stem for stem, _, _ in plots.FIGURES}
    assert {p.stem for p in written} == stems
    for path in written:
        assert path.stat().st_size > 5_000, f"{path} is too small to be a real figure"


# --- the figures report the RECORD, not the code -------------------------------------------------

def test_detection_matrix_reads_the_record():
    fig = plots.fig_detection_matrix(_fake_detection())
    text = _canvas_text(fig)
    assert "zz_fault_alpha" in text and "zz_fault_beta" in text
    assert "3x5x7" in text, "the shape must come from the record"
    # 0.7654321 s renders as 765.4 ms; the point is that the digits are the record's.
    assert "765.4" in text and "111.22" in text


def test_detection_matrix_shows_the_unmeasured_layer_rather_than_dropping_it():
    """An unrun layer that simply vanished from the figure would read as 'nothing to see here'."""
    fig = plots.fig_detection_matrix(_fake_detection())
    text = _canvas_text(fig).lower()
    assert "rtl" in text and "not measured" in text
    assert "synthetic reason" in text, "the reason it was not measured must be carried through"


def test_cost_to_detect_reads_the_record():
    fig = plots.fig_cost_to_detect(_fake_detection())
    text = _canvas_text(fig).lower()
    assert "not measured" in text and "synthetic reason" in text
    assert "111.22" in text, "the expensive layer's measured mean must come from the record"


def test_obligation_coverage_reads_the_record():
    fig = plots.fig_obligation_coverage(_fake_coverage())
    text = _canvas_text(fig)
    assert _MAGIC_TARGET in text, "target names must come from the data, never from the code"
    assert "3 of 7" in text
    assert "zz synthetic reason" in text, "each omission's reason belongs in the caption"


def test_formal_scaling_annotates_the_derived_mesh_tile():
    fig = plots.fig_formal_scaling(_fake_scaling())
    text = _canvas_text(fig)
    assert "64x64x64" in text and _MAGIC_TARGET in text
    assert "edge 64" in text, "the tile edge must be the DERIVED one from the record"


def test_a_shape_the_solver_did_not_settle_is_not_reported_as_verified():
    """A fast 'unknown' is a solver giving up. If the figure counted it as verified, the scaling
    curve would advertise the cheapest possible lie."""
    rec = _fake_scaling()
    rec["points"][0]["status"] = "unknown"
    rec["points"][0]["verified"] = False
    text = _canvas_text(plots.fig_formal_scaling(rec))
    assert "1/2 verified" in text and "unknown" in text


# --- and nothing is baked into the plotting module ------------------------------------------------

def _plots_source() -> str:
    return (merlin_dir() / "python" / "merlin" / "verify" / "plots.py").read_text(encoding="utf-8")


def test_no_measurement_is_hardcoded_in_the_plotting_module():
    """Values that came off a real run must not appear as literals; they belong in the records."""
    source = _plots_source()
    latest = _real_records()
    baked = [tok for tok in _measured_tokens(latest) if tok in source]
    assert not baked, f"measured values pasted into the plotting code: {baked}"


def test_no_target_name_is_hardcoded_in_the_plotting_module():
    """Same rule the repo-wide gate enforces, asserted here where the figures label their axes."""
    source = _plots_source()
    root = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    assert names, "no capsule targets found; this test would be vacuous"
    found = [n for n in names if n in source]
    assert not found, f"target names in library code: {found}"


def _real_records() -> dict:
    """The most recent collected records, if any. Absent records make the check vacuous, so say so."""
    from merlin.common.paths import artifacts_dir

    latest = artifacts_dir() / "verification" / "v1" / "latest"
    out = {}
    for name in (plots.DETECTION_FILE, plots.COVERAGE_FILE, plots.SCALING_FILE):
        path = latest / name
        if path.is_file():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _measured_tokens(records: dict) -> list[str]:
    """Rendered forms of the measured numbers -- the shapes a hardcoded value would actually take."""
    tokens: list[str] = []
    det = records.get(plots.DETECTION_FILE)
    if det:
        tokens += [f"{d['seconds']:.4f}" for d in det["detections"]]
    scaling = records.get(plots.SCALING_FILE)
    if scaling:
        tokens += [f"{p['seconds']:.4f}" for p in scaling["points"]]
    return [t for t in tokens if float(t) != 0.0]


def test_the_internal_note_is_derived_from_the_records_not_typed():
    """The candid note must be generated, and must carry the shape and bound it describes.

    The hand-written predecessor claimed a mesh-tile figure that matched no committed record and was
    2x optimistic against the freshest one, and it lived in a superseded directory whose records were
    never written. Both failures are structural, so the fix is structural: derive the note.
    """
    from merlin.verify.plots import internal_note, DETECTION_FILE, COVERAGE_FILE, SCALING_FILE

    det = {
        "schema": "verify_detection_matrix/v2",
        "shape": {"m": 7, "k": 9, "n": 11, "reuse": 3},
        "timeout_ms": 41_000,
        "layers": ["static", "formal"],
        "layers_not_measured": {},
        "false_positives": [],
        "detections": [
            {"fault": "only_static", "layer": "static", "detected": True, "seconds": 0.1,
             "diagnostic": "", "outcome": "detected"},
            {"fault": "only_static", "layer": "formal", "detected": False, "seconds": 0.2,
             "diagnostic": "unsat", "outcome": "clean"},
            {"fault": "gave_up", "layer": "static", "detected": False, "seconds": 0.1,
             "diagnostic": "", "outcome": "clean"},
            {"fault": "gave_up", "layer": "formal", "detected": False, "seconds": 9.9,
             "diagnostic": "timeout", "outcome": "abstained"},
        ],
        "faults": [{"name": "only_static", "summary": "", "expected": []},
                   {"name": "gave_up", "summary": "", "expected": []}],
    }
    cov = {"declared_total": 13, "emitted_total": 5, "omitted_total": 8,
           "baseline": {"emitted": 0, "source": "nothing consumed them"}, "targets": []}
    sca = {"reuse": 3, "timeout_ms": 41_000, "derived_mesh_edges": {},
           "points": [{"m": 7, "k": 7, "n": 7, "product": 343, "seconds": 1.25, "status": "unsat",
                       "verified": True, "n_outputs": 1, "mesh_tile_for": ["some_target"]}]}

    note = internal_note({DETECTION_FILE: det, COVERAGE_FILE: cov, SCALING_FILE: sca})

    # values that appear nowhere in the repo must appear in the note -- i.e. it was READ, not typed
    assert "7x9x11" in note, "the note does not carry the shape it describes"
    assert "41000" in note, "the note does not carry the solver bound"
    assert "5 of 13" in note, "the note does not carry the derived coverage"
    assert "1.25 s" in note, "the note does not carry the measured mesh-tile time"
    # an abstention must not be counted as a static-only catch
    assert "ALONE: **1**" in note, "abstention leaked into the caught-alone count"
    assert "only_static" in note and "abstentions: **1**" in note
