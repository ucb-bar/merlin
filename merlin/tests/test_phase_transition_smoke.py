"""Headless smoke test for the phase-transition experiment."""
import csv
import io

from merlin.dse.experiment import CSV_COLUMNS, phase_transition

MINI_AXES = {"H": [1, 2, 8, 32], "reuse_count": [1, 2, 4, 8],
             "dtype": ["i8"], "epilogue": [True]}


def test_phase_transition_writes_csv_with_crossover(tmp_path):
    res = phase_transition(axes=MINI_AXES, out_dir=tmp_path)
    csv_path = tmp_path / "phase_transition.csv"
    assert csv_path.is_file()

    reader = csv.DictReader(io.StringIO(csv_path.read_text()))
    assert reader.fieldnames == CSV_COLUMNS
    rows = list(reader)
    assert rows

    # There must be a crossover: the winning contract is not the same at every horizon.
    best_by_h = {r["H"]: r["contract"] for r in rows if r["best"] == "True"}
    assert len(set(best_by_h.values())) > 1
    # Specifically, opaque wins at H=1 but not at H=32.
    assert best_by_h["1"] == "I0"
    assert best_by_h["32"] != "I0"


def test_exploitability_reports_emitted(tmp_path):
    res = phase_transition(axes=MINI_AXES, out_dir=tmp_path)
    assert (tmp_path / "exploitability_resident_packed_tensor.yaml").is_file()
    assert (tmp_path / "exploitability_accumulator_commit.yaml").is_file()
    assert res["exploitability"]["resident_packed_tensor"]["rows"][0]["exploitability"] == 0.0
