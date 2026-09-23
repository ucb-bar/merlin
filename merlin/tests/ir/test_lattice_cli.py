"""Explicit SMT output admission without invoking a solver or hardware."""

import pytest

from merlin.verify import lattice


@pytest.mark.parametrize("args", [["--emit-counterexamples"], ["--smt-profile", "output.yaml"]])
def test_incomplete_output_arguments_refuse_before_sweep(monkeypatch, args):
    monkeypatch.setattr(lattice, "sweep", lambda *a, **k: pytest.fail("sweep ran before admission"))
    with pytest.raises(SystemExit) as exc:
        lattice.main(["--target", "fixture", *args])
    assert exc.value.code == 2


def test_directory_output_refuses_before_sweep(monkeypatch, tmp_path):
    monkeypatch.setattr(lattice, "sweep", lambda *a, **k: pytest.fail("sweep ran before admission"))
    with pytest.raises(SystemExit) as exc:
        lattice.main(["--target", "fixture", "--emit-counterexamples", "--smt-profile", str(tmp_path)])
    assert exc.value.code == 2


def test_cli_forwards_declared_output(monkeypatch, tmp_path):
    record = {"points_refuted": 1}
    observed = []
    monkeypatch.setattr(lattice, "sweep", lambda *a, **k: record)
    monkeypatch.setattr(lattice, "emit_counterexamples", lambda rec, *, profile: observed.append((rec, profile)))
    output = tmp_path / "declared.yaml"
    assert lattice.main(["--target", "fixture", "--json", "--emit-counterexamples", "--smt-profile", str(output)]) == 1
    assert observed == [(record, output)]


def test_no_refutation_does_not_create_sidecar(tmp_path):
    output = tmp_path / "absent" / "declared.yaml"
    lattice.emit_counterexamples({"results": [{"status": "unsat"}]}, profile=output)
    assert not output.parent.exists()


def test_refutation_writes_only_selected_sidecar(monkeypatch, tmp_path):
    import yaml

    from merlin.verify import counterexamples

    evidence = []
    monkeypatch.setattr(counterexamples, "write_evidence", lambda target, records: evidence.append((target, records)))
    output = tmp_path / "selected" / "shapes.yaml"
    result = {"status": "sat", "m": 2, "k": 3, "n": 4, "cell": "contraction/i8"}
    lattice.emit_counterexamples(
        {"target": "fixture", "results": [result], "lattice_source": "declared-facts"}, profile=output
    )
    document = yaml.safe_load(output.read_text())
    assert document["provenance"]["lattice_source"] == "declared-facts"
    assert document["capsules"][0]["name"] == "CX_contraction_i8_2x3x4"
    assert evidence == [("fixture", [result])]
