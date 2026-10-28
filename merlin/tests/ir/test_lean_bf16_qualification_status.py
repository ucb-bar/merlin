"""A qualification runner cannot report success without executed expected cases."""

import json

import lean_bf16_policy_qualification as qualification
import pytest


@pytest.mark.parametrize("case", [None, "unparsed_positive"])
def test_incomplete_qualification_batch_fails(tmp_path, monkeypatch, case):
    root = tmp_path / "repo"
    lean = root / "merlin" / "lean"
    lean.mkdir(parents=True)
    (lean / "SPEC_LOCK").write_text("{}")
    monkeypatch.setattr(qualification, "repo_root", lambda: root)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    checker = tmp_path / "checker"
    checker.write_bytes(b"unused checker fixture")
    rows = (
        []
        if case is None
        else [
            {
                "name": case,
                "expected": "raw_policy_match",
                "sites": 1,
                "before": {"export": None},
                "after": {"export": None},
            }
        ]
    )
    (inputs / "MATRIX.json").write_text(json.dumps({"cases": rows}))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="qualification"):
        qualification.checks(output, inputs, checker)
    assert (output / "SOURCE-CUT.json").is_file()
    if case is not None:
        result = json.loads((output / "RESULT.json").read_text())
        assert result["cases"][0]["outcome"] == "AUTHORITATIVE_MLIR_REFUSAL_NOT_LEAN_TEST"
