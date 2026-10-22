import copy
import hashlib
import json

import pytest
from merlin_experiments.phase1 import joint_calibration_review_successor as S
from merlin_experiments.phase1 import joint_calibration_admission as A


def pin(tmp, name, value):
    p = tmp / name
    p.write_text(json.dumps(value))
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def fixture(tmp):
    evidence = pin(tmp, "evidence.json", {"public": "structural residual remains"})
    old = dict(schema="merlin.native_calibration_admission_review.v1", review_id="original",
               deny=[dict(model_sha256="a" * 64, price_paths=[["price"]], reason="unqualified")])
    new = copy.deepcopy(old); new["deny"][0]["evidence"] = [evidence]
    previous = pin(tmp, "previous.json", old)
    replacement = pin(tmp, "replacement.json", new)
    successor = dict(schema=S.SCHEMA, review_id="evidence-only", previous_review=previous,
                     review=replacement, fit_authorized=False, native_continuation_authorized=False)
    selection = dict(environment={A.ENV: json.dumps(previous)}, run_dir="same", policy="same")
    return selection, successor, new, evidence


def test_repaired_review_preserves_denial_and_original_selection(tmp_path, monkeypatch):
    selection, successor, new, evidence = fixture(tmp_path)
    original = copy.deepcopy(selection)
    result = S.select(selection, pin(tmp_path, "successor.json", successor))
    assert selection == original
    assert result["run_dir"] == "same" and result["policy"] == "same"
    assert result["calibration_review_successor"]["fit_authorized"] is False
    monkeypatch.setenv(A.ENV, result["environment"][A.ENV])
    runtime = pin(tmp_path, "runtime.json", dict(model_sha256="a" * 64))
    with pytest.raises(ValueError, match="blocked_structural_model_admission: unqualified"):
        A.assert_allowed(dict(runtime=runtime, parameters=[dict(path=["price"])]))


@pytest.mark.parametrize("damage", ["model", "price", "reason", "remove", "add", "top", "oldpin", "fit", "continuation", "evidence", "overflow"])
def test_successor_cannot_change_authority_or_stale_evidence(tmp_path, damage):
    selection, successor, new, evidence = fixture(tmp_path)
    if damage == "model": new["deny"][0]["model_sha256"] = "b" * 64
    elif damage == "price": new["deny"][0]["price_paths"] = [["other"]]
    elif damage == "reason": new["deny"][0]["reason"] = "qualified"
    elif damage == "remove": new["deny"] = []
    elif damage == "add": new["deny"].append(copy.deepcopy(new["deny"][0]))
    elif damage == "top": new["review_id"] = "changed"
    elif damage == "oldpin": successor["previous_review"] = evidence
    elif damage == "fit": successor["fit_authorized"] = True
    elif damage == "continuation": successor["native_continuation_authorized"] = True
    elif damage == "overflow": new["deny"][0]["evidence"] *= 17
    else:
        from pathlib import Path
        Path(evidence["path"]).write_text("changed")
    successor["review"] = pin(tmp_path, "replacement.json", new)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        S.select(selection, pin(tmp_path, "successor.json", successor))
