import hashlib
import json

import pytest

from merlin_experiments.tracking import operator_comparison as C


def spec_for(path, *, mode="declared"):
    reference = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"schema": C.SCHEMA, "visibility": "operator_private", "profiles": [
        {"id": "control", "label": "Control", "role": "oracle",
         "documents": [{"name": "Options", "stage": "Scheduling", "mode": mode,
                        "reference": reference, "pointer": "/options"}]},
        {"id": "candidate", "label": "Candidate", "role": "experiment", "documents": []},
    ]}


def test_private_access_required_before_reading_records(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"options":{"prefetch":true}}')
    spec = spec_for(path)
    path.unlink()
    with pytest.raises(ValueError, match="operator_private"):
        C.comparison(spec)


def test_declared_choices_do_not_become_applied_or_missing_zero(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"options":{"prefetch":true,"retention":false}}')
    report = C.comparison(spec_for(path), operator_private=True)
    assert len(report["matrix"]) == 2
    for row in report["matrix"]:
        assert row["cells"]["control"]["mode"] == "declared"
        assert "candidate" not in row["cells"]
    page = C.render(report)
    assert "Not recorded" in page
    assert "OPERATOR PRIVATE" in page
    assert "not prove that a pass executed" in page


def test_changed_evidence_withholds_old_values(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"options":{"tile":16}}')
    spec = spec_for(path, mode="schedule_recipe")
    path.write_text('{"options":{"tile":32}}')
    report = C.comparison(spec, operator_private=True)
    assert report["matrix"] == []
    assert report["profiles"][0]["records"][0]["data"] is None
    assert report["inventory"][0]["status"] == "identity mismatch; values withheld"


def test_json_pointer_and_html_are_data_not_executable(tmp_path):
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"options": {"a/b": {"~token": "</pre><script>BAD()</script>"}}}))
    spec = spec_for(path)
    spec["profiles"][0]["documents"][0]["pointer"] = "/options/a~1b/~0token"
    report = C.comparison(spec, operator_private=True)
    assert report["matrix"][0]["cells"]["control"]["value"].startswith("</pre>")
    page = C.render(report)
    assert "<script>BAD()" not in page
    assert "&lt;script&gt;BAD()" in page


def test_output_cannot_write_into_author_tree(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"options":{"tile":16}}')
    spec = spec_for(path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec["profiles"][0]["workspace"] = str(workspace)
    specification = tmp_path / "spec.json"
    specification.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="observed author/source tree"):
        C.main(["--operator-private", "--spec", str(specification),
                "--output", str(workspace / "comparison.html")])
    assert not (workspace / "comparison.html").exists()


def test_comparison_module_is_author_withheld():
    from merlin.common.access import author_withheld

    assert author_withheld(C.__name__)


def test_import_inventory_never_executes_candidate_or_claims_pass_use(tmp_path):
    tree = tmp_path / "submission"
    tree.mkdir()
    (tree / "backend.py").write_text(
        "import merlin.compile.scheduling.dense_ready_readout\nraise RuntimeError('must not execute')\n")
    spec = {"schema": C.SCHEMA, "visibility": "operator_private", "profiles": [
        {"id": "candidate", "label": "Candidate", "role": "experiment",
         "imports_root": str(tree), "documents": []}]}
    report = C.comparison(spec, operator_private=True)
    row = next(r for r in report["matrix"] if r["field"] == "merlin.compile.scheduling.dense_ready_readout")
    assert row["cells"]["candidate"]["mode"] == "imported"


def test_detailed_schedule_retains_order_without_flooding_matrix(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"options":["tile","retain","emit"]}')
    spec = spec_for(path, mode="schedule_recipe")
    spec["profiles"][0]["documents"][0]["matrix"] = False
    report = C.comparison(spec, operator_private=True)
    assert report["matrix"] == []
    assert report["profiles"][0]["records"][0]["data"] == ["tile", "retain", "emit"]
