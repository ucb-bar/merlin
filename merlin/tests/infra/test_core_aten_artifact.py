"""Tamper and claim-boundary checks for saved operator-coverage evidence."""
from __future__ import annotations

import csv
import importlib.util
import io
import json

import pytest

from merlin.common.paths import repo_root

spec = importlib.util.spec_from_file_location(
    "verify_core_aten_artifact", repo_root() / "build_tools/scripts/verify_core_aten_artifact.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


def rebind(root, manifest, name):
    path = root / name
    manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": audit.digest(path)}


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "catalog.json").write_text(json.dumps([
        {"n": "0", "overload": "aten.mm.default", "klass": "DEVICE57", "form_lane": "integer"},
        {"n": "1", "overload": "aten.sin.default", "klass": "HOST136", "form_lane": ""},
    ]))
    cases = [
        {"overload": "mm.default", "klass": "DEVICE57", "ok": "PASS", "result": "lowered-on-device",
         "device_insns": "8", "numerical": "pass", "host_compute": "pass", "trace_check": "pass", "meets_full": "YES"},
        {"overload": "sin.default", "klass": "HOST136", "ok": "FAIL", "result": "declined",
         "device_insns": "0", "numerical": "", "host_compute": "", "trace_check": "", "meets_full": "NO"},
    ]
    write_csv(root / "cases.csv", cases)
    table = [dict(r, proven=("YES" if r["klass"] == "DEVICE57" else "NO"),
                  credited="2", credited_graded="1", credited_pass="1", credited_fail="0", credited_not_graded="1") for r in cases]
    write_csv(root / "table.csv", table)
    write_csv(root / "credits.csv", [{"overload": "aten.mm.default", "tier": "tier1", "tier1_capsules": "checked;probe"}])
    write_csv(root / "corpus.csv", [
        {"capsule": "checked", "group": "core", "outcome": "pass"},
        {"capsule": "refusal", "group": "core", "outcome": "fail"},
        {"capsule": "model", "group": "core", "outcome": "unavailable"},
    ])
    manifest = {"schema": audit.SCHEMA, "source_commit": "0" * 40,
                "expected_class_counts": {"DEVICE57": 1, "HOST136": 1},
                "catalog": "catalog.json", "table": "table.csv", "case_results": "cases.csv",
                "credit_map": "credits.csv", "corpus_results": "corpus.csv", "files": {}}
    for path in root.iterdir():
        rebind(root, manifest, path.name)
    return root, manifest


def test_ungraded_probes_host_failures_and_unavailable_models_stay_separate(evidence):
    root, manifest = evidence
    result = audit.validate(root, manifest)
    assert result["device_supported_on_graded_cases"] == 1
    assert result["host_reported_supported"] == 0
    assert result["ungraded_device_credit_slots"] == 1
    assert result["corpus"] == {"core": {"pass": 1, "fail": 1, "unavailable": 1}}
    assert result["device_details"][0]["ungraded_test_names"] == ["probe"]
    assert not result["archived_programs_executed"]


@pytest.mark.parametrize("change", ["edit", "remove", "add", "symlink"])
def test_changed_or_indirect_payload_is_rejected(evidence, change, tmp_path):
    root, manifest = evidence
    if change == "edit":
        (root / "cases.csv").write_text("changed")
    elif change == "remove":
        (root / "cases.csv").unlink()
    elif change == "add":
        (root / "unexpected.py").write_text("raise RuntimeError('never execute')")
    else:
        (root / "cases.csv").rename(tmp_path / "external.csv")
        (root / "cases.csv").symlink_to(tmp_path / "external.csv")
    with pytest.raises(ValueError):
        audit.validate(root, manifest)


@pytest.mark.parametrize("field,value", [("numerical", "fail"), ("trace_check", ""), ("host_compute", "unknown"), ("device_insns", "0")])
def test_full_claim_cannot_survive_missing_execution_checks(evidence, field, value):
    root, manifest = evidence
    rows = list(csv.DictReader((root / "cases.csv").open()))
    rows[0][field] = value
    write_csv(root / "cases.csv", rows)
    rebind(root, manifest, "cases.csv")
    with pytest.raises(ValueError, match="FULL verdict differs"):
        audit.validate(root, manifest)


def test_ungraded_probe_cannot_be_reclassified_as_a_pass(evidence):
    root, manifest = evidence
    rows = list(csv.DictReader((root / "table.csv").open()))
    rows[0].update(credited_pass="2", credited_graded="2", credited_not_graded="0")
    write_csv(root / "table.csv", rows)
    rebind(root, manifest, "table.csv")
    with pytest.raises(ValueError, match="credited test counts differ"):
        audit.validate(root, manifest)


def test_support_requires_at_least_one_graded_credit(evidence):
    root, manifest = evidence
    rows = list(csv.DictReader((root / "credits.csv").open()))
    rows[0]["tier1_capsules"] = "probe"
    write_csv(root / "credits.csv", rows)
    rebind(root, manifest, "credits.csv")
    rows = list(csv.DictReader((root / "table.csv").open()))
    rows[0].update(credited="1", credited_graded="0", credited_pass="0", credited_not_graded="1")
    write_csv(root / "table.csv", rows)
    rebind(root, manifest, "table.csv")
    with pytest.raises(ValueError, match="support verdict differs"):
        audit.validate(root, manifest)


def test_duplicate_operator_rows_are_rejected_even_with_matching_checksums(evidence):
    root, manifest = evidence
    rows = list(csv.DictReader((root / "cases.csv").open()))
    write_csv(root / "cases.csv", [*rows, rows[0]])
    rebind(root, manifest, "cases.csv")
    with pytest.raises(ValueError, match="duplicate overload"):
        audit.validate(root, manifest)


def test_catalog_and_grade_classifications_must_match(evidence):
    root, manifest = evidence
    rows = list(csv.DictReader((root / "cases.csv").open()))
    rows[0]["klass"] = "HOST136"
    write_csv(root / "cases.csv", rows)
    rebind(root, manifest, "cases.csv")
    with pytest.raises(ValueError, match="classification differs"):
        audit.validate(root, manifest)
