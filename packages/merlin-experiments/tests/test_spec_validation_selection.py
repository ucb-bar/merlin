"""The spec-validation selection keeps every case identity and writes nothing else."""

import hashlib
import json
import runpy
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root

SCRIPT = repo_root() / "build_tools/scripts/make_spec_validation_selection.py"


@pytest.fixture
def writer():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture
def suites(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"cases": [{"overload": f"aten.op{n}.default"} for n in (2, 0, 1)]}))
    bounded = tmp_path / "bounded.json"
    cases = [dict(case_id=f"aten.op{n % 3}.default::{n}", overload=f"aten.op{n % 3}.default") for n in range(7)]
    bounded.write_text(
        json.dumps({"selected_cases": cases[::-1], "denominator_sha256": "d" * 64, "suite_sha256": "s" * 64})
    )
    return {"corpus_193": corpus, "bounded": bounded}


def test_selection_retains_every_case_identity_and_binds_suite_bytes(writer, suites):
    document = writer["build"](suites)
    assert document["schema"] == "merlin.phase0.spec_validation_selection.v1"
    corpus, bounded = document["suites"]["corpus_193"], document["suites"]["bounded"]
    assert corpus["case_ids"] == [f"aten.op{n}.default" for n in range(3)]
    assert bounded["case_ids"] == sorted(f"aten.op{n % 3}.default::{n}" for n in range(7))
    assert (corpus["case_count"], bounded["case_count"], document["case_count"]) == (3, 7, 10)
    assert bounded["denominator_sha256"] == "d" * 64 and bounded["suite_sha256"] == "s" * 64
    assert corpus["denominator_sha256"] is None and corpus["suite_sha256"] is None
    assert bounded["file_sha256"] == hashlib.sha256(suites["bounded"].read_bytes()).hexdigest()
    assert document["selection_sha256"] == writer["selection_digest"](document)


def test_selection_carries_no_lanes_cohorts_recipe_or_descriptor(writer, suites):
    document = writer["build"](suites)
    assert set(document) == {"schema", "suites", "case_count", "selection_sha256"}
    for item in document["suites"].values():
        assert set(item) == {"file_sha256", "denominator_sha256", "suite_sha256", "case_count", "case_ids"}
    assert "lane" not in json.dumps(document) and "cohort" not in json.dumps(document)


def test_selection_digest_is_stable_and_path_independent(writer, suites, tmp_path):
    first = writer["build"](suites)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for source, path in list(suites.items()):
        suites[source] = moved / path.name
        suites[source].write_bytes(path.read_bytes())
    assert writer["build"](suites) == first
    suites["bounded"].write_bytes(suites["bounded"].read_bytes() + b"\n")
    assert writer["build"](suites)["selection_sha256"] != first["selection_sha256"]


def test_selection_refuses_duplicate_or_empty_suites(writer, tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"selected_cases": [{"case_id": "a", "overload": "x"}] * 2}))
    with pytest.raises(ValueError, match="duplicate case identities"):
        writer["build"]({"bounded": duplicate})
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"cases": []}))
    with pytest.raises(ValueError, match="non-empty"):
        writer["build"]({"corpus_193": empty})
    with pytest.raises(ValueError, match="at least one suite"):
        writer["build"]({})
    with pytest.raises(ValueError, match="at least one suite"):
        writer["build"]({"public": duplicate})


def test_cli_writes_only_the_selection_file_and_never_replaces_it(suites, tmp_path):
    output = tmp_path / "out" / "selection.json"
    output.parent.mkdir()
    argv = [sys.executable, str(SCRIPT), "--corpus-193", str(suites["corpus_193"]), "--bounded", str(suites["bounded"])]
    subprocess.run([*argv, "--output", str(output)], check=True, capture_output=True)
    assert [path.name for path in output.parent.iterdir()] == ["selection.json"]
    document = json.loads(output.read_bytes())
    assert document["case_count"] == 10
    before = output.read_bytes()
    again = subprocess.run([*argv, "--output", str(output)], capture_output=True)
    assert again.returncode != 0 and output.read_bytes() == before
