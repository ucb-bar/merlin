"""Data/source-association and diagnostic controls; no mocked positive proofs."""

import hashlib
import json
from copy import deepcopy

import pytest
from configuration_elision_qualification_classifier import OBLIGATION, classify
from configuration_elision_qualification_review import audit_decoding


def decoded_fixture():
    source, final = "before", "after"
    effect = {"reads": [], "writes": [[0, 3]], "clobbers": [], "ordered": True, "pureWrite": True}
    atom = {
        "effect": effect,
        "eraseAllowed": True,
        "decoded": {"instruction": "set", "arguments": {"slot": 0, "a": 3}, "word": 19},
    }
    entry = {"name": "run", "code": [atom]}
    projection = {
        "fields": ["configuration:0"],
        "before": {"entries": [entry]},
        "after": {"entries": [deepcopy(entry)]},
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "final_sha256": hashlib.sha256(final.encode()).hexdigest(),
    }
    row = {
        "instruction": "set",
        "bindings": {"slot": 0, "a": 3},
        "encoded_word": 19,
        "effect": {
            "reads": [],
            "writes": [["configuration:0", 3]],
            "clobbers": [],
            "ordered": True,
            "pure_write": True,
        },
    }
    producer = {
        "source_decoding": [{"function": "run", "operations": [row]}],
        "final_decoding": [{"function": "run", "operations": [deepcopy(row)]}],
    }
    return projection, producer, source, final


@pytest.mark.parametrize(
    "mutation", ["source", "final", "roster", "duplicate_field", "word", "value", "negative", "boolean"]
)
def test_independent_decoding_audit_refuses_wrong_associations(mutation):
    projection, producer, source, final = decoded_fixture()
    if mutation in {"source", "final"}:
        projection[mutation + "_sha256"] = "0" * 64
    elif mutation == "roster":
        projection["before"]["entries"] = []
    elif mutation == "duplicate_field":
        projection["fields"] *= 2
    elif mutation == "word":
        projection["before"]["entries"][0]["code"][0]["decoded"]["word"] += 1
    else:
        pair = projection["before"]["entries"][0]["code"][0]["effect"]["writes"][0]
        pair[0 if mutation in {"negative", "boolean"} else 1] = {"negative": -1, "boolean": False, "value": 5}[mutation]
    with pytest.raises(ValueError):
        audit_decoding(projection, producer, source, final)


def diagnostic(tmp_path, *, text=None, returncode=1, stderr=""):
    path = tmp_path / "ConfigurationElisionQuery.lean"
    query = "namespace ActualConfigurationElision\n" + OBLIGATION + "\nend ActualConfigurationElision\n"
    path.write_text(query)
    (tmp_path / "query.process.json").write_text(json.dumps({"command": ["lean", str(path)], "returncode": returncode}))
    (tmp_path / "query.stdout").write_text(
        text
        if text is not None
        else str(path) + ":2:75: error: Tactic `decide` proved that the proposition\n  False\nis false\n"
    )
    (tmp_path / "query.stderr").write_text(stderr)
    return query


@pytest.mark.parametrize(
    "marker", ["heartbeat", "maximum recursion depth", "out of memory", "failed to import", "panic"]
)
def test_resource_failures_cannot_be_counted_as_expected_refusals(tmp_path, marker):
    query = diagnostic(tmp_path)
    with (tmp_path / "query.stdout").open("a") as stream:
        stream.write(marker)
    assert classify(tmp_path, query)["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("code", [0, 2, -11, None])
def test_nonordinary_exits_cannot_be_counted_as_expected_refusals(tmp_path, code):
    query = diagnostic(tmp_path, returncode=code)
    assert classify(tmp_path, query)["status"] == "UNAVAILABLE"


def test_only_actual_bound_closed_false_diagnostic_counts(tmp_path):
    query = diagnostic(tmp_path)
    assert classify(tmp_path, query)["status"] == "ACTUAL_EXPECTED_LEAN_OBLIGATION_REFUSAL"
    assert classify(tmp_path, query + "\n")["status"] == "UNAVAILABLE"
