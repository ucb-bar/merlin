"""No native checking: exact versioned declaration/audit separation."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from merlin.verify import lean_graph_loops_shape_after_declarations as emission
from merlin.verify.lean_graph_loops_shape_after_declarations import separate_audits


def test_all_retained_queries_preserve_declarations_and_root_rosters(tmp_path):
    selected = os.environ.get("MERLIN_GRAPH_SHAPE_AFTER_PACKET")
    if not selected:
        pytest.skip("select the immutable source-candidate-v1 query inventory")
    root = Path(os.environ["MERLIN_GRAPH_SHAPE_AFTER_REPO"])
    records = json.loads((Path(selected) / "QUERIES.json").read_text())
    assert len(records) == 25
    for index, row in enumerate(records):
        original = (root / row["path"]).read_bytes()
        assert hashlib.sha256(original).hexdigest() == row["sha256"]
        converted = separate_audits(original.decode(), row["support_roots"], final_root=row["final_root"])
        # Only known command lines may disappear. Everything proving the
        # before/after relation, including bad candidates, remains byte-exact.
        expected = b"".join(
            line for line in original.splitlines(keepends=True) if line.startswith(b"#print axioms ") is False
        )
        assert converted["query"].encode() == expected
        assert not any(line.lstrip().startswith("#") for line in converted["query"].splitlines())
        assert set((converted["final_root"], *converted["support_roots"])) == set(row["support_roots"])
        assert converted["final_root"] not in converted["support_roots"]
        assert converted["legacy_query_sha256"] == row["sha256"]
        assert converted["query_sha256"] != row["sha256"]
        assert not any(
            converted[key] for key in ("lean_invoked", "lean_accepted", "qualified", "pipeline_receipt_issued")
        )
        (tmp_path / f"Case{index:02d}.lean").write_text(converted.pop("query"))
        (tmp_path / f"Case{index:02d}.json").write_text(json.dumps(dict(converted, legacy=row), indent=2) + "\n")
        assert (root / row["path"]).read_bytes() == original


@pytest.mark.parametrize("command", ("#check proof", "#eval 1", "#print axioms unexpected", " #print axioms proof"))
def test_unexpected_command_is_not_silently_removed(command):
    with pytest.raises(ValueError, match="unexpected generated query command"):
        separate_audits("theorem proof : True := .intro\n" + command + "\n", ["proof"], final_root="proof")


def test_missing_root_audit_is_an_emitter_contract_error():
    with pytest.raises(ValueError, match="do not cover"):
        separate_audits("theorem proof : True := .intro\n", ["proof"], final_root="proof")


@pytest.mark.parametrize("roots,final", ((["proof", "proof"], "proof"), (["proof"], "absent")))
def test_invalid_roster_is_refused(roots, final):
    with pytest.raises(ValueError, match="distinct complete root roster"):
        separate_audits("#print axioms proof\n", roots, final_root=final)


def test_escaped_metadata_is_preserved_and_repeated_expected_audits_are_deduplicated():
    literal = 'def metadata := "line\\n#print axioms opaque"\n'
    result = separate_audits(literal + "#print axioms proof\n#print axioms proof\n", ["proof"], final_root="proof")
    assert result["query"] == literal
    assert result["support_roots"] == []


@pytest.mark.parametrize("family", ("pair", "chain"))
def test_preparation_wrappers_keep_data_and_cannot_issue_acceptance(monkeypatch, family):
    root = emission.PAIR_ROOT if family == "pair" else emission.EPOCH_ROOT
    legacy = dict(query="def input := 4\n#print axioms " + root + "\n", expected_roots=[root], model={"sentinel": 9})
    seen = []

    def prepare(*args, **kwargs):
        seen.append((args, kwargs))
        return legacy

    monkeypatch.setattr(emission, "prepare_" + family + "_v1", prepare)
    result = getattr(emission, "prepare_" + family)("source", index_bits=32)
    assert seen == [(("source",), {"index_bits": 32})]
    assert result["model"] == legacy["model"]
    assert result["query"] == "def input := 4\n"
    assert result["final_root"] == root
    assert result["expected_compile_stdout"] == ""
    assert not result["lean_invoked"] and not result["pipeline_receipt_issued"]
    assert "#print" in legacy["query"]  # The historical result was not mutated.
