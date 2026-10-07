"""``--only-group`` builds a PARTIAL whole-model program, marked so, and every whole-model reader refuses it.

A partial program runs and prints every group line (the others are the target's library calls), so
nothing in its log says it is not the candidate's whole model. These hold that the marker is put where
each reader looks and that each reader -- the verdict, the grade, the gate, the service builder -- says no.
Nothing here names a target: the builder's own collaborators are replaced where a real one would need a
target's toolchain.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from merlin.perf import whole_model_build as W
from merlin.perf import whole_model_partial as P

_GROUPS = [0, 1, 2, 3, 4]


class _Stated(Exception):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        super().__init__("stated")


@pytest.fixture
def statement(monkeypatch):
    """``W.state``: with no package, a five-group model; with one, record what it was asked and stop."""

    def state(capsule, *, target, package_dir=None, **kwargs):
        if package_dir is None:
            return {"whole_program": {"per_group": [{"group": g} for g in _GROUPS]}}
        raise _Stated(kwargs)

    monkeypatch.setattr(W, "state", state)


def test_group_names_parse_and_junk_is_refused():
    assert P.parse_groups(["g12,g3", 7, "4"]) == [3, 4, 7, 12]
    with pytest.raises(ValueError, match="names no group"):
        P.parse_groups(["conv"])


def test_only_groups_declines_every_other_group_and_refuses_one_the_model_lacks(statement):
    assert P.unasked(object(), target="fixture", only=["g1,g3"]) == [0, 2, 4]
    with pytest.raises(W.WholeModelBuildError, match="does not have"):
        P.unasked(object(), target="fixture", only=["g9"])


def test_the_build_asks_the_package_for_the_named_groups_alone(statement, monkeypatch, tmp_path):
    capsule = SimpleNamespace(name="m", interface=tmp_path / "m.mlir")
    monkeypatch.setattr(W, "load_model_capsule", lambda path: capsule)
    monkeypatch.setattr(W, "machine_header", lambda *a: {"sha256": "0" * 64})
    monkeypatch.setattr(W, "require_datapath_facts", lambda target: {})
    monkeypatch.setattr(W, "corpus_binder", lambda *a, **k: SimpleNamespace(binder=None, record={}))
    monkeypatch.setattr("merlin.runtime.backends.base.whole_model_driver", lambda target: SimpleNamespace())
    with pytest.raises(_Stated) as stated:
        W.build(
            tmp_path / "pkg",
            "m",
            target="fixture",
            machine="mach",
            header="h.h",
            out=tmp_path / "out",
            oracle=False,
            decline=["residual_add"],
            only_groups=["g2"],
        )
    assert stated.value.kwargs["decline"] == ["residual_add", 0, 1, 3, 4]


def test_a_partial_build_is_marked_on_the_record_the_oracle_and_beside_them(tmp_path):
    (tmp_path / "oracle.json").write_text(json.dumps({"groups": {}, "argmax": 3}))
    record = {"attribution": {"per_group": [{"group": 2, "on": "package"}, {"group": 3, "on": "vendor"}]}}
    note = P.mark(record, tmp_path, ["g2"])
    assert record[P.MARKER] == note and note["only_groups"] == ["g2"] and note["package_answered"] == ["g2"]
    assert json.loads((tmp_path / P.MARKER_FILE).read_text()) == note
    assert json.loads((tmp_path / "oracle.json").read_text())[P.MARKER] == note


_EXPECTATIONS = {"groups": {"1": {"compare": "exact", "sum": 1, "fnv1a": 2, "inputs_from": []}}, "argmax": 0}


def test_a_verdict_refuses_partial_expectations():
    from merlin.perf import whole_model_verdict as V

    V.Expectations.from_record(_EXPECTATIONS)  # the same expectations, unmarked, are read
    with pytest.raises(V.VerdictRefusal, match="partial build"):
        V.Expectations.from_record({**_EXPECTATIONS, P.MARKER: {"only_groups": ["g1"]}})


def test_the_grade_refuses_a_partial_oracle_even_when_every_line_agrees():
    from merlin.perf import whole_model_oracle as O

    oracle = {"groups": {"1": {"fnv1a": 2, "sum": 1}}, "argmax": 0}
    uart = (
        "GM_LOCAL 1 mismatches=0 of=4 first=-1\nGM_GROUP 1 matmul 10 sum=1 fnv1a=2\nGM_ARGMAX got=0 want=0 agrees=1\n"
    )
    assert O.grade(uart, oracle)["quotable"] is True
    refused = O.grade(uart, {**oracle, P.MARKER: {"only_groups": ["g1"]}})
    assert refused["quotable"] is False and "partial build" in refused["refused"]


def test_the_whole_model_gate_refuses_a_partial_build(monkeypatch, tmp_path):
    from merlin.perf import whole_model_builder as B
    from merlin.perf import whole_model_gate as G

    monkeypatch.setattr(B, "build", lambda *a, **k: {P.MARKER: {"only_groups": ["g1"]}, "attribution": {}})
    model = {"name": "m", "machine": "mach", "header": "h.h", "capsule": "c"}
    result = G.run_model(tmp_path / "pkg", model, target="fixture", roles=(), out=tmp_path / "gate")
    assert result["status"] == G.FAIL and "partial build" in result["checks"]["build"]["error"]


def test_the_service_builder_forwards_only_groups(monkeypatch, tmp_path):
    from merlin.perf import whole_model_builder as B
    from merlin.perf import whole_model_open as WO

    def fake_build(package_dir, model_capsule, **kwargs):
        raise _Stated(kwargs)

    monkeypatch.setattr(W, "build", fake_build)
    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: False)
    with pytest.raises(_Stated) as stated:
        B.build(
            "pkg",
            target="fixture",
            out_dir=tmp_path / "out",
            model_capsule="m",
            machine="mach",
            header="h.h",
            only_groups=["g2"],
        )
    assert stated.value.kwargs["only_groups"] == ["g2"]


def test_the_builder_cli_forwards_only_group_and_says_the_build_is_partial(monkeypatch, tmp_path, capsys):
    from merlin.perf import whole_model_build_cli as CLI

    def fake_build(*args, **kwargs):
        assert kwargs["only_groups"] == ["g3"]
        return {
            "elf": "p.elf",
            "elf_sha256": "0" * 64,
            "oracle": None,
            P.MARKER: {"only_groups": ["g3"]},
            "attribution": {"counts": {"package": 1}, "per_group": []},
        }

    monkeypatch.setattr(CLI, "build", fake_build)
    argv = [
        "build",
        "--capsule",
        str(tmp_path / "m"),
        "--target",
        "fixture",
        "--machine",
        "mach",
        "--header",
        "h.h",
        "--only-group",
        "g3",
    ]
    assert CLI.main(argv) == 0
    assert "PARTIAL BUILD (only g3)" in capsys.readouterr().out
