"""Actual producer inputs with synthetic proof boundaries; no SDK credit."""

import json
from pathlib import Path

import pytest
from host_graph_fusion_cases import mutate, pair

from merlin.verify import compiler_host_graph_fusion as gate
from merlin.verify import host_graph_fusion_intake as intake


def setup_pair(tmp_path, *, newlines=False):
    raw = pair(context=True)
    if newlines:
        raw = tuple(text.replace("\n", "\r\n") for text in raw)
    paths = tuple(tmp_path / name for name in ("source.mlir", "target.mlir"))
    for path, text in zip(paths, raw, strict=True):
        path.write_bytes(text.encode())
    candidate = intake.prepare(*raw)
    lean = tmp_path / "sdk" / "lean"
    lean.parent.mkdir()
    lean.touch()  # Existence only; every proof invocation is intercepted below.
    return raw, paths, candidate, lean


def invoke(route, tmp_path, paths, lean):
    if route == "job":
        return gate.check(
            {
                "before": str(paths[0]),
                "after": str(paths[1]),
                "directory": str(tmp_path / "job"),
                "params": {"index_bits": 64, "sdk_bin": str(lean.parent)},
            }
        )
    saved = tmp_path / "saved.json"
    saved.write_text(
        json.dumps(
            {
                "schema": "merlin.host-graph-fusion.checked.v1",
                "buffers": [gate.file_ref(path) for path in paths],
                "status": "descriptive-only",
            }
        )
    )
    return gate.replay(saved, tmp_path / "replay", lean=lean)


@pytest.mark.parametrize("route", ["job", "replay"])
@pytest.mark.parametrize("side", [0, 1], ids=["source", "target"])
def test_transient_valid_read_cannot_replace_the_referenced_actual_input(tmp_path, monkeypatch, route, side):
    raw, paths, candidate, lean = setup_pair(tmp_path)
    wrong = raw[0].replace("arith.addi", "arith.subi", 1) if side == 0 else mutate(raw[1], "arithmetic")
    changed = tuple(wrong if index == side else text for index, text in enumerate(raw))
    assert intake.prepare(*changed).query != candidate.query  # A typed actual semantic mutation.
    paths[side].write_bytes(wrong.encode())
    original_check = gate._check_input_refs
    original_read = Path.read_bytes

    def arm_transient_read(refs, actual_paths):
        original_check(refs, actual_paths)  # References bind the actual mutant file.

        def transient_read(path):
            if path == paths[side]:
                monkeypatch.setattr(Path, "read_bytes", original_read)
                return raw[side].encode()  # A good pair must not reach proof instead.
            return original_read(path)

        monkeypatch.setattr(Path, "read_bytes", transient_read)

    monkeypatch.setattr(gate, "_check_input_refs", arm_transient_read)
    monkeypatch.setattr(gate, "check_boundary", lambda *a, **k: pytest.fail("transient pair reached proof"))
    with pytest.raises(gate.GateUnavailable, match="source changed during independent read"):
        invoke(route, tmp_path, paths, lean)
    assert original_read(paths[side]) == wrong.encode()


@pytest.mark.parametrize("route", ["job", "replay"])
@pytest.mark.parametrize("side", [0, 1], ids=["source", "target"])
@pytest.mark.parametrize("phase", ["proof", "validation", "delivery"])
def test_original_input_is_reopened_after_proof_validation_and_delivery(tmp_path, monkeypatch, route, side, phase):
    raw, paths, candidate, lean = setup_pair(tmp_path)
    sentinel = object()
    report = {"candidate": candidate.describe(), "statements": dict.fromkeys(candidate.roots, "HOST TEST ONLY")}

    def change(at):
        if at == phase:
            paths[side].write_bytes((raw[side] + "\n").encode())

    def prove(before, after, *args, **kwargs):
        assert (before, after) == raw
        change("proof")
        return sentinel

    def validate(completion):
        assert completion is sentinel
        change("validation")
        return report

    def deliver(completion):
        validate(completion)
        change("delivery")
        return raw[1]

    monkeypatch.setattr(gate, "check_boundary", prove)
    monkeypatch.setattr(gate, "validate_completion", validate)
    monkeypatch.setattr(gate, "deliver", deliver)
    with pytest.raises(gate.GateUnavailable, match="source changed"):
        invoke(route, tmp_path, paths, lean)


@pytest.mark.parametrize("route", ["job", "replay"])
@pytest.mark.parametrize("newlines", [False, True], ids=["LF", "CRLF"])
def test_unchanged_inputs_reach_the_exact_raw_boundary(tmp_path, monkeypatch, route, newlines):
    raw, paths, candidate, lean = setup_pair(tmp_path, newlines=newlines)
    completion = gate.Completion(candidate.identity, str(tmp_path / "fake-report"))
    report = {"candidate": candidate.describe(), "statements": dict.fromkeys(candidate.roots, "HOST TEST ONLY")}

    def prove(before, after, *args, **kwargs):
        assert (before, after) == raw
        return completion

    monkeypatch.setattr(gate, "check_boundary", prove)
    monkeypatch.setattr(gate, "validate_completion", lambda value: report)
    monkeypatch.setattr(gate, "deliver", lambda value: raw[1])
    result = invoke(route, tmp_path, paths, lean)
    if route == "job":
        assert result["source_sha256"] == gate.digest(raw[0])
        assert result["final_sha256"] == gate.digest(raw[1])
    else:
        assert result is completion
