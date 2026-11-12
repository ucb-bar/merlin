from __future__ import annotations

from verification_witnesses.host_compute_audit import _ELEMENTS, _REQUANT, _graded_run, _loop, pytest


def test_a_pass_that_does_the_units_work_on_the_host_fails_the_grade(tmp_path, monkeypatch) -> None:
    from merlin.development.targetgen import offload_census as OC

    buffer = {
        "tensors": {"acc": {"shape": [32, 32], "dtype": "i32"}, "y": {"shape": [32, 32], "dtype": "i8"}},
        "kernel_abi": {"kind": "whole_program", "args": [], "outputs": ["y"]},
        "commands": [{"opcode": "COMMIT", "operands": {"src": "acc", "dst": "y"}}],
    }
    host_loop = '"builtin.module"() ({' + _loop("entry", _ELEMENTS, _REQUANT) + "\n}) : () -> ()"
    for name in ("does_it_on_the_host", "already_failing", "whole_model"):
        _graded_run(tmp_path, name, host_loop, buffer)
    results = [
        {"capsule": "does_it_on_the_host", "kind": "layer", "status": "pass"},
        {
            "capsule": "already_failing",
            "kind": "layer",
            "status": "fail",
            "failure": {"category": "functional_mismatch"},
        },
        {"capsule": "whole_model", "kind": "model", "status": "pass"},
        {"capsule": "left_no_artifacts", "kind": "layer", "status": "pass"},
    ]
    vetoed = OC.apply_host_compute_veto(results, tmp_path)
    assert [row["program"] for row in vetoed] == ["does_it_on_the_host"]
    (first, failing, model, bare) = results
    assert (first["status"], first["failure"]["category"]) == ("fail", OC.VETO_CATEGORY)
    import json

    stored = json.loads((tmp_path / "does_it_on_the_host" / "capsule_result.json").read_text())
    assert (stored["status"], stored["status_before_host_compute_veto"]) == ("fail", "pass")
    assert stored["failure"]["plane"] == "host_compute" and stored["host_compute"]["verdict"]
    assert "reported, not failed" in stored["failure"]["detail"][:800]
    untouched = json.loads((tmp_path / "already_failing" / "capsule_result.json").read_text())
    assert "status_before_host_compute_veto" not in untouched
    assert failing["failure"] == {"category": "functional_mismatch"}
    assert model["status"] == "pass" and bare["status"] == "pass"
    monkeypatch.setenv(OC.VETO_ENV, "0")
    again = [{"capsule": "does_it_on_the_host", "kind": "layer", "status": "pass"}]
    assert OC.apply_host_compute_veto(again, tmp_path) and again[0]["status"] == "pass"
    assert again[0]["host_compute"]["veto_suppressed_by"] == OC.VETO_ENV
