"""Mandatory full checks stop at known build failures without claiming later work."""

from types import SimpleNamespace

from merlin.model_check import commands as C
from merlin.model_check import full
from merlin.model_check import report as R


def test_full_check_does_not_build_later_programs_or_final_images_after_required_failure(tmp_path, monkeypatch):
    model = SimpleNamespace(name="synthetic", programs=[SimpleNamespace(name=s) for s in ("first", "later")])
    monkeypatch.setattr(C, "load_models", lambda _: SimpleNamespace(model=lambda _: model))
    bundle = SimpleNamespace(manifest=lambda: {}, inputs=lambda *_: ["visible/v00"], root=tmp_path)
    monkeypatch.setattr(C, "_bundle", lambda _: bundle)
    monkeypatch.setattr(full, "validate", lambda _: None)
    attempted = []

    def fail(*_args, **kwargs):
        attempted.append(_args[3])
        return {
            "program": _args[3],
            "build": {"built": 0, "refused": 1},
            "refused": {"a": "first failed"},
            "kernels": [
                {"symbol": "a", "status": "refused"},
                {"symbol": "b", "status": "unfinished", "mismatches": ["not attempted"]},
            ],
        }

    monkeypatch.setattr(C, "_replay_program", fail)
    monkeypatch.setattr(
        full, "check", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("final image started"))
    )
    args = SimpleNamespace(
        mode="full",
        models=tmp_path,
        model="synthetic",
        program=None,
        host_package=tmp_path,
        contract=None,
        rtl_facts=None,
        split="visible",
        engine="spike",
        work=tmp_path / "work",
    )
    result = C.check(args, R.new_report("check", model=model.name, split="visible", submission={}))
    assert attempted == ["first"] and result["programs_unattempted"] == ["later"]
    assert result["gates"]["build"]["status"] == "fail" and result["gates"]["build"]["partial"]
    assert result["gates"]["kernel_exact"]["status"] == "fail" and result["gates"]["kernel_exact"]["partial"]
    assert result["comparison"] == {
        "selected": "host-twin",
        "status": "not_run",
        "detail": "a required accelerator kernel could not be built",
    }
    assert result["model_metrics"] is None and R.settle(result)["status"] == "fail"
