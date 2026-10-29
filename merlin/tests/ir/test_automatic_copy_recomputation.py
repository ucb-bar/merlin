"""Pure preparation regression: no compiler, code generator or native execution."""

import importlib

import pytest
from test_recomputation import proposal

from merlin.llvmlower import automatic_copies
from merlin.llvmlower.source_recomputation import SourceRecomputationSelection
from merlin.transforms.certificates import ir_text


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_automatic_copies_preserve_recomputation_requested_features(tmp_path, monkeypatch, selected, explicit):
    lower = importlib.import_module("merlin.llvmlower.lower")
    requested = frozenset({"fold_uniform_fill_copy"}) if explicit else None
    observed = {}
    settings = []
    original_select = automatic_copies.select
    explicit_features = original_select(requested, serial=False)[0]

    def inspect_select(features, *, serial):
        result = original_select(features, serial=serial)
        observed["report"] = result[1]
        return result

    class ReachedLowering(Exception):
        pass

    def stop_before_tools(source, **kwargs):
        observed["effective"] = kwargs["features"]
        raise ReachedLowering

    def original_owner(**kwargs):
        settings.append(kwargs["settings"]["features"])

    monkeypatch.setenv("MERLIN_COMPILER_DRIVER", "python")
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_COPIES", "1")
    monkeypatch.setenv("MERLIN_PASSES", "")
    monkeypatch.setattr(automatic_copies, "select", inspect_select)
    monkeypatch.setattr(lower, "preprocess_text", lambda text, **_: (text, {}))
    monkeypatch.setattr(lower, "lower_to_llvm_ir", stop_before_tools)
    plan = proposal()
    selection = SourceRecomputationSelection(plan, verify_current=original_owner) if selected else None
    with pytest.raises(ReachedLowering):
        lower.lower_model(
            ir_text(plan.source), tmp_path, features=requested, source_recomputation=selection, index_bits=64
        )
    report = observed["report"]
    assert report["enabled"] is (not selected)
    if selected:
        assert report["disabled_reason"] == "automatic copy augmentation with source recomputation is not qualified"
        assert observed["effective"] == explicit_features
        assert settings and all(value == (sorted(requested) if explicit else None) for value in settings)
    else:
        assert "disabled_reason" not in report
        assert automatic_copies.FEATURES <= observed["effective"]
    if explicit:
        assert "fold_uniform_fill_copy" in observed["effective"]
