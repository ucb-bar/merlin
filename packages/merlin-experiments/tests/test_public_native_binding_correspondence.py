"""Neutral original-owner transport tests, not target timing qualification."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2.contracts import StageGateError
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source
from test_public_native_exploration import selection
from test_public_source_diagnostic import inputs as public_inputs

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "public-source-policy"
    root.mkdir()
    return public_inputs.__wrapped__(root, monkeypatch)


class NeutralCorrespondenceOwner:
    identity = "neutral-original-source-native-correspondence"

    def __init__(self, compiler_binding, native_binding):
        self.compiler_binding = compiler_binding
        self.native_binding = native_binding
        self._original = (compiler_binding, native_binding)

    def revalidate(self):
        if (self.compiler_binding, self.native_binding) != self._original:
            raise ValueError("original source/native proof changed")

    def verify_compiler_binding(self, compiler_binding, *, native_binding):
        self.revalidate()
        if (compiler_binding, native_binding) != self._original:
            raise ValueError("full original compiler/native binding mismatch")


def setup(source, inputs):
    args, *_ = inputs
    selected = selection(source)
    native = replace(selected.correspondence.binding, hardware=args["compiler_pass_selection"].binding.hardware)
    selected = replace(
        selected,
        expected_rtl_facts_sha256=native.hardware,
        correspondence=ReviewedPassCapability("timing_model_correspondence", native, selected.correspondence.owner),
        native_semantics=ReviewedPassCapability("native_timing_semantics", native, selected.native_semantics.owner),
    )
    compiler = args["compiler_pass_selection"].binding
    assert compiler.software != native.software
    original = NeutralCorrespondenceOwner(compiler, native)
    bridge = ReviewedPassCapability(N.BRIDGE_MECHANISM, native, original)
    return args, selected, bridge


def test_distinct_software_namespaces_require_explicit_live_correspondence(source, inputs):
    args, selected, bridge = setup(source, inputs)
    legacy = N.prepare_public_native_exploration(selected)
    assert legacy.record()["schema"] == N.SCHEMA
    assert "compiler_source_native_correspondence" not in legacy.record()
    with pytest.raises(StageGateError, match="another hardware/software"):
        D.prepare_public_source_diagnostic(**args, native_exploration=legacy)
    native = N.prepare_public_native_exploration(selected, binding_correspondence=bridge)
    policy = D.prepare_public_source_diagnostic(**args, native_exploration=native)
    assert native.record()["schema"] == "merlin.public_native_exploration.v2"
    assert native.record()["compiler_source_native_correspondence"] == bridge.record()
    assert policy._ordinary_screen is None
    assert policy.binding()["formal_admission"] is False
    assert not hasattr(policy.feedback_evaluator, "certificate")
    assert native.record()["measured_cycles"] is None and native.record()["promotion_authority"] == "NONE"


@pytest.mark.parametrize("field", ["hardware", "software", "numerical_policy", "runtime", "library"])
def test_bridge_rejects_every_other_native_binding_field(source, inputs, field):
    _, selected, bridge = setup(source, inputs)
    wrong = ReviewedPassCapability(N.BRIDGE_MECHANISM, replace(bridge.binding, **{field: "foreign"}), bridge.owner)
    with pytest.raises(StageGateError, match="exact reviewed bridge"):
        N.prepare_public_native_exploration(selected, binding_correspondence=wrong)


@pytest.mark.parametrize("field", ["hardware", "software", "numerical_policy", "runtime", "library"])
def test_bridge_rejects_every_other_compiler_binding_field(source, inputs, field):
    args, selected, bridge = setup(source, inputs)
    native = N.prepare_public_native_exploration(selected, binding_correspondence=bridge)
    policy = D.prepare_public_source_diagnostic(**args, native_exploration=native)
    changed = SimpleNamespace(
        feedback_evaluator=policy.feedback_evaluator,
        _compiler_pass_selection=SimpleNamespace(binding=replace(bridge.owner.compiler_binding, **{field: "foreign"})),
    )
    with pytest.raises((ValueError, StageGateError)):
        native.verify_for(changed)


@pytest.mark.parametrize("fault", ["json", "mechanism", "replacement", "method", "proof"])
def test_saved_foreign_or_mutated_bridge_is_not_authority(source, inputs, monkeypatch, fault):
    _, selected, bridge = setup(source, inputs)
    if fault == "json":
        with pytest.raises(StageGateError):
            N.prepare_public_native_exploration(selected, binding_correspondence=bridge.record())
        return
    if fault == "mechanism":
        with pytest.raises(StageGateError):
            N.prepare_public_native_exploration(
                selected,
                binding_correspondence=ReviewedPassCapability("another_mechanism", bridge.binding, bridge.owner),
            )
        return
    native = N.prepare_public_native_exploration(selected, binding_correspondence=bridge)
    if fault == "replacement":
        object.__setattr__(
            native, "binding_correspondence", ReviewedPassCapability(N.BRIDGE_MECHANISM, bridge.binding, bridge.owner)
        )
    elif fault == "method":
        monkeypatch.setattr(NeutralCorrespondenceOwner, "verify_compiler_binding", lambda *_a, **_k: None)
    else:
        bridge.owner.compiler_binding = replace(bridge.owner.compiler_binding, software="forged-proof")
    with pytest.raises((ValueError, StageGateError)):
        native.verify()
