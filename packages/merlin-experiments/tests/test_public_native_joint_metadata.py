"""Original metadata custody controls only; no target timing qualification."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2.contracts import StageGateError
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source
from test_public_native_binding_correspondence import inputs as inputs
from test_public_native_binding_correspondence import setup

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapabilities, ReviewedPassCapability


class CountingOwner:
    def __init__(self, identity):
        self.identity = identity
        self._original_identity = identity
        self.checks = 0
        self.valid = True

    def revalidate(self):
        self.checks += 1
        if not self.valid or self.identity != self._original_identity:
            raise ValueError("original native/count proof changed")


def original(source, inputs, monkeypatch):
    args, selected, bridge = setup(source, inputs)
    count, semantics = CountingOwner("original-count"), CountingOwner("original-semantics")
    binding = selected.correspondence.binding
    selected = replace(
        selected,
        correspondence=ReviewedPassCapability("timing_model_correspondence", binding, count),
        native_semantics=ReviewedPassCapability("native_timing_semantics", binding, semantics),
    )
    runtime_calls = []
    verify = N.ArchiveSpikeRuntime.verify

    def current_runtime(self):
        runtime_calls.append(self)
        return verify(self)

    monkeypatch.setattr(N.ArchiveSpikeRuntime, "verify", current_runtime)
    native = N.prepare_public_native_exploration(
        selected, binding_correspondence=bridge, metadata_protocol=N.JOINT_METADATA_PROTOCOL
    )
    policy = D.prepare_public_source_diagnostic(**args, native_exploration=native)
    return native, policy, bridge, count, semantics, runtime_calls


def test_optin_coalesces_same_original_roster_only_and_keeps_current_checks(source, inputs, monkeypatch):
    native, policy, bridge, count, semantics, runtime_calls = original(source, inputs, monkeypatch)
    verifier = bridge.owner
    counts = (count.checks, semantics.checks, verifier.checks, verifier.binding_checks, len(runtime_calls))
    policy.binding()
    assert count.checks == counts[0] + 2
    assert semantics.checks == counts[1] + 2
    assert verifier.checks == counts[2] + 3  # original boundaries plus actual binding callback
    assert verifier.binding_checks == counts[3] + 1
    assert len(runtime_calls) == counts[4] + 5  # runtime remains fresh at every verify
    policy.binding()
    assert count.checks == counts[0] + 4  # no cross-operation cache
    assert semantics.checks == counts[1] + 4
    assert native.record()["schema"] == "merlin.public_native_exploration.v2"
    assert native.record()["promotion_authority"] == "NONE"


@pytest.mark.parametrize("field", ["hardware", "software", "numerical_policy", "runtime", "library"])
def test_each_current_compiler_binding_field_remains_checked(source, inputs, monkeypatch, field):
    native, policy, bridge, *_ = original(source, inputs, monkeypatch)
    changed = SimpleNamespace(
        feedback_evaluator=policy.feedback_evaluator,
        _compiler_pass_selection=SimpleNamespace(binding=replace(bridge.owner.compiler_binding, **{field: "foreign"})),
    )
    with pytest.raises((ValueError, StageGateError)):
        with native._metadata_operation():
            native.verify_for(changed)


@pytest.mark.parametrize("selected_name", ["correspondence", "native_semantics"])
def test_equal_record_foreign_capability_object_rejected(source, inputs, monkeypatch, selected_name):
    native, *_ = original(source, inputs, monkeypatch)
    original_cap = getattr(native.selection, selected_name)
    replacement = ReviewedPassCapability(original_cap.mechanism, original_cap.binding, original_cap.owner)
    assert replacement.record() == original_cap.record()
    object.__setattr__(native.selection, selected_name, replacement)
    with pytest.raises(StageGateError, match="capability roster"):
        native.verify()


def test_arbitrary_containing_registry_cannot_select_joint_mode(source, inputs, monkeypatch):
    native, _, bridge, *_ = original(source, inputs, monkeypatch)
    roster = (native.selection.correspondence, native.selection.native_semantics, bridge)
    foreign = ReviewedPassCapabilities(roster)
    with pytest.raises(StageGateError, match="issued roster"):
        N._bridge_verification(bridge, bridge.binding, foreign, expected_roster=roster)


@pytest.mark.parametrize("owner_index", [3, 4])
def test_final_original_count_or_native_failure_prevents_return(source, inputs, monkeypatch, owner_index):
    value = original(source, inputs, monkeypatch)
    native, policy = value[:2]
    owner = value[owner_index]
    with pytest.raises(ValueError, match="proof changed"):
        with native._metadata_operation():
            native.record()
            owner.valid = False
    assert N.METADATA._ACTIVE.get() is None
    with pytest.raises(ValueError):
        policy.binding()


def test_primary_exception_kept_when_final_original_check_also_fails(source, inputs, monkeypatch):
    native, _, _, count, *_ = original(source, inputs, monkeypatch)
    with pytest.raises(RuntimeError, match="original body failure") as caught:
        with native._metadata_operation():
            count.valid = False
            raise RuntimeError("original body failure")
    assert any("final" in note.lower() for note in caught.value.__notes__)
    assert N.METADATA._ACTIVE.get() is None


def test_changed_original_callback_refuses_inside_metadata(source, inputs, monkeypatch):
    native, *_ = original(source, inputs, monkeypatch)
    with pytest.raises(ValueError, match="implementation changed"):
        with native._metadata_operation():
            monkeypatch.setattr(CountingOwner, "revalidate", lambda self: None)


def test_changed_source_bytes_refuse_before_consumption(source, inputs, monkeypatch):
    native, *_ = original(source, inputs, monkeypatch)
    with pytest.raises(StageGateError, match="evidence changed"):
        with native._metadata_operation():
            source["observations_path"].write_text("changed original observations")
            native.record()


def test_runtime_mutation_is_not_cached_by_joint_registry(source, inputs, monkeypatch):
    native, *_ = original(source, inputs, monkeypatch)
    with pytest.raises(StageGateError, match="runtime source/tool/model changed"):
        with native._metadata_operation():
            native.selection.runtime.observer.write_text("changed observer")
            native.record()


def test_current_observation_outside_metadata_transaction(source, inputs, monkeypatch, tmp_path):
    def observe(*_args, **_kwargs):
        assert N.METADATA._ACTIVE.get() is None
        return {"status": "UNKNOWN", "reason": "neutral current observation"}

    monkeypatch.setattr(N.OBSERVATION, "observe", observe)
    native, policy, *_ = original(source, inputs, monkeypatch)
    assert native.observe_member(policy, None, tmp_path, {}, sandbox=None, remaining_s=5)["status"] == "UNKNOWN"


@pytest.mark.parametrize("protocol", [True, "foreign", {}, []])
def test_unknown_protocol_never_issues_owner(source, inputs, protocol):
    _, selected, bridge = setup(source, inputs)
    with pytest.raises((StageGateError, TypeError)):
        N.prepare_public_native_exploration(selected, binding_correspondence=bridge, metadata_protocol=protocol)


def test_joint_protocol_requires_live_bridge(source):
    from test_public_native_exploration import selection

    with pytest.raises(StageGateError, match="explicit original bridge"):
        N.prepare_public_native_exploration(selection(source), metadata_protocol=N.JOINT_METADATA_PROTOCOL)


@pytest.mark.parametrize("selected_name", ["correspondence", "native_semantics"])
@pytest.mark.parametrize("field", ["hardware", "software", "numerical_policy", "runtime", "library"])
def test_each_original_native_binding_field_stays_structurally_checked(
    source, inputs, monkeypatch, selected_name, field
):
    native, *_ = original(source, inputs, monkeypatch)
    cap = getattr(native.selection, selected_name)
    changed = replace(cap.binding, **{field: "foreign"})
    with pytest.raises(ValueError, match="implementation changed"):
        with native._metadata_operation():
            object.__setattr__(cap, "_binding", changed)
            native.record()


def test_callback_code_mutation_with_same_function_object_refuses(source, inputs, monkeypatch):
    native, *_ = original(source, inputs, monkeypatch)
    function = CountingOwner.revalidate
    with pytest.raises(ValueError, match="implementation changed"):
        with native._metadata_operation():
            monkeypatch.setattr(function, "__code__", (lambda self: None).__code__)
            native.record()


def test_different_nested_registry_gets_its_own_fresh_boundaries(source, inputs, monkeypatch):
    native, _, _, count, semantics, _ = original(source, inputs, monkeypatch)
    foreign = ReviewedPassCapabilities((native.selection.correspondence,))
    before = count.checks, semantics.checks
    with native._metadata_operation():
        with N.METADATA.scoped_registry_metadata_verification(foreign):
            native.selection.correspondence.record()
    assert count.checks == before[0] + 4  # outer and distinct inner entry/exit
    assert semantics.checks == before[1] + 2


def test_weak_joint_registry_does_not_keep_dropped_native_owner_alive(source, inputs, monkeypatch):
    import gc
    import weakref

    native, policy, *_ = original(source, inputs, monkeypatch)
    registry = weakref.ref(N._ISSUED[native][7])
    owner = weakref.ref(native)
    del policy, native
    gc.collect()
    assert owner() is None and registry() is None


def test_protocol_cannot_be_selected_by_a_spoofed_equality_object(source, inputs):
    class Spoof:
        def __eq__(self, _):
            return True

    _, selected, bridge = setup(source, inputs)
    with pytest.raises(StageGateError, match="exact original opt-in"):
        N.prepare_public_native_exploration(selected, binding_correspondence=bridge, metadata_protocol=Spoof())
