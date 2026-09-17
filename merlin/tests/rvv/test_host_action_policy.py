"""Frozen host-action catalogue and the model-blind whole-model bridge."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.kernels.microkernel import VL_DYNAMIC, MicrokernelSpec
from merlin.llvmlower.host_policy import (
    OUTPUT_TILE_AXIS,
    OUTPUT_TILE_SEMANTICS_VERSION,
    HostLoweringEffects,
    HostPolicy,
    HostPolicyMechanismError,
    VectorLengthIntent,
    materialize_lowering_effects,
)
from merlin.mining.host_actions import (
    HOST_ACTIONS,
    ConflictingHostActionsError,
    EffectRequirement,
    HostActionId,
    HostActionResolutionError,
    HostPrecision,
    IllegalPrecisionError,
    IncompleteHostMechanismError,
    MechanismStatus,
    NonCanonicalHostPolicyError,
    UnimplementedHostActionError,
    UnknownHostActionError,
    action_registry,
    load_canonical_host_policy,
    resolve_host_policy,
)


def test_registry_is_total_against_the_frozen_optimization_space():
    spec = load_yaml(
        repo_root() / "merlin/experiments/cpu_host_compiler_v0/optimization_space_v1.yaml")
    yaml_actions = spec["actions"]
    assert [entry["id"] for entry in yaml_actions] == [action.id.value for action in HOST_ACTIONS]
    assert set(action_registry()) == set(HostActionId)
    assert len(action_registry()) == len(HOST_ACTIONS) == len(HostActionId)
    for expected, action in zip(yaml_actions, HOST_ACTIONS):
        assert (expected["id"], expected["group"], expected["action_class"],
                expected["stage"], expected["value"]) == (
                    action.id.value, action.group, action.action_class.value,
                    action.stage, action.value)


def test_catalogue_and_registry_are_immutable():
    assert isinstance(HOST_ACTIONS, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        HOST_ACTIONS[0].stage = 999
    with pytest.raises(TypeError):
        action_registry()[HostActionId.DYNAMIC_VL] = HOST_ACTIONS[0]
    plan = resolve_host_policy(["output_tile_8"], precision="fp32")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.policy.output_tile = 4
    with pytest.raises(TypeError, match="tuple"):
        dataclasses.replace(plan.policy, selected_actions=["output_tile_8"])


def test_selection_is_in_canonical_catalogue_order():
    plan = resolve_host_policy(
        ["deterministic_static_partition", "dynamic_vl"],
        precision="fp32",
        worker_count=4,
    )
    assert plan.policy.selected_actions == (
        "dynamic_vl", "deterministic_static_partition")
    assert [effect.key for effect in plan.effect_requirements] == [
        "microkernel.vl_strategy", "lower.parallel_harts",
        "runtime.omp_static_schedule"]
    assert plan.effect_requirements[-2].value == 4
    assert plan.effect_requirements[-1].value == "exact_cover"


def test_unknown_duplicate_and_group_conflicts_are_rejected():
    with pytest.raises(UnknownHostActionError):
        resolve_host_policy(["not_an_action"], precision="fp32")
    with pytest.raises(ConflictingHostActionsError, match="duplicate"):
        resolve_host_policy(["dynamic_vl", "dynamic_vl"], precision="fp32")
    with pytest.raises(ConflictingHostActionsError, match="mutually exclusive"):
        resolve_host_policy(["output_tile_4", "output_tile_16"], precision="fp32")
    with pytest.raises(ConflictingHostActionsError, match="cannot be composed"):
        resolve_host_policy(["dynamic_vl", "output_tile_8"], precision="fp32")
    with pytest.raises(UnknownHostActionError, match="string/path"):
        resolve_host_policy("dynamic_vl", precision="fp32")
    with pytest.raises(UnknownHostActionError, match="exact strings"):
        resolve_host_policy([Path("dynamic_vl")], precision="fp32")


def test_unknown_output_tile_semantics_fails_closed():
    for invalid in (True, 1.0, "1"):
        with pytest.raises(HostActionResolutionError, match="semantics version"):
            resolve_host_policy(["output_tile_8"], precision="fp32",
                                output_tile_semantics_version=invalid)
    with pytest.raises(HostActionResolutionError, match="semantics version"):
        resolve_host_policy(["output_tile_8"], precision="fp32",
                            output_tile_semantics_version=OUTPUT_TILE_SEMANTICS_VERSION + 1)
    plan = resolve_host_policy(["output_tile_8"], precision="fp32")
    assert plan.policy.output_tile_axis is OUTPUT_TILE_AXIS
    assert plan.policy.output_tile_semantics_version == OUTPUT_TILE_SEMANTICS_VERSION
    assert not plan.ready
    assert plan.effect_requirements == ()


def test_precision_legality_is_explicit_and_fail_closed():
    with pytest.raises(IllegalPrecisionError, match="bf16_f32acc"):
        resolve_host_policy(["output_tile_8"], precision=HostPrecision.BF16_F32ACC)
    with pytest.raises(IllegalPrecisionError, match="int8_w8a8"):
        resolve_host_policy(["dynamic_vl"], precision=HostPrecision.INT8_W8A8)
    assert not resolve_host_policy(["output_tile_8"], precision="int8_w8a8").ready
    with pytest.raises(IllegalPrecisionError):
        resolve_host_policy(["dynamic_vl"], precision=Path("fp32"))


def test_output_tiles_are_partial_until_exact_whole_model_nr_is_guaranteed():
    for action_id in (
            HostActionId.OUTPUT_TILE_4,
            HostActionId.OUTPUT_TILE_8,
            HostActionId.OUTPUT_TILE_16):
        action = action_registry()[action_id]
        assert action.status is MechanismStatus.PARTIAL
        assert action.effect_requirements == ()
        plan = resolve_host_policy([action_id], precision="fp32")
        assert not plan.ready
        with pytest.raises(IncompleteHostMechanismError, match="exact NR"):
            plan.lowering_effects(base_microkernel=MicrokernelSpec())


def test_unimplemented_actions_are_rejected_instead_of_becoming_metadata():
    for action in HOST_ACTIONS:
        if action.status is MechanismStatus.UNIMPLEMENTED:
            with pytest.raises(UnimplementedHostActionError, match=action.id.value):
                resolve_host_policy([action.id], precision="fp32")


def test_partial_actions_are_visible_but_missing_mechanisms_block_materialization():
    for action in HOST_ACTIONS:
        if action.status is not MechanismStatus.PARTIAL:
            continue
        workers = 2 if action.id is HostActionId.PERSISTENT_WORKER_POOL else 1
        plan = resolve_host_policy([action.id], precision="fp32", worker_count=workers)
        assert not plan.ready
        assert plan.blockers and action.id.value in plan.blockers[0]
        assert plan.effect_requirements == ()
        with pytest.raises(IncompleteHostMechanismError):
            plan.require_ready()
        with pytest.raises(IncompleteHostMechanismError):
            plan.lowering_effects(base_microkernel=MicrokernelSpec())


def test_materializer_refuses_a_manual_policy_with_a_missing_mechanism():
    partial = resolve_host_policy(["lmul_4"], precision="fp32").policy
    with pytest.raises(IncompleteHostMechanismError, match="exact dtype-aware"):
        materialize_lowering_effects(partial, base_microkernel=MicrokernelSpec())


def test_existing_microkernel_and_static_partition_seams_are_real_effects():
    plan = resolve_host_policy(
        ["dynamic_vl", "deterministic_static_partition"],
        precision="fp32", worker_count=4)
    effects = plan.lowering_effects(base_microkernel=MicrokernelSpec(MR=2, NR=16, KC=32))
    assert effects.microkernel == MicrokernelSpec(MR=2, NR=16, KC=32, vl_strategy=VL_DYNAMIC)
    assert effects.parallel_harts == 4
    assert effects.parallel_schedule == "exact_cover"
    assert effects.mechanisms == (
        "kernels.microkernel.vl_strategy",
        "llvmlower.lower.parallel_harts",
        "runtime.omp_static_schedule.exact_cover",
    )


def test_persistent_pool_is_a_real_openmp_runtime_effect_not_metadata():
    plan = resolve_host_policy(
        ["deterministic_static_partition", "persistent_worker_pool"],
        precision="fp32", worker_count=4)
    assert plan.ready
    effects = plan.lowering_effects()
    assert effects.parallel_harts == 4
    assert effects.parallel_schedule == "exact_cover"
    assert effects.openmp_runtime == "merlin_pthread_pool"
    assert effects.worker_pool_size == 4
    assert effects.mechanisms == (
        "llvmlower.lower.parallel_harts",
        "runtime.omp_static_schedule.exact_cover",
        "runtime.libomp_pthread.openmp_abi",
        "runtime.libomp_pthread.session_resident_pool",
    )


def test_persistent_pool_requires_a_parallel_lowering_consumer():
    with pytest.raises(HostActionResolutionError, match="deterministic_static_partition"):
        resolve_host_policy(["persistent_worker_pool"], precision="fp32", worker_count=4)


def test_microkernel_patch_requires_an_explicit_base_policy():
    plan = resolve_host_policy(["dynamic_vl"], precision="fp32")
    with pytest.raises(HostPolicyMechanismError, match="base_microkernel"):
        plan.lowering_effects()


def test_parallel_actions_require_multiple_workers():
    with pytest.raises(ValueError, match="worker_count >= 2"):
        resolve_host_policy(["deterministic_static_partition"], precision="fp32")


def test_serialization_is_deterministic_and_contains_no_workload_identity():
    first = resolve_host_policy(
        ["deterministic_static_partition", "dynamic_vl"],
        precision="fp32", worker_count=8)
    second = resolve_host_policy(
        ["dynamic_vl", "deterministic_static_partition"],
        precision=HostPrecision.FP32, worker_count=8)
    assert first.canonical_json() == second.canonical_json()
    assert first.canonical_json().endswith("\n")
    lowered = first.lowering_effects(base_microkernel=MicrokernelSpec())
    assert lowered.canonical_dict() == first.lowering_effects(
        base_microkernel=MicrokernelSpec()).canonical_dict()
    for forbidden in ("model_id", "capsule", "holdout", "gemma", "llama", "resnet"):
        assert forbidden not in first.canonical_json().lower()


def test_canonical_json_round_trip_rejects_action_order_and_group_ambiguity():
    plan = resolve_host_policy(
        ["dynamic_vl", "deterministic_static_partition"],
        precision="fp32", worker_count=4)
    encoded = plan.canonical_json()
    assert load_canonical_host_policy(encoded) == plan

    reordered = json.loads(encoded)
    reordered["actions"].reverse()
    reordered["policy"]["selected_actions"].reverse()
    reordered_text = json.dumps(reordered, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(NonCanonicalHostPolicyError, match="canonical action-ordered"):
        load_canonical_host_policy(reordered_text)

    conflicting = json.loads(
        resolve_host_policy(["output_tile_4"], precision="fp32").canonical_json())
    conflicting["policy"]["selected_actions"].append("output_tile_16")
    conflicting_text = json.dumps(conflicting, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(ConflictingHostActionsError, match="mutually exclusive"):
        load_canonical_host_policy(conflicting_text)


def test_canonical_json_rejects_duplicate_keys_and_noncanonical_bytes():
    plan = resolve_host_policy(["output_tile_8"], precision="fp32")
    with pytest.raises(NonCanonicalHostPolicyError, match="duplicate"):
        load_canonical_host_policy('{"policy":{},"policy":{}}\n')
    with pytest.raises(NonCanonicalHostPolicyError, match="canonical action-ordered"):
        load_canonical_host_policy(plan.canonical_json().rstrip("\n"))


def test_public_execution_boundary_rejects_forged_policy_and_action_objects():
    plan = resolve_host_policy(["dynamic_vl"], precision="fp32")

    unknown = dataclasses.replace(
        plan.policy, selected_actions=("models/gemma2/dynamic_vl",), vector_length=None)
    with pytest.raises(UnknownHostActionError):
        materialize_lowering_effects(unknown, base_microkernel=MicrokernelSpec())

    mismatched = dataclasses.replace(plan.policy, vector_length=None)
    with pytest.raises(NonCanonicalHostPolicyError, match="fields do not match"):
        materialize_lowering_effects(mismatched, base_microkernel=MicrokernelSpec())

    cloned_action = dataclasses.replace(plan.actions[0])
    forged_plan = dataclasses.replace(plan, actions=(cloned_action,))
    with pytest.raises(NonCanonicalHostPolicyError, match="catalogue entries"):
        forged_plan.lowering_effects(base_microkernel=MicrokernelSpec())


def test_direct_policy_construction_rejects_illegal_precision_combinations():
    with pytest.raises(ValueError, match="only for fp32"):
        HostPolicy(
            precision=HostPrecision.INT8_W8A8,
            worker_count=1,
            selected_actions=("dynamic_vl",),
            vector_length=VectorLengthIntent.DYNAMIC,
        )
    with pytest.raises(ValueError, match="fp32 or int8_w8a8"):
        HostPolicy(
            precision=HostPrecision.BF16_F32ACC,
            worker_count=1,
            selected_actions=("output_tile_8",),
            output_tile=8,
        )


def test_nested_contract_values_are_strictly_typed_and_immutable():
    with pytest.raises(TypeError, match="semantics_version"):
        EffectRequirement("microkernel.NR", 8, "kernels.microkernel.NR", True)
    with pytest.raises(TypeError, match="semantics_version"):
        EffectRequirement("microkernel.NR", 8, "kernels.microkernel.NR", 1.0)
    with pytest.raises(TypeError, match="legal_precisions"):
        dataclasses.replace(HOST_ACTIONS[0], legal_precisions=[HostPrecision.FP32])
    with pytest.raises(TypeError, match="effect_requirements"):
        dataclasses.replace(
            HOST_ACTIONS[0],
            effect_requirements=[HOST_ACTIONS[0].effect_requirements[0]],
        )
    with pytest.raises(TypeError, match="tuple"):
        HostLoweringEffects(None, None, None, [])
    with pytest.raises(TypeError, match="output_tile_semantics_version"):
        HostPolicy(
            precision=HostPrecision.FP32,
            worker_count=1,
            selected_actions=(),
            output_tile_semantics_version=True,
        )


def test_public_boundary_rejects_permissively_constructed_microkernel_values():
    plan = resolve_host_policy(["dynamic_vl"], precision="fp32")
    malformed = MicrokernelSpec(MR="4")
    with pytest.raises(TypeError, match="base_microkernel.MR"):
        plan.lowering_effects(base_microkernel=malformed)


def test_materialized_effect_requirements_are_exactly_aligned(monkeypatch):
    import merlin.llvmlower.host_policy as host_policy_module

    plan = resolve_host_policy(
        ["dynamic_vl", "deterministic_static_partition"],
        precision="fp32",
        worker_count=4,
    )
    effects = plan.lowering_effects(base_microkernel=MicrokernelSpec())
    assert tuple(effect.mechanism for effect in plan.effect_requirements) == effects.mechanisms

    def forge_effects(policy, *, base_microkernel):
        return HostLoweringEffects(
            microkernel=base_microkernel.with_(vl_strategy=VL_DYNAMIC),
            parallel_harts=policy.worker_count,
            parallel_schedule="exact_cover",
            mechanisms=("kernels.microkernel.vl_strategy",),
        )

    monkeypatch.setattr(
        host_policy_module, "_materialize_validated_lowering_effects", forge_effects)
    with pytest.raises(NonCanonicalHostPolicyError, match="mechanisms differ"):
        materialize_lowering_effects(plan.policy, base_microkernel=MicrokernelSpec())

    baseline = resolve_host_policy([], precision="fp32")
    unclaimed = HostLoweringEffects(
        microkernel=None,
        parallel_harts=4,
        parallel_schedule="exact_cover",
        openmp_runtime=None,
        worker_pool_size=None,
        mechanisms=(),
    )
    with pytest.raises(NonCanonicalHostPolicyError, match="not required"):
        baseline.validate_materialized_effects(unclaimed)
