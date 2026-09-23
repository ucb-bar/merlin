"""Frozen host-action catalogue and deterministic resolver for whole-model lowering.

The catalogue mirrors ``cpu_host_compiler_v0/optimization_space_v1.yaml`` but is a
compiler-facing contract, not an experiment reader.  Keeping the entries typed and total
makes drift detectable in tests while leaving the frozen experiment untouched.

Resolution separates three facts which previous metadata-only paths conflated:

* ``implemented`` -- the action has a concrete effect accepted by
  :func:`merlin.llvmlower.host_policy.materialize_lowering_effects`;
* ``partial`` -- related machinery exists, but not the exact action contract; and
* ``unimplemented`` -- no matching compiler mechanism exists.

Unknown and unimplemented selections are rejected immediately.  A partial selection may
be inspected as a non-ready plan, but ``require_ready`` and materialization fail closed.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ..kernels.microkernel import MicrokernelSpec
from ..llvmlower.host_policy import (
    OUTPUT_TILE_AXIS,
    OUTPUT_TILE_SEMANTICS_VERSION,
    HostLoweringEffects,
    HostPolicy,
    HostPrecision,
    VectorLengthIntent,
    materialize_lowering_effects,
)


class HostActionId(str, Enum):
    DYNAMIC_VL = "dynamic_vl"
    FIXED_VLEN_256_GUARDED = "fixed_vlen_256_guarded"
    LMUL_1 = "lmul_1"
    LMUL_2 = "lmul_2"
    LMUL_4 = "lmul_4"
    OUTPUT_TILE_4 = "output_tile_4"
    OUTPUT_TILE_8 = "output_tile_8"
    OUTPUT_TILE_16 = "output_tile_16"
    K_UNROLL_2 = "k_unroll_2"
    K_UNROLL_4 = "k_unroll_4"
    K_UNROLL_8 = "k_unroll_8"
    RHS_PANEL_8 = "rhs_panel_8"
    RHS_PANEL_16 = "rhs_panel_16"
    FUSE_LEGAL_EPILOGUES = "fuse_legal_epilogues"
    RETAIN_IMMUTABLE_PACKS = "retain_immutable_packs"
    DETERMINISTIC_STATIC_PARTITION = "deterministic_static_partition"
    PERSISTENT_WORKER_POOL = "persistent_worker_pool"


class ActionClass(str, Enum):
    HEURISTIC = "heuristic"
    KNOB = "knob"
    PASS = "pass"


class MechanismStatus(str, Enum):
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    UNIMPLEMENTED = "unimplemented"


class PolicyField(str, Enum):
    VECTOR_LENGTH = "vector_length"
    LMUL = "lmul"
    OUTPUT_TILE = "output_tile"
    K_UNROLL = "k_unroll"
    RHS_PANEL = "rhs_panel"
    FUSE_LEGAL_EPILOGUES = "fuse_legal_epilogues"
    RETAIN_IMMUTABLE_PACKS = "retain_immutable_packs"
    STATIC_PARTITION = "deterministic_static_partition"
    PERSISTENT_WORKER_POOL = "persistent_worker_pool"


# Deliberately enumerated instead of deriving them from the enum.  Adding a new
# precision must not silently make every old action legal for that arithmetic.
_FP32_ONLY = (HostPrecision.FP32,)
_FP32_INT8 = (HostPrecision.FP32, HostPrecision.INT8_W8A8)
_CURRENT_PRECISIONS = (
    HostPrecision.FP32,
    HostPrecision.INT8_W8A8,
    HostPrecision.BF16_F32ACC,
    HostPrecision.FP16_F32ACC,
)


@dataclass(frozen=True)
class EffectRequirement:
    """One concrete effect an implemented action requires from downstream lowering."""

    key: str
    value: str | int | bool
    mechanism: str
    semantics_version: int | None = None

    def __post_init__(self) -> None:
        if type(self.key) is not str or not self.key:
            raise TypeError("effect key must be a non-empty string")
        if type(self.value) not in (str, int, bool) or (
                type(self.value) is str and not self.value):
            raise TypeError("effect value must be a non-empty string, int, or bool")
        if type(self.mechanism) is not str or not self.mechanism:
            raise TypeError("effect mechanism must be a non-empty string")
        if self.semantics_version is not None and (
                type(self.semantics_version) is not int or self.semantics_version < 1):
            raise TypeError("effect semantics_version must be None or a positive int")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "mechanism": self.mechanism,
            "semantics_version": self.semantics_version,
            "value": self.value,
        }


@dataclass(frozen=True)
class HostAction:
    id: HostActionId
    group: str
    action_class: ActionClass
    stage: int
    value: str | int | bool
    policy_field: PolicyField
    policy_value: str | int | bool
    legal_precisions: tuple[HostPrecision, ...]
    status: MechanismStatus
    existing_mechanisms: tuple[str, ...] = ()
    missing_mechanisms: tuple[str, ...] = ()
    effect_requirements: tuple[EffectRequirement, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, HostActionId):
            raise TypeError("action id must be a HostActionId")
        if type(self.group) is not str or not self.group:
            raise TypeError("action group must be a non-empty string")
        if not isinstance(self.action_class, ActionClass):
            raise TypeError("action_class must be an ActionClass")
        if type(self.stage) is not int or self.stage < 0:
            raise TypeError("action stage must be a non-negative int")
        for name in ("value", "policy_value"):
            value = getattr(self, name)
            if type(value) not in (str, int, bool) or (type(value) is str and not value):
                raise TypeError(f"action {name} must be a non-empty string, int, or bool")
        if not isinstance(self.policy_field, PolicyField):
            raise TypeError("policy_field must be a PolicyField")
        if not isinstance(self.legal_precisions, tuple) or not self.legal_precisions or any(
                not isinstance(precision, HostPrecision)
                for precision in self.legal_precisions):
            raise TypeError("legal_precisions must be a non-empty tuple of HostPrecision")
        if len(set(self.legal_precisions)) != len(self.legal_precisions):
            raise ValueError("legal_precisions must not contain duplicates")
        if not isinstance(self.status, MechanismStatus):
            raise TypeError("status must be a MechanismStatus")
        for name in ("existing_mechanisms", "missing_mechanisms"):
            mechanisms = getattr(self, name)
            if not isinstance(mechanisms, tuple) or any(
                    type(mechanism) is not str or not mechanism
                    for mechanism in mechanisms):
                raise TypeError(f"{name} must be a tuple of non-empty strings")
        if not isinstance(self.effect_requirements, tuple) or any(
                not isinstance(effect, EffectRequirement)
                for effect in self.effect_requirements):
            raise TypeError("effect_requirements must be a tuple of EffectRequirement")


def _effect(key: str, value: str | int | bool, mechanism: str, *,
            semantics_version: int | None = None) -> tuple[EffectRequirement, ...]:
    return (EffectRequirement(key, value, mechanism, semantics_version),)


# Catalogue order is the canonical action order.  Stages are checked nondecreasing below and the
# full order is checked against the frozen YAML in tests; no sort on id can accidentally change it.
HOST_ACTIONS: tuple[HostAction, ...] = (
    HostAction(HostActionId.DYNAMIC_VL, "vector_length", ActionClass.HEURISTIC, 10, "scalable",
               PolicyField.VECTOR_LENGTH, VectorLengthIntent.DYNAMIC.value,
               _FP32_ONLY, MechanismStatus.IMPLEMENTED,
               ("scalable MicrokernelSpec.vl_strategy lowering",), (),
               _effect("microkernel.vl_strategy", "dynamic",
                       "kernels.microkernel.vl_strategy")),
    HostAction(HostActionId.FIXED_VLEN_256_GUARDED, "vector_length", ActionClass.HEURISTIC, 10, 256,
               PolicyField.VECTOR_LENGTH, VectorLengthIntent.FIXED_256_GUARDED.value,
               _CURRENT_PRECISIONS, MechanismStatus.PARTIAL,
               ("march_with_vlen compiler constraint", "independent CSR VLEN measurement"),
               ("one typed seam composing the compile constraint with the same-run CSR guard",)),
    HostAction(HostActionId.LMUL_1, "lmul", ActionClass.KNOB, 20, 1,
               PolicyField.LMUL, 1, _CURRENT_PRECISIONS, MechanismStatus.PARTIAL,
               ("lmul_widen_n relative schedule edit",),
               ("exact dtype-aware LMUL=1 control",)),
    HostAction(HostActionId.LMUL_2, "lmul", ActionClass.KNOB, 20, 2,
               PolicyField.LMUL, 2, _CURRENT_PRECISIONS, MechanismStatus.PARTIAL,
               ("lmul_widen_n relative schedule edit",),
               ("exact dtype-aware LMUL=2 control",)),
    HostAction(HostActionId.LMUL_4, "lmul", ActionClass.KNOB, 20, 4,
               PolicyField.LMUL, 4, _CURRENT_PRECISIONS, MechanismStatus.PARTIAL,
               ("lmul_widen_n relative schedule edit",),
               ("exact dtype-aware LMUL=4 control independent of the base schedule",)),
    HostAction(HostActionId.OUTPUT_TILE_4, "output_tile", ActionClass.KNOB, 30, 4,
               PolicyField.OUTPUT_TILE, 4, _FP32_INT8,
               MechanismStatus.PARTIAL,
               ("MicrokernelSpec.NR shape-dependent upper bound",),
               ("an exact NR=4 effect for every supported contraction shape",)),
    HostAction(HostActionId.OUTPUT_TILE_8, "output_tile", ActionClass.KNOB, 30, 8,
               PolicyField.OUTPUT_TILE, 8, _FP32_INT8,
               MechanismStatus.PARTIAL,
               ("MicrokernelSpec.NR shape-dependent upper bound",),
               ("an exact NR=8 effect for every supported contraction shape",)),
    HostAction(HostActionId.OUTPUT_TILE_16, "output_tile", ActionClass.KNOB, 30, 16,
               PolicyField.OUTPUT_TILE, 16, _FP32_INT8,
               MechanismStatus.PARTIAL,
               ("MicrokernelSpec.NR shape-dependent upper bound",),
               ("an exact NR=16 effect for every supported contraction shape",)),
    HostAction(HostActionId.K_UNROLL_2, "k_unroll", ActionClass.KNOB, 40, 2,
               PolicyField.K_UNROLL, 2, _FP32_INT8,
               MechanismStatus.UNIMPLEMENTED, (),
               ("a reduction-loop unroll factor; MicrokernelSpec.KC is measured inert and "
                "unroll_m is M-axis",)),
    HostAction(HostActionId.K_UNROLL_4, "k_unroll", ActionClass.KNOB, 40, 4,
               PolicyField.K_UNROLL, 4, _FP32_INT8,
               MechanismStatus.UNIMPLEMENTED, (),
               ("a reduction-loop unroll factor; MicrokernelSpec.KC is measured inert and "
                "unroll_m is M-axis",)),
    HostAction(HostActionId.K_UNROLL_8, "k_unroll", ActionClass.KNOB, 40, 8,
               PolicyField.K_UNROLL, 8, _FP32_INT8,
               MechanismStatus.UNIMPLEMENTED, (),
               ("a reduction-loop unroll factor; MicrokernelSpec.KC is measured inert and "
                "unroll_m is M-axis",)),
    HostAction(HostActionId.RHS_PANEL_8, "rhs_layout", ActionClass.HEURISTIC, 50, "panel8",
               PolicyField.RHS_PANEL, 8, _FP32_INT8, MechanismStatus.PARTIAL,
               ("transform.structured.pack for linalg.matmul",),
               ("an exact width-8 RHS panel for every supported contraction family",)),
    HostAction(HostActionId.RHS_PANEL_16, "rhs_layout", ActionClass.HEURISTIC, 50, "panel16",
               PolicyField.RHS_PANEL, 16, _FP32_INT8, MechanismStatus.PARTIAL,
               ("transform.structured.pack for linalg.matmul",),
               ("an exact width-16 RHS panel for every supported contraction family",)),
    HostAction(HostActionId.FUSE_LEGAL_EPILOGUES, "epilogue_fusion", ActionClass.PASS, 60,
               "legal_only", PolicyField.FUSE_LEGAL_EPILOGUES, True, _CURRENT_PRECISIONS,
               MechanismStatus.PARTIAL,
               ("post-vectorization linalg-fuse-elementwise-ops stage",),
               ("typed legality-scoped epilogue matching instead of broad environment-gated "
                "fusion",)),
    HostAction(HostActionId.RETAIN_IMMUTABLE_PACKS, "pack_lifetime", ActionClass.PASS, 70,
               "session_resident", PolicyField.RETAIN_IMMUTABLE_PACKS, True,
               _FP32_INT8, MechanismStatus.UNIMPLEMENTED, (),
               ("a session-owned immutable prepack cache outside the inference call",)),
    HostAction(HostActionId.DETERMINISTIC_STATIC_PARTITION, "parallel_schedule",
               ActionClass.HEURISTIC, 80, "exact_cover", PolicyField.STATIC_PARTITION, True,
               _CURRENT_PRECISIONS, MechanismStatus.IMPLEMENTED,
               ("parallel_harts OpenMP lowering",
                "merlin_omp_static_split exact-cover runtime"), (),
               (
                   EffectRequirement("lower.parallel_harts", "policy.worker_count",
                                     "llvmlower.lower.parallel_harts"),
                   EffectRequirement("runtime.omp_static_schedule", "exact_cover",
                                     "runtime.omp_static_schedule.exact_cover"),
               )),
    HostAction(HostActionId.PERSISTENT_WORKER_POOL, "worker_lifetime", ActionClass.PASS, 90,
               "session_resident", PolicyField.PERSISTENT_WORKER_POOL, True,
               _CURRENT_PRECISIONS, MechanismStatus.IMPLEMENTED,
               ("pthread-backed __kmpc_* runtime provider",
                "process-lifetime lazy worker pool"), (),
               (
                   EffectRequirement("runtime.openmp_abi_provider", "merlin_pthread_pool",
                                     "runtime.libomp_pthread.openmp_abi"),
                   EffectRequirement("runtime.worker_pool_size", "policy.worker_count",
                                     "runtime.libomp_pthread.session_resident_pool"),
               )),
)


_ACTION_BY_ID: Mapping[HostActionId, HostAction] = MappingProxyType(
    {action.id: action for action in HOST_ACTIONS}
)

if len(_ACTION_BY_ID) != len(HostActionId) or len(HOST_ACTIONS) != len(HostActionId):
    raise RuntimeError("host action catalogue must contain every HostActionId exactly once")
if tuple(action.stage for action in HOST_ACTIONS) != tuple(
        sorted(action.stage for action in HOST_ACTIONS)):
    raise RuntimeError("host action catalogue must be in nondecreasing stage order")
for _action in HOST_ACTIONS:
    if _action.status is MechanismStatus.IMPLEMENTED and not _action.effect_requirements:
        raise RuntimeError(f"implemented host action {_action.id.value!r} has no concrete effect")
    if _action.status is not MechanismStatus.IMPLEMENTED and _action.effect_requirements:
        raise RuntimeError(f"incomplete host action {_action.id.value!r} must not emit effects")
    if _action.status is not MechanismStatus.IMPLEMENTED and not _action.missing_mechanisms:
        raise RuntimeError(f"incomplete host action {_action.id.value!r} must name its blockers")


class HostActionResolutionError(ValueError):
    pass


class UnknownHostActionError(HostActionResolutionError):
    pass


class ConflictingHostActionsError(HostActionResolutionError):
    pass


class IllegalPrecisionError(HostActionResolutionError):
    pass


class UnimplementedHostActionError(HostActionResolutionError):
    pass


class IncompleteHostMechanismError(HostActionResolutionError):
    pass


class NonCanonicalHostPolicyError(HostActionResolutionError):
    pass


@dataclass(frozen=True)
class ResolvedHostPolicy:
    policy: HostPolicy
    actions: tuple[HostAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy, HostPolicy):
            raise TypeError("policy must be a HostPolicy")
        if not isinstance(self.actions, tuple) or any(
                not isinstance(action, HostAction) for action in self.actions):
            raise TypeError("actions must be a tuple of HostAction")

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(
            f"{action.id.value}: {missing}"
            for action in self.actions
            if action.status is not MechanismStatus.IMPLEMENTED
            for missing in action.missing_mechanisms
        )

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def effect_requirements(self) -> tuple[EffectRequirement, ...]:
        return tuple(effect for action in self.actions for effect in self._effects_for(action))

    def _effects_for(self, action: HostAction) -> tuple[EffectRequirement, ...]:
        """Bind catalogue-level requirements to this policy's concrete resource values."""

        return tuple(
            EffectRequirement(
                key=effect.key,
                value=(self.policy.worker_count
                       if effect.value == "policy.worker_count" else effect.value),
                mechanism=effect.mechanism,
                semantics_version=effect.semantics_version,
            )
            for effect in action.effect_requirements
        )

    def require_ready(self) -> ResolvedHostPolicy:
        canonical = self._revalidate()
        if canonical.blockers:
            raise IncompleteHostMechanismError(
                "host policy is not executable because exact mechanisms are missing: "
                + "; ".join(canonical.blockers)
            )
        return self

    def lowering_effects(
        self, *, base_microkernel: MicrokernelSpec | None = None,
    ) -> HostLoweringEffects:
        self._revalidate()
        self.require_ready()
        return materialize_lowering_effects(self.policy, base_microkernel=base_microkernel)

    def validate_materialized_effects(self, effects: HostLoweringEffects) -> None:
        """Require exact agreement between catalogue requirements and emitted effects."""

        canonical = self._revalidate()
        canonical.require_ready()
        if not isinstance(effects, HostLoweringEffects):
            raise TypeError("effects must be HostLoweringEffects")
        requirements = canonical.effect_requirements
        required_mechanisms = tuple(requirement.mechanism for requirement in requirements)
        if effects.mechanisms != required_mechanisms:
            raise NonCanonicalHostPolicyError(
                "materialized mechanisms differ from the selected action requirements: "
                f"expected {required_mechanisms!r}, got {effects.mechanisms!r}"
            )
        observed: dict[str, str | int | bool | None] = {
            "microkernel.vl_strategy": (
                effects.microkernel.vl_strategy if effects.microkernel is not None else None),
            "lower.parallel_harts": effects.parallel_harts,
            "runtime.omp_static_schedule": effects.parallel_schedule,
            "runtime.openmp_abi_provider": effects.openmp_runtime,
            "runtime.worker_pool_size": effects.worker_pool_size,
        }
        for requirement in requirements:
            if requirement.key not in observed:
                raise NonCanonicalHostPolicyError(
                    f"no materialized-effect reader exists for {requirement.key!r}"
                )
            if observed[requirement.key] != requirement.value:
                raise NonCanonicalHostPolicyError(
                    f"materialized effect {requirement.key!r} is "
                    f"{observed[requirement.key]!r}, expected {requirement.value!r}"
                )
        required_keys = {requirement.key for requirement in requirements}
        parallel_keys = {"lower.parallel_harts", "runtime.omp_static_schedule"}
        if effects.parallel_harts is not None and not parallel_keys.issubset(required_keys):
            raise NonCanonicalHostPolicyError(
                "materialized parallel effects were not required by the selected actions"
            )

    def _revalidate(self) -> ResolvedHostPolicy:
        """Rebuild this plan through catalogue authority before compiler execution."""

        canonical = resolve_host_policy(
            self.policy.selected_actions,
            precision=self.policy.precision,
            worker_count=self.policy.worker_count,
            output_tile_semantics_version=self.policy.output_tile_semantics_version,
        )
        if self.policy != canonical.policy:
            raise NonCanonicalHostPolicyError(
                "host policy fields do not match the selected catalogue actions"
            )
        if len(self.actions) != len(canonical.actions) or any(
                supplied is not expected
                for supplied, expected in zip(self.actions, canonical.actions)):
            raise NonCanonicalHostPolicyError(
                "resolved actions must be the canonical catalogue entries"
            )
        return canonical

    def canonical_dict(self) -> dict[str, object]:
        return {
            "actions": [
                {
                    "action_class": action.action_class.value,
                    "effect_requirements": [
                        effect.canonical_dict() for effect in self._effects_for(action)
                    ],
                    "existing_mechanisms": list(action.existing_mechanisms),
                    "group": action.group,
                    "id": action.id.value,
                    "missing_mechanisms": list(action.missing_mechanisms),
                    "stage": action.stage,
                    "status": action.status.value,
                    "value": action.value,
                }
                for action in self.actions
            ],
            "blockers": list(self.blockers),
            "output_tile_semantics": {
                "axis": OUTPUT_TILE_AXIS.value,
                "version": OUTPUT_TILE_SEMANTICS_VERSION,
            },
            "policy": self.policy.canonical_dict(),
            "ready": self.ready,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")) + "\n"


def action_registry() -> Mapping[HostActionId, HostAction]:
    """Read-only, total action registry."""

    return _ACTION_BY_ID


def _parse_action_ids(selected: Iterable[str | HostActionId]) -> tuple[HostActionId, ...]:
    if isinstance(selected, (str, bytes)):
        raise UnknownHostActionError(
            "selected host actions must be an iterable of exact action ids, not a string/path"
        )
    parsed: list[HostActionId] = []
    for raw in selected:
        if not isinstance(raw, HostActionId) and type(raw) is not str:
            raise UnknownHostActionError(
                f"host action ids must be HostActionId or exact strings, got {raw!r}"
            )
        try:
            parsed.append(raw if isinstance(raw, HostActionId) else HostActionId(raw))
        except ValueError as exc:
            raise UnknownHostActionError(
                f"unknown host action {raw!r}; known actions: "
                f"{[action.id.value for action in HOST_ACTIONS]}"
            ) from exc
    if len(set(parsed)) != len(parsed):
        duplicates = sorted({item.value for item in parsed if parsed.count(item) > 1})
        raise ConflictingHostActionsError(f"duplicate host actions selected: {duplicates}")
    selected_set = set(parsed)
    return tuple(action.id for action in HOST_ACTIONS if action.id in selected_set)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NonCanonicalHostPolicyError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_canonical_host_policy(text: str) -> ResolvedHostPolicy:
    """Parse only the exact canonical representation emitted by ``canonical_json``.

    The action ids are resolved again through the frozen catalogue.  The reconstructed
    object and bytes must then match exactly, so reordered actions, group ambiguity,
    stale statuses/effects, duplicate keys, unknown fields, and noncanonical whitespace
    cannot acquire a compiler meaning through JSON.
    """

    try:
        raw = json.loads(text, object_pairs_hook=_unique_json_object)
    except NonCanonicalHostPolicyError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise NonCanonicalHostPolicyError(f"invalid host policy JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise NonCanonicalHostPolicyError("host policy JSON must be an object")
    policy = raw.get("policy")
    if not isinstance(policy, dict):
        raise NonCanonicalHostPolicyError("host policy JSON needs an object-valued 'policy'")
    selected = policy.get("selected_actions")
    if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        raise NonCanonicalHostPolicyError("policy.selected_actions must be a list of strings")
    try:
        plan = resolve_host_policy(
            selected,
            precision=policy["precision"],
            worker_count=policy["worker_count"],
            output_tile_semantics_version=policy["output_tile_semantics_version"],
        )
    except KeyError as exc:
        raise NonCanonicalHostPolicyError(f"host policy JSON is missing {exc.args[0]!r}") from exc
    except HostActionResolutionError:
        raise
    except (TypeError, ValueError) as exc:
        raise NonCanonicalHostPolicyError(f"invalid host policy values: {exc}") from exc
    if raw != plan.canonical_dict() or text != plan.canonical_json():
        raise NonCanonicalHostPolicyError(
            "host policy JSON is not the canonical action-ordered representation"
        )
    return plan


def resolve_host_policy(
    selected: Iterable[str | HostActionId],
    *,
    precision: HostPrecision | str,
    worker_count: int = 1,
    output_tile_semantics_version: int = OUTPUT_TILE_SEMANTICS_VERSION,
) -> ResolvedHostPolicy:
    """Resolve selected action ids into one canonical, model-blind policy.

    Selection order never affects the result.  Unknown ids, duplicate/group conflicts,
    illegal precisions, unknown output-tile semantics and unimplemented actions are hard
    errors.  Partial actions produce an explicitly non-ready plan for gap accounting; the
    compiler bridge cannot materialize it.
    """

    if not isinstance(precision, HostPrecision) and type(precision) is not str:
        raise IllegalPrecisionError(
            f"host precision must be HostPrecision or an exact string, got {precision!r}"
        )
    try:
        typed_precision = (precision if isinstance(precision, HostPrecision)
                           else HostPrecision(precision))
    except ValueError as exc:
        raise IllegalPrecisionError(
            f"unknown host precision {precision!r}; known: {[p.value for p in HostPrecision]}"
        ) from exc
    if type(output_tile_semantics_version) is not int:
        raise HostActionResolutionError("output-tile semantics version must be an int")
    if output_tile_semantics_version != OUTPUT_TILE_SEMANTICS_VERSION:
        raise HostActionResolutionError(
            "unsupported output-tile semantics version "
            f"{output_tile_semantics_version}; supported version is "
            f"{OUTPUT_TILE_SEMANTICS_VERSION} ({OUTPUT_TILE_AXIS.value})"
        )

    action_ids = _parse_action_ids(selected)
    actions = tuple(_ACTION_BY_ID[action_id] for action_id in action_ids)
    selected_ids = set(action_ids)
    if HostActionId.PERSISTENT_WORKER_POOL in selected_ids \
            and HostActionId.DETERMINISTIC_STATIC_PARTITION not in selected_ids:
        raise HostActionResolutionError(
            "persistent_worker_pool requires deterministic_static_partition so the selected "
            "OpenMP runtime has compiler-emitted parallel regions to consume"
        )
    if HostActionId.DYNAMIC_VL in selected_ids and any(
            output in selected_ids for output in (
                HostActionId.OUTPUT_TILE_4,
                HostActionId.OUTPUT_TILE_8,
                HostActionId.OUTPUT_TILE_16,
            )):
        raise ConflictingHostActionsError(
            "dynamic_vl and exact output_tile actions cannot be composed by the current "
            "whole-model resolver"
        )
    grouped: dict[str, HostAction] = {}
    for action in actions:
        prior = grouped.get(action.group)
        if prior is not None:
            raise ConflictingHostActionsError(
                f"actions {prior.id.value!r} and {action.id.value!r} are mutually exclusive "
                f"members of group {action.group!r}"
            )
        grouped[action.group] = action
        if typed_precision not in action.legal_precisions:
            raise IllegalPrecisionError(
                f"action {action.id.value!r} is not legal for precision {typed_precision.value!r}; "
                f"legal precisions: {[p.value for p in action.legal_precisions]}"
            )
    unavailable = [action for action in actions
                   if action.status is MechanismStatus.UNIMPLEMENTED]
    if unavailable:
        detail = "; ".join(
            f"{action.id.value}: {', '.join(action.missing_mechanisms)}"
            for action in unavailable
        )
        raise UnimplementedHostActionError(
            "selected action(s) have no compiler mechanism and cannot be metadata-only: " + detail
        )

    fields: dict[PolicyField, str | int | bool] = {
        action.policy_field: action.policy_value for action in actions
    }
    vector_raw = fields.get(PolicyField.VECTOR_LENGTH)
    if type(worker_count) is not int:
        raise HostActionResolutionError("worker_count must be an int")
    policy = HostPolicy(
        precision=typed_precision,
        worker_count=worker_count,
        selected_actions=tuple(action.id.value for action in actions),
        vector_length=(VectorLengthIntent(str(vector_raw)) if vector_raw is not None else None),
        lmul=_optional_int(fields.get(PolicyField.LMUL)),
        output_tile=_optional_int(fields.get(PolicyField.OUTPUT_TILE)),
        output_tile_axis=OUTPUT_TILE_AXIS,
        output_tile_semantics_version=output_tile_semantics_version,
        k_unroll=_optional_int(fields.get(PolicyField.K_UNROLL)),
        rhs_panel=_optional_int(fields.get(PolicyField.RHS_PANEL)),
        fuse_legal_epilogues=bool(fields.get(PolicyField.FUSE_LEGAL_EPILOGUES, False)),
        retain_immutable_packs=bool(fields.get(PolicyField.RETAIN_IMMUTABLE_PACKS, False)),
        deterministic_static_partition=bool(fields.get(PolicyField.STATIC_PARTITION, False)),
        persistent_worker_pool=bool(fields.get(PolicyField.PERSISTENT_WORKER_POOL, False)),
    )
    return ResolvedHostPolicy(policy=policy, actions=actions)


def validate_host_policy_for_execution(policy: HostPolicy) -> ResolvedHostPolicy:
    """Re-resolve a direct policy and require exact catalogue equality and readiness."""

    if not isinstance(policy, HostPolicy):
        raise TypeError("policy must be a HostPolicy")
    canonical = resolve_host_policy(
        policy.selected_actions,
        precision=policy.precision,
        worker_count=policy.worker_count,
        output_tile_semantics_version=policy.output_tile_semantics_version,
    )
    if policy != canonical.policy:
        raise NonCanonicalHostPolicyError(
            "host policy fields do not match the selected catalogue actions"
        )
    canonical.require_ready()
    return canonical


def _optional_int(value: str | int | bool | None) -> int | None:
    return None if value is None else int(value)
