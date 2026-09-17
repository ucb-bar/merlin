"""Typed, model-blind policy input for whole-model CPU lowering.

This module is deliberately smaller than a target package.  It records the host-side
decisions selected by mining without referring to a model, capsule, or target-specific
shape.  :mod:`merlin.mining.host_actions` owns the frozen action catalogue and produces
these policies; this module owns the compiler-side types and the bridge to mechanisms
that already exist.

An intent is not evidence that a mechanism exists.  ``materialize_lowering_effects``
therefore implements only lowering seams that are present today:

* dynamic VL on the target-agnostic ``MicrokernelSpec`` used by the RVV package resolver; and
* ``parallel_harts`` used by the whole-model OpenMP/static-partition pipeline.

Every other field fails closed.  In particular, a named action never becomes a piece of
metadata which a later report could mistake for an implemented compiler change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..kernels.microkernel import VL_DYNAMIC, VL_STRATEGIES, MicrokernelSpec


class HostPrecision(str, Enum):
    """Compiled arithmetic contract, using the package vocabulary already in use."""

    FP32 = "fp32"
    INT8_W8A8 = "int8_w8a8"
    BF16_F32ACC = "bf16_f32acc"
    FP16_F32ACC = "fp16_f32acc"


class VectorLengthIntent(str, Enum):
    DYNAMIC = "dynamic"
    FIXED_256_GUARDED = "fixed_256_guarded"


class OutputTileAxis(str, Enum):
    """Meaning of ``HostPolicy.output_tile``.

    Version 1 is the innermost PARALLEL output dimension of a contraction: N for
    ``linalg.matmul`` and ``linalg.batch_matmul``.  It is not an output-area tile, an M
    tile, a reduction tile, or a byte count.  Version 1 asks for an *exact* value on
    this axis.  The existing whole-model ``MicrokernelSpec.NR`` resolver treats NR as
    a shape-dependent upper bound, so it is not by itself an implementation of this
    contract.
    """

    CONTRACTION_INNER_PARALLEL = "contraction_inner_parallel"


OUTPUT_TILE_SEMANTICS_VERSION = 1
OUTPUT_TILE_AXIS = OutputTileAxis.CONTRACTION_INNER_PARALLEL


@dataclass(frozen=True)
class HostPolicy:
    """Immutable compiler intent resolved from a set of host optimization actions.

    ``selected_actions`` is already in catalogue order.  Optional values mean the
    corresponding group was not selected; they do not invent a default action.
    """

    precision: HostPrecision
    worker_count: int
    selected_actions: tuple[str, ...]
    vector_length: VectorLengthIntent | None = None
    lmul: int | None = None
    output_tile: int | None = None
    output_tile_axis: OutputTileAxis = OUTPUT_TILE_AXIS
    output_tile_semantics_version: int = OUTPUT_TILE_SEMANTICS_VERSION
    k_unroll: int | None = None
    rhs_panel: int | None = None
    fuse_legal_epilogues: bool = False
    retain_immutable_packs: bool = False
    deterministic_static_partition: bool = False
    persistent_worker_pool: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.precision, HostPrecision):
            raise TypeError("precision must be a HostPrecision")
        if isinstance(self.worker_count, bool) or not isinstance(self.worker_count, int):
            raise TypeError("worker_count must be an int")
        if self.worker_count < 1:
            raise ValueError(f"worker_count must be positive, got {self.worker_count!r}")
        if not isinstance(self.selected_actions, tuple) or any(
                type(action) is not str or not action for action in self.selected_actions):
            raise TypeError("selected_actions must be a tuple of non-empty strings")
        if len(set(self.selected_actions)) != len(self.selected_actions):
            raise ValueError("selected_actions must not contain duplicates")
        if type(self.output_tile_semantics_version) is not int:
            raise TypeError("output_tile_semantics_version must be an int")
        if self.output_tile_semantics_version != OUTPUT_TILE_SEMANTICS_VERSION:
            raise ValueError(
                "unsupported output-tile semantics version "
                f"{self.output_tile_semantics_version}; supported version is "
                f"{OUTPUT_TILE_SEMANTICS_VERSION}"
            )
        if self.output_tile_axis is not OUTPUT_TILE_AXIS:
            raise ValueError(
                f"unsupported output-tile axis {self.output_tile_axis!r}; "
                f"version {OUTPUT_TILE_SEMANTICS_VERSION} requires {OUTPUT_TILE_AXIS.value!r}"
            )
        for name in ("lmul", "output_tile", "k_unroll", "rhs_panel"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an int or None")
                if value < 1:
                    raise ValueError(f"{name} must be positive, got {value!r}")
        if self.vector_length is not None and not isinstance(
                self.vector_length, VectorLengthIntent):
            raise TypeError("vector_length must be a VectorLengthIntent or None")
        for name in ("fuse_legal_epilogues", "retain_immutable_packs",
                     "deterministic_static_partition", "persistent_worker_pool"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        # These are the explicit arithmetic contracts supported by the selected
        # mechanisms.  The action resolver checks the same sets; keeping this guard
        # at the compiler input type prevents a directly-constructed policy from
        # bypassing action legality.
        microkernel_precisions = (HostPrecision.FP32, HostPrecision.INT8_W8A8)
        if self.vector_length is VectorLengthIntent.DYNAMIC \
                and self.precision is not HostPrecision.FP32:
            raise ValueError("dynamic vector length is implemented only for fp32")
        if any(value is not None for value in (
                self.output_tile, self.k_unroll, self.rhs_panel)) \
                and self.precision not in microkernel_precisions:
            raise ValueError(
                "output tiling, K unrolling, and RHS panels are legal only for "
                "fp32 or int8_w8a8"
            )
        if self.retain_immutable_packs and self.precision not in microkernel_precisions:
            raise ValueError("immutable packs are legal only for fp32 or int8_w8a8")
        if self.vector_length is VectorLengthIntent.DYNAMIC and self.output_tile is not None:
            raise ValueError(
                "dynamic vector length and exact output tiling cannot be composed by the "
                "current whole-model resolver"
            )
        if (self.deterministic_static_partition or self.persistent_worker_pool) \
                and self.worker_count < 2:
            raise ValueError("parallel runtime actions require worker_count >= 2")

    def canonical_dict(self) -> dict[str, object]:
        """Plain deterministic representation; no enum or set ordering leaks into it."""

        return {
            "fuse_legal_epilogues": self.fuse_legal_epilogues,
            "k_unroll": self.k_unroll,
            "lmul": self.lmul,
            "output_tile": self.output_tile,
            "output_tile_axis": self.output_tile_axis.value,
            "output_tile_semantics_version": self.output_tile_semantics_version,
            "persistent_worker_pool": self.persistent_worker_pool,
            "precision": self.precision.value,
            "retain_immutable_packs": self.retain_immutable_packs,
            "rhs_panel": self.rhs_panel,
            "selected_actions": list(self.selected_actions),
            "static_partition": self.deterministic_static_partition,
            "vector_length": self.vector_length.value if self.vector_length else None,
            "worker_count": self.worker_count,
        }


class HostPolicyMechanismError(ValueError):
    """A policy cannot be materialized by the existing whole-model compiler seams."""


@dataclass(frozen=True)
class HostLoweringEffects:
    """Existing compiler inputs produced by a ready host policy.

    ``microkernel`` is consumed by the existing target resolver and
    ``parallel_harts`` by ``llvmlower.lower``.  ``mechanisms`` is descriptive
    provenance for these concrete effects, not a substitute for either effect.
    """

    microkernel: MicrokernelSpec | None
    parallel_harts: int | None
    parallel_schedule: str | None
    mechanisms: tuple[str, ...]
    openmp_runtime: str | None = None
    worker_pool_size: int | None = None

    def __post_init__(self) -> None:
        if self.microkernel is not None and not isinstance(self.microkernel, MicrokernelSpec):
            raise TypeError("microkernel must be a MicrokernelSpec or None")
        if self.microkernel is not None:
            _validate_microkernel_spec(self.microkernel, name="microkernel")
        if self.parallel_harts is not None and (
                isinstance(self.parallel_harts, bool)
                or not isinstance(self.parallel_harts, int)
                or self.parallel_harts < 2):
            raise ValueError("parallel_harts must be None or an int >= 2")
        if self.parallel_schedule is not None and (
                type(self.parallel_schedule) is not str or not self.parallel_schedule):
            raise TypeError("parallel_schedule must be None or a non-empty string")
        if (self.parallel_harts is None) != (self.parallel_schedule is None):
            raise ValueError(
                "parallel_harts and parallel_schedule must be present or absent together"
            )
        if self.openmp_runtime is not None and self.openmp_runtime != "merlin_pthread_pool":
            raise ValueError(
                "openmp_runtime must be None or the concrete 'merlin_pthread_pool' provider"
            )
        if self.worker_pool_size is not None and (
                isinstance(self.worker_pool_size, bool)
                or not isinstance(self.worker_pool_size, int)
                or self.worker_pool_size < 2):
            raise ValueError("worker_pool_size must be None or an int >= 2")
        if (self.openmp_runtime is None) != (self.worker_pool_size is None):
            raise ValueError(
                "openmp_runtime and worker_pool_size must be present or absent together"
            )
        if self.openmp_runtime is not None and self.parallel_harts is None:
            raise ValueError("the persistent OpenMP runtime requires a parallel lowering")
        if not isinstance(self.mechanisms, tuple) or any(
                type(name) is not str or not name for name in self.mechanisms):
            raise TypeError("mechanisms must be a tuple of non-empty strings")
        if len(set(self.mechanisms)) != len(self.mechanisms):
            raise ValueError("mechanisms must not contain duplicates")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "mechanisms": list(self.mechanisms),
            "microkernel": self.microkernel.to_knobs() if self.microkernel else None,
            "openmp_runtime": self.openmp_runtime,
            "parallel_harts": self.parallel_harts,
            "parallel_schedule": self.parallel_schedule,
            "worker_pool_size": self.worker_pool_size,
        }


def _validate_microkernel_spec(spec: MicrokernelSpec, *, name: str) -> None:
    """Reject permissively-constructed ``MicrokernelSpec`` objects at this boundary."""

    for field in ("MR", "NR", "KC"):
        value = getattr(spec, field)
        if type(value) is not int or value < 1:
            raise TypeError(f"{name}.{field} must be a positive int")
    for field in ("unroll_m", "pack", "k_block"):
        if type(getattr(spec, field)) is not bool:
            raise TypeError(f"{name}.{field} must be a bool")
    if type(spec.vl_strategy) is not str or spec.vl_strategy not in VL_STRATEGIES:
        raise TypeError(f"{name}.vl_strategy must be one of {VL_STRATEGIES!r}")


def materialize_lowering_effects(
    policy: HostPolicy,
    *,
    base_microkernel: MicrokernelSpec | None = None,
) -> HostLoweringEffects:
    """Turn a catalogue-resolved policy into verified existing lowering inputs.

    The micro-kernel decisions are patches over an explicit base.  Requiring the base is
    important: constructing ``MicrokernelSpec()`` here would silently select MR/KC/packing
    decisions which were not among the chosen actions.

    This is a public compiler execution boundary.  It deliberately imports the action
    resolver lazily and reconstructs the policy from its selected action ids before any
    effect is emitted.  A caller therefore cannot use a hand-built ``HostPolicy`` to
    smuggle unknown/model-specific actions, mismatched fields, or illegal precision
    combinations past the frozen catalogue.
    """

    if not isinstance(policy, HostPolicy):
        raise TypeError("policy must be a HostPolicy")
    if base_microkernel is not None:
        if not isinstance(base_microkernel, MicrokernelSpec):
            raise TypeError("base_microkernel must be a MicrokernelSpec or None")
        _validate_microkernel_spec(base_microkernel, name="base_microkernel")

    # Lazy to avoid a module-import cycle: host_actions owns catalogue authority,
    # while this module owns the effect types and low-level materializer.
    from ..mining.host_actions import validate_host_policy_for_execution

    plan = validate_host_policy_for_execution(policy)
    effects = _materialize_validated_lowering_effects(
        plan.policy, base_microkernel=base_microkernel)
    plan.validate_materialized_effects(effects)
    return effects


def _materialize_validated_lowering_effects(
    policy: HostPolicy,
    *,
    base_microkernel: MicrokernelSpec | None,
) -> HostLoweringEffects:
    """Materialize an already catalogue-revalidated, ready policy."""

    unsupported: list[str] = []
    if policy.vector_length is VectorLengthIntent.FIXED_256_GUARDED:
        unsupported.append("fixed_vlen_256_guarded")
    if policy.lmul is not None:
        unsupported.append("exact_lmul")
    if policy.output_tile is not None:
        unsupported.append("exact_output_tile")
    if policy.k_unroll is not None:
        unsupported.append("reduction_unroll")
    if policy.rhs_panel is not None:
        unsupported.append("rhs_panel")
    if policy.fuse_legal_epilogues:
        unsupported.append("legality_scoped_epilogue_fusion")
    if policy.retain_immutable_packs:
        unsupported.append("session_resident_immutable_pack")
    if unsupported:
        raise HostPolicyMechanismError(
            "host policy needs compiler/runtime mechanisms not present in the bridge: "
            + ", ".join(unsupported)
        )

    microkernel = base_microkernel
    mechanisms: list[str] = []
    needs_microkernel = policy.vector_length is VectorLengthIntent.DYNAMIC
    if needs_microkernel and microkernel is None:
        raise HostPolicyMechanismError(
            "dynamic VL actions patch the existing micro-kernel policy; "
            "an explicit base_microkernel is required"
        )
    if policy.vector_length is VectorLengthIntent.DYNAMIC:
        assert microkernel is not None
        microkernel = microkernel.with_(vl_strategy=VL_DYNAMIC)
        mechanisms.append("kernels.microkernel.vl_strategy")
    parallel_harts = None
    parallel_schedule = None
    openmp_runtime = None
    worker_pool_size = None
    if policy.deterministic_static_partition:
        parallel_harts = policy.worker_count
        parallel_schedule = "exact_cover"
        mechanisms += [
            "llvmlower.lower.parallel_harts",
            "runtime.omp_static_schedule.exact_cover",
        ]
    if policy.persistent_worker_pool:
        openmp_runtime = "merlin_pthread_pool"
        worker_pool_size = policy.worker_count
        mechanisms += [
            "runtime.libomp_pthread.openmp_abi",
            "runtime.libomp_pthread.session_resident_pool",
        ]
    return HostLoweringEffects(
        microkernel=microkernel,
        parallel_harts=parallel_harts,
        parallel_schedule=parallel_schedule,
        openmp_runtime=openmp_runtime,
        worker_pool_size=worker_pool_size,
        mechanisms=tuple(mechanisms),
    )
