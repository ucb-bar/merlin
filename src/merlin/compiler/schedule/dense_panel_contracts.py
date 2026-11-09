"""Explicit target-owned issue, configuration and completion premises for dense panels."""

from __future__ import annotations

from dataclasses import dataclass

from merlin.compiler.schedule.dense_contracts import DenseTiledIntegerReadout


@dataclass(frozen=True)
class DensePanelOverlapAdmission:
    """Original selected implementation premises, never inferred from capacity.

    Qualification must cover ordered issue/backpressure, disjoint load/execute
    overlap, load configuration snapshotting, paired preload/compute consumption,
    disjoint output compute issue and full-device visibility before drain return.
    Bytes/identities alone are not qualification. The normal execution owner
    supplies and freezes this contract; candidate IR cannot grant itself one.
    Issued counts are finite compiler epoch budgets; pending counts are selected
    physical queue bounds. Qualified ready/valid backpressure enforces pending
    capacity while a larger epoch issues, without fabricated completion tokens.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    max_issued_compute_pairs: int
    max_issued_loads: int
    pending_compute_pairs: int
    pending_loads: int
    execution: str = "ordered_issue_disjoint_load_compute_full_drain"
    load_configuration: str = "ordered_snapshot_before_data_issue"
    descriptor: str = "preload_immediately_paired_with_compute"
    max_issued_readouts: int = 1
    pending_readouts: int = 1
    readout_configuration: str = "separate_drained_readout"

    def __post_init__(self):
        if (
            any(
                type(v) is not str or not v
                for v in (self.hardware_identity, self.numerical_policy_identity, self.qualification_identity)
            )
            or any(
                type(v) is not int or v <= 0
                for v in (
                    self.max_issued_compute_pairs,
                    self.max_issued_loads,
                    self.pending_compute_pairs,
                    self.pending_loads,
                    self.max_issued_readouts,
                    self.pending_readouts,
                )
            )
            or self.execution != "ordered_issue_disjoint_load_compute_full_drain"
            or self.load_configuration != "ordered_snapshot_before_data_issue"
            or self.descriptor != "preload_immediately_paired_with_compute"
            or self.readout_configuration not in ("separate_drained_readout", "ordered_snapshot_before_store_issue")
            or self.max_issued_readouts > 1
            and self.readout_configuration != "ordered_snapshot_before_store_issue"
        ):
            raise ValueError("panel overlap requires explicit selected issue, configuration and completion premises")


@dataclass(frozen=True)
class DenseReadoutConfigurationSnapshot:
    """Target-owned capture of each exact tiled readout's numerical state.

    The selected implementation must consume configuration in store order and
    retain that store's scale/activation through backpressure and completion.
    This does not permit combining differently configured stores, changing
    coefficients, or issuing a readout before its accumulator producer.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    readout_identity: str
    execution: str = "ordered_per_store_snapshot_until_completion"

    def validate_for(self, adapter, readout):
        if (
            any(
                type(value) is not str or not value.strip()
                for value in (
                    self.hardware_identity,
                    self.numerical_policy_identity,
                    self.qualification_identity,
                    self.readout_identity,
                )
            )
            or self.hardware_identity != adapter.configuration_identity
            or self.numerical_policy_identity != adapter.numerical_policy_identity
            or type(readout) is not DenseTiledIntegerReadout
            or self.readout_identity != readout.identity
            or self.execution != "ordered_per_store_snapshot_until_completion"
        ):
            raise ValueError("varying readout requires its exact selected configuration snapshot mapping")
        readout.__post_init__()


@dataclass(frozen=True)
class DenseReadoutComputeAdmission:
    """Additional target conjunction; pure readout/reduction laws do not imply it.

    A store reads a completed ACC region while computation writes a disjoint
    region. Accepted requests retain their numerical configuration, finite
    queues apply backpressure, and one full drain closes both lifetimes before
    either region is reused. The source owner must qualify this conjunction.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    store_qualification_identity: str
    max_issued_readouts: int
    pending_readouts: int
    execution: str = "disjoint_accumulator_readout_compute_full_drain"
    configuration_snapshot: DenseReadoutConfigurationSnapshot | None = None

    def validate_for(self, adapter, original):
        if (
            any(
                type(value) is not str or not value.strip()
                for value in (
                    self.hardware_identity,
                    self.numerical_policy_identity,
                    self.qualification_identity,
                    self.store_qualification_identity,
                )
            )
            or any(type(value) is not int or value <= 0 for value in (self.max_issued_readouts, self.pending_readouts))
            or self.execution != "disjoint_accumulator_readout_compute_full_drain"
            or self.hardware_identity != adapter.configuration_identity
            or self.numerical_policy_identity != adapter.numerical_policy_identity
            or self.hardware_identity != original.hardware_identity
            or self.numerical_policy_identity != original.numerical_policy_identity
            or self.max_issued_readouts > original.max_issued_readouts
            or self.pending_readouts != original.pending_readouts
            or original.readout_configuration != "ordered_snapshot_before_store_issue"
        ):
            raise ValueError("readout/compute needs the selected separate snapshot and completion conjunction")


@dataclass(frozen=True)
class DenseReadyReadoutAdmission:
    """Separate causal ACC producer-to-readout completion capability.

    Ordered issue alone is insufficient. The selected implementation must keep
    overlapping stores dependent on producer completion and pending ACC writes,
    snapshot each readout, apply finite-queue backpressure, and fully drain before
    ownership reuse. No compiler-chosen cycle spacing is part of this contract.
    Varying tiled numerical state additionally needs ``configuration_snapshot``
    bound to that complete original readout; uniform-only callers grant none.
    Concurrent disjoint input loads additionally require the separately selected
    ``disjoint_load_qualification_identity`` for the complete LD/compute/readout
    conjunction. Its absence keeps the original refusal of mixed load epochs.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    store_qualification_identity: str
    max_issued_readouts: int
    pending_readouts: int
    execution: str = "ordered_accumulator_raw_readout_full_drain"
    configuration_snapshot: DenseReadoutConfigurationSnapshot | None = None
    disjoint_load_qualification_identity: str | None = None

    def validate_for(self, adapter, original):
        if self.execution != "ordered_accumulator_raw_readout_full_drain":
            raise ValueError("causal readout requires the separate producer completion execution law")
        if self.disjoint_load_qualification_identity is not None and (
            type(self.disjoint_load_qualification_identity) is not str
            or not self.disjoint_load_qualification_identity.strip()
        ):
            raise ValueError("ready readout/load overlap needs its separate selected execution conjunction")
        DenseReadoutComputeAdmission(
            self.hardware_identity,
            self.numerical_policy_identity,
            self.qualification_identity,
            self.store_qualification_identity,
            self.max_issued_readouts,
            self.pending_readouts,
            configuration_snapshot=self.configuration_snapshot,
        ).validate_for(adapter, original)
