"""Gemmini physical-layout alternatives backed by supplied hardware facts.

The target adapter contains the target vocabulary and the q545-derived cost guard.  It does not
inspect model names or shapes beyond the convolution's general batch extent.  Mesh geometry and
evidence identities are inputs so a configuration change cannot silently inherit a stale rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from merlin.perf.physical_layout import (
    CapacityDemand,
    LayoutOp,
    LayoutRefusal,
    OperatorEncoding,
    PhysicalLayoutError,
)


UNDERFILLED_TRANSPOSED_INPUT_REFUSAL = (
    "hardware_cost_guard_transposed_nchw_underfills_systolic_rows"
)


@dataclass(frozen=True)
class NativeConvLayoutFacts:
    """Configuration-bound facts needed to expose native convolution alternatives."""

    mesh_rows: int
    capabilities: tuple[str, ...]
    non_transposed_encoding: str
    transposed_encoding: str
    parameter_encoding: str
    configuration_identity: str
    cost_guard_evidence: str | None
    cost_guard_configuration_identity: str | None
    transient_storage_space: str | None = None
    transient_capacity_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.mesh_rows) is not int or self.mesh_rows <= 0:
            raise PhysicalLayoutError(
                "native convolution mesh_rows must be a positive derived fact")
        if (not self.non_transposed_encoding.strip() or not self.transposed_encoding.strip()
                or not self.parameter_encoding.strip()):
            raise PhysicalLayoutError("native convolution encodings must be named")
        if not self.configuration_identity.strip():
            raise PhysicalLayoutError("native convolution configuration identity is required")
        if self.non_transposed_encoding == self.transposed_encoding:
            raise PhysicalLayoutError("transposed and non-transposed encodings must differ")
        if (any(not item.strip() for item in self.capabilities)
                or len(set(self.capabilities)) != len(self.capabilities)):
            raise PhysicalLayoutError("native convolution capabilities must be unique names")
        if type(self.transient_capacity_bytes) is not int or self.transient_capacity_bytes < 0:
            raise PhysicalLayoutError("native convolution transient capacity must be nonnegative")
        if (self.transient_storage_space is None) != (self.transient_capacity_bytes == 0):
            raise PhysicalLayoutError(
                "transient storage space must be supplied exactly when its demand is nonzero")


@dataclass(frozen=True)
class NativeConvLayoutRequest:
    name: str
    activation: str
    parameters: tuple[str, ...]
    outputs: tuple[str, ...]
    batch: int
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not self.name.strip() or not self.activation.strip()
                or not self.parameters or not self.outputs):
            raise PhysicalLayoutError(
                "native convolution request requires name, activation, parameters, and outputs")
        if type(self.batch) is not int or self.batch <= 0:
            raise PhysicalLayoutError("native convolution batch must be a positive static extent")
        if not self.provenance or any(not item.strip() for item in self.provenance):
            raise PhysicalLayoutError("native convolution request requires source provenance")


@dataclass(frozen=True)
class NativeConvLayoutDecision:
    op: LayoutOp | None
    refusals: tuple[LayoutRefusal, ...]
    receipt: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_encodings": (
                [option.encoding_id for option in self.op.options] if self.op is not None else []),
            "refusals": [item.to_dict() for item in self.refusals],
            "receipt": dict(self.receipt),
        }


def native_convolution_layout_decision(
        request: NativeConvLayoutRequest, facts: NativeConvLayoutFacts) -> NativeConvLayoutDecision:
    """Return exact native choices and refuse a hardware-proven underfilled transpose.

    The non-transposed input route receives one preference point because it can be propagated
    through the graph.  The lexicographic target-neutral planner will therefore retain that layout
    across compatible producers, residual branches, and consumers before minimizing conversions.
    """

    capabilities = set(facts.capabilities)
    refusals: list[LayoutRefusal] = []
    options: list[OperatorEncoding] = []
    demands = ()
    if facts.transient_storage_space is not None:
        demands = (CapacityDemand(
            facts.transient_storage_space,
            facts.transient_capacity_bytes,
            "configuration-derived native convolution working-set bound",
        ),)
    parameter_encodings = tuple(
        (name, facts.parameter_encoding) for name in request.parameters)

    if "native_conv_non_transposed_input" in capabilities:
        options.append(OperatorEncoding(
            "native_conv_non_transposed_input",
            facts.non_transposed_encoding,
            preference_weight=1,
            required_capabilities=("native_conv_non_transposed_input",),
            capacity_demands=demands,
            port_encodings=parameter_encodings,
            provenance=("target native convolution non-transposed input capability",),
        ))
    else:
        refusals.append(LayoutRefusal(
            request.name,
            "native_non_transposed_input_unavailable",
            "target facts do not expose a non-transposed native convolution input path",
            request.provenance,
        ))

    active_rows = min(request.batch, facts.mesh_rows)
    if "native_conv_transposed_input" in capabilities:
        if active_rows < facts.mesh_rows:
            evidence_matches = (
                facts.cost_guard_evidence is not None
                and facts.cost_guard_evidence.strip()
                and facts.cost_guard_configuration_identity == facts.configuration_identity
            )
            if not evidence_matches:
                refusals.append(LayoutRefusal(
                    request.name,
                    "missing_hardware_cost_guard_evidence",
                    "transposed native convolution underfills the mesh, but no configuration-bound "
                    "hardware evidence licenses an automatic cost decision",
                    request.provenance,
                ))
            else:
                refusals.append(LayoutRefusal(
                    request.name,
                    UNDERFILLED_TRANSPOSED_INPUT_REFUSAL,
                    f"transposed input exposes {active_rows} of {facts.mesh_rows} systolic rows",
                    (facts.cost_guard_evidence,),
                ))
        else:
            options.append(OperatorEncoding(
                "native_conv_transposed_input",
                facts.transposed_encoding,
                required_capabilities=("native_conv_transposed_input",),
                capacity_demands=demands,
                port_encodings=parameter_encodings,
                provenance=("target native convolution transposed input capability",),
            ))
    else:
        refusals.append(LayoutRefusal(
            request.name,
            "native_transposed_input_unavailable",
            "target facts do not expose a transposed native convolution input path",
            request.provenance,
        ))

    op = None
    if options:
        op = LayoutOp(
            request.name,
            "native_convolution",
            (request.activation, *request.parameters),
            request.outputs,
            True,
            tuple(options),
            request.provenance,
            (request.activation, *request.outputs),
        )
    receipt = (
        ("active_transposed_systolic_rows", active_rows),
        ("available_systolic_rows", facts.mesh_rows),
        ("batch", request.batch),
        ("configuration_identity", facts.configuration_identity),
        ("cost_guard_evidence", facts.cost_guard_evidence),
        ("cost_guard_configuration_identity", facts.cost_guard_configuration_identity),
        ("graph_propagated_non_transposed_preferred", True),
        ("transposed_systolic_row_utilization", active_rows / facts.mesh_rows),
    )
    return NativeConvLayoutDecision(op, tuple(refusals), receipt)
