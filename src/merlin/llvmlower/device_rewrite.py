"""The record of one device rewrite: what moved, under which ABI, and what was declined.

Dependency-free data shared by the device routes and the source-composition passes that only
inspect a finished rewrite. ``merlin.llvmlower.device_offload`` performs rewrites and re-exports
these same objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BY_CONTRACTION",
    "BY_GROUP",
    "CONTRACTION_ACCESS",
    "DeviceRewrite",
    "Routed",
    "SIDECAR_NAME",
]

#: Where the rewrite records what it minted, beside the prepared module. One file per device, so two
#: devices in one system do not overwrite each other's signature set.
SIDECAR_NAME = "device_signatures.json"

#: The two units this module can move, named because they are DIFFERENT PROGRAMS. A contraction is
#: the multiply-accumulate alone, so the bias, the requantize, the activation and the pooling the
#: layer carries stay on the host; a closed compute group is the layer, readout and all. A build that
#: reported only "N kernels" over a mix of the two would say nothing about which of a model's layers
#: kept their readout -- which is why the granularity is a parameter, recorded in the sidecar, rather
#: than a property of whichever caller happened to run.
BY_CONTRACTION = "contraction"
BY_GROUP = "group"

#: The per-argument ``bufferization.access`` of a contraction's callee: two operands read, the
#: destination written. A group's call derives its own per symbol (``DeviceRewrite.arg_access``).
CONTRACTION_ACCESS = ("read", "read", "write")

@dataclass(frozen=True)
class Routed:
    """One contraction -- or one closed compute group -- that was moved, and where it went."""

    symbol: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...]
    dtypes: tuple[str, str, str]
    fqn: str = ""
    operation_id: str = ""
    #: The compute group this call is, when the route was ``BY_GROUP``; ``None`` per contraction.
    group: int | None = None
    source_operation_ordinal: int = -1
    source_region: str = ""
    tensor_types: tuple[str, str, str] = ()
    #: Exact captured root identity, when complete provenance survived preparation.
    source_region_id: str | None = None
    source_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeviceRewrite:
    """What the rewrite did, and -- as importantly -- what it declined and why."""

    device: str
    routed: tuple[Routed, ...] = ()
    #: symbol -> (parallel..., reduction) extents, the callee's identity.
    signatures: dict[str, tuple[int, ...]] = field(default_factory=dict)
    #: (symbol-or-"all", reason). A decline is reported, never silent.
    skipped: tuple[tuple[str, str], ...] = ()
    package_sha256: str | None = None
    transport: str | None = None
    abi_sha256: str | None = None
    certification_sha256: tuple[str, ...] = ()
    release_review_digest: str | None = None
    software_spec_sha256: str | None = None
    capability_contract_sha256: str | None = None
    #: symbol -> exact Phase 0 interface bytes and hash, if selected by operation ID.
    expected_interfaces: dict[str, dict[str, str]] = field(default_factory=dict)
    #: :data:`BY_CONTRACTION` or :data:`BY_GROUP` -- what unit was moved. See the constants.
    granularity: str = BY_CONTRACTION
    #: symbol -> the group's own stated program (``group_command.GroupProgram.to_dict()``), whose
    #: ``entry`` carries the epilogue, the multiplier, the convolution geometry and the committed
    #: type. Empty for a contraction route, which states no program: there is nothing but extents.
    programs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: symbol -> the corpus entry a backend is asked to emit for that symbol. The map
    #: :func:`merlin.llvmlower.device_build.build_device_objects` consumes as ``entries=``.
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The ``group_prepack.prepack`` record for the routed groups, or ``None`` when no weights
    #: manifest was readable -- which is a different fact from "no group needed a prepack".
    prepack: dict[str, Any] | None = None
    #: symbol -> per-argument ``bufferization.access``. The contraction route mints one fixed shape
    #: (read, read, write); a group's call takes whatever free values its members read, so the
    #: accesses are DERIVED per symbol and the printer repair has to be told them.
    arg_access: dict[str, tuple[str, ...]] = field(default_factory=dict)
    model_sha256: str | None = None
    #: symbol -> the host call's operands under the logical kernel ABI, in call order: each
    #: ``{"role", "shape", "dtype"}`` (activation, stored weight, folded bias, destinations). Empty
    #: for a call that passes the fixed (lhs, rhs, out) triple.
    call_buffers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: The laid-out weights bound as trailing ``@forward`` arguments (:mod:`merlin.llvmlower.blob_weights`),
    #: or ``None`` when they stayed constants.
    blob_weights: Any = None
    #: Original checked placement record; absent on historical selector routes.
    offload_selection: dict | None = None

    @property
    def moved(self) -> int:
        return len(self.routed)

    def to_sidecar(self) -> dict[str, Any]:
        """The sidecar's content, as plain JSON-able data."""
        record = {
            "device": self.device,
            "model_sha256": self.model_sha256,
            "granularity": self.granularity,
            "signatures": {s: list(k) for s, k in self.signatures.items()},
            "routed": [
                {
                    "symbol": r.symbol,
                    "parallel": list(r.parallel),
                    "reduction": list(r.reduction),
                    "dtypes": list(r.dtypes),
                    "fqn": r.fqn,
                    "operation_id": r.operation_id,
                    "group": r.group,
                    "source_operation_ordinal": r.source_operation_ordinal,
                    "source_region": r.source_region,
                    "tensor_types": list(r.tensor_types),
                    **(
                        {"source_region_id": r.source_region_id, "source_node_ids": list(r.source_node_ids)}
                        if r.source_region_id is not None and r.source_node_ids
                        else {}
                    ),
                }
                for r in self.routed
            ],
            "package_sha256": self.package_sha256,
            "transport": self.transport,
            "abi_sha256": self.abi_sha256,
            "certification_sha256": list(self.certification_sha256),
            "release_review_digest": self.release_review_digest,
            "software_spec_sha256": self.software_spec_sha256,
            "capability_contract_sha256": self.capability_contract_sha256,
            "expected_interfaces": self.expected_interfaces,
            # THE STATEMENT TRAVELS WITH THE SIGNATURES, for the reason the signatures do at all: the
            # rewrite runs inside the lowering subprocess and the build that generates the callee runs
            # outside it. A build handed only extents rebuilds a bare contraction and drops whatever
            # readout the layer carries, and nothing in the artifact would say so.
            "programs": {s: dict(p) for s, p in self.programs.items()},
            "entries": {s: dict(e) for s, e in self.entries.items()},
            "prepack": self.prepack,
            "arg_access": {s: list(a) for s, a in self.arg_access.items()},
            "call_buffers": {s: [dict(b) for b in bs] for s, bs in self.call_buffers.items()},
            "skipped": [[s, why] for s, why in self.skipped],
        }
        if self.offload_selection is not None:
            record["offload_selection"] = self.offload_selection
        return record

    def write_sidecar(self, directory: str | Path) -> Path:
        path = Path(directory) / SIDECAR_NAME
        path.write_text(json.dumps(self.to_sidecar(), indent=1), encoding="utf-8")
        if self.blob_weights is not None:
            self.blob_weights.write(directory)
        return path
