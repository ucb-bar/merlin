"""Bind an admitted offer through outlining and admit its actual product receipt.

Receipts establish compiler custody and explicit realization claims. Independent
semantic verification and execution admission remain separate authorities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from merlin.compiler.invocation import request_identity


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class LoweringTicket:
    partition: object
    admitted: object
    region: object
    entry: str
    _request: tuple
    _record: str
    _owners: tuple

    @classmethod
    def bind(cls, partition, region, entry, request):
        partition.revalidate()
        item = next((item for item in partition.selection.selected
                     if item.alternative.id == region.alternative), None)
        if item is None or item.proposal.lowering is None:
            raise ValueError("selected lowering requires an original recipe-bearing offer")
        ticket = cls(partition, item, region, entry, request_identity(request), "",
                     (partition, item, region))
        object.__setattr__(ticket, "_record", _json(ticket._describe()))
        ticket.revalidate(request)
        return ticket

    def _describe(self):
        selection, proposal, region = self.partition.selection, self.admitted.proposal, self.region
        return {
            "schema": "merlin.selected_lowering.v1", "proposal": asdict(proposal),
            "topology_sha256": selection.topology.sha256,
            "storage_bound_bytes": selection.storage_bound_bytes,
            "storage_scope": "source liveness plus maximum serial scratch; local device resources checked by backend",
            "partition": {"alternative": region.alternative, "provider": region.provider,
                          "symbol": region.symbol, "source_ordinals": region.source_ordinals,
                          "inputs": region.inputs, "outputs": region.outputs, "entry": self.entry},
            "source_observations": {"proposal": self.admitted.source_observation(),
                                    "partition": self.partition.source_observation()},
            "request": (self._request[0], str(self._request[1]), self._request[2]),
        }

    @property
    def sha256(self):
        return hashlib.sha256(self._record.encode()).hexdigest()

    def record(self):
        return json.loads(self._record)

    def revalidate(self, request):
        if any(a is not b for a, b in zip((self.partition, self.admitted, self.region), self._owners, strict=True)):
            raise ValueError("selected lowering ticket owner changed")
        self.partition.revalidate()
        self.admitted.revalidate()
        if (not any(item is self.admitted for item in self.partition.selection.selected)
                or not any(region is self.region for region in self.partition.regions)):
            raise ValueError("selected lowering ticket lost original partition ownership")
        proposal, region = self.admitted.proposal, self.region
        if (proposal.lowering is None or region.alternative != proposal.alternative.id
                or region.provider != proposal.provider or region.source_ordinals != proposal.operation_ordinals
                or region.inputs != tuple(rep.buffer for rep in proposal.alternative.inputs)
                or region.outputs != tuple(rep.buffer for rep in proposal.alternative.outputs)):
            raise ValueError("selected lowering ticket differs from selected source/ports/recipe")
        if request_identity(request) != self._request or _json(self._describe()) != self._record:
            raise ValueError("selected lowering ticket or request changed")
        if region.request_source(self.entry).text != request.source.text:
            raise ValueError("selected lowering body differs from original outlined source")

    def acknowledgment(self, request, result):
        """Provider's explicit receipt over actual artifacts, never a correctness certificate."""
        self.revalidate(request)
        result.verify()
        return {"ticket_sha256": self.sha256,
                "recipe": asdict(self.admitted.proposal.lowering),
                "artifacts": [{"path": a.path.relative_to(result.directory).as_posix(),
                               "kind": a.kind, "role": a.role, "sha256": a.sha256} for a in result.artifacts]}

    def admit(self, request, result):
        actual = self.acknowledgment(request, result)
        # Normalize immutable tuples to their portable JSON representation.
        if _json(result.metadata.get("selected_lowering")) != _json(actual):
            raise ValueError("backend omitted or changed the exact selected lowering acknowledgment/products")
        required = set(self.admitted.proposal.lowering.artifacts)
        if not required <= {(a.kind, a.role) for a in result.artifacts}:
            raise ValueError("backend omitted selected lowering artifact kinds/roles")
