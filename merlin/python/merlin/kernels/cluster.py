"""Cluster dossiers by their static-fact signature so the agent reviews ONE representative per
cluster (~dozens) instead of every kernel (hundreds). The signature (dossier.signature()) groups
kernels that make the same RVV decisions + structure; within a cluster they differ only in shape/
naming, so a single representative carries the cluster's lesson. This is what makes the
representative agent mode cheap and consistent (vs per-kernel)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dossier import KernelDossier


@dataclass
class Cluster:
    signature: tuple
    members: list[KernelDossier] = field(default_factory=list)

    @property
    def representative(self) -> KernelDossier:
        """The smallest-MR member (the simplest exemplar of the cluster's decision/structure)."""
        return min(self.members,
                   key=lambda d: ((d.decisions.get("register_block") or {}).get("mr") or 0,
                                  d.path))

    def summary(self) -> dict[str, Any]:
        rep = self.representative
        return {
            "signature": list(self.signature),
            "n_members": len(self.members),
            "sources": sorted({d.source for d in self.members}),
            "representative": rep.path,
            "decisions": rep.decisions,
            "struct": rep.struct,
            "motifs": rep.motifs,
            "members": [d.path for d in self.members],
        }


def cluster_dossiers(dossiers: list[KernelDossier]) -> list[Cluster]:
    """Group by signature; return clusters sorted by size (largest first — most representative)."""
    by_sig: dict[tuple, Cluster] = {}
    for d in dossiers:
        c = by_sig.setdefault(d.signature(), Cluster(d.signature()))
        c.members.append(d)
    return sorted(by_sig.values(), key=lambda c: len(c.members), reverse=True)


def representatives(dossiers: list[KernelDossier]) -> list[KernelDossier]:
    """One representative dossier per cluster — the agent's review set in representative mode."""
    return [c.representative for c in cluster_dossiers(dossiers)]
