"""Kernel dossier — compose all mining layers into the single object the agent reads (never raw C).

A dossier folds together, per kernel:
  * identity        — source / framework / path / op / dtype / shape
  * decisions       — the RVV decision vector (features/rvv_intrinsics)
  * struct          — loop nest/order, prepack idiom, op counts (features/ast_struct, tree-sitter)
  * motifs          — the classified motif set (what promotes to policy)
  * framework_contract — caller-side prepack/transpose/layout assumptions (NOT in the code/asm)
  * asm             — objdump instruction text, when a build-to-asm is available (S8.3); else None
  * agent_notes     — sparse, agent-filled judgment (algorithm, exemplary?, caveats); None until run

This is the unit the cluster step groups and the agent step (dual-mode) annotates. Deterministic
layers are always populated; asm/agent_notes are optional so the dossier degrades gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .classify import classify_motifs
from .features import extract_all
from .framework_contracts import load_contract
from .types import NormalizedKernel


@dataclass
class KernelDossier:
    source: str
    framework: str
    path: str
    op: str
    dtype: str
    decisions: dict[str, Any]                 # f["rvv"]
    struct: dict[str, Any]                     # f["struct"]
    motifs: list[str]
    framework_contract: dict[str, Any]
    shape: dict[str, Any] = field(default_factory=dict)
    asm: str | None = None                     # objdump text (S8.3 build_asm), when available
    agent_notes: dict[str, Any] | None = None  # sparse agent judgment (S8.6), when run

    def signature(self) -> tuple:
        """A deterministic clustering key from the static facts (used by cluster.py)."""
        d, s = self.decisions, self.struct
        mr = (d.get("register_block") or {}).get("mr")
        return (self.op, self.dtype,
                d.get("lmul_class"), d.get("fma_form"), bool(d.get("int_widening")),
                mr, s.get("loop_nest_depth"), tuple(s.get("loop_order", [])[:3]),
                bool(s.get("pointer_advance_prepack")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "framework": self.framework, "path": self.path,
            "op": self.op, "dtype": self.dtype, "shape": self.shape,
            "decisions": self.decisions, "struct": self.struct, "motifs": self.motifs,
            "framework_contract": self.framework_contract,
            "has_asm": self.asm is not None, "agent_notes": self.agent_notes,
        }


def build_dossier(nk: NormalizedKernel, *, asm: str | None = None) -> KernelDossier:
    """Assemble a dossier for one normalized kernel. ``asm`` (objdump text) is passed in when a
    build-to-asm exists; otherwise the dossier carries the code-level layers only."""
    features, _fired = extract_all(nk)
    return KernelDossier(
        source=nk.source, framework=nk.source, path=nk.path, op=nk.op, dtype=nk.dtype,
        decisions=features.get("rvv", {}), struct=features.get("struct", {}),
        motifs=sorted(classify_motifs(features, nk.op)),
        framework_contract=load_contract(nk.source),
        shape=getattr(nk, "shape", {}) or {}, asm=asm)
