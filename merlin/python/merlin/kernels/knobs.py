"""Motif -> RVV-knob mapping + the gap-router: turn a structural divergence (S4 compare) into
concrete fork proposals, each tagged with the lever class and the mined evidence that justifies it.

Honest forkable/deferred split (proven by the manual forks):
  * FORKABLE NOW (lever "knob") — edits expressible in the transform schedule today: tile/vector
    width (toward higher LMUL), contraction lowering strategy, lowering-pattern set.
  * DEFERRED (lever "lowering_pattern"/"llvm_requirement") — needs a schedule/pipeline FEATURE that
    does not exist yet, recorded as a work-item rather than auto-applied. The headline example: the
    fused-vfmacc gap needs fast-math `contract` injection at MLIR emission (outerproduct is a no-op
    because `transform.structured.vectorize` lowers the matmul straight to mul+add — no
    vector.contract is ever formed). The router surfaces it; it is not yet a one-knob fork.

`propose_forks(divergences, knobs)` returns a list of `ForkProposal` the beam expands.

SINGLE-ROUTER NOTE: the CCA<->lever router `kernels.action_catalog` (the one the bijection contract
enforces) is the source of truth for what the compiler exposes. `mining.fork_from_action.
propose_forks_from_cca` derives the beam proposer from it (consuming typed CCA `Divergence`s) and is the
successor to this motif-string router. This module is retained for the existing motif-string beam path
until the beam is cut over to CCA divergences (WS-D). NB: the `fma_form` "work-item" note below is
historical — fused vfmacc is now a certified `impr_features:fused_vfmacc_contraction` PASS
(see action_catalog), so that gap is CLOSED; the note is kept only to document the original routing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ForkProposal:
    overrides: dict[str, Any]          # knob overrides applied to the parent (empty for deferred)
    lever: str                          # knob | lowering_pattern | llvm_requirement | feature | work_item
    targets: str                        # which divergence/decision this addresses
    evidence: list[str]                 # mined policy / kernel ids justifying it
    forkable: bool                      # True => beam can mint+certify; False => recorded work-item
    note: str = ""
    # the typed CompilerAction this proposal came from (CCA-native proposer only; None for the legacy
    # motif router). Carries intended_facet so the beam can AUDIT the minted fork (did the emitted asm
    # achieve the promise?) via search_step.audit_fork. Left as Any to avoid importing action_catalog.
    action: Any = None


def _wider_n_overrides(knobs: dict, factor: int) -> dict:
    """Scale the N tile/vector dim of every contraction op (toward higher LMUL grouping)."""
    new = []
    for m in knobs.get("op_match", []):
        tile, vec = list(m["tile"]), list(m["vector"])
        # N is the second-to-last dim (…, M, N, K) for both matmul and batch_matmul here.
        if len(tile) >= 3:
            tile[-2] *= factor
            vec[-2] *= factor
        new.append({"op": m["op"], "tile": tile, "vector": vec})
    return {"op_match": new}


# decision key -> list of (lever, forkable, override-builder | None, evidence-policy, note)
_ROUTES: dict[str, list[dict]] = {
    "lmul_class": [
        {"lever": "knob", "forkable": True, "policy": "lmul_grouping_policy",
         "build": lambda k: _wider_n_overrides(k, 2),
         "note": "widen N tile/vector x2 to push vector grouping toward higher LMUL"},
        {"lever": "knob", "forkable": True, "policy": "lmul_grouping_policy",
         "build": lambda k: _wider_n_overrides(k, 4),
         "note": "widen N tile/vector x4"},
    ],
    "fma_form": [
        {"lever": "knob", "forkable": True, "policy": "fma_broadcast_policy",
         "build": lambda k: {"contraction_strategy": "outerproduct"},
         "note": "try outerproduct contraction lowering (NOTE: proven no-op; kept so the beam "
                 "records it as explored/pruned)"},
        {"lever": "llvm_requirement", "forkable": False, "policy": "fma_broadcast_policy",
         "build": None,
         "note": "RECOVER FUSED vfmacc: inject fast-math `contract` at MLIR emission so clang fuses "
                 "fmul+fadd -> fmuladd -> vfmacc. Not a schedule knob today (needs a lowering "
                 "feature: set fastmath on arith ops / a contract pass). Work-item."},
    ],
    "vl_strategy": [
        {"lever": "llvm_requirement", "forkable": False, "policy": "vl_tail_policy",
         "build": None,
         "note": "expert uses vsetvl-loop (VL-polymorphic); we emit vsetivli (fixed immediate). "
                 "Needs a scalable-vector / VL-loop lowering path. Work-item."},
    ],
    "int_widening": [
        {"lever": "knob", "forkable": True, "policy": "int8_widening_policy",
         "build": lambda k: {"dtype_strategy": "int8_w8a8"},
         "note": "route i8 matmul through the vwmacc integer datapath (passes_quant_int)"},
    ],
}


def propose_forks(divergences: list[str], knobs: dict[str, Any]) -> list[ForkProposal]:
    """From S4 divergence strings (e.g. "lmul_class: expert='m4' vs ours='m2'") + the parent knobs,
    enumerate candidate forks. Forkable proposals carry knob overrides; deferred ones are recorded
    work-items (lever-2/3) the beam reports but cannot auto-apply yet."""
    keys = [d.split(":")[0].strip() for d in divergences]
    out: list[ForkProposal] = []
    for key in keys:
        for route in _ROUTES.get(key, []):
            overrides = route["build"](knobs) if route["build"] else {}
            out.append(ForkProposal(
                overrides=overrides, lever=route["lever"], targets=key,
                evidence=[route["policy"]], forkable=route["forkable"], note=route["note"]))
    return out
