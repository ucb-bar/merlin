"""Consistency invariants: automatic "do the results make sense?" checks.

Runs inside ``kernel-extract`` over the full corpus + promoted artifacts. Three kinds:

* **Structural invariants** that must hold by construction (composite motifs are subsets of
  their parts; target-restricted motifs only fire on their targets; promoted evidence ids
  exist in the corpus; the motif table equals a recount). A violation is a bug in the
  marker/classify layer, found *before* anyone trusts a policy.
* **Surprise list**: motifs firing on op families where the decision makes no sense (e.g.
  ``packed_rhs`` on an elementwise ``vadd``). Each entry is either a marker over-firing or a
  genuine insight — both actionable, so they are listed, never hidden.
* **Dedup accounting** from content hashes (reported by the caller's diagnostics).

Advisory by design: results land in the report (and ``--json``); only ``--strict`` callers
turn hard violations into a nonzero exit.
"""
from __future__ import annotations

import collections

from merlin.kernels.markers import target_family
from merlin.kernels.policy import MotifStat, PromotionResult

# Composite motif -> parts it must be a subset of (mirrors classify.py by construction).
_SUBSET_RULES = (
    ("reused_packed_rhs", ("packed_rhs",)),
    ("accumulator_commit", ("accumulator_lifetime", "epilogue_before_commit")),
)

# Motifs that only make sense for specific ISA families.
_TARGET_RESTRICTED = {"vector_length_polymorphic": {"rvv"}}

# Op families where a weight-reuse / dataflow motif firing is a *surprise* worth auditing.
_ELEMENTWISE_OPS = {
    "vadd", "vmul", "vsub", "vdiv", "vclamp", "velu", "vsigmoid", "vtanh", "vgelu",
    "vsqrt", "vrsqrt", "vexp", "vcvt", "copy", "scal", "swap", "axpy", "axpby",
    "amax", "amin", "asum", "sum", "max", "min", "rot", "nrm2", "dropout", "transpose",
}
_SURPRISE_MOTIFS = ("packed_rhs", "reused_packed_rhs", "weight_stationary_dataflow")


def check_invariants(records: list[dict], stats: dict[str, MotifStat],
                     promo: PromotionResult) -> dict:
    """Return ``{checks: [...], surprises: [...], total_violations: int}``."""
    checks: list[dict] = []

    def _check(name: str, bad: list[str]) -> None:
        checks.append({"name": name, "status": "ok" if not bad else "VIOLATED",
                       "violations": len(bad), "examples": bad[:5]})

    subset_bad: dict[str, list[str]] = {f"{c} ⊆ {' ∩ '.join(p)}": [] for c, p in _SUBSET_RULES}
    target_bad: list[str] = []
    surprise_counter: collections.Counter = collections.Counter()
    surprise_example: dict = {}
    recount: collections.Counter = collections.Counter()
    corpus_eids: set[str] = set()

    for rec in records:
        ev = rec.get("evidence", {}) or {}
        motifs = set(ev.get("motifs", []))
        corpus_eids.add(ev.get("id", ""))
        recount.update(motifs)
        for child, parents in _SUBSET_RULES:
            if child in motifs and not all(p in motifs for p in parents):
                subset_bad[f"{child} ⊆ {' ∩ '.join(parents)}"].append(rec.get("path", "?"))
        for motif, fams in _TARGET_RESTRICTED.items():
            if motif in motifs and target_family(rec.get("target", "")) not in fams:
                target_bad.append(f"{motif} on {rec.get('target')}: {rec.get('path', '?')}")
        op = rec.get("op", "unknown")
        if op in _ELEMENTWISE_OPS:
            for motif in _SURPRISE_MOTIFS:
                if motif in motifs:
                    key = (motif, op, rec.get("source", "?"))
                    surprise_counter[key] += 1
                    surprise_example.setdefault(key, rec.get("path", "?"))

    for name, bad in subset_bad.items():
        _check(f"subset: {name}", bad)
    _check("target-restricted motifs fire only on their targets", target_bad)

    # Motif table counts must equal an independent recount.
    count_bad = [f"{m}: table={st.kernel_count} recount={recount.get(m, 0)}"
                 for m, st in stats.items() if st.kernel_count != recount.get(m, 0)]
    _check("motif table equals recount", count_bad)

    # Every promoted artifact's evidence ids must exist in the corpus.
    ev_bad: list[str] = []
    for artifact in (*promo.candidates, *promo.rules, *promo.runtime_candidates):
        name = artifact.get("name") or artifact.get("policy", "?")
        for eid in artifact.get("evidence", []):
            if eid not in corpus_eids:
                ev_bad.append(f"{name}: {eid}")
    _check("promoted evidence ids exist in corpus", ev_bad)

    # Dispatch metrics must exist wherever many_small_dispatches fired.
    dm_bad = [rec.get("path", "?") for rec in records
              if "many_small_dispatches" in (rec.get("evidence", {}) or {}).get("motifs", [])
              and not (rec.get("features", {}) or {}).get("dispatch_metrics")]
    _check("many_small_dispatches implies dispatch metrics", dm_bad)

    surprises = [{"motif": m, "op": op, "source": src, "count": n,
                  "example": surprise_example[(m, op, src)]}
                 for (m, op, src), n in surprise_counter.most_common(12)]
    return {
        "checks": checks,
        "surprises": surprises,
        "total_violations": sum(c["violations"] for c in checks),
    }
