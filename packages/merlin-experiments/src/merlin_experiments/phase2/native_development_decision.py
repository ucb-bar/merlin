"""Opt-in queued native priorities, separate from measured promotion and plateau.

The launcher validates the original complete queued feedback before calling
this projection. Original native owners are reopened; no saved verdict creates
an owner. Decisions cover the exact declared selected roster, never a subset
chosen after observing failures or UNKNOWN predictions.
"""

from . import _public_native_feedback as FEEDBACK
from .contracts import StageGateError, document_sha256

POLICY = "merlin.qualified_native_development_priority.v1"


def decide(document, *, owner, required_members):
    from .public_native_exploration import PublicNativeExploration

    if type(owner) is not PublicNativeExploration:
        raise StageGateError("native decision requires its original live exploration owner")
    owner.verify()
    if (
        type(required_members) is not tuple
        or not 1 <= len(required_members) <= 4096
        or any(
            type(key) is not tuple or len(key) != 2 or any(type(part) is not str or not part for part in key)
            for key in required_members
        )
        or len(set(required_members)) != len(required_members)
    ):
        raise StageGateError("native decision requires its exact bounded declared roster")
    if type(document) is not dict:
        raise StageGateError("native decision requires complete queued feedback")
    selected = document.get("selected_members")
    if (
        type(selected) is not list
        or any(type(key) is not list or len(key) != 2 or any(type(part) is not str for part in key) for key in selected)
        or sorted(selected) != sorted([list(key) for key in required_members])
    ):
        raise StageGateError("native decision cannot drop or replace selected members")
    if type(document.get("cells")) is not list or any(
        type(cell) is not dict
        or any(type(cell.get(part)) is not str for part in ("family", "capsule"))
        or (
            (cell["family"], cell["capsule"]) in required_members
            and any(
                type(cell.get(arm)) is not dict or "native_exploration" not in cell[arm]
                for arm in ("baseline", "candidate")
            )
        )
        for cell in document["cells"]
    ):
        raise StageGateError("native decision requires original arm evidence")
    FEEDBACK.validate(document, owner=owner)
    cells = {(cell["family"], cell["capsule"]): cell for cell in document["cells"]}
    if len(cells) != len(document["cells"]) or not set(required_members) <= set(cells):
        raise StageGateError("native decision lost original complete cells")
    rows = []
    for key in required_members:
        cell = cells[key]
        row = dict(
            family=key[0],
            capsule=key[1],
            status="UNKNOWN",
            recommendation=None,
            conservative_improvement_fraction=None,
            reason="original arm evidence incomplete",
        )
        arms = [cell[arm] for arm in ("baseline", "candidate")]
        if all(type(arm) is dict and arm.get("functional_status") == "PASS" for arm in arms):
            projections = [arm["native_exploration"] for arm in arms]
            if all(value["status"] == "predicted" for value in projections):
                compiled = [arm.get("compiled", {}) for arm in arms]
                signatures = [(value.get("elf_sha256"), value.get("llvm_mlir_sha256")) for value in compiled]
                if any(value.get("status") != "observed" for value in compiled) or any(
                    type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                    for signature in signatures
                    for digest in signature
                ):
                    row["reason"] = "complete original compiled product identities unavailable"
                elif signatures[0][1] == signatures[1][1]:
                    row.update(status="unchanged_kernel_ir", reason="same observed kernel IR; no distinct experiment")
                else:
                    baseline, candidate = [value["prediction"]["interval"] for value in projections]
                    saved = 1 - candidate[1] / baseline[0]
                    row["reason"] = "qualified residual intervals overlap or improvement is below one percent"
                    if saved >= 0.01:
                        row.update(
                            status="qualified_interval_priority",
                            recommendation="candidate",
                            conservative_improvement_fraction=saved,
                            reason="separated qualified intervals with at least one percent conservative improvement",
                        )
                    elif baseline[1] < candidate[0]:
                        row.update(
                            status="qualified_interval_priority",
                            recommendation="baseline",
                            reason="separated qualified intervals favor baseline",
                        )
            else:
                row["reason"] = "one or both original native domains remain UNKNOWN"
        rows.append(row)
    recommendation = "candidate" if all(row["recommendation"] == "candidate" for row in rows) else None
    if any(row["recommendation"] == "baseline" for row in rows):
        recommendation = "baseline"
    owner.verify()
    return dict(
        schema="merlin.native_development_decision.v1",
        policy=POLICY,
        feedback_sha256=document_sha256(document),
        selection_sha256=document_sha256(owner.record()),
        required_members=[list(key) for key in required_members],
        members=rows,
        next_exploration_parent=recommendation,
        measured_cycles=None,
        promotion_authority="NONE",
        plateau_evidence=False,
        full_search_ready=False,
        scope="qualified interval priorities only; no measured incumbent, aggregate objective or convergence",
    )
