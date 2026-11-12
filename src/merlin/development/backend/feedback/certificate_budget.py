"""Apply an explicitly declared per-capsule certificate cost budget.

Unpriced infrastructure failures retain the existing admission behavior;
a declared, evaluated ceiling may decline work without changing verdicts.
"""

from __future__ import annotations

from pathlib import Path

from merlin.development.backend.context import InvocationContext


def affordable_cert_items(want, *, cert_tier, log, context: InvocationContext, capsules_root: Path | None = None):
    """``want`` minus the capsules a DECLARED per-capsule budget says cost too much at ``cert_tier``.

    OFF UNLESS DECLARED. `tier_policy.ceiling_budget_seconds` is opt-in and has no default, so with no
    budget in force this returns ``want`` unchanged before touching a manifest, a cost fit or the disk.
    A budget nobody declared, silently deciding which capsules get certified, is exactly the quiet
    coverage loss the ceiling exists to make visible.

    A COST VERDICT MAY DECLINE; AN INFRASTRUCTURE FAILURE MAY NOT. `oracle_ceiling` fails closed on an
    UNKNOWN cost, which is right -- "not shown to fit" is not "fits". But a manifest that will not load
    is not a statement about cost, and dropping the capsule for it would be a silent coverage loss
    wearing a budget's clothes. Those allow, loudly. This mirrors how promotion is wrapped at its call
    site: an optimisation, never a gate.

    NOT `schedule(cost_s=..., budget_s=...)`: that budget is a TOTAL across the queue and its cost is
    per-TIER, so it answers "how many may we buy" and cannot answer "may THIS capsule be bought".
    """
    from merlin.development.targetgen.tier_policy import ceiling_budget_seconds, oracle_ceiling

    if ceiling_budget_seconds() is None:
        return want, {}
    try:
        from merlin.development.targetgen.capsule_common import load_capsule
        from merlin.development.targetgen.target_experiment import load_target_experiment

        te = load_target_experiment(context.descriptor)
        by_name = {
            d.name: d
            for root in ((capsules_root,) if capsules_root is not None else te.graded_roots())
            for d in Path(root).iterdir()
            if (d / "capsule.yaml").is_file()
        }
    except Exception as exc:  # noqa: BLE001 -- cannot read the corpus: certify everything, and say so
        print(
            f"[promote] budget declared but the corpus could not be read ({type(exc).__name__}); "
            f"every capsule admitted to {cert_tier}",
            file=log,
            flush=True,
        )
        return want, {}
    kept, declined = [], {}
    for w in want:
        d = by_name.get(w.capsule)
        if d is None:
            print(
                f"[promote] {w.capsule}: no manifest under the graded roots; admitted to "
                f"{cert_tier} rather than dropped on a budget it was never priced against",
                file=log,
                flush=True,
            )
            kept.append(w)
            continue
        try:
            ceiling = oracle_ceiling(te.target, load_capsule(d), cert_tier)
        except Exception as exc:  # noqa: BLE001 -- an unaskable question is not a NO
            print(
                f"[promote] {w.capsule}: cost ceiling could not be evaluated "
                f"({type(exc).__name__}); admitted to {cert_tier}",
                file=log,
                flush=True,
            )
            kept.append(w)
            continue
        if ceiling.allowed:
            kept.append(w)
        else:
            declined[w.capsule] = {
                "source": ceiling.source,
                "axis": ceiling.axis,
                "reason": ceiling.reason,
                "record": ceiling.record,
            }
            print(f"[promote] {w.capsule}: NOT enqueued for {cert_tier} -- {ceiling.reason}", file=log, flush=True)
    return kept, declined
