"""Capsule evaluation: eligibility."""

from __future__ import annotations

from merlin.targetgen.runner_config import _config_for_target


def _dtype_has_no_datapath(region, cmap, _el) -> tuple[bool, str]:
    """Is the region's operand dtype absent from EVERY capability this target declares?

    This is the ONLY hard structural fact in an eligibility verdict, and so the only ground on which a
    capsule may be withheld from a suite. If no datapath holds the operand format, no arrangement of the
    program puts it on the device.

    The other two axes of the verdict are NOT hard, and both have cost us a real capability:

    * **family** — families compose. Attention on a systolic array is a contraction plus a transposing
      movement, so "no capability for family X" is not proof the device cannot do X. Measured: a float
      target PASSED two capsules its contract calls ineligible on family grounds.
    * **rank** — rank is the one thing a compiler exists to change. A rank-4 convolution reaches a
      rank-2 mesh through im2col, and this compiler ships that lowering
      (``convolution_im2col_matmul``). Measured: ``RP14_patch_embed_bf16_pt`` was withheld as
      "rank 4 not in contraction legal ranks [2, 3]" while the lowering that turns it into a rank-2
      contraction sat unused in the tree.

    Compares through eligibility's OWN alias-aware check, never string equality: a contract spells the
    format ``int8`` while a capsule region reports ``i8``, and a raw ``!=`` reads a native-dtype capsule
    as having no datapath at all.
    """
    dt = getattr(region, "in_dtype", None)
    all_dtypes = tuple({x for cap in cmap.values() for x in (getattr(cap, "dtypes", ()) or ())})
    if dt is None or not all_dtypes or _el._dtype_ok(dt, all_dtypes):
        return False, ""
    return True, f"operand dtype {dt!r} is in no capability this target declares"


def _forbids_the_mesh(capsule: dict) -> bool:
    """Whether this capsule's whole purpose is NOT to be accelerated.

    Shared by the suite's withholding rule and the gate's denominator, because those two drifting apart
    is a defect this module has already paid for once: they ran different tests, and a capsule the
    suite graded and scored was silently absent from the gate that decided whether a whole-model
    capstone could launch. One predicate, used by both.

    Such a capsule declares ``forbid: [on_mesh]`` -- it exists to prove the compiler leaves work on the
    host -- so the absent datapath that would withhold any other capsule is its PREMISE. Withholding it
    deletes the only test of the negative lane.
    """
    return "on_mesh" in {str(x) for x in ((capsule.get("lanes") or {}).get("forbid") or ())}


def _split_ineligible(op_caps: list[dict], target: str) -> tuple[list[dict], list[dict]]:
    """Partition op capsules into (to grade, not-gradeable-on-this-target).

    Ineligibility is decided by the target's OWN declared capability, so this adds no new judgement --
    it acts on the verdict :mod:`eligibility` already computes. Fails OPEN in every uncertain case
    (no capability map, an undetermined verdict, any exception): a capsule is only withheld when the
    contract positively says the family/dtype/rank is not supported. Withholding on doubt would let a
    suite shrink itself into a better score, which is the opposite of the point.
    """
    try:
        from merlin.targetgen import coverage_report as _cr
        from merlin.targetgen import eligibility as _el

        cmap = _el.capability_map_for_target(target)
    except Exception:  # noqa: BLE001 - never block a grade
        return op_caps, []
    if not cmap:
        return op_caps, []
    keep, withheld = [], []
    for c in op_caps:
        try:
            region = _cr._capsule_region(c)
            v = _el.is_eligible(region, cmap)
        except Exception:  # noqa: BLE001
            keep.append(c)
            continue
        if getattr(v, "undetermined", False) or getattr(v, "eligible", True):
            keep.append(c)
            continue
        # WITHHOLD ONLY ON A HARD STRUCTURAL FACT: an operand dtype or rank that appears in NO declared
        # capability at all. Then no datapath on the device can hold the operands and no composition can
        # rescue it, whatever family the op belongs to -- which also covers a capsule whose family our
        # taxonomy cannot name (pooling, whole-model), where withholding on the family verdict alone
        # would drop it for OUR ignorance rather than the hardware's limits.
        #
        # A merely UNDECLARED FAMILY is graded. Families compose -- attention on a systolic array is a
        # contraction plus a transposing movement -- so "no capability for family X" is not proof the
        # device cannot do X. Measured: a float target PASSED two capsules its contract calls ineligible
        # on family grounds; withholding those would have hidden a contract under-declaration, which is a
        # finding we want, not noise to suppress.
        # Compare dtypes through eligibility's OWN alias-aware check, never by string equality: the
        # contract spells the format `int8` while a capsule region reports `i8`, so a raw `!=` reads a
        # native-dtype capsule as having no datapath at all and withholds it for a spelling difference.
        # See :func:`_dtype_has_no_datapath` for why the dtype is the only hard fact here, and why
        # family and rank must not withhold. State the fact that ACTUALLY triggered withholding:
        # eligibility's own reason describes its FAMILY verdict, which would send someone to fix the
        # taxonomy when the hardware is the constraint.
        dtype_absent, why = _dtype_has_no_datapath(region, cmap, _el)
        if not dtype_absent:
            keep.append(c)
            continue
        # A CAPSULE THAT FORBIDS THE MESH IS ASKING FOR EXACTLY THIS, so withholding it deletes the
        # test. The rule above is "no datapath can hold these operands, so do not ask the device to
        # certify this arithmetic" -- sound for a capsule that WANTS to be accelerated. A host-lane
        # capsule wants the opposite: it declares `forbid: [on_mesh]` and exists to prove the compiler
        # leaves the work on the host, and the absent datapath is its premise rather than its problem.
        # Withheld, the negative lane is never graded, and an axis that derives such capsules produces
        # nothing while looking like it ran -- the exact silence this corpus is built to prevent. The
        # forbid IS gradeable: a decoded accelerator instruction violates it.
        # Explicit frozen host-only work is also gradeable through the existing
        # Merlin host provider. That route rechecks source eligibility and credits
        # infrastructure only, never candidate accelerator capability.
        from merlin.targetgen.owned_host_leaf_execution import declared_host_only

        if _forbids_the_mesh(c) or declared_host_only(c):
            keep.append(c)
            continue
        withheld.append(
            {
                "capsule": c.get("name"),
                "kind": c.get("kind"),
                "label": c.get("label"),
                "status": "not_graded",
                "ineligible": True,
                "failure": {
                    "plane": "capability",
                    "category": "NOT_GRADED",
                    "detail": f"not graded on this target: {why} "
                    f"(eligibility also reports: {getattr(v, 'reason', '') or 'ineligible'})",
                },
            }
        )
    return keep, withheld


def _gate_counts(result: dict, capsules: list[dict], target: str) -> bool:
    """Whether a graded op-capsule result belongs in the whole-model gate's denominator.

    True unless the capsule is provably ineligible -- and "provably" means the SAME hard structural fact
    that :func:`_split_ineligible` uses to withhold a capsule from the suite: no declared capability
    holds its operand dtype. Fails OPEN whenever that cannot be decided.

    These two ran different tests and disagreed. This read the raw ``is_eligible`` verdict, so a capsule
    whose FAMILY the taxonomy cannot name was dropped from the gate denominator while the suite graded
    it and counted it in the score. Measured on gemmini: ``C7_attention_qk_i8`` (declared
    ``op: attention_qk``, which the taxonomy maps to the ``attention`` composite) was excluded here,
    lifting the gate fraction from 18/23 = 0.78 to 18/22 = 0.82 and launching a whole-model capstone
    that should have deferred -- a five-and-a-half-hour simulation inside a four-hour round. Meanwhile
    the identical computation spelled ``op: matmul`` (``C5_attention_qk_matmul``) counted normally.
    One test, used by both, or the suite and its gate are scoring different corpora.
    """
    name = result.get("capsule")
    cap = next((c for c in capsules if c.get("name") == name), None)
    if cap is None:
        return True
    try:
        from merlin.targetgen import coverage_report as _cr
        from merlin.targetgen import eligibility as _el

        cmap = _el.capability_map_for_target(target)
        if not cmap:
            return True
        region = _cr._capsule_region(cap)
        verdict = _el.is_eligible(region, cmap)
    except Exception:  # noqa: BLE001 - a gate must not crash a grade
        return True
    if getattr(verdict, "undetermined", False) or getattr(verdict, "eligible", True):
        return True
    from merlin.targetgen.owned_host_leaf_execution import declared_host_only

    if _forbids_the_mesh(cap) or declared_host_only(cap):
        return True
    dtype_absent, _why = _dtype_has_no_datapath(region, cmap, _el)
    return not dtype_absent


def suite_for(target: str, *, dtype: str = "i8xi8_i32") -> str:
    """The suite path segment ``run_capsule`` writes results under for this target (``cfg.suite``).

    Any reader that re-globs the on-disk ``capsule_result.json`` MUST resolve the suite through this,
    not a reference-target constant: ``run_capsule`` lays results at
    ``<runs_root>/runs/<cfg.suite>/<capsule>/`` where ``cfg.suite`` is the TARGET's own suite
    (e.g. '<target>-capsule-bench'). Using a fixed reference suite made another target's self-check
    glob an empty directory and report ``n_capsules: 0`` on every call — the feedback loop went blind while
    the in-memory grade was correct. Derived from the target's RunnerConfig, so it stays target-agnostic."""
    return _config_for_target(target, None, dtype).suite
