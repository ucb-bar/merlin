"""Preserve declared model accelerator, lane and semantic obligations."""

from __future__ import annotations

from merlin.capture.demands import model_op_demands_checked


def model_accelerator_demand(linalg_mlir: str, binding) -> tuple[str | None, list[str]]:
    """Derive accelerator family/classes from original source and target roles.
    Unsupported source families, absent capability or missing role census yield
    ``(None, [])``. The caller must preserve that zero *mandatory* device demand
    instead of inventing a hardware obligation. Public host programs retain
    complete source/numerical checks and optional legal accelerator lowering.
    """
    from merlin.targetgen import eligibility as _el
    from merlin.targetgen import semantic_families as _sf
    from merlin.targetgen.corpus_instruction_roles import classes_for_checked_source

    target = getattr(binding, "target", None)
    if not target or not linalg_mlir:
        return None, []
    try:
        cap_map = _el.capability_map_for_target(target)
    except Exception:  # noqa: BLE001 — no contract yet: demand nothing
        return None, []
    if not cap_map:
        return None, []

    in_fmt = binding.cap_dtype(binding.operand_dtype)
    try:
        # A provenance family is not authority by itself. The checked reader joins every claimed
        # contraction to a parsed, typed contraction of the same shape and refuses omissions/extras.
        demands = model_op_demands_checked(linalg_mlir, in_fmt)
    except Exception:  # noqa: BLE001 — unreadable capture: demand nothing
        return None, []

    # Ops of the model this target's hardware is declared able to run. Asked of the eligibility oracle,
    # the same independent denominator ARR uses -- not of routing, which is the thing under test.
    eligible_ops: list[str] = []
    eligible_families: list[str] = []
    for d in demands:
        if d.op in eligible_ops:
            continue
        if not d.source_formats_complete:
            continue
        desc = _el.RegionDescriptor(
            source=d.site or d.op,
            op=d.op,
            family=d.family,
            in_dtype=d.admission_input_fmt,
            weight_dtype=d.admission_weight_fmt,
            m=d.m,
            k=d.k,
            n=d.n,
            form=d.form,
        )
        if _el.is_eligible(desc, cap_map).eligible:
            eligible_ops.append(d.op)
            # ``int_matmul`` is a capture/routing spelling, not a semantic-family key. Its family
            # was independently checked against typed IR above; retain that canonical result.
            # Other operations keep the pre-existing op-vocabulary path rather than allowing an
            # unverified provenance label to create a new whole-model demand.
            family = d.family if d.family == "contraction" else _sf.from_op(d.op)
            if family:
                eligible_families.append(family)
    if not eligible_ops:
        return None, []

    # The capsule's family is the one the accelerator is actually FOR. A model is a composition and the
    # closed vocabulary has no name for it, so naming the family it owes -- rather than adding a bogus
    # ``model`` entry to the vocabulary -- is what lets the eligibility oracle resolve the capsule at all,
    # and therefore what lets must_accelerate mean anything. Contraction when present (every declared
    # matrix unit exists for it); otherwise the first eligible family, in program order.
    families = eligible_families
    if not families:
        return None, []
    family = "contraction" if "contraction" in families else families[0]

    # Classes come from the binding's OWN deriver -- the same callable every op capsule is built with, so
    # the capstone and the op capsules can never disagree about what this target's contraction sequence
    # is. (It resolves three regimes: a declared matrix unit, a self-hosted ISA taxonomy, or a RoCC
    # encoding map. Reaching past it to one of those directly is how the capstone came out empty on a
    # target whose op capsules carry the full eight-class sequence.)
    classes: list[str] = []
    for op in eligible_ops:
        try:
            got = classes_for_checked_source(binding, op=op, output_dtype=in_fmt)
        except Exception:  # noqa: BLE001 — an underivable class list demands nothing
            continue
        for c in got or ():
            if c not in classes:
                classes.append(c)
    return family, _in_issue_order(classes, binding, in_fmt)


def _in_issue_order(classes: list[str], binding, in_fmt: str) -> list[str]:
    """Put a UNION of per-op class lists back into the target's own issue order.

    The union above is built in first-encounter order, which was harmless only while every op returned
    the identical full sequence. Now that a movement op correctly owes no contraction classes, the op
    the model happens to mention first decides the order -- and a model whose first eligible op is a
    copy reported its store class BEFORE its multiply, i.e. a sequence the target cannot issue.

    The reference order is asked of the binding, not written here: a contraction is the op whose class
    list IS this target's full sequence, so its order is the canonical one, and using it keeps this
    agnostic to how the target spells or orders its classes. A class the reference does not contain is
    kept, at the end -- an unplaceable class is not evidence it is unnecessary, and dropping it would be
    the silent drop the parsing rule forbids.
    """
    try:
        reference = list(binding.classes_for(op="matmul", output_dtype=in_fmt) or ())
    except Exception:  # noqa: BLE001 -- no reference: keep the order we have
        return classes
    rank = {c: i for i, c in enumerate(reference)}
    return sorted(classes, key=lambda c: (rank.get(c, len(rank)), classes.index(c)))


def _model_semantic_block(entry: dict, family: str | None, classes) -> dict:
    """The generalization-intent block for a whole-model capsule.

    Three things are decided here, and the profile entry's ``generalization`` block overrides each --
    the same authored-override channel ``corpus_spec._semantic_block`` already honours for every other
    capsule kind. The model path used to ignore it entirely, which caused two failures at once:

    * ``must_accelerate`` was FORCED false by the mere presence of ``lanes.require``. The reasoning is
      sound for a whole real model, whose norms have nowhere but the host to go -- but it is wrong for a
      capsule whose subject is the seam itself. The host-island capsule needs BOTH: its two int8 GEMMs
      must reach the mesh (``must_accelerate``) *and* its LayerNorm island must land on the host lane
      (``lanes.require``). Forcing them apart made declaring the lane contract silently WEAKEN the mesh
      assertion, so the automatic withholding is now a default an author can override, not a mandate.
    * ``not_asserted_reason`` could only ever say "the demand could not be derived". A capsule that
      withholds the assertion ON PURPOSE had no way to say so from the profile, so the reason was
      hand-written into the generated capsule and deleted by the next regeneration.
    """
    grounded = bool(family and classes)
    authored = entry.get("generalization")
    authored = dict(authored) if isinstance(authored, dict) else {}
    interop = bool((entry.get("lanes") or {}).get("require"))
    block: dict = {}
    if family:
        block["semantic_family"] = family
    block["generalization_axis"] = authored.get("generalization_axis", "model")
    block["must_accelerate"] = bool(authored.get("must_accelerate", grounded and not interop))
    # A WITHHELD ASSERTION ALWAYS SAYS WHY. The two reasons are different facts and must not be
    # conflated: "we could not derive the demand" is about our own reach, while "host-lane work is the
    # subject" is about the capsule. Deriving both means a synthesized interop capsule carries the
    # explanation without anyone typing it, and the profile text is an override rather than the only
    # source -- silence here reads as an author who simply never made a claim.
    reason = authored.get("not_asserted_reason")
    if not reason and not grounded:
        reason = (
            "the accelerator demand could not be derived from this model's linalg against this "
            "target's declared capabilities and role census, so must_accelerate is withheld "
            "rather than asserted ungrounded"
        )
    if not reason and interop:
        reason = (
            "interop capsule: composition across lanes is the behaviour under test, so host-lane "
            "work is expected and required. The accelerator demand is expressed as lanes.require, "
            "which fails unless every named lane actually carried work."
        )
    if reason and not block["must_accelerate"]:
        block["not_asserted_reason"] = reason
    block["eligible"] = authored.get("eligible", "auto")
    return block


def _checked_lanes(entry: dict, binding) -> list[str]:
    """The lanes an interop capsule requires, REFUSED at generation time if this target cannot populate one.

    A required lane is a bar the submission must clear. A bar the target's declared compute units make
    unreachable is not a hard capability test, it is a wall: no compiler can put work on a lane the router
    has nothing to route there. Measured on the corpus this check was written for -- a capsule required
    ``in_contract_vector_scalar`` on a target whose every declared unit is a mesh kind, so the lane was
    empty by construction and the capsule was unpassable however good the backend was.

    Raising here (rather than shipping the capsule and failing it forever) keeps the failure where someone
    can fix it: at authoring time, naming the lane and what the target actually offers."""
    want = [str(x) for x in (entry.get("lanes") or {}).get("require") or []]
    target = getattr(binding, "target", None)
    if not target:
        return want  # no target to judge against: leave the declaration
    try:
        from merlin.targetgen.routing import reachable_lanes

        have = reachable_lanes(target)
    except Exception:  # noqa: BLE001 - never block generation on this
        return want
    missing = [ln for ln in want if ln not in have]
    if missing:
        raise ValueError(
            f"capsule {entry.get('name')!r} requires lane(s) {missing} that target {target!r} cannot "
            f"populate (its declared compute units make {sorted(have)} reachable). A required lane the "
            f"router can put nothing on is unpassable by construction -- declare a lane this target owns, "
            f"or give the target a compute unit that serves the one you want"
        )
    return want
