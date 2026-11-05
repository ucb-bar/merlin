"""Preserve authored generalization intent and derive its semantic family."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding


def _semantic_block(entry: dict, binding: CorpusBinding) -> dict:
    """The capsule's generalization-intent block: half derived, half authorial.

    ``semantic_family`` is DERIVED from the op — the same mapping the eligibility oracle and the
    leave-one-family-out splitter already fall back to, so a generated capsule and a graded capsule can
    never disagree about which family they belong to.

    ``must_accelerate`` / ``generalization_axis`` are NOT derivable: they are the author's claim about
    what this capsule is for, so they come from the profile entry (or the profile's
    ``datapath.semantic_defaults``). ⚠️ ``must_accelerate`` defaults to FALSE on purpose. It is the
    strongest assertion in the block — an eligible region that falls back is a hard failure — and
    defaulting it true would turn every generated capsule into a gate the moment this lands.

    Emitting this here is what stops a corpus regeneration from silently deleting the annotation:
    before, the 40 radiance blocks were hand-written and ``_write_capsule`` dumps the dict wholesale,
    so a regen would have zeroed the only working generalization annotation in the repo.
    """

    from merlin.targetgen import semantic_families as _sf

    from .corpus_spec import _AXIS_BY_KIND, _fused_families

    defaults = dict(getattr(binding, "semantic_defaults", None) or {})
    # ⚠️ NOT ``entry["semantic"]`` -- a profile entry already uses that key for a free-form op-semantics
    # label ("quantized_linear", "matmul_mxu0") that flows into ``operation.attributes.semantic``.
    # Reusing it here would collide with existing entries across target profiles and crash the
    # generator, so the generalization-intent override gets a key of its own.
    raw = entry.get("generalization")
    authored = dict(raw) if isinstance(raw, dict) else {}
    fam = authored.get("semantic_family") or _sf.from_op(entry.get("op", "matmul"))
    block: dict = {}
    if fam:
        block["semantic_family"] = fam
    # The axis follows the capsule KIND, which is a fact about what the capsule tests rather than an
    # opinion: an ``isa`` capsule exercises a primitive the corpus has SEEN, a ``model_slice`` exercises
    # a COMPOSITION of primitives, and a whole ``model`` is the end-to-end integration case. Deriving it
    # keeps 40 annotations from having to be re-typed and keeps a new capsule from being mislabelled.
    kind = entry.get("kind", "isa")
    axis = authored.get("generalization_axis") or defaults.get("generalization_axis") or _AXIS_BY_KIND.get(kind, "seen")
    block["generalization_axis"] = axis
    # A whole-model capsule spans families by construction, is never held out (see
    # generalization_splits), and is expected to contain regions the target cannot run -- so it must
    # never carry must_accelerate, whatever the corpus default says.
    #
    # ⚠️ NEITHER MAY A CAPSULE THAT FORBIDS THE MESH. `lanes.forbid: [on_mesh]` is the negative lane:
    # the capsule exists to prove the compiler does NOT put this work on the accelerator, and
    # `must_accelerate: true` demands the opposite. Emitting both produces a capsule that cannot pass
    # under any behaviour, and it is produced by exactly the combination that looks safest -- a
    # corpus-wide `semantic_defaults: {must_accelerate: true}` meeting a host-only capsule. Measured:
    # declaring that default on one profile gave `SY_host_only_elementwise_map` and
    # `SY_host_only_reduction` `forbid: [on_mesh]` alongside `must_accelerate: true`. The lane
    # declaration is the more specific statement and wins, structurally rather than by an exception
    # list -- the same treatment `kind == "model"` already gets above.
    lanes = entry.get("lanes") or {}
    forbids_mesh = "on_mesh" in {str(x) for x in (lanes.get("forbid") or ())}
    must_default = False if (kind == "model" or forbids_mesh) else bool(defaults.get("must_accelerate", False))
    # An AUTHORED `true` cannot override the negative lane either: the contradiction is the same
    # whichever half asserted it, and a capsule declaring both is an authoring error to report rather
    # than a preference to honour.
    authored_must = authored.get("must_accelerate", must_default)
    if forbids_mesh and bool(authored_must):
        raise ValueError(
            f"{entry.get('name', '<unnamed>')}: lanes.forbid names 'on_mesh' while must_accelerate is "
            f"true. The negative lane exists to prove this work does NOT reach the accelerator, so the "
            f"two demands cannot both hold and the capsule could not pass under any behaviour"
        )
    block["must_accelerate"] = bool(authored_must)
    # WHY the strongest assertion was withheld. The schema has a field for it, and for the interop and
    # host-island capsules it is the whole contract: they withhold ``must_accelerate`` on purpose,
    # because host-lane work is the behaviour under test rather than a fallback failure, and the demand
    # they DO make is `lanes.require`. That reason was hand-written into the capsule after generation
    # and therefore deleted by the next regeneration -- measured on three gemmini capsules at once --
    # which left a withheld assertion looking like an author who simply never made one. Same argument
    # as the block around it: emit it from the profile so a regen preserves it.
    reason = authored.get("not_asserted_reason", defaults.get("not_asserted_reason"))
    if reason:
        block["not_asserted_reason"] = reason
    block["eligible"] = authored.get("eligible", defaults.get("eligible", "auto"))
    # FAMILIES THIS CAPSULE FUSES, so that a fused-only family can be covered at all.
    #
    # A target may declare a family reachable ONLY in composition -- `elementwise_map` with
    # `composed_with: [contraction]` on the target measured here. A standalone elementwise capsule would
    # then be the WRONG capsule (the eligibility oracle refuses it as a false fallback), so the only
    # capsule that can ever evidence that cell is a contraction with an epilogue. Crediting a capsule for
    # exactly one family therefore left such a cell permanently uncoverable while the requirement kept
    # demanding it -- a gap no capsule could close, reported forever as corpus debt.
    #
    # Derived, not declared: each epilogue stage the capsule carries is resolved to its family through
    # the same op->family table everything else uses, and kept only when the manifest says that family
    # composes with this capsule's own. A stage whose family does not resolve is recorded rather than
    # dropped, because silently withholding coverage a capsule earned is the failure this replaces.
    fused, unresolved = _fused_families(entry, fam, binding)
    if fused:
        block["composed_families"] = sorted(fused)
    if unresolved:
        block["composed_families_unresolved"] = sorted(unresolved)
    # ANY OTHER AUTHORED KEY SURVIVES. The block above consumes the keys it knows and, until this,
    # silently dropped the rest -- so a capsule synthesized to discharge a NAMED obligation could not
    # say which one, and the coverage reader had to re-derive the answer from attributes. That works
    # only when the obligation is expressible in attributes: a `padUNKNOWN` convolution window is not,
    # because the member can only spell zero padding, so the obligation it was built for could never
    # be matched by it. Carrying the reference is what closes that, and it is general rather than a
    # special case for one axis.
    for key, value in authored.items():
        if key not in block and key not in (
            "must_accelerate",
            "eligible",
            "not_asserted_reason",
            "semantic_family",
            "generalization_axis",
        ):
            block[key] = value
    return block
