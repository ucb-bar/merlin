"""Which precision a target should compile a model in, and why the others were refused.

`workload_spec.precision_preference` is a RANKING, and until now it could only ever filter: synthesis
compared it against the dtypes a target admits and reported the survivors. Nobody could ask the direct
question -- "what is the best format this target supports for this model" -- so the answer lived in
whichever profile entry happened to name a dtype.

THREE THINGS DECIDE IT, and they are kept apart because they fail differently:

``admitted``
    the target's capability manifest says the hardware has a datapath for it. A preference can never
    widen this; a format the hardware lacks is not a choice, it is absent.
``expressible``
    the format registry knows the format, so the compiler can spell it. A token nobody can decode is a
    typo, not a capability gap, and reporting it as one sends the reader to the wrong place.
``certified``
    an accuracy gate has measured that the model survives it. This is the one that is usually NOT
    established, and it is reported as unestablished rather than assumed -- picking the narrowest format
    the hardware admits and calling it "best" is exactly how a compiler ships a model that runs fast and
    answers wrong.

The result therefore names a candidate and states which of the three it rests on. A caller that needs
accuracy assurance must read ``certified`` and refuse to proceed on ``not_established``; a caller that
only needs a legal datapath can use ``admitted`` alone.
"""
from __future__ import annotations

from typing import Any


def _is_expressible(token: str) -> bool | None:
    """Does the format registry know this token? ``None`` when the registry itself cannot be consulted.

    `conformance.capsule_dtype` passes an unknown token THROUGH unchanged rather than raising -- it maps
    spellings, it does not validate them -- so an unmappable token would otherwise be reported as a
    capability gap when it is a typo. The registry is the thing that knows.
    """
    try:
        from merlin.common import quant_formats as QF

        return bool(QF.has(token))
    except Exception:                              # noqa: BLE001 -- an unreadable registry answers nothing
        return None


def best_format(target: str, *, preference: list | tuple | None = None,
                family: str = "contraction", admitted: "set | frozenset | None" = None) -> dict:
    """The highest-ranked precision ``target`` admits for ``family``, with the rejections named.

    ``preference`` defaults to the target's declared ``workload_spec.precision_preference``. Every
    rejected token carries WHY, because "we preferred int8 and this target has no int8 datapath" and
    "int8 is a typo" send a reader to different places.

    ``admitted`` (capsule spellings) lets a caller that ALREADY holds the derived requirement pass it in
    instead of having this re-read the manifest. That is not an optimisation: :mod:`corpus_synth` is
    supposed to decide from the requirement document alone, and a second read of the manifest there
    would be a second source of truth for what the target admits -- the exact defect that produces two
    answers to one question.
    """
    from merlin.targetgen import conformance as CF

    # BOTH SIDES IN ONE SPELLING. `admitted` speaks the registry's tokens and a cell speaks the capsule's
    # ("int8" vs "i8"); comparing them raw reported every admitted dtype as not-admitted, which is the
    # same spelling bug `corpus_synth.filtered_precision` was written to avoid.
    if admitted is not None:
        admitted = {str(d) for d in admitted}
    else:
        admitted_by_family = CF.admitted(target)
        admitted = set()
        for d in (admitted_by_family.get(family) or ()):
            try:
                admitted.add(CF.capsule_dtype(str(d)))
            except Exception:                      # noqa: BLE001 -- keep an unmappable token visible
                admitted.add(str(d))

    if preference is None:
        preference = _declared_preference(target)
    ranked = [str(p) for p in (preference or ())]

    chosen, rejected = None, []
    registry_readable = True
    for token in ranked:
        known = _is_expressible(token)
        if known is None:
            registry_readable = False
        if known is False:
            rejected.append({"format": token, "why": "not_expressible",
                             "detail": "no registry entry decodes this token; it is a spelling error "
                                       "rather than a capability gap"})
            continue
        capsule_spelling = CF.capsule_dtype(token)
        if capsule_spelling not in admitted:
            rejected.append({"format": token, "capsule_dtype": capsule_spelling,
                             "why": "not_admitted",
                             "detail": f"the capability manifest declares no {family} datapath for it"})
            continue
        chosen = {"format": token, "capsule_dtype": capsule_spelling}
        break

    # NO PREFERENCE IS NOT NO ANSWER. A target that declares none has not said which of its admitted
    # formats it wants, and reporting `chosen: None` beside an empty rejection list reads as "nothing
    # qualified" when the truth is "nobody was asked".
    no_preference = not ranked
    return {
        "target": target, "family": family, "preference": ranked,
        "status": ("no_preference_declared" if no_preference
                   else "ok" if chosen else "no_admitted_format_in_preference"),
        "admitted": sorted(admitted),
        "chosen": chosen,
        "rejected": rejected,
        # THE HALF THAT IS NOT ESTABLISHED, said plainly. Choosing the narrowest admitted format and
        # calling it best is how a compiler ships a model that runs fast and answers wrong.
        "certified": {
            "status": "not_established",
            "detail": ("no accuracy gate has measured this model in this format on this target; "
                       "`admitted` says the hardware has the datapath, not that the model survives it"),
        },
        "registry_readable": registry_readable,
        "basis": ("preference RANKS; the manifest ADMITS; the registry EXPRESSES. A preference can "
                  "never widen what the hardware has, and none of the three is an accuracy claim"),
    }


def _declared_preference(target: str) -> list:
    """The target's own declared precision preference, or an empty ranking.

    The descriptor is located by ``corpora.descriptor_path`` rather than by assembling the path here:
    where a target's descriptor lives is the corpora module's fact, and a second copy of that layout is
    a second thing to update when it moves.
    """
    try:
        from merlin.targetgen.corpora import descriptor_path
        from merlin.targetgen.target_experiment import load_target_experiment

        p = descriptor_path(target)
        if p is None or not p.is_file():
            return []
        ws: dict[str, Any] = dict(getattr(load_target_experiment(p), "workload_spec", None) or {})
        return [str(x) for x in (ws.get("precision_preference") or ())]
    except Exception:                              # noqa: BLE001
        return []
