"""Which captured models MEASURE the generalization claim, and are therefore barred from deriving it.

``conformance.required_cells`` derives the requirement from what real captures CONTAIN, the synthesized
corpus covers those cells, and coverage is then reported over captured models. One capture doing both
jobs makes the claim circular -- the corpus was built from the model it is said to generalize to.
Measured before this split existed: every bundle under ``out/artifacts/recaptures/`` fed the derivation,
the four claim models included, and lstmnetvit was already in both roles.

The declaration is ``merlin/contract/claim_models.yaml`` (tracked and reviewed). This module is the only
reader, so "is this model held out?" has one answer.

⚠️ MATCHING IS STRUCTURAL, ON TOKEN BOUNDARIES. A bundle is named ``<model>_<dtype>_<variant>``
(``resnet50_v1_5_fp32_consistent``, ``tiny_llama_fp32_full``), so a claim model matches a bundle when its
tokens are a whole-token PREFIX of the bundle's. Substring matching would be wrong in both directions:
``small_llama`` contains no claim model but shares a token with ``tiny_llama``, and a future
``lstmnetvit2`` must NOT match ``lstmnetvit``. Nothing here uses regex -- the pattern that silently
matches too much or too little is the failure this repo's parsing rule exists to prevent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

_CONTRACT = ("contract", "claim_models.yaml")


def _doc() -> dict:
    import yaml

    from merlin.common.paths import merlin_dir
    p = merlin_dir().joinpath(*_CONTRACT)
    if not p.is_file():
        raise FileNotFoundError(
            f"no claim-model declaration at {p}; the derivation/claim split cannot be applied, and "
            f"deriving the requirement from every capture would make the coverage claim circular")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not doc.get("claim_models"):
        raise ValueError(f"{p} declares no `claim_models`")
    return doc


def claim_models() -> tuple[str, ...]:
    """The held-out model names, as declared."""
    return tuple(str(m) for m in _doc()["claim_models"])


def exclusion_rule() -> str:
    """The prose standard a reviewer applies. Returned rather than paraphrased at call sites."""
    return str((_doc().get("exclusion") or {}).get("rule") or "")


def forbidden_sources() -> tuple[str, ...]:
    """The artifact classes that carry claim-model facts and therefore may not enter derivation."""
    return tuple(str(s) for s in (_doc().get("exclusion") or {}).get("forbidden_sources") or ())


def known_derivation_gaps() -> tuple[dict, ...]:
    """Requirement branches left underived BECAUSE a claim model was held out.

    Recorded rather than silently accepted: holding out the only convolution-dominated capture removes
    the conv branch from the requirement, and the honest response is a different derivation model, not
    re-admitting the claim model.
    """
    return tuple(_doc().get("known_derivation_gaps") or ())


def _tokens(name: str) -> tuple[str, ...]:
    return tuple(t for t in str(name).split("_") if t)


def model_of(bundle: str) -> str | None:
    """The claim model a bundle name belongs to, or ``None``.

    Whole-token prefix, longest match first: ``tiny_llama`` and a hypothetical ``tiny`` must not both
    claim ``tiny_llama_fp32_full``, and the more specific declaration is the one that means it.
    """
    bt = _tokens(bundle)
    best: str | None = None
    best_len = 0
    for m in claim_models():
        mt = _tokens(m)
        if len(mt) > len(bt) or bt[:len(mt)] != mt:
            continue
        if len(mt) > best_len:
            best, best_len = m, len(mt)
    return best


def is_claim_bundle(bundle: str) -> bool:
    """True when this bundle carries a held-out model's data."""
    return model_of(bundle) is not None


def partition(captures: dict[str, Path | str]) -> tuple[dict, dict]:
    """``(derivation, claim)`` -- split a capture map by the declared holdout.

    Everything that is not a claim bundle derives, per the declared
    ``derivation_policy``: a newly captured model joins derivation by DEFAULT, which is the safe
    direction, because the unsafe one is a claim model quietly entering it.
    """
    derivation, claim = {}, {}
    for name, path in (captures or {}).items():
        (claim if is_claim_bundle(name) else derivation)[name] = path
    return derivation, claim


def covered_claim_models(captures: Iterable[str]) -> dict[str, list[str]]:
    """``{claim model: [bundles carrying it]}`` -- including the empty lists.

    An uncaptured claim model is reported with no bundles rather than omitted: a claim measured over a
    model nobody captured is a claim about nothing, and that has to be visible.
    """
    out: dict[str, list[str]] = {m: [] for m in claim_models()}
    for name in captures:
        m = model_of(name)
        if m is not None:
            out[m].append(str(name))
    return {k: sorted(v) for k, v in out.items()}
