"""Numerical and provenance policy for optional cross-framework comparisons.

These historical study classifications and bars do not define capture-bundle
semantics or target capabilities. Keep the recorded evidence and refusal rules
beside the baseline arms that use them.
"""

from __future__ import annotations


# Models whose torch loader instantiates RANDOM weights (no pretrained checkpoint) AND whose capture
# bundle ships no ``weights.safetensors``. Their captured golden came from a *different* seeded
# instantiation, so a re-exported fp32 run can never reproduce it — even when the lowering is exact.
#
# MEASURED evidence (do not add a model here without it — an unjustified entry silently WEAKENS that
# model's correctness gate):
#   bitvla/fp32  cos=0.013323 vs captured golden, but 0.9999999999987 (rel 1.6e-06) vs compute_golden
#   openvla/fp32 cos=0.009043 vs captured golden; TVM's host-VM lowering cos=1.0 on the same model
#
# Deliberately EXCLUDED despite being random-init-ish: ``rdt2`` — its recomputed golden reproduces the
# capture exactly (norm 11.4439 both), so its captured golden IS reachable and must stay the gate.
# ``rdt`` is excluded too (recompute 53.24 vs capture 52.88 — close but unresolved; measure first).
# The llama family (tiny_llama/small_llama) loads real checkpoints and passes against captured goldens.
#
# For these models the correctness gate becomes LOWERING-EXACTNESS — "the framework reproduces eager
# torch for THIS instantiation" — NOT a semantic match against a trained model. Every such cell must
# say so; see ``lowering_exactness_note``. This mirrors the TVM arm (openvla cos=1.0) and the int8
# path, which already recomputes its reference for the same reason.
RANDOM_INIT_GOLDEN_UNREPRODUCIBLE: frozenset[str] = frozenset({"bitvla", "openvla", "smolvla"})


#: Models whose captured weights are RANDOM-INIT rather than a trained checkpoint, with the evidence
#: for that classification. This is a DIFFERENT question from whether the golden is reproducible, and
#: conflating the two mislabels a whole class of cell: smolvla is random-init yet perfectly
#: reproducible (a seeded re-instantiation matches its weights.safetensors bit-for-bit, 500/500
#: tensors), so a passing cosine on it measures framework-vs-eager agreement, NOT a semantic match
#: against trained weights -- while the reproducibility-keyed label called it "semantic".
#: Every entry is declared with what was actually checked; an unlisted model is treated as trained,
#: which is the status quo and is why the label still says what evidence backs it.
RANDOM_INIT_WEIGHTS: dict[str, str] = {
    "smolvla": (
        "the loader defaults M2M_SMOLVLA_PRETRAINED=0 and the capture was taken with it "
        "unset; the pretrained checkpoint on disk was deliberately NOT used, because "
        "capturing with it would have made the int8 cell a different model from the "
        "already-accepted fp32 cell. The PARAMETER TENSORS are reproducible (two independent "
        "processes agree on the params digest, which equals the digest of the bundle's own "
        "weights.safetensors over all 500 tensors) -- but the GOLDEN is not: measured "
        "2026-09-05, an eager fp32 re-instantiation through the loader scores cos 0.027 / "
        "rel 1.56 against golden.npy, BEFORE any quantization or export, and cos -0.031 "
        "with the zero-parameter perturbation the capture harness applies. A matching "
        "weights digest therefore does not imply a reachable golden here, which is why this "
        "model is also in RANDOM_INIT_GOLDEN_UNREPRODUCIBLE."
    ),
    "bitvla": "random-init capture; the golden's weights are additionally unrecoverable",
    "openvla": "random-init capture; the golden's weights are additionally unrecoverable",
}


def weights_are_random_init(model: str) -> bool:
    """True if ``model``'s captured weights are random-init rather than a trained checkpoint."""
    return model in RANDOM_INIT_WEIGHTS


def random_init_evidence(model: str) -> str:
    """What was checked to classify ``model``'s weights. Empty when it is not classified."""
    return RANDOM_INIT_WEIGHTS.get(model, "")


def golden_unreproducible(model: str) -> bool:
    """True if ``model``'s captured golden cannot be reproduced by re-instantiating its loader."""
    return model in RANDOM_INIT_GOLDEN_UNREPRODUCIBLE


def lowering_exactness_note(model: str) -> str:
    """The honesty label when gated on lowering-exactness, not semantics."""
    return (
        f" random-init({model}): gate=LOWERING-EXACTNESS (framework vs eager-torch on THIS seeded "
        "instantiation), NOT a semantic match — the captured golden's weights are unrecoverable "
        "(bundle ships no weights.safetensors), so cos measures lowering fidelity only."
    )


#: The fp32-tier cosine floor our own gate enforces (``zephyr_model._gate`` T2).
FP32_TIER_MIN_COS = 0.99

#: How many times the model's OWN quantization noise an int8 arm may deviate before it is called
#: wrong. Same constant our gate already applies to our arm as ``quant_excess`` -- named once so the
#: reference and we are judged by the same rule rather than by two numbers that happen to differ.
QUANT_EXCESS_K = 4.0

#: Absolute int8-vs-fp32 relative bar, used where a bundle ships no W8A8 reference to derive a floor
#: from. A bar is never LOOSENED below this; the derived one only ever raises it.
ABSOLUTE_INT8_REL = 5e-2


def quantization_floor(root) -> dict:
    """How far a bundle's own W8A8 reference sits from its fp32 golden.

    This is the yardstick an int8 arm is fairly judged against, DERIVED per bundle from the two
    goldens the bundle already ships -- never a constant. It matters because a constant does the
    wrong thing at both ends:

    * ``tiny_llama_int8_consistent``'s W8A8 reference is cos 0.9487 / rel 0.958 from its fp32 golden
      and FLIPS THE ARGMAX. An implementation reproducing that reference bit-for-bit would still
      fail a tier asking cos > 0.99 with a matching argmax, so the tier decides nothing there.
    * ExecuTorch's qd8 arm on that bundle measured rel 0.106 -- nearly ten times CLOSER to fp32 than
      quantizing the model at all is -- and was refused against an absolute 0.05. Mis-specified, not
      strict.

    Missing or unreadable goldens give UNKNOWN (None), never a comfortable True.
    """
    from pathlib import Path as _P

    import numpy as np

    root = _P(root)
    out: dict = {"floor_cos": None, "floor_rel": None, "floor_argmax": None, "fp32_tier_reachable": None, "note": ""}
    fp32_p, w8a8_p = root / "golden.npy", root / "golden_w8a8.npy"
    if not (fp32_p.is_file() and w8a8_p.is_file()):
        out["note"] = (
            "bundle ships fewer than both goldens, so its quantization floor is UNKNOWN "
            "and an int8 arm here can only be judged against the absolute bar"
        )
        return out
    try:
        f = np.load(fp32_p, allow_pickle=False).astype("float64").ravel()
        q = np.load(w8a8_p, allow_pickle=False).astype("float64").ravel()
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"goldens unreadable ({type(exc).__name__}: {exc}); floor UNKNOWN"
        return out
    if f.size != q.size:
        # Two references of different extent are not two views of the same output, so the distance
        # between them is not this model's quantization floor. Truncating to the shorter one would
        # produce a confident number from a comparison that was never valid.
        out["note"] = (
            f"the fp32 golden has {f.size} elements and the W8A8 reference {q.size}: "
            "they do not describe the same output, so the quantization floor is UNKNOWN "
            "and an int8 arm here can only be judged against the absolute bar"
        )
        return out
    if np.array_equal(f, q):
        # The two references are the SAME ARRAY. This is not a bundle whose quantization happens to
        # be lossless -- it is a bundle whose W8A8 golden is a copy of its fp32 one, so the pair
        # measures nothing. resnet50_v1_5_int8_w8a8_consistent ships exactly this (both files
        # sha256 3f640279a604...), and the unguarded path returned floor_rel 0.0 / floor_cos 1.0 /
        # fp32_tier_reachable True, which then derived the TIGHTEST POSSIBLE bar from the LEAST
        # informative comparison -- the same trap the all-zeros branch below exists to close. The
        # real floor for that bundle, taken against the fp32 bundle's own golden, is cos 0.999314 /
        # rel 0.029439, and the fp32 tier IS reachable; the identical-file path said so for the
        # wrong reason and would have failed a correct implementation on a fabricated bar.
        out["note"] = (
            "the fp32 and W8A8 goldens are byte-identical, so this pair measures no "
            "quantization distance at all: the floor is UNKNOWN, not zero, and an int8 "
            "arm here can only be judged against the absolute bar"
        )
        return out
    denom = float(np.linalg.norm(q) * np.linalg.norm(f))
    if not denom:
        # One of the references is all zeros, so there is no direction to take a cosine of and no
        # scale to take a relative error against. UNKNOWN, not a floor of zero -- a floor of zero
        # would derive the TIGHTEST possible bar from the least informative reference.
        out["note"] = (
            "a golden is all zeros, so this bundle has no measurable quantization floor "
            "and an int8 arm here can only be judged against the absolute bar"
        )
        return out
    out["floor_cos"] = float(q @ f / denom)
    out["floor_rel"] = float(np.abs(q - f).max()) / max(1e-9, float(np.abs(f).max()))
    out["floor_argmax"] = bool(int(np.argmax(q)) == int(np.argmax(f)))
    out["fp32_tier_reachable"] = bool(
        out["floor_cos"] is not None and out["floor_cos"] > FP32_TIER_MIN_COS and out["floor_argmax"]
    )
    if not out["fp32_tier_reachable"]:
        out["note"] = (
            f"the fp32 tier is UNREACHABLE on this bundle: its own W8A8 reference scores cos "
            f"{out['floor_cos']:.6f} / rel {out['floor_rel']:.4f} / argmax {out['floor_argmax']} "
            f"against the fp32 golden, so an implementation matching that reference bit-for-bit "
            f"would still fail a tier asking cos > {FP32_TIER_MIN_COS} and a matching argmax. Read "
            "a failure there as a property of quantizing THIS model, not as evidence about the "
            "implementation."
        )
    return out


def int8_accuracy_bar(root) -> dict:
    """The (min_cos, max_rel) an int8 arm on this bundle is judged by, and how it was derived.

    Derived from :func:`quantization_floor`, so the reference arm and our arm are held to the same
    RULE (deviate by at most ``QUANT_EXCESS_K`` times the model's own quantization noise) rather
    than to two absolute numbers that happen to differ. The derived bar never falls below the
    absolute one; it only rises where the model's own quantization is itself far from fp32.
    """
    floor = quantization_floor(root)
    cos_thr, rel_thr = FP32_TIER_MIN_COS, ABSOLUTE_INT8_REL
    basis = "absolute (no W8A8 reference to derive a floor from)"
    if floor.get("floor_rel") is not None:
        rel_thr = max(ABSOLUTE_INT8_REL, QUANT_EXCESS_K * floor["floor_rel"])
        # An arm cannot be required to sit CLOSER to fp32 than quantizing the model at all does.
        cos_thr = min(FP32_TIER_MIN_COS, float(floor["floor_cos"]))
        basis = (
            f"derived: rel = max({ABSOLUTE_INT8_REL:g}, {QUANT_EXCESS_K:g} x floor_rel "
            f"{floor['floor_rel']:.4f}) = {rel_thr:.4f}; cos = min({FP32_TIER_MIN_COS:g}, "
            f"floor_cos {floor['floor_cos']:.6f}) = {cos_thr:.6f}"
        )
    return {"cos_threshold": cos_thr, "rel_threshold": rel_thr, "basis": basis, "floor": floor}
