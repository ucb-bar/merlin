"""Can a capsule's declared tolerance tell a working kernel from one that computes nothing?

A capsule grades by comparing the device output to a golden under ``numeric_policy``
(``|obs - exp| <= atol + rtol * |exp|``). If ``atol`` is comparable to the golden's own spread, then a
CONSTANT answer — the untouched zero fill, or any single value near the mean — satisfies the comparison
at every element, and the capsule passes whatever the kernel did. Such a capsule cannot fail, so its
pass is not evidence.

This is not hypothetical. Atlas's ``AF2_softmax`` declared ``atol: 0.25`` against a golden whose entire
range is ``[0.0139, 0.1523]``: while its operands were never reaching the device, the kernel returned a
constant ``softmax(0) = 1/32`` and the capsule PASSED, which is why it was the sole apparent survivor of
a corpus-wide preload defect and read as a working elementwise path. A whole-model capstone in the same
corpus accepted an all-zeros answer on the same grounds.

Target-agnostic: the candidates are derived from the capsule's OWN golden and compared under its OWN
declared policy. Nothing here knows a target, a dtype palette, or an op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def degenerate_answers(expected: np.ndarray) -> dict[str, np.ndarray]:
    """The constant answers to test this golden against, keyed by name.

    ``zeros`` is the untouched output buffer a kernel that never stores leaves behind. ``mean`` is the
    obvious guess. ``midrange`` is the OPTIMAL constant under a max-error criterion -- it minimises the
    worst element error -- so a policy that rejects it rejects EVERY constant, which is the property the
    ceiling below is derived from. Testing only zeros and mean would leave a band of tolerances that
    reject those two while still admitting a better constant.
    """
    if expected.size == 0:
        return {}
    mid = 0.5 * (float(expected.max()) + float(expected.min()))
    return {
        "zeros": np.zeros_like(expected),
        "mean": np.full_like(expected, expected.mean()),
        "midrange": np.full_like(expected, mid),
    }


def _accepts(observed: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> bool:
    return bool(np.all(np.abs(observed - expected) <= atol + rtol * np.abs(expected)))


def audit_capsule(capsule: dict, capsule_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Every ``{output, answer, atol, rtol, spread}`` this capsule would ACCEPT without computing.

    Empty means the capsule is falsifiable. Returns empty (rather than raising) for a capsule with no
    float tolerance policy or no readable golden — those are graded some other way and are not this
    check's business.
    """
    from . import capsule_golden

    policy = capsule.get("numeric_policy") or {}
    if policy.get("compare") != "tolerance_float":
        return []
    try:
        golden = capsule_golden.golden(capsule, capsule_dir)
    except Exception:  # noqa: BLE001 - a golden we cannot read is not a verdict
        return []
    return audit_outputs(policy, golden)


class UnmeasuredPolicy(ValueError):
    """A numeric policy cannot be assessed from the supplied oracle outputs."""


def audit_outputs(policy: dict, outputs, *, require_measurable: bool = False) -> list[dict[str, Any]]:
    """Test the existing constant candidates against already selected outputs.

    Strict callers distinguish unavailable, empty, ragged or non-finite outputs
    from an assessed policy. Legacy capsule queries retain their skip behavior.
    This function never discovers files, computes an oracle or changes a policy.
    """
    if policy.get("compare") != "tolerance_float":
        return []
    atol, rtol = float(policy.get("atol", 0.0)), float(policy.get("rtol", 0.0))
    if require_measurable and (not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0):
        raise UnmeasuredPolicy("tolerances must be finite and nonnegative")
    if require_measurable and (not isinstance(outputs, dict) or not outputs):
        raise UnmeasuredPolicy("no named oracle outputs could be measured")
    found: list[dict[str, Any]] = []
    for name, values in (outputs or {}).items():
        try:
            exp = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            if require_measurable:
                raise UnmeasuredPolicy(f"output {name!r} is not a comparable numeric tensor") from exc
            # A RAGGED golden (per-output arrays of differing shape) is not one comparable tensor, so
            # "the constant nearest every element" is not defined over it. Skip that output rather than
            # crash: a checker that dies on one corpus shape takes the whole gate down with it, and a
            # gate that cannot run is indistinguishable from one that found nothing.
            continue
        if exp.dtype == object or exp.size == 0:
            if require_measurable:
                raise UnmeasuredPolicy(f"output {name!r} is empty or nonnumeric")
            continue
        if require_measurable and not np.isfinite(exp).all():
            raise UnmeasuredPolicy(f"output {name!r} contains non-finite values")
        for answer, cand in degenerate_answers(exp).items():
            if _accepts(cand, exp, atol, rtol):
                found.append(
                    {
                        "output": name,
                        "answer": answer,
                        "atol": atol,
                        "rtol": rtol,
                        "spread": float(exp.max() - exp.min()),
                        "range": [float(exp.min()), float(exp.max())],
                    }
                )
    return found


def max_falsifiable_atol(expected: np.ndarray, rtol: float = 0.0) -> float:
    """The largest ``atol`` under which the best constant answer still FAILS this golden — the ceiling a
    falsifiable policy must stay below. Derived from the golden alone: the constant nearest to every
    element is the midrange, so the tightest element error any constant must incur is half the spread,
    less whatever the relative term already forgives there."""
    exp = np.asarray(expected, dtype=np.float64)
    if exp.size == 0:
        return 0.0
    mid = 0.5 * (exp.max() + exp.min())
    slack = np.abs(mid - exp) - rtol * np.abs(exp)
    return float(max(slack.max(), 0.0))


class UnfalsifiablePolicy(ValueError):
    """A capsule whose numeric policy no tolerance can make falsifiable, named so it can be fixed."""


def falsifiable_policy(policy: dict, outputs, *, name: str = "?") -> tuple[dict, dict]:
    """``(policy, provenance)`` with an absolute tolerance a constant answer cannot survive.

    THE TOLERANCE IS A PROPERTY OF THE DATAPATH; THE CEILING IS A PROPERTY OF THE GOLDEN. A profile
    declares ONE absolute tolerance for a whole target, which is the right shape for a datapath error
    budget and the wrong shape for a small-magnitude output: measured, a softmax capsule whose golden
    spans 0.0139..0.1523 was graded at ``atol: 0.25``, so zeros, the mean and the midrange all passed it.
    The capsule reported a numeric pass and proved nothing.

    A bounded-integer policy is a hardware error contract: if its golden cannot falsify the declared
    bound, raise and require a wider stimulus. For float tolerance policies, an absolute tolerance at
    or above the ceiling is replaced by the profile's OWN relative tolerance at the golden's scale
    (``rtol * max|golden|``). A capsule that is still unfalsifiable afterwards raises.
    """
    pol = dict(policy or {})
    atol = pol.get("atol")
    if atol is None:
        return pol, {"status": "not_applicable", "why": "policy declares no absolute tolerance"}
    rtol = float(pol.get("rtol") or 0.0)
    ceilings, scales = [], []
    for values in (outputs or {}).values():
        try:
            exp = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError):
            continue  # ragged golden: a ceiling is undefined, not zero
        if exp.dtype == object or exp.size == 0:
            continue
        ceilings.append(max_falsifiable_atol(exp, rtol))
        scales.append(float(np.abs(exp).max()))
    if not ceilings:
        return pol, {"status": "not_measured", "why": "no golden output could be sized"}
    ceiling, scale = min(ceilings), max(scales)
    prov = {
        "status": "ok",
        "declared_atol": float(atol),
        "ceiling_atol": ceiling,
        "basis": "the largest atol under which the best constant answer still fails this golden",
    }
    if float(atol) < ceiling:
        prov["falsifiable"] = True
        return pol, prov
    if pol.get("compare") == "bounded_int":
        raise UnfalsifiablePolicy(
            f"{name}: declared integer error bound {atol} reaches the golden's falsifiability "
            f"ceiling {ceiling}; widen the stimulus instead of silently changing the device's "
            "arithmetic contract"
        )
    derived = rtol * scale
    if not (derived < ceiling):
        raise UnfalsifiablePolicy(
            f"{name}: no absolute tolerance is both falsifiable and derivable here — the declared "
            f"atol {atol} and the relative tolerance at this golden's scale ({rtol} * {scale} = "
            f"{derived}) both reach the falsifiability ceiling {ceiling}. The golden has too little "
            f"spread to grade; change the capsule's shape or stimulus rather than its tolerance"
        )
    pol["atol"] = derived
    prov.update(
        falsifiable=True,
        applied_atol=derived,
        golden_scale=scale,
        rtol=rtol,
        why=(
            "the declared absolute tolerance reached the falsifiability ceiling, so the "
            "profile's own relative tolerance was applied at the golden's scale instead"
        ),
    )
    return pol, prov
