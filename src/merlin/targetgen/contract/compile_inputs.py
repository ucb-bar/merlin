"""Independent logical operands and explicit prepack or final-profile selections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _recorded_operands(cb: dict[str, Any]) -> dict[str, list] | None:
    """The operands an independent-float or WHOLE-PROGRAM buffer must run on, or ``None``.

    A capsule graded under a float policy cannot have had its answer recomputed on the integer engine,
    so its golden is the INDEPENDENT one — computed off-device, on the operands the runner attached to
    the buffer as ``canonical_inputs``. The device is only answerable against that golden if it runs
    on those same operands; nothing consumed them on this path, so the harness materialized each leaf
    from its NAME and the device computed the right function of the wrong inputs — a guaranteed
    mismatch, reported as a functional failure of the submission.

    THE FLOAT CONDITION remains load-bearing for ordinary buffers. A per-op integer buffer is graded
    against a golden recomputed from deterministic name-materialized fill, and may carry unrelated
    historical operands (``GS0_matmul_spec`` is the guard). A WHOLE-PROGRAM buffer is different: the
    runner has already replaced/re-keyed ``canonical_inputs`` with the exact semantic stimulus because
    its compiler may rename leaves positionally. Discarding that table rematerializes from ``arg0``
    instead of ``A0`` and guarantees a false mismatch. The explicit ABI discriminator separates that
    exception without changing legacy per-op integer behavior.
    """
    from merlin.runtime.backends import base as _backends
    from merlin.runtime.commandbuffer import declared_output_dtypes

    recorded = cb.get("canonical_inputs") or {}
    tensors = cb.get("tensors") or {}
    if not recorded:
        return None
    dtypes = declared_output_dtypes(cb)
    outputs = [n for n, s in tensors.items() if (s or {}).get("role") == "output"]
    whole_program = (cb.get("kernel_abi") or {}).get("kind") == "whole_program"
    if not whole_program and (not outputs or not all(_backends.float_format_of(dtypes.get(n, "")) for n in outputs)):
        return None
    return {
        name: spec["values"]
        for name, spec in recorded.items()
        if isinstance(spec, dict) and spec.get("values") is not None and name in tensors
    } or None


def _explicit_prepack_inputs(inputs, authorizations) -> None:
    """Never satisfy a host immutable-payload grant from candidate recorded operands."""
    if authorizations is None:
        return
    if not isinstance(authorizations, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("host prepack authorization requires explicit logical inputs")
    from merlin.runtime.prepack_authority import HostPrepackAuthorization

    if any(type(grant) is not HostPrepackAuthorization for grant in authorizations.values()):
        raise ValueError("prepack requires an exact host authorization object")
    if set(authorizations) - set(inputs):
        raise ValueError("authorized prepack input is missing from explicit logical inputs")


def _strict_warm_profile(profile, cb=None):
    """Validate the explicit final-profile capability without naming a target."""
    if profile is None:
        return None
    from merlin.perf.warm_profile_harness import require_strict_final_warm_profile

    validated = require_strict_final_warm_profile(profile)
    if cb is not None and (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
        raise ValueError("strict final warm profiling requires an explicit whole-program kernel ABI")
    return validated


# Preserve historical callable and record pickle lookup identities.
_recorded_operands.__module__ = "merlin.targetgen.contract.compile"
_explicit_prepack_inputs.__module__ = "merlin.targetgen.contract.compile"
_strict_warm_profile.__module__ = "merlin.targetgen.contract.compile"
