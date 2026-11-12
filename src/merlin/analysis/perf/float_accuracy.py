"""An open model's end result against the FLOAT model its capture quantized, at the capture's tolerance.

The whole-model gate's other end-result checks are self-consistency: the program against its own
reference arm (bit-identical), the oracle, and the capsule's golden, which is the QUANTIZED model run in
torch. None of them can see that a quantization scheme made the model wrong: a variant agrees with
itself however far it is from the network it was meant to approximate. This check asks that question.

The reference is the capture's own ``float_reference`` (the loader's model, run before any
quantization or precision cast, recorded beside the golden by the capture worker). The threshold is the
capsule's declared ``numeric_policy`` (``atol``/``rtol``): an element is within when
``|program - float| <= atol + rtol * |float|``, and the check passes only when every element is. It
FAILS CLOSED -- passed False with the reason -- when the capture declares no float reference or no
tolerance, when the program printed less than the whole output, or when the output is not the single
tensor the reference names. Nothing here knows a target or a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The golden-document key the capture worker writes the float model's outputs under.
FLOAT_REFERENCE = "float_reference"


def _refused(why: str, **extra: Any) -> dict[str, Any]:
    return {"passed": False, "basis": "float reference", "note": why, **extra}


def declared_policy(capsule: Path) -> tuple[float, float] | None:
    """``(atol, rtol)`` the capsule declares for its float output, or ``None`` when it declares none."""
    import yaml

    path = Path(capsule) / "capsule.yaml"
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    policy = document.get("numeric_policy") if isinstance(document, Mapping) else None
    if not isinstance(policy, Mapping) or policy.get("atol") is None or policy.get("rtol") is None:
        return None
    return float(policy["atol"]), float(policy["rtol"])


def declared_reference(capsule: Path):
    """The float model's single output the capture recorded, as float64, or ``(None, why)``."""
    import numpy as np

    from merlin.development.targetgen.golden_store import load_golden

    golden = load_golden(capsule)
    if not isinstance(golden, Mapping):
        return None, "the capsule has no golden document"
    reference = golden.get(FLOAT_REFERENCE)
    outputs = reference.get("outputs") if isinstance(reference, Mapping) else None
    if not isinstance(outputs, Mapping) or not outputs:
        return None, "the capture declares no float reference (its golden is the quantized model's output)"
    if len(outputs) != 1:
        return None, f"the float reference names {len(outputs)} outputs; the program prints one"
    (values,) = outputs.values()
    return np.asarray(values, np.float64).reshape(-1), None


def compare(program, reference, atol: float, rtol: float) -> dict[str, Any]:
    """Element-wise agreement under ``atol``/``rtol``; both operands must be finite."""
    import numpy as np

    a, b = np.asarray(program, np.float64), np.asarray(reference, np.float64)
    gap = np.abs(a - b)
    # An infinite reference otherwise makes both sides infinite: inf <= inf
    # would accept a finite candidate. Nonfinite operands are always violations.
    within = int((np.isfinite(a) & np.isfinite(b) & (gap <= atol + rtol * np.abs(b))).sum())
    scale = np.maximum(np.abs(b), np.finfo(np.float64).tiny)
    norms = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "within": within,
        "of": int(b.size),
        "max_abs": float(gap.max()) if b.size else 0.0,
        "max_rel": float((gap / scale).max()) if b.size else 0.0,
        "cosine": float(a @ b / norms) if norms > 0 else None,
        "policy": {"atol": atol, "rtol": rtol},
    }


def check(console: str, capsule: str | Path) -> dict[str, Any]:
    """The program's printed output (``OUT`` line) against the capture's float reference."""
    from merlin.analysis.perf import whole_model_reference as R

    capsule = Path(capsule)
    policy = declared_policy(capsule)
    if policy is None:
        return _refused("the capsule declares no numeric_policy atol/rtol to judge against")
    reference, why = declared_reference(capsule)
    if reference is None:
        return _refused(why)
    bits = R.output_bits(console)
    if not bits:
        return _refused("the program printed no output values")
    if len(bits) != reference.size:
        return _refused(
            f"the program printed {len(bits)} of the {reference.size} output elements; "
            "accuracy is judged over the whole output"
        )
    result = compare(R._floats(bits), reference, *policy)  # noqa: SLF001 -- the module's own bit decoding
    result.update(basis="float reference", passed=result["within"] == result["of"])
    return result
