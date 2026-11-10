"""Concrete independent source observations for closed component mechanisms."""

from __future__ import annotations

import math
from fractions import Fraction

SOURCE_NUMERIC_EFFECTS = frozenset({"quantizer_tie_input", "quantizer_clamp_input", "producer_signed_zero_observer"})


def verify(capsule, golden):
    operation = capsule["operation"]
    if operation["op"] != "producer_quantizer_observer":
        return {"effects": [], "scope": "no selected closed source mechanism witness"}
    import numpy as np

    from merlin.targetgen.component_sources import parameters

    attrs = operation["attributes"]
    selected = parameters({"dtype": "f32", **attrs})
    if any(attrs.get(key) != value for key, value in selected.items()):
        raise ValueError("source quantizer omitted its exact scalar or signed storage bounds")
    outputs = golden.get("outputs") or {}
    names = attrs.get("outs") or []
    if len(names) != 4 or set(names) != set(outputs) or golden.get("golden_source") != "host_torch_eager":
        raise ValueError("source observer omitted a complete independently recorded publication")
    inputs = (golden.get("oracle_provenance") or {}).get("inputs") or {}
    if "X" not in inputs:
        raise ValueError("source observer omitted its actual producer operand")
    source = inputs["X"]
    x = np.asarray(source["decoded"], dtype=np.float32).reshape(source["shape"])
    producer = np.asarray(outputs[names[0]], dtype=np.float32)
    expected = x * np.float32(selected["producer_scale"])
    if not np.array_equal(producer, expected) or not np.array_equal(np.signbit(producer), np.signbit(expected)):
        raise ValueError("published source producer differs from its exact typed source operands")
    normalized = producer / np.float32(selected["quant_scale"])
    if not np.isfinite(normalized).all():
        raise ValueError("source observer quantizer has an unmodeled nonfinite conversion")
    # Fraction's nearest-even integer projection is independent of the source
    # Torch round/clamp/cast path. Power-of-two division is source-format exact
    # except under/overflow, which the typed f32 normalization observes first.
    rounded = [round(Fraction(float(value))) for value in normalized.flat]
    codes = np.asarray(
        [min(selected["quant_max"], max(selected["quant_min"], value)) for value in rounded], dtype=np.float32
    ).reshape(producer.shape)
    observed_codes = np.asarray(outputs[names[1]], dtype=np.float32)
    decoded = np.asarray(outputs[names[2]], dtype=np.float32)
    if not np.array_equal(observed_codes, codes) or not np.array_equal(decoded, codes * selected["quant_scale"]):
        raise ValueError("published quantizer codes or decoded values differ from exact RNE/clamp source semantics")
    effects = set()
    tie_count = sum(abs(Fraction(float(value)) % 1) == Fraction(1, 2) for value in normalized.flat)
    clamp_count = sum(value < selected["quant_min"] or value > selected["quant_max"] for value in rounded)
    negative_zero_count = int(np.sum((producer == 0) & np.signbit(producer)))
    if tie_count:
        effects.add("quantizer_tie_input")
    if clamp_count:
        effects.add("quantizer_clamp_input")
    if negative_zero_count:
        effects.add("producer_signed_zero_observer")
    return {
        "effects": sorted(effects),
        "producer_elements": math.prod(producer.shape),
        "ties": tie_count,
        "clamps": clamp_count,
        "negative_zero_producer_elements": negative_zero_count,
        "outputs": names,
        "scope": "independent complete source producer/RNE/clamp observations; candidate and FENV unverified",
    }
