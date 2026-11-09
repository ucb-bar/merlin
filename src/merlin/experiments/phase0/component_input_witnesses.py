"""Join selected deterministic palettes to actual independently recorded operands."""

from __future__ import annotations

import math

from merlin.targetgen.input_palette import findings, pattern, realize

NUMERIC_INPUT_EFFECTS = frozenset(
    {
        "signed_input",
        "signed_zero_input",
        "subnormal_input",
        "maximum_finite_input",
        "nonfinite_input",
        "rounding_tie_input",
        "cancellation_input",
    }
)


def verify(capsule, golden):
    palette = capsule.get("input_palette")
    if palette is None:
        return {"effects": [], "inputs": {}, "scope": "no selected numerical input palette"}
    used, records, effects = set(), {}, set()
    provenance = (golden.get("oracle_provenance") or {}).get("inputs") or {}
    for index, row in enumerate(capsule.get("inputs") or []):
        name, shape, dtype = row["name"], row["shape"], row["dtype"]
        values = realize(palette, name=name, shape=shape, dtype=dtype, index=index)
        if values is None:
            continue
        used.add(pattern(palette, name=name, dtype=dtype, index=index)["name"])
        observed = provenance.get(name)
        if golden.get("golden_source") in {"host_torch_eager", "ieee_simt_f32_accumulate", "specir_refmodel_float"}:
            if (
                not isinstance(observed, dict)
                or observed.get("shape") != shape
                or len(observed.get("decoded") or []) != len(values)
            ):
                raise ValueError("independent golden omitted the selected full numerical stimulus")
            for wanted, actual in zip(values, observed["decoded"], strict=True):
                same = (math.isnan(wanted) and math.isnan(actual)) or wanted == actual
                if not same or (wanted == 0 and math.copysign(1, wanted) != math.copysign(1, actual)):
                    raise ValueError("independent recorded numerical stimulus differs from the selected palette")
        elif golden.get("golden_source") not in {"merlin_tensor_int", "merlin_tensor_component_program"}:
            raise ValueError("selected independent golden source has no exact numerical input-palette witness")
        observed_effects = findings(palette, name=name, shape=shape, dtype=dtype, index=index)
        effects.update(observed_effects)
        records[name] = {"shape": shape, "dtype": dtype, "effects": observed_effects, "elements": len(values)}
    if any(row["name"] not in used for row in palette["inputs"]) or not records:
        raise ValueError("selected numerical palette has no concrete generated tensor witness")
    return {
        "effects": sorted(effects),
        "inputs": records,
        "scope": "actual source input patterns; arithmetic order, FENV and candidate execution unverified",
    }
