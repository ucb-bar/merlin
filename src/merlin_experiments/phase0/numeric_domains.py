"""Observed full input/output domains under explicitly selected numerical rules.

This is generation-time program admission. It never accepts a candidate's output
or changes any numerical oracle, tolerance, rounding order or acceptance gate.
"""

from __future__ import annotations

import math

from merlin.targetgen import golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves


def _scalars(value):
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _scalars(child)
    else:
        if type(value) not in (int, float):
            raise ValueError("numerical domain requires complete numeric tensors")
        yield value


def screen(capsule, directory, semantics):
    policies = {field: semantics[field] for field in ("input_domain", "output_domain") if field in semantics}
    if not policies:
        return None
    try:
        golden = golden_store.load_golden(directory)
        if not isinstance(golden, dict) or not golden.get("outputs"):
            raise ValueError("selected numerical domain has no independent full-output reference")
        inputs = (golden.get("oracle_provenance") or {}).get("inputs") or {}
        if inputs:
            observed_inputs = {row["name"]: inputs[row["name"]]["decoded"] for row in capsule["inputs"]}
        else:
            observed_inputs = {name: value.data for name, value in materialize_capsule_leaves(capsule).items()}
        observations = {}
        refused = []
        for field, values in (("input_domain", observed_inputs), ("output_domain", golden["outputs"])):
            if field not in policies:
                continue
            finite = nonfinite = 0
            for tensor in values.values():
                for value in _scalars(tensor):
                    if math.isfinite(value):
                        finite += 1
                    else:
                        nonfinite += 1
            observations[field] = {
                "tensors": len(values),
                "finite_elements": finite,
                "nonfinite_elements": nonfinite,
                "policy": policies[field]["nonfinite"],
            }
            if not values or not finite + nonfinite:
                raise ValueError("selected numerical domain has no concrete complete tensor observations")
            if nonfinite and policies[field]["nonfinite"] == "forbid":
                refused.append(field + " has forbidden nonfinite values")
        return {
            "status": "unsupported" if refused else "within_domain",
            "policies": policies,
            "observations": observations,
            "reason": "; ".join(refused) or "complete numerical values match selected domains",
            "scope": "source operands and independent reference outputs; candidate arithmetic unverified",
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "unknown", "policies": policies, "reason": str(exc), "observations": {}}
