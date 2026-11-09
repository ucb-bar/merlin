"""Project admitted numerical candidates through the existing TorchAO capture policy.

These recipes are prospective preparation inputs. Actual framework realization,
independent numerical checks and placement remain separate responsibilities.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

from merlin.target.spec.quantization import semantic_extensions


def capture_recipe_candidates(
    spec: Mapping,
    quantization_contract: Mapping,
    *,
    preparation_profile=None,
    model: str | None = None,
    alternative_sha256: str | None = None,
) -> list[dict]:
    """Project saved hardware recipes through explicit SW format/operation scopes.

    These are diagnostic transformation inputs, not reviewed accelerator support.
    Framework observer policy is recorded from the existing adapter, not inferred
    from RTL or added to the authored SW spec. Actual calibration and transformed
    precision must still be observed by the capture producer.
    """
    from merlin.compiler.quantization.recipe import digest
    from merlin.compiler.quantization.torchao.application import DEFAULT_OBSERVER_EPSILON, _observer_name
    from merlin.targetgen.semantic_families import from_op

    preparation = quantization_contract.get("preparation_profile")
    if preparation_profile is not None:
        from merlin.compiler.quantization.preparation import select_recipe

        if preparation is None or preparation["identity"] != preparation_profile.identity():
            raise ValueError("capture preparation profile differs from the selected quantization contract")
    elif preparation is not None or model is not None or alternative_sha256 is not None:
        raise ValueError("capture requires the explicitly selected application preparation profile")
    operations = {row["id"]: row for row in spec.get("operations") or []}
    # These records are preserved by intake, but the generic recipe adapter
    # cannot realize their encoding/output semantics. Even a caller-supplied
    # stale diagnostic contract cannot turn them into executable recipes.
    if semantic_extensions(spec):
        return []
    result = []
    for declaration in quantization_contract.get("formats") or []:
        selected = declaration["declaration"]
        rows = [
            operations[identity]
            for identity in selected["eligible_operations"]
            if operations[identity]["placement"] == "accelerator"
        ]
        families = {family for row in rows for family in row.get("families", [])}
        families.update(family for row in rows for op in row.get("ops", []) if (family := from_op(op)))
        for match in declaration.get("hardware_matches") or []:
            if match["status"] != "candidate" or not rows:
                continue
            recipe = copy.deepcopy(match["candidate"]["recipe"])
            # Dynamic module transforms cannot enforce FX signature constraints.
            # Preserve them in the diagnostic contract, not as executable scoped recipes.
            if recipe["activation"]["mode"] != "static":
                continue
            allowed = [family for family in recipe["families"] if family in families]
            if not allowed:
                continue
            for family in set(recipe["families"]) - set(allowed):
                recipe.setdefault("unquantized", {})[family] = "not selected by the SW quantization operation scope"
            recipe["families"] = allowed
            recipe["accumulator_dtype"] = selected["accumulator_dtype"]
            recipe["software_admission"] = {
                "target": spec["target"],
                "status": spec["status"],
                "operations": copy.deepcopy(rows),
            }
            # This authored numerical choice determines which independent
            # framework reference may judge the integer rewrite. Hardware
            # facts derive the format, not the reference semantics.
            recipe["software_numerical_engine"] = spec["numerical_semantics"]["model"]["engine"]
            framework = selected.get("framework") or {}
            unsupported = set(framework) - {
                "activation_observer",
                "weight_observer",
                "observer_epsilon",
                "calibration_count",
            }
            if unsupported:
                raise ValueError(f"capture adapter cannot enforce framework parameters: {sorted(unsupported)!r}")
            for role in ("activation", "weight"):
                tensor = recipe[role]
                tensor["observer"] = framework.get(
                    f"{role}_observer", _observer_name(tensor, is_weight=role == "weight")
                )
            recipe["framework_capture_policy"] = {
                "observer_epsilon": framework.get("observer_epsilon", DEFAULT_OBSERVER_EPSILON),
                "basis": "selected framework parameters or existing deterministic adapter policy; not RTL semantics",
            }
            if "calibration_count" in framework:
                # How many samples of the loader's calibration stream the static observers see.
                # Absent, the adapter's own default holds and the recipe's digest is unchanged.
                count = framework["calibration_count"]
                if type(count) is not int or count < 1:
                    raise ValueError("framework calibration_count must be a positive integer")
                recipe["framework_capture_policy"]["calibration_count"] = count
            recipe["recipe_sha256"] = digest(recipe)
            row = {
                "format_id": selected["id"],
                "unit": match["unit"],
                "recipe": recipe,
                "qualification": "diagnostic prospective transformation; no target admission",
            }
            methodology = quantization_contract.get("methodology")
            if preparation_profile is not None:
                row["recipe"] = select_recipe(
                    recipe,
                    preparation_profile,
                    preparation.get("output_path"),
                    model=model,
                    alternative_sha256=alternative_sha256,
                )
                row["preparation_profile"] = preparation_profile.identity()
            elif methodology is not None:
                # The target's declared methodology over the derived output path. A recipe it cannot derive
                # stays ``underivable`` with the reasons, never the hardware-only recipe it replaces.
                from merlin.targetgen import quant_methodology as QM

                declared, path = methodology["declaration"], methodology.get("output_path")
                row["recipe"] = QM.derive(recipe, declared, path)
                row["model_variants"] = [
                    {
                        "model": variant["model"],
                        "name": variant["name"],
                        "recipe": QM.derive(recipe, declared, path, model=variant["model"]),
                    }
                    for variant in declared.get("model_variants") or ()
                ]
            result.append(row)
    return result
