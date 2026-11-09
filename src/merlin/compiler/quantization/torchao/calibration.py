"""Recipe sample limits, observer reuse, GPTQ hooks and measured agreement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .portable import helper

_specs = helper("_recipe_specs")
RecipeError = _specs.RecipeError
_recipe_operators = _specs._recipe_operators


AGREEMENT_SAMPLE_LIMIT = 64


def _verify_dynamic_sites(graph_module: Any, rows: list, example_inputs: tuple, calibration: Any) -> None:
    """Check every rewritten site on the agreement samples: the capture input, then the calibration stream."""
    DC = helper("_dynamic_contractions")

    samples = [tuple(example_inputs), *(calibration or ())][:AGREEMENT_SAMPLE_LIMIT]
    DC.verify_sites(graph_module, rows, samples)


def _dynamic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals and sites of a dynamic rewrite, for the capture's stats."""
    done = [r for r in rows if r.get("status") == "rewritten"]
    return {
        "rewritten": len(done),
        "kept_in_float": [r for r in rows if r.get("status") != "rewritten"],
        **{key: sum(int(r.get(key, 0)) for r in done) for key in ("int8_macs", "device_calls", "host_elements")},
        "sites": done,
    }


def _weight_rounding(recipe: Mapping[str, Any], prepared: Any = None) -> Any:
    """How ``recipe`` rounds its weights: ``None`` for round-to-nearest, else GPTQ (``_gptq.py``).

    Without ``prepared`` this returns the recipe's validated GPTQ parameters. With a prepared PT2E
    module it returns the hook that accumulates Hessians during calibration (``track``) and sets
    GPTQ's codes into the stored weights before ``convert_pt2e`` folds them (``apply``). The
    algorithm and hook timing are documented in ``_gptq.py``. The portable loader
    binds that exact selected helper file, as it does ``quant_layer_plan``.
    """
    _gptq = helper("_gptq")

    try:
        parameters = _gptq.rounding_of(recipe["weight"])
    except _gptq.GPTQError as error:
        raise RecipeError(str(error)) from error
    if parameters is None or prepared is None:
        return parameters
    import torch

    return _gptq.PT2EWeightRounding(prepared, parameters, contractions=_recipe_operators(torch)["contraction"])


def _as_tuple(sample: Any) -> tuple:
    return tuple(sample) if isinstance(sample, (tuple, list)) else (sample,)


def calibration_limit(recipe: Mapping[str, Any], default: int) -> int:
    """How many calibration samples a static recipe observes: its policy's count, else ``default``.

    The count is part of the recipe (``framework_capture_policy.calibration_count``) because it
    changes the scales, so two captures of one recipe observe the same number of samples and a
    receipt keyed on the recipe digest names it. Which samples is the loader's calibration stream.
    """
    policy = recipe.get("framework_capture_policy") or {}
    value = policy.get("calibration_count", default)
    if type(value) is not int or value < 1:
        raise RecipeError("framework_capture_policy.calibration_count must be a positive integer")
    return value


def _RepeatedObservationSkip(prepared: Any):
    """The observer owner supplies guarded, exact min/max observation reuse."""
    RO = helper("_recipe_observers")  # noqa: PLC0415 -- capture interpreter sibling

    return RO.RepeatedObservationSkip(prepared)


def agreement(
    reference: Any, candidate: Any, samples: Iterable[Any], *, limit: int = AGREEMENT_SAMPLE_LIMIT
) -> dict[str, Any]:
    """How all quantized tensor outputs compare with the floating-point model on real samples.

    A contract change is never an optimisation: moving a weight scale from per-channel to
    per-tensor changes the numbers, and this is the receipt that says by how much. Reported per
    stream, not judged here; the consumer holds it to the workload's own bar.
    """
    RA = helper("_recipe_agreement")

    try:
        return RA.compare(reference, candidate, samples, limit=limit, as_tuple=_as_tuple)
    except RA.AgreementError as error:
        raise RecipeError(str(error)) from error
