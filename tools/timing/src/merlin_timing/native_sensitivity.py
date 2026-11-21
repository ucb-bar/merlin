"""TRAIN-only native parameter sensitivity and missing-evidence diagnostics.

Native predictions are supplied by the original evaluator. This analysis neither
executes that evaluator nor certifies its source, physical prices or timing.
"""

from __future__ import annotations

import itertools
import math
import statistics

from . import contracts as C
from .fitting import quantile

SCHEMA = "merlin.native_sensitivity_input.v1"
MAX_EVALUATIONS = 128
MAX_SCALAR_WORK = 4_000_000


def _norm(values):
    value = math.hypot(*values)
    if not math.isfinite(value):
        raise C.ContractError("native response exceeds the finite diagnostic range")
    return value


def _cosine(left, right):
    a_scale, b_scale = _norm(left), _norm(right)
    return sum((a / a_scale) * (b / b_scale) for a, b in zip(left, right)) if a_scale and b_scale else None


def _correlation(left, right):
    a, b = statistics.mean(left), statistics.mean(right)
    return _cosine([x - a for x in left], [x - b for x in right])


def _rank(vectors):
    basis = []
    for vector in vectors:
        scale = _norm(vector)
        if not scale:
            continue
        residual = [value / scale for value in vector]
        # Reorthogonalize for stable rank diagnostics; this is not a fit.
        for _ in range(2):
            for column in basis:
                projection = sum(a * b for a, b in zip(residual, column))
                residual = [a - projection * b for a, b in zip(residual, column)]
        remaining = _norm(residual)
        if remaining > 1e-8:
            basis.append([value / remaining for value in residual])
    return len(basis)


def _metrics(errors):
    return {
        "cases": len(errors),
        "relative_rmse": _norm(errors) / math.sqrt(len(errors)),
        "median_absolute_relative_error": statistics.median(abs(x) for x in errors),
        "p95_absolute_relative_error": quantile([abs(x) for x in errors], 0.95),
    }


def analyze(dataset: dict, document: dict) -> dict:
    """Compare local native responses using only admitted public TRAIN labels.

    Every probe must keep the same original source cases and declared parameter
    roster. Missing predictions stay explicit; they cannot improve an error score.
    Secants at different parameter contexts are descriptive, not global physical
    identifiability or permission to extrapolate.
    """
    C.dataset(dataset)
    if dataset["profile"] != "phase1-public":
        raise C.ContractError("native sensitivity requires public Phase 1 evidence")
    C.closed(document, {"schema", "dataset_sha256", "evaluations"}, "native sensitivity input")
    if document["schema"] != SCHEMA or document["dataset_sha256"] != C.digest(dataset):
        raise C.ContractError("native sensitivity requires the exact original dataset")
    evaluations = document["evaluations"]
    if not isinstance(evaluations, list) or not 2 <= len(evaluations) <= MAX_EVALUATIONS:
        raise C.ContractError(f"declare 2..{MAX_EVALUATIONS} native evaluations")
    cases = {case["case_id"]: case for case in dataset["cases"] if case["split"] == "train"}
    records = C.grouped_records(dataset)
    points, unavailable = {}, []
    for key, case in cases.items():
        point, reason = C.case_point(case, records.get(key, []), dataset["feature_names"])
        if point is None:
            unavailable.append({"case_id": key, "reason": reason})
        else:
            points[key] = point
    if len(points) < 2 or len({cases[key]["source_group_sha256"] for key in points}) < 2:
        raise C.ContractError("native sensitivity needs independent available TRAIN cases")
    parameters, structure, seen, repeated_points = None, None, set(), {}
    scores = []
    for evaluation in evaluations:
        C.closed(evaluation, {"model_sha256", "fixed_model_sha256", "parameters", "predictions"}, "native evaluation")
        identity = C.sha(evaluation["model_sha256"], "native model identity")
        current_structure = C.sha(evaluation["fixed_model_sha256"], "fixed native structure")
        if identity in seen:
            raise C.ContractError("native model identity is duplicated")
        seen.add(identity)
        if structure is not None and structure != current_structure:
            raise C.ContractError("parameter probes cannot change fixed native structure")
        structure = current_structure
        knobs = evaluation["parameters"]
        if not isinstance(knobs, dict) or not 1 <= len(knobs) <= 64:
            raise C.ContractError("declare a bounded native parameter roster")
        for key, value in knobs.items():
            C.text(key, "parameter name")
            C.number(value, "parameter value")
        if parameters is not None and set(parameters) != set(knobs):
            raise C.ContractError("native probes change the parameter roster")
        parameters = knobs
        predictions = evaluation["predictions"]
        if not isinstance(predictions, dict) or set(predictions) != set(cases):
            raise C.ContractError("native probe must preserve every declared TRAIN case and exclude other splits")
        for value in predictions.values():
            if value is not None:
                C.number(value, "native prediction", positive=True)
        parameter_point = tuple(sorted(knobs.items()))
        if parameter_point in repeated_points and predictions != repeated_points[parameter_point]:
            raise C.ContractError("identical parameter probes disagree on original predictions")
        repeated_points[parameter_point] = predictions
        missing = sorted(key for key in points if predictions[key] is None)
        errors = {key: predictions[key] / point["cycles"] - 1 for key, point in points.items()} if not missing else {}
        by_family = {}
        for key, error in errors.items():
            by_family.setdefault(cases[key]["mechanism_family"], []).append(error)
        scores.append(
            {
                "model_sha256": identity,
                "predicted_available_train_cases": len(points) - len(missing),
                "missing_predictions": missing,
                "metrics": _metrics(list(errors.values())) if errors else None,
                "by_family": {name: _metrics(values) for name, values in sorted(by_family.items())},
            }
        )
    assert parameters is not None
    work = len(evaluations) ** 2 * len(points) * len(parameters)
    if work > MAX_SCALAR_WORK:
        raise C.ContractError("native sensitivity exceeds its declared scalar-work budget")
    keys = sorted(points)
    responses = {name: [] for name in parameters}
    for left, right in itertools.combinations(evaluations, 2):
        changed = [name for name in parameters if left["parameters"][name] != right["parameters"][name]]
        if len(changed) != 1 or any(side["predictions"][key] is None for side in (left, right) for key in keys):
            continue
        name = changed[0]
        a, b = left["parameters"][name], right["parameters"][name]
        scale = max(abs(a), abs(b), 1)
        vector = [
            (right["predictions"][key] - left["predictions"][key]) / points[key]["cycles"] * scale / (b - a)
            for key in keys
        ]
        context_distance = sum(
            left["parameters"][key] != evaluations[0]["parameters"][key] for key in parameters if key != name
        )
        responses[name].append((context_distance, -abs(b - a), left["model_sha256"], right["model_sha256"], vector))
    sensitivity, vectors = [], {}
    for name, alternatives in sorted(responses.items()):
        if not alternatives:
            sensitivity.append({"parameter": name, "status": "no_complete_one_parameter_contrast"})
            continue
        context, _, left, right, vector = min(alternatives, key=lambda row: row[:4])
        vectors[name] = vector
        responding = [key for key, value in zip(keys, vector) if abs(value) > 1e-12]
        sensitivity.append(
            {
                "parameter": name,
                "status": "responsive" if responding else "no_observed_response",
                "models": [left, right],
                "other_parameters_differing_from_seed": context,
                "complete_contrasts": len(alternatives),
                "rms_relative_response_per_scaled_unit": _norm(vector) / math.sqrt(len(vector)),
                "responding_cases": len(responding),
                "responding_source_groups": len({cases[key]["source_group_sha256"] for key in responding}),
            }
        )
    coupled = []
    for left, right in itertools.combinations(sorted(vectors), 2):
        cosine = _cosine(vectors[left], vectors[right])
        if cosine is not None and abs(cosine) >= 0.995:
            coupled.append({"parameters": [left, right], "response_cosine": cosine})
    baseline = evaluations[0]["predictions"]
    correlations = []
    if all(baseline[key] is not None for key in keys):
        residual = [(points[key]["cycles"] - baseline[key]) / points[key]["cycles"] for key in keys]
        for name in dataset["feature_names"]:
            correlation = _correlation([points[key]["features"][name] for key in keys], residual)
            if correlation is not None:
                correlations.append({"feature": name, "signed_residual_correlation": correlation})
    return {
        "schema": "merlin.native_sensitivity_report.v1",
        "dataset_sha256": C.digest(dataset),
        "fixed_model_sha256": structure,
        "declared_train_cases": len(cases),
        "available_train_cases": len(points),
        "available_train_source_groups": len({cases[key]["source_group_sha256"] for key in points}),
        "unavailable_train_cases": unavailable,
        "evaluations": scores,
        "parameter_sensitivity": sensitivity,
        "response_matrix_rank": _rank(list(vectors.values())),
        "response_matrix_columns": len(vectors),
        "scalar_work_bound": work,
        "maximum_scalar_work": MAX_SCALAR_WORK,
        "coupled_responses": coupled,
        "baseline_feature_residual_correlations": sorted(
            correlations, key=lambda row: -abs(row["signed_residual_correlation"])
        ),
        "held_or_calibration_labels_used": False,
        "physical_price_identification": "NOT_ESTABLISHED",
        "qualification_issued": False,
        "launch_authority": False,
        "limits": [
            "Caller-supplied native results require independent source/runtime custody.",
            "Local secants can cross different parameter contexts and nonlinear scheduling knees.",
            "Zero response may reflect masking, quantization or missing excitation, not a free operation.",
            "Residual correlations do not establish causality or missing resource semantics.",
            "No unseen workload, capacity or attention domain is qualified by these diagnostics.",
        ],
    }
