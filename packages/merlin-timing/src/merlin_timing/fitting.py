"""Interpretable nonnegative empirical screening fits using only the standard library."""

from __future__ import annotations

import math

from . import contracts as C


def quantile(values: list[float], probability: float) -> float:
    """Conservative nearest-rank empirical quantile; no distribution is assumed."""
    if not values:
        raise C.ContractError("a residual quantile needs observations")
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))]


def _condition(gram: list[list[float]], maximum: float) -> float:
    """Infinity-norm Gram condition bounds scaled design conditioning conservatively."""
    size = len(gram)
    matrix = [row[:] + [float(i == j) for j in range(size)] for i, row in enumerate(gram)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            raise C.ContractError("features/intercept are rank deficient")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        value = matrix[column][column]
        matrix[column] = [x / value for x in matrix[column]]
        for row in range(size):
            if row != column:
                factor = matrix[row][column]
                matrix[row] = [x - factor * y for x, y in zip(matrix[row], matrix[column])]
    norm = max(sum(abs(x) for x in row) for row in gram)
    inverse = max(sum(abs(x) for x in row[size:]) for row in matrix)
    condition = math.sqrt(norm * inverse)
    if not math.isfinite(condition) or condition > maximum:
        raise C.ContractError("scaled features exceed the declared maximum condition")
    return condition


def design_check(matrix: list[list[float]], *, maximum_condition: float) -> dict:
    """Apply the fit's own identifiability gates to a design that has no labels yet.

    The gates are exactly those :func:`nnls` enforces after :func:`fit` scales each column by its
    largest value: two distinct points per parameter, and a full-rank scaled Gram whose condition
    bound stays under ``maximum_condition``. Planning an experiment with them means a design that
    passes here cannot later fail the fit for lack of identifiability; it says nothing about labels.
    """
    C.number(maximum_condition, "maximum condition", positive=True)
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise C.ContractError("design check requires a nonempty rectangular feature matrix")
    width, count = len(matrix[0]), len(matrix)
    scales = [max(1.0, max(abs(float(row[i])) for row in matrix)) for i in range(width)]
    scaled = [[float(row[i]) / scales[i] for i in range(width)] for row in matrix]
    if len({tuple(row) for row in scaled}) < 2 * width:
        return {"identifiable": False, "reason": "fit requires two distinct points per parameter", "rows": count}
    gram = [[sum(row[i] * row[j] for row in scaled) / count for j in range(width)] for i in range(width)]
    try:
        condition = _condition(gram, maximum_condition)
    except C.ContractError as exc:
        return {"identifiable": False, "reason": str(exc), "rows": count}
    return {"identifiable": True, "reason": None, "rows": count, "scaled_design_condition_upper_bound": condition}


def nnls(
    matrix: list[list[float]], labels: list[float], *, maximum_condition: float, max_iterations: int = 20000
) -> tuple[list[float], dict]:
    """Projected coordinate NNLS with an explicit conditioning and KKT residual gate."""
    C.number(maximum_condition, "maximum condition", positive=True)
    if maximum_condition < 1:
        raise C.ContractError("maximum condition must be at least one")
    width, count = len(matrix[0]), len(matrix)
    if len({tuple(row) for row in matrix}) < 2 * width:
        raise C.ContractError("fit requires two distinct points per parameter")
    gram = [[sum(row[i] * row[j] for row in matrix) / count for j in range(width)] for i in range(width)]
    rhs = [sum(row[i] * label for row, label in zip(matrix, labels)) / count for i in range(width)]
    condition = _condition(gram, maximum_condition)
    coefficients = [0.0] * width
    tolerance = 1e-10
    for iteration in range(max_iterations):
        for i in range(width):
            gradient = sum(gram[i][j] * coefficients[j] for j in range(width)) - rhs[i]
            coefficients[i] = max(0.0, coefficients[i] - gradient / gram[i][i])
        gradients = [sum(gram[i][j] * coefficients[j] for j in range(width)) - rhs[i] for i in range(width)]
        kkt = max(abs(g) if c > 0 else max(0.0, -g) for c, g in zip(coefficients, gradients))
        if kkt <= tolerance * max(1.0, max(abs(x) for x in rhs)):
            return coefficients, {
                "algorithm": "scaled_projected_coordinate_nnls",
                "iterations": iteration + 1,
                "scaled_design_condition_upper_bound": condition,
                "maximum_condition": maximum_condition,
                "kkt_residual": kkt,
            }
    raise C.ContractError("projected NNLS failed its KKT convergence gate")


def predict(model: dict, values: dict, *, domain_sha256: str) -> tuple[float | None, str | None]:
    if domain_sha256 != model["domain_sha256"]:
        return None, "device, engine, feature schema, scope or memory regime differs from fit"
    C.features(values, model["feature_names"])
    for name in model["feature_names"]:
        value = values[name]
        lo, hi = model["training_ranges"][name]
        if value is None or not lo <= value <= hi:
            return None, f"feature {name} is unknown or outside the training domain"
    estimate = model["fixed_cycles"] + sum(
        values[name] * model["coefficients"][name] for name in model["feature_names"]
    )
    if not math.isfinite(estimate) or estimate <= 0:
        return None, "screen has no positive finite prediction"
    return estimate, None


def residual_interval(points: list[tuple[str, str, float, float]]) -> dict:
    groups = sorted({group for _, group, _, _ in points})
    if len(points) < 2 or len(groups) < 2:
        return {
            "status": "unknown",
            "reason": "two independent calibration source groups are required",
            "n_cases": len(points),
            "source_group_sha256s": groups,
            "relative_residual_quantiles": None,
        }
    residuals = [(actual - predicted) / predicted for _, _, predicted, actual in points]
    return {
        "status": "empirical",
        "reason": "calibration residuals; not a physical or guaranteed interval",
        "n_cases": len(points),
        "source_group_sha256s": groups,
        "relative_residual_quantiles": [quantile(residuals, 0.05), quantile(residuals, 0.95)],
    }


def fit(
    document: dict,
    *,
    include_fixed: bool = True,
    maximum_condition: float = 1e6,
    training_intake: str = "all_declared",
    loss: str = "absolute_squared",
) -> dict:
    C.dataset(document)
    if training_intake not in {"all_declared", "eligible_records_v1"}:
        raise C.ContractError("select all_declared or eligible_records_v1 training intake")
    if loss not in {"absolute_squared", "relative_squared"}:
        raise C.ContractError("select absolute_squared or relative_squared fitting loss")
    names = document["feature_names"]
    rows = C.grouped_records(document)
    points, declared, unavailable = [], [], []
    for case in document["cases"]:
        if case["split"] != "train":
            continue
        declared.append(case["case_id"])
        point, reason = C.case_point(case, rows.get(case["case_id"], []), names)
        if point is None:
            unavailable.append({"case_id": case["case_id"], "reason": reason})
        else:
            points.append(point)
    if not points or (unavailable and training_intake == "all_declared"):
        raise C.ContractError("every declared training case requires complete original passing records")
    if len({point["case"]["source_group_sha256"] for point in points}) < 2:
        raise C.ContractError("fit requires independent training source groups")
    scales = [max(1.0, max(point["features"][name] for point in points)) for name in names]
    label_scale = max(point["cycles"] for point in points)
    matrix = [[float(point["features"][name]) / scale for name, scale in zip(names, scales)] for point in points]
    if include_fixed:
        matrix = [[1.0] + row for row in matrix]
    labels = [point["cycles"] / label_scale for point in points]
    if loss == "relative_squared":
        # Equal relative error across latency scales; only training labels
        # choose these weights. Held membership and validation gates stay fixed.
        matrix = [[value / label for value in row] for row, label in zip(matrix, labels)]
        labels = [1.0] * len(labels)
    coefficients, solver = nnls(matrix, labels, maximum_condition=maximum_condition)
    solver["include_fixed"] = include_fixed
    fixed = coefficients[0] * label_scale if include_fixed else 0.0
    slopes = coefficients[1:] if include_fixed else coefficients
    prices = {name: coefficient * label_scale / scale for name, coefficient, scale in zip(names, slopes, scales)}
    model = {
        "schema": C.MODEL_V3
        if loss == "relative_squared"
        else C.MODEL_V2
        if training_intake == "eligible_records_v1"
        else C.MODEL,
        "kind": "empirical_screen",
        "profile": document["profile"],
        "domain": document["domain"],
        "domain_sha256": C.digest(document["domain"]),
        "feature_names": names,
        "coefficients": prices,
        "fixed_cycles": fixed,
        "parameters": {**prices, "fixed_cycles": fixed},
        "training_ranges": {
            name: [min(point["features"][name] for point in points), max(point["features"][name] for point in points)]
            for name in names
        },
        "training_case_ids": [point["case"]["case_id"] for point in points],
        "training_group_sha256s": sorted({point["case"]["source_group_sha256"] for point in points}),
        "calibration_case_ids": [],
        "dataset_sha256": C.digest(document),
        "solver": solver,
        "residual_interval": {},
        "status": "diagnostic",
        "engine_qualification": "not_established",
    }
    if training_intake == "eligible_records_v1" or loss == "relative_squared":
        model["training_intake"] = {
            "policy": training_intake,
            "declared_training_case_ids": declared,
            "unavailable_training_cases": unavailable,
            "evaluation_denominator": "all original declared held cases; unchanged dataset and source splits",
        }
    if loss == "relative_squared":
        model["fit_loss"] = loss
    residuals = []
    for case in document["cases"]:
        if case["split"] != "calibration":
            continue
        point, _ = C.case_point(case, rows.get(case["case_id"], []), names)
        if point is not None:
            estimate, _ = predict(model, point["features"], domain_sha256=model["domain_sha256"])
            if estimate is not None:
                residuals.append((case["case_id"], case["source_group_sha256"], estimate, point["cycles"]))
    model["calibration_case_ids"] = [case_id for case_id, _, _, _ in residuals]
    model["residual_interval"] = residual_interval(residuals)
    return model


def model(document: dict) -> dict:
    v3 = document.get("schema") == C.MODEL_V3
    v2 = document.get("schema") in {C.MODEL_V2, C.MODEL_V3}
    C.closed(
        document,
        {
            "schema",
            "kind",
            "profile",
            "domain",
            "domain_sha256",
            "feature_names",
            "coefficients",
            "fixed_cycles",
            "parameters",
            "training_ranges",
            "training_case_ids",
            "training_group_sha256s",
            "calibration_case_ids",
            "dataset_sha256",
            "solver",
            "residual_interval",
            "status",
            "engine_qualification",
        }
        | ({"training_intake"} if v2 else set())
        | ({"fit_loss"} if v3 else set()),
        "screen model",
    )
    if (
        document["schema"] not in {C.MODEL, C.MODEL_V2, C.MODEL_V3}
        or document["kind"] != "empirical_screen"
        or document["profile"] not in C.PROFILES
        or document["status"] != "diagnostic"
        or document["engine_qualification"] != "not_established"
    ):
        raise C.ContractError("unsupported screen model or unearned engine qualification")
    C.domain(document["domain"])
    if C.digest(document["domain"]) != document["domain_sha256"]:
        raise C.ContractError("model domain digest differs from its facts")
    C.sha(document["dataset_sha256"], "fitted dataset")
    if v3 and document["fit_loss"] != "relative_squared":
        raise C.ContractError("unsupported versioned screen fitting loss")
    if v2:
        intake = C.closed(
            document["training_intake"],
            {"policy", "declared_training_case_ids", "unavailable_training_cases", "evaluation_denominator"},
            "training intake",
        )
        if (
            intake["policy"] not in ({"all_declared", "eligible_records_v1"} if v3 else {"eligible_records_v1"})
            or intake["evaluation_denominator"]
            != "all original declared held cases; unchanged dataset and source splits"
        ):
            raise C.ContractError("unsupported eligible-record training policy or changed evaluation denominator")
        declared, unavailable = intake["declared_training_case_ids"], intake["unavailable_training_cases"]
        if not isinstance(declared, list) or not isinstance(unavailable, list) or len(declared) > C.MAX_CASES:
            raise C.ContractError("training intake exceeds the declared case budget")
        for case_id in declared:
            C.text(case_id, "declared training case")
        unavailable_ids = []
        for row in unavailable:
            C.closed(row, {"case_id", "reason"}, "unavailable training case")
            C.text(row["case_id"], "unavailable training case")
            C.text(row["reason"], "training unavailability reason")
            unavailable_ids.append(row["case_id"])
        used = document["training_case_ids"]
        if (
            len(set(declared)) != len(declared)
            or len(set(unavailable_ids)) != len(unavailable_ids)
            or set(used) & set(unavailable_ids)
            or set(declared) != set(used) | set(unavailable_ids)
        ):
            raise C.ContractError("training intake does not preserve the complete declared training roster")
    names = document["feature_names"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or len(names) > C.MAX_FEATURES:
        raise C.ContractError("model requires distinct feature names")
    C.closed(document["coefficients"], set(names), "coefficients")
    C.closed(document["training_ranges"], set(names), "training ranges")
    for name in names:
        C.number(document["coefficients"][name], f"price {name}")
        bounds = document["training_ranges"][name]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not 0 <= C.number(bounds[0], "minimum") <= C.number(bounds[1], "maximum")
        ):
            raise C.ContractError("training ranges must be ordered nonnegative bounds")
    C.number(document["fixed_cycles"], "fixed cycles")
    if document["parameters"] != {**document["coefficients"], "fixed_cycles": document["fixed_cycles"]}:
        raise C.ContractError("diagnostic parameter seeds differ from screen prices")
    C.closed(
        document["solver"],
        {
            "algorithm",
            "iterations",
            "scaled_design_condition_upper_bound",
            "maximum_condition",
            "kkt_residual",
            "include_fixed",
        },
        "solver",
    )
    if type(document["solver"]["include_fixed"]) is not bool:
        raise C.ContractError("intercept selection must be explicit")
    if document["solver"]["algorithm"] != "scaled_projected_coordinate_nnls":
        raise C.ContractError("unsupported solver")
    for key in ("iterations", "scaled_design_condition_upper_bound", "maximum_condition", "kkt_residual"):
        C.number(document["solver"][key], f"solver {key}")
    C.closed(
        document["residual_interval"],
        {"status", "reason", "n_cases", "source_group_sha256s", "relative_residual_quantiles"},
        "residual interval",
    )
    residuals = document["residual_interval"]
    if residuals["status"] not in {"unknown", "empirical"}:
        raise C.ContractError("unknown residual interval status")
    bounds = residuals["relative_residual_quantiles"]
    if bounds is not None:
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds)
            or bounds[0] > bounds[1]
        ):
            raise C.ContractError("residual quantiles require ordered finite bounds")
    if (bounds is None) != (residuals["status"] == "unknown"):
        raise C.ContractError("residual interval status disagrees with its evidence")
    return document
