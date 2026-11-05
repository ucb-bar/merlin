"""Pure completeness predicate for selected full-chain quantized-reference reports."""

def report_complete(report: dict, model, policy: dict, package: dict) -> bool:
    """Require real selected gates and every declared step, not a CLI pass string."""
    gates = report.get("gates") or {}
    if report.get("status") != "pass" or report.get("failures") or report.get("model") != model.name:
        return False
    if (
        report.get("command") != "check"
        or report.get("split") != policy["split"]
        or report.get("agent_view") is not False
    ):
        return False
    if report.get("submission", {}).get("tree_sha256") != package["model_check_tree_sha256"]:
        return False
    comparison = report.get("comparison") or {}
    if comparison.get("selected") != "quantized-reference" or comparison.get("float_policy") != {
        "atol": policy["atol"],
        "rtol": policy["rtol"],
        "equal_nan": False,
        "finite_required": True,
    }:
        return False
    for name in ("build", "kernel_exact", "quantized_reference"):
        if gates.get(name, {}).get("status") != "pass" or gates[name].get("partial", False) is not False:
            return False
    expected = sum(program.steps for program in model.programs)
    gate = gates["quantized_reference"]
    if (
        gate.get("inputs_completed"),
        gate.get("inputs_expected"),
        gate.get("steps_completed"),
        gate.get("steps_expected"),
    ) != (1, 1, expected, expected):
        return False
    bindings = [
        {"program": ref["program"], "step": step, "output": int(ref["output"])}
        for ref in model.judged["outputs"]
        for step in range(model.program(ref["program"]).steps)
    ]
    if (
        gate.get("steps_declared") != expected
        or gate.get("outputs_compared") != len(bindings)
        or gate.get("outputs_expected") != len(bindings)
    ):
        return False
    metrics = report.get("model_metrics") or {}
    per_input = metrics.get("per_input") or []
    if metrics.get("source") != "measured" or metrics.get("inputs") != 1 or len(per_input) != 1:
        return False
    row = per_input[0]
    validation = row.get("reference_validation") or {}
    if (
        row.get("id") != policy["input"]
        or row.get("split") != policy["split"]
        or row.get("pass") is not True
        or "reference_consistent" not in row
        or row["reference_consistent"] is not None
    ):
        return False
    if (
        validation.get("source") != "sealed_int8"
        or validation.get("strict_arity") is not True
        or validation.get("unchanged") is not True
    ):
        return False
    compared = row.get("quantized_reference") or []
    outputs = validation.get("outputs") or []
    if len(compared) != len(bindings) or len(outputs) != len(bindings):
        return False
    for binding, measured, reference in zip(bindings, compared, outputs, strict=True):
        if any(measured.get(key) != value or reference.get(key) != value for key, value in binding.items()):
            return False
        if measured.get("pass") is not True or measured.get("mismatched") != 0:
            return False
        if measured.get("dtype") != reference.get("dtype") or measured.get("shape") != reference.get("shape"):
            return False
    if gate.get("float_policy") != comparison["float_policy"]:
        return False
    programs = {row.get("program"): row for row in report.get("programs", [])}
    if len(programs) != len(report.get("programs", [])) or set(programs) != {
        program.name for program in model.programs
    }:
        return False
    for program in model.programs:
        rows = programs[program.name].get("runs", [])
        if len(rows) != program.steps or {row.get("step") for row in rows} != set(range(program.steps)):
            return False
        if any(
            row.get("status") != "completed" or row.get("engine") != "spike" or row.get("input") != policy["input"]
            for row in rows
        ):
            return False
    return True
