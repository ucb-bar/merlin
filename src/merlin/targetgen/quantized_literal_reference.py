"""Qualify a proved source literal under the selected rational numeric contract."""

import json


def qualify(directory, report, plan, contract):
    from merlin.targetgen.core_aten_numeric import _source_reference

    records = [r for r in report["cases"] if r.get("status") == "bundled"]
    request = dict(contract=contract, records=records, output_count=report["output_count"])
    (directory / "quantized-native-request.json").write_text(json.dumps(request) + "\n")
    process = _source_reference(directory, "--quantized-native")
    if process.returncode:
        raise ArithmeticError("independent quantized literal qualification failed: " + process.stderr[-3000:])
    reference = json.loads((directory / "quantized-native-reference.json").read_text())
    plan.update(
        quantized_contract=contract,
        precision_class="b",
        host_floor=dict(status="not_applicable", comparison="independent_quantized_exact_rational"),
        quantized_non_contraction_sites=plan["exact_dtype_sites"],
        integer_reference=dict(
            status="prepared",
            engine="independent_source_quantized_exact_rational",
            expected_output_bytes=reference["expected_output_bytes"],
        ),
    )
    for r in records:
        r["numeric_admission"] = plan
    (directory / "spec-int8-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan
