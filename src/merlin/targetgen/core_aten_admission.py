"""Transactional native admission from general proofs and construction policy."""

import copy
import json


def prepare(directory, report, implementation, **kwargs):
    from merlin.common.yaml import load_yaml
    from merlin.frontends.linalg_mlir import parse_mlir_text

    selected = kwargs.get("software_spec")
    software = load_yaml(selected) if selected is not None else {}
    native_contract = software.get("numerical_semantics", {}).get("native_unit_contract", {})
    excluded = native_contract.get("excluded_constructions")
    if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
        raise ValueError("native placement requires explicit excluded_constructions")
    from merlin.targetgen.native_unit_policy import construction_evidence

    stages = []

    def observe(module, stage):
        result_abi = {}
        for record in report.get("cases", ()):
            if record.get("status") not in {None, "bundled"}:
                continue
            indices, abis = record.get("output_indices", ()), record.get("output_abi", ())
            if len(indices) != len(abis):
                continue  # No logical-word proof is available without a complete ABI.
            result_abi.update(zip(indices, abis, strict=True))
        decision = construction_evidence(module, excluded=excluded, result_abi=result_abi)
        stages.append(dict(stage=stage, **decision))

    # Preparation can change the whole exported ABI and its inputs. A rejected
    # placement must restore the original host program and all of its sidecars.
    original_files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    original_report = copy.deepcopy(report)
    original_cases = list(report["cases"])
    observe(parse_mlir_text((directory / "model.mlir").read_text()), "source")
    plan = implementation(directory, report, construction_observer=observe, **kwargs)
    observe(parse_mlir_text((directory / "model.mlir").read_text()), "prepared")
    native = dict(
        schema="merlin.native_unit_policy.v1",
        status="host" if any(s["status"] == "host" for s in stages) else "eligible",
        stages=stages,
        evidence=[dict(stage=s["stage"], **e) for s in stages for e in s["evidence"]],
    )
    # One numeric candidate belongs only to bundled records. Some preparation
    # paths annotate all retained corpus rows, including not-in-shard metadata;
    # restore those rows before publishing this candidate's policy evidence.
    for record, saved in zip(report["cases"], original_report["cases"], strict=True):
        if saved.get("status") not in {None, "bundled"}:
            record.clear()
            record.update(saved)
    if not any(
        plan.get(key, 0)
        for key in (
            "exact_dtype_sites",
            "native_contractions",
            "quantized_contractions",
            "quantized_non_contraction_sites",
        )
    ):
        plan["native_admission"] = dict(
            status="host",
            reason_code="non_native_unit_construction"
            if native["status"] == "host"
            else plan.get("refusal_code", "no_admitted_device_candidate"),
            native_unit_policy=native,
        )
        for record in report["cases"]:
            if record.get("status") in {None, "bundled"}:
                record["numeric_admission"] = plan
        (directory / "spec-int8-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
        return plan
    from merlin.targetgen.operator_review import review_record

    admission = dict(
        status="device",
        reason_code="native_family_proved",
        native_unit_operator_review=review_record(native_contract),
        native_unit_policy=native,
    )
    # The first refusal names the reason; later native checks do not overwrite it.
    if native["status"] == "host":
        admission.update(status="host", reason_code="non_native_unit_construction")
    elif not admission["native_unit_operator_review"]["operator_accepted"]:
        admission.update(status="host", reason_code="native_unit_operator_acceptance_required")
    from merlin.targetgen.native_precision_contract import proves_quantized_model, requires_quantized_model

    if requires_quantized_model(parse_mlir_text(original_files["model.mlir"].decode())):
        admission["precision_class"] = "quantized_exact"
        if admission["status"] == "device" and not proves_quantized_model(plan, software):
            admission.update(status="host", reason_code="declared_quantized_precision_requires_quantized_model")
    if admission["status"] == "device" and plan.get("capture_recipe"):
        from merlin.targetgen.native_integer_work import uncovered

        missing = uncovered(parse_mlir_text((directory / "model.mlir").read_text()), plan.get("schedule_proof"))
        if missing:
            admission.update(
                status="host", reason_code="native_integer_family_work_uncovered", uncovered_source_ordinals=missing
            )
    # Static descriptor queries are exported through a tensor wrapper, but
    # carry no tensor data. A capture-owned proof distinguishes them from real
    # literal fills and rank-zero tensor arithmetic without an overload roster.
    from merlin.targetgen.core_aten_capture import descriptor_specialization_matches

    folds = []
    for record in original_report["cases"]:
        if record.get("status") not in {None, "bundled"} or not record.get("capture_directory"):
            continue
        from pathlib import Path

        receipt_path = Path(record["capture_directory"]) / "capture.json"
        receipt = json.loads(receipt_path.read_text())
        specialization = receipt.get("capture_meta", {}).get("static_metadata_specialization")
        if specialization and descriptor_specialization_matches(record["case"], receipt):
            folds.append(specialization)
    if folds:
        admission.update(status="host", reason_code="constant_metadata_fold_no_tensor_data", descriptor_folds=folds)
    # F1 kernels have private results. A checked storage contract copies their
    # exact bytes into the borrowed result view before returning its descriptor.
    aliases = [
        record["case"]["output_input_aliases"]
        for record in original_report["cases"]
        if record.get("status") in {None, "bundled"} and (record.get("case") or {}).get("output_input_aliases")
    ]
    from merlin.targetgen.materialized_movement import alias_materialization_valid

    if aliases and not alias_materialization_valid(directory, report, plan):
        admission.update(
            status="host",
            reason_code="native_output_alias_contract_unproved",
            required_output_input_aliases=aliases,
        )
    plan["native_admission"] = admission
    if admission["status"] != "device":
        (directory / "refused-device-source.mlir").write_bytes((directory / "model.mlir").read_bytes())
        (directory / "refused-device-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
        for name, data in original_files.items():
            (directory / name).write_bytes(data)
        # The runner retains references to these records across preparation.
        # Preserve their identities so later routing/build updates reach the
        # restored report rather than abandoned candidate dictionaries.
        for record, saved in zip(original_cases, original_report["cases"], strict=True):
            record.clear()
            record.update(saved)
        original_report["cases"] = original_cases
        report.clear()
        report.update(original_report)
        plan = dict(
            selection=plan["selection"],
            status="host",
            sites=[],
            quantized_contractions=0,
            native_contractions=0,
            quantized_non_contraction_sites=0,
            exact_dtype_sites=0,
            refusal_code=admission["reason_code"],
            host_reason=admission["reason_code"],
            native_admission=admission,
        )
    for record in report["cases"]:
        if record.get("status") in {None, "bundled"}:
            record.update(numeric_admission=plan, numeric_selection=plan["selection"]["identities"])
    (directory / "spec-int8-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan
