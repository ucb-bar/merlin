"""Opt-in Core ATen numeric policy preparation and separate approximation metrics."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def _source_reference(directory, *arguments):
    """Run the source interpreter in the declared capture environment.

    Source evaluation needs PyTorch and model2MLIR's xDSL interpreter, while the
    compiler environment deliberately declares only its lowering dependencies.
    Merlin's source import roots are explicit for non-editable installations too.
    """
    from merlin.common.paths import python_import_roots
    from merlin.integrations.model2mlir import capture_python
    from merlin.targetgen import core_aten_w8a8_reference

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in python_import_roots())
    return subprocess.run(
        [str(capture_python()), "-m", core_aten_w8a8_reference.__name__, str(directory.resolve()), *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def map_reference_boundary(fresh, reference, output_count):
    """Project independent user outputs and observed post-call inputs onto exported roles."""
    outputs = [None] * output_count
    for indices, values in (
        (fresh["output_indices"], reference["outputs"]),
        (fresh["semantic_boundary"]["post_indices"], reference["post_inputs"]),
    ):
        for index, value in zip(indices, values, strict=True):
            if outputs[index] is not None and not np.array_equal(outputs[index], value, equal_nan=True):
                raise ValueError("aliased user/post reference observations disagree")
            outputs[index] = value
    if any(value is None for value in outputs):
        raise ValueError("independent reference does not cover the complete exported boundary")
    return outputs


def map_reference_byte_boundary(fresh, reference, output_count):
    """Cover every result and post-input dtype with exact transport bytes."""
    expected = {}
    for indices, values in (
        (fresh["output_indices"], reference["output_bytes"]),
        (fresh["semantic_boundary"]["post_indices"], reference["post_input_bytes"]),
    ):
        for index, value in zip(indices, values, strict=True):
            key = str(index)
            if key in expected and expected[key] != value:
                raise ValueError("aliased exact reference byte observations disagree")
            expected[key] = value
    if set(expected) != {str(i) for i in range(output_count)}:
        raise ValueError("exact reference bytes do not cover every exported result")
    return expected


def restore_caller_layouts(module, original, fresh, semantic_path):
    """Retain source caller storage for inputs and their actual exported post-call observations."""
    from xdsl.dialects.builtin import StringAttr

    from merlin.llvmlower.result_layout import ATTRIBUTE
    from merlin.targetgen.core_aten_batch import _tensor_documents

    layout = json.loads(module.attributes[ATTRIBUTE].data)
    arguments = _tensor_documents(original["case"]["arguments"])
    if len(arguments) != len(layout["inputs"]):
        raise ValueError("caller layout count differs from recaptured inputs")
    boundary = json.loads(semantic_path.read_text())
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    returned = next(op for op in function.body.block.ops if op.name == "func.return")
    for ordinal, document in enumerate(arguments):
        caller = {
            key: copy.deepcopy(document[key]) for key in ("dtype", "shape", "stride", "storage_offset", "requires_grad")
        }
        if any(caller[key] != layout["inputs"][ordinal][key] for key in ("dtype", "shape")):
            raise ValueError("caller tensor type differs from recaptured input")
        layout["inputs"][ordinal] = caller
        index = fresh["semantic_boundary"]["post_indices"][ordinal]
        entry = boundary["outputs"][index]
        mutation = entry.get("role") == "user_input_mutation"
        passthrough = returned.operands[index] is function.body.block.args[ordinal]
        if mutation or passthrough:
            if entry["shape"] != caller["shape"] or entry["dtype"] != caller["dtype"]:
                raise ValueError("recaptured mutation changes the caller tensor type")
            entry.update(caller)
            # Read-only observations borrow caller storage, including expanded views.
            # Mutation write-back remains subject to the lowerer's overlap refusal.
            layout["results"][index] = {**(layout["results"][index] or {}), **caller, "alias_inputs": [ordinal]}
            if mutation:
                layout["results"][index].update(role="user_input_mutation", input_index=ordinal)
    expected = _tensor_documents(original["case"]["expected"]) if "expected" in original["case"] else []
    for document, index in zip(expected, fresh["output_indices"] if expected else [], strict=True):
        # Quantization may choose a different dense layout (for example its DQ
        # loses channels-last storage). Restore the caller's complete owned result
        # contract through result_layout, which allocates and copies to those
        # real strides after bufferization. Aliases above retain their own proof.
        record, entry = layout["results"][index], boundary["outputs"][index]
        if record is None or record.get("alias_inputs") or entry.get("alias_inputs"):
            continue
        if (
            any(record.get(key) != document.get(key) for key in ("dtype", "shape"))
            or len(record.get("stride") or ()) != len(document["shape"])
            or any(type(stride) is not int or stride < 0 for stride in document["stride"])
        ):
            continue
        record["stride"] = list(document["stride"])
        entry["stride"] = list(document["stride"])
        record["storage_offset"] = entry["storage_offset"] = document["storage_offset"]
    module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout, sort_keys=True))
    semantic_path.write_text(json.dumps(boundary, sort_keys=True) + "\n")


def expose_results(module, values):
    """Append values to forward's results, with unconstrained layouts; return the prior result count."""
    from xdsl.dialects.builtin import FunctionType, StringAttr

    from merlin.llvmlower.result_layout import ATTRIBUTE

    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    returned = next(op for op in function.body.block.ops if op.name == "func.return")
    count = len(returned.operands)
    returned.operands = [*returned.operands, *values]
    function.properties["function_type"] = FunctionType.from_lists(
        [arg.type for arg in function.body.block.args], [value.type for value in returned.operands]
    )
    if ATTRIBUTE in module.attributes:
        layout = json.loads(module.attributes[ATTRIBUTE].data)
        layout["results"].extend([None] * len(values))
        module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout, sort_keys=True))
    return count


def bind_device_contraction(op, declarations, region_id):
    """Bind a device contraction's declarations, region and transfer contracts, in this attribute order."""
    from xdsl.dialects.builtin import StringAttr

    op.attributes["prov.declarations"] = StringAttr(declarations)
    op.attributes["prov.region_id"] = StringAttr(region_id)
    for contract in ("operand_load", "accumulator_readout"):
        op.attributes[f"prov.{contract}"] = StringAttr(f"transfer_contracts.{contract}")


def scalarized_contraction_candidates(module):
    """Find frontend contraction families expressed as ordered scalar loops.

    This requests a recapture, never device admission. The captured integer
    contractions still need the ordinary declaration, source-reference and
    provider schedule proofs before they can replace the host program.
    """
    candidates = []
    for ordinal, op in enumerate(module.walk()):
        if op.name != "linalg.generic" or getattr(op.attributes.get("prov.family"), "data", None) != "contraction":
            continue
        children = list(op.body.walk())
        if any(child.name == "scf.for" for child in children) and any(
            child.name == "arith.mulf"
            and all(getattr(value.owner, "name", None) != "arith.constant" for value in child.operands)
            for child in children
        ):
            candidates.append(ordinal)
    return candidates


def prepare_spec_int8(
    directory: Path,
    report: dict,
    **kwargs,
):
    """Admit a device candidate, or keep the original host program byte for byte.

    Source specializations and closed normalizations rewrite model.mlir and
    its sidecars while a device candidate is formed. A case that does not end
    on the device runs its original captured host program: every file present
    before preparation is restored. New evidence files (admission and refusal
    receipts) are kept.
    """
    snapshot = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    (directory / "model.source.mlir").write_bytes(snapshot["model.mlir"])
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.targetgen.floating_contract import requires_quantized_model

    source_module = parse_mlir_file(directory / "model.mlir")
    if requires_quantized_model(source_module)["quantized"]:
        from merlin.targetgen.core_aten_spec_int8 import select_spec

        selection = select_spec(kwargs["target"], kwargs["facts"], kwargs["software_spec"], kwargs["host_capabilities"])
        numeric = selection["spec"].get("numerical_semantics") or {}
        storage = (numeric.get("quantized_native_fallback") or {}).get("operand_storage", [])
        unsupported = sorted(
            {
                abi["dtype"]
                for record in report["cases"]
                if record.get("status") == "bundled"
                for abi in record.get("input_abi", [])
                if abi["dtype"].startswith("complex") and abi["dtype"] not in storage
            }
        )
        if (
            numeric.get("component_domain", {}).get("floating_storage_arithmetic") == "declared_quantized_model"
            and unsupported
        ):
            from merlin.targetgen.core_aten_exact_dtype import _host_plan

            return _host_plan(
                directory,
                report["cases"],
                selection,
                "quantized_operand_storage_unproved",
                "declared quantized arithmetic has no validated operand storage: " + ", ".join(unsupported),
            )
    result = _prepare_spec_int8_candidates(directory, report, **kwargs)
    if (result or {}).get("native_admission", {}).get("status") != "device":
        for name, data in snapshot.items():
            (directory / name).write_bytes(data)
    if report.get("native_certificate") == "unavailable":
        result["native_certificate"] = "unavailable"
        result["native_certificate_refusal"] = report["native_certificate_refusal"]
        admission = result.get("native_admission", {})
        # A native component refusal is decisive only when it supplies the
        # final host reason. Construction/ABI policy can independently require
        # host execution, and an exact quantized candidate needs no IEEE proof.
        if (
            admission.get("status") != "device"
            and admission.get("reason_code") == "native_dtype_exact_components_unavailable"
        ):
            from merlin.targetgen.native_cpu_errors import require_certificate_available

            require_certificate_available(result.get("refusal_details"))
        (directory / "spec-int8-admission.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _prepare_spec_int8_candidates(
    directory: Path,
    report: dict,
    **kwargs,
):
    from merlin.targetgen.core_aten_admission import prepare
    from merlin.targetgen.source_power_specialization import specialize

    specialize(directory, report)
    from merlin.targetgen.source_mask_endpoints import specialize as specialize_mask

    specialize_mask(directory, report)
    result = prepare(directory, report, _prepare_spec_int8, **kwargs)
    if result.get("native_admission", {}).get("status") != "device":
        from merlin.targetgen.source_integer_remainder import specialize as specialize_remainder

        remainder = specialize_remainder(directory, report, kwargs.get("software_spec"))
        if remainder is not None:
            candidate = prepare(directory, report, _prepare_spec_int8, **kwargs)
            if candidate.get("native_admission", {}).get("status") == "device":
                from merlin.targetgen.quantized_literal_reference import qualify

                return qualify(directory, report, candidate, remainder[0]["quantized_contract"])
            (directory / "model.mlir").write_bytes(remainder[1])
    if result.get("native_admission", {}).get("status") != "device" and result.get("refusal_code") not in {
        "empty_tensor_no_device_work",
        "native_output_alias_contract_unproved",
        "input_metadata_mutation_requires_host",
        "constant_metadata_fold_no_tensor_data",
    }:
        from merlin.targetgen.quantized_native import prepare as quantized_fallback

        fallback = quantized_fallback(directory, report, **kwargs)
        if fallback is not None:
            return fallback
    return result


def normalize_closed_identities(module):
    """Apply the closed, payload-independent identity normalizations in order.

    Returns the number of folds. Shared by native preparation and by the
    floating-contract decision, which must see the same normalized program.
    """
    from merlin.llvmlower import image_folds as image
    from merlin.llvmlower import structural_folds as structural
    from merlin.llvmlower.static_masked_sums import expose_static_masked_sums
    from merlin.targetgen.closed_scalar import fold_closed_generics, fold_unit_scales

    order = (
        expose_static_masked_sums,
        structural.fold_static_unit_indices,
        image.fold_integral_float_rounding,
        image.fold_boolean_unit_image,
        image.fold_integer_image_predicates,
        structural.fold_empty_reductions,
        image.fold_boolean_image_predicates,
        image.fold_boolean_domain,
        structural.normalize_singleton_boundaries,
        image.fold_nan_guarded_neutral,
        image.fold_integral_image_identities,
        structural.fold_singleton_reductions,
        structural.fold_singleton_selectors,
        expose_static_masked_sums,
        structural.fold_singleton_carriers,
        structural.prune_unused_tensor_inputs,
        image.fold_ieee_identity_selectors,
    )
    folds = sum(fold(module) for fold in order)
    # x * 1 and x / 1 are exact identities (movement, graded bit-exact). The
    # literal operand they read becomes unused and is pruned with them.
    for _ in range(3):  # a pruned literal operand can expose a round trip across generics
        unit_scales = fold_unit_scales(module)
        if not unit_scales:
            break
        folds += unit_scales + structural.prune_unused_tensor_inputs(module)

    # An elementwise generic whose inputs are all closed literals (for
    # example an empty reduction narrowed to f16/bf16 storage) is a uniform
    # literal fill; evaluate it exactly once instead of leaving a closed
    # conversion chain for the native word proofs.
    folds += fold_closed_generics(module)
    return folds


def _prepare_spec_int8(
    directory: Path,
    report: dict,
    *,
    target: str,
    facts: dict,
    software_spec: Path,
    host_capabilities: Path,
    provider=None,
    package=None,
    construction_observer=None,
):
    """Emit admitted sites and preserve independently refused source contractions."""
    import hashlib

    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.targetgen.core_aten_spec_int8 import admission_plan, host_readout_chain, provider_proof, select_spec

    selection = select_spec(target, facts, software_spec, host_capabilities)
    from merlin.targetgen.core_aten_exact_dtype import empty_host_plan, publish_plan

    empty = empty_host_plan(directory, report, selection)
    if empty is not None:
        return empty
    module = parse_mlir_file(directory / "model.mlir")
    folds = normalize_closed_identities(module)
    if folds:
        from merlin.llvmlower.entry_weight_projection import _pure_dce

        for function in module.walk():
            if function.name == "func.func":
                _pure_dce(function)
        (directory / "model.mlir").write_text(str(module) + "\n")
    if any("prov.input_metadata_mutation" in op.attributes for op in module.walk()):
        # Resizing/restriding caller storage is a host descriptor effect. Device
        # materialization cannot replace that effect or prove its alias ABI.
        from merlin.targetgen.core_aten_exact_dtype import retain_native_host

        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(
                dict(
                    family="movement",
                    code="input_metadata_mutation_requires_host",
                    reason="input_metadata_mutation_requires_host",
                )
            )
            + "\n"
        )
        return retain_native_host(directory, report, selection)
    original = (directory / "model.mlir").read_bytes()
    prepare = getattr(provider, "prepare_bundle", None)
    preparation = prepare(directory, target=target, facts=facts) if prepare is not None else {}
    if construction_observer is not None:
        construction_observer(parse_mlir_file(directory / "model.mlir"), "provider_prelude")
    import functools

    from merlin.kernels.shapes import observe_contractions
    from merlin.targetgen.core_aten_components import prepare_integer_components
    from merlin.targetgen.core_aten_exact_dtype import prepare_bitwise
    from merlin.targetgen.native_byte_activation import prepare as byte_activation
    from merlin.targetgen.native_quantized_activation import prepare as quantized_activation
    from merlin.targetgen.native_wide_activation import prepare as wide_activation

    components = functools.partial(
        prepare_integer_components, limb_attribute=f"{target}.limb", construction_observer=construction_observer
    )
    # Each candidate parses its own copy: a declined candidate's in-place
    # rewrites cannot change the next candidate's source or its ordinals.
    for candidate in (prepare_bitwise, byte_activation, wide_activation, components, quantized_activation):
        plan = candidate(directory, report, selection, provider, package, parse_mlir_file(directory / "model.mlir"))
        if plan is not None:
            return plan
    from merlin.targetgen.core_aten_exact_dtype import native_dtype_request, retain_native_host

    if native_dtype_request(report):
        return retain_native_host(directory, report, selection)
    module = parse_mlir_file(directory / "model.mlir")
    native = any(shape.dtypes == ("i8", "i8", "i32") for _, shape in observe_contractions(module))
    if native:
        proof, proof_error = provider_proof(provider, module, package=package, policy=selection["spec"])
        plan = admission_plan(module, selection, native=True, provider_proof=proof)
        if proof_error:
            plan["provider_refusal"] = proof_error
        selected = [site for site in plan["sites"] if site["status"] == "admitted"]
        plan["native_contractions"] = len(selected)
        plan["source_preparation"] = preparation
        if selected and (provider is None or not hasattr(provider, "admit_numeric_schedule")):
            for site in selected:
                site.update(status="refused", reasons=["qualified provider schedule binding unavailable"])
            selected = []
            plan.update(status="refused", native_contractions=0)
        if selected:
            contractions = dict((list(module.walk()).index(op), op) for op, _ in observe_contractions(module))
            for site in selected:
                step = next(row for row in site["steps"] if row["step"] == 3)
                bind_device_contraction(
                    contractions[site["source_operation_ordinal"]],
                    json.dumps({"software": step["software"]["declaration"]}),
                    f"native_int8_contraction_{site['source_operation_ordinal']}",
                )
            from merlin.targetgen.core_aten_spec_int8 import covering_proof

            covered = [contractions[site["source_operation_ordinal"]] for site in selected]
            proof = covering_proof(provider, module, covered, package=package, policy=selection["spec"])
            plan["schedule_proof"] = proof
            # Expose raw accumulators as well as every user/post-call result. The
            # oracle interprets source integer arithmetic, independently of lowering
            # and simulator outputs; eager is retained for boundary semantics.
            accumulators = [
                contractions[site["source_operation_ordinal"]].results[0]
                for site in selected
                if site.get("readout") != "int32_host_exact"
            ]
            expose_results(module, accumulators)
            module.verify()
            (directory / "model.native_reference.mlir").write_text(str(module) + "\n")
            (directory / "native-int8-request.json").write_text(
                json.dumps(dict(selected_ordinals=[], quant_min=-128, quant_max=127, eps=2**-12, dispatch=[])) + "\n"
            )
            process = _source_reference(directory, "--native")
            if process.returncode:
                for site in selected:
                    site.update(
                        status="refused", reasons=["exact source reference unavailable: " + process.stderr[-4000:]]
                    )
                plan.update(status="refused", native_contractions=0)
                (directory / "model.mlir").write_bytes(original)
                return publish_plan(directory, report["cases"], selection, plan)
            reference = json.loads((directory / "native-int8-reference.json").read_text())
            if len(reference["output_bytes"]) != report["output_count"] + len(accumulators):
                raise ValueError("native integer reference does not cover the complete exported boundary")
            plan["integer_reference"] = dict(
                status="prepared",
                engine="independent_pytorch_source_integer_interpreter",
                expected_output_bytes={str(i): raw for i, raw in enumerate(reference["output_bytes"])},
            )
            plan["host_int8_narrowing"] = any(
                abi["dtype"] == "i8"
                for record in report["cases"]
                if record["status"] == "bundled"
                for abi in record["output_abi"]
            )
            report["output_count"] += len(accumulators)
            (directory / "model.mlir").write_text(str(module) + "\n")
        return publish_plan(directory, report["cases"], selection, plan)
    plan = admission_plan(module, selection, planned_exact_readout=True)
    plan["source_sha256_before"] = hashlib.sha256(original).hexdigest()
    selected = [site for site in plan["sites"] if site["status"] == "admitted"]
    # A program with no admitted contraction is still worth recapturing when its floating sites would
    # be admitted as int8 work (affine maps, movements, window sums): the static recipe quantizes them.
    from merlin.targetgen.core_aten_spec_int8 import non_contraction_sites

    prospective = non_contraction_sites(
        module, selection, contract=selection.get("capability_contract"), prospective=True
    )
    plan["prospective_non_contraction_sites"] = prospective
    # A view result (permute, transpose) aliases its input: it is host metadata, and a recapture would
    # materialize it as a fresh tensor and break the call's alias contract.
    views = any(
        (record.get("case") or {}).get("output_input_aliases")
        for record in report["cases"]
        if record["status"] == "bundled"
    )
    scalarized = scalarized_contraction_candidates(module)
    plan["scalarized_contraction_recapture_candidates"] = scalarized
    from merlin.llvmlower.quantized_guard_domain import Unproved, prove_affine

    domain_candidates = []
    for ordinal, operation in enumerate(module.walk()):
        if operation.name != "linalg.generic":
            continue
        try:
            candidate = prove_affine(operation, conditional=True)
        except (Unproved, ValueError, OverflowError, ZeroDivisionError):
            continue
        domain_candidates.append(dict(source_operation_ordinal=ordinal, **candidate))
    # This requests conversion only. Actual admission must discharge the
    # finite-domain obligations against the newly frozen program qparams.
    plan["conditional_quantized_affine_candidates"] = domain_candidates
    non_contraction = (
        not selected
        and not views
        and (any(site["status"] == "admitted" for site in prospective) or bool(scalarized) or bool(domain_candidates))
    )
    recipes = [
        r
        for r in selection["recipes"]
        if r["activation"]
        and r["activation"]["dtype"] in {"int8", "uint8"}
        and r["activation"]["mode"] == "static"
        and r["activation"]["granularity"] == "tensor"
        and (
            -128 <= r["activation"]["quant_min"] < r["activation"]["quant_max"] <= 127
            if r["activation"]["dtype"] == "int8"
            else 0 <= r["activation"]["quant_min"] < r["activation"]["quant_max"] <= 255
        )
    ]
    if non_contraction and len(recipes) != 1:
        non_contraction = False
        plan["refusal_code"] = "no_static_recipe"
    if selected and (len(recipes) != 1 or provider is None or not hasattr(provider, "admit_numeric_schedule")):
        for site in selected:
            site.update(
                status="refused",
                reasons=["static activation recipe or qualified provider schedule binding unavailable"],
            )
        selected = []
    reference = None
    if selected or non_contraction:
        (directory / "model.before_spec_int8.mlir").write_bytes(original)

        def declaration_keys(site):
            def key(step):
                row = next((row for row in site["steps"] if row["step"] == step), {})
                return json.dumps(
                    {
                        authority: row[authority]["declaration"]
                        for authority in ("software", "host")
                        if authority in row
                    },
                    sort_keys=True,
                )

            return dict(quantize=key(1), contraction=key(3), cast=key(5), mul=key(6))

        from merlin.targetgen.capsule_source import NumericCaptureRefused, PytorchRefSource
        from merlin.targetgen.core_aten_batch import build_core_aten_batch
        from merlin.targetgen.core_aten_capture import case_capture_name, loader_source
        from merlin.targetgen.quant_recipe import digest

        records = [r for r in report["cases"] if r["status"] == "bundled"]
        if len(records) != 1:
            raise ValueError("static capture needs single-case attribution")
        source_case = records[0]["case"]
        capture_root = directory / "capture-int8"
        capture_dir = capture_root / case_capture_name(source_case)
        capture_dir.mkdir(parents=True, exist_ok=True)
        loader = capture_dir / "capsule.pytorch.py"
        loader.write_text(loader_source(source_case))
        recipe = dict(
            recipes[0],
            calibration_domain=selection["spec"]["numerical_semantics"]["calibration_domain"],
            quantization_nonfinite=selection["spec"]["numerical_semantics"].get("quantization_nonfinite"),
            readout_specials=selection["spec"]["numerical_semantics"].get("readout_specials"),
            f32_boundary_overflow=selection["spec"]["numerical_semantics"].get("f32_boundary_overflow"),
            accumulator_dtype=selection["readout_facets"][0]["accumulator_dtype"],
            software_admission=selection["spec"],
            software_numerical_engine="integer_reference",
            readout_numerics="integer_readout",
            **({"host_readout_declarations": declaration_keys(selected[0])["mul"]} if selected else {}),
        )
        # A Core ATen call's contraction operands are its run-time inputs, not stored parameters.
        recipe["runtime_weight_contractions"] = True
        recipe["recipe_sha256"] = digest(recipe)
        try:

            def complex_argument(value):
                if isinstance(value, dict):
                    return (
                        value.get("kind") == "complex"
                        or str(value.get("dtype", "")).startswith("complex")
                        or any(complex_argument(item) for item in value.values())
                    )
                return isinstance(value, list) and any(complex_argument(item) for item in value)

            if complex_argument(source_case["arguments"]):
                raise NumericCaptureRefused(
                    "static int8 recipe does not admit complex tensors or scalar arguments", code="complex_calibration"
                )
            artifact = PytorchRefSource().capture_loader(loader, "int8", recipe=recipe, workdir=capture_dir / "work")
            refused_recipes = [
                row
                for row in (artifact.meta.get("quantization_stats") or {}).get("integer_recipes", ())
                if "refused" in row
            ]
            if refused_recipes:
                raise NumericCaptureRefused(
                    "integer recipes lack an exact-arithmetic proof",
                    code="exact_recipe_refused",
                    details={"integer_recipes": refused_recipes},
                )
            from merlin.targetgen.pt2e_reference_selection import EXACT_ENGINE

            # Declared admission consumes the exact oracle's sites and post-call inputs; a graph
            # whose golden reference had to be framework float has no such reference to consume.
            golden_selection = (artifact.meta.get("integerization_receipt") or {}).get("reference_selection")
            if golden_selection and golden_selection.get("engine") != EXACT_ENGINE:
                raise NumericCaptureRefused(
                    "declared software admission requires the exact-rational reference: "
                    + str(golden_selection.get("reason")),
                    code="exact_reference_unavailable",
                    details={"reference_selection": golden_selection},
                )
            from merlin.targetgen.core_aten_spec_int8 import case_readout

            # The readout the case was quantized for governs its admission and its recorded spec.
            try:
                selection = case_readout(
                    selection, (artifact.meta.get("quantization_stats") or {}).get("quantization_scheme")
                )
            except ValueError as error:
                raise NumericCaptureRefused(str(error), code="case_quantization_scheme_unresolved") from error
        except NumericCaptureRefused as exc:
            # Only explicit frontend refusals grant original host fallback.
            for site in selected:
                site.update(
                    status="refused",
                    refusal_code=exc.code,
                    reasons=[str(exc)],
                )
            non_contraction = False
            plan.update(status="refused", quantized_contractions=0, quantized_non_contraction_sites=0)
            plan["refusal_code"] = exc.code
            plan["refusal_details"] = exc.details
            selected = []
        if selected or non_contraction:
            reference_path = Path(artifact.meta["weights_manifest"]).parent / "spec-int8-reference.json"
            reference = json.loads(reference_path.read_text())
            (directory / "spec-int8-reference.json").write_bytes(reference_path.read_bytes())
            (capture_dir / "capsule.linalg.mlir").write_text(artifact.linalg_mlir)
            (capture_dir / "inputs.json").write_text(json.dumps(artifact.inputs))
            (capture_dir / "golden.json").write_text(json.dumps(artifact.golden))
            (capture_dir / "capture.json").write_text(json.dumps({"capture_meta": artifact.meta}))
            prepared_dir = directory / "capture-bundle"
            prepared_report = build_core_aten_batch({"cases": [source_case]}, capture_root, prepared_dir)
            prepared_module = parse_mlir_file(prepared_dir / "model.mlir")
            if recipe.get("quantization_nonfinite") == "byte_storage_zero_nan_saturating_infinity":
                from xdsl.dialects.builtin import UnitAttr

                for operation in prepared_module.walk():
                    if getattr(operation.attributes.get("prov.aten"), "data", None) == "m2m_q.quantize.default":
                        operation.attributes["merlin.quantizer_nan_storage_zero"] = UnitAttr()
            from merlin.llvmlower.structural_folds import fold_scalar_tensor_roundtrips

            scalar_carriers = fold_scalar_tensor_roundtrips(prepared_module)
            from merlin.llvmlower.quantized_guard_domain import canonicalize_quantized_affine

            domain_proofs = canonicalize_quantized_affine(
                prepared_module,
                exact_rational=selection["spec"].get("numerical_semantics", {}).get("model", {}).get("engine")
                == "integer_reference",
            )
            capture_proof, capture_proof_error = provider_proof(
                provider, prepared_module, package=package, policy=selection["spec"]
            )
            captured_plan = admission_plan(prepared_module, selection, captured=True, provider_proof=capture_proof)
            residuals = artifact.meta.get("integerization_receipt", {}).get("ieee_quantizer_residuals", [])
            active_residuals = [row for row in residuals if row.get("beta") != 0]
            if active_residuals:
                captured_plan["ieee_special_family"] = dict(
                    family="F3", proof="declared_single_residual_boundary", readouts=active_residuals
                )
            captured_plan["scalar_tensor_carriers_folded"] = scalar_carriers
            captured_plan["quantized_affine_domain_proofs"] = domain_proofs
            if capture_proof_error:
                captured_plan["provider_refusal"] = capture_proof_error
            from merlin.targetgen.core_aten_spec_int8 import source_declaration_continuity

            continuity = source_declaration_continuity(plan, captured_plan)
            captured_plan["source_declaration_continuity"] = continuity
            if continuity["status"] != "preserved":
                for site in captured_plan["sites"]:
                    site["status"] = "refused"
                    site["reasons"].append("exact source operator declaration was not preserved through integerization")
                captured_plan["status"] = "refused"
            island_sites = [site for site in captured_plan["non_contraction_sites"] if site["quantized_island"]]
            non_contraction_admitted = (
                not captured_plan["sites"]
                and bool(island_sites)
                and all(site["status"] == "admitted" for site in island_sites)
                and all(row["host"]["status"] == "admitted" for row in captured_plan["quantization_sites"])
            )
            if (
                continuity["status"] != "preserved"
                or any(site["status"] != "admitted" for site in captured_plan["sites"])
                or (not captured_plan["sites"] and not non_contraction_admitted)
            ):
                # Refuse the complete capture before replacing the host source.
                plan = captured_plan
                selected = []
                plan["source_sha256_before"] = hashlib.sha256(original).hexdigest()
                plan["source_sha256_after"] = plan["source_sha256_before"]
                return publish_plan(directory, records, selection, plan)
            selected = captured_plan["sites"]
            if len(selected) != len(reference["scales"]) or any(
                scale.get("accumulator") is None for scale in reference["scales"]
            ):
                for site in selected:
                    site.update(status="refused", reasons=["exact reference does not cover every emitted accumulator"])
                plan = {**captured_plan, "status": "refused", "quantized_contractions": 0}
                plan["source_sha256_before"] = hashlib.sha256(original).hexdigest()
                plan["source_sha256_after"] = plan["source_sha256_before"]
                return publish_plan(directory, records, selection, plan)
            module = prepared_module
            plan = captured_plan
            if non_contraction_admitted:
                plan["status"] = "admitted"
            plan["source_sha256_before"] = hashlib.sha256(original).hexdigest()
            plan["capture_recipe"] = recipe
            plan["capture_meta"] = artifact.meta
            observations = []
            from xdsl.dialects.builtin import StringAttr

            from merlin.kernels.shapes import observe_contractions

            contractions = [op for op, _ in observe_contractions(module)]
            for site, contraction, scale in zip(selected, contractions, reference["scales"], strict=True):
                keys, region = declaration_keys(site), f"spec_int8_contraction_{site['source_operation_ordinal']}"
                if scale.get("readout", ()) is None:
                    # The reference states this GEMM's accumulator only: it reaches its readout through
                    # further exact integer work (a convolution_backward's tap accumulation or layout
                    # change), so there is no direct cast/multiply chain to bind, and the final outputs
                    # are compared exactly instead.
                    bind_device_contraction(contraction, keys["contraction"], region)
                    site["host_readout_bridge"] = {"status": "composed_integer_readout"}
                    observations.append(() if site.get("readout") == "int32_host_exact" else (contraction.results[0],))
                    continue
                # Readout preserves two separately rounded frontend f32 multiplies.
                (cast, first_readout, readout), restored_readout, bridge = host_readout_chain(
                    contraction, with_bridge=True
                )
                site["host_readout_bridge"] = bridge
                site["host_readout_operation_ordinals"] = [
                    list(module.walk()).index(o) for o in (first_readout, readout)
                ]
                bind_device_contraction(contraction, keys["contraction"], region)
                for op, key in ((cast, "cast"), (first_readout, "mul"), (readout, "mul")):
                    op.attributes["prov.declarations"] = StringAttr(keys[key])
                readout.attributes["prov.role"] = StringAttr("spec_int8_host_readout")
                observations.append((contraction.results[0], restored_readout))
            operations = list(module.walk())
            for row in plan["quantization_sites"]:
                keys = {
                    authority: row[authority]["declaration"] for authority in ("software", "host") if authority in row
                }
                operations[row["source_operation_ordinal"]].attributes["prov.declarations"] = StringAttr(
                    json.dumps(keys, sort_keys=True)
                )
            # The provider proves contraction schedules; a contraction-free program has none to prove.
            from merlin.targetgen.core_aten_spec_int8 import covering_proof

            policy = selection["spec"]
            proof = covering_proof(provider, module, contractions, package=package, policy=policy) if selected else None
            plan["schedule_proof"] = proof
            plan["quantized_contractions"] = len(selected)
            plan["quantized_non_contraction_sites"] = sum(
                site["status"] == "admitted" and site["quantized_island"] for site in plan["non_contraction_sites"]
            )
            plan["calibration"].update(
                performed=True,
                recipe=recipe,
                graph="TorchAO PT2E prepared floating graph",
                scales_frozen_before_conversion=True,
            )
            # Preserve caller-owned inputs; bind observed user and post-call
            # values through the fresh conversion's actual output-role mapping.
            fresh = prepared_report["cases"][0]
            reference["outputs"] = map_reference_boundary(fresh, reference, prepared_report["output_count"])
            restore_caller_layouts(module, records[0], fresh, prepared_dir / "semantic_io.json")
            caller_case = records[0]["case"]
            records[0].update(
                {
                    k: v
                    for k, v in fresh.items()
                    if k not in ("routing", "capture_directory", "case", "input_indices", "input_abi")
                }
            )
            records[0]["case"] = caller_case
            if prepared_report["input_count"] != report["input_count"]:
                raise ValueError("PT2E recapture changed the caller input ABI")
            for name in ("weights.safetensors.manifest.json", "semantic_io.json"):
                path = prepared_dir / name
                if path.exists():
                    (directory / name).write_bytes(path.read_bytes())
            report["output_count"] = prepared_report["output_count"]
        if not selected and not non_contraction:
            plan["status"] = "refused"
        if selected or non_contraction:
            records = [r for r in report["cases"] if r["status"] == "bundled"]
            if len(records) != 1:
                raise ValueError("static int8 needs single-case attribution")
            record = records[0]
            record["fp32_case"] = copy.deepcopy(record["case"])
            record["reference_engine"] = reference["engine"]
            # Keep the full original result structure, including complex components.
            from merlin.targetgen.core_aten_batch import _tensor_documents

            record["case"] = copy.deepcopy(record["case"])

            def values(document, index):
                value = reference["outputs"][index]
                if document["dtype"] in {"complex64", "complex128"}:

                    def encode(v):
                        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                            return dict(kind="complex", real=v[0], imag=v[1])
                        return [encode(x) for x in v]

                    value = encode(value)
                document["values"] = value

            for document, index in zip(
                _tensor_documents(record["case"]["expected"]), record["output_indices"], strict=True
            ):
                values(document, index)
            if record.get("semantic_boundary"):
                for document, index in zip(
                    _tensor_documents(record["case"]["post_arguments"]),
                    record["semantic_boundary"]["post_indices"],
                    strict=True,
                ):
                    values(document, index)
            record["case"]["comparison_parameters"] = dict(rtol=0.0, atol=0.0, equal_nan=True)
            # Retain actual i32 and host readout buffers as additional grading outputs.
            acc_index = expose_results(module, [v for pair in observations for v in pair])
            expected_bytes = map_reference_byte_boundary(fresh, reference, acc_index)
            for site, scale, observed in zip(selected, reference["scales"], observations, strict=True):
                if observed:
                    expected_bytes[str(acc_index)] = np.asarray(scale["accumulator"], np.int32).tobytes().hex()
                if len(observed) == 2:
                    expected_bytes[str(acc_index + 1)] = np.asarray(scale["readout"], np.float32).tobytes().hex()
                binding = next(
                    b for b in proof["bindings"] if b["source_operation_ordinal"] == site["source_operation_ordinal"]
                )
                schedule = next(k for k in proof["kernels"] if k["symbol"] == binding["symbol"])
                actual_bound = (
                    schedule["padded_primitive_k"]
                    * scale["operands"][0]["quantized_max_abs"]
                    * scale["operands"][1]["quantized_max_abs"]
                )
                site["accumulator_headroom"] = dict(
                    mesh_uniform_bound=schedule["max_partial_sum_bound"],
                    mesh_actual_operand_bound=actual_bound,
                    mesh_positive_limit=schedule["mac_positive_limit"],
                    uniform_headroom=schedule["headroom"],
                    actual_operand_headroom=schedule["mac_positive_limit"] - actual_bound,
                    full_i32_max_abs=scale["accumulator_max_abs"],
                    full_i32_headroom=(1 << 31) - 1 - scale["accumulator_max_abs"],
                    k_split=schedule["k_split"],
                    padded_primitive_k=schedule["padded_primitive_k"],
                )
                site.update(
                    accumulator_output_index=acc_index if observed else None,
                    readout_output_index=acc_index + 1 if len(observed) == 2 else None,
                    actual_dispatch="provider catalog with independently checked WS schedule",
                    scales={k: v for k, v in scale.items() if k not in {"accumulator", "readout"}},
                )
                acc_index += len(observed)
            plan["integer_reference"] = dict(
                status="prepared",
                engine=reference["engine"],
                expected_output_bytes=expected_bytes,
                final_comparison="exact bit patterns for every result dtype; signed zero distinct",
                accumulator_comparison="exact i32",
            )
            plan["scales"] = [
                {k: v for k, v in scale.items() if k not in {"accumulator", "readout"}} for scale in reference["scales"]
            ]
            record["scales"] = plan["scales"]
            products = {json.dumps(scale.get("product")): scale.get("product") for scale in plan["scales"]}
            if len(products) == 1 and isinstance(next(iter(products.values())), (int, float)):
                record["output_scale"] = next(iter(products.values()))
            report["output_count"] += sum(len(observed) for observed in observations)
            module.verify()
            (directory / "model.mlir").write_text(str(module) + "\n")
    if (
        not plan.get("quantized_contractions", 0)
        and not plan.get("quantized_non_contraction_sites", 0)
        and (directory / "model.mlir").read_bytes() == original
    ):
        # A failed prospective int8 declaration or calibration does not rule
        # out a complete native integer component program. Its own typed
        # family, transfer, source-bit and provider proofs are still required.
        if any(abi["dtype"] == "f32" for record in report["cases"] for abi in record.get("input_abi", [])):
            components = prepare_integer_components(
                directory,
                report,
                selection,
                provider,
                package,
                parse_mlir_file(directory / "model.mlir"),
                limb_attribute=f"{target}.limb",
                construction_observer=construction_observer,
            )
            if components is not None:
                components["quantization_fallback"] = dict(
                    refusal_code=plan.get("refusal_code", "int8_family_not_admitted"),
                    method="native_exact_components_without_calibration",
                )
                (directory / "spec-int8-admission.json").write_text(json.dumps(components, indent=2) + "\n")
                return components
    plan["source_sha256_after"] = hashlib.sha256((directory / "model.mlir").read_bytes()).hexdigest()
    document = {"selection": selection, **plan}
    (directory / "spec-int8-admission.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    for case in report["cases"]:
        if case["status"] == "bundled":
            case["numeric_admission"] = plan
            case["numeric_selection"] = selection["identities"]
    return document


def prepare_w8a8(directory: Path, report: dict, *, triples, ranks) -> dict:
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.contraction_quantization import SPEC, quantize_module, selected_contractions

    module = parse_mlir_file(directory / "model.mlir")
    selected = list(selected_contractions(module, triples, ranks))
    if not selected:
        return {"quantized_contractions": 0, "scales": []}
    # Single-case attribution is mandatory for both oracle replacement and device evidence.
    records = [r for r in report["cases"] if r["status"] == "bundled"]
    if len(records) != 1:
        raise ValueError("W8A8 execution requires one case per shard")
    original = directory / "model.before_w8a8.mlir"
    original.write_bytes((directory / "model.mlir").read_bytes())
    operations = list(module.walk())
    request = dict(
        selected_ordinals=[operations.index(op) for op in selected],
        quant_min=SPEC.quant_min,
        quant_max=SPEC.quant_max,
        eps=SPEC.eps,
    )
    (directory / "w8a8-request.json").write_text(json.dumps(request, indent=2) + "\n")
    record = records[0]
    record["fp32_case"] = copy.deepcopy(record["case"])
    record["case"] = copy.deepcopy(record["case"])
    record["w8a8_policy"] = policy_record()
    accumulators = []
    count = quantize_module(module, selected, accumulators=accumulators)
    record["w8a8_accumulators"] = accumulators
    record["w8a8_selected_ordinals"] = request["selected_ordinals"]
    report["output_count"] += len(accumulators)
    # Add observation-only boundary results without changing user result indices.
    semantic = directory / "semantic_io.json"
    if semantic.exists():
        data = json.loads(semantic.read_text())
        data["outputs"].extend([None] * len(accumulators))
        semantic.write_text(json.dumps(data, indent=2) + "\n")
    (directory / "model.mlir").write_text(str(module) + "\n")
    return dict(quantized_contractions=count, scales=[], policy=record["w8a8_policy"])


def complete_w8a8_reference(directory, report, dispatch):
    """Qualify only after source-bound provider selection exposes actual primitives."""
    from merlin.targetgen.core_aten_batch import _tensor_documents

    request_path = directory / "w8a8-request.json"
    request = json.loads(request_path.read_text())
    request["dispatch"] = dispatch
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    process = _source_reference(directory)
    (directory / "w8a8-reference.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise ValueError(f"independent W8A8 reference failed: {process.stderr[-2000:]}")
    reference = json.loads((directory / "w8a8-reference.json").read_text())
    record = next(r for r in report["cases"] if r["status"] == "bundled")
    record["scales"] = reference["scales"]
    record["reference_engine"] = reference["engine"]
    record["w8a8_reference_bytes"] = reference["output_bytes"]
    record["w8a8_dispatch"] = dispatch
    expected = _tensor_documents(record["case"]["expected"])
    if len(expected) != len(record["output_indices"]):
        raise ValueError("W8A8 oracle needs tensor output documents")

    def reference_values(document, index):
        values = reference["outputs"][index]
        if document["dtype"] in {"complex64", "complex128"}:
            packed = np.asarray(values)
            if packed.shape != (*document["shape"], 2):
                raise ValueError("complex oracle output does not match the physical ABI")

            def encode(value):
                if len(value) == 2 and all(isinstance(v, (float, int)) for v in value):
                    return dict(kind="complex", real=value[0], imag=value[1])
                return [encode(v) for v in value]

            values = encode(values)
        return values

    for document, index in zip(expected, record["output_indices"], strict=True):
        document["values"] = reference_values(document, index)
    if record.get("semantic_boundary"):
        posts = _tensor_documents(record["case"]["post_arguments"])
        for document, index in zip(posts, record["semantic_boundary"]["post_indices"], strict=True):
            document["values"] = reference_values(document, index)
    record["case"]["comparison_parameters"] = dict(rtol=0.0, atol=0.0, equal_nan=False, storage_bits=True)
    report["numeric_preparation"].update(
        scales=reference["scales"], reference_engine=reference["engine"], dispatch=dispatch
    )


def policy_record():
    from merlin.llvmlower.contraction_quantization import SPEC

    return dict(
        name="int8_w8a8",
        operand_range=[SPEC.quant_min, SPEC.quant_max],
        eps=SPEC.eps,
        granularity="dynamic_per_tensor",
        quantization="f32_reciprocal_first_roundeven_clamp",
        dequantization="f32_acc_cast_then_f64_scale_product_and_multiply_then_one_f32_round_then_storage_cast",
        nonfinite_inputs="sanitize_before_integer_conversion_and_propagate_canonical_nan_to_entire_contraction",
        nonrepresentable_finite_result="runtime_assert_and_oracle_refusal",
        subnormals="quantize_to_zero_via_eps_floor",
        signed_zero="distinct_output_bit_patterns",
        correctness_gate="raw_i32_accumulator_and_final_float_storage_bits",
        nan_comparison="canonical_nan_storage_bits_only",
    )


def validate_primitive_dispatch(numeric, *, full_k, primitive_ks):
    from merlin.llvmlower.contraction_quantization import SPEC
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    bound = integer_split_k_limit(numeric, tile_dim=1, full_k=full_k, operand_range=(SPEC.quant_min, SPEC.quant_max))
    if bound is None or not primitive_ks:
        raise ValueError("W8A8 dispatch lacks selected internal arithmetic or observed primitive K")
    if any(type(k) is not int or not 0 < k <= bound["max_true_k"] for k in primitive_ks):
        raise ValueError("W8A8 dispatched primitive K exceeds internal MAC partial-sum bound")
    return dict(full_k=full_k, primitive_ks=primitive_ks, numerical_semantics=numeric["numerical_semantics"], **bound)


def checked_w8a8_builder(builder, *, target, package, directory, report):
    """Screen the actual emitted target IR before catalog selection and linking.

    Operation/extent names are target-owned observation data, never ISA guesses.
    The catalog's target IR hash binds the observations to the compiled object.
    """
    import hashlib

    from xdsl.dialects.llvm import LLVM
    from xdsl.parser import Parser

    from merlin.common.paths import repo_root
    from merlin.frontends.linalg_mlir import make_context
    from merlin.targetgen.software_spec import load_software_spec, software_spec_path_for_recipe
    from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment

    te = load_target_experiment(descriptor_for(target))
    spec_path = software_spec_path_for_recipe(repo_root() / te.numeric_profile)
    spec = load_software_spec(spec_path, target=target)
    if spec.get("status") != "reviewed":
        raise ValueError("W8A8 requires a reviewed internal arithmetic declaration")
    observation = spec["numerical_semantics"]["internal_arithmetic"].get("primitive_k_observation")
    if not observation:
        raise ValueError("W8A8 target lacks primitive K observation contract")

    def checked(prepared, workdir):
        manifest_path, object_path = builder(prepared, workdir)
        manifest = json.loads(Path(manifest_path).read_text())
        digest = manifest["compilation"]["target_ir_sha256"]
        sources = [p for p in Path(workdir).glob("*.mlir") if hashlib.sha256(p.read_bytes()).hexdigest() == digest]
        if len(sources) != 1:
            raise ValueError("W8A8 catalog lacks a unique hash-bound target IR")
        ctx = make_context()
        ctx.load_dialect(LLVM)
        from merlin.targetgen.plugins import load_module

        identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()
        dialect_module = load_module(
            package, observation["dialect_module"], package_name=f"core_aten_backend_{identity}"
        )
        ctx.load_dialect(getattr(dialect_module, observation["dialect_symbol"]))
        module = Parser(ctx, sources[0].read_text()).parse_module()
        primitives, callees = {}, {}
        for function in module.body.block.ops:
            if function.name != "llvm.func":
                continue
            ks, calls = [], []
            for op in function.walk():
                name = op.op_name.data if op.name == "builtin.unregistered" else op.name
                if op.name == "llvm.call":
                    calls.append(op.callee.root_reference.data)
                if name == observation["operation"]:
                    value = op.attributes.get(observation["extent_attribute"]) or op.properties.get(
                        observation["extent_attribute"]
                    )
                    if value is None or not hasattr(value, "value"):
                        raise ValueError("W8A8 primitive has no static declared K extent")
                    ks.append(value.value.data)
            primitives[function.sym_name.data] = ks
            callees[function.sym_name.data] = calls

        def reachable_primitives(symbol, visited=()):
            if symbol in visited or symbol not in primitives:
                raise ValueError("W8A8 catalog has recursive or unobserved primitive calls")
            return primitives[symbol] + [
                k for callee in callees[symbol] for k in reachable_primitives(callee, (*visited, symbol))
            ]

        kernels = {k["symbol"]: k for k in manifest["kernels"]}
        # Follow actual LLVM calls rather than assuming a batched wrapper layout.
        dispatch_by_region = {}
        for binding in manifest["bindings"]:
            kernel = kernels[binding["symbol"]]
            proof = validate_primitive_dispatch(
                spec, full_k=kernel["dimensions"]["k"], primitive_ks=reachable_primitives(binding["symbol"])
            )
            proof.update(
                target_ir_sha256=digest,
                software_spec_path=str(spec_path.resolve()),
                target_ir_path=str(sources[0].resolve()),
                object_path=str(Path(object_path).resolve()),
                software_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                object_sha256=hashlib.sha256(Path(object_path).read_bytes()).hexdigest(),
                observation=observation,
            )
            dispatch_by_region[binding["region"]] = proof
        record = next(r for r in report["cases"] if r["status"] == "bundled")
        dispatch = [dispatch_by_region[f"w8a8_contraction_{ordinal}"] for ordinal in record["w8a8_selected_ordinals"]]
        complete_w8a8_reference(directory, report, dispatch)
        (directory / "w8a8-dispatch.json").write_text(json.dumps(dispatch, indent=2) + "\n")
        return manifest_path, object_path

    return checked


def exact_w8a8_gate(record, outputs):
    """Compare raw readout words and final storage bytes, retaining signed zero."""
    if len(record.get("w8a8_accumulators", [])) != len(record["scales"]):
        return dict(
            name="raw_i32_accumulator_and_final_float_storage_bits",
            passed=False,
            reason="raw accumulator observation roster differs from oracle",
            accumulators=[],
            final_float_storage=[],
        )
    accumulator_details = []
    for abi, site in zip(record.get("w8a8_accumulators", []), record["scales"], strict=True):
        expected = np.asarray(site["accumulator"], dtype=np.int32).reshape(abi["shape"]).tobytes()
        raw = outputs[abi["index"]] if abi["index"] < len(outputs) else None
        accumulator_details.append(dict(index=abi["index"], passed=raw == expected))
    final = []
    references = record.get("w8a8_reference_bytes", [])
    indices = [*record["output_indices"], *record.get("semantic_boundary", {}).get("post_indices", [])]
    for index in dict.fromkeys(indices):
        passed = index < len(references) and index < len(outputs) and outputs[index].hex() == references[index]
        final.append(dict(index=index, passed=passed))
    passed = bool(accumulator_details) and all(d["passed"] for d in [*accumulator_details, *final])
    return dict(
        name="raw_i32_accumulator_and_final_float_storage_bits",
        passed=passed,
        accumulators=accumulator_details,
        final_float_storage=final,
    )


def fp32_error(record, outputs) -> list[dict]:
    """Report compiled-output error vs original golden, without affecting correctness."""
    from merlin.targetgen.core_aten_batch_grade import _DTYPES, _complex_values, _leaves

    errors = []
    for index, abi, expected in zip(
        record["output_indices"], record["output_abi"], _leaves(record["fp32_case"]["expected"]), strict=True
    ):
        complex_type = {"complex<f32>": np.complex64, "complex<f64>": np.complex128}.get(abi["dtype"])
        compute_type = np.complex128 if complex_type else np.float64
        actual = np.frombuffer(outputs[index], complex_type or _DTYPES[abi["dtype"]]).astype(compute_type)
        golden = np.asarray(_complex_values(expected["values"]), dtype=compute_type).reshape(-1)
        difference = np.abs(actual - golden)
        relative = np.divide(difference, np.abs(golden), out=np.where(difference == 0, 0.0, np.inf), where=golden != 0)
        norm = float(np.linalg.norm(actual) * np.linalg.norm(golden))
        cosine = float(np.real(np.vdot(actual, golden)) / norm) if norm else float(np.array_equal(actual, golden))
        error_energy = float(np.sum(difference**2))
        golden_energy = float(np.sum(np.abs(golden) ** 2))
        rmse = float(np.sqrt(error_energy / difference.size)) if difference.size else 0.0
        sqnr = (
            float(10 * np.log10(golden_energy / error_energy))
            if golden_energy and error_energy
            else (float("inf") if error_energy == 0 else float("-inf"))
        )
        normalized_l2 = (
            float(np.sqrt(error_energy / golden_energy))
            if golden_energy
            else (0.0 if error_energy == 0 else float("inf"))
        )
        # An output epilogue can combine several contraction sites. Report each
        # site's LSB explicitly; no invented single scale for composed outputs.
        lsb_errors = []
        for i, site in enumerate(record.get("scales", [])):
            lsb = site.get("output_lsb", site.get("product"))
            if lsb is None and site.get("lhs") is not None and site.get("rhs") is not None:
                lsb = (np.asarray(site["lhs"]) * np.asarray(site["rhs"])).tolist()
            scalar = isinstance(lsb, (int, float)) and np.isfinite(lsb) and lsb > 0
            lsb_errors.append(
                dict(
                    site=i,
                    lsb=lsb,
                    max_abs=float(np.max(difference, initial=0) / lsb) if scalar else None,
                    rmse=rmse / lsb if scalar else None,
                )
            )
        errors.append(
            dict(
                max_abs=float(np.max(difference, initial=0)),
                max_rel=float(np.max(relative, initial=0)),
                relative_inf_at_zero_golden=int(np.sum((golden == 0) & (difference != 0))),
                rmse=rmse,
                sqnr_db=sqnr,
                cosine=cosine,
                normalized_l2=normalized_l2,
                error_in_output_lsbs=lsb_errors,
                contraction_diagnostics=[
                    dict(
                        operands=site.get("operands"),
                        accumulator=site.get("accumulator_metrics"),
                        k_split=site.get("k_split"),
                    )
                    for site in record.get("scales", [])
                ],
                output_lsb_error="not_applicable"
                if not isinstance(record.get("output_scale"), (int, float)) or not record["output_scale"]
                else float(np.max(difference, initial=0) / record["output_scale"]),
                output_lsb_unit="contraction readout scale product; list gives each stage when no single scale exists",
                contraction_output_lsb_errors=[
                    dict(
                        scale=scale["product"],
                        max_error=float(np.max(difference, initial=0) / scale["product"])
                        if isinstance(scale["product"], (int, float))
                        and np.isfinite(scale["product"])
                        and scale["product"] > 0
                        else None,
                    )
                    for scale in record.get("scales", [])
                    if "product" in scale
                ],
                output_f32_ulp_error=float(
                    np.max(
                        np.divide(difference, np.abs(np.spacing(golden.astype(np.float32))).astype(np.float64)),
                        initial=0,
                    )
                )
                if not complex_type
                else "complex_components_reported_separately",
                relative_error_at_zero=int(np.count_nonzero((golden == 0) & (difference != 0))),
            )
        )
    return errors
