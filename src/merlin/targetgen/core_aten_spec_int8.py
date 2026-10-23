"""Fail-closed admission for a declared static int8 full-call pipeline.

A scale's storage type is not its product arithmetic. Phase-0 form resolution
must not fill numerical semantics from a header or a target name. In particular,
an admitted MAC does not grant a narrowing readout or a host transfer.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from merlin.targetgen.operator_review import review_accepted
from merlin.targetgen.software_spec import admit_operation, load_software_spec

QUANTIZE = "quantized_decomposed.quantize_per_tensor.default"
DEQUANTIZE = "quantized_decomposed.dequantize_per_tensor.default"


def quantization_schema():
    """The versioned frontend-owned vocabulary, also available in installed packages."""
    import json

    from merlin.common.paths import contract_dir

    schema = json.loads((contract_dir() / "quant_schema.json").read_text())
    if schema.get("schema_version") != 1 or schema.get("unknown_keys") != "reject":
        raise ValueError("unsupported frontend quantization schema")
    return schema


def validate_quantization_attributes(op, op_kind=None):
    """Quantization metadata is a closed vocabulary, including on MACs and seeds."""
    allowed = quantization_schema()["attributes"]
    unknown = sorted(
        key
        for key in op.attributes
        if key.startswith("prov.quant.")
        and (key not in allowed or (op_kind is not None and op_kind not in allowed[key]["op_kinds"]))
    )
    if unknown:
        raise ValueError(f"unknown readout quantization attributes: {unknown}")


def host_signature(op: str, rank: int, parameters: dict) -> dict:
    return {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32" if op == QUANTIZE else "i8", "f32", "i64"],
        "ordered_result_dtypes": ["i8" if op == QUANTIZE else "f32"],
        "operand_dtype": "f32" if op == QUANTIZE else "i8",
        "rank": rank,
        "quantization_parameters": parameters,
    }


def select_spec(target: str, facts: dict, spec_path: Path, host_path: Path) -> dict:
    """Resolve authored forms through the existing Phase-0 mechanism, retaining evidence."""
    from merlin.common.yaml import load_yaml
    from merlin.targetgen.host_capabilities import validate_host_capabilities
    from merlin.targetgen.quant_recipe import derive_candidates
    from merlin.targetgen.readout_facet import for_target
    from merlin.targetgen.spec_fact_drift import fact_capabilities, resolve_spec
    from merlin.targetgen.target_registry import load_contract

    spec = load_software_spec(spec_path, target=target)
    host = validate_host_capabilities(load_yaml(host_path))
    contract = load_contract(target)
    facets = for_target(target, facts=facts, contract=contract)
    candidates = [candidate.to_dict() for candidate in derive_candidates(contract, facets)]
    recipes = [candidate["recipe"] for candidate in candidates if candidate["recipe"] is not None]
    capabilities = fact_capabilities(
        target=target,
        contract=contract,
        raw_facts=facts,
        readout_facets=[facet.to_dict() for facet in facets],
        quantization_candidates=candidates,
    )
    resolved, resolution = resolve_spec(spec, capabilities)
    return {
        "spec": resolved,
        "host": host,
        # The selected hardware capability declaration, for the HW screen of non-contraction sites.
        "capability_contract": contract,
        "resolution": resolution,
        "recipes": recipes,
        "quantization_candidates": candidates,
        "readout_facets": [facet.to_dict() for facet in facets],
        "identities": {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("software_spec", spec_path), ("host_capabilities", host_path))
        },
    }


def declaration_location(selection, authority, key, fields=()):
    """Bind a decision to the exact authored YAML node, including rank restrictions."""
    import yaml

    identity = selection.get("identities", {}).get("software_spec" if authority == "software" else "host_capabilities")
    if not identity:
        return {"declaration": key}
    root = yaml.compose(Path(identity["path"]).read_text())

    def field(node, name):
        if not isinstance(node, yaml.MappingNode):
            return None
        return next((v for k, v in node.value if k.value == name), None)

    operations = field(root, "operations")
    if authority == "software":
        node = field(operations, key) or operations
    else:
        node = next((n for n in operations.value if field(n, "id").value == key), operations)
    if fields:
        for name in fields:
            node = field(node, name) or node
        return {**identity, "declaration": key, "line": node.start_mark.line + 1}
    signature = field(node, "signature") if isinstance(node, yaml.MappingNode) else None
    rank_node = field(signature or node, "ranks") if isinstance(signature or node, yaml.MappingNode) else None
    return {
        **identity,
        "declaration": key,
        "line": node.start_mark.line + 1,
        "restriction_line": (rank_node or signature or node).start_mark.line + 1,
    }


def pipeline_refusals(selection):
    reasons = []
    if not review_accepted(selection["spec"], selection["host"], default="unreviewed"):
        reasons.append("software and host declarations must both be reviewed")
    return reasons


def readout_refusals(selection, decision):
    """Check the numerical contract of the declaration actually admitted at step 6."""
    reasons = []
    for authority, document in (("software", selection["spec"]), ("host", selection["host"])):
        key = decision[authority].get("declaration")
        row = next((r for r in document["operations"] if r["id"] == key), {})
        location = declaration_location(selection, authority, key)
        if "aten.mul.Tensor" not in row.get("ops", []) or row.get("placement") != "host":
            reasons.append(
                f"{location.get('path', '')}:{location.get('line', '')} {key}: "
                "required host aten.mul.Tensor declaration absent"
            )
        if authority == "host":
            contract = row.get("numerical_contract", {})
            semantics = contract.get("semantics", {})
            for field, expected in (
                ("arithmetic", "ieee754_binary32"),
                ("rounding", "round_to_nearest_even"),
                ("per_element", "correctly_rounded_single_operation"),
            ):
                if not review_accepted(contract, default="unreviewed") or semantics.get(field) != expected:
                    where = declaration_location(selection, authority, key, ("numerical_contract", "semantics", field))
                    reasons.append(
                        f"{where.get('path', '')}:{where.get('line', '')} {key}: "
                        f"required {field}={expected} declaration absent"
                    )
    return reasons


def provider_proof(provider, source, *, package, policy):
    """(schedule proof, None), (None, provider ValueError text), or (None, None) without a prover."""
    if not callable(getattr(provider, "admit_numeric_schedule", None)):
        return None, None
    try:
        return provider.admit_numeric_schedule(str(source), package=package, numeric_policy=policy), None
    except ValueError as exc:
        return None, str(exc)


def covering_proof(provider, module, operations, *, package, policy):
    """The provider's proof of module (its ValueError propagates) covering each operation, else ValueError."""
    proof = provider.admit_numeric_schedule(str(module), package=package, numeric_policy=policy)
    for op in operations:
        failure = provider_coverage(module, proof, op)
        if failure:
            raise ValueError(failure)
    return proof


def provider_coverage(module, proof, op) -> str | None:
    """A source-bound, proven catalog binding for this precise contraction."""
    if not isinstance(proof, dict) or proof.get("status") != "proven":
        return "selected provider coverage proof unavailable"
    if proof.get("source_sha256") != hashlib.sha256(str(module).encode()).hexdigest():
        return "selected provider coverage proof is for different source bytes"
    ordinal = list(module.walk()).index(op)
    bindings = [row for row in proof.get("bindings", ()) if row.get("source_operation_ordinal") == ordinal]
    if len(bindings) != 1:
        return "selected provider must prove exactly one binding for the proposed contraction"
    binding = bindings[0]
    kernels = [row for row in proof.get("kernels", ()) if row.get("symbol") == binding.get("symbol")]
    if len(kernels) != 1 or kernels[0].get("status") != "proven":
        return "selected provider binding lacks a proven kernel"
    from merlin.kernels.shapes import accumulator_init

    try:
        readout = device_readout(op)
    except ValueError as exc:
        return str(exc)
    kernel = kernels[0]
    schedule = kernel.get("schedule", {})
    if tuple(str(getattr(value.type, "element_type", "")) for value in op.operands[:2]) == ("i32", "i32"):
        failure = centered_operand_coverage(op, kernel)
        if failure:
            return failure
    if any(
        getattr(row.attributes.get("prov.quant.host_only"), "data", None) == "true"
        or getattr(row.attributes.get("prov.quant.numerics"), "data", None) == "float_readout"
        for row in (op, readout["operation"] if readout else op)
    ):
        return "host-only float_readout cannot feed the device"
    if readout is not None:
        if readout.get("integer_continuation"):
            # This binding replaces only the raw producer. All integer host work
            # and its final rational boundary remain the captured source.
            if (
                schedule.get("output_dtype") != "i32"
                or schedule.get("scale") != 1
                or schedule.get("relu") is not False
                or schedule.get("bias") is not False
                or type(kernel.get("full_result_bound")) is not int
                or not 0 <= kernel["full_result_bound"] <= 2**31 - 1
            ):
                return "selected provider must prove bounded from-zero raw int32 continuation readout"
            if set(binding.get("replaced_ordinals", ())) & set(
                readout["integer_continuation"]["retained_operation_ordinals"]
            ):
                return "selected provider must retain the proved integer continuation on the host"
        else:
            if (
                schedule.get("readout") != "full_i32"
                or schedule.get("requantization") != "retained_source_exact_integer"
            ):
                return "selected provider must prove full int32 readout with exact host requantization"
            for field, name in (("scale_numerator", "numerator"), ("scale_denominator", "denominator")):
                if schedule.get(field) != [v[name] for v in readout["scale_ratios"]]:
                    return "selected provider readout scale differs from the exact source contract"
    if (readout is not None and not readout.get("integer_continuation")) or accumulator_init(op) == "int32_seed":
        if kernels[0].get("exact_arithmetic") is not True:
            return "selected provider has no exact-arithmetic proof for seeded/readout semantics"
    if readout is not None and list(module.walk()).index(readout["operation"]) in binding.get("replaced_ordinals", ()):
        return "selected provider must retain the exact source readout on the host"
    return None


def centered_operand_coverage(op, kernel):
    """Authenticate the provider's byte-plane proof, never an i32-to-i8 cast.

    The selected provider proves the complete source selector/correction graph
    and emitted stream. Its enclosing source hash and binding are checked by
    provider_coverage; this validates the exact representation/readout contract.
    """
    schedule = kernel.get("schedule", {})
    refusal = "centered i32 operands require a source-bound exact byte-product/correction proof and full int32 readout"
    if not isinstance(schedule, dict):
        return refusal
    proof = schedule.get("centered_byte_contraction", {})
    if not isinstance(proof, dict):
        return refusal
    planes = proof.get("operand_planes", [])
    bounds = proof.get("operand_bounds", [])
    digests = proof.get("selector_sha256", [])
    if any(not isinstance(items, (list, tuple)) for items in (planes, bounds, digests)):
        return refusal
    low, high = proof.get("accumulator_min"), proof.get("accumulator_max")
    if (
        kernel.get("exact_arithmetic") is not True
        or kernel.get("device_readout") != "full_i32"
        or type(proof.get("version")) is not int
        or proof.get("version") != 1
        or proof.get("rule") != "closed_byte_selectors_plus_literal_integer_planes"
        or proof.get("unsigned_packing") != "xor_0x80_then_add_128"
        or proof.get("rounding") != "none"
        or proof.get("device_scales") != "unit"
        or proof.get("readout") != "full_i32"
        or proof.get("source_operand_types") != [str(v.type) for v in op.operands[:2]]
        or len(planes) != 2
        or any(type(v) is not int or v < 1 for v in planes)
        or proof.get("product_count") != math.prod(planes)
        or len(bounds) != 2
        or any(type(v) is not int or not 0 <= v <= 383 for v in bounds)
        or len(digests) != 2
        or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in digests)
        or proof.get("prefix_bound") != "abs(seed)+sum_k(sum_plane_abs(A)*sum_plane_abs(B))"
        or type(low) is not int
        or type(high) is not int
        or not -(1 << 31) <= low <= high < (1 << 31)
        or schedule.get("accumulator_min") != low
        or schedule.get("accumulator_max") != high
        or kernel.get("accumulator_min") != low
        or kernel.get("accumulator_max") != high
    ):
        return refusal
    return None


def mq_host_only(op):
    """A host-only marker is a refusal even on an unseeded native contraction."""
    return (
        getattr(op.attributes.get("prov.quant.host_only"), "data", None) == "true"
        or getattr(op.attributes.get("prov.quant.numerics"), "data", None) == "float_readout"
    )


def admission_plan(
    module, selection, *, captured=False, native=False, provider_proof=None, planned_exact_readout=False
):
    """Independently screen all six steps at each source contraction's actual ranks."""
    from merlin.kernels.shapes import accumulator_init, observe_contractions, zero_initialised
    from merlin.llvmlower.contraction_quantization import closed_float_mac
    from merlin.targetgen.core_aten_readout import host_readout_bridge, planned_readout_geometry
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    spec, host = selection["spec"], selection["host"]
    quantization_errors = []
    for operation in module.walk():
        try:
            validate_quantization_attributes(operation)
        except ValueError as exc:
            quantization_errors.append(str(exc))
    ordinals = {op: i for i, op in enumerate(module.walk())}
    sites = []
    for op, shape in observe_contractions(module):
        rank = len(op.results[0].type.get_shape())
        steps, reasons = [], pipeline_refusals(selection) + quantization_errors
        # Exact integer capture retains the source quantize boundary on the host.
        # The device only returns full int32 tiles; no cast or product can round.
        try:
            readout = device_readout(op) if (captured or native) else None
        except ValueError as exc:
            readout = None
            reasons.append(str(exc))
        if captured and not native and readout is None:
            reasons.append("captured floating readout lacks a full-domain exact-arithmetic proof")
        if mq_host_only(op):
            reasons.append("host-only float_readout cannot feed the device")
        seeded = accumulator_init(op) == "int32_seed"
        centered = shape.dtypes == ("i32", "i32", "i32") and (captured or native)
        if seeded or readout is not None or centered:
            failure = provider_coverage(module, provider_proof, op)
            if failure:
                reasons.append(failure)
        geometry = planned_readout_geometry(op) if not native and readout is None else None
        readout_proof = None
        readout_rank = rank
        if geometry is not None:
            if captured:
                try:
                    _, _, readout_proof = host_readout_bridge(op)
                    readout_rank = readout_proof["cast_rank"]
                except ValueError as exc:
                    reasons.append(str(exc))
            else:
                readout_rank = len(geometry["flat_shape"])
                readout_proof = {
                    "status": "planned_static_geometry_requires_capture_proof",
                    "cast_rank": readout_rank,
                    **geometry,
                }

        def screen(step, name, signature, placement="host"):
            row = {"step": step, "op": name, "signature": signature}
            for authority, document in (("software", spec), ("host", host)):
                if placement in {"accelerator", "fused_accelerator"} and authority == "host":
                    continue
                if name == "contraction":
                    from merlin.targetgen.software_spec import admit_contraction_operation

                    source = getattr(op.attributes.get("prov.aten"), "data", name)
                    decision = admit_contraction_operation(document, source, signature, placement)
                    row["source_operator"] = source
                else:
                    decision = admit_operation(document, name, signature, placement)
                if decision.get("declaration"):
                    decision["source"] = declaration_location(selection, authority, decision["declaration"])
                row[authority] = decision
                if decision["status"] != "admitted":
                    where = decision.get("source", {})
                    reasons.append(
                        f"step {step} {name} rank {signature['rank']} {authority}: "
                        f"{where.get('path', '')}:{where.get('restriction_line', '')} "
                        f"{decision.get('declaration', '')}: {decision['reason']}"
                    )
            steps.append(row)

        if not native:
            for src in op.operands[:2]:
                screen(
                    1,
                    QUANTIZE,
                    host_signature(
                        QUANTIZE, len(src.type.get_shape()), {"zero_point": 0, "quant_min": -128, "quant_max": 127}
                    ),
                )
        transfers = spec.get("transfer_contracts", {})
        crossings = [
            (2, "operand_load", "host", "accelerator", "int8", [len(v.type.get_shape()) for v in op.operands[:2]]),
            (4, "accumulator_readout", "accelerator", "host", "i32", [rank]),
        ]
        if seeded:
            crossings.insert(1, (2, "accumulator_seed_load", "host", "accelerator", "i32", [rank]))
        for step, key, source, destination, dtype, ranks in crossings:
            from merlin.targetgen.transfer_contracts import screen_transfer_contract

            transfer = next((r for r in transfers.get("declarations", []) if r["id"] == key), {})
            decision = screen_transfer_contract(
                spec,
                source_placement=source,
                destination_placement=destination,
                operand_dtype=dtype,
                result_dtype=dtype,
                operand_layout="row_major_contiguous",
                result_layout="row_major_contiguous",
            )
            admitted = decision["status"] == "admitted" and key in decision["matching_declarations"]
            if step == 4:
                admitted &= transfer.get("semantics", {}).get("valid_window") == "required"
            row = {
                "step": step,
                "declaration": f"transfer_contracts.{key}",
                "status": "admitted" if admitted else "refused",
                "ranks": ranks,
                "constraints": transfer,
            }
            identity = selection.get("identities", {}).get("software_spec")
            if identity:
                import yaml

                root = yaml.compose(Path(identity["path"]).read_text())
                node = next(v for k, v in root.value if k.value == "transfer_contracts")
                node = next(v for k, v in node.value if k.value == key)
                row["source"] = {**identity, "line": node.start_mark.line + 1}
            steps.append(row)
            if not admitted:
                reasons.append(f"step {step} {row['declaration']} refused at {row.get('source')}")
        signature = dict(
            family="contraction",
            ordered_operand_dtypes=["i8", "i8", "i32"],
            ordered_result_dtypes=["i32"],
            operand_dtype="i8",
            accum_dtype="i32",
            readout_dtype="i32",
            rank=rank,
            layout="row_major_contiguous",
            tails="zero_pad_valid_window",
            broadcasting="independent_batches" if rank > 2 else "none",
            aliasing="disjoint_inputs_outputs",
        )
        screen(3, "contraction", signature, "accelerator")
        cast_obligation = None
        # A recapture explicitly requesting integer_readout has no i32->f32
        # cast. This provisional plan must still be requalified against the
        # actual captured source and provider proof before any device routing.
        planned_integer_readout = planned_exact_readout and not captured and not native
        if readout is not None or planned_integer_readout:
            policy = (spec.get("numerical_semantics") or {}).get("readout") or {}
            declared = host.get("exact_readout") or {}
            admitted = (
                policy.get("strategy") == "exact_rational_host"
                and review_accepted(declared, default="unreviewed")
                and declared.get("api") == "merlin.runtime.exact_readout.requantize_int32"
                and declared.get("rounding") == "half_to_even"
            )
            steps.append(
                dict(
                    step=5,
                    op="exact_rational_host_readout",
                    placement="host",
                    status="admitted" if admitted else "refused",
                    device_readout={key: value for key, value in (readout or {}).items() if key != "operation"},
                    qualification="pending_integer_capture_and_provider_proof"
                    if planned_integer_readout
                    else "captured",
                    rank=readout_rank,
                    source=declared,
                )
            )
            if not admitted:
                reasons.append("exact rational host readout requires reviewed API selection and exact host policy")
            # The device returns the full accumulator. The retained source
            # readout owns coefficient signs and activation order; no device
            # ReLU or floating scale replaces that host expression.
        if not native and readout is None and not planned_integer_readout:
            screen(
                5,
                "aten._to_copy.default",
                dict(
                    family="elementwise_map",
                    ordered_operand_dtypes=["i32", "f32"],
                    ordered_result_dtypes=["f32"],
                    operand_dtype="i32",
                    rank=readout_rank,
                ),
            )
            cast_decision = steps[-1]["host"]
            cast_key = cast_decision.get("declaration", "host_i32_to_f32_cast")
            cast_declaration = next((r for r in host["operations"] if r["id"] == cast_key), {})
            contract = cast_declaration.get("numerical_contract", {})
            exact_range = contract.get("semantics", {}).get("exact_range", "")
            location = declaration_location(
                selection, "host", cast_key, ("numerical_contract", "semantics", "exact_range")
            )
            prefix, separator, exponent = exact_range.partition("|x| <= 2^")
            exact_limit = 2 ** int(exponent) if separator and not prefix and exponent.isdecimal() else None
            operand_ranges = [row["signature"]["quantization_parameters"] for row in steps if row["step"] == 1]
            cast_bound = math.prod(shape.reduction) * math.prod(
                max(abs(r["quant_min"]), abs(r["quant_max"])) for r in operand_ranges
            )
            cast_obligation = dict(
                full_k=math.prod(shape.reduction),
                operand_ranges=operand_ranges,
                max_abs_bound=cast_bound,
                exact_limit=exact_limit,
                source=location,
            )
            if not review_accepted(contract, default="unreviewed") or exact_limit is None or cast_bound > exact_limit:
                reasons.append(
                    f"step 5 {location.get('path', '')}:{location.get('line', '')} {cast_key}: "
                    f"accumulator bound {cast_bound} exceeds or lacks declared exact_range {exact_range!r}"
                )
            screen(
                6,
                "aten.mul.Tensor",
                dict(
                    family="elementwise_map",
                    ordered_operand_dtypes=["f32", "f32", "f32"],
                    ordered_result_dtypes=["f32"],
                    operand_dtype="f32",
                    rank=readout_rank,
                ),
            )
            steps[-1]["readout_multiply"] = 1
            if all(steps[-1][a]["status"] == "admitted" for a in ("software", "host")):
                reasons.extend(readout_refusals(selection, steps[-1]))
            screen(
                6,
                "aten.mul.Tensor",
                dict(
                    family="elementwise_map",
                    ordered_operand_dtypes=["f32", "f32", "f32"],
                    ordered_result_dtypes=["f32"],
                    operand_dtype="f32",
                    rank=readout_rank,
                ),
            )
            steps[-1]["readout_multiply"] = 2
            if all(steps[-1][a]["status"] == "admitted" for a in ("software", "host")):
                reasons.extend(readout_refusals(selection, steps[-1]))
        expected = ("i8", "i8", "i32") if captured or native else ("f32", "f32", "f32")
        if shape.dtypes != expected and not centered:
            reasons.append(f"source dtype {shape.dtypes} is not {expected}")
        closed = (
            (closed_centered_integer_mac(op) if centered else closed_integer_mac(op))
            if captured or native
            else closed_float_mac(op)
        )
        if not closed or not (zero_initialised(op) or (seeded and readout is not None)):
            reasons.append(
                "source is not a closed unflagged MAC from zero (or from an int32 seed it reads out on device)"
            )
        try:
            bound = integer_split_k_limit(spec, tile_dim=1, full_k=math.prod(shape.reduction))
            if bound is None:
                reasons.append("no declared internal partial-sum arithmetic")
        except (ValueError, IndexError) as exc:
            bound = None
            reasons.append(str(exc))
        sites.append(
            dict(
                source_operation_ordinal=ordinals[op],
                source_operation=op.name,
                source_dtypes=list(shape.dtypes),
                result_rank=rank,
                host_readout_bridge=readout_proof,
                readout=(
                    "int32_host_exact"
                    if readout is not None
                    else "int32_host_exact_pending_capture"
                    if planned_integer_readout
                    else "host"
                ),
                accumulator_init="int32_seed" if seeded else "zero",
                steps=sorted(steps, key=lambda s: s["step"]),
                partial_sum_obligation=bound,
                cast_exact_range_obligation=cast_obligation,
                status="refused" if reasons else "admitted",
                reasons=reasons,
            )
        )
    quantizers = []
    if captured:
        from merlin.common import mlir_query as mq

        for op in module.walk():
            if mq.op_name(op) != "quant_ext.quantize_per_tensor":
                continue
            from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

            parameters = {
                "zero_point": _constant_of(op.operands[2]),
                "quant_min": op.properties["quant_min"].value.data,
                "quant_max": op.properties["quant_max"].value.data,
            }
            signature = host_signature(QUANTIZE, len(op.operands[0].type.get_shape()), parameters)
            signature["ordered_operand_dtypes"] = [str(v.type.element_type) for v in op.operands]
            signature["ordered_result_dtypes"] = [str(op.results[0].type.element_type)]
            row = {"source_operation_ordinal": ordinals[op], "source_operation": mq.op_name(op), "signature": signature}
            for authority, document in (("software", spec), ("host", host)):
                decision = admit_operation(document, QUANTIZE, signature, "host")
                if decision.get("declaration"):
                    decision["source"] = declaration_location(selection, authority, decision["declaration"])
                row[authority] = decision
                if decision["status"] != "admitted":
                    location = decision.get("source", {})
                    for site in sites:
                        site["status"] = "refused"
                        site["reasons"].append(
                            f"step 1 captured quantizer {authority}: {location.get('path', '')}:"
                            f"{location.get('restriction_line', '')} {decision['reason']}"
                        )
            quantizers.append(row)
    return dict(
        sites=sites,
        # Elementwise, reduction and movement sites, screened by the same authorities; informational
        # for the contraction pipeline above, whose status they do not change.
        non_contraction_sites=non_contraction_sites(module, selection, contract=selection.get("capability_contract")),
        quantization_sites=quantizers,
        quantized_contractions=0,
        scales=[],
        status="admitted"
        if any(s["status"] == "admitted" for s in sites)
        else "refused"
        if sites
        else "not_applicable",
        calibration=dict(
            mode="static_per_tensor",
            source="declared_observer_domain",
            disclosure="Static observer ranges are declared by the numerical contract, independently of evaluation payloads.",
            performed=False,
        ),
        integer_reference=dict(status="not_run"),
    )


def host_readout_chain(contraction, *, with_bridge=False):
    """Require the exact captured cast/readout chain, including any proved views."""
    from merlin.targetgen.core_aten_readout import host_readout_bridge

    bridge = host_readout_bridge(contraction)
    return bridge if with_bridge else bridge[0]


def source_declaration_continuity(initial, captured):
    """Refuse loss of exact frontend obligations through an overload rewrite."""
    from collections import Counter

    initial_steps = [step for site in initial["sites"] for step in site["steps"] if step["step"] == 3]
    required = Counter(
        step["source_operator"] for step in initial_steps if step["software"].get("applicable_exact_source_declaration")
    )
    observed = Counter(
        step.get("source_operator") for site in captured["sites"] for step in site["steps"] if step["step"] == 3
    )
    missing = required - observed
    return {
        "status": "refused" if missing else "preserved",
        "initial_source_admissions": initial_steps,
        "captured_source_operator_counts": dict(observed),
        "unpreserved_exact_source_operators": dict(missing),
    }


def device_readout(op) -> dict | None:
    """The fused int8 readout that consumes ``op``'s accumulator, or None.

    The frontend's ``integer_readout`` numerics emit it as the accumulator's ONLY user: a parallel
    ``linalg.generic`` tagged ``prov.role = "readout"`` with an i8 result, its scale, zero point, range
    and ReLU carried as ``prov.quant.*`` attributes. Read from those attributes, never from a name; an
    absent or malformed one means there is no readout this plan can state.
    """
    from merlin.common import mlir_query as mq

    validate_quantization_attributes(op, "contraction")

    result = op.results[0] if len(op.results) == 1 else None
    users = list(result.uses) if result is not None else []
    reader = users[0].operation if len(users) == 1 else None
    continuation = None
    if reader is None or mq.op_name(reader) != "linalg.generic" or mq.attr_str(reader, "prov.role") != "readout":
        from merlin.targetgen.integer_continuation import readout_continuation

        found = readout_continuation(op) if result is not None else None
        if found is None:
            return None
        reader, continuation = found
    validate_quantization_attributes(reader, "integer_readout")
    if mq.op_name(reader) != "linalg.generic" or mq.attr_str(reader, "prov.role") != "readout":
        return None
    output_dtype = str(getattr(reader.results[0].type, "element_type", ""))
    if output_dtype not in {"i8", "f32"}:
        return None
    # Both boundaries execute on the host. The device only computes the closed
    # integer MAC and returns all 32 bits; an exact f32 result rounds once there.
    attributes = reader.attributes

    def number(key):
        value = attributes.get(key)
        data = getattr(getattr(value, "value", None), "data", None)
        return data if isinstance(data, (int, float)) and not isinstance(data, bool) else None

    scale = attributes.get("prov.quant.scale")
    if scale is not None and hasattr(scale, "data") and isinstance(scale.data, tuple):
        scales = [getattr(getattr(item, "value", None), "data", None) for item in scale.data]
    else:
        scales = [number("prov.quant.scale")]
    from fractions import Fraction

    ratios = []
    if "prov.quant.scale_numerator" in attributes or "prov.quant.scale_denominator" in attributes:
        try:
            numerators = [int(item.data) for item in attributes["prov.quant.scale_numerator"].data]
            denominators = [int(item.data) for item in attributes["prov.quant.scale_denominator"].data]
            if not numerators or any(d <= 0 for d in denominators):
                raise ValueError("nonpositive denominator or empty scale")
            ratios = [Fraction(n, d) for n, d in zip(numerators, denominators, strict=True)]
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("readout requires paired integer scale numerators and positive denominators") from exc
        scales = []
    if not ratios:
        raise ValueError("device readout requires exact scale_numerator/scale_denominator")
    if mq.attr_str(reader, "prov.quant.host_only") == "true" or mq.attr_str(op, "prov.quant.host_only") == "true":
        raise ValueError("host-only quantization cannot feed the device")
    if mq.attr_str(reader, "prov.quant.numerics") != "integer_readout":
        raise ValueError("device readout requires integer_readout numerics")
    parameters = {key: number(f"prov.quant.{key}") for key in ("zero_point", "quant_min", "quant_max")}
    relu = mq.attr_str(reader, "prov.quant.relu")
    if output_dtype == "f32":
        if mq.attr_str(reader, "prov.quant.arithmetic") != "exact":
            raise ValueError("floating host readout requires exact rational arithmetic")
        if any(value is not None for value in parameters.values()):
            raise ValueError("floating host readout cannot carry integer quantization bounds")
        parameters = {}
        relu = relu or "false"
    if (
        (not scales and not ratios)
        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in scales)
        or any(type(value) is not int for value in parameters.values())
        or relu not in {"true", "false"}
    ):
        raise ValueError("readout requires finite scales and canonical integer zero_point/quant_min/quant_max")
    storage_dtype = mq.attr_str(reader, "prov.quant.storage_dtype")
    storage_min, storage_max = (0, 255) if storage_dtype == "uint8" else (-128, 127)
    if (
        output_dtype == "i8"
        and not storage_min
        <= parameters["quant_min"]
        <= parameters["zero_point"]
        <= parameters["quant_max"]
        <= storage_max
    ):
        raise ValueError("readout quantization bounds do not fit declared byte storage")
    if storage_dtype not in ({None, "", "int8", "uint8"} if output_dtype == "i8" else {None, "", "float32"}):
        raise ValueError("readout storage dtype does not match its result")
    count = len(ratios or scales)
    granularity = "tensor" if count == 1 else "channel"
    if mq.attr_str(reader, "prov.quant.granularity") not in {None, "", granularity}:
        raise ValueError("readout granularity contradicts its scales")
    if count > 1 and count != reader.results[0].type.get_shape()[-1]:
        raise ValueError("readout channel scales do not match its trailing dimension")
    rounding = mq.attr_str(reader, "prov.quant.rounding")
    if rounding not in {None, "", "half_to_even"}:
        raise ValueError(f"unsupported readout rounding {rounding!r}")
    for key, expected in (("saturate", "true"), ("accumulator", "i32")):
        value = mq.attr_str(reader, f"prov.quant.{key}")
        if value not in {None, "", expected}:
            raise ValueError(f"unsupported readout {key} {value!r}")
    arithmetic = mq.attr_str(reader, "prov.quant.arithmetic")
    if arithmetic not in {None, "", "exact"}:
        raise ValueError(f"unsupported readout arithmetic {arithmetic!r}")
    exact_parameters = {}
    for key in ("lhs_scale", "rhs_scale", "output_scale", "alpha", "beta"):
        value = attributes.get(f"prov.quant.{key}")
        if value is None:
            continue
        values = list(value.data) if hasattr(value, "data") and isinstance(value.data, tuple) else [value]
        numbers = [getattr(getattr(item, "value", None), "data", None) for item in values]
        if any(not isinstance(n, (int, float)) or not math.isfinite(n) for n in numbers):
            raise ValueError(f"readout {key} requires finite numeric parameters")
        exact_parameters[key] = numbers
    return {
        "output_dtype": output_dtype,
        "storage_dtype": storage_dtype or ("int8" if output_dtype == "i8" else "float32"),
        "scales": [float(v) for v in scales],
        "scale_ratios": [{"numerator": str(v.numerator), "denominator": str(v.denominator)} for v in ratios],
        "scale_signs": [(v > 0) - (v < 0) for v in (ratios or scales)],
        "granularity": granularity,
        **{key: int(value) for key, value in parameters.items()},
        "relu": relu == "true",
        "arithmetic": arithmetic or "unspecified",
        "exact_parameters": exact_parameters,
        "operation": reader,
        **({"integer_continuation": continuation} if continuation is not None else {}),
    }


def closed_centered_integer_mac(op):
    """A closed i32 MAC is structural only; its byte lowering needs provider proof."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import i32
    from xdsl.dialects.linalg.ops import YieldOp

    if len(op.operands) != 3 or len(op.results) != 1 or not hasattr(op, "body"):
        return False
    block = op.body.block
    if tuple(arg.type for arg in block.args) != (i32, i32, i32) or len(block.ops) != 3:
        return False
    mul, add, yielded = list(block.ops)
    return (
        all(getattr(v.type, "element_type", None) == i32 for v in (*op.operands, *op.results))
        and isinstance(mul, arith.MuliOp)
        and not mul.overflow_flags.data
        and tuple(mul.operands) == tuple(block.args[:2])
        and isinstance(add, arith.AddiOp)
        and not add.overflow_flags.data
        and tuple(add.operands) in ((mul.result, block.args[2]), (block.args[2], mul.result))
        and isinstance(yielded, YieldOp)
        and tuple(yielded.operands) == (add.result,)
    )


def closed_integer_mac(op):
    """Only signed i8 extensions and a closed, unflagged i32 multiply/add."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import i8, i32
    from xdsl.dialects.linalg.ops import YieldOp

    if len(op.operands) != 3 or len(op.results) != 1 or not hasattr(op, "body"):
        return False
    block = op.body.block
    if len(block.args) != 3 or len(block.ops) != 5:
        return False
    a, b, mul, add, yielded = list(block.ops)
    return (
        tuple(arg.type for arg in block.args) == (i8, i8, i32)
        and all(getattr(v.type, "element_type", None) == i8 for v in op.operands[:2])
        and all(getattr(v.type, "element_type", None) == i32 for v in (*op.operands[2:], *op.results))
        and isinstance(a, arith.ExtSIOp)
        and isinstance(b, arith.ExtSIOp)
        and a.result.type == b.result.type == i32
        and tuple(a.operands) == (block.args[0],)
        and tuple(b.operands) == (block.args[1],)
        and isinstance(mul, arith.MuliOp)
        and not mul.overflow_flags.data
        and tuple(mul.operands) == (a.result, b.result)
        and isinstance(add, arith.AddiOp)
        and not add.overflow_flags.data
        and tuple(add.operands) in ((mul.result, block.args[2]), (block.args[2], mul.result))
        and isinstance(yielded, YieldOp)
        and tuple(yielded.operands) == (add.result,)
    )


#: Carriers that are views or scalar plumbing, not independent device work (C52: views stay host
#: metadata and are never materialised), so they are not screened as sites.
_NON_SITE_CARRIERS = frozenset(
    {
        "tensor.empty",
        "tensor.extract_slice",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.reshape",
        "tensor.cast",
        "tensor.dim",
        "tensor.extract",
        "tensor.yield",
        "linalg.yield",
        "linalg.index",
    }
)
#: Tensor-dialect carriers whose work is data motion.
_TENSOR_MOVEMENT = frozenset({"tensor.concat", "tensor.insert_slice", "tensor.pad", "tensor.splat"})


def _site_family(op) -> str | None:
    """The semantic family of a top-level non-contraction site, structurally; None when not a site."""
    from merlin.common import mlir_query as mq
    from merlin.targetgen import semantic_families as sf
    from merlin.xdsl_dialects.lowering import contraction_coverage as cc

    name = mq.op_name(op)
    if name in _NON_SITE_CARRIERS:
        return None
    if name == "linalg.generic":
        return {
            "sum-reduction": "reduction",
            "max-reduction": "reduction",
            "absmax": "reduction",
            "other-reduction": "reduction",
            "movement": "movement",
            "elementwise": "elementwise_map",
        }.get(cc.classify_generic(op))
    if name == "linalg.reduce":
        return "reduction"
    if name in _TENSOR_MOVEMENT:
        return "movement"
    if name.startswith("linalg."):
        family = sf.from_op(name.rpartition(".")[2])
        return family if family in {"movement", "elementwise_map", "reduction"} else None
    return None


_DEQUANTIZE_OPS = frozenset({"quant_ext.dequantize_per_tensor"})
_QUANTIZE_OPS = frozenset({"quant_ext.quantize_per_tensor"})
#: Value producers that carry no data from a host tensor: a constant, a splat of one, an empty init.
_CONSTANT_PRODUCERS = frozenset({"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"})


def quantized_islands(module) -> set[int]:
    """``id``s of the floating-point operations that compute only between int8 quantization edges.

    PT2E leaves a quantized non-contraction operation as ``dequantize -> f32 op -> quantize``. Such an
    op computes on int8 codes and its result is read only as int8: it is int8 work whose floating
    spelling is the frontend's, exactly as a backend that matches the sandwich treats it. An op belongs
    to an island when every tensor operand comes from a dequantize, a constant producer or another
    island op, and every user is a quantize or another island op (a fixpoint, so chains -- a window
    sum followed by its division, the slices of a pad -- form one island).
    """
    from merlin.common import mlir_query as mq

    operations = [op for op in module.walk() if mq.op_name(op) not in {"builtin.module", "func.func", "func.return"}]

    def floating(op) -> bool:
        return any("f32" in str(getattr(result.type, "element_type", "")) for result in op.results)

    def ranked(op) -> bool:
        return any(list(getattr(result.type, "get_shape", lambda: [])()) for result in op.results)

    candidates = {
        id(op): op
        for op in operations
        if floating(op)
        and mq.op_name(op) not in _DEQUANTIZE_OPS | _QUANTIZE_OPS | {"tensor.empty", "arith.constant"}
        # A rank > 0 fill is the work of fill/full/full_like; a scalar one is a quantization parameter.
        and (mq.op_name(op) not in _CONSTANT_PRODUCERS or ranked(op))
        and not _nested_in_linalg(op)
    }
    changed = True
    while changed:
        changed = False
        for key, op in list(candidates.items()):
            sourced = all(
                (owner := getattr(value, "owner", None)) is not None
                and hasattr(owner, "results")
                and (
                    mq.op_name(owner) in _DEQUANTIZE_OPS
                    or mq.op_name(owner) in _CONSTANT_PRODUCERS
                    or id(owner) in candidates
                )
                for value in op.operands
                if hasattr(value.type, "get_shape") or str(value.type) == "f32"
            )
            consumed = all(
                mq.op_name(use.operation) in _QUANTIZE_OPS or id(use.operation) in candidates
                for result in op.results
                for use in result.uses
            ) and any(result.uses for result in op.results)
            if not (sourced and consumed):
                del candidates[key]
                changed = True
    return set(candidates)


def _dense_tensor(tensor_type) -> bool:
    """A ranked tensor type with no layout encoding (the default dense row-major layout)."""
    from xdsl.dialects.builtin import NoneAttr

    encoding = getattr(tensor_type, "encoding", None)
    return encoding is None or isinstance(encoding, NoneAttr)


def _nested_in_linalg(op) -> bool:
    parent = op.parent_op()
    while parent is not None:
        if getattr(parent, "name", "").startswith("linalg."):
            return True
        parent = parent.parent_op()
    return False


#: Floating element types a prospective screen reads as the int8 work the static recipe makes of them.
_PROSPECTIVE_FLOATS = ("f32", "f64", "f16", "bf16")


def non_contraction_sites(module, selection, *, contract=None, prospective=False):
    """Screen every top-level non-contraction site of an integer program for device placement.

    A site is one elementwise, reduction or movement operation outside the contractions
    :func:`admission_plan` screens. Each passes through the same authorities a contraction does: the
    operand-load and readout transfer contracts, SW :func:`admit_operation` (standalone, and fused
    behind a contraction producer when one feeds it) and, given the selected capability contract,
    HW :func:`eligibility.is_eligible`. Nothing here selects a backend or a lowering: an admitted site
    is a placement candidate only. A float site is screened like any other and refused by precision,
    which is the honest answer until a recipe quantizes it.

    ``prospective`` screens a FLOATING source program as the int8 program its static recipe would
    make of it (every floating site as int8 work), the way a contraction is screened prospectively
    before its recapture; it decides whether a recapture is worth making, never what is placed.
    """
    from merlin.common import mlir_query as mq
    from merlin.kernels.shapes import observe_contractions
    from merlin.targetgen import access_observations
    from merlin.targetgen import semantic_families as sf
    from merlin.targetgen.eligibility import (
        RegionDescriptor,
        capability_map_from_contract,
        is_eligible,
        undetermined_families_from_contract,
    )
    from merlin.targetgen.transfer_contracts import screen_transfer_contract
    from merlin.xdsl_dialects.lowering import contraction_coverage as cc

    spec = selection["spec"]
    capability_map = capability_map_from_contract(contract) if contract is not None else None
    undetermined = undetermined_families_from_contract(contract) if contract is not None else None
    contractions = {id(op) for op, _shape in observe_contractions(module)}
    ordinals = {id(op): i for i, op in enumerate(module.walk())}
    islands = quantized_islands(module)
    quantization_errors = []
    for operation in module.walk():
        try:
            validate_quantization_attributes(operation)
        except ValueError as exc:
            quantization_errors.append(str(exc))

    def precision(value) -> str | None:
        dtype = mq.type_shape_dtype(value.type)[1]
        return dtype or None

    def nested(op) -> bool:
        parent = op.parent_op()
        while parent is not None:
            if mq.op_name(parent).startswith("linalg."):
                return True
            parent = parent.parent_op()
        return False

    def produced_by_contraction(op) -> bool:
        return any(id(getattr(value, "owner", None)) in contractions for value in op.operands)

    sites = []

    def quantization_parameter(op) -> bool:
        """A scalar or a value read only by quantize/dequantize ops: a scale or zero point, not work."""
        results = list(op.results)
        if (
            mq.op_name(op) in {"arith.constant", "tensor.splat"}
            and results
            and all(not list(getattr(r.type, "get_shape", lambda: [0])()) for r in results)
        ):
            return True
        users = [use.operation for result in results for use in result.uses]
        return (
            bool(users)
            and all(mq.op_name(user) in _DEQUANTIZE_OPS | _QUANTIZE_OPS for user in users)
            and all(use.index != 0 for result in results for use in result.uses)
        )

    for op in module.walk():
        if id(op) in contractions or nested(op) or quantization_parameter(op):
            continue
        family = _site_family(op)
        if family is None:
            continue
        name = mq.op_name(op)
        tensors = [value for value in op.operands if hasattr(value.type, "get_shape")]
        results = list(op.results)
        operand_dtypes = [precision(value) for value in op.operands]
        result_dtypes = [precision(value) for value in results]
        rank = len(results[0].type.get_shape()) if results and hasattr(results[0].type, "get_shape") else None
        source = mq.attr_str(op, "prov.aten") or name
        observations = access_observations.observe(op, family)
        if name.startswith("tensor.") and all(
            _dense_tensor(value.type) for value in [*op.operands, *op.results] if hasattr(value.type, "get_shape")
        ):
            # Tensor-dialect values are immutable SSA with the default dense row-major layout.
            observations = {**observations, "layout": "row_major_contiguous", "aliasing": "disjoint_inputs_outputs"}
        form = sf.carrier_form(op) if family == "movement" else None
        kind = cc.reduction_kind(op) if family == "reduction" else None
        in_dtype = precision(tensors[0]) if tensors else (operand_dtypes[0] if operand_dtypes else None)
        out_dtype = result_dtypes[0] if result_dtypes else None
        # Prospectively, a float64/float16 result is int8 work too: the recipe spells a float dtype
        # conversion or a fill at float32 and reads its result out at the requested width
        # (_recipe_quantizer.read_out_floating_widths).
        grid = _PROSPECTIVE_FLOATS if prospective else ("f32",)
        quantized = id(op) in islands or (prospective and any(dtype in grid for dtype in result_dtypes))
        if quantized:
            # Int8 work in the frontend's floating spelling (dequantize -> op -> quantize): its codes are
            # int8 on both edges, so it is screened as the int8 operation the island computes.
            operand_dtypes = ["i8" if dtype in grid else dtype for dtype in operand_dtypes]
            result_dtypes = ["i8" if dtype in grid else dtype for dtype in result_dtypes]
            in_dtype = "i8" if in_dtype is None or in_dtype in grid else in_dtype
            out_dtype = "i8" if out_dtype in grid else out_dtype
        signature = {
            "family": family,
            "operand_dtype": in_dtype,
            "ordered_operand_dtypes": operand_dtypes,
            "ordered_result_dtypes": result_dtypes,
            "rank": rank,
            # The pipeline's own calibration policy is static per-tensor (admission_plan).
            "scale_granularity": "tensor",
            **{key: value for key, value in observations.items() if value is not None},
        }
        if family == "elementwise_map":
            signature.update(access_observations.primitive_signature(op))
        elif family == "movement":
            signature["broadcasting"] = (
                "outer"
                if name == "tensor.splat" and form == "broadcast"
                else access_observations.primitive_signature(op)["broadcasting"]
                if form == "broadcast"
                else "none"
            )
        if form is not None:
            signature["form"] = form
        if kind is not None:
            signature["kind"] = kind
        steps, reasons = [], list(pipeline_refusals(selection)) + quantization_errors
        for step, key, origin, destination, dtype in (
            (2, "operand_load", "host", "accelerator", in_dtype),
            (4, "readout", "accelerator", "host", out_dtype),
        ):
            decision = screen_transfer_contract(
                spec,
                source_placement=origin,
                destination_placement=destination,
                operand_dtype=dtype,
                result_dtype=dtype,
                operand_layout="row_major_contiguous",
                result_layout="row_major_contiguous",
            )
            steps.append({"step": step, "transfer": key, "dtype": dtype, **decision})
        placements = [("accelerator", signature)]
        if produced_by_contraction(op) and family == "elementwise_map":
            placements.append(("fused_accelerator", {**signature, "composed_with": ["contraction"]}))
        software = [admit_operation(spec, source, observed, placement) for placement, observed in placements]
        index = next(
            (i for i, d in enumerate(software) if d["status"] == "admitted"),
            next((i for i, d in enumerate(software) if d["status"] == "unknown"), 0),
        )
        chosen = software[index]
        fused = ("contraction",) if placements[index][0] == "fused_accelerator" else None
        if chosen.get("declaration"):
            chosen = {**chosen, "source": declaration_location(selection, "software", chosen["declaration"])}
        steps.append({"step": 3, "software": chosen, "placement": placements[index][0]})
        hardware = None
        if capability_map is not None:
            verdict = is_eligible(
                RegionDescriptor(
                    op=name.rpartition(".")[2],
                    family=family,
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    rank=rank,
                    layout=signature.get("layout"),
                    form=form,
                    kind=kind,
                    scale_granularity=None,
                ),
                capability_map,
                undetermined=undetermined,
                fused_with=fused,
            )
            hardware = {
                "status": "admitted" if verdict.eligible else "unknown" if verdict.undetermined else "unsupported",
                "reason": verdict.reason,
                "refusal": verdict.refusal,
            }
            steps.append({"step": 3, "hardware": hardware})
        statuses = [row["status"] for row in steps if "transfer" in row] + [chosen["status"]]
        statuses.append(hardware["status"] if hardware is not None else "unknown")
        for row in steps:
            if "transfer" in row and row["status"] not in {"admitted", "not_applicable"}:
                reasons.append(
                    f"step {row['step']} {row['transfer']} {row['dtype']}: {row.get('reason', row['status'])}"
                )
        if chosen["status"] != "admitted":
            reasons.append(f"step 3 {source} software {chosen.get('declaration', '')}: {chosen['reason']}")
        if hardware is None:
            reasons.append("step 3 hardware: no selected capability contract")
        elif hardware["status"] != "admitted":
            reasons.append(f"step 3 hardware: {hardware['reason']}")
        status = (
            "admitted"
            if not reasons
            else "refused"
            if any(value == "unsupported" for value in statuses) or pipeline_refusals(selection)
            else "unknown"
        )
        sites.append(
            {
                "source_operation_ordinal": ordinals[id(op)],
                "source_operation": name,
                "source_operator": source,
                "family": family,
                "quantized_island": quantized,
                "signature": signature,
                "steps": steps,
                "status": status,
                "reasons": reasons,
                "scope": "placement screen only; no backend, lowering or numerics is selected here",
            }
        )
    return sites
