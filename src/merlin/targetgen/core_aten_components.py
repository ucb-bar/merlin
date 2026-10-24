"""Qualify exact native component sources against the selected provider."""

import functools
import hashlib
import json


class Refusal(Exception):
    """A declined candidate; its keyword fields become the refusal JSON."""

    def __init__(self, **fields):
        super().__init__(fields)


def refusing(name, **defaults):
    """Write a candidate's Refusal to directory/name (defaults first) and return None."""

    def wrap(prepare):
        @functools.wraps(prepare)
        def run(directory, *args, **kwargs):
            try:
                return prepare(directory, *args, **kwargs)
            except Refusal as refusal:
                (directory / name).write_text(json.dumps(dict(defaults, **refusal.args[0])) + "\n")
                return None

        return run

    return wrap


def require_component_identity(directory, plan):
    """Bind general component proofs to source, never to runtime tensor bytes."""
    if not plan.get("exact_dtype_sites"):
        return
    source = (directory / "model.mlir").read_text().rstrip("\n")
    proof = plan.get("schedule_proof") or plan.get("provider_catalog") or {}
    if hashlib.sha256(source.encode()).hexdigest() != proof.get("source_sha256"):
        raise ValueError("exact dtype source changed after its provider proof")
    bounds = plan.get("input_bound_proof")
    if bounds is not None and (bounds.get("method") != "full_dtype_storage_domain" or "input_sha256" in bounds):
        raise ValueError("exact components require an input-independent dtype domain")
    if bounds is not None and plan.get("native_semantic_proof", {}).get("status") != "proven":
        raise ValueError("exact components require a general native semantic proof")


def unlowered_data_reductions(module, limb_attribute):
    """A device prelude cannot stand in for any retained floating data reduction."""
    from xdsl.dialects.builtin import IndexType, IntegerType

    result = []
    for ordinal, op in enumerate(module.walk()):
        parent = op
        while parent is not None and "merlin.exact_host_arithmetic" not in parent.attributes:
            parent = parent.parent_op()
        if parent is not None:
            continue
        # Exact component readout uses integer arithmetic. A retained math
        # intrinsic is source computation, even when a conversion before it
        # happened to produce an eligible integer device region.
        if op.name.startswith("math."):
            result.append(ordinal)
            continue
        if op.name in {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "arith.cmpf",
            "arith.cmpi",
            "arith.maximumf",
            "arith.minimumf",
            "arith.maxnumf",
            "arith.minnumf",
            "arith.maxsi",
            "arith.maxui",
            "arith.minsi",
            "arith.minui",
            "arith.divsi",
            "arith.divui",
            "arith.remsi",
            "arith.remui",
            "arith.floordivsi",
        }:
            # Index predicates select addresses or representation components;
            # they are not retained numerical comparisons of tensor values.
            if op.operands and all(isinstance(value.type, IndexType) for value in op.operands):
                continue
            result.append(ordinal)
            continue
        if op.name not in {"linalg.generic", "linalg.reduce", "scf.for"} or limb_attribute in op.attributes:
            continue
        children = list(op.walk())[1:]
        reduction = (
            op.name in {"linalg.reduce", "scf.for"}
            or any(it.data.value == "reduction" for it in op.get_iterator_types())
            or any(child.name == "scf.for" for child in children)
        )
        if not reduction:
            continue
        for child in children:
            if child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"}:
                if any(getattr(value.owner, "name", None) != "arith.constant" for value in child.operands):
                    result.append(ordinal)
                    break
            if child.name != "arith.muli":
                continue
            typ = child.results[0].type
            if not isinstance(typ, IntegerType) and str(typ) not in {"f16", "bf16", "f32", "f64"}:
                continue
            if all(getattr(value.owner, "name", None) != "arith.constant" for value in child.operands):
                result.append(ordinal)
                break
    return result


@refusing("exact-dtype-refusal.json", family="F2")
def prepare_integer_components(
    directory,
    report,
    selection,
    provider,
    package,
    module,
    *,
    limb_attribute,
    construction_observer=None,
    quantized_contract=None,
):
    from merlin.llvmlower.integer_components import rewrite_integer_components
    from merlin.targetgen.core_aten_spec_int8 import provider_proof
    from merlin.targetgen.software_spec import admit_operation
    from merlin.targetgen.transfer_contracts import screen_transfer_contract

    if not callable(getattr(provider, "admit_numeric_schedule", None)):
        return None
    if any(key.startswith("prov.quant.") for op in module.walk() for key in op.attributes):
        return None
    from merlin.targetgen.native_dtype_domain import component_semantics, dtype_domain

    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        return None
    records = [record for record in report["cases"] if record.get("status") in {None, "bundled"}]
    native_types = [str(arg.type.element_type) for arg in functions[0].body.block.args]
    domains = [dtype_domain(dtype) for dtype in native_types]
    bounds = [domain["absolute_bound"] for domain in domains]
    dtype_bounds = list(bounds)
    grids = [domain["finite_grid"] for domain in domains]

    def observe(stage):
        if construction_observer is not None:
            construction_observer(module, stage)

    if quantized_contract is not None:
        from merlin.llvmlower.floating_components import rewrite_floating_components
        from merlin.llvmlower.quantized_native import finite_source, quantize_entries

        finite_source(module)
        known = quantize_entries(module, quantized_contract)
        from merlin.llvmlower.native_sign_components import rewrite_native_sign_components

        sign_bounds, sign_proofs = rewrite_native_sign_components(module, defer_floating_negation=True)
        known_bounds, typed_proofs = rewrite_floating_components(
            module,
            [None] * len(native_types),
            known_grids=known,
            allow_reassociation=True,
            quantized_contract=quantized_contract,
        )
        known_bounds.update(sign_bounds)
        bounds = [0] * len(native_types)
        literal_folds = boolean_count = 0
        semantics = dict(status="proven", method="declared_quantized_rational_family")
    else:
        from merlin.llvmlower.uniform_numeric_literals import fold_uniform_numeric_literals

        literal_folds = fold_uniform_numeric_literals(module)
        semantics = component_semantics(module)
        if semantics["status"] != "proven":
            raise Refusal(**semantics)
        from merlin.llvmlower.boolean_components import rewrite_boolean_components

        known_bounds, boolean_count = rewrite_boolean_components(module)
        observe("boolean_components")
        sign_proofs = []
        from merlin.llvmlower.native_cpu_layout import input_layouts, profile

        loop = semantics.get("cpu_loop_profile")
        profiles = (
            {
                value: profile(layout, block=loop["reduced_vector_elements"], grain=loop["parallel_grain"])
                for value, layout in input_layouts(report, functions[0].body.block.args).items()
            }
            if loop
            else {}
        )
        scalar_primitives = {
            operation.name for operation in module.walk() if operation.name.startswith(("arith.", "math."))
        }
        pure_float_copy = bool(scalar_primitives & {"arith.extf", "arith.truncf"}) and scalar_primitives <= {
            "arith.extf",
            "arith.truncf",
            "arith.constant",
        }
        floating_storage = set(native_types) & {"f16", "bf16", "f32", "f64"}
        if len(floating_storage) > 1 and not pure_float_copy:
            raise Refusal(
                code="native_mixed_storage_promotion_unproved",
                reason="mixed-storage TensorIterator temporary conversion needs a complete scalar/vector copy profile",
            )
        copy_profiles = None
        one_tensor_storage_context = bool(records) and all(
            len(record.get("input_indices", ())) == 1 for record in records
        )
        if pure_float_copy:
            if not one_tensor_storage_context:
                raise Refusal(
                    code="native_copy_layout_context_unproved",
                    reason="CPU copy profile requires one original tensor; additional destination layouts are unproved",
                )
            copy_profiles = {
                value: profile(layout, block=loop["float_copy_vector_elements"], grain=loop["parallel_grain"])
                for value, layout in input_layouts(report, functions[0].body.block.args).items()
            }
            if any(copy_profiles.get(value) is None for value in functions[0].body.block.args):
                raise Refusal(code="native_copy_layout_unproved", reason="copy CPU loop layout has no certified form")
        if semantics.get("cpu_reduced_storage_profile_required"):
            candidates = [p for p in profiles.values() if p is not None]
            output_shape = (
                list(records[0]["output_abi"][0]["shape"])
                if len(records) == 1 and records[0].get("output_abi")
                else None
            )
            selected_profile = next((p for p in candidates if p["shape"] == output_shape), None)
            from math import prod

            complete_context = all(
                0 <= index < len(functions[0].body.block.args) and functions[0].body.block.args[index] in profiles
                for record in records
                for index in record.get("input_indices", ())
            )
            if (
                not complete_context
                or selected_profile is None
                or any(
                    p is None
                    or (
                        prod(p["shape"]) != 1
                        and any(p[k] != selected_profile[k] for k in ("shape", "axes", "row", "vector_prefix"))
                    )
                    for p in profiles.values()
                )
            ):
                from merlin.llvmlower.native_cpu_layout import joint_profile

                selected_profile = (
                    joint_profile(
                        list(input_layouts(report, functions[0].body.block.args).values()),
                        output_shape,
                        block=loop["reduced_vector_elements"],
                        grain=loop["parallel_grain"],
                    )
                    if complete_context
                    else None
                )
                if selected_profile is None:
                    raise Refusal(
                        code="native_reduced_binary_layout_unproved",
                        reason=(
                            "reduced-storage scalar/vector readout needs equal dense/rank-one input"
                            " profiles or one scalar"
                        ),
                    )
            scalars = sum(
                prod(p["shape"]) == 1 or all(s == 0 for s, n in zip(p["stride"], p["shape"]) if n > 1)
                for p in profiles.values()
            )
            if scalars > 1:
                selected_profile = dict(selected_profile, vector_prefix=0)
            from xdsl.dialects.builtin import StringAttr

            for operation in module.walk():
                if operation.name == "linalg.generic":
                    operation.attributes["merlin.cpu_storage_profile"] = StringAttr(
                        json.dumps(selected_profile, sort_keys=True)
                    )
        for operation in module.walk():
            if operation.name != "linalg.generic" or len(operation.inputs) != 1:
                continue
            source = operation.inputs[0]
            if str(source.type.element_type) not in {"f16", "bf16"}:
                continue
            if any(
                child.name == "arith.negf"
                or (
                    child.name == "arith.cmpf"
                    and not (child.operands[0] is child.operands[1] and child.predicate.value.data in {7, 14})
                )
                for child in operation.body.block.ops
            ):
                if profiles.get(source) is None or not one_tensor_storage_context:
                    raise Refusal(
                        code="native_cpu_word_layout_unproved",
                        reason=(
                            "reduced storage unary CPU loop requires one original tensor and a certified "
                            "dense or rank-one layout below its parallel grain"
                        ),
                    )
        if any(
            dtype in {"i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64", "complex<f32>"} for dtype in native_types
        ):
            from merlin.llvmlower.native_sign_components import rewrite_native_sign_components

            sign_bounds, sign_proofs = rewrite_native_sign_components(
                module, canonical_bf16_nan=False, cpu_profiles=profiles
            )
            observe("native_sign_prelude")
            known_bounds.update(sign_bounds)
        typed_proofs = []
        if semantics.get("method") == "closed_typed_pointwise_IEEE_composition_v1":
            from merlin.llvmlower.fixed_point_ir import FORMAT
            from merlin.llvmlower.floating_components import rewrite_floating_components
            from merlin.llvmlower.ieee_conversion_components import rewrite_ieee_conversions
            from merlin.llvmlower.ieee_dyadic_ir import finite_dtype_grid
            from merlin.llvmlower.integer_float_boundaries import rewrite_integer_float_boundaries
            from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives
            from merlin.llvmlower.significand_products import rewrite_significand_products

            expose_pointwise_primitives(module)
            literal_folds += fold_uniform_numeric_literals(module)
            from merlin.llvmlower.integer_division_components import rewrite_integer_divisions

            added, divisions = rewrite_integer_divisions(
                module, wrapping_floor_widths=semantics.get("integer_floor_wrap_widths", ())
            )
            known_bounds.update(added)
            typed_proofs.extend(divisions)
            added, conversions = rewrite_ieee_conversions(module, copy_profiles=copy_profiles)
            known_bounds.update(added)
            typed_proofs.extend(conversions)
            added, products = rewrite_significand_products(module)
            known_bounds.update(added)
            typed_proofs.extend(products)
            conversion_grids = {}
            added, _ = rewrite_integer_float_boundaries(module, bounds, output_grids=conversion_grids)
            known_bounds.update(added)
            from merlin.llvmlower.ieee_zero_division import rewrite_zero_division

            rewrite_zero_division(module)
            from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

            for operation in module.walk():
                for value in operation.results:
                    token = str(getattr(value.type, "element_type", ""))
                    if token in FORMAT and _constant_of(value) is None:
                        # Any finite word of this declared type lies on its full
                        # dtype grid, including word selects and NaN-priority DAGs.
                        # Packing sanitizes specials; complete-source admission
                        # still requires every arithmetic producer on device.
                        conversion_grids.setdefault(value, finite_dtype_grid(token))
            try:
                added, floating_proofs = rewrite_floating_components(
                    module, grids, known_grids=conversion_grids, ieee_pointwise=True
                )
            except ValueError as exc:
                raise Refusal(code="native_typed_representation_unproved", reason=str(exc))
            known_bounds.update(added)
            typed_proofs.extend(floating_proofs)
            from xdsl.dialects.builtin import UnitAttr

            for operation in module.walk():
                if operation.name in {"arith.cmpf", "arith.cmpi"}:
                    operation.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
            observe("typed_ieee_components")
        if any(b is None for b in bounds):
            if not sign_proofs and not typed_proofs:
                raise Refusal(
                    code="native_float_representation_semantics_unproved",
                    reason="no full-domain native IEEE producer theorem for this source",
                )
            # Sign/selection producers read raw storage words. They do not use a
            # numerical grid, even for signaling NaNs, infinities and signed zeros.
            bounds = [0 if b is None else b for b in bounds]
    if any(proof.get("arithmetic") == "source_compare_select_retained_host" for proof in sign_proofs):
        # An identity component producer followed by the source threshold on
        # the host is not native coverage of that threshold's arithmetic.
        raise Refusal(
            code="native_family_arithmetic_retained_host",
            reason="device components only copy storage; source compare/select remains on host",
        )
    if semantics.get("method") == "typed_storage_and_modular_ring_v1":
        from merlin.llvmlower.integer_division_components import rewrite_integer_divisions

        added, divisions = rewrite_integer_divisions(module)
        known_bounds.update(added)
        typed_proofs.extend(divisions)
    plans = rewrite_integer_components(
        module,
        limb_attribute=limb_attribute,
        input_bounds=bounds,
        known_bounds=known_bounds,
    )
    observe("integer_components")
    if not plans:
        raise Refusal(
            code="native_component_producer_unavailable",
            reason="no complete exact integer component producer matches the source",
        )
    from merlin.llvmlower.entry_weight_projection import _pure_dce
    from merlin.llvmlower.literal_zero_components import fold_literal_zero_components

    unfolded = module.clone()
    folded_zero_producers = fold_literal_zero_components(module, limb_attribute)
    for function in functions:
        _pure_dce(function)
    if not any(limb_attribute in operation.attributes for operation in module.walk()):
        # Keep the general exact producer if literal folding would erase every
        # kernel while typed/IEEE readout still depends on runtime storage.
        module = unfolded
        functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
        for function in functions:
            _pure_dce(function)
        folded_zero_producers = 0
    observe("literal_zero_components")
    remaining_products = unlowered_data_reductions(module, limb_attribute)
    if remaining_products:
        if quantized_contract is not None:
            (directory / "quantized-native-uncovered.mlir").write_text(str(module))
        raise Refusal(
            reason="native arithmetic remains in host source; partial preludes do not place the operation",
            source_operation_ordinals=remaining_products,
        )
    for origin, destination, dtype in (("host", "accelerator", "i8"), ("accelerator", "host", "i32")):
        decision = screen_transfer_contract(
            selection["spec"],
            source_placement=origin,
            destination_placement=destination,
            operand_dtype=dtype,
            result_dtype=dtype,
            operand_layout="row_major_contiguous",
            result_layout="row_major_contiguous",
        )
        if decision["status"] != "admitted":
            raise Refusal(
                code="native_component_transfer_unadmitted",
                reason="exact component transfer has no admitted declaration",
                transfer=decision,
            )
    from merlin.xdsl_dialects._common import text

    source = text(module, generic=True) if quantized_contract is not None else str(module)
    proof, error = provider_proof(provider, source, package=package, policy=selection["spec"])
    if error is not None:
        raise Refusal(
            reason="selected provider refuses exact components: " + error,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        )
    if proof.get("status") != "proven" or proof.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest():
        reason = proof.get("reason", "component provider proof unavailable")
        raise Refusal(reason=reason, provider_proof=proof)
    sites = []
    for ordinal, op in enumerate(module.walk()):
        if limb_attribute not in op.attributes:
            continue
        binding = [
            b
            for b in proof.get("bindings", [])
            if b["source_operation_ordinal"] == ordinal and b.get("numeric", {}).get("family") in {"F2", "F4"}
        ]
        contraction = any(it.data.value == "reduction" for it in op.get_iterator_types())
        family = (
            "contraction"
            if binding and binding[0]["numeric"].get("form") == "bilinear"
            else "reduction"
            if contraction
            else "elementwise_map"
        )
        signature = dict(
            family=family,
            numerical_family="F5" if boolean_count else "F2",
            operand_dtype="i8",
            accum_dtype="i32",
            readout_dtype="i32",
            rank=len(op.results[0].type.get_shape()),
            layout="row_major_contiguous",
            aliasing="disjoint_inputs_outputs",
        )
        decision = admit_operation(selection["spec"], family, signature, "accelerator")
        host = admit_operation(selection["host"], family, signature, "host")
        if decision["status"] != "admitted" or host["status"] != "admitted" or len(binding) != 1:
            raise Refusal(
                reason="component software/host/source binding incomplete",
                software=decision,
                host=host,
                bindings=binding,
            )
        numeric = binding[0]["numeric"]
        from merlin.targetgen.native_mac_bound import require_mesh_mac_bound

        mesh = selection["spec"].get("numerical_semantics", {}).get("native_unit_contract", {}).get("mesh_mac", {})
        require_mesh_mac_bound(numeric, mesh_signed_bits=mesh.get("signed_bits"))
        if numeric.get("readout") != "full_i32" or numeric.get("recombination") != "retained_source_host_exact_integer":
            raise ValueError("component producer would replace the exact host epilogue")
        sites.append(
            dict(source_operation_ordinal=ordinal, family=family, status="admitted", software=decision, numeric=numeric)
        )
    plan = dict(
        status="admitted",
        numerical_family="F5" if boolean_count else "F2",
        literal_zero_producers_folded=folded_zero_producers,
        quantized_contractions=0,
        native_contractions=0,
        quantized_non_contraction_sites=0,
        exact_dtype_sites=len(sites),
        sites=sites,
        schedule_proof=proof,
        limb_plans=plans,
        native_semantic_proof=semantics,
        native_sign_proofs=sign_proofs,
        typed_ieee_proofs=typed_proofs,
        source_uniform_literal_folds=literal_folds,
        boolean_semantics={"source": "closed_boolean_and", "integer_components": boolean_count},
        input_bound_proof=dict(
            method="full_dtype_storage_domain",
            dtypes=native_types,
            absolute_bounds=dtype_bounds,
            dyadic_grids=[
                None if value is None else dict(binary_exponent=value[0], integer_absolute_bound=value[1])
                for value in grids
            ],
        ),
        host_floor=dict(
            status="required",
            comparison="native_eager_bits_every_boundary_result",
            order_dependent_reduction=False,
            finite_subproblem_overlay=False,
        ),
    )
    if quantized_contract is not None:
        plan.update(
            quantized_contract=quantized_contract,
            host_floor=dict(status="not_applicable", comparison="independent_quantized_exact_rational"),
            quantized_non_contraction_sites=len(sites),
        )
    from merlin.targetgen.core_aten_exact_dtype import publish_plan

    (directory / "model.mlir").write_text(source + "\n")
    return publish_plan(directory, records, selection, plan)
