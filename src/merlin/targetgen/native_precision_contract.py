"""A native IEEE proof cannot replace a requested quantized-output bit contract."""


def requires_quantized_model(module):
    """Does the IR's floating work require the declared quantized model?

    Decided from the normalized IR alone (floating_contract): live floating
    arithmetic on data from floating storage is discharged only by the
    declared quantized model; movement, exact conversions out of integer
    storage, literal fills and identities keep their eager-bit proofs.
    """
    from merlin.targetgen.floating_contract import requires_quantized_model as decide

    return decide(module)["quantized"]


def proves_quantized_model(plan, software):
    from merlin.llvmlower.quantized_native import parameters

    engine = plan.get("integer_reference", {}).get("engine")
    if plan.get("quantized_contract") is not None:
        return (
            plan["quantized_contract"] == parameters(software)
            and engine == "independent_source_quantized_exact_rational"
        )
    # The legacy static recipe independently seals input scales and its
    # rational output reference. An eager/native component proof does neither.
    recipe = plan.get("capture_recipe") or {}
    numeric = software.get("numerical_semantics", {})
    domain = recipe.get("calibration_domain")
    return bool(
        domain
        and domain == numeric.get("calibration_domain")
        and domain.get("evaluation_inputs") == "never_read"
        and recipe.get("quantization_nonfinite") == numeric.get("quantization_nonfinite")
        and engine == "independent_pt2e_exact_rational"
    )
