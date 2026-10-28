"""Resolve software declarations from one selected hardware and composition context.

Both admission/validation and Phase 0 evidence use this owner. It grants no
execution or validation authority; unresolved facts retain the resolver's UNKNOWNs.
"""

from __future__ import annotations

from pathlib import Path


def resolve_selected_spec(
    software_spec,
    *,
    target,
    facts,
    contract,
    readout_facets=(),
    quantization_candidates=(),
    taxonomy=None,
    prohibited_roles=(),
    data_only=False,
    composition_validation=None,
):
    """Resolve exactly the supplied selection, including its composition evidence."""
    from merlin.targetgen import spec_fact_drift

    fact_contract = contract
    if data_only and not (contract.get("output_epilogue") or {}).get("output_path"):
        fact_contract = {**contract, "output_epilogue": _derived_output_epilogue(target, contract, facts)}
    fact_view = spec_fact_drift.fact_capabilities(
        target=target,
        contract=fact_contract,
        raw_facts=facts,
        readout_facets=readout_facets,
        quantization_candidates=quantization_candidates,
        taxonomy=taxonomy,
        prohibited_roles=prohibited_roles,
        **({"data_only": True} if data_only else {}),
    )
    if data_only:
        _add_composition_facts(fact_view, target, fact_contract, facts, software_spec)
    resolved, derivation = spec_fact_drift.resolve_spec(
        software_spec,
        fact_view,
        composition_validation=composition_validation,
    )
    derivation["prohibited_instruction_roles"] = sorted(set(prohibited_roles))
    return resolved, derivation


def _derived_output_epilogue(target, contract, raw_facts) -> dict:
    """The contract's output-epilogue record, read from the selected facts and the elaboration probes.

    The probes file is the one beside the target's own declarations, not beside the selected contract:
    a prepared contract is a derived artifact in a run tree and carries no probes, which left
    ``accumulate_on_load`` UNKNOWN and with it every accumulator stage of the output path. A probes file
    that is absent leaves the fact UNKNOWN, as before.
    """
    from merlin.targetgen import target_registry
    from merlin.targetgen.rtl import output_epilogue, semantic_facts

    probes = Path(target_registry.resolve(target).base) / "contracts" / semantic_facts.PROBES_FILE
    semantic = semantic_facts.derive(
        target, probes_file=probes if probes.is_file() else None, only=("accumulate_on_load",)
    )
    try:
        return output_epilogue.derive(raw_facts, contract=dict(contract), semantic_facts=semantic)
    except Exception as exc:  # noqa: BLE001 -- the record is all-UNKNOWN with the reason, as contract derivation does
        return output_epilogue.underived(f"{type(exc).__name__}: {exc}")


def _add_composition_facts(fact_view, target, contract, raw_facts, software_doc) -> None:
    """Give scheme resolution the operand, accumulator and panel facts the fusion enumeration reads.

    ``fact_capabilities`` reports only the output path and producer placement for a composition, so a
    scheme field that needs the operand width, the K-extension bound or the weight-panel width stayed
    UNKNOWN here although the selected facts and contract establish it. The same derivation the fusion
    enumeration consumes supplies them; a fact it cannot establish stays absent and so stays UNKNOWN.
    """
    import json

    from merlin.compile.fusion_enumeration import _spec_input
    from merlin.targetgen import composition_facts
    from merlin.targetgen.spec_fact_drift import FactField

    composition = fact_view["composition"]
    derived = composition_facts.derive(
        facts=raw_facts,
        contract=contract,
        software_spec=_spec_input(software_doc),
        output_path=composition.get("output_path"),
    )
    values, proof = derived["facts"], derived["evidence"]
    wanted = {}
    if "weight_panel_width" in values:
        wanted["weight_panel_width"] = ("weight_panel_width", values["weight_panel_width"])
    bound = values.get("k_extension_within")
    if isinstance(bound, dict) and "max_k" in bound:
        wanted["max_k_extension"] = ("k_extension_within", bound["max_k"])
    if type(values.get("operand_signed_bits")) is int:
        wanted["dtype"] = ("operand_signed_bits", f"i{values['operand_signed_bits']}")
    # Ordinary authored chains need these independently derived constraints too.
    # Missing values stay absent; never fill them from authored declarations.
    from merlin.targetgen.spec_fact_drift import _valid_composition_fact
    for key in ("dataflows", "pool_window_within_register_fields", "max_requires_store_path_pooling",
                "k_extension_within", "shared_input_resident"):
        if key in values and _valid_composition_fact(key, values[key]):
            value = values[key]
            members = tuple(value) if key == "dataflows" else (value,)
            composition[key] = FactField(members, True, (json.dumps(proof[key], sort_keys=True, default=str),))
    for key, (source, value) in wanted.items():
        if key not in composition:
            composition[key] = FactField((value,), True, (json.dumps(proof[source], sort_keys=True, default=str),))
