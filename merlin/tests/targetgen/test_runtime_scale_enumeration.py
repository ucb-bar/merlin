"""Runtime enumeration integration; shared files are delivered as deferred patches."""

from __future__ import annotations

import copy
import json
import random

import pytest
import yaml
from test_runtime_scale_recognition import FACTS_FILE, ROOT, TRANSPORT, read, source

from merlin.common.jsonio import canonical_json, canonical_sha256
from merlin.compile import composition_fusion as CF
from merlin.compile import fusion_enumeration as FE
from merlin.targetgen.rtl.output_epilogue import derive as derive_output

ENUM_PATCH = "S/deferred-patches/L1-fusion_enumeration.py.patch"
PLAN_PATCH = "S/deferred-patches/L1-composition_fusion.py.patch"
requires_enumeration = pytest.mark.skipif(
    not hasattr(FE, "_scale_source_variants"), reason=f"requires deferred patch {ENUM_PATCH}"
)


@pytest.fixture(scope="module")
def selected():
    assert FACTS_FILE.is_file(), "select actual target facts with RT_FACTS"
    facts = json.loads(FACTS_FILE.read_text())
    contract = yaml.safe_load((ROOT / "examples/gemmini/support/contracts/target_contract.yaml").read_text())
    # Derive the physical readout from selected FIRRTL, rather than fabricating its stages.
    contract["output_epilogue"] = derive_output(facts, contract=contract)
    assert contract["output_epilogue"]["output_path"]
    from merlin.targetgen.software_spec import load_software_spec

    spec = load_software_spec(ROOT / "examples/gemmini/target/software-spec.yaml", target="gemmini")
    # Use the same independent placement/numeric resolution as the ordinary CLI.
    from merlin.targetgen.spec_fact_drift import _resolve_chain, fact_capabilities, resolve_spec

    spec = FE._spec_input(spec)
    capabilities = fact_capabilities(target="gemmini", contract=contract, raw_facts=facts, data_only=True)
    spec, _ = resolve_spec(spec, capabilities)
    producers = sorted({"movement" if form == "window_movement" else form for form, _ in FE._producer_forms(spec)})
    placement = _resolve_chain(
        {"producer": producers, "constraints": []},
        capabilities,
        {"output_path": contract["output_epilogue"]["output_path"]},
    )
    spec["composition_resolution"] = {"chains": [placement]}
    return dict(target="gemmini", facts=facts, contract=contract, spec=spec)


def runtime_mixed(row):
    sem = row["semantic"]
    nodes = sem.get("nodes", [])
    if (
        row["state"] != "candidate"
        or sem.get("structure") != "shared_input"
        or sem.get("dataflow") != "WS"
        or len(nodes) != 2
    ):
        return False
    if any(
        n["form"] != "contraction" or n["realization"] != "matmul" or n["seed"] or n["residual"] is not None
        for n in nodes
    ):
        return False
    if any(
        n["regime"]["readout_strategy"] != "f32_product_rne" or n["regime"]["weight_scale_granularity"] != "tensor"
        for n in nodes
    ):
        return False
    narrow = [
        n for n in nodes if n["readout"]["exit"] == "requant" and n["regime"].get("scale_source") == "runtime_input"
    ]
    raw = [n for n in nodes if n["readout"]["exit"] == "accumulator" and n["readout"]["stages"] == []]
    return len(narrow) == len(raw) == 1 and [
        s["kind"] for s in narrow[0]["readout"]["stages"] if s["kind"] != "activation"
    ] == ["convert", "scale", "requant"]


@requires_enumeration
def test_runtime_variants_survive_shared_representative_canonicalization(selected):
    manifest = FE.enumerate_candidates(**selected)
    witnesses = [row for row in manifest["rows"] if runtime_mixed(row)]
    assert witnesses, "runtime_scale_mixed_ws selection must be nonempty"
    assert {
        any(s["kind"] == "activation" for n in r["semantic"]["nodes"] for s in n["readout"]["stages"])
        for r in witnesses
    } == {False, True}
    assert manifest["normalized_facts"]["fields"]["runtime_readout_scale"]["value"] == TRANSPORT
    print("runtime_scale_mixed_ws IDs:", [r["candidate_id"] for r in witnesses])
    import os
    from pathlib import Path

    if destination := os.environ.get("L1_ENUM_EVIDENCE"):
        Path(destination).write_text(
            json.dumps(
                {
                    "candidate_count": manifest["candidate_count"],
                    "summary": manifest["summary"],
                    "identity": manifest["identity"],
                    "selection": "runtime_scale_mixed_ws",
                    "rows": witnesses,
                },
                indent=2,
            )
            + "\n"
        )
    assert all(r["candidate_id"] == canonical_sha256(r["semantic"]) for r in witnesses)
    again = FE.enumerate_candidates(**selected)
    assert canonical_json(again) == canonical_json(manifest)
    permuted = copy.deepcopy(selected)
    random.Random(1701).shuffle(permuted["spec"]["quantization"]["schemes"]["schemes"])
    permuted["facts"]["facts"]["interfaces"].reverse()
    assert canonical_json(FE.enumerate_candidates(**permuted)) == canonical_json(manifest)
    # A runtime narrow route must coexist with the original constant route.
    for row in witnesses:
        nodes = copy.deepcopy(row["semantic"]["nodes"])
        runtime = next(n for n in nodes if n["regime"].get("scale_source") == "runtime_input")
        runtime["regime"].pop("scale_source")
        assert any(
            r["state"] == "candidate"
            and r["semantic"].get("structure") == "shared_input"
            and r["semantic"].get("dataflow") == "WS"
            and [n["scheme"] for n in r["semantic"].get("nodes", [])] == [n["scheme"] for n in nodes]
            and not any(n["regime"].get("scale_source") == "runtime_input" for n in r["semantic"].get("nodes", []))
            for r in manifest["rows"]
        )


@requires_enumeration
def test_declared_campaign_executes_both_runtime_witnesses(selected):
    from merlin.targetgen.required_regimes import regime_identity, represents_required_regime, required_regimes
    from merlin.targetgen.software_spec import load_software_spec

    declared = load_software_spec(ROOT / "examples/gemmini/target/software-spec.yaml", target="gemmini")
    manifest = FE.enumerate_candidates(**selected)
    witnesses = {r["candidate_id"] for r in manifest["rows"] if runtime_mixed(r)}
    campaign = FE.stratify_candidates(manifest, declared)
    executed = {r["candidate_id"] for r in campaign["rows"] if r["state"] == "candidate"}
    assert len(witnesses) == 2 and witnesses <= executed
    coverage = {canonical_json(e["regime"]): e for e in campaign["campaign"]["regime_coverage"]}
    for chain, scheme, axes in required_regimes(declared):
        eligible = {
            r["candidate_id"]
            for r in manifest["rows"]
            if r["state"] == "candidate" and represents_required_regime(r["semantic"], chain, scheme, axes)
        }
        entry = coverage[canonical_json(regime_identity(chain, scheme, axes))]
        assert entry["executed"] == len(eligible & executed) >= min(2, len(eligible))
    FE.verify_stratification(campaign, declared, required=True)
    FE.verify_stratification(campaign, required=True)


@requires_enumeration
def test_enumeration_remains_independent_of_composition_catalog(selected):
    minimal = copy.deepcopy(selected)
    # Keep this authority test inexpensive; choose by numeric semantics, never by ID.
    schemes = minimal["spec"]["quantization"]["schemes"]["schemes"]
    minimal["spec"]["quantization"]["schemes"]["schemes"] = [
        s
        for s in schemes
        if s["weight_scale_granularity"] == "tensor"
        and s["readout_strategy"] == "f32_product_rne"
        and s["operand_sum_rounding"] == "single_rounding"
    ]
    first = canonical_json(FE.enumerate_candidates(**minimal))
    minimal["spec"]["compositions"] = {"chains": [{"id": "does_not_select_any_hardware_behavior"}]}
    assert canonical_json(FE.enumerate_candidates(**minimal)) == first
    del minimal["spec"]["compositions"]
    assert canonical_json(FE.enumerate_candidates(**minimal)) == first
    renamed = copy.deepcopy(minimal)
    renamed["spec"]["quantization"]["schemes"]["schemes"][0]["id"] = "arbitrary_runtime_compatible_numeric_scheme"
    assert any(runtime_mixed(row) for row in FE.enumerate_candidates(**renamed)["rows"])


@requires_enumeration
@pytest.mark.parametrize("missing", ["transport", "policy"])
def test_runtime_variants_require_policy_and_transport(selected, missing):
    minimal = copy.deepcopy(selected)
    schemes = minimal["spec"]["quantization"]["schemes"]["schemes"]
    minimal["spec"]["quantization"]["schemes"]["schemes"] = [
        s
        for s in schemes
        if s["weight_scale_granularity"] == "tensor"
        and s["readout_strategy"] == "f32_product_rne"
        and s["operand_sum_rounding"] == "single_rounding"
    ]
    if missing == "transport":
        minimal["contract"]["readout_semantics"]["scalar_abi"]["scale_dtype"] = "f16"
    else:
        del minimal["spec"]["quantization"]["numerics"]["policies"]["scale_source"]["runtime_input"]
    manifest = FE.enumerate_candidates(**minimal)
    runtime = [
        r
        for r in manifest["rows"]
        if any(n["regime"].get("scale_source") == "runtime_input" for n in r["semantic"].get("nodes", []))
    ]
    assert runtime and not any(r["state"] == "candidate" for r in runtime)
    reason = "runtime scale transport" if missing == "transport" else "runtime scale policy"
    assert all(any(reason in issue["reason"] for issue in r["reasons"]) for r in runtime)


@pytest.mark.skipif(
    "runtime_scale_facts" not in __import__("inspect").signature(CF.plan_contraction_readout).parameters,
    reason=f"requires deferred patch {PLAN_PATCH}",
)
def test_runtime_plan_forwards_transport_and_keeps_raw_facts_digest():
    from merlin.verify.lean_contract import document_digest

    module, chain, decision = read(source())
    from test_runtime_scale_recognition import PATH

    scheme = {"readout_strategy": "f32_product_rne"}
    raw = {"selected_facts": "unaltered"}
    plan = CF.plan_contraction_readout(
        module, chain.root, path=PATH, scheme=scheme, facts=raw, runtime_scale_facts=TRANSPORT
    )
    assert plan.decision["proof"] == decision.proof
    assert plan.decision["facts_sha256"] == document_digest(raw)
    assert plan.decision["proof"]["runtime_scale_facts_sha256"] == document_digest(TRANSPORT)
    ops = tuple(module.walk())
    assert plan.operation_ordinals == tuple(ops.index(op) for op in (chain.root, *chain.ops))
    assert len(plan.operation_ordinals) == 2

    candidate = {
        "route": "epilogue",
        "operation_ordinals": list(plan.operation_ordinals),
        "detail": {
            "root_ordinal": ops.index(chain.root),
            "composition": {"producer": "contraction", "exits": ["requant"]},
        },
    }
    composed = CF.plan_composition(
        module, candidate, path=PATH, scheme=scheme, facts=raw, runtime_scale_facts=TRANSPORT
    )
    assert composed.record() == plan.record()

    for ordinals in (plan.operation_ordinals[:1], (*plan.operation_ordinals, plan.operation_ordinals[-1])):
        changed = {**candidate, "operation_ordinals": list(ordinals)}
        with pytest.raises(ValueError):
            CF.plan_composition(module, changed, path=PATH, scheme=scheme, facts=raw, runtime_scale_facts=TRANSPORT)
