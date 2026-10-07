"""Atlas reference metadata stays loadable without implying RTL qualification."""

from __future__ import annotations

import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import capability_manifests, target_registry
from merlin.targetgen.software_spec import admit_operation, capability_contract, load_software_spec


def test_atlas_reference_contract_is_selected_and_matches_derivation_intent(monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.delenv("MERLIN_TARGETS_DIR", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    selected = repo_root() / "examples/atlas/target"
    info = target_registry.resolve("atlas")
    assert info.kind == "reference"
    assert info.contract_path == selected / "contracts/target_contract.yaml"

    contract = capability_manifests.validate(info.load_contract())
    residual = yaml.safe_load((selected / "contracts/residual.yaml").read_text(encoding="utf-8"))
    assert residual.pop("facts_source") == "rtl"
    # The authored part is exactly the residual. Units the deriver synthesized from the RTL facts are
    # recorded beside it and marked in `derived_compute_units` (test_lane_datapaths re-derives them);
    # they are never authored intent, so they are set aside for this comparison.
    derived_units = set(contract.get("derived_compute_units") or ())
    authored = {k: v for k, v in contract.items() if k != "derived_compute_units"}
    authored["compute_units"] = [u for u in contract["compute_units"] if u["name"] not in derived_units]
    assert residual == authored
    assert contract["status"] == "prototype" and contract["requires_human_review"] is True
    assert "mesh" not in contract["capabilities"]
    assert "encoding" not in contract
    assert "scaling" not in contract["compute_units"][0]
    assert "requant" not in contract["compute_units"][0]
    assert contract["compute_units"][0]["accumulate"] == [{"in": "fp8_e4m3", "weight": "fp8_e4m3", "acc": "bf16"}]

    spec = load_software_spec(selected / "software-spec.yaml", target="atlas")
    assert capability_contract(spec, base_contract=contract) == contract
    signature = {
        "family": "contraction",
        "operand_dtype": "fp8_e4m3",
        "accum_dtype": "bf16",
        "rank": 2,
        "layout": "row_major_contiguous",
        "tails": "zero_pad_valid_window",
        "broadcasting": "none",
        "aliasing": "disjoint_inputs_outputs",
    }
    decision = admit_operation(spec, "matmul", signature, "accelerator")
    assert decision["constraints_status"] == "matched"
    assert decision["status"] == "unknown"
