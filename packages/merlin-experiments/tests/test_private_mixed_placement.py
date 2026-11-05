"""Intended host work cannot hide an omitted independently eligible device root."""

import json
from copy import deepcopy

import pytest
from merlin_experiments.phase1.feedback.private_group_provenance import verify_routed_source_coverage


def _records():
    source = {
        "eligible_groups": [4],
        "eligible_group_provenance": [
            {"source_group": 4, "source_region_id": "public-region", "source_node_ids": ["a", "b"]}
        ],
        "host_groups": [0, 1, 2],
    }
    source["accelerator_source_demand"] = {
        "status": "source_demand_derived", "n_unknown_host_groups": 0, "n_missing_direct_groups": 0,
        "n_unresolved_raw_regions": 0, "n_demanded_groups": 1,
        "demanded_group_provenance": deepcopy(source["eligible_group_provenance"]),
    }
    emitted = {
        "routed": [{"group": 7, "source_region_id": "public-region", "source_node_ids": ["a", "b"]}],
        "skipped": [["other-host-region", "the placement did not select this group's root"]],
    }
    return emitted, source


def test_complete_original_eligibility_allows_intended_host_regions():
    emitted, source = _records()
    assert verify_routed_source_coverage(emitted, source) == {7: 4}


@pytest.mark.parametrize("change", ["omission", "duplicate", "stale", "fake_eligibility", "missing_provenance"])
def test_intended_host_reason_cannot_excuse_bad_original_coverage(change):
    emitted, source = _records()
    if change == "omission":
        source["eligible_groups"].append(9)
        source["eligible_group_provenance"].append(
            {"source_group": 9, "source_region_id": "missing-region", "source_node_ids": ["c"]}
        )
    elif change == "duplicate":
        emitted["routed"].append(deepcopy(emitted["routed"][0]))
    elif change == "stale":
        emitted["routed"][0]["source_node_ids"] = ["old-node"]
    elif change == "fake_eligibility":
        source["eligible_groups"] = [0]
    else:
        emitted["routed"][0].pop("source_region_id")
    with pytest.raises(ValueError):
        verify_routed_source_coverage(emitted, source)


@pytest.mark.parametrize("refusal", [
    ["eligible", "no device form: unsupported operand"],
    ["eligible", "the group has no host destination"],
    ["eligible"], {}, None,
])
def test_actual_or_malformed_refusals_remain_fatal(refusal):
    emitted, source = _records()
    emitted["skipped"] = [refusal]
    with pytest.raises(ValueError, match="device refusal"):
        verify_routed_source_coverage(emitted, source)


def test_real_postbuild_consumer_continues_to_existing_symbol_policy(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import private_full_models as models

    emitted, source = _records()
    emitted.update(granularity="group", device="fixture")
    sidecar = tmp_path / "device.json"
    sidecar.write_text(json.dumps(emitted))
    elf = tmp_path / "program.elf"
    elf.write_bytes(b"synthetic bytes; no executable or compiler used")
    capture_tree = {"sha256": "a" * 64}
    package_tree = {"sha256": "b" * 64}
    receipt = dict(
        status="compiled", execution_route="device_requested_dispatch_unverified",
        inputs={"capture_tree": capture_tree, "device": {"package_tree": package_tree}},
        output={"elf": str(elf), "elf_sha256": models.file_sha256(elf),
                "device_sidecar": {"path": str(sidecar), "sha256": models.file_sha256(sidecar)}},
    )
    monkeypatch.setattr(models, "_require_static_build_inputs", lambda *a, **kw: None)
    monkeypatch.setattr(models.compilation_support, "verify", lambda *a, **kw: {})

    def symbol_policy(_emitted):
        raise ValueError("existing downstream symbol policy remains mandatory")

    monkeypatch.setattr(models, "_routed_kernel_symbols", symbol_policy)
    with pytest.raises(ValueError, match="downstream symbol policy"):
        models._verify_compiled_program(
            receipt, program="public-program", source=source, stage_tree=capture_tree,
            package_digest=package_tree, target="fixture", board="fixture-board",
            catalog=tmp_path / "catalog", dts=tmp_path / "dts", device_selected=True,
        )
