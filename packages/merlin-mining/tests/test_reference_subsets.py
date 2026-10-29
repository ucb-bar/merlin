"""Offline subset isolation, conservative observations and reproducible diversity selection."""

from __future__ import annotations

import hashlib

import pytest

from merlin.mining.reference_database import write_database
from merlin.mining.reference_database_io import digest, store_body, write_json
from merlin.mining.reference_subset_cards import diversity_order, features, known_leaves
from merlin.mining.reference_subsets import prepare, query, verify_bank


def toy_corpus(root):
    root.mkdir()
    artifacts, bodies, rows = [], [], []
    for model in range(3):
        original = root / f"model-{model}.elf"
        original.write_bytes(f"original-{model}".encode())
        artifacts.append(
            dict(
                id=f"model-{model}",
                path=original.name,
                format="elf",
                model=str(model),
                sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                origin={},
            )
        )
        for index in range(2):
            body = store_body(root, "machine", f"body-{model}-{index}".encode(), ".bin")
            bodies.append(body)
            parent = f"mappings/model-{model}/region-{index}"
            analysis = dict(
                cca=dict(
                    op="accelerator_region",
                    memory=None,
                    compute=dict(widening=False),
                    provenance=dict(unknowns=["selected_isa_correspondence_not_established"]),
                ),
                accelerator_sites=2,
                cfg=dict(natural_loops=index),
                schedule_model=dict(complete=False),
                evidence=dict(evidence_id=f"evidence-{model}-{index}"),
                mechanism_projection=dict(
                    role_sites={f"role-{model}": 2},
                    instruction_classes={f"ISA_{model}": 2},
                    counted_control=dict(status="unknown"),
                    field_fingerprint_status="complete",
                    field_fingerprints=[
                        dict(
                            instruction=f"ISA_{model}",
                            field="extent",
                            sites=2,
                            distinct_values=1,
                            missing_sites=0,
                            unresolved_sites=index,
                        )
                    ],
                    field_profiles={f"ISA_{model}.extent": [dict(value=model + 2, sites=2 - index)]},
                    mechanism_observations=dict(repetition_census="complete_recovered_field_tuples"),
                ),
            )
            write_json(root / parent / "analysis.json", analysis)
            write_json(root / parent / "cca.json", analysis["cca"])
            (root / parent / "cca.mlir").write_text("test-only unused MLIR")
            (root / parent / "instructions.dis").write_text("test-only listing")
            rows.append(
                dict(
                    id=f"model-{model}:region-{index}",
                    artifact_id=f"model-{model}",
                    kernel_id=body["id"],
                    address=model * 256 + index * 16,
                    size_bytes=body["size_bytes"],
                    accelerator_sites=2,
                    boundary_status="inferred_static_span",
                    analysis=parent + "/analysis.json",
                )
            )
    write_database(root / "database.sqlite", artifacts, bodies, rows, [])
    write_json(
        root / "SHA256SUMS.json",
        {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()},
    )
    write_json(
        root / "manifest.json",
        dict(database_sha256=digest(root / "database.sqlite"), file_inventory_sha256=digest(root / "SHA256SUMS.json")),
    )
    return [a["id"] for a in artifacts]


def test_subset_assets_are_nested_but_raw_files_are_identical(tmp_path):
    corpus = tmp_path / "corpus"
    models = toy_corpus(corpus)
    selection = prepare(corpus, tmp_path / "out", models, [1, 2, 3, 4])
    banks = [verify_bank(tmp_path / "out" / "conditions" / c["label"]) for c in selection["conditions"]]
    assert [len(b["units"]) for b in banks] == [0, 1, 2, 3, 4, 6]
    for small, large in zip(banks, banks[1:]):
        assert {u["kernel_id"] for u in small["units"]} < {u["kernel_id"] for u in large["units"]}
        assert small["raw"] == large["raw"]
    assert set(
        selection["order"][0]["models"] + selection["order"][1]["models"] + selection["order"][2]["models"]
    ) == set(models)
    raw = tmp_path / "out" / "conditions" / "raw"
    with pytest.raises(ValueError, match="unavailable"):
        query(raw)
    small = tmp_path / "out" / "conditions" / "cca-1"
    excluded = next(u["kernel_id"] for u in banks[-1]["units"] if u["kernel_id"] != banks[1]["units"][0]["kernel_id"])
    with pytest.raises(ValueError, match="outside"):
        query(small, kernel_id=excluded)
    answer = query(small, limit=1)
    assert len(answer["results"]) == 1 and answer["results"][0]["performance"] == "unknown"
    assert answer["results"][0]["fields"][0]["value_profile"]
    assert answer["results"][0]["unknowns"] == ["selected_isa_correspondence_not_established"]
    assert answer["results"][0]["count_basis"] == "listed_instruction_sites"
    full = tmp_path / "out" / "conditions" / "cca-6"
    values = query(full, instruction="ISA_0", field="extent", limit=2)
    assert [row["known_field_fraction"] for row in values["results"]] == [1, 0.5]
    assert all(row["observed_values_available"] for row in values["results"])
    with pytest.raises(ValueError, match="finite"):
        query(small, limit=9)
    (small / "unselected-analysis.json").write_text("{}")
    with pytest.raises(ValueError, match="unselected asset"):
        verify_bank(small)
    (small / "unselected-analysis.json").unlink()
    hidden = small / "unselected" / "bank.json"
    hidden.parent.mkdir()
    hidden.write_text("{}")
    with pytest.raises(ValueError, match="unselected asset"):
        verify_bank(small)


def test_known_observations_preserve_false_and_unknown_fields_remain_absent():
    assert known_leaves(
        dict(memory=None, dispatch=dict(loop_offloaded=False, n_fences=0), provenance=dict(secret=1))
    ) == {
        "dispatch.loop_offloaded": False,
        "dispatch.n_fences": 0,
    }


def test_features_use_observations_not_claimed_performance_or_gap_labels():
    base = dict(accelerator_sites=4, mechanism_projection=dict(role_sites=dict(load=2)))
    with_claims = dict(base, performance="winner", model_gap="expert_recipe", next_action="change compiler")
    assert features(base) == features(with_claims)


def test_diversity_order_does_not_depend_on_input_order_or_workload_names():
    units = [
        dict(kernel_id=str(i), size_bytes=i + 1, models={str(i % 3)}, features={"class:" + str(i % 2), "role:load"})
        for i in range(6)
    ]
    left = [u["kernel_id"] for u in diversity_order(units)]
    right = [u["kernel_id"] for u in diversity_order(list(reversed(units)))]
    assert left == right


def test_query_detects_mutated_card(tmp_path):
    write_json(tmp_path / "card.json", {})
    write_json(
        tmp_path / "bank.json",
        dict(schema="merlin.reference_subset_bank.v1", cca_enabled=True, files={"card.json": "0" * 64}, units=[]),
    )
    with pytest.raises(ValueError, match="pin mismatch"):
        query(tmp_path)
