"""Original archive joins only; synthetic fixtures grant no hardware authority."""

import copy
import hashlib
import json

import pytest
from merlin_timing import archive as A
from merlin_timing import contracts as C
from merlin_timing import count_intake as I
from merlin_timing.cli import main
from test_archive import fixture, per_record, reports_for, write_archive


def inputs(tmp_path):
    index, payload, binding = fixture(tmp_path)
    for number, item in enumerate(binding["cases"]):
        selected = [r for r in index["records"] if r["result"] in item["results"]]
        source = selected[0]["artifacts"]["input_interface"]
        data = f"synthetic complete source {number}".encode()
        source["sha256"], source["bytes"] = hashlib.sha256(data).hexdigest(), len(data)
        payload["run/" + source["path"]] = data
        case = item["case"]
        case["source"] = dict(case["source"], source_sha256=source["sha256"])
        case["source_group_sha256"] = source["sha256"]
        case["workload_sha256"] = C.workload(case)
    per_record(index)
    path, digest = write_archive(tmp_path, index, payload)
    binding["archive_sha256"] = digest
    return A.inspect_archive(path, digest), binding, path, digest


def test_unreviewed_plan_keeps_every_original_and_all_counts_unknown(tmp_path):
    inventory, _, _, _ = inputs(tmp_path)
    before = copy.deepcopy(inventory)
    plan = I.prepare(inventory)
    assert plan == I.prepare(inventory) and inventory == before
    assert plan["total_records"] == plan["unresolved_records"] == 6
    assert len(plan["source_groups"]) == 3
    assert all(len(g["records"]) == 2 and g["split"] is None for g in plan["source_groups"])
    assert all(r["fresh_count_report"] is None and r["qualified_timing"] is None for r in plan["requests"])
    assert not plan["launch_authority"] and plan["qualification_authority"] == "NONE"
    assert plan["observation_plan"] is None and plan["launches"] == []
    assert plan["domain"] is None
    assert [r["original_label"]["cycles"] for r in plan["requests"]] == [
        r["row"]["cycles"] for r in inventory["records"]
    ]


def test_reviewed_plan_joins_through_original_attach(tmp_path):
    inventory, binding, _, _ = inputs(tmp_path)
    plan = I.prepare(inventory, binding)
    original = A.prepare(inventory, binding)
    assert plan["observation_plan"] == original
    assert all(not row["missing"] for row in plan["requests"])
    dataset = A.attach(original, reports_for(original))
    assert len(dataset["records"]) == 6


def test_different_functional_wrapper_is_never_selected(tmp_path):
    inventory, _, _, _ = inputs(tmp_path)
    row = inventory["records"][0]["row"]
    row["executions"] = {"L2": {"elf_sha256": C.digest("different functional caller")}}
    request = I.prepare(inventory)["requests"][0]
    assert request["elf_sha256"] == row["elf_sha256"]
    assert request["alternate_functional_execution"] == row["executions"]["L2"]
    assert request["elf_sha256"] != request["alternate_functional_execution"]["elf_sha256"]


@pytest.mark.parametrize("damage", ["elf", "source", "group", "omit", "duplicate"])
def test_stale_original_and_source_split_bindings_refuse(tmp_path, damage):
    inventory, binding, _, _ = inputs(tmp_path)
    if damage == "elf":
        inventory["records"][0]["row"]["artifacts"]["elf"] = dict(
            inventory["records"][0]["row"]["artifacts"]["elf"], sha256=C.digest("stale")
        )
    elif damage == "source":
        binding["cases"][0]["case"]["source"]["source_sha256"] = C.digest("foreign")
        binding["cases"][0]["case"]["workload_sha256"] = C.workload(binding["cases"][0]["case"])
    elif damage == "group":
        binding["cases"][0]["case"]["source_group_sha256"] = C.digest("foreign group")
    elif damage == "omit":
        binding["cases"].pop()
    else:
        inventory["records"].append(copy.deepcopy(inventory["records"][0]))
        inventory["capsule_records"] += 1
    with pytest.raises(C.ContractError):
        I.prepare(inventory, binding)


def test_hidden_and_failed_records_remain_refusals_without_hidden_names(tmp_path):
    inventory, _, _, _ = inputs(tmp_path)
    inventory["records"][0]["row"]["visibility"] = "hidden"
    inventory["records"][1]["row"]["measured_now"] = False
    plan = I.prepare(inventory)
    assert plan["total_records"] == 6 and plan["refused_records"] == 2
    assert "capsule" not in plan["requests"][0] and "elf_member" not in plan["requests"][0]
    assert [r["record_id"] for r in plan["requests"]] == [r["record_id"] for r in inventory["records"]]


def test_cli_reopens_real_archive_and_refuses_missing_raw_record(tmp_path):
    inventory, _, path, digest = inputs(tmp_path)
    out = tmp_path / "count-plan.json"
    args = ["archive-count-plan", "--archive", str(path), "--archive-sha256", digest, "--output", str(out)]
    assert main(args) == 0
    assert json.loads(out.read_text())["total_records"] == len(inventory["records"])
    assert main(args) == 2  # immutable result; never overwrite
    path.write_bytes(path.read_bytes()[:-10240])
    assert main([*args[:-1], str(tmp_path / "invalid.json")]) == 2
    assert not (tmp_path / "invalid.json").exists()


def test_per_record_missing_review_stays_unknown_and_attach_refuses_counts(tmp_path):
    inventory, binding, _, _ = inputs(tmp_path)
    binding["schema"] = A.BINDING_V2
    for item in binding["cases"]:
        scope = item.pop("counter_scope")
        item["counter_scopes"] = dict.fromkeys(item["results"], scope)
    missing = binding["cases"][0]["results"][0]
    binding["cases"][0]["counter_scopes"][missing] = None
    plan = I.prepare(inventory, binding)
    request = next(r for r in plan["requests"] if r["result"] == missing)
    assert request["scope_binding"] is None and request["missing"] == ["original_counter_pc_review"]
    assert len(plan["observation_plan"]["observations"]) == 6
    with pytest.raises(C.ContractError, match="counter-window review"):
        A.attach(plan["observation_plan"], reports_for(plan["observation_plan"]))


def test_compiler_variants_cannot_leak_source_group_splits(tmp_path):
    inventory, binding, _, _ = inputs(tmp_path)
    case = binding["cases"][1]["case"]
    case["source_group_sha256"] = binding["cases"][0]["case"]["source_group_sha256"]
    with pytest.raises(C.ContractError, match="crosses dataset splits"):
        I.prepare(inventory, binding)


@pytest.mark.parametrize("damage", ["source_bytes", "raw_record"])
def test_cli_checks_original_payload_even_with_new_archive_digest(tmp_path, damage):
    index, payload, _ = fixture(tmp_path)
    per_record(index)
    row = index["records"][0]
    if damage == "source_bytes":
        payload["run/" + row["artifacts"]["input_interface"]["path"]] = b"changed source"
    else:
        del payload["run/" + row["result"]]
    path, digest = write_archive(tmp_path, index, payload)
    out = tmp_path / "refused.json"
    assert main(["archive-count-plan", "--archive", str(path), "--archive-sha256", digest, "--output", str(out)]) == 2
    assert not out.exists()


def test_new_path_never_reinterprets_legacy_archive_version(tmp_path):
    inventory, binding, _, _ = inputs(tmp_path)
    inventory["archive_schema"] = A.ARCHIVE
    with pytest.raises(C.ContractError, match="per-record v3"):
        I.prepare(inventory, binding)
    # Legacy behavior remains available through its original entry point.
    assert A.prepare(inventory, binding)["schema"] == A.PLAN


def test_original_join_group_also_colocates_differently_encoded_sources(tmp_path):
    inventory, binding, _, _ = inputs(tmp_path)
    for number in (0, 2):
        inventory["records"][number]["row"]["join"] = {"source_group_key": C.digest("same original group")}
    with pytest.raises(C.ContractError, match="archive source variants cross"):
        I.prepare(inventory, binding)
