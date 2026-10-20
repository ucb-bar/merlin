"""Original archive intake with synthetic bytes; no physical qualification."""

import copy
import hashlib
import json

import pytest
from merlin_experiments.phase2 import timing_domain_readiness as T
from merlin_timing import archive as A
from merlin_timing import contracts as C
from test_archive import fixture, per_record, reports_for, write_archive
from test_archive_source_domain import catalog, interface, source

from merlin.targetgen.contract.interface_emit import emit_interface_mlir


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def store(path, data):
    path.write_text(json.dumps(data))
    return pin(path)


def original_archive(tmp_path):
    index, payload, binding = fixture(tmp_path)
    for number, selected in enumerate(binding["cases"]):
        raw = emit_interface_mlir(interface(m=number+2)).encode()
        digest = hashlib.sha256(raw).hexdigest()
        results = set(selected["results"])
        for row in index["records"]:
            if row["result"] in results:
                old = row["artifacts"]["input_interface"]
                old.update(sha256=digest, bytes=len(raw))
                payload["run/"+old["path"]] = raw
        selected["case"]["source"]["source_sha256"] = digest
        selected["case"]["source_group_sha256"] = digest
        selected["case"]["mechanism_family"] = "matmul"
        selected["case"]["workload_sha256"] = C.workload(selected["case"])
    per_record(index)
    path, checksum = write_archive(tmp_path, index, payload)
    binding["archive_sha256"] = checksum
    plan = A.prepare(A.inspect_archive(path, checksum), binding)
    return {"archive": pin(path), "binding": store(tmp_path/"binding.json", binding),
            "observations": store(tmp_path/"observations.json", reports_for(plan))}


def cohort(tmp_path):
    path = tmp_path/"source.mlir"
    path.write_text(source())
    rows = [{"identity": "objective", "role": "objective", "source_requirements": {
        "parsed_operations": {"source": pin(path)}}},
        {"identity": "guard", "role": "functional_guard", "source_requirements": {
            "parsed_operations": {"reason": "original source cost refusal"}}}]
    return store(tmp_path/"cohort.json", {"members": rows}), path


def test_cli_reopens_actual_archive_wrappers_and_keeps_every_member_and_repeat(tmp_path):
    archive = original_archive(tmp_path)
    roster, _ = cohort(tmp_path)
    selected = {"schema": T.INPUT_SCHEMA, "cohort": roster, "archives": [archive]}
    inputs = store(tmp_path/"inputs.json", selected)
    output = tmp_path/"readiness.json"
    assert T.main(["--inputs", inputs["path"], "--inputs-sha256", inputs["sha256"], "--output", str(output)]) == 0
    result = json.loads(output.read_bytes())
    assert result["declared_members"] == 2 and result["no_member_dropped"]
    assert [m["original_role"] for m in result["members"]] == ["objective", "functional_guard"]
    assert result["members"][1]["status"] == "SOURCE_UNAVAILABLE_OR_REFUSED"
    assert result["archives"][0]["original_record_count"] == 6
    assert sum(len(row["records"]) for row in result["archives"][0]["records"]) == 6
    assert result["live_qualification"] is False and result["new_measurements"] == 0
    assert result["archives"][0]["review"]["status"] == "REFUSED"  # Fixture lacks held ranking evidence.
    assert T.main(["--inputs", inputs["path"], "--inputs-sha256", inputs["sha256"], "--output", str(output)]) == 2


@pytest.mark.parametrize("changed", ["elf_archive", "binding", "observations", "source", "selection"])
def test_stale_original_pins_refuse_before_output(tmp_path, changed):
    archive = original_archive(tmp_path)
    roster, source_path = cohort(tmp_path)
    inputs = store(tmp_path/"inputs.json", {"schema": T.INPUT_SCHEMA, "cohort": roster, "archives": [archive]})
    path = {"elf_archive": archive["archive"]["path"], "binding": archive["binding"]["path"],
            "observations": archive["observations"]["path"], "source": str(source_path),
            "selection": inputs["path"]}[changed]
    with open(path, "ab") as stream:
        stream.write(b"changed")
    output = tmp_path/"refused.json"
    assert T.main(["--inputs", inputs["path"], "--inputs-sha256", inputs["sha256"], "--output", str(output)]) == 2
    assert not output.exists()


def test_missing_original_raw_report_stays_missing_not_zero_or_dropped(tmp_path):
    archive = original_archive(tmp_path)
    roster, _ = cohort(tmp_path)
    path = tmp_path/"observations.json"
    reports = json.loads(path.read_bytes())
    reports["reports"].pop(next(iter(reports["reports"])))
    archive["observations"] = store(path, reports)
    result = T.analyze({"schema": T.INPUT_SCHEMA, "cohort": roster, "archives": [archive]})
    assert result["archives"][0]["original_record_count"] == 6
    assert result["archives"][0]["original_raw_report_count"] == 5
    # Original attach refuses to choose the other, available repeat as a
    # representative of the incomplete case. Preserve both distinctions.
    assert result["archives"][0]["eligible_dataset_records"] == 4
    assert len(result["archives"][0]["primary_raw_activity"]) == 5
    assert sum(not row["native_observation_available"] for case in result["archives"][0]["records"]
               for row in case["records"]) == 2


def test_member_profile_is_exact_joint_semantics_not_independent_axis_ranges(tmp_path):
    roster, path = cohort(tmp_path)
    document = json.loads((tmp_path/"cohort.json").read_bytes())
    support = catalog(T.DOMAIN.typed_profile(source()))
    assert T._members(document, [support])[0]["status"] == "EXACT_TRAIN_SOURCE_SUPPORTED_COUNTS_PENDING"
    path.write_text(source(m=3))
    document["members"][0]["source_requirements"]["parsed_operations"]["source"] = pin(path)
    assert T._members(document, [support])[0]["status"] == "EXACT_TRAIN_GEOMETRY_MISSING"
    path.write_text(source().replace("constant 0", "constant 1"))
    document["members"][0]["source_requirements"]["parsed_operations"]["source"] = pin(path)
    assert T._members(document, [support])[0]["status"] == "UNSUPPORTED_COMPLETE_SEMANTICS"


def test_different_counter_wrapper_cannot_rebind_original_scope(tmp_path):
    archive = original_archive(tmp_path)
    roster, _ = cohort(tmp_path)
    binding = json.loads((tmp_path/"binding.json").read_bytes())
    binding["review"]["counter"]["clock"] = "foreign clock"
    archive["binding"] = store(tmp_path/"foreign-binding.json", binding)
    with pytest.raises(ValueError):
        T.analyze({"schema": T.INPUT_SCHEMA, "cohort": roster, "archives": [archive]})


def test_union_split_or_domain_refusal_is_not_repaired_by_selecting_a_subset(tmp_path, monkeypatch):
    from merlin_timing import archive_union as UNION

    archive = original_archive(tmp_path)
    roster, _ = cohort(tmp_path)
    component = T._component(archive)
    foreign = copy.deepcopy(component)
    foreign[1]["cases"][0]["split"] = "test"
    records = iter((component, foreign))
    monkeypatch.setattr(T, "_component", lambda _spec: next(records))
    result = T.analyze({"schema": T.INPUT_SCHEMA, "cohort": roster, "archives": [archive, archive]})
    assert result["joint"]["status"] == "REFUSED" and len(result["archives"]) == 2
    assert result["declared_members"] == 2 and result["live_qualification"] is False
    with pytest.raises(ValueError):
        UNION.join((component[1], foreign[1]))
