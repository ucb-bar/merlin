"""Synthetic archive integration and adversarial controls; no hardware verdicts."""

import base64
import hashlib
import io
import json
import tarfile
from pathlib import PurePosixPath

import pytest
from data_helpers import sample
from merlin_timing import archive as A
from merlin_timing import contracts as C
from merlin_timing import validation as V
from merlin_timing.cli import main


def fixture(tmp_path):
    data = sample()
    cases = [data["cases"][index] for index in (0, 16, 24)]
    pins = {role: C.digest(role) for role in ("accessor", "isa", "clock")}
    selection = {
        "schema": "merlin.kernel_call_timing.v1",
        "selection_sha256": C.digest("selected-contract"),
        "boundary": "immediately before kernel entry through explicitly selected completion",
        "counter_semantics": {
            "unit": "cycles",
            "width_bits": 64,
            "wrap": "modulo",
            "clock": "test-clock",
            "source_pins": pins,
        },
        "max_delta_cycles": 100000,
    }
    sim = {"sha256": C.digest("synthetic-gsim"), "provenance": "synthetic"}
    review = {
        "timing_selection": selection,
        "cycle_scope": "entry-through-completion",
        "sim_provenance_sha256": C.digest(sim),
        "cache_protocol": None,
        "counter": {
            "unit": "cycles",
            "width_bits": 64,
            "wrap": "modulo",
            "clock": "test-clock",
            "max_delta_cycles": 100000,
            "sources": pins,
        },
    }
    domain = dict(
        data["domain"],
        target_model_sha256=hashlib.sha256(b"synthetic frozen model").hexdigest(),
        label_engine_sha256=sim["sha256"],
        scope_sha256=C.digest(
            {
                "schema": "merlin.kernel_call_scope.v1",
                "cycle_scope": review["cycle_scope"],
                "timing_selection_sha256": selection["selection_sha256"],
                "selection_record_sha256": C.digest(selection),
            }
        ),
        memory_regime_sha256=C.digest({"cache_warmth": "unknown", "protocol": None}),
    )
    rows, payload, bindings = [], {}, []
    for number, case in enumerate(cases):
        parent = f"_qa_work/candidate/runs/suite/capsule{number}"
        artifacts = {}
        marker = (
            "/* merlin.kernel_call_timing.v1 selection_sha256="
            + selection["selection_sha256"]
            + " boundary=entry-through-completion record_b64="
            + base64.b64encode(json.dumps(selection).encode()).decode()
            + " */"
        )
        for key, name, content in [
            ("elf", "package_kernel.elf", f"synthetic ELF {number}".encode()),
            ("harness", "harness.c", (marker + f"\nsynthetic input caller {number}").encode()),
            ("input_interface", "input.interface.mlir", b"synthetic input interface"),
            ("command_buffer", "command_buffer.json", b"{}"),
            ("static_instruction_trace", "instruction_trace.json", b"static only"),
            ("numeric_report", "numeric_report.yaml", b"status: pass\n"),
        ]:
            path = parent + "/generated/" + name
            payload["run/" + path] = content
            artifacts[key] = {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        artifacts["functional_console"] = None
        case = dict(
            case,
            elf_sha256=artifacts["elf"]["sha256"],
            input_sha256=C.digest(
                {"schema": "merlin.embedded_input_context.v1", "harness_sha256": artifacts["harness"]["sha256"]}
            ),
            scope_sha256=domain["scope_sha256"],
        )
        case["workload_sha256"] = C.workload(case)
        shape = {"input": {"shape": [7, 11], "dtype": "i8"}}
        results = []
        for repeat in range(2):
            # Original paid repeats have different result paths but the same ELF.
            result = f"_qa_work/paid{repeat}/runs/suite/capsule{number}/capsule_result.json"
            console_path = str(PurePosixPath(result).parent / "artifacts" / "rtl_console.log")
            console_raw = f"METRIC cycles {100 + repeat}\n".encode()
            payload["run/" + console_path] = console_raw
            record_artifacts = dict(
                artifacts,
                rtl_console={
                    "path": console_path,
                    "sha256": hashlib.sha256(console_raw).hexdigest(),
                    "bytes": len(console_raw),
                },
            )
            row = {
                "result": result,
                "capsule": f"capsule{number}",
                "status": "pass",
                "result_status": "pass",
                "engine": "gsim",
                "cycles": 100 + repeat,
                "cycles_measured": True,
                "measured_now": True,
                "console_cycle_verified": True,
                "cycle_scope": review["cycle_scope"],
                "timing_selection_sha256": selection["selection_sha256"],
                "timing_selection": selection,
                "shape": shape,
                "candidate_sha256": C.digest("candidate"),
                "command_buffer_sha256": artifacts["command_buffer"]["sha256"],
                "elf": artifacts["elf"]["path"],
                "elf_sha256": case["elf_sha256"],
                "harness_sha256": artifacts["harness"]["sha256"],
                "artifacts": record_artifacts,
                "sim_provenance": sim,
                "derived_from_rtl": True,
                "cycle_accurate": True,
                "timing": {"elapsed_seconds": 0.1},
            }
            original = {
                "capsule": row["capsule"],
                "status": row["result_status"],
                "tiers": {
                    "L3": {
                        key: row[key]
                        for key in (
                            "status",
                            "engine",
                            "cycles",
                            "derived_from_rtl",
                            "cycle_accurate",
                            "sim_provenance",
                            "timing",
                        )
                    }
                },
                "command_buffer_artifact": {
                    "artifact_sha256": row["command_buffer_sha256"],
                    "command_buffer": {"tensors": shape},
                },
                "original_compiler_child": {"candidate_sha256": row["candidate_sha256"]},
            }
            payload["run/" + result] = json.dumps(original).encode()
            rows.append(row)
            results.append(result)
        bindings.append(
            {
                "case": case,
                "results": results,
                "candidate_sha256": C.digest("candidate"),
                "shape_sha256": C.digest(shape),
                "counter_scope": {
                    "begin_pc": 4096,
                    "end_pc": 8192,
                    "occurrence": 0,
                    "evidence_sha256": C.digest(["review", number]),
                },
            }
        )
    index = {
        "schema": A.ARCHIVE,
        "records": rows,
        "capsule_records": len(rows),
        "gsim_pass_records": len(rows),
        "gsim_pass_with_cycles": len(rows),
        "gsim_pass_complete": len(rows),
        "timing_ready": True,
    }
    binding = {
        "schema": A.BINDING,
        "archive_sha256": "",
        "domain": domain,
        "feature_names": data["feature_names"],
        "review": review,
        "cases": bindings,
    }
    return index, payload, binding


def write_archive(tmp_path, index, payload, extra=None):
    path = tmp_path / "capsules.tar"
    with tarfile.open(path, "w") as tar:
        # Index deliberately last: verification must not depend on member order.
        for name, data in [*payload.items(), ("index.json", json.dumps(index).encode())]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        if extra is not None:
            tar.addfile(extra, io.BytesIO(b"x") if extra.isreg() else None)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def prepared(tmp_path):
    index, payload, binding = fixture(tmp_path)
    path, checksum = write_archive(tmp_path, index, payload)
    binding["archive_sha256"] = checksum
    inventory = A.inspect_archive(path, checksum)
    return A.prepare(inventory, binding), inventory, binding, path


def reports_for(plan):
    reports = {}
    data = plan["dataset_template"]
    for row in plan["observations"]:
        case = next(case for case in data["cases"] if case["case_id"] == row["case_id"])
        report = {
            "schema": "merlin.annotated_spike_report.v1",
            "profile": "phase1-public",
            "domain": data["domain"],
            "domain_sha256": C.digest(data["domain"]),
            "model_sha256": data["domain"]["target_model_sha256"],
            "feature_engine_sha256": data["domain"]["feature_engine_sha256"],
            **{key: case[key] for key in ("elf_sha256", "input_sha256", "scope_sha256", "device_sha256")},
            "functional_status": "PASS",
            "execution_completed": True,
            "raw_functional_exit_code": 0,
            "timing_complete": True,
            "estimated_cycles": 100,
            "feature_counts": {"host_instructions": 10},
            "unresolved": [],
            "cycle_breakdown": {},
            "scope_binding": row["scope_binding"],
        }
        reports[row["record_id"]] = {"sha256": C.digest(report), "report": report}
    return {"schema": "merlin.phase1.spike_observations.v1", "plan_sha256": C.digest(plan), "reports": reports}


def test_archive_to_original_report_dataset_preserves_repeats_and_scope(tmp_path):
    plan, inventory, _, path = prepared(tmp_path)
    assert path.stat().st_size < 200000
    assert inventory["capsule_records"] == 6 and len(plan["dataset_template"]["cases"]) == 3
    assert plan["launches"] == [] and plan["cache_warmth"] == "unknown"
    assert plan["static_trace_is_dynamic_timeline"] is False
    data = A.attach(plan, reports_for(plan))
    assert len(data["records"]) == 6
    assert {row["label"]["cycles"] for row in data["records"]} == {100, 101}
    assert len(C.grouped_records(data)) == 3
    assert not V.validate(data, prediction_kind="event_timing")["qualified"]


@pytest.mark.parametrize("damage", ["hash", "missing", "summary", "selection", "duplicate", "traversal", "link"])
def test_archive_corruption_refuses(tmp_path, damage):
    index, payload, _ = fixture(tmp_path)
    extra = None
    if damage == "hash":
        payload[next(iter(payload))] += b"changed"
    elif damage == "missing":
        del payload[next(iter(payload))]
    elif damage == "summary":
        index["capsule_records"] += 1
    elif damage == "selection":
        index["records"][0]["timing_selection_sha256"] = "0" * 64
    else:
        extra = tarfile.TarInfo(
            {"duplicate": next(iter(payload)), "traversal": "run/../escape", "link": "run/link"}[damage]
        )
        if damage == "link":
            extra.type = tarfile.SYMTYPE
            extra.linkname = "outside"
        else:
            extra.size = 1
    path, checksum = write_archive(tmp_path, index, payload, extra)
    with pytest.raises(C.ContractError):
        A.inspect_archive(path, checksum)


@pytest.mark.parametrize(
    "damage", ["clock", "maximum", "warmth", "simulator", "candidate", "shape", "input", "omit", "split", "pc"]
)
def test_binding_changes_refuse(tmp_path, damage):
    _, inventory, binding, _ = prepared(tmp_path)
    if damage == "clock":
        binding["review"]["counter"]["clock"] = ""
    elif damage == "maximum":
        binding["review"]["counter"]["max_delta_cycles"] = 2**63
    elif damage == "warmth":
        binding["review"]["cache_protocol"] = {"state": "warm", "receipt_sha256": C.digest("unbound")}
    elif damage == "simulator":
        binding["review"]["sim_provenance_sha256"] = "0" * 64
    elif damage in {"candidate", "shape"}:
        binding["cases"][0][damage + "_sha256"] = "0" * 64
    elif damage == "input":
        binding["cases"][0]["case"]["input_sha256"] = "0" * 64
    elif damage == "omit":
        binding["cases"][0]["results"].pop()
    elif damage == "split":
        binding["cases"][1]["case"]["source_group_sha256"] = binding["cases"][0]["case"]["source_group_sha256"]
    elif damage == "pc":
        binding["cases"][0]["counter_scope"]["end_pc"] += 1
    with pytest.raises(C.ContractError):
        A.prepare(inventory, binding)


def test_missing_repeat_never_selects_faster_label(tmp_path):
    plan, _, _, _ = prepared(tmp_path)
    reports = reports_for(plan)
    reports["reports"].pop(plan["observations"][1]["record_id"])
    data = A.attach(plan, reports)
    assert len(data["cases"]) == 3 and len(data["records"]) == 4
    assert plan["observations"][0]["case_id"] not in C.grouped_records(data)


def test_changed_native_scope_refuses(tmp_path):
    plan, _, _, _ = prepared(tmp_path)
    reports = reports_for(plan)
    row = next(iter(reports["reports"].values()))
    row["report"]["scope_binding"] = dict(row["report"]["scope_binding"], end_pc=9998)
    row["sha256"] = C.digest(row["report"])
    with pytest.raises(C.ContractError):
        A.attach(plan, reports)


def test_archive_cli_preserves_existing_output_and_requires_external_digest(tmp_path):
    _, _, _, path = prepared(tmp_path)
    out = tmp_path / "inventory.json"
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    args = ["archive-inspect", "--archive", str(path), "--archive-sha256", checksum, "--output", str(out)]
    assert main(args) == 0
    original = out.read_bytes()
    assert main(args) == 2 and out.read_bytes() == original
    with pytest.raises(C.ContractError):
        A.inspect_archive(path, "0" * 64)


@pytest.mark.parametrize("damage", ["nonempty_clock", "source_pin", "different_limit"])
def test_review_cannot_change_original_counter_facts(tmp_path, damage):
    _, inventory, binding, _ = prepared(tmp_path)
    counter = binding["review"]["counter"]
    if damage == "nonempty_clock":
        counter["clock"] = "another-clock"
    elif damage == "source_pin":
        counter["sources"]["clock"] = C.digest("another-source")
    else:
        counter["max_delta_cycles"] -= 1
    with pytest.raises(C.ContractError):
        A.prepare(inventory, binding)


@pytest.mark.parametrize("damage", ["console", "caller", "original_result", "candidate", "shape"])
def test_original_context_is_checked_after_index_hashes_match(tmp_path, damage):
    index, payload, _ = fixture(tmp_path)
    row = index["records"][0]
    if damage in {"console", "caller"}:
        role = "rtl_console" if damage == "console" else "harness"
        entry = row["artifacts"][role]
        raw = b"METRIC cycles 100\nMETRIC cycles 100\n" if damage == "console" else b"missing timing marker"
        payload["run/" + entry["path"]] = raw
        # Update every shared reference, so independent content checking is needed.
        for record in index["records"]:
            ref = record["artifacts"].get(role)
            if ref and ref["path"] == entry["path"]:
                ref.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
                if role == "harness":
                    record["harness_sha256"] = ref["sha256"]
    elif damage == "original_result":
        original = json.loads(payload["run/" + row["result"]])
        original["tiers"]["L3"]["cycles"] += 1
        payload["run/" + row["result"]] = json.dumps(original).encode()
    elif damage == "candidate":
        row["candidate_sha256"] = C.digest("different candidate")
    else:
        row["shape"] = {"input": {"dtype": "i8", "shape": [9, 17]}}
    path, checksum = write_archive(tmp_path, index, payload)
    with pytest.raises(C.ContractError):
        A.inspect_archive(path, checksum)


def test_failed_and_unavailable_records_with_null_artifacts_remain_auditable(tmp_path):
    index, payload, _ = fixture(tmp_path)
    failed = dict(
        index["records"][0],
        result="_qa_work/failed/runs/suite/capsule/capsule_result.json",
        status="unavailable",
        result_status="fail",
        cycles=None,
        cycles_measured=False,
        console_cycle_verified=False,
        elf=None,
        elf_sha256=None,
        artifacts={key: None for key in index["records"][0]["artifacts"]},
    )
    index["records"].append(failed)
    index["capsule_records"] += 1
    payload["run/" + failed["result"]] = b'{"status":"fail"}'
    path, checksum = write_archive(tmp_path, index, payload)
    inventory = A.inspect_archive(path, checksum)
    assert len(inventory["records"]) == 7 and inventory["gsim_pass_complete"] == 6


def test_handoff_records_scope_selection_and_stratified_error_without_qualification(tmp_path):
    plan, _, _, _ = prepared(tmp_path)
    data = A.attach(plan, reports_for(plan))
    identities = {
        key: C.digest(key)
        for key in (
            "spike_executable_sha256",
            "spike_configuration_sha256",
            "model_artifact_sha256",
            "training_code_sha256",
        )
    }
    identities["model_artifact_sha256"] = data["domain"]["target_model_sha256"]
    handoff = A.handoff(plan, data, identities)
    assert handoff["phase2_ranking_eligible"] is False
    assert handoff["firesim_cycle_accuracy_established"] is False
    assert handoff["held_case_errors"][0]["absolute_cycle_error"] == 0.5
    assert handoff["errors_by_shape"] and handoff["errors_by_mechanism_family"]
    changed = json.loads(json.dumps(data))
    changed["records"][0]["label"]["cycles"] += 1
    with pytest.raises(C.ContractError):
        A.handoff(plan, changed, identities)


def test_all_archive_cli_stages_join_the_same_originals(tmp_path):
    _, _, binding, path = prepared(tmp_path)
    binding_path = tmp_path / "binding.json"
    C.write(binding_path, binding)
    plan_path = tmp_path / "plan.json"
    assert (
        main(
            [
                "archive-prepare",
                "--archive",
                str(path),
                "--archive-sha256",
                binding["archive_sha256"],
                "--binding",
                str(binding_path),
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = C.read(plan_path)
    report_path = tmp_path / "reports.json"
    C.write(report_path, reports_for(plan))
    dataset_path = tmp_path / "dataset.json"
    assert (
        main(["archive-attach", "--plan", str(plan_path), "--reports", str(report_path), "--output", str(dataset_path)])
        == 0
    )
    selected = {}
    for name, payload in {
        "spike-executable": b"synthetic executable",
        "spike-configuration": b"synthetic config",
        "model-artifact": b"synthetic frozen model",
        "training-code": b"synthetic fitting source",
    }.items():
        selected[name] = tmp_path / name
        selected[name].write_bytes(payload)
    handoff_path = tmp_path / "handoff.json"
    assert (
        main(
            [
                "archive-handoff",
                "--plan",
                str(plan_path),
                "--dataset",
                str(dataset_path),
                *[arg for name, value in selected.items() for arg in ("--" + name, str(value))],
                "--output",
                str(handoff_path),
            ]
        )
        == 0
    )
    handoff = C.read(handoff_path)
    assert handoff["archive_sha256"] == binding["archive_sha256"]
    assert handoff["phase2_ranking_eligible"] is False
    assert handoff["artifact_identities"]["spike_executable_sha256"] == A.file_identity(selected["spike-executable"])


@pytest.mark.parametrize("kind", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_SPARSE])
def test_extended_metadata_refuses_before_reading_a_large_payload(tmp_path, kind):
    header = tarfile.TarInfo("oversized-metadata")
    header.type = kind
    header.size = 1024**3
    path = tmp_path / "metadata.tar"
    path.write_bytes(header.tobuf(format=tarfile.USTAR_FORMAT))
    with pytest.raises(C.ContractError, match="metadata|sparse"):
        A.inspect_archive(path, hashlib.sha256(path.read_bytes()).hexdigest())


def per_record(index, *, visibility="public"):
    """The fixture as a producer v3 index: explicit visibility and a per-record eligibility verdict."""
    for row in index["records"]:
        row.setdefault("visibility", visibility)
        row["training_eligible"] = A._training_eligible(row)
    eligible = [row for row in index["records"] if row["training_eligible"]]
    public = sum(row["visibility"] == "public" for row in eligible)
    index.update(
        schema=A.TRAINING_ARCHIVE,
        training_eligible_records=len(eligible),
        training_eligible_public_records=public,
        training_eligible_hidden_records=len(eligible) - public,
        training_ready=public > 0,
    )
    return index


def per_record_prepared(tmp_path, change=None):
    index, payload, binding = fixture(tmp_path)
    if change is not None:
        change(index, payload)
    per_record(index)
    index["gsim_pass_complete"] = sum(map(A._complete, index["records"]))
    index["timing_ready"] = index["gsim_pass_records"] == index["gsim_pass_complete"]
    path, checksum = write_archive(tmp_path, index, payload)
    binding["archive_sha256"] = checksum
    return A.inspect_archive(path, checksum), binding


def failed_capsule(index, payload):
    """The first capsule failed a host-side rule after its gSIM tier passed: not timing_ready."""
    for row in index["records"][:2]:
        row["result_status"] = "fail"
        original = json.loads(payload["run/" + row["result"]])
        original["status"] = "fail"
        payload["run/" + row["result"]] = json.dumps(original).encode()


def test_per_record_archive_prepares_measurements_of_failed_capsules(tmp_path):
    inventory, binding = per_record_prepared(tmp_path, failed_capsule)
    assert inventory["archive_schema"] == A.TRAINING_ARCHIVE
    assert inventory["timing_ready"] is False and inventory["training_ready"] is True
    assert inventory["training_eligible_public_records"] == 6
    plan = A.prepare(inventory, binding)
    assert all(row["label"]["status"] == "pass" for row in plan["observations"])
    data = A.attach(plan, reports_for(plan))
    assert len(data["records"]) == 6


def test_v2_archive_still_requires_every_capsule_to_pass(tmp_path):
    index, payload, binding = fixture(tmp_path)
    failed_capsule(index, payload)
    index["gsim_pass_complete"] = 4
    index["timing_ready"] = False
    path, checksum = write_archive(tmp_path, index, payload)
    binding["archive_sha256"] = checksum
    with pytest.raises(C.ContractError, match="incomplete timing archive"):
        A.prepare(A.inspect_archive(path, checksum), binding)


def test_hidden_records_never_enter_public_calibration(tmp_path):
    def hide_first(index, payload):
        for row in index["records"][:2]:
            row["visibility"] = "hidden"

    inventory, binding = per_record_prepared(tmp_path, hide_first)
    assert inventory["training_eligible_public_records"] == 4
    assert inventory["training_ready"] is True
    with pytest.raises(C.ContractError, match="hidden"):
        A.prepare(inventory, binding)


@pytest.mark.parametrize("damage", ["eligibility", "visibility", "summary"])
def test_per_record_declarations_are_recomputed(tmp_path, damage):
    index, payload, _ = fixture(tmp_path)
    per_record(index)
    if damage == "eligibility":
        index["records"][0]["training_eligible"] = False
    elif damage == "visibility":
        index["records"][0]["visibility"] = "unknown"
    else:
        index["training_eligible_hidden_records"] = 1
    path, checksum = write_archive(tmp_path, index, payload)
    with pytest.raises(C.ContractError):
        A.inspect_archive(path, checksum)


def test_executed_failed_repeat_cannot_be_dropped_but_unexecuted_result_can(tmp_path):
    def add(status):
        def change(index, payload):
            row = dict(index["records"][0], result=f"_qa_work/{status}/runs/suite/capsule0/capsule_result.json")
            row.update(status=status, cycles=None, cycles_measured=False, console_cycle_verified=False)
            index["records"].append(row)
            index["capsule_records"] += 1
            payload["run/" + row["result"]] = json.dumps({"status": "fail", "tiers": {}}).encode()

        return change

    inventory, binding = per_record_prepared(tmp_path, add("fail"))
    with pytest.raises(C.ContractError, match="omitted"):
        A.prepare(inventory, binding)
    binding["cases"][0]["results"].append("_qa_work/fail/runs/suite/capsule0/capsule_result.json")
    plan = A.prepare(inventory, binding)
    failed = [row for row in plan["observations"] if row["result"].startswith("_qa_work/fail/")]
    assert failed[0]["label"]["status"] == "unknown"
    (tmp_path / "unexecuted").mkdir()
    inventory, binding = per_record_prepared(tmp_path / "unexecuted", add("unavailable"))
    assert len(A.prepare(inventory, binding)["observations"]) == 6


def test_carried_verdict_is_not_a_passing_measurement(tmp_path):
    index, payload, _ = fixture(tmp_path)
    index["records"][1]["measured_now"] = False
    path, checksum = write_archive(tmp_path, index, payload)
    with pytest.raises(C.ContractError, match="summary counts"):
        A.inspect_archive(path, checksum)
    index["gsim_pass_records"] = index["gsim_pass_with_cycles"] = index["gsim_pass_complete"] = 5
    path, checksum = write_archive(tmp_path, index, payload)
    assert A.inspect_archive(path, checksum)["gsim_pass_records"] == 5
