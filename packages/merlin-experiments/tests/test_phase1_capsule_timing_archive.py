from __future__ import annotations

import base64
import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from merlin_experiments.phase1 import capsule_timing_archive as A


def _capsule(root, run, name, cycles):
    capsule = root / "_qa_work" / run / "runs" / "gemmini-capsule-bench" / name
    (capsule / "generated").mkdir(parents=True, exist_ok=True)
    (capsule / "artifacts").mkdir(parents=True, exist_ok=True)
    (capsule / "artifacts" / "rtl_console.log").write_text(
        f"METRIC cycles {cycles}\nDONE\n" if type(cycles) is int and cycles > 0 else "DONE\n"
    )
    (capsule / "generated" / "package_kernel.elf").write_bytes(name.encode())
    (capsule / "generated" / "input.interface.mlir").write_text("module {}\n")
    (capsule / "generated" / "command_buffer.json").write_text("{}\n")
    (capsule / "generated" / "instruction_trace.json").write_text("[]\n")
    (capsule / "generated" / "numeric_report.yaml").write_text("status: pass\n")
    selection = {
        "schema": "merlin.kernel_call_timing.v1",
        "selection_sha256": "a" * 64,
        "boundary": "immediately before kernel entry through explicitly selected completion",
        "counter_semantics": {
            "unit": "cycles",
            "width_bits": 64,
            "wrap": "modulo",
            "clock": "selected-clock",
            "source_pins": [{"path": "/original/source", "sha256": "b" * 64}],
        },
    }
    encoded = base64.b64encode(json.dumps(selection).encode()).decode()
    (capsule / "generated" / "harness.c").write_text(
        f"/* merlin.kernel_call_timing.v1 selection_sha256={'a' * 64} "
        f"boundary=entry-through-completion record_b64={encoded} */\n"
    )
    (capsule / "capsule_result.json").write_text(
        json.dumps(
            {
                "capsule": name,
                "status": "pass",
                "command_buffer_artifact": {
                    "artifact_sha256": "c" * 64,
                    "command_buffer": {"tensors": {"input": {"dtype": "i8", "shape": [4, 8]}}},
                },
                "original_compiler_child": {"candidate_sha256": "d" * 64},
                "tiers": {
                    "L3": {
                        "status": "pass",
                        "engine": "gsim",
                        "derived_from_rtl": True,
                        "cycle_accurate": True,
                        "cycles": cycles,
                        "timing": {"sim_active_s": 1.5},
                        "sim_provenance": {"sha256": "e" * 64},
                    }
                },
            }
        )
    )
    return capsule


def test_archive_requires_real_cycles_and_keeps_capsule_artifacts(tmp_path):
    first = _capsule(tmp_path, "runs_r0000_t000001", "A", 1024)
    second = _capsule(tmp_path, "fruns_900", "B", 0)
    report = A.audit(tmp_path)
    assert report["gsim_pass_records"] == 2
    assert report["gsim_pass_with_cycles"] == 1
    assert report["timing_ready"] is False
    output = tmp_path / "capsules.tar"
    with pytest.raises(A.TimingArchiveError, match="1/2"):
        A.archive(tmp_path, output)
    assert not output.exists()

    _capsule(tmp_path, "fruns_900", "B", 2048)
    result = A.archive(tmp_path, output)
    assert result["timing_ready"] is True
    assert result["gsim_pass_complete"] == 2
    assert result["artifact_files"] == 16
    assert result["archive_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with tarfile.open(output) as tar:
        names = set(tar.getnames())
        assert "index.json" in names
        for capsule in (first, second):
            relative = capsule.relative_to(tmp_path).as_posix()
            assert f"run/{relative}/capsule_result.json" in names
            assert f"run/{relative}/generated/package_kernel.elf" in names
        index = json.load(tar.extractfile("index.json"))
        assert index["gsim_pass_with_cycles"] == 2
        assert all(row["elf_sha256"] for row in index["records"])
        assert all(row["shape"] == {"input": {"dtype": "i8", "shape": [4, 8]}} for row in index["records"])
        assert all(row["command_buffer_sha256"] == "c" * 64 for row in index["records"])
        assert all(row["candidate_sha256"] == "d" * 64 for row in index["records"])
        assert all(row["timing_selection_sha256"] == "a" * 64 for row in index["records"])
        assert all(
            row["timing_selection"]["counter_semantics"]["clock"] == "selected-clock" for row in index["records"]
        )
        assert all(row["console_cycle_verified"] for row in index["records"])


def test_archive_refuses_symlinked_capsule_artifact(tmp_path):
    capsule = _capsule(tmp_path, "runs_r0000_t000001", "A", 10)
    (capsule / "generated" / "indirect").symlink_to(capsule / "capsule_result.json")
    with pytest.raises(A.TimingArchiveError, match="symlink"):
        A.archive(tmp_path, tmp_path / "capsules.tar")


def test_archive_requires_elf_and_simulator_identity(tmp_path):
    capsule = _capsule(tmp_path, "runs_r0000_t000001", "A", 10)
    (capsule / "generated" / "package_kernel.elf").unlink()
    report = A.audit(tmp_path)
    assert report["gsim_pass_with_cycles"] == 1
    assert report["gsim_pass_complete"] == 0
    assert report["timing_ready"] is False
    with pytest.raises(
        A.TimingArchiveError, match="ELF, compiler, command-buffer, simulator and kernel-scope identities"
    ):
        A.archive(tmp_path, tmp_path / "capsules.tar")


def test_archive_requires_compiler_identity_for_spike_import(tmp_path):
    capsule = _capsule(tmp_path, "runs_r0000_t000001", "A", 10)
    result = capsule / "capsule_result.json"
    document = json.loads(result.read_text())
    document["original_compiler_child"].pop("candidate_sha256")
    result.write_text(json.dumps(document))
    report = A.audit(tmp_path)
    assert report["gsim_pass_with_cycles"] == 1
    assert report["gsim_pass_complete"] == 0
    assert report["timing_ready"] is False


def test_archive_refuses_unscoped_cycle_marker(tmp_path):
    capsule = _capsule(tmp_path, "runs_r0000_t000001", "A", 10)
    (capsule / "generated" / "harness.c").write_text("int main(void){return 0;}\n")
    report = A.audit(tmp_path)
    assert report["gsim_pass_with_cycles"] == 1
    assert report["gsim_pass_complete"] == 0
    assert report["records"][0]["cycle_scope"] is None


def test_archive_refuses_duplicate_console_cycle_markers(tmp_path):
    capsule = _capsule(tmp_path, "runs_r0000_t000001", "A", 10)
    with (capsule / "artifacts" / "rtl_console.log").open("a") as stream:
        stream.write("METRIC cycles 10\n")
    report = A.audit(tmp_path)
    assert report["records"][0]["console_cycle_verified"] is False
    assert report["timing_ready"] is False


def test_run_finalization_writes_audit_and_training_tar_only_when_ready(tmp_path):
    _capsule(tmp_path, "runs_r0000_t000001", "A", None)
    status = A.finalize_run(tmp_path)
    assert status["timing_ready"] is False
    assert (tmp_path / "capsule_timing_audit.json").is_file()
    assert not (tmp_path / "capsule_timing.tar").exists()
    evidence = status["evidence_archive"]
    with tarfile.open(evidence) as raw:
        index = json.load(raw.extractfile("index.json"))
        assert index["timing_ready"] is False
        assert index["export_scope"].startswith("diagnostic_all_results")
        assert any(name.endswith("/capsule_result.json") for name in raw.getnames())
    assert json.loads((tmp_path / "capsule_timing_archive_status.json").read_text()) == status

    _capsule(tmp_path, "runs_r0000_t000001", "A", 100)
    status = A.finalize_run(tmp_path)
    assert status["timing_ready"] is True
    assert status["archive_sha256"] == hashlib.sha256((tmp_path / "capsule_timing.tar").read_bytes()).hexdigest()
    with tarfile.open(tmp_path / "capsule_timing.tar") as tar:
        assert "index.json" in tar.getnames()
    assert A.finalize_run(tmp_path) == status
    _capsule(tmp_path, "fruns_900", "B", 200)
    extended = A.finalize_run(tmp_path)
    assert extended["timing_ready"] is True
    assert extended["archive"] != status["archive"]
    assert (tmp_path / "capsule_timing.tar").is_file()


def test_carried_rtl_pass_is_not_a_new_timing_measurement(tmp_path):
    _capsule(tmp_path, "fresh", "A", 10)
    carried = _capsule(tmp_path, "reused", "B", 20)
    result = carried / "capsule_result.json"
    document = json.loads(result.read_bytes())
    document["tiers"]["L3"]["carried"] = {"source": "original exact program certificate"}
    result.write_text(json.dumps(document))
    (carried / "artifacts" / "rtl_console.log").unlink()
    audit = A.audit(tmp_path)
    assert audit["timing_ready"] is True
    assert audit["gsim_pass_records"] == 1
    assert audit["gsim_reused_records"] == 1
    assert audit["gsim_pass_complete"] == 1


def _graded(root, area, name, cycles, *, result_status="pass"):
    """A capsule graded into a formal ``grading_<visibility>`` suite instead of ``_qa_work``."""
    capsule = _capsule(root, "unused", name, cycles)
    target = root / area / "runs" / "gemmini-capsule-bench" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    capsule.rename(target)
    result = target / "capsule_result.json"
    document = json.loads(result.read_text())
    document["status"] = result_status
    if result_status != "pass":
        document["failure"] = {"category": "host_compute_in_accelerator_group"}
    result.write_text(json.dumps(document))
    return target


def _record(capsule, name, *, executable, inputs=(), outputs=(), finished=0):
    """One retained process record, as `merlin.common.invocation_record` writes it."""
    folder = capsule / "generated" / "invocations" / name
    folder.mkdir(parents=True)

    def pins(rows):
        return [{"path": str(capsule / "generated" / file), "sha256": digest} for file, digest in rows]

    (folder / "invocation.json").write_text(
        json.dumps(
            {
                "schema": "merlin.invocation_record.v1",
                "executable": {"path": "/engine", "sha256": executable},
                "inputs": pins(inputs),
                "outputs": pins(outputs),
                "finished_ns": finished,
            }
        )
    )


def test_formal_grading_suites_are_indexed_with_visibility(tmp_path):
    _graded(tmp_path, "grading_public", "P", 300)
    _graded(tmp_path, "grading_hidden", "H", 400)
    report = A.audit(tmp_path)
    rows = {row["capsule"]: row for row in report["records"]}
    assert report["schema"] == "merlin.phase1.capsule_timing_archive.v3"
    assert report["records_by_area"] == {"authoring": 0, "grading_public": 1, "grading_hidden": 1}
    assert (rows["P"]["area"], rows["P"]["visibility"]) == ("grading_public", "public")
    assert (rows["H"]["area"], rows["H"]["visibility"]) == ("grading_hidden", "hidden")
    assert report["training_eligible_records"] == 2
    assert report["training_eligible_public_records"] == 1
    assert report["training_eligible_hidden_records"] == 1
    subset = A.training_index(report)
    assert [row["capsule"] for row in subset["records"]] == ["P"]
    assert subset["export_scope"] == "training_eligible_public"
    assert subset["source_audit"]["training_eligible_hidden_records"] == 1


def test_training_eligibility_is_per_record_and_independent_of_capsule_verdict(tmp_path):
    _graded(tmp_path, "grading_public", "FAILED_CAPSULE", 300, result_status="fail")
    unattributed = _graded(tmp_path, "grading_public", "NO_COMPILER", 500)
    result = unattributed / "capsule_result.json"
    document = json.loads(result.read_text())
    document["original_compiler_child"] = {}
    result.write_text(json.dumps(document))
    report = A.audit(tmp_path)
    rows = {row["capsule"]: row for row in report["records"]}
    # The capsule failed a host-side rule, but its gSIM tier measured and verified these bytes.
    assert rows["FAILED_CAPSULE"]["training_eligible"] is True
    assert rows["FAILED_CAPSULE"]["result_failure"] == "host_compute_in_accelerator_group"
    assert rows["NO_COMPILER"]["training_eligible"] is False
    assert rows["NO_COMPILER"]["training_exclusions"] == ["identity_missing:candidate"]
    assert report["timing_ready"] is False
    assert report["training_ready"] is True
    assert report["training_exclusions"] == {"identity_missing:candidate": 1}


def test_training_archive_keeps_indexed_evidence_and_executed_repeats_only(tmp_path):
    keep = _capsule(tmp_path, "runs_a", "A", 100)
    repeat = _capsule(tmp_path, "runs_b", "A", 100)
    failed = json.loads((repeat / "capsule_result.json").read_text())
    failed["tiers"]["L3"]["status"] = "fail"
    (repeat / "capsule_result.json").write_text(json.dumps(failed))
    gated = _capsule(tmp_path, "runs_c", "A", None)
    unexecuted = json.loads((gated / "capsule_result.json").read_text())
    unexecuted["tiers"] = {}
    (gated / "capsule_result.json").write_text(json.dumps(unexecuted))
    (keep / "generated" / "unindexed_scratch.bin").write_bytes(b"x" * 100)
    destination = tmp_path / "out" / "training.tar"
    result = A.training_archive(tmp_path, destination)
    assert result["capsule_records"] == 2
    assert result["training_eligible_public_records"] == 1
    assert result["gsim_pass_records"] == 1
    with tarfile.open(destination) as tar:
        names = set(tar.getnames())
        index = json.load(tar.extractfile("index.json"))
    assert {row["result"] for row in index["records"]} == {
        (keep / "capsule_result.json").relative_to(tmp_path).as_posix(),
        (repeat / "capsule_result.json").relative_to(tmp_path).as_posix(),
    }
    assert not any(name.endswith("unindexed_scratch.bin") for name in names)
    for row in index["records"]:
        assert "run/" + row["result"] in names
        assert all("run/" + entry["path"] in names for entry in row["artifacts"].values() if entry)
    with pytest.raises(A.TimingArchiveError, match="already exists"):
        A.training_archive(tmp_path, destination)


def test_training_archive_refuses_evidence_changed_after_audit(tmp_path):
    capsule = _capsule(tmp_path, "runs_a", "A", 100)
    index = A.audit(tmp_path)
    (capsule / "generated" / "package_kernel.elf").write_bytes(b"rebuilt")
    with pytest.raises(A.TimingArchiveError, match="changed after its audit"):
        A.training_archive(tmp_path, tmp_path / "training.tar", index=index)
    assert not (tmp_path / "training.tar").exists()


def test_per_engine_executions_are_recovered_from_process_records(tmp_path):
    capsule = _capsule(tmp_path, "runs_a", "A", 100)
    result = capsule / "capsule_result.json"
    document = json.loads(result.read_text())
    document["tiers"]["L2"] = {"status": "pass", "engine": "spike", "sim_provenance": {"sha256": "f" * 64}}
    result.write_text(json.dumps(document))
    elf = hashlib.sha256((capsule / "generated" / "package_kernel.elf").read_bytes()).hexdigest()
    harness = hashlib.sha256((capsule / "generated" / "harness.c").read_bytes()).hexdigest()
    spike_elf, kernel = "1" * 64, "2" * 64
    builder = "0" * 64
    _record(capsule, "a", executable=builder, inputs=[("harness.c", "3" * 64)], outputs=[("harness.o", "4" * 64)])
    _record(
        capsule,
        "b",
        executable=builder,
        inputs=[("kernel.o", kernel), ("harness.o", "4" * 64)],
        outputs=[("package_kernel.elf", spike_elf)],
        finished=1,
    )
    _record(capsule, "c", executable="f" * 64, inputs=[("package_kernel.elf", spike_elf)], finished=2)
    _record(
        capsule, "d", executable=builder, inputs=[("harness.c", harness)], outputs=[("harness.o", "5" * 64)], finished=3
    )
    _record(
        capsule,
        "e",
        executable=builder,
        inputs=[("kernel.o", kernel), ("harness.o", "5" * 64)],
        outputs=[("package_kernel.elf", elf)],
        finished=4,
    )
    _record(capsule, "f", executable="e" * 64, inputs=[("package_kernel.elf", elf)], finished=5)
    row = A.audit(tmp_path)["records"][0]
    assert row["executions"]["L2"]["elf_sha256"] == spike_elf
    assert row["executions"]["L2"]["harness_sha256"] == "3" * 64
    assert row["executions"]["L3"] == {
        "source": "invocation_records",
        "engine_sha256": "e" * 64,
        "runs": 1,
        "elf_sha256": elf,
        "kernel_object_sha256": kernel,
        "harness_sha256": harness,
    }
    assert row["join"]["same_elf"] is False and row["join"]["same_kernel_object"] is True
    assert row["join"]["gsim_elf_sha256"] == elf and row["join"]["spike_elf_sha256"] == spike_elf
    assert row["training_eligible"] is True
    # A later rebuild left a different program in place: the gSIM label no longer describes these bytes.
    (capsule / "generated" / "package_kernel.elf").write_bytes(b"rebuilt for another tier")
    row = A.audit(tmp_path)["records"][0]
    assert row["training_exclusions"] == ["current_program_differs_from_gsim_execution"]


def test_finalize_writes_training_archive_for_an_incomplete_run(tmp_path, capsys):
    _graded(tmp_path, "grading_public", "PASS", 300)
    _graded(tmp_path, "grading_public", "HOST_FAIL", 400, result_status="fail")
    _graded(tmp_path, "grading_hidden", "HIDDEN", 500)
    assert A.main([str(tmp_path), "--finalize"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["timing_ready"] is False and status["training_ready"] is True
    assert status["training_eligible_public_records"] == 2
    assert status["training_eligible_hidden_records"] == 1
    training = Path(status["training_archive"])
    assert training.name.startswith("capsule_timing_training_")
    assert status["training_archive_sha256"] == hashlib.sha256(training.read_bytes()).hexdigest()
    with tarfile.open(training) as tar:
        index = json.load(tar.extractfile("index.json"))
        assert not any("grading_hidden" in name for name in tar.getnames())
    assert {row["capsule"] for row in index["records"]} == {"PASS", "HOST_FAIL"}
    # Re-finalizing an unchanged run (an operator after a stopped supervisor) rewrites nothing.
    assert A.finalize_run(tmp_path) == status
