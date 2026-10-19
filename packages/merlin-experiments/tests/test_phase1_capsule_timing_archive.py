from __future__ import annotations

import base64
import hashlib
import json
import tarfile

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


def test_timing_selection_requires_one_real_comment_not_quoted_or_duplicate_text():
    from merlin_experiments.phase1 import capsule_timing_archive as archive

    marker = "/* merlin.kernel_call_timing.v1 selection_sha256=" + "a" * 64
    marker += " boundary=entry-through-completion record_b64=e30= */"
    assert archive._kernel_scopes('const char *text = "' + marker + '";') == []
    assert archive._kernel_scopes("// " + marker + "\n") == []
    assert archive._kernel_scopes(marker) == [("a" * 64, "e30=")]
    assert len(archive._kernel_scopes(marker + "\n" + marker)) == 2
    assert archive._kernel_scopes(marker.replace("selection_sha256=", "missing=")) == [("", "")]


def test_cycle_records_require_canonical_positive_ascii_integers():
    from merlin_experiments.phase1 import capsule_timing_archive as archive

    text = "\n".join(
        [
            "METRIC cycles 23",
            "METRIC cycles 0",
            "METRIC cycles 023",
            "METRIC cycles -1",
            "METRIC cycles ٢٣",
            "METRIC cycles 3 extra",
        ]
    )
    assert archive._console_cycles(text) == [23]
