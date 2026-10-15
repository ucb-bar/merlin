from __future__ import annotations

import json
import tarfile

import pytest
from merlin_experiments.phase1 import capsule_timing_archive as A


def _capsule(root, run, name, cycles):
    capsule = root / "_qa_work" / run / "runs" / "gemmini-capsule-bench" / name
    (capsule / "generated").mkdir(parents=True, exist_ok=True)
    (capsule / "generated" / "package_kernel.elf").write_bytes(name.encode())
    (capsule / "capsule_result.json").write_text(
        json.dumps(
            {
                "capsule": name,
                "tiers": {
                    "L3": {
                        "status": "pass",
                        "engine": "gsim",
                        "cycles": cycles,
                        "timing": {"sim_active_s": 1.5},
                        "sim_provenance": {"sha256": "emulator-digest"},
                    }
                },
            }
        )
    )
    return capsule


def test_archive_requires_real_cycles_and_keeps_capsule_artifacts(tmp_path):
    first = _capsule(tmp_path, "runs_r0000_t000001", "A", 1024)
    second = _capsule(tmp_path, "runs_r0000_t000001", "B", 0)
    report = A.audit(tmp_path)
    assert report["gsim_pass_records"] == 2
    assert report["gsim_pass_with_cycles"] == 1
    assert report["timing_ready"] is False
    output = tmp_path / "capsules.tar"
    with pytest.raises(A.TimingArchiveError, match="1/2"):
        A.archive(tmp_path, output)
    assert not output.exists()

    _capsule(tmp_path, "runs_r0000_t000001", "B", 2048)
    result = A.archive(tmp_path, output)
    assert result["timing_ready"] is True
    assert result["artifact_files"] == 4
    with tarfile.open(output) as tar:
        names = set(tar.getnames())
        assert "index.json" in names
        for capsule in (first, second):
            relative = capsule.relative_to(tmp_path).as_posix()
            assert f"run/{relative}/capsule_result.json" in names
            assert f"run/{relative}/generated/package_kernel.elf" in names
        index = json.load(tar.extractfile("index.json"))
        assert index["gsim_pass_with_cycles"] == 2


def test_archive_refuses_symlinked_capsule_artifact(tmp_path):
    capsule = _capsule(tmp_path, "runs_r0000_t000001", "A", 10)
    (capsule / "generated" / "indirect").symlink_to(capsule / "capsule_result.json")
    with pytest.raises(A.TimingArchiveError, match="symlink"):
        A.archive(tmp_path, tmp_path / "capsules.tar")
