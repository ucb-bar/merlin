"""A full Spike certificate must survive reopening exact bytes and reject stale tails."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.runtime.backends import output_identity, zephyr_model

_spec = importlib.util.spec_from_file_location(
    "source_output_certificate", repo_root() / "examples" / "kodiak" / "source_output_certificate.py"
)
assert _spec is not None and _spec.loader is not None
cert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cert)
flow = cert.flow


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    job = tmp_path / "run_123"
    bundle = tmp_path / "capture"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}\n")
    golden = np.array([[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype="<f4")
    np.save(bundle / "golden.npy", golden)
    job.mkdir()
    framework = job / "framework_result.json"
    framework.write_text('{"tool":"merlin-compile","run":"host"}\n')
    elf = job / "device" / "zephyr.elf"
    elf.parent.mkdir()
    elf.write_bytes(b"exact staged ELF")
    evidence_dir = job / "full_output" / "device"
    evidence_dir.mkdir(parents=True)
    raw = evidence_dir / "raw.bin"
    raw.write_bytes(golden.tobytes())
    hash_value = zephyr_model.out_hash_fnv1a32(golden)
    bits = golden.ravel()[:3].view("<u4").tolist()
    console = evidence_dir / "console.txt"
    console.write_text(
        f"METRIC build_hash abc123\nHASH fnv1a32 {golden.size} {hash_value}\nOUT 3 {' '.join(map(str, bits))}\nDONE\n"
    )
    identity = output_identity.output_identity(zephyr_model._parse_console(console.read_text(), 0))
    files = {
        "raw": raw,
        "console": console,
        "debugger_commands": evidence_dir / "commands.txt",
        "debugger_log": evidence_dir / "debugger.log",
        "disassembly": evidence_dir / "disassembly.txt",
        "spike_version": evidence_dir / "spike_version.txt",
    }
    files["debugger_commands"].write_text("resolve MERLIN_OUTPUT_PTR[0]; dump output\n")
    files["debugger_log"].write_text("HALT_PC=0x80001000 OUTPUT=0x80002000\n")
    files["disassembly"].write_text("80001000: halt after HASH and before OUT\nMERLIN_OUTPUT_PTR\n")
    files["spike_version"].write_text("Spike test version\n")
    spike = job / "spike"
    spike.write_bytes(b"pinned spike executable")
    capture_sha = flow._capture_sha256(bundle)
    manifest = {
        "schema": flow.SCHEMA,
        "name": "tinyllama_source",
        "bundle": str(bundle),
        "capture_sha256": capture_sha,
        "board": "firesim_kodiak_opu",
        "hw_config": "FireSimKodiakOPUConfig",
        "harts": 1,
        "legs": {
            "device": {
                "elf": str(elf),
                "elf_sha256": _sha(elf),
                "build_hash": "abc123",
                "spike_output_identity": identity,
            }
        },
        "source": {
            "dtype": "int8_w8a8",
            "capture_sha256": capture_sha,
            "reference": {
                "file": "golden.npy",
                "kind": "torchao_w8a8_framework",
                "sha256": _sha(bundle / "golden.npy"),
            },
            "framework_result": str(framework),
            "framework_result_sha256": _sha(framework),
        },
        "jobs": {},
    }
    (job / flow.MANIFEST).write_text(json.dumps(manifest))
    return job, bundle, files, spike, manifest


def _create(job, files, spike):
    return cert.create(
        job,
        "device",
        evidence_files=files,
        spike=spike,
        output_address=0x80002000,
        halt_pc=0x80001000,
    )


def test_complete_spike_certificate_rechecks_evidence_and_ignores_mutable_queue_jobs(tmp_path):
    job, _, files, spike, manifest = _job(tmp_path)
    certificate = _create(job, files, spike)
    result = cert.verify(certificate)
    assert result["verification"]["scope"] == "complete_spike_output_vs_torchao_golden"
    assert result["verification"]["full_gate"]["comparison_complete"] is True
    assert result["verification"]["per_token_argmax"] == [2, 2]
    manifest["jobs"] = {"device": 123456}
    (job / flow.MANIFEST).write_text(json.dumps(manifest))
    assert cert.verify(certificate)["plan_sha256"] == result["plan_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("raw_tail", "raw evidence changed"),
        ("elf", "staged ELF changed"),
        ("capture", "source capture changed"),
        ("reference", "TorchAO golden changed"),
        ("console", "console evidence changed"),
        ("debugger_missing", "missing or symlinked evidence file"),
    ],
)
def test_certificate_rejects_stale_or_missing_evidence(tmp_path, mutation, message):
    job, bundle, files, spike, _ = _job(tmp_path)
    certificate = _create(job, files, spike)
    if mutation == "raw_tail":
        raw = bytearray(files["raw"].read_bytes())
        raw[-1] ^= 1
        files["raw"].write_bytes(raw)
    elif mutation == "elf":
        (job / "device" / "zephyr.elf").write_bytes(b"different ELF")
    elif mutation == "capture":
        (bundle / "model.mlir").write_text("module { different }\n")
    elif mutation == "reference":
        np.save(bundle / "golden.npy", np.ones((2, 3), dtype="<f4"))
    elif mutation == "console":
        files["console"].write_text(files["console"].read_text().replace("abc123", "wrong"))
    elif mutation == "debugger_missing":
        files["debugger_log"].unlink()
    with pytest.raises(cert.CertificateError, match=message):
        cert.verify(certificate)


def test_prefix_only_dump_cannot_create_full_output_certificate(tmp_path):
    job, _, files, spike, _ = _job(tmp_path)
    files["raw"].write_bytes(files["raw"].read_bytes()[: 3 * 4])
    with pytest.raises(cert.CertificateError, match="complete f32 output"):
        _create(job, files, spike)


def test_corrupted_unprinted_tail_is_rejected_even_before_sealing(tmp_path):
    job, _, files, spike, _ = _job(tmp_path)
    raw = bytearray(files["raw"].read_bytes())
    raw[-1] ^= 1
    files["raw"].write_bytes(raw)
    with pytest.raises(cert.CertificateError, match="whole-output FNV differs"):
        _create(job, files, spike)


def test_same_run_console_and_pointer_evidence_are_required(tmp_path):
    job, _, files, spike, _ = _job(tmp_path)
    files["console"].write_text(files["console"].read_text().replace("abc123", "wrong"))
    with pytest.raises(cert.CertificateError, match="console image build hash"):
        _create(job, files, spike)
    job, _, files, spike, _ = _job(tmp_path / "other")
    files["debugger_commands"].write_text("dump fixed 0x80002000\n")
    with pytest.raises(cert.CertificateError, match="MERLIN_OUTPUT_PTR"):
        _create(job, files, spike)
