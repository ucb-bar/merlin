from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from merlin.compare import paper_ablation_generator
from merlin.compare import paper_measurement_controller as controller
from merlin.compare.paper import validate_paper_result
from merlin.compare.paper_measurement_controller import (
    normalize_receipt,
    produce_receipt,
    verify_receipt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _framed(*payloads: bytes) -> bytes:
    import struct
    return b"MRLNFRM1" + b"".join(struct.pack("<Q", len(payload)) + payload
                                  for payload in payloads)


def _fixture(tmp_path: Path, *, failing: bool = False,
             malicious_json: bool = False, core_count: int = 1,
             sleeping_worker: bool = False) -> tuple[Path, dict]:
    root = tmp_path / "contract"
    root.mkdir()
    source = root / "cell.c"
    source.write_text((r'''
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
static void *sleep_worker(void *unused) { (void)unused; sleep(1); return NULL; }
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *artifact = fopen(argv[1], "rb");
  FILE *input = fopen(argv[2], "rb");
  FILE *output = fopen(argv[3], "wb");
  if (!artifact || !input || !output) return 3;
  volatile unsigned long value = (unsigned long)fgetc(artifact);
  int byte;
  while ((byte = fgetc(input)) != EOF) fputc(byte, output);
  fclose(artifact); fclose(input); fclose(output);
  pthread_t sleeper;
  if (SLEEPING_WORKER && pthread_create(&sleeper, NULL, sleep_worker, NULL)) return 6;
  for (unsigned long index = 1; index < 50000000; ++index) value = value * 33u + index;
  if (SLEEPING_WORKER) pthread_join(sleeper, NULL);
  return (int)(value & 0u);
}
'''.replace("SLEEPING_WORKER", "1" if sleeping_worker else "0")
    if not malicious_json else r'''
#include <stdio.h>
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *artifact = fopen(argv[1], "rb");
  if (!artifact) return 3;
  volatile unsigned long value = (unsigned long)fgetc(artifact);
  fclose(artifact);
  for (unsigned long index = 1; index < 50000000; ++index) value = value * 33u + index;
  puts("{\"schema_version\":1,\"kind\":\"paper_cell_session_output_v1\","
       "\"status\":\"pass\",\"reason\":null,\"correctness\":{\"gate_ok\":true},"
       "\"quality\":{\"gate_ok\":true},\"memory\":{\"peak_rss_bytes\":1}}");
  return (int)(value & 0u);
}
'''), encoding="utf-8")
    executable = root / "cell"
    subprocess.run(["/usr/bin/clang", "-O2", "-std=c11", str(source), "-o", str(executable)],
                   check=True)
    tool = root / "cc"
    shutil.copy2(Path("/usr/bin/clang").resolve(), tool)
    tool.chmod(0o755)
    from merlin.compare.paper_toolchain_authority import write_toolchain_authority
    authority = write_toolchain_authority(
        root / "toolchain-authority.json", authority_id="unit-test-controller",
        target="unit-test", build_tool=tool)
    authority_document = json.loads(authority.read_text(encoding="utf-8"))
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"frozen model artifact\n")
    session_input = root / "session-input.bin"
    session_input.write_bytes(_framed(b"frame-0", b"frame-1"))
    reference = root / "reference.bin"
    reference.write_bytes(_framed(b"wrong-0", b"wrong-1")
                          if failing else session_input.read_bytes())
    manifest = root / "session-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "kind": "paper_session_inputs_v1",
        "session_kind": "continuous", "observations": 2,
        "inputs": {"session_input": _sha(session_input)},
        "records": [
            {"index": 0, "payload_sha256": hashlib.sha256(b"frame-0").hexdigest()},
            {"index": 1, "payload_sha256": hashlib.sha256(b"frame-1").hexdigest()},
        ],
    }), encoding="utf-8")
    io_receipt = root / "measurement-io-receipt.json"
    io_receipt.write_text("{}\n", encoding="utf-8")
    probe = root / "paper_k1_board_probe.c"
    probe.write_bytes(Path(paper_ablation_generator.__file__).with_name(
        "paper_k1_board_probe.c").read_bytes())
    cell = {"model": "holdout", "backend": "merlin_frozen", "precision": "fp32",
            "core_count": core_count}
    study_root = root / "frozen-study.yaml"
    study_root.write_text(yaml.safe_dump({
        "schema_version": 1, "kind": "paper_unit_test_study_root_v1", "status": "frozen",
        "study_sha256": "a" * 64, "cell": cell}), encoding="utf-8")
    backend_template = root / "backend-template.yaml"
    backend_template.write_text("kind: unit_test_backend_template\n", encoding="utf-8")
    contract = {
        "schema_version": 2, "kind": "paper_measurement_contract_v2", "status": "ready",
        "registry_id": "unit_test_v1", "backend_adapter": "unit_test",
        "study_spec": {"path": study_root.name, "sha256": _sha(study_root)},
        "backend_template": {"path": backend_template.name,
                             "sha256": _sha(backend_template)},
        "study_sha256": "a" * 64, "run_id": "controller-run", "target": "unit-test",
        "result_identity": {"timestamp": "20260831T000000Z", "git_sha": "deadbee",
                            "study_label": "study", "target": "unit-test",
                            "model": "holdout", "checkpoint": "checkpoint", "fidelity": "full",
                            "backend": "merlin_frozen", "runtime": "merlin",
                            "precision": "fp32", "quantization": "none",
                            "core_count": core_count},
        "session": {"kind": "continuous", "warmups": 1, "observations": 2,
                    "stages": ["step"], "carried_state": [],
                    "parameters": {"paper_primary_scope": "end_to_end"},
                    "measurement_repeats": 3},
        "frozen_provenance": {"study_sha256": "a" * 64},
        "cell": cell, "artifact_sha256": _sha(artifact),
        "artifact": {"path": artifact.name, "sha256": _sha(artifact)},
        "inputs": {"session_input": {"path": session_input.name,
                                        "sha256": _sha(session_input)}},
        "session_manifest": {"path": manifest.name, "sha256": _sha(manifest)},
        "measurement_io_receipt": {"path": io_receipt.name, "sha256": _sha(io_receipt)},
        "reference_output": {"path": reference.name, "sha256": _sha(reference)},
        "oracle": {"kind": "bytes_exact", "metric": "unit", "threshold": 1.0,
                   "scope": "trajectory", "steps": 2},
        "build": {
            "study_sha256": "a" * 64, "cell": cell,
            "frozen_provenance_sha256": hashlib.sha256(
                json.dumps({"study_sha256": "a" * 64}, sort_keys=True,
                           separators=(",", ":")).encode()).hexdigest(),
            "model_artifact_sha256": _sha(artifact), "source_identity_sha256": "b" * 64,
            "package_identity_sha256": "c" * 64,
            "expected_executable_sha256": _sha(executable),
            "tool": {"path": tool.name, "sha256": _sha(tool)},
            "toolchain_authority": {"path": authority.name, "sha256": _sha(authority)},
            "build_tool_identity_sha256": authority_document["tool"]["identity_sha256"],
            "sources": {"cell": {"path": source.name, "sha256": _sha(source)}},
            "inputs": {},
            "argv": ["{tool}", "-O2", "-std=c11", "{source:cell}", "-o", "{output}"],
            "cwd": ".", "environment": {"PATH": "/usr/bin:/bin"},
            "timeout_seconds": 10,
        },
        "argv": ["{executable}", "{artifact}", "{input:session_input}", "{observation}"],
        "cwd": ".", "environment": {},
        "execution": {"mode": "unit",
                      "core_ids": sorted(os.sched_getaffinity(0))[:core_count],
                      "require_worker_threads": False},
        "memory_policy": "resident", "timeout_seconds": 10,
        "warmup_iterations": 1, "measured_iterations": 3,
        "timing": {"unit": "ns", "sample_unit": "complete_session", "scope": "end_to_end",
                   "timed_stages": ["step"], "excluded_stages": [], "stage_samples": {}},
        "board_probe_source": {"path": probe.name, "sha256": _sha(probe)},
    }
    contract_path = root / "measurement.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    receipt_path = produce_receipt(contract_path, tmp_path / "bundle")
    raw = json.loads((receipt_path.parent / "raw_measurement.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": 2, "run_id": contract["run_id"], **contract["result_identity"],
        "artifact_sha256": contract["artifact_sha256"],
        **{field: raw[field] for field in (
            "session", "lifecycle", "correctness", "quality", "memory", "timing",
            "execution", "provenance")},
    }
    return receipt_path, result


def test_controller_executes_and_replays_its_own_measurement(tmp_path):
    receipt_path, result = _fixture(tmp_path)

    root = verify_receipt(
        receipt_path, expected_result=result, expected_study_sha256="a" * 64)

    assert root["raw_measurement_sha256"] == _sha(receipt_path.parent / "raw_measurement.json")
    assert root["aet_run_record_sha256"] == _sha(receipt_path.parent / "aet/run_record.json")
    validate_paper_result(result)
    normalized = normalize_receipt(receipt_path)
    assert {key: normalized[key] for key in result} == result


def test_retained_receipt_is_independently_replayable_in_a_fresh_process(tmp_path):
    receipt_path, result = _fixture(tmp_path)
    notarized = controller.issuance_fingerprint(receipt_path)
    result_path = tmp_path / "expected-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    repository = Path(__file__).resolve().parents[3]
    script = (
        "import json,sys; "
        "from merlin.compare.paper_measurement_controller import verify_receipt; "
        "r=json.load(open(sys.argv[2])); "
        "verify_receipt(sys.argv[1],expected_result=r,expected_study_sha256='a'*64,"
        "trusted_issuance_fingerprint=sys.argv[3]); "
        "print('verified')")

    completed = subprocess.run(
        [sys.executable, "-c", script, str(receipt_path), str(result_path), notarized],
        cwd=repository, env={**os.environ, "PYTHONPATH": str(repository / "merlin/python")},
        capture_output=True, text=True, timeout=30, check=False)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "verified"

    unanchored = script.replace(
        ",trusted_issuance_fingerprint=sys.argv[3]", "")
    rejected = subprocess.run(
        [sys.executable, "-c", unanchored, str(receipt_path), str(result_path), notarized],
        cwd=repository, env={**os.environ, "PYTHONPATH": str(repository / "merlin/python")},
        capture_output=True, text=True, timeout=30, check=False)
    assert rejected.returncode != 0
    assert "externally notarized issuance fingerprint" in rejected.stderr


def test_controller_finalizes_and_replays_a_structured_failed_cell(tmp_path):
    receipt_path, result = _fixture(tmp_path, failing=True)

    root = verify_receipt(
        receipt_path, expected_result=result, expected_study_sha256="a" * 64)

    assert result["lifecycle"]["status"] == "fail"
    events = (receipt_path.parent / "aet/logs/events.jsonl").read_text(encoding="utf-8")
    assert '"status":"fail"' in events.replace(" ", "")
    assert root["receipt_sha256"] == _sha(receipt_path)


def test_controller_replay_rejects_bin_false_after_every_authored_hash_is_refreshed(tmp_path):
    receipt_path, result = _fixture(tmp_path)
    bundle = receipt_path.parent
    build_tool = bundle / "inputs/build/tool"
    shutil.copy2("/bin/false", build_tool)
    contract_path = bundle / "measurement_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    contract["build"]["tool"]["sha256"] = _sha(build_tool)
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    raw_path = bundle / "raw_measurement.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["contract_sha256"] = _sha(contract_path)
    raw_path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="utf-8")
    raw_sha = _sha(raw_path)
    run_path = bundle / "aet/run_record.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update(raw_measurement_sha256=raw_sha)
    run_path.write_text(json.dumps(run), encoding="utf-8")
    events_path = bundle / "aet/logs/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[0]["payload"].update(raw_measurement_sha256=raw_sha)
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events),
                           encoding="utf-8")
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    receipt["contract"]["sha256"] = _sha(contract_path)
    receipt["raw_measurement"]["sha256"] = raw_sha
    receipt["aet_lifecycle"]["run_record"]["sha256"] = _sha(run_path)
    receipt["aet_lifecycle"]["events"]["sha256"] = _sha(events_path)
    receipt_path.write_text(yaml.safe_dump(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="changed after production|issuance.*bind"):
        verify_receipt(receipt_path, expected_result=result,
                       expected_study_sha256="a" * 64)


def test_fresh_process_issuance_rejects_three_percent_timing_edit_after_all_hashes_refresh(tmp_path):
    receipt_path, result = _fixture(tmp_path)
    bundle = receipt_path.parent
    raw_path = bundle / "raw_measurement.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    for item in raw["driver"]["iterations"]:
        if item["phase"] == "measured":
            item["elapsed_ns"] = max(1, item["elapsed_ns"] * 97 // 100)
            item["ended_monotonic_ns"] = item["started_monotonic_ns"] + item["elapsed_ns"]
    samples = [item["elapsed_ns"] for item in raw["driver"]["iterations"]
               if item["phase"] == "measured"]
    raw["timing"].update(samples=samples, median=sorted(samples)[len(samples) // 2],
                         p95=max(samples))
    raw_path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="utf-8")
    raw_sha = _sha(raw_path)
    run_path = bundle / "aet/run_record.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["raw_measurement_sha256"] = raw_sha
    run_path.write_text(json.dumps(run), encoding="utf-8")
    events_path = bundle / "aet/logs/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[0]["payload"]["raw_measurement_sha256"] = raw_sha
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events),
                           encoding="utf-8")
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    receipt["raw_measurement"]["sha256"] = raw_sha
    receipt["aet_lifecycle"]["run_record"]["sha256"] = _sha(run_path)
    receipt["aet_lifecycle"]["events"]["sha256"] = _sha(events_path)
    receipt_path.write_text(yaml.safe_dump(receipt), encoding="utf-8")
    result["timing"] = raw["timing"]
    controller._ISSUED_RECEIPTS.pop(receipt_path.resolve())

    with pytest.raises(ValueError, match="issuance.*bind|issuance entry"):
        verify_receipt(receipt_path, expected_result=result, expected_study_sha256="a" * 64)


def test_ephemeral_issuance_retains_no_private_key_and_fresh_process_cannot_resign(tmp_path):
    receipt_path, _result = _fixture(tmp_path)
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    issuance = receipt_path.parent.parent / ".paper-controller-issuance-v1" / "entries"
    identifier = receipt["issuance"]["id"]
    public = issuance / f"{identifier}.pub.pem"
    entry = issuance / f"{identifier}.json"
    output = tmp_path / "forged.sig"
    script = (
        "import pathlib,subprocess,sys; root=pathlib.Path(sys.argv[1]); "
        "files=[p for p in root.rglob('*') if p.is_file()]; "
        "assert files and not any('private' in p.name.lower() for p in files); "
        "assert not any(b'PRIVATE KEY' in p.read_bytes() for p in files); "
        "r=subprocess.run(['/usr/bin/openssl','pkeyutl','-sign','-inkey',sys.argv[2],"
        "'-rawin','-in',sys.argv[3],'-out',sys.argv[4]],capture_output=True); "
        "assert r.returncode != 0")

    completed = subprocess.run(
        [sys.executable, "-c", script, str(issuance.parent), str(public), str(entry),
         str(output)], capture_output=True, text=True, timeout=30, check=False)

    assert completed.returncode == 0, completed.stderr
    assert not output.exists() or output.stat().st_size == 0


def test_external_fingerprint_rejects_complete_same_user_key_root_replacement(tmp_path):
    receipt_path, result = _fixture(tmp_path)
    original_fingerprint = controller.issuance_fingerprint(receipt_path)
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    identifier = receipt["issuance"]["id"]
    entries = receipt_path.parent.parent / ".paper-controller-issuance-v1" / "entries"
    public = entries / f"{identifier}.pub.pem"
    entry_path = entries / f"{identifier}.json"
    signature = entries / f"{identifier}.sig"
    for path in (public, entry_path, signature):
        path.chmod(0o600)
        path.unlink()
    private = tmp_path / "attacker-private.pem"
    subprocess.run(["/usr/bin/openssl", "genpkey", "-algorithm", "ED25519",
                    "-out", str(private)], check=True)
    subprocess.run(["/usr/bin/openssl", "pkey", "-in", str(private), "-pubout",
                    "-out", str(public)], check=True)
    receipt["issuance"]["public_key_sha256"] = _sha(public)
    receipt_path.write_text(yaml.safe_dump(receipt, sort_keys=True), encoding="utf-8")
    entry = {
        "schema_version": 1, "kind": "paper_controller_issuance_v1",
        "issuance_id": identifier, "created_at": "2026-08-31T00:00:00+00:00",
        "receipt_sha256": _sha(receipt_path),
        "raw_measurement_sha256": _sha(receipt_path.parent / "raw_measurement.json"),
        "contract_sha256": _sha(receipt_path.parent / "measurement_contract.yaml"),
        "run_id": receipt["run_id"], "command_sha256": receipt["command_sha256"],
    }
    entry_path.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(["/usr/bin/openssl", "pkeyutl", "-sign", "-inkey", str(private),
                    "-rawin", "-in", str(entry_path), "-out", str(signature)], check=True)
    private.unlink()
    controller._ISSUED_RECEIPTS.pop(receipt_path.resolve())

    with pytest.raises(ValueError, match="externally notarized fingerprint"):
        verify_receipt(
            receipt_path, expected_result=result, expected_study_sha256="a" * 64,
            trusted_issuance_fingerprint=original_fingerprint)


def test_controller_rejects_handwritten_receipt_without_replayable_roots(tmp_path):
    receipt = tmp_path / "receipt.yaml"
    receipt.write_text("schema_version: 4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="closed|identity"):
        verify_receipt(receipt, expected_result={}, expected_study_sha256="a" * 64)


def test_digest_bound_elf_that_reads_one_byte_loops_and_prints_pass_json_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not evidence JSON|observation"):
        _fixture(tmp_path, malicious_json=True)


def test_two_core_contract_rejects_one_busy_thread_allowed_on_both_cores(tmp_path):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("test host exposes fewer than two CPUs")
    with pytest.raises(ValueError, match="simultaneous work on every requested core"):
        _fixture(tmp_path, core_count=2)


def test_two_core_contract_rejects_a_sleeping_second_worker(tmp_path):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("test host exposes fewer than two CPUs")
    with pytest.raises(ValueError, match="simultaneous work on every requested core"):
        _fixture(tmp_path, core_count=2, sleeping_worker=True)


def test_quality_is_gated_per_frame_not_by_flattened_trajectory_cosine():
    import struct

    reference_frames = [struct.pack("<ff", 1.0, 1.0) for _ in range(1000)]
    observed_frames = list(reference_frames)
    observed_frames[517] = struct.pack("<ff", 1.0, -1.0)
    reference = _framed(*reference_frames)
    observed = _framed(*observed_frames)
    contract = {"oracle": {"kind": "float32_cosine", "metric": "logit_cosine",
                            "threshold": 0.99, "scope": "trajectory", "steps": 1000}}

    correctness, quality = controller._oracle(observed, reference, contract)

    assert quality["gate_ok"] is False
    assert correctness["gate_ok"] is False
    assert quality["per_step"][517] == {"index": 517, "value": 0.0, "gate_ok": False}
    assert sum(row["gate_ok"] for row in quality["per_step"]) == 999
