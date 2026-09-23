"""Static cache ownership preserves exact artifacts without qualifying or executing them."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase2 import contracts, static_cache


@pytest.fixture(autouse=True)
def refuse_processes(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("static cache operations must not launch a process")

    monkeypatch.setattr(subprocess, "Popen", refused)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    capsules = [digest("primary"), digest("training")]
    candidate = digest("compiler")
    analyses, artifacts = [], {}
    for index, capsule in enumerate(capsules):
        lowered = f"synthetic lowered {index}"
        command = json.dumps({"commands": [index]})
        emission = {
            "candidate_lowered_sha256": digest(lowered),
            "candidate_command_buffer_sha256": digest(command),
        }
        analyses.append(
            {
                "candidate_sha256": candidate,
                "workload": {"capsule_sha256": capsule},
                "emission": emission,
                "probe_receipts": [{"synthetic": "never import"}],
                "diagnostics": {
                    "elapsed_seconds": 5,
                    "static_fact": index,
                    "emission_execution": {"command": "never replay"},
                },
            }
        )
        artifacts[capsule] = {
            "candidate_sha256": candidate,
            **emission,
            "lowered_text": lowered,
            "command_buffer_text": command,
            "command_buffer": json.loads(command),
            "decoded_trace": [],
            "task_instruction_evidence": {},
            "interface": "not a portable artifact",
            "parsed_lowered_module": object(),
            "baseline_artifacts": {
                "lowered_text": "baseline",
                "lowered_sha256": digest("baseline"),
                "command_buffer_text": "{}",
                "command_buffer_sha256": digest("{}"),
                "probe_receipts": [{"unverified": True}],
                "elapsed_seconds": 3,
            },
        }
    binding = {"candidate_sha256": candidate, "policy_sha256": digest("policy")}
    record = {
        "iteration": 0,
        "candidate_sha256": candidate,
        "analysis": analyses[0],
        "portfolio": {"members": [{"analysis": item} for item in analyses]},
        "cross_run_static_analysis_binding": binding,
    }
    return output, capsules, record, artifacts


def publish(evidence):
    output, capsules, record, artifacts = evidence
    return static_cache.persist_static_analysis_bundle(
        record,
        artifacts[capsules[0]],
        output=output,
        capsule_sha256s=capsules,
        portfolio_sha256=digest("portfolio"),
        portfolio_artifacts=artifacts,
    )


def load(evidence, reference):
    output, capsules, record, _ = evidence
    return static_cache.load_static_analysis_bundle(
        checkpoint_parent=output,
        checkpoint_bundle_ref=reference,
        iteration_bundle_ref=reference,
        binding=record["cross_run_static_analysis_binding"],
        candidate_sha256=record["candidate_sha256"],
        portfolio_sha256=digest("portfolio"),
        capsule_sha256s=capsules,
    )


def replace_bundle(reference, document):
    path = Path(reference["path"])
    path.chmod(0o644)
    path.write_text(json.dumps(document))
    path.chmod(0o444)
    reference["sha256"] = contracts.sha256_file(path)


def test_two_member_publication_roundtrip_strips_dynamic_and_local_evidence(evidence):
    output, capsules, record, artifacts = evidence
    reference = publish(evidence)
    path = Path(reference["path"])
    original = path.read_bytes()
    assert path.parent == output
    assert path.stat().st_mode & 0o222 == 0
    assert reference["sha256"] == contracts.sha256_file(path)
    members = list(load(evidence, reference))
    assert len(members) == 2
    for index, member in enumerate(members):
        assert member.analysis["workload"]["capsule_sha256"] == capsules[index]
        assert "probe_receipts" not in member.analysis
        assert member.analysis["diagnostics"] == {"static_fact": index}
        decoded = member.decode_artifacts(analysis=member.analysis)
        assert decoded["command_buffer"] == artifacts[capsules[index]]["command_buffer"]
        assert "interface" not in decoded
        assert "parsed_lowered_module" not in decoded
        assert "elapsed_seconds" not in decoded["baseline_artifacts"]
        assert "probe_receipts" not in decoded["baseline_artifacts"]
    members[0].analysis["workload"]["capsule_sha256"] = "changed detached copy"
    assert record["analysis"]["workload"]["capsule_sha256"] == capsules[0]
    assert path.read_bytes() == original
    assert list(load(evidence, reference))[0].analysis["workload"]["capsule_sha256"] == capsules[0]


@pytest.mark.parametrize("primary", [{}, {"decoded_trace": []}])
def test_incomplete_primary_does_not_publish_bundle(evidence, primary):
    output, capsules, record, _ = evidence
    assert (
        static_cache.persist_static_analysis_bundle(
            record,
            primary,
            output=output,
            capsule_sha256s=capsules,
            portfolio_sha256=digest("portfolio"),
        )
        is None
    )
    assert list(output.iterdir()) == []


def test_missing_training_artifacts_persist_analysis_but_cannot_import(evidence):
    _, capsules, _, artifacts = evidence
    artifacts.pop(capsules[1])
    reference = publish(evidence)
    document = json.loads(Path(reference["path"]).read_bytes())
    assert len(document["member_analyses"]) == 2
    assert document["portfolio_member_artifacts"][1]["artifacts"] is None
    assert document["portfolio_member_artifacts"][1]["status"] == "member_published_no_reusable_artifacts"
    members = list(load(evidence, reference))
    members[0].decode_artifacts(analysis=members[0].analysis)
    with pytest.raises(ValueError, match="no reusable"):
        members[1].decode_artifacts(analysis=members[1].analysis)


def test_explicitly_empty_primary_member_is_not_treated_as_missing_training(evidence):
    output, capsules, record, artifacts = evidence
    primary = artifacts[capsules[0]]
    artifacts[capsules[0]] = {}
    with pytest.raises(ValueError, match="objective"):
        static_cache.persist_static_analysis_bundle(
            record,
            primary,
            output=output,
            capsule_sha256s=capsules,
            portfolio_sha256=digest("portfolio"),
            portfolio_artifacts=artifacts,
        )
    assert list(output.iterdir()) == []


@pytest.mark.parametrize(
    "mutation", ["order", "binding", "candidate", "portfolio", "coverage", "analysis_type", "member_type", "workload"]
)
def test_bundle_identity_and_coverage_changes_are_refused(evidence, mutation):
    reference = publish(evidence)
    document = json.loads(Path(reference["path"]).read_bytes())
    if mutation == "order":
        document["portfolio_member_artifacts"].reverse()
    elif mutation in ("binding", "candidate", "portfolio"):
        field = {"binding": "binding", "candidate": "candidate_sha256", "portfolio": "portfolio_sha256"}[mutation]
        document[field] = "changed"
    elif mutation == "coverage":
        document["member_analyses"].pop()
    elif mutation == "analysis_type":
        document["member_analyses"][1] = []
    elif mutation == "member_type":
        document["portfolio_member_artifacts"][1] = []
    elif mutation == "workload":
        document["member_analyses"][1]["workload"]["capsule_sha256"] = digest("changed")
    replace_bundle(reference, document)
    with pytest.raises(ValueError):
        list(load(evidence, reference))


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "lowered", "hash", "json", "malformed_json", "candidate", "baseline"]
)
def test_member_artifact_mutations_are_refused_at_decode(evidence, mutation):
    reference = publish(evidence)
    document = json.loads(Path(reference["path"]).read_bytes())
    artifact = document["portfolio_member_artifacts"][1]["artifacts"]
    if mutation == "missing":
        artifact.pop("decoded_trace")
    elif mutation == "extra":
        artifact["unexpected"] = True
    elif mutation == "lowered":
        artifact["lowered_text"] += " changed"
    elif mutation == "hash":
        artifact["candidate_lowered_sha256"] = digest("changed")
    elif mutation == "json":
        artifact["command_buffer"] = {"unbound": True}
    elif mutation == "malformed_json":
        artifact["command_buffer_text"] = "{"
        artifact["candidate_command_buffer_sha256"] = digest("{")
        document["member_analyses"][1]["emission"]["candidate_command_buffer_sha256"] = digest("{")
    elif mutation == "candidate":
        artifact["candidate_sha256"] = digest("changed")
    elif mutation == "baseline":
        artifact["baseline_artifacts"]["lowered_text"] += " changed"
    replace_bundle(reference, document)
    members = list(load(evidence, reference))
    members[0].decode_artifacts(analysis=members[0].analysis)
    with pytest.raises(ValueError):
        members[1].decode_artifacts(analysis=members[1].analysis)


def test_later_member_identity_is_not_validated_before_primary_consumption(evidence):
    reference = publish(evidence)
    document = json.loads(Path(reference["path"]).read_bytes())
    document["portfolio_member_artifacts"][1]["capsule_sha256"] = digest("changed")
    replace_bundle(reference, document)
    members = iter(load(evidence, reference))
    first = next(members)
    first.decode_artifacts(analysis=first.analysis)
    with pytest.raises(ValueError):
        next(members)


@pytest.mark.parametrize("mutation", ["writable", "linked", "escaped", "bytes", "relative"])
def test_bundle_storage_changes_are_refused(evidence, tmp_path, mutation):
    reference = publish(evidence)
    path = Path(reference["path"])
    if mutation == "writable":
        path.chmod(0o644)
    elif mutation == "linked":
        original = path.with_suffix(".original")
        path.rename(original)
        path.symlink_to(original)
    elif mutation == "escaped":
        outside = tmp_path / "outside.json"
        path.rename(outside)
        reference["path"] = str(outside)
    elif mutation == "relative":
        reference["path"] = path.name
    elif mutation == "bytes":
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)
    with pytest.raises(ValueError):
        load(evidence, reference)


def test_publication_never_overwrites_existing_bundle(evidence):
    output, _, _, _ = evidence
    reference = publish(evidence)
    path = Path(reference["path"])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        publish(evidence)
    assert path.read_bytes() == original
    assert list(output.iterdir()) == [path]


def test_checkpoint_and_iteration_must_pin_same_bundle(evidence):
    output, capsules, record, _ = evidence
    reference = publish(evidence)
    changed = {**reference, "sha256": digest("changed")}
    with pytest.raises(ValueError):
        static_cache.load_static_analysis_bundle(
            checkpoint_parent=output,
            checkpoint_bundle_ref=reference,
            iteration_bundle_ref=changed,
            binding=record["cross_run_static_analysis_binding"],
            candidate_sha256=record["candidate_sha256"],
            portfolio_sha256=digest("portfolio"),
            capsule_sha256s=capsules,
        )


def test_pinned_reader_hashes_and_parses_one_buffer(evidence, monkeypatch):
    reference = publish(evidence)
    path = Path(reference["path"])
    original = path.read_bytes()
    replacement = json.loads(original)
    replacement["candidate_sha256"] = digest("substituted after read")
    real_read = Path.read_bytes
    reads = []

    def swap_after_read(selected):
        payload = real_read(selected)
        if selected == path:
            reads.append(selected)
            path.chmod(0o644)
            path.write_text(json.dumps(replacement))
            path.chmod(0o444)
        return payload

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    result = static_cache.load_pinned_read_only_mapping(path, reference["sha256"], label="synthetic bundle")
    assert result == json.loads(original)
    assert reads == [path]


@pytest.mark.parametrize("name", ["../outside.json", "nested/bundle.json", "..", ".", "", "/absolute.json"])
def test_atomic_publication_refuses_unsafe_names_without_writing(tmp_path, name):
    output = tmp_path / "cache"
    output.mkdir()
    with pytest.raises((ValueError, contracts.StageGateError)):
        static_cache.atomic_static_write(name, {"synthetic": True}, output=output)
    assert list(tmp_path.rglob("*")) == [output]


@pytest.mark.parametrize("serialization_error", [False, True])
def test_temporary_name_collision_preserves_preexisting_file(tmp_path, monkeypatch, serialization_error):
    monkeypatch.setattr(static_cache.os, "getpid", lambda: 123)
    monkeypatch.setattr(static_cache.time, "time_ns", lambda: 456)
    temporary = tmp_path / ".bundle.json.123.456.tmp"
    original = b"preexisting publisher bytes"
    temporary.write_bytes(original)
    record = {"synthetic": object() if serialization_error else True}
    expected = TypeError if serialization_error else FileExistsError
    with pytest.raises(expected):
        static_cache.atomic_static_write("bundle.json", record, output=tmp_path)
    assert temporary.read_bytes() == original
    assert list(tmp_path.iterdir()) == [temporary]


@pytest.mark.parametrize("failure", ["serialization", "write", "link"])
def test_failed_publication_cleans_only_its_own_temporary_file(tmp_path, monkeypatch, failure):
    record = {"synthetic": True}
    if failure == "serialization":
        record["synthetic"] = object()
    elif failure == "link":

        def refuse_link(*args, **kwargs):
            raise OSError("synthetic link failure")

        monkeypatch.setattr(static_cache.os, "link", refuse_link)
    else:
        real_open = Path.open

        class PartialWrite:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def write(self, payload):
                self.stream.write(payload[:1])
                self.stream.flush()
                raise OSError("synthetic partial write failure")

        def partial_open(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            if mode == "xb" and path.parent == tmp_path:
                return PartialWrite(stream)
            return stream

        monkeypatch.setattr(Path, "open", partial_open)
    with pytest.raises(TypeError if failure == "serialization" else OSError):
        static_cache.atomic_static_write("bundle.json", record, output=tmp_path)
    assert list(tmp_path.iterdir()) == []
