"""Staging preserves exact bytes/versions and never dispatches cloud execution."""

import hashlib
from pathlib import Path

import pytest
from merlin_experiments.post_freeze import elf_upload as U
from merlin_experiments.post_freeze import firesim as F
from test_post_freeze_firesim import case as case
from test_post_freeze_firesim import direct_observation, save


@pytest.fixture
def frozen_request(case):
    path, *_ = direct_observation(case)
    body = '{"schemaVersion":"2.2","description":"pinned deployment"}\n'
    selection = Path(F.load(path)["selection"]["path"])
    spec = F.load(selection)
    spec["deployment"].update(ssm_document_version="7", ssm_document_sha256=hashlib.sha256(body.encode()).hexdigest(),
                              manager_entrypoint_sha256="a" * 64)
    manager_path = Path(spec["deployment"]["manager_contract"]["path"])
    helpers = {}
    for name in ("elf_custody", "chain_observation", "manager_result"):
        raw = ("# " + name + " source\n").encode()
        helpers[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    save(manager_path, {"schema": "merlin.post_freeze.firesim.manager_contract.v1",
                        "boot_argument_prefix": "+prog0=", "helpers": helpers})
    spec["deployment"]["manager_contract"] = F.pin(manager_path)
    save(selection, spec)
    save(path, F.prepare(selection))
    return path, body


class FakeAWS:
    def __init__(self, document):
        self.document = {"Name": "FsimRunElf", "DocumentVersion": "7", "Status": "Active", "Content": document,
                         "UnfilteredField": "retained"}
        self.calls = []
        self.objects = {}
        self.fault = lambda kind, key, response, destination: None
        self.after_document = lambda: None

    def __call__(self, args):
        self.calls.append(args)
        kind = args[:2]
        if kind == ["ssm", "get-document"]:
            assert args[args.index("--document-version") + 1] == "7"
            self.after_document()
            return dict(self.document)
        assert kind[0] == "s3api" and kind[1] in {"put-object", "get-object"}, "execution is forbidden"
        key = args[args.index("--key") + 1]
        if kind[1] == "put-object":
            assert args[args.index("--if-none-match") + 1] == "*"
            if key in self.objects:
                raise ValueError("PreconditionFailed: object already exists")
            raw = Path(args[args.index("--body") + 1]).read_bytes()
            version = "version-" + str(len(self.objects) + 1)
            self.objects[key] = (version, raw)
            response = {"VersionId": version, "ETag": "not-a-byte-proof", "UnfilteredField": [1, 2]}
            destination = None
        else:
            version, raw = self.objects[key]
            assert args[args.index("--version-id") + 1] == version
            destination = Path(args[-1])
            destination.write_bytes(raw)
            response = {"VersionId": version, "ContentLength": len(raw), "UnfilteredField": [3, 4]}
        self.fault(kind[1], key, response, destination)
        return response


def stage(frozen_request, tmp_path, *, fake=None, output="stage", identifier="unique-submission"):
    path, document = frozen_request
    client = FakeAWS(document) if fake is None else fake
    return U.stage(path, model="example", output=tmp_path / output, submission_id=identifier, aws=client), client


def test_staging_reopens_versioned_bytes_and_preserves_raw_request(frozen_request, tmp_path):
    path, _ = frozen_request
    original = path.read_bytes()
    result, fake = stage(frozen_request, tmp_path)
    elf = F.load(F.reopen(result["elf_upload"]))
    uploaded_request = F.load(F.reopen(result["request_upload"]))
    assert elf["schema"] == F.UPLOAD_SCHEMA
    assert F.reopen(uploaded_request["downloaded_request"]).read_bytes() == original
    assert fake.objects[uploaded_request["key"]][1] == original
    assert len(fake.objects) == 2
    assert elf["key"] == "firesim-elfs/unique-submission.elf"
    assert uploaded_request["key"] == "firesim-requests/unique-submission.json"
    assert elf["version_id"] == "version-1" and uploaded_request["version_id"] == "version-2"
    assert F.load(F.reopen(elf["s3_get_object"]))["UnfilteredField"] == [3, 4]
    assert F.load(F.reopen(elf["s3_put_object"]))["UnfilteredField"] == [1, 2]
    assert F.load(F.reopen(result["ssm_document"]))["UnfilteredField"] == "retained"
    command = F.load(F.reopen(result["send_command"]))
    assert command["DocumentVersion"] == "7"
    assert "CommandId" not in command and "Command" not in command
    assert command["DocumentHashType"] == "Sha256"
    assert command["DocumentHash"] == hashlib.sha256(frozen_request[1].encode()).hexdigest()
    assert command["Parameters"] == {
        "elfKey": [elf["key"]], "elfVersion": ["version-1"],
        "requestKey": [uploaded_request["key"]], "requestVersion": ["version-2"],
        "requestSha256": [hashlib.sha256(original).hexdigest()], "model": ["example"],
        "elfSha256": [elf["downloaded_elf"]["sha256"]], "elfBytes": [str(elf["downloaded_elf"]["bytes"])],
        "entrypointSha256": ["a" * 64], "jobWallSeconds": ["120"], "submissionId": ["unique-submission"]}
    assert result["planned_command_evidence"] == {"bucket": "example", "key": "firesim-commands/unique-submission.json"}
    assert result["paid_jobs_launched"] == result["ssm_commands_sent"] == 0
    assert result["manager_started"] is result["scientific_qualification"] is False
    assert all(call[:2] != ["ssm", "send-command"] for call in fake.calls)


@pytest.mark.parametrize("field,value", [
    ("ssm_document_version", None), ("ssm_document_version", "$LATEST"), ("ssm_document_version", "$DEFAULT"),
    ("ssm_document_version", True), ("ssm_document_version", "01"), ("ssm_document_version", 0),
    ("ssm_document_version", 7),
    ("ssm_document_sha256", "wrong"), ("manager_entrypoint_sha256", "wrong"),
])
def test_required_pinned_deployment_refuses_before_aws(frozen_request, tmp_path, field, value):
    path, document = frozen_request
    selection = Path(F.load(path)["selection"]["path"])
    spec = F.load(selection)
    spec["deployment"][field] = value
    save(selection, spec)
    fake = FakeAWS(document)
    with pytest.raises(ValueError):
        save(path, F.prepare(selection))
        stage(frozen_request, tmp_path, fake=fake)
    assert not fake.calls


@pytest.mark.parametrize("field,value", [("DocumentVersion", "8"), ("Name", "Other"),
                                         ("Status", "Failed"), ("Content", "changed")])
def test_wrong_document_never_uploads(frozen_request, tmp_path, field, value):
    fake = FakeAWS(frozen_request[1])
    fake.document[field] = value
    with pytest.raises(ValueError, match="document version/content"):
        stage(frozen_request, tmp_path, fake=fake)
    assert not fake.objects
    assert (tmp_path / "stage/ssm-document-response.json").is_file()


@pytest.mark.parametrize("field,value", [("VersionId", None), ("VersionId", "null"),
                                         ("VersionId", ""), ("VersionId", 7)])
def test_unversioned_upload_refuses(frozen_request, tmp_path, field, value):
    fake = FakeAWS(frozen_request[1])
    def mutate(kind, key, response, destination):
        if kind == "put-object":
            response[field] = value
    fake.fault = mutate
    with pytest.raises(ValueError, match="VersionId"):
        stage(frozen_request, tmp_path, fake=fake)
    assert len(fake.calls) == 2
    assert not (tmp_path / "stage/stage.json").exists()


@pytest.mark.parametrize("fault", ["version", "size", "checksum", "elf_bytes", "request_bytes"])
def test_actual_download_drift_refuses(frozen_request, tmp_path, fault):
    fake = FakeAWS(frozen_request[1])
    def mutate(kind, key, response, destination):
        if kind != "get-object":
            return
        if fault in {"elf_bytes", "request_bytes"}:
            selected = key.endswith(".elf") if fault == "elf_bytes" else key.endswith(".json")
            if selected:
                destination.write_bytes(b"x" * response["ContentLength"])
        elif key.endswith(".elf"):
            response[{"version": "VersionId", "size": "ContentLength", "checksum": "ChecksumSHA256"}[fault]] = "wrong"
    fake.fault = mutate
    with pytest.raises(ValueError, match="downloaded bytes"):
        stage(frozen_request, tmp_path, fake=fake)
    assert not (tmp_path / "stage/stage.json").exists()


@pytest.mark.parametrize("which", ["request", "elf", "compiler"])
def test_source_mutation_during_staging_refuses(frozen_request, tmp_path, which):
    fake = FakeAWS(frozen_request[1])
    record = F.load(frozen_request[0])
    boot = next(x for x in record["models"][0]["assets"] if x["role"] == "boot_elf")
    target = {"request": frozen_request[0], "elf": Path(boot["path"]),
              "compiler": Path(record["compiler"]["path"]) / "compiler.py"}[which]
    fake.after_document = lambda: target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        stage(frozen_request, tmp_path, fake=fake)
    assert not (tmp_path / "stage/stage.json").exists()


def test_no_overwrite_or_reused_submission(frozen_request, tmp_path):
    _, fake = stage(frozen_request, tmp_path)
    before = dict(fake.objects)
    with pytest.raises(ValueError, match="PreconditionFailed"):
        stage(frozen_request, tmp_path, fake=fake, output="other-stage")
    assert fake.objects == before
    assert (tmp_path / "other-stage/elf-put-failure.json").is_file()


@pytest.mark.parametrize("identifier", ["..", "unsafe/key", "newline\n", "x" * 65])
def test_submission_token_refuses_before_aws(frozen_request, tmp_path, identifier):
    fake = FakeAWS(frozen_request[1])
    with pytest.raises(ValueError, match="safe token"):
        stage(frozen_request, tmp_path, fake=fake, identifier=identifier)
    assert not fake.calls


def test_cli_requires_explicit_upload_opt_in(monkeypatch, frozen_request, tmp_path):
    monkeypatch.setattr(U, "stage", lambda *a, **k: pytest.fail("no default uploads"))
    with pytest.raises(SystemExit) as error:
        U.main([str(frozen_request[0]), "--model", "example", "--output", str(tmp_path / "unused")])
    assert error.value.code == 2


def test_generated_submission_id(frozen_request, tmp_path):
    result, _ = stage(frozen_request, tmp_path, identifier=None)
    assert F._token(result["submission_id"])
    assert F.load(F.reopen(result["send_command"]))["DocumentVersion"] == "7"


def test_changed_frozen_source_is_refused_before_any_aws(frozen_request, tmp_path):
    fake = FakeAWS(frozen_request[1])
    compiler = Path(F.load(frozen_request[0])["compiler"]["path"]) / "compiler.py"
    compiler.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        stage(frozen_request, tmp_path, fake=fake)
    assert not fake.calls


@pytest.mark.parametrize("location", ["existing", "compiler", "symlink", "parent_traversal"])
def test_output_must_be_fresh_and_disjoint(frozen_request, tmp_path, location):
    fake = FakeAWS(frozen_request[1])
    path = tmp_path / "stage"
    if location == "existing":
        path.mkdir()
    elif location == "compiler":
        path = Path(F.load(frozen_request[0])["compiler"]["path"]) / "new-output"
    elif location == "symlink":
        path.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        path = tmp_path / "uncreated" / ".." / "stage"
    with pytest.raises(ValueError, match="fresh ordinary"):
        U.stage(frozen_request[0], model="example", output=path, aws=fake)
    assert not fake.calls


def test_aws_cli_uses_bounded_structured_argv(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        body = '{"IfNoneMatch":""}' if "--generate-cli-skeleton" in args else '{"raw":"response"}'
        return SimpleNamespace(stdout=body)
    monkeypatch.setattr(U.subprocess, "run", run)
    assert U._aws_cli("selected-region")(["ssm", "get-document", "--document-version", "7"]) == {"raw": "response"}
    assert calls[0][0] == ["aws", "--region", "selected-region", "s3api", "put-object",
                           "--generate-cli-skeleton", "input"]
    assert calls[0][1]["env"]["AWS_EC2_METADATA_DISABLED"] == "true"
    assert calls[0][1]["timeout"] == 30
    assert calls[1][0] == ["aws", "--region", "selected-region", "ssm", "get-document",
                           "--document-version", "7", "--output", "json"]
    assert calls[1][1]["env"]["AWS_PAGER"] == ""
    assert {key: value for key, value in calls[1][1].items() if key != "env"} == {
        "check": True, "capture_output": True, "text": True, "timeout": 120}


def test_old_cli_refuses_without_any_aws_api_call(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='{"Bucket":"","Key":""}')
    monkeypatch.setattr(U.subprocess, "run", run)
    with pytest.raises(ValueError, match="upgrade before staging"):
        U._aws_cli("selected-region")
    assert len(calls) == 1 and "--generate-cli-skeleton" in calls[0]


def test_aws_failure_retains_exact_stdout_and_stderr(frozen_request, tmp_path):
    def fail(_args):
        raise U.subprocess.TimeoutExpired(["aws"], 120, output=b"raw stdout\xff", stderr=b"raw stderr\xfe")
    with pytest.raises(U.subprocess.TimeoutExpired):
        stage(frozen_request, tmp_path, fake=fail)
    failure = F.load(tmp_path / "stage/ssm-document-failure.json")
    assert F.reopen(failure["stdout"]).read_bytes() == b"raw stdout\xff"
    assert F.reopen(failure["stderr"]).read_bytes() == b"raw stderr\xfe"


def test_download_symlink_is_refused_without_changing_source_permissions(frozen_request, tmp_path):
    fake = FakeAWS(frozen_request[1])
    boot = next(x for x in F.load(frozen_request[0])["models"][0]["assets"] if x["role"] == "boot_elf")
    source = Path(boot["path"])
    mode = source.stat().st_mode
    def redirect(kind, key, response, destination):
        if kind == "get-object":
            destination.unlink()
            destination.symlink_to(source)
    fake.fault = redirect
    with pytest.raises(ValueError, match="regular artifact"):
        stage(frozen_request, tmp_path, fake=fake)
    assert source.stat().st_mode == mode


def test_blocked_request_never_stages(frozen_request, tmp_path):
    path, body = frozen_request
    selection = Path(F.load(path)["selection"]["path"])
    spec = F.load(selection)
    spec["deployment"]["manager_contract"] = None
    save(selection, spec)
    save(path, F.prepare(selection))
    assert F.load(path)["status"] == "blocked"
    fake = FakeAWS(body)
    with pytest.raises(ValueError, match="fully prepared"):
        stage(frozen_request, tmp_path, fake=fake)
    assert not fake.calls
