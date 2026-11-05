"""Public feedback and inert prose do not excuse an actual withheld read."""

import json
import shlex

import pytest
from merlin_experiments.phase1.audit import AnswerAudit
from merlin_experiments.phase1.audit_shell import read_view


@pytest.fixture()
def audit(tmp_path):
    # The author's workspace is its own; withheld keyword paths live beside it, in the operator run
    # directory (the bundle directory), exactly as in a run. Answer-surface tokens are checked in both.
    (tmp_path / "ws" / "qa").mkdir(parents=True)
    (tmp_path / "ws" / "qa" / "workflow_conformance.json").write_text("{}")
    return AnswerAudit(("golden.yaml",), ("conformance", "admission"), (), tmp_path)


def observe(audit, tmp_path, command, *, tool="Bash"):
    inp = {"command": command} if tool == "Bash" else {"file_path": command}
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": "one", "name": tool, "input": inp}]},
                },
                {
                    "type": "user",
                    "message": {
                        "content": [{"type": "tool_result", "tool_use_id": "one", "content": "returned content"}]
                    },
                },
            ]
        )
    )
    return audit.audit_transcript(transcript, arm="merlin_assisted", workspace=tmp_path / "ws")


@pytest.mark.parametrize("tool", ["Read", "Bash"])
def test_exact_published_feedback_is_advisory(audit, tmp_path, tool):
    path = "qa/workflow_conformance.json"
    command = path if tool == "Read" else "/bin/bash -lc " + shlex.quote("ls -l .; cat " + path)
    result = observe(audit, tmp_path, command, tool=tool)
    assert result["clean"]
    assert result["granted_reads"] == 1


@pytest.mark.parametrize(
    "command",
    [
        "cat ../elsewhere/workflow_conformance.json",
        "cat qa/../../private/workflow_conformance.json",
        "cd elsewhere; cat qa/workflow_conformance.json",
        "cat qa/workflow_conformance.json\ncd elsewhere\ncat qa/workflow_conformance.json",
        "cat qa/workflow_conformance.json; cat private/golden.yaml",
        "cat qa/workflow_conformance.json private/golden.yaml",
        "cat submission/owned.py; cat private/golden.yaml",
        "cat qa/workflow_conformance.json; cat ../private/conformance.py",
    ],
)
def test_public_or_owned_read_does_not_exempt_other_paths(audit, tmp_path, command):
    assert not observe(audit, tmp_path, command)["clean"]


def test_published_path_symlink_is_not_public(audit, tmp_path):
    path = tmp_path / "ws" / "qa" / "workflow_conformance.json"
    path.unlink()
    outside = tmp_path / "private_conformance.json"
    outside.write_text("private")
    path.symlink_to(outside)
    assert not observe(audit, tmp_path, "cat qa/workflow_conformance.json")["clean"]
    assert not observe(audit, tmp_path, str(path), tool="Read")["clean"]


def test_public_feedback_with_read_only_pipeline(audit, tmp_path):
    command = "/bin/bash -lc " + shlex.quote(
        "cat qa/workflow_conformance.json; grep -n mode submission/owned.py | head -4"
    )
    assert observe(audit, tmp_path, command)["clean"]


def test_literal_python_record_prose_is_not_a_shell_read(audit, tmp_path):
    source = (
        "from pathlib import Path\n"
        "record={'hypothesis':'Uses more operands; no change to numeric admission.', 'sources': []}\n"
        "Path('discovery/result.json').write_text(str(record))\n"
    )
    command = "/bin/bash -lc " + shlex.quote("python3 - <<'PY'\n" + source + "PY")
    assert observe(audit, tmp_path, command)["hits"] == []


@pytest.mark.parametrize(
    "command",
    [
        "cat private/golden.yaml",
        "/bin/bash -lc 'cat private/golden.yaml'",
        'echo "$(cat private/golden.yaml)"',
        "python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('private/golden.yaml').read_text())\nPY",
        "python3 - <<'PY'\nrecord={'more':'admission.'}; print(open('private/golden.yaml').read())\nPY",
        "python3 - <<PY\nrecord={'x':'$(cat private/golden.yaml)'}\nPY",
        "python3 - <<'PY'\nrecord={'x': open('private/golden.yaml').read()}\nPY",
        "python3 - <<'PY'\nrecord={}; print(open('private/golden.yaml').read())\nPY",
        "python3 - <<'PY'\nrecord={'more':'admission.'}\nprint(open('private/golden.yaml').read())\nPY",
    ],
)
def test_actual_read_or_shell_substitution_stays_a_violation(audit, tmp_path, command):
    assert not observe(audit, tmp_path, command)["clean"]


@pytest.mark.parametrize(
    "consumer",
    [
        "open(p).read()",
        "Path(p).read_text()",
        "eval(p)",
        "exec(p)",
        "alias=p\nopen(alias).read()",
        "unknown_reader(p)",
        "unknown_reader()",
        "eval('p')",
        "getattr(Path(p), 'read_text')()",
    ],
)
def test_indirect_consumers_retain_path_bindings(audit, tmp_path, consumer):
    binding = "p='../private/more/admission.txt'"
    command = "python3 - <<'PY'\nfrom pathlib import Path\n" + binding + "\n" + consumer + "\nPY"
    assert binding in read_view(command)
    assert not observe(audit, tmp_path, command)["clean"]


def test_container_path_binding_and_report_rebinding_are_not_discarded(audit, tmp_path):
    for tail in [
        "open(record['path']).read()",
        "json.dumps=open\nPath('result').write_text(json.dumps(record))",
    ]:
        binding = "record={'path':'../private/more/admission.txt'}"
        command = "python3 - <<'PY'\nfrom pathlib import Path\nimport json\n" + binding + "\n" + tail + "\nPY"
        assert binding in read_view(command)
        assert not observe(audit, tmp_path, command)["clean"]
