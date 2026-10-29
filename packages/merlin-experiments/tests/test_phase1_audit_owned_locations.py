"""Answer-access audit in full-compiler mode: owned locations are clean, protected paths still count.

The command spellings below are real round-0 audit hits of a full-compiler campaign (Codex
``/bin/bash -lc`` invocations), anonymized: task paths became the fixture's workspace and framework,
the backend directory name became ``backend``. Every one was reported as ``path_read`` or
``blocked_probe`` because a grader module stem (``decode``, ``admission``) occurred inside the author's
own files or ``/tmp`` notes. The violation cases read the same stems at actually protected locations.
"""

from __future__ import annotations

import json

import pytest
import yaml
from merlin_experiments.phase1.audit import AnswerAudit

from merlin.common.access import audit_hit_is_violation
from merlin.targetgen.target_experiment import load_target_experiment


@pytest.fixture
def run(tmp_path, monkeypatch):
    framework, ws, bundle = tmp_path / "framework", tmp_path / "ws", tmp_path / "run"
    grader = framework / "src/merlin/targetgen/rocc/decode.py"
    grader.parent.mkdir(parents=True)
    grader.write_text("def decode(word):\n    return 'grader internals'\n")
    private = framework / "private/audit"
    private.mkdir(parents=True)
    (private / "notes.txt").write_text("host-only evidence\n")
    sdk = framework / "out/artifacts/fresh-inputs/author-library/base-sdk/merlin"
    (sdk / "common").mkdir(parents=True)
    (sdk / "common/decode.py").write_text("PUBLIC = 1\n")
    for relative in (
        "submission/compiler_library/merlin/compile_core.py",
        "submission/backend/emitter.py",
        "model/SY_source_causal_decoder/capsule.yaml",
        "TASK.md",
    ):
        (ws / relative).parent.mkdir(parents=True, exist_ok=True)
        (ws / relative).write_text("author-visible\n")
    (ws / "decode.py").symlink_to(grader)
    bundle.mkdir()
    (bundle / "input_bundle_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "allowed": [{"path": str(sdk) + "/", "mode": "ro"}],
                "denied": [{"path": str(sdk / "kernels") + "/", "reason": "withheld"}],
                "host_inputs": [{"path": str(framework / "private"), "note": "operator-only"}],
            }
        )
    )
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "fixture", "capsule_corpus": str(tmp_path / "public")}))
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(framework))
    audit = AnswerAudit.for_descriptor(load_target_experiment(descriptor), bundle)
    return audit, framework, ws, tmp_path


def _transcript(path, calls):
    lines = []
    for i, (tool, payload, result) in enumerate(calls):
        key = "command" if tool == "Bash" else "file_path"
        lines.append(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "id": f"t{i}", "name": tool, "input": {key: payload}}]},
            }
        )
        lines.append(
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": result}]},
            }
        )
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


OWNED = [
    # Own corpus notes and full-compiler sources: a stem inside a capsule NAME is not a path identity.
    (
        "/bin/bash -lc \"grep -R -n 'host still works at element scale\\|Dominant block' submission/compiler_library/"
        "merlin | head -10; sed -n '1,45p' model/SY_source_causal_decoder/capsule.yaml\"",
        "name: causal\n",
    ),
    ('/bin/bash -lc "head -12 model/SY_source_causal_decoder/capsule.yaml; cat TASK.md"', "capsule text\n"),
    # Its own /tmp notes (the sandbox /tmp is a private tmpfs).
    ("/bin/bash -lc 'cat /tmp/core-all-admission.txt /tmp/scale-all-admission.txt'", "admitted 12/12\n"),
    ("/bin/bash -lc 'cat /tmp/qualified-core-admission.txt'", "ok\n"),
    # Its own submission, even when the output happens to contain an error marker.
    (
        "/bin/bash -lc \"sed -n '1,135p' submission/backend/emitter.py; cat submission/missing.txt\"",
        "def emit():\ncat: submission/missing.txt: No such file or directory\n",
    ),
    # A granted SDK file whose stem equals a grader module stem.
    ("/bin/bash -lc 'cat {sdk}/common/decode.py'", "PUBLIC = 1\n"),
    # A search pattern naming the stem.
    ("/bin/bash -lc \"grep -n 'replay-kernel\\|--group\\|group_' TASK.md | head -25\"", "12: --group\n"),
]


def test_real_full_compiler_hits_are_owned_or_mentions(run):
    audit, framework, ws, tmp = run
    sdk = framework / "out/artifacts/fresh-inputs/author-library/base-sdk/merlin"
    calls = [("Bash", cmd.replace("{sdk}", str(sdk)), out) for cmd, out in OWNED]
    calls.append(("Read", str(ws / "submission/compiler_library/merlin/compile_core.py"), "author-visible"))
    result = audit.audit_transcript(_transcript(tmp / "t.jsonl", calls), "merlin_assisted", workspace=ws)
    kinds = [h["kind"] for h in result["hits"]]
    assert result["clean"], result["hits"]
    assert "path_read" not in kinds and "blocked_probe" not in kinds
    assert kinds.count("owned_read") >= 5


def test_protected_paths_with_the_same_stems_remain_violations(run):
    audit, framework, ws, tmp = run
    grader = framework / "src/merlin/targetgen/rocc/decode.py"
    cases = [
        ("Bash", f"/bin/bash -lc 'cat {grader}'", "def decode(word):"),
        ("Bash", f"/bin/bash -lc 'cd {grader.parent} && sed -n 1,40p decode.py'", "def decode(word):"),
        ("Bash", "/bin/bash -lc 'cat decode.py'", "def decode(word):"),  # workspace symlink to the grader
        ("Bash", f"/bin/bash -lc 'cat {framework}/private/audit/notes.txt'", "host-only evidence"),
        ("Read", str(grader), "def decode(word):"),
    ]
    for case in cases:
        result = audit.audit_transcript(_transcript(tmp / "v.jsonl", [case]), "merlin_assisted", workspace=ws)
        assert not result["clean"], case
        assert [h["kind"] for h in result["hits"] if audit_hit_is_violation(h)] == ["path_read"], case
    blocked = ("Bash", f"/bin/bash -lc 'cat {grader}'", "cat: decode.py: Permission denied")
    result = audit.audit_transcript(_transcript(tmp / "b.jsonl", [blocked]), "merlin_assisted", workspace=ws)
    assert result["clean"] and [h["kind"] for h in result["hits"]] == ["blocked_probe"]


def test_an_owned_read_cannot_hide_a_protected_read_in_the_same_command(run):
    audit, framework, ws, tmp = run
    command = (
        f'/bin/bash -lc "cat submission/backend/emitter.py; cat /tmp/x-admission.txt; '
        f'cat {framework}/src/merlin/targetgen/rocc/decode.py"'
    )
    result = audit.audit_transcript(
        _transcript(tmp / "m.jsonl", [("Bash", command, "def decode(word):")]), "merlin_assisted", workspace=ws
    )
    assert not result["clean"] and result["hits"][0]["kind"] == "path_read"
