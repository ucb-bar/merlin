"""Fixed-payload offline CLI/bwrap double; not an OS isolation implementation."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def emit(document):
    print(json.dumps(document), flush=True)


root = Path(os.environ["PHASE1_FIXTURE_ROOT"]).resolve(strict=True)
if Path(sys.argv[0]).name == "bwrap":
    args = sys.argv[1:]
    if args[:1] == ["--args"]:
        descriptor = int(args[1])
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            args = stream.read().decode().rstrip("\x00").split("\x00") + args[2:]
    # Never execute shell text: inspect the real policy argv and only dispatch one
    # fixed local payload. All unknown shapes refuse before any process execution.
    assert args[-3:-1] == ["bash", "-c"], args[-4:]
    script = args[-1]
    assert "--unshare-pid" in args and "--die-with-parent" in args
    workspace = Path(args[args.index("--chdir") + 1]).resolve(strict=True)
    assert workspace.is_relative_to(root), workspace
    from merlin.targetgen.sandbox.toolchain import sandbox_env
    from merlin.targetgen.target_experiment import load_target_experiment

    prefix = sandbox_env(load_target_experiment(root / "target_experiment.yaml"), workspace) + " "
    assert script.startswith(prefix)
    payload = script[len(prefix) :]
    # Admit only the complete current visibility-probe grammar, never execute it.
    # The random control path must name the host-created workspace sentinel.
    repo = Path(os.environ["MERLIN_REPO_ROOT"])
    corpus = root / "corpus/isa"
    from merlin.targetgen.sandbox.bwrap import bundle_snapshot_root

    snapshot = bundle_snapshot_root(workspace)
    frozen = (
        snapshot / "repo" / corpus.relative_to(repo)
        if corpus.is_relative_to(repo)
        else (snapshot / "external" / Path(*corpus.parts[1:]))
    )
    roots = {corpus, repo / "merlin/contract/capsules", frozen}
    roots.update(load_target_experiment(root / "target_experiment.yaml").graded_roots())
    control_words = shlex.split(payload.partition("; ")[0])
    is_probe = (
        len(control_words) == 6 and control_words[:2] == ["test", "-s"] and control_words[3:] == ["||", "exit", "1"]
    )
    probe = None
    if is_probe:
        control = Path(control_words[2])
        assert control.parent == workspace and control.name.startswith(".mask-control-")
        assert control.read_bytes() == b"probe control\n"
        patterns = [
            "golden.*",
            "*.golden.*",
            "expected_command_buffer*",
            "expected_instruction_coverage.yaml",
            "*.safetensors",
            "*.safetensors.manifest.json",
        ]
        expression = " -o ".join("-name " + shlex.quote(pattern) for pattern in patterns)
        emit_files = shlex.quote('for f do if test -s "$f"; then printf "LEAK:%s\\n" "$f"; fi; done')
        commands = []
        for directory in sorted(roots):
            quoted = shlex.quote(str(directory))
            commands.append(
                f"if test -e {quoted}; then find {quoted} \\( {expression} -o "
                f"\\( -path '*/hidden/*' -name capsule.yaml \\) \\) "
                f"-exec sh -c {emit_files} sh {{}} + || exit 1; fi"
            )
        probe = "; ".join([f"test -s {shlex.quote(str(control))} || exit 1", *commands, "printf 'DONE\\n'"])
    if payload == probe:
        # Simulated transport response, NOT proof that host files were masked.
        (root / "mask_probe_payload.txt").write_text(script)
        print("DONE")
        raise SystemExit(0)
    assert payload.startswith("claude --print "), payload
    command = shlex.split(payload)
    expected = [
        "claude",
        "--print",
        "--model",
        "claude-fixture",
        "--effort",
        "high",
        "--permission-mode",
        "bypassPermissions",
        "--add-dir",
        str(workspace),
        "--output-format",
        "stream-json",
        "--verbose",
        "<",
        str(workspace / "TASK.md"),
    ]
    assert command == expected, command
    os.chdir(workspace)
    os.execv(sys.executable, [sys.executable, str(root / "bin/claude"), *command[1:-2]])

assert Path(sys.argv[0]).name == "claude"
workspace = Path.cwd().resolve(strict=True)
assert workspace.is_relative_to(root)
assert sys.argv[1:3] == ["--print", "--model"]
calls = root / "provider_calls.jsonl"
first = not calls.exists()


def process_identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19], "parent": int(fields[1])}


ancestor = os.getppid()
while b"run_baseline_qa_loop.py" not in Path(f"/proc/{ancestor}/cmdline").read_bytes():
    ancestor = process_identity(ancestor)["parent"]
    assert ancestor > 1, "missing controller ancestor"
brokers = []
for value in Path(f"/proc/{ancestor}/task/{ancestor}/children").read_text().split():
    pid = int(value)
    command = Path(f"/proc/{pid}/cmdline").read_bytes()
    if b"merlin_experiments.phase1.brokers." in command:
        brokers.append(process_identity(pid))
assert len(brokers) >= 2, "real selfcheck and simjob brokers were not started"
with calls.open("a") as stream:
    stream.write(json.dumps({"pid": os.getpid(), "workspace": str(workspace), "brokers": brokers}) + "\n")
partial = workspace / "submission/partial.txt"
if first:
    partial.write_text("preserved across resume\n")
    emit({"type": "rate_limit_event", "rate_limit_info": {"rateLimitType": "seven_day", "status": "rejected"}})
else:
    assert partial.read_text() == "preserved across resume\n"
    # Actual public shim -> actual host broker -> actual selfcheck worker. The
    # incomplete submission must receive an honest failure, never fabricated success.
    emit(
        {
            "type": "assistant",
            "message": {
                "model": "claude-fixture",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "offline-selfcheck",
                        "name": "Bash",
                        "input": {
                            "command": "python agent_selfcheck.py --sim spike --capsules public_member --timeout 1"
                        },
                    }
                ],
            },
        }
    )
    check = subprocess.run(
        [
            sys.executable,
            str(workspace / "agent_selfcheck.py"),
            "--sim",
            "spike",
            "--capsules",
            "public_member",
            "--timeout",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=7,
    )
    (root / "selfcheck.json").write_text(
        json.dumps({"rc": check.returncode, "stdout": check.stdout, "stderr": check.stderr})
    )
    assert check.returncode != 0, check.stdout
    assert "selfcheck_request_id" in check.stdout, check.stdout + check.stderr
    emit(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "offline-selfcheck",
                        "content": check.stdout,
                        "is_error": check.returncode != 0,
                    }
                ]
            },
        }
    )
    emit(
        {
            "type": "assistant",
            "message": {
                "model": "claude-fixture",
                "content": [{"type": "text", "text": "Synthetic incomplete authoring."}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        }
    )
    emit(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "Incomplete fixture; no compiler delivered.",
        }
    )
