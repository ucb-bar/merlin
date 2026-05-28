"""Deterministic replay of the TargetGen executor with hand-written
``manualllm`` responses.

The executor's lifecycle is non-trivial: each task transitions through
``planned`` → ``preflight_ready`` → ``prompt_emitted`` → ``awaiting_response``
→ (operator gate?) → ``proposal_ready`` → ``completed`` (or ``blocked``
on a mutation-policy gate). Without this test, we have no way to catch a
regression in the lifecycle that doesn't involve a real LLM.

By design, tasks with ``mutation_policy != planner_generated_only``
terminate at ``blocked`` after the executor emits a ``branch_gate``
operator request. The executor then **halts the entire pipeline** — it
does not advance past a blocked task. The operator must explicitly enable
live mutation out-of-band (e.g., switch to a dedicated branch and resume
with ``live_mutation_enabled=True``) before the remaining tasks run.
This is the safety contract that keeps Claude Code from autonomously
editing IREE/LLVM submodules.

The contract this test enforces:

  * Every ``planner_generated_only`` task that runs reaches ``completed``.
  * The first mutation-gated task in pipeline order reaches ``blocked``
    after the executor emits exactly one ``branch_gate`` operator
    request. The pipeline halts there.
  * No task ends stuck in ``awaiting_response`` / ``prompt_emitted`` /
    ``awaiting_operator`` once the executor has been driven to a fixed
    point.

When the executor adds new tasks or changes mutation boundaries, this
test fails until the fixture's response set is refreshed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

CAPABILITY = REPO_ROOT / "target_specs" / "examples" / "gemmini_mx" / "capability.yaml"
RESPONSES = Path(__file__).parent / "fixtures" / "manualllm_replay" / "gemmini_mx"

pytestmark = [pytest.mark.integration]


def _merlin(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    base_env = dict(__import__("os").environ)
    base_env["PYTHONPATH"] = str(TOOLS_DIR) + ":" + base_env.get("PYTHONPATH", "")
    if env:
        base_env.update(env)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "merlin.py"), *args],
        capture_output=True,
        text=True,
        env=base_env,
        cwd=REPO_ROOT,
        check=False,
    )


def _read_state(target_dir: Path) -> dict:
    return json.loads((target_dir / "execution_state.json").read_text())


def _read_open_requests(target_dir: Path) -> list[dict]:
    requests_dir = target_dir / "operator_requests"
    if not requests_dir.exists():
        return []
    out: list[dict] = []
    for path in sorted(requests_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "open":
            out.append(payload)
    return out


def test_manualllm_replay_drives_gemmini_mx_to_completion(tmp_path: Path) -> None:
    if not CAPABILITY.exists():
        pytest.skip(f"capability spec missing: {CAPABILITY}")
    if not RESPONSES.is_dir():
        pytest.skip(f"manualllm fixture missing at {RESPONSES}")

    out_dir = tmp_path / "manualllm_out"
    target_dir = out_dir / "gemmini_mx"

    # 1. Bootstrap: orchestrate to emit prompts under a new target_dir.
    rc = _merlin(
        "targetgen",
        "orchestrate",
        str(CAPABILITY),
        "--out-dir",
        str(out_dir),
        "--prompt-backend",
        "manualllm",
    )
    assert rc.returncode == 0, f"orchestrate failed:\n{rc.stderr}"

    prompts_dir = target_dir / "prompts"
    assert prompts_dir.is_dir(), f"prompts dir not created: {prompts_dir}"

    # 2. Stage hand-written responses next to each prompt.
    bundle = json.loads((target_dir / "execution_bundle.json").read_text())
    expected_responses = sorted(t["response_packet"] for t in bundle["tasks"] if t.get("response_packet"))
    fixture_responses = sorted(p.name for p in RESPONSES.glob("*.response.md"))
    missing = set(expected_responses) - set(fixture_responses)
    if missing:
        pytest.fail(
            f"manualllm fixture under {RESPONSES} is missing responses for: "
            f"{sorted(missing)}.\nExpected (from execution bundle): "
            f"{expected_responses}\nProvided: {fixture_responses}"
        )

    for fixture in RESPONSES.glob("*.response.md"):
        shutil.copy2(fixture, prompts_dir / fixture.name)

    # 3. Drive the executor to a fixed point. The lifecycle is:
    #     - each `execute` advances one transition (read response, emit
    #       operator request, etc.) and returns.
    #     - once a task hits `blocked` (mutation gate), the entire
    #       pipeline halts.
    # The fixed point is reached when consecutive `execute` calls produce
    # identical task statuses AND no open operator requests remain.
    max_iterations = 25
    prev_statuses: dict[str, str] | None = None
    statuses: dict[str, str] = {}
    for iteration in range(max_iterations):
        rc = _merlin(
            "targetgen",
            "execute",
            "--from-dir",
            str(target_dir),
            "--prompt-backend",
            "manualllm",
            "--resume",
        )
        assert rc.returncode == 0, (
            f"execute iteration {iteration} failed (exit {rc.returncode}):\n"
            f"STDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}"
        )

        state = _read_state(target_dir)
        statuses = {tid: t["status"] for tid, t in state["tasks"].items()}
        requests = _read_open_requests(target_dir)

        if statuses == prev_statuses and not requests:
            break

        for req in requests:
            choice = (
                "continue_without_mutation"
                if any(o["id"] == "continue_without_mutation" for o in req["options"])
                else req["recommended_option"]
            )
            ans = _merlin(
                "targetgen",
                "answer",
                "--target-dir",
                str(target_dir),
                "--question-id",
                req["id"],
                "--choice",
                choice,
            )
            assert ans.returncode == 0, f"answer failed for {req['id']}:\n{ans.stderr}"
        prev_statuses = statuses
    else:
        pytest.fail(
            f"executor did not reach a fixed point within {max_iterations} iterations.\n" f"final statuses: {statuses}"
        )

    # 4. Verify the lifecycle contract:
    #    - planner_generated_only tasks that the executor reached must have
    #      completed.
    #    - The first mutation-gated task in pipeline order must be `blocked`
    #      and must have produced exactly one branch_gate operator request.
    #    - No task should be stuck in a non-terminal lifecycle state
    #      (awaiting_response / prompt_emitted / awaiting_operator).
    bundle_tasks = [t for t in bundle["tasks"]]  # already in pipeline order
    final_state = _read_state(target_dir)

    non_terminal = {
        "preflight_ready",
        "prompt_emitted",
        "awaiting_response",
        "awaiting_operator",
        "proposal_ready",
    }
    stuck = [tid for tid, t in final_state["tasks"].items() if t["status"] in non_terminal]
    assert not stuck, f"executor reached fixed point with tasks stuck in non-terminal " f"states: {stuck}"

    saw_first_mutation_block = False
    for t in bundle_tasks:
        tid = t["id"]
        status = final_state["tasks"][tid]["status"]
        policy = t["mutation_policy"]
        if not saw_first_mutation_block:
            if policy == "planner_generated_only":
                assert status == "completed", (
                    f"non-mutation task {tid} (before first gate) did not " f"complete: status={status}"
                )
            else:
                assert status == "blocked", (
                    f"first mutation-gated task {tid} did not reach `blocked`; " f"status={status}"
                )
                saw_first_mutation_block = True
        else:
            # After the first gate the executor halts; remaining tasks may
            # still be `planned`. Just assert nothing has spuriously advanced.
            assert status in {
                "planned",
                "completed",
                "blocked",
            }, f"task {tid} after halted gate has unexpected status {status}"

    assert saw_first_mutation_block, (
        "no mutation-gated task exists in this fixture; the test only "
        "covers the planner_generated_only path. Refresh the fixture to "
        "include a mutation task."
    )

    # The blocked task should have emitted a branch_gate operator request.
    requests_dir = target_dir / "operator_requests"
    request_ids = {p.stem for p in requests_dir.glob("*.json")} if requests_dir.exists() else set()
    blocked_task_ids = [tid for tid, t in final_state["tasks"].items() if t["status"] == "blocked"]
    for tid in blocked_task_ids:
        assert any(rid.startswith(tid) for rid in request_ids), (
            f"blocked task {tid} produced no operator request; " f"the safety contract was bypassed"
        )
