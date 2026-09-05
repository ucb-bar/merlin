#!/usr/bin/env python3
"""Launch ONE isolated, instrumented target-generation experiment, then grade it.

The session that BUILT this harness is not an experiment run. A valid run is created only here: a
fresh run dir, a workspace assembled from exactly one declared input bundle (denied paths masked),
a non-interactive agent launch with full transcript + token/cost capture, a frozen submission, and a
separate full-access grading phase.

Agent launch:
  - real:  claude --print --model <m> --output-format stream-json --verbose < TASK.md   (in workspace)
  - dummy: --dummy-agent copies a known-good reference submission (pipeline validation, no LLM)

Sandbox:
  - bwrap (real runs): bind allowed paths RO, fresh writable workspace, /scratch* hidden.
  - none (dummy/dev): copy-based workspace; denied paths asserted absent.

Usage:
  run_agent_experiment.py --arm raw_baseline --run-id rb_0001 --model opus [--sandbox bwrap]
  run_agent_experiment.py --arm raw_baseline --run-id dry_0001 --dummy-agent \
      --dummy-submission out/artifacts/targets/<target>/agent_spec_v1_mlir_oot --no-oracle-grade \
      --grade-capsules /tmp/grade_subset
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

import _common as C

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.common import arrival_stamp as AS  # noqa: E402  (one arrival-time convention for every driver's transcript)
from merlin.targetgen import experiment_tokens as ET  # noqa: E402

ARM_BUNDLE = {"raw_baseline": "raw_baseline_public_v0", "merlin_assisted": "merlin_assisted_public_v0",
              "cpp_merlininfra": "cpp_merlininfra_hwbringup_v0"}


def _load_bundle(arm: str) -> dict:
    b = C.BUNDLES / ARM_BUNDLE[arm] / "input_bundle_manifest.yaml"
    return yaml.safe_load(b.read_text())


def _is_answer_file(p: Path) -> bool:
    """Is this an ANSWER surface the bring-up agent must not have?

    Keyed on the corpus's derived naming convention: golden outputs, expected command buffers, and the
    expected instruction coverage, and the private external model instance from which a model golden
    derives. Never a per-capsule literal. Kept in lockstep with the canonical ``answer_surfaces`` bwrap
    mask; this is defense in depth for the non-sandbox (``--sandbox none``) assembly path.
    """
    n = p.name
    return (n in ("golden.yaml", "golden.npy") or ".golden." in n
            or n.startswith("expected_command_buffer") or n == "expected_instruction_coverage.yaml"
            or n.endswith(".safetensors")
            or n.endswith(".safetensors.manifest.json"))


def _link_filtered(src: Path, dst: Path) -> None:
    """Materialize ``src`` at ``dst`` for the agent workspace, EXCLUDING any answer surface. A dir with no
    answer file below it is symlinked whole (cheap); a dir that contains one is mirrored as a tree of
    per-file symlinks with the answer files omitted; a plain file is symlinked (unless it is itself an
    answer file, then skipped). So the assembled workspace is answer-free by construction even without the
    sandbox mask."""
    if src.is_dir():
        if not any(_is_answer_file(f) for f in src.rglob("*") if f.is_file()):
            dst.symlink_to(src)
            return
        dst.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            if child.is_file() and _is_answer_file(child):
                continue
            _link_filtered(child, dst / child.name)
    elif not _is_answer_file(src):
        dst.symlink_to(src)


def assemble_workspace(bundle: dict, ws: Path, *,
                       _policy_test_live_inputs: bool = False) -> list[str]:
    """Freeze the bundle inputs, then expose friendly links in the workspace.

    The links retain the paths prompts already teach, but the real bwrap mount at
    each link target comes from the run-private snapshot, never from the live
    worktree.  Answer surfaces are still filtered from this convenience view and
    independently masked by the canonical bwrap answer-surface pass.
    """
    from merlin.targetgen.sandbox import bwrap as _BW

    if not _policy_test_live_inputs:
        _BW.materialize_bundle_inputs(ws, bundle, repo=C.REPO)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "submission").mkdir(exist_ok=True)
    skipped: list[str] = []
    for entry in bundle.get("allowed", []):
        # THROUGH the shared resolver, which is the only thing that keeps this in lockstep with the
        # sandbox binder. Restating two of its three cases here and omitting the third -- the grant whose
        # home the TARGET REGISTRY owns, because a target's package does not always live under
        # merlin/targets/ -- silently dropped exactly one kind of input. Measured on
        # merlincirct_atlasp1arm4: the RTL-checks arm is granted `merlin/targets/atlas/contracts/
        # rtl_facts/`, TASK.md tells it to derive the ISA/mesh/datapath from that path and
        # ALLOWED_MERLIN_TOOLS.md lists it by name; the binder resolved it and bound the bytes, this
        # loop did not and exposed no link, so the arm whose whole purpose is compiling FROM RTL facts
        # had no way to find them. 503 tool calls, one touched that surface, five RTL-workflow
        # conformance checks failed, and it read as an agent ignoring its tooling.
        src = _BW.resolve_grant(entry["path"], C.REPO)
        if not src.exists():
            skipped.append(entry["path"])
            continue
        # Honor an explicit ``as:`` alias (e.g. hwbringup set mounted as ``<target>/``) so the workspace
        # entry matches the name the prompt tells the agent to read; else fall back to the basename.
        dst = ws / (entry.get("as") or Path(entry["path"]).name)
        if dst.exists() or dst.is_symlink():
            dst = ws / entry["path"].replace("/", "_").rstrip("_")
        try:
            _link_filtered(src, dst)          # answer surfaces omitted from the materialized tree
        except FileExistsError:
            pass
    if skipped:
        # Never silent: a granted tool that is not in the workspace is indistinguishable, to the agent,
        # from a tool it was never granted -- and it will not ask.
        print(f"[workspace] {len(skipped)} granted path(s) could not be placed: {sorted(skipped)}",
              flush=True)
    return [Path(d["path"]).name for d in bundle.get("denied", [])]


def assert_isolation(ws: Path, bundle: dict) -> list[str]:
    """Confirm no denied path is reachable as a workspace entry. Returns violations (empty = ok)."""
    viol = []
    for d in bundle.get("denied", []):
        target = (C.REPO / d["path"]).resolve()
        for entry in ws.iterdir():
            try:
                if entry.is_symlink() and entry.resolve() == target:
                    viol.append(f"denied path reachable: {entry.name} -> {d['path']}")
                elif entry.resolve() == target:
                    viol.append(f"denied path present: {d['path']}")
            except Exception:
                pass
    return viol


def _dummy_transcript(path: Path) -> None:
    """A minimal stream-json transcript so token/cost extraction is exercised on the dummy path."""
    evs = [
        {"type": "assistant", "message": {"id": "dummy1", "model": "claude-opus-4-8",
            "usage": {"input_tokens": 4200, "cache_creation_input_tokens": 800,
                      "cache_read_input_tokens": 15000, "output_tokens": 950},
            "content": [{"type": "thinking", "text": "..."},
                        {"type": "tool_use", "name": "Write"},
                        {"type": "tool_use", "name": "Bash"}]}},
        {"type": "result", "subtype": "success"},
    ]
    path.write_text("\n".join(json.dumps(e) for e in evs) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_BUNDLE))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--task", default=str(C.EXP / "task" / "TASK.md"))
    ap.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap")
    ap.add_argument("--allow-unsandboxed", action="store_true",
                    help="permit a real (non-dummy) run with --sandbox none (NOT for trusted results)")
    ap.add_argument("--dummy-agent", action="store_true",
                    help="copy a known-good submission instead of launching an LLM (pipeline test)")
    ap.add_argument("--dummy-submission",
                    default=str(C.REPO / "out/artifacts/targets" / C.TARGET / "agent_spec_v1_mlir_oot"))
    ap.add_argument("--no-oracle-grade", action="store_true")
    ap.add_argument("--grade-capsules", default="",
                    help="comma-separated capsule roots for the public/dev grade. Empty (the default) "
                         "RESOLVES them from the target descriptor — the primary corpus plus its "
                         "sibling categories — which is the only spelling that grades the whole suite "
                         "for the target actually under test.")
    ap.add_argument("--grade-hidden-capsules", default="",
                    help="comma-separated capsule roots for the hidden grade. Empty resolves the "
                         "target's own hidden/ dir from the descriptor.")
    ap.add_argument("--skip-hidden", action="store_true")
    a = ap.parse_args(argv)

    # A real (non-dummy) run MUST be sandboxed: without bwrap the agent can read any absolute path
    # (incl. denied /scratch* dirs), so workspace assembly alone does not isolate.
    if not a.dummy_agent and a.sandbox != "bwrap" and not a.allow_unsandboxed:
        print("REFUSING: a real run requires --sandbox bwrap (or explicit --allow-unsandboxed). "
              "Workspace assembly alone does not hide denied absolute paths.", file=sys.stderr)
        return 4

    bundle = _load_bundle(a.arm)
    run_dir = C.RUNS / a.arm / a.run_id
    if run_dir.exists():
        print(f"run dir exists, refusing to overwrite: {run_dir}", file=sys.stderr)
        return 2
    ws = run_dir / "workspace"
    run_dir.mkdir(parents=True)

    # provenance + bundle copy + task
    shutil.copy(C.BUNDLES / ARM_BUNDLE[a.arm] / "input_bundle_manifest.yaml",
                run_dir / "input_bundle_manifest.yaml")
    shutil.copy(a.task, run_dir / "TASK.md")
    denied_names = assemble_workspace(
        bundle, ws, _policy_test_live_inputs=a.sandbox != "bwrap")
    from merlin.targetgen.sandbox import bwrap as _BW
    snapshot = _BW.snapshot_record(ws) if a.sandbox == "bwrap" else None
    # Verification-spec contract: the QA acceptance spec the agent builds to (target ops, dtypes, numeric
    # acceptance policy, datapath coverage) — DERIVED from the answer-free capsule declarations, never a
    # golden. Written into the workspace so it sits alongside the RTL/docs; TASK.md points at it. Advisory:
    # never block a run if it cannot render.
    try:
        from merlin.targetgen.verification_spec import write_spec as _write_vspec
        from merlin.targetgen.target_experiment import load_target_experiment as _lte
        _desc = C.EXP / "target_experiment.yaml"
        if _desc.is_file():
            _write_vspec(_lte(_desc), ws)
            with open(run_dir / "TASK.md", "a", encoding="utf-8") as _tf:
                _tf.write(
                    "\n\n## Verification spec (your acceptance contract)\n"
                    "Read `verification_spec.md` in your workspace — the verification team's spec of the "
                    "operations, datatypes, numeric acceptance policy, and datapath coverage you must "
                    "satisfy to pass. There is no answer key: validate by computing each operation's "
                    "expected result yourself from the declared inputs, running your artifact on the RTL, "
                    "and debugging with the disassembler / trace / hardware-state tools.\n")
    except Exception as _e:  # noqa: BLE001 — the spec is a convenience contract; never block a run on it
        print(f"[verification_spec] skipped: {type(_e).__name__}: {_e}", file=sys.stderr)
    viol = assert_isolation(ws, bundle)
    (run_dir / "environment.yaml").write_text(yaml.safe_dump({
        "run_id": a.run_id, "arm": a.arm, "model": a.model, "effort": a.effort,
        "sandbox": a.sandbox, "dummy_agent": a.dummy_agent,
        "repo_sha": C.repo_sha(), "bundle_id": bundle["bundle_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "isolation_violations": viol,
        "denied_paths_checked": denied_names,
        "bundle_input_snapshot": snapshot,
    }, sort_keys=False))
    if viol:
        print(f"ISOLATION FAILURE: {viol}", file=sys.stderr)
        return 3

    transcript = run_dir / "transcript.jsonl"
    t0 = time.time()
    exit_code = 0
    if a.dummy_agent:
        # harness stand-in: emit a known-good submission + a synthetic transcript (no LLM)
        src = C.REPO / a.dummy_submission
        dst = run_dir / "submission"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("build", "__pycache__"))
        _dummy_transcript(transcript)
        (run_dir / "claude_stdout.log").write_text("[dummy-agent] copied known-good submission\n")
        (run_dir / "claude_stderr.log").write_text("")
    else:
        cmd = (f'claude --print --model {a.model} --effort {a.effort} '
               f'--permission-mode bypassPermissions --output-format stream-json --verbose '
               f'< {run_dir / "TASK.md"}')
        if a.sandbox == "bwrap":
            cmd = _bwrap_wrap(cmd, ws, bundle)
        # Streamed, not redirected: a straight stdout redirect leaves no process able to observe a
        # line, so the transcript carries no per-event wall time. arrival_stamp appends `arrived_at`
        # to every event, in the same shape the codex driver writes.
        exit_code = AS.stream_stamped(
            ["bash", "-c", cmd], cwd=ws, transcript=transcript,
            stderr_path=run_dir / "claude_stderr.log",
            raw_path=run_dir / "claude_stream.raw.jsonl")
        # the agent writes its package into workspace/submission; capture it
        wsub = ws / "submission"
        if wsub.exists():
            dst = run_dir / "submission"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(wsub, dst, ignore=shutil.ignore_patterns("build", "__pycache__"))
    wall = round(time.time() - t0, 3)

    # process metrics
    summ = ET.parse_transcript(transcript)
    ET.write_cost_yaml(summ, run_dir / "cost_time_toolcalls.yaml",
                       wall_time_seconds=wall, model=a.model, exit_code=exit_code)

    # separate full-access grading phase
    # WHICH capsules this run is graded on is part of the experiment's definition, not a CLI default.
    # The default used to be the shared parent of every target's corpus: for the target whose corpus IS
    # that parent it happened to be right, and for every other target it graded foreign capsules (or,
    # for the hidden phase, another target's hidden set entirely). Resolve from the descriptor.
    pub_roots, hid_roots = _grade_roots(a)
    grade_cmd = [sys.executable, str(C.EXP / "scripts" / "grade_agent_run.py"),
                 "--run-dir", str(run_dir), "--arm", a.arm, "--model", a.model,
                 "--capsules", pub_roots]
    if hid_roots:
        grade_cmd += ["--hidden-capsules", hid_roots]
    elif not a.skip_hidden:
        # A hidden phase with no hidden capsules scores 0/0 and reads as a pass. Say so and skip it.
        print("[run_agent_experiment] no hidden capsules for this target — skipping the hidden phase "
              "rather than recording a 0/0 that looks like one", flush=True)
        grade_cmd.append("--skip-hidden")
    if a.no_oracle_grade:
        grade_cmd.append("--no-oracle")
    if a.skip_hidden:
        grade_cmd.append("--skip-hidden")
    subprocess.run(grade_cmd, cwd=str(C.REPO))
    print(f"run complete: {run_dir}")
    return 0


def _grade_roots(a) -> tuple[str, str]:
    """``(public_roots, hidden_roots)`` as comma-separated strings, from the flags or the descriptor.

    Explicit flags win, so a one-off subset run stays possible; otherwise the descriptor decides, which
    is what keeps "the corpus this target declares" and "the corpus this run is graded on" the same
    thing. Falling back to a fixed path is deliberately NOT an option here — that fallback is the defect.
    """
    from merlin.targetgen.target_experiment import load_target_experiment
    desc = C.EXP / "target_experiment.yaml"          # C.EXP honors MERLIN_TARGET_EXPERIMENT
    if a.grade_capsules and (a.grade_hidden_capsules or a.skip_hidden):
        return a.grade_capsules, a.grade_hidden_capsules
    if not desc.is_file():
        raise SystemExit(f"no target descriptor at {desc}: cannot resolve which capsules to grade on, "
                         "and defaulting to a fixed path is how a target gets graded on another "
                         "target's capsules. Pass --grade-capsules explicitly.")
    te = load_target_experiment(desc)
    pub = a.grade_capsules or ",".join(str(r) for r in te.graded_roots())
    hid = a.grade_hidden_capsules or ",".join(str(r) for r in te.hidden_roots())
    return pub, hid


def bwrap_argv(ws: Path, bundle: dict) -> list[str]:
    """Deny-by-default bwrap argv prefix (delegates to the shared, target-agnostic sandbox base): system
    RO, /scratch* tmpfs-hidden, ONLY the bundle's allowed paths bound RO, denied sub-paths re-masked,
    workspace writable+last. Reused by the launcher (real runs) and the pre-flight canary probe."""
    from merlin.targetgen.sandbox import bwrap as _BW
    return _BW.base_argv(ws, bundle, repo=C.REPO)


def _bwrap_wrap(cmd: str, ws: Path, bundle: dict) -> str:
    return " ".join(bwrap_argv(ws, bundle)) + f" bash -c '{cmd}'"


if __name__ == "__main__":
    raise SystemExit(main())
