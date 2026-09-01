#!/usr/bin/env python3
"""Raw-baseline capsule_bench_v0 PILOT run with a redacted QA-gate iterate-to-pass loop.

A fresh, sandboxed Claude agent authors `submission/` from the allowed bundle ONLY, with goldens
withheld (masked). Between rounds an operator-side QA gate (`qa_check.py`) grades the current
submission against the public pilot capsules and writes a REDACTED verdict (pass/fail + failure
plane + trace violations — never expected/golden values) into the agent's `qa/verdict.json`. The
agent reads it and iterates. The loop ends when all 4 pilot capsules pass or a round/wall cap is hit.

Design = multi-round relaunch (robust; no daemon). Each round is a fresh agent context that resumes
from its own `submission/` + `docs/iteration_notes.md` + `qa/verdict.json`. Process telemetry
(wall/cost/tokens/tool-calls) is SUMMED across rounds = total effort to pass.

After convergence: copy the final submission into the run dir, freeze, and run the official
public+hidden grading record via `grade_agent_run.py` (against the pilot capsule subset).

This driver builds the QA substrate (UNCOUNTED). Only the agent's autonomous authoring counts.

Usage:
  run_baseline_qa_loop.py --run-id rb_pilot_0001 [--model claude-opus-4-8] [--max-rounds 6]
                          [--round-timeout 3600] [--no-oracle] [--skip-hidden] [--sandbox bwrap|none]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import yaml

import _common as C
import run_agent_experiment as RX  # reuse bundle/workspace/bwrap primitives
import _ratelimit as RL  # five-hour rate-limit detection + reset epoch (shared)
import resume_on_quota as ROQ  # cut-short (timeout / weekly-quota) detection + checkpoint/resume policy

SCRIPTS = Path(__file__).resolve().parent  # this scripts/ dir (selfcheck_broker.py, selfcheck_shim.py)
_MODEL_HOST_SNAPSHOT_ROOT_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"
_MODEL_HOST_SNAPSHOT_REQUIRED_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"


def _strip_build_state(root: Path) -> None:
    """Delete all cmake/ninja build state under `root` so a graded copy builds from scratch in its OWN
    path. Excluding a dir named 'build' is not enough — a stale CMakeCache / ninja state elsewhere pins the
    original source dir and makes cmake error ('source does not match cache'). The abc9 baseline L3-0/20
    'build' bug. Guarantees a clean, relocatable build for every grade."""
    for pat in ("CMakeCache.txt", "CMakeFiles", "build.ninja", ".ninja_deps", ".ninja_log",
                "cmake_install.cmake"):
        for p in list(Path(root).rglob(pat)):
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except Exception:
                pass

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.common.paths import ext_path  # noqa: E402
from merlin.targetgen import experiment_tokens as ET  # noqa: E402
from merlin.targetgen import tool_registry as _TR  # noqa: E402  (the arm-gated tool catalog)

def _repo_root():
    from pathlib import Path as _P
    p = _P(__file__).resolve()
    while p != p.parent and not (p / "merlin" / "python").is_dir():
        p = p.parent
    return p
_ROOT = _repo_root()

ARM = "raw_baseline"  # default arm; overridable via --arm (the QA loop is arm-agnostic)
# Loop+gate capsule set: the target's PUBLIC capsules, DERIVED from the descriptor's capsule_corpus
# (+ sibling corpora) and materialized per-target — NOT a committed gemmini set. So an atlas run grades
# atlas fp8/bf16 capsules (arc L3, float-tolerance), a gemmini run its i8 set (spike L2, exact_int); no
# target leak. `None` = "resolve lazily from the descriptor" (see _pilot_subset); run_fullsuite overrides
# it with the full 25-capsule set for the final grade, so it stays a settable module attribute.
PILOT_SUBSET = None
_DERIVED_PILOT = None  # cache of the descriptor-derived set (populated on first _pilot_subset() call)
VERILATOR_ATTEMPTS = 3  # cycle-accurate L3 checkpoint chances after spike-convergence (1 fix-round between)


def _l3_barrier_decision(l3_all_pass: bool, rnd: int, max_rounds: int) -> str:
    """The NON-TERMINAL barrier decision (extracted so it is unit-testable — this is the abc7 regression).
    Returns: 'done' (L3 passed → the ONLY success exit), 'budget' (round budget exhausted → stop honestly,
    reason=max_rounds), or 'iterate' (L3 failed but rounds remain → feed back + fix round, do NOT end)."""
    if l3_all_pass:
        return "done"
    if rnd >= max_rounds:
        return "budget"
    return "iterate"


def _l3_fix_verdict(vv: dict, attempt: int) -> dict:
    """Redacted L3 feedback for the next fix round, including Arm4's answer-free RTL readback."""
    out = {
        "stage": "verilator_checkpoint", "attempt": attempt, "all_pass": False,
        "n_passed": vv["n_passed"], "n_capsules": vv["n_capsules"],
        "note": "CYCLE-ACCURATE (verilator/L3) results — fix the capsules failing at L3, then "
                "re-run agent_selfcheck (spike) clean before declaring READY again.",
        "per_capsule": vv["per_capsule"],
    }
    if vv.get("rtl_checks") is not None:
        out["rtl_checks"] = vv["rtl_checks"]
        out["rtl_checks_note"] = (
            "Answer-free RTL structural feedback regenerated from this L3 attempt's emitted "
            "lowering/trace. Read and address it during this fix round.")
    return out


def _authoring_completion(numeric_all_pass: bool, workflow_conformant: bool) -> bool:
    """The agent may leave the authoring loop only when both in-sandbox gates are evidenced."""
    return bool(numeric_all_pass and workflow_conformant)


def _formal_completion(numeric_all_pass: bool, workflow_conformant: bool,
                       official_grade_complete: bool) -> bool:
    """Formal success additionally requires the outer, post-freeze public+hidden L3 grade."""
    return bool(numeric_all_pass and workflow_conformant and official_grade_complete)


def _workflow_conformance(tpath: Path, submission_dir: Path, arm: str, endpoint_kind: str,
                          resolved_tools) -> tuple[dict, bool]:
    """Recompute the workflow verdict from the *current* authoring transcript and submission.

    The resolved bundle tools, not the driver's legacy arm spelling, select Arm4's extra requirements.
    Every caller gets the same fail-closed exception handling so an unavailable checker can never leave a
    stale ``True`` from an earlier round in force.
    """
    try:
        import conformance as _CONF
        conf = _CONF.compute(tpath, submission_dir, arm, endpoint_kind,
                             resolved_tools=resolved_tools)
    except Exception as exc:  # noqa: BLE001 — conformance is a formal gate; unavailable means false
        conf = {"conformant": False, "error": f"{type(exc).__name__}: {exc}"}
    return conf, conf.get("conformant") is True


def _official_grade_result(returncode: int, run_dir: Path, *, required_tier: str = "L3") -> dict:
    """Validate the official grader's exit status *and* its claim-bearing manifest.

    The subprocess return code is necessary but not sufficient: this rejects a stale/malformed manifest,
    a vacuous 0/0 phase, and a phase whose status says complete without the matching L3/RTL evidence.
    """
    failures: list[str] = []
    if returncode != 0:
        failures.append(f"grader_exit_nonzero:{returncode}")
    manifest_path = run_dir / "run_manifest.yaml"
    try:
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001 — malformed/missing evidence is a refusal, not a crash
        manifest = {}
        failures.append(f"manifest_unreadable:{type(exc).__name__}")
    if not isinstance(manifest, Mapping):
        manifest = {}
        failures.append("manifest_malformed")

    completion = manifest.get("completion")
    if not isinstance(completion, Mapping):
        failures.append("completion_status_missing")
    else:
        if completion.get("formal_grade_complete") is not True:
            failures.append("grader_reported_incomplete")
        if completion.get("required_tier") != required_tier:
            failures.append("required_tier_mismatch")

    for phase_name in ("public_dev", "hidden"):
        phase = manifest.get(phase_name)
        if not isinstance(phase, Mapping):
            failures.append(f"{phase_name}:status_missing")
            continue
        n_capsules, n_passed = phase.get("n_capsules"), phase.get("n_passed")
        if (not isinstance(n_capsules, int) or isinstance(n_capsules, bool)
                or n_capsules <= 0):
            failures.append(f"{phase_name}:capsule_set_empty_or_malformed")
        if (not isinstance(n_passed, int) or isinstance(n_passed, bool)
                or not isinstance(n_capsules, int) or isinstance(n_capsules, bool)
                or n_passed != n_capsules):
            failures.append(f"{phase_name}:not_all_capsules_passed")
        if phase.get("formal_complete") is not True:
            failures.append(f"{phase_name}:formal_complete_not_true")
        if phase.get("gradeable") is not True:
            failures.append(f"{phase_name}:not_gradeable")
        if phase.get("integrity_status") != "clean":
            failures.append(f"{phase_name}:integrity_not_clean")
        if phase.get("numeric_all_exact") is not True:
            failures.append(f"{phase_name}:numeric_exactness_not_complete")
        if phase.get("trace_all_pass") is not True:
            failures.append(f"{phase_name}:trace_conformance_not_complete")
        unmeasured = phase.get("unmeasured_counts")
        for field in (
            "n_not_graded_ineligible", "n_gated_deferred", "n_screened_only",
            "n_budget_exhausted", "n_incomplete", "n_not_gradeable_no_oracle",
        ):
            if not isinstance(unmeasured, Mapping) or unmeasured.get(field) != 0:
                failures.append(f"{phase_name}:{field}_not_zero")
        tiers = phase.get("tier_reached")
        if not isinstance(tiers, Mapping) or tiers.get(required_tier) != n_capsules:
            failures.append(f"{phase_name}:not_all_capsules_reached_{required_tier}")
        evidence = phase.get("pass_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("rtl_backed") != n_capsules:
            failures.append(f"{phase_name}:not_all_capsules_rtl_backed")
    return {
        "complete": not failures,
        "grader_returncode": int(returncode),
        "manifest": str(manifest_path),
        "failures": failures,
    }


def _spend_over_cap(this_round_cost) -> tuple[bool, float, float]:
    """Enforce the experiment's DOLLAR ceiling in code. Append this round's cost to a SHARED spend ledger
    (all arm processes of the batch write to ``MERLIN_SPEND_LEDGER``) and return ``(over_cap, total, cap)``
    against ``MERLIN_MAX_SPEND_USD``. Once the running total across EVERY arm crosses the cap, each arm stops
    before its next round — a soft ceiling whose overshoot is bounded by one in-flight round per arm (a
    round's true cost is only known once it completes). No cap / no ledger configured → never over. The cost
    is the now-authoritative, subagent-inclusive figure from :func:`experiment_tokens.parse_transcript`."""
    import os as _os
    cap = float(_os.environ.get("MERLIN_MAX_SPEND_USD") or 0)
    ledger = _os.environ.get("MERLIN_SPEND_LEDGER")
    if cap <= 0 or not ledger:
        return False, 0.0, 0.0
    import fcntl
    # An UNMEASURED round is not a free one. `float(cost or 0)` booked $0 for any round whose usage never
    # arrived -- and the usage of a round killed by --round-timeout never arrives at all, because the
    # driver only emits it on `turn.completed` and a killed turn never completes. Measured: three separate
    # A/B legs across v9, v10 and v11 each burned a full four-hour round that the ledger recorded as
    # costing nothing. A metered run could therefore overrun its ceiling by an arbitrary amount while
    # every gate reported it comfortably under.
    #
    # Unmeasured rounds are written with `cost: null` + `unmeasured: true` and counted separately, so the
    # cap is enforced on what is KNOWN while the unknown is visible rather than silently zero. The cap
    # cannot be enforced against a number nobody has; what it can do is refuse to pretend the number is 0.
    _unmeasured = this_round_cost is None
    c = None if _unmeasured else float(this_round_cost)
    p = Path(ledger)
    p.parent.mkdir(parents=True, exist_ok=True)
    total, n_unmeasured = 0.0, 0
    with open(p, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"cost": c, "unmeasured": _unmeasured}) + "\n")
        f.flush()
        f.seek(0)
        for line in f:
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 — a malformed ledger line must not defeat the cap
                continue
            if row.get("unmeasured") or row.get("cost") is None:
                n_unmeasured += 1
                continue
            try:
                total += float(row.get("cost") or 0)
            except Exception:  # noqa: BLE001
                continue
        fcntl.flock(f, fcntl.LOCK_UN)
    if n_unmeasured:
        print(f"  [spend] ${total:.2f} of ${cap:.2f} measured, plus {n_unmeasured} UNMEASURED round(s) "
              f"whose usage never arrived — the true total is a LOWER BOUND", flush=True)
    return total >= cap, total, cap


def _verilator_per_capsule_timeout() -> int:
    """Per-capsule L3 (verilator RTL cert) timeout, from a T_obs that is POSITIVELY confirmed to be THIS
    target's sim (generous 2x, min 900s). The readiness gate writes scripts/.oracle_timing.json — but that
    path is a symlink shared across targets, so a radiance run must NOT inherit a GemminiRocketConfig T_obs
    measured for a different, far lighter RTL (which would floor to 900s and mass-timeout every L3 capsule,
    the abc9 "L3 0/N = timeout not skill" trap). A measurement is trusted only when it is target-scoped:
    the file ``.oracle_timing.<target>.json`` OR a legacy ``.oracle_timing.json`` whose ``target`` field
    matches. An unconfirmed / foreign-config measurement is ignored in favor of a conservative bound and a
    loud log — never a silent under-time."""
    import math
    tgt = _te().target
    for p in (C.EXP / "scripts" / f".oracle_timing.{tgt}.json",
              C.EXP / "scripts" / ".oracle_timing.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if p.name.endswith(f".{tgt}.json") or d.get("target") == tgt:
            return max(900, int(math.ceil(2 * float(d["verilator_per_capsule_s"]))))
        print(f"[timeout] ignoring {p.name}: T_obs measured on config={d.get('config')!r} "
              f"target={d.get('target')!r}, not {tgt!r} — needs a target-scoped measurement",
              file=sys.stderr)
    print(f"[timeout] no target-confirmed L3 timing for {tgt!r}; using conservative 2400s. Run readiness "
          f"or the L3 measurement to record scripts/.oracle_timing.{tgt}.json", file=sys.stderr)
    return 2400


def _cycle_accurate_checkpoint_enabled() -> tuple[bool, str]:
    """Whether to run the cycle-accurate RTL-cert (verilator/L3) barrier for THIS target.

    Target-agnostic + DERIVED, never a per-target literal: the barrier's cycle-accurate cert tiers are the
    checkpoint ladder (:func:`qa_checkpoint_adapters`) MINUS the fast loop tier (:func:`qa_loop_adapters`) —
    e.g. verilator L3 (and VCS L4) once spike/cyclotron L2 is the loop gate. The barrier is a pass-gate ONLY
    when at least one of those cert tiers is a MANDATORY tier of the pilot corpus (listed in a public
    capsule's ``required_oracle_tiers``). A target whose corpus makes every cert tier OPTIONAL — a prototype
    / not-RTL-certified accelerator graded on its functional oracle (its required tier is the fast
    functional sim, e.g. cyclotron L2) — SKIPS the barrier, so a normal run is never blocked on a
    slow/hanging RTL sim (the atlas-0/N `oracle_unavailable` trap, here on the driver side). An explicit
    ``MERLIN_CAPSULE_L3_CHECKPOINT`` env var forces it on (opt-in RTL cert when the sim is available and
    there is time) or off, overriding the derivation."""
    env = os.environ.get("MERLIN_CAPSULE_L3_CHECKPOINT")
    if env is not None:
        on = env.strip().lower() in ("1", "true", "yes", "on")
        return on, f"MERLIN_CAPSULE_L3_CHECKPOINT={env!r} (explicit opt-{'in' if on else 'out'})"
    from merlin.targetgen import capsule_runner as _CR
    try:
        te = _te()
        ck = _CR.qa_checkpoint_adapters(te.target, te.sim_via)
        loop = _CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=_declared_loop_tiers())
    except Exception:  # noqa: BLE001 — no resolvable checkpoint oracle -> nothing to gate on
        ck, loop = {}, {}
    cert_tiers = set(ck) - set(loop)   # the cycle-accurate cert tiers held back from the fast loop
    if not cert_tiers:
        return False, "no cycle-accurate cert tier beyond the fast loop oracle for this target"
    mandatory: set[str] = set()
    for cf in Path(_pilot_subset()).rglob("capsule.yaml"):
        try:
            doc = yaml.safe_load(cf.read_text()) or {}
        except Exception:  # noqa: BLE001
            continue
        mandatory |= set(doc.get("required_oracle_tiers") or [])
    hit = sorted(cert_tiers & mandatory)
    if hit:
        return True, f"cert tier(s) {hit} are mandatory in the pilot corpus"
    return False, (f"cert tier(s) {sorted(cert_tiers)} are OPTIONAL for the pilot corpus (functional-tier "
                   f"pass bar) — skipping the cycle-accurate RTL barrier; set "
                   f"MERLIN_CAPSULE_L3_CHECKPOINT=1 to opt in")
_EXPERIMENT = "full"    # set in main(): 'full' (abc1) or 'realistic' (abc2, whole-repo + self-check)
_ARM = ""               # set in main(): the arm being run (enforces the per-arm language mandate)
_ADD_TOOLS: tuple = ()  # set in main(): --with-tool    (ABLATION: grant these on top of the rung)
_DROP_TOOLS: tuple = () # set in main(): --without-tool (ABLATION: withhold these from the rung)
_DRIVER = "auto"        # set in main(): agent driver (auto|converse|claudecode|opencode); auto routes by model
_SUBAGENT_MODEL = ""    # set in main(): delegate/subagent model for tier-within-agent (converse driver)
_BACKGROUND_MODEL = ""  # set in main(): background/mechanical model (converse driver; reserved)
READY_MARKER = "READY_FOR_BARRIER"  # realistic: agent drops submission/<this> to self-declare done
# Merlin-arm-only docs staged into the workspace alongside the (shared) graded task.
MERLIN_WS_DOCS = ("TASK_ADDENDUM.md", "ALLOWED_MERLIN_TOOLS.md", "MERLIN_PROVENANCE_TEMPLATE.md")
_TREATMENT_BUNDLE_DECLARATIONS = (
    "input_bundle_manifest.yaml", "allowed_files.txt", "tools.txt",
)


def _file_digest(path: Path) -> tuple[str, int]:
    """Return ``(sha256, bytes)`` for one ordinary file, refusing symlink indirection."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"treatment input is not an ordinary file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _treatment_file_row(name: str, path: Path, *, required: bool) -> dict:
    """A stable row for a treatment file, including intentional absence of optional docs."""
    if path.exists() or path.is_symlink():
        digest, size = _file_digest(path)
        return {"name": name, "path": str(path), "present": True,
                "n_bytes": size, "sha256": digest}
    if required:
        raise RuntimeError(f"required treatment input is missing: {path}")
    return {"name": name, "path": str(path), "present": False,
            "n_bytes": 0, "sha256": None}


def _treatment_snapshot_record(ws: Path, run_dir: Path, bundle_dir: Path,
                               resolved_tools) -> dict:
    """Seal the exact prompt/docs and tool declarations that define one agent treatment.

    The bundle input payload already has its own immutable snapshot.  These files live outside that
    payload but still determine what the agent is told and which brokers start, so they need an equally
    explicit binding.  Optional declarations are represented by an absence row: creating one after setup
    is drift too (notably ``tools.txt``, whose later appearance changes tool resolution).
    """
    specs: list[tuple[str, Path, bool]] = [
        ("served/TASK.md", ws / "TASK.md", True),
        ("archived/TASK.md", run_dir / "TASK.md", True),
        ("archived_bundle/input_bundle_manifest.yaml",
         run_dir / "input_bundle_manifest.yaml", True),
    ]
    specs.extend((f"served/{name}", ws / name, False) for name in MERLIN_WS_DOCS)
    specs.extend((f"source_bundle/{name}", bundle_dir / name,
                  name == "input_bundle_manifest.yaml")
                 for name in _TREATMENT_BUNDLE_DECLARATIONS)
    rows = [_treatment_file_row(name, path, required=required)
            for name, path, required in specs]
    if len({row["name"] for row in rows}) != len(rows):
        raise RuntimeError("treatment input names are not unique")
    by_name = {row["name"]: row for row in rows}
    if (by_name["archived_bundle/input_bundle_manifest.yaml"]["sha256"]
            != by_name["source_bundle/input_bundle_manifest.yaml"]["sha256"]):
        raise RuntimeError(
            "source input bundle manifest changed while the run was being staged")

    tool_ids = list(resolved_tools)
    if any(not isinstance(tool, str) or not tool for tool in tool_ids):
        raise RuntimeError("resolved tool ids must be non-empty strings")
    if len(set(tool_ids)) != len(tool_ids):
        raise RuntimeError("resolved tool ids must be unique")
    aggregate = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["name"]):
        aggregate.update(row["name"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(b"1" if row["present"] else b"0")
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update((row["sha256"] or "-").encode("ascii"))
        aggregate.update(b"\n")
    for index, tool_id in enumerate(tool_ids):
        aggregate.update(f"tool\0{index}\0{tool_id}\n".encode("utf-8"))
    return {
        "version": 1,
        "content_sha256": aggregate.hexdigest(),
        "n_files_present": sum(bool(row["present"]) for row in rows),
        "files": rows,
        "resolved_tool_ids": tool_ids,
    }


def _verify_treatment_snapshot(expected: Mapping, ws: Path, run_dir: Path,
                               bundle_dir: Path, resolved_tools) -> dict:
    """Recompute a treatment binding and fail closed on any prompt/tool drift."""
    if not isinstance(expected, Mapping) or expected.get("version") != 1:
        raise RuntimeError("persisted treatment snapshot is missing or malformed")
    try:
        observed = _treatment_snapshot_record(ws, run_dir, bundle_dir, resolved_tools)
    except RuntimeError as exc:
        raise RuntimeError(f"experiment treatment drifted after setup: {exc}") from exc
    if dict(expected) != observed:
        expected_rows = {row.get("name"): row for row in expected.get("files", [])
                         if isinstance(row, Mapping)}
        observed_rows = {row["name"]: row for row in observed["files"]}
        changed = sorted(name for name in set(expected_rows) | set(observed_rows)
                         if expected_rows.get(name) != observed_rows.get(name))
        if expected.get("resolved_tool_ids") != observed["resolved_tool_ids"]:
            changed.append("resolved_tool_ids")
        raise RuntimeError(
            "experiment treatment drifted after setup: "
            + (", ".join(changed) if changed else "aggregate/metadata mismatch"))
    return observed


def _snapshot_path_for_live_path(snapshot_root: Path, live_path: Path, repo: Path) -> Path:
    """Map a declared live destination to its private bundle-snapshot location."""
    live_path = live_path.absolute()
    repo = repo.absolute()
    try:
        return snapshot_root / "repo" / live_path.relative_to(repo)
    except ValueError:
        return snapshot_root / "external" / Path(*live_path.parts[1:])


def _hidden_snapshot_dir(snapshot_root: Path, te, repo: Path) -> Path:
    """Resolve this target's hidden sibling inside the immutable bundle snapshot, never live."""
    corpus = Path(te.capsule_corpus)
    live_corpus = corpus if corpus.is_absolute() else repo / corpus
    return _snapshot_path_for_live_path(snapshot_root, live_corpus.parent / "hidden", repo)


def _subtree_snapshot_record(root: Path) -> dict:
    """Canonical file/count/byte binding for an operator-only frozen subtree."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"hidden snapshot subtree is missing or unsafe: {root}")
    rows: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"hidden snapshot subtree contains a symlink: {path}")
        if path.is_dir():
            continue
        digest, size = _file_digest(path)
        rows.append({"path": path.relative_to(root).as_posix(),
                     "n_bytes": size, "sha256": digest})
    if not rows:
        raise RuntimeError(f"hidden snapshot subtree is empty: {root}")
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(row["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(row["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    return {
        "version": 1,
        "path": str(root.resolve(strict=True)),
        "content_sha256": aggregate.hexdigest(),
        "n_files": len(rows),
        "n_bytes": sum(row["n_bytes"] for row in rows),
        "n_capsules": sum(Path(row["path"]).name == "capsule.yaml" for row in rows),
    }


def _verify_subtree_snapshot(expected: Mapping) -> tuple[Path, dict]:
    """Verify a persisted hidden-subtree record and return its already-frozen path."""
    if not isinstance(expected, Mapping) or expected.get("version") != 1:
        raise RuntimeError("persisted hidden subtree snapshot is missing or malformed")
    raw_path = expected.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("persisted hidden subtree snapshot path is missing")
    root = Path(raw_path)
    observed = _subtree_snapshot_record(root)
    if dict(expected) != observed:
        raise RuntimeError(
            f"hidden capsule snapshot drifted after setup at {root}: "
            f"expected {expected.get('content_sha256')}, observed {observed['content_sha256']}")
    return root, observed


def _verify_persisted_run_inputs(environment: Mapping, *, identity: Mapping,
                                 task_scope: Mapping, ws: Path, run_dir: Path,
                                 bundle_dir: Path, resolved_tools,
                                 expected_hidden_dir: Path | None = None) -> Path | None:
    """Common fail-closed gate for resume and the post-authoring official grade."""
    if not isinstance(environment, Mapping):
        raise RuntimeError("persisted environment record is missing or malformed")
    for field, observed in identity.items():
        if environment.get(field) != observed:
            raise RuntimeError(
                f"persisted environment {field} changed "
                f"({environment.get(field)!r} != {observed!r})")
    if environment.get("task_scope") != dict(task_scope):
        raise RuntimeError("descriptor-derived task scope drifted")
    _verify_treatment_snapshot(
        environment.get("treatment_snapshot"), ws, run_dir, bundle_dir, resolved_tools)
    if expected_hidden_dir is None:
        return None
    hidden_dir, _ = _verify_subtree_snapshot(environment.get("hidden_capsule_snapshot"))
    if hidden_dir.resolve(strict=True) != expected_hidden_dir.resolve(strict=True):
        raise RuntimeError(
            "hidden capsule record does not name this target's private bundle-snapshot subtree")
    return hidden_dir


def _te():
    """This experiment's target descriptor (honors MERLIN_TARGET_EXPERIMENT via C.EXP)."""
    from merlin.targetgen.target_experiment import load_target_experiment
    return load_target_experiment(C.EXP / "target_experiment.yaml")


def _pilot_subset():
    """The public-capsule set to grade against. An explicit override (``PILOT_SUBSET`` set by
    run_fullsuite) wins; otherwise it is DERIVED from the descriptor's capsule_corpus and materialized
    per-target (target-agnostic — gemmini→i8/L2, atlas→fp8/L3), cached for the run."""
    global _DERIVED_PILOT
    if PILOT_SUBSET is not None:
        return PILOT_SUBSET
    if _DERIVED_PILOT is None:
        from merlin.targetgen.contract.materialize import public_capsules_for
        _DERIVED_PILOT = public_capsules_for(_te())
    return _DERIVED_PILOT


def _declared_loop_tiers() -> set:
    """The oracle tiers THIS target's graded corpus declares in ``required_oracle_tiers``.

    Read from the corpus roots (not the materialized subset, which is derived FROM this) and handed to
    ``qa_loop_adapters`` so the per-round loop tier is one the capsules actually declared, rather than
    whichever endpoint tier happens to be fastest."""
    from merlin.targetgen.contract.materialize import declared_oracle_tiers
    try:
        return declared_oracle_tiers(*_te().graded_roots())
    except Exception:  # noqa: BLE001 — no resolvable corpus -> legacy fastest-tier behavior
        return set()


def _manifest():
    """This target's capability manifest (endpoint kind + sim tiers) — the second input render_prompt
    needs. Derived from the committed target_contract, so any target's runner works unchanged."""
    from merlin.targetgen.target_experiment import load_capability_manifest
    return load_capability_manifest(C.TARGET)


def answer_files() -> list[Path]:
    """Every answer-bearing value file the agent must not see (goldens + model weights)."""
    from merlin.targetgen.sandbox import golden_files, weight_files
    te = _te()
    return [*golden_files(te), *weight_files(te)]


def _denied_subpaths_under(rel: str, bundle: dict) -> list[str]:
    """Return bundle-denied paths that fall strictly INSIDE the allowed dir `rel`, as paths relative
    to `rel` (deny-wins). Lets the launcher copy an allowed tool dir MINUS its denied sub-paths
    (e.g. allow merlin/.../generate/ but exclude .../generate/runtime_adapter.py)."""
    base = rel.rstrip("/") + "/"
    out = []
    for d in bundle.get("denied", []):
        dp = d["path"].rstrip("/")
        if dp.startswith(base):
            out.append(dp[len(base):])
    return out


def assemble_copy_workspace(bundle: dict, ws: Path) -> dict:
    """sandbox=none isolation: build the workspace from COPIES of the allowed materials with every
    answer-bearing file dropped, and SYMLINK only answer-free large/toolchain paths. The agent's cwd
    is this workspace; it never receives goldens, hidden capsules, the reference, or prior backends.
    (This is the ``--sandbox none`` ESCAPE HATCH: FS isolation is by construction + a post-run transcript
    audit — detection, not prevention. bwrap is now the default and the preferred, enforced path; use
    ``none`` only for local dev where bwrap is unavailable.)

    Allowed Merlin authoring-tool dirs (under merlin/) are COPIED minus any deny-wins sub-path
    (e.g. runtime_adapter.py, xdsl_dialects/lowering/) so the workspace carries no in-workspace
    pointer to the oracle-callable helpers. (Under --sandbox none the real merlin package is still
    importable from disk; the strengthened transcript audit + the submission integrity scan are the
    load-bearing boundary — this copy-minus removes the convenience pointer and declares intent.)"""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "submission").mkdir(exist_ok=True)
    drop_dirs = {"hidden"}  # capsules/hidden = answers, never staged
    report = {"copied": [], "symlinked": [], "copied_minus": [], "answer_files_dropped": 0,
              "tool_subpaths_excluded": []}

    def _is_answer_name(n: str) -> bool:
        # Any golden (golden.yaml/json/mlir, any nesting) or example expected-output, matched by pattern
        # so nested / non-.yaml / non-_g0 answers do not escape the copy drop, plus externalized model
        # weights and their optional provenance manifest (parity with answer_surfaces.{golden,weight}_files).
        return (n.startswith("golden.") or n.startswith("expected_command_buffer")
                or n.endswith(".safetensors") or n.endswith(".safetensors.manifest.json"))

    def _ignore(dirpath, names):
        skip = set()
        for n in names:
            full = Path(dirpath) / n
            if n in drop_dirs or n == "__pycache__" or _is_answer_name(n):
                skip.add(n)
                if full.is_file():
                    report["answer_files_dropped"] += 1
        return skip

    for entry in bundle.get("allowed", []):
        src = C.REPO / entry["path"]
        if not src.exists():
            continue
        rel = entry["path"].rstrip("/")
        # Optional `as:` = explicit workspace destination (needed for out-of-repo absolute paths like the
        # vanilla gemmini repo, and to place single tools at a friendly top-level name). abc1 manifests
        # have no `as:` so their behavior is unchanged.
        dst_rel = (entry.get("as") or rel).lstrip("/")
        # The bench_contract tree is the ONLY answer-bearing input: copy it ONCE minus answers and
        # skip every redundant merlin/contract/... sub-entry (those symlink straight at real goldens).
        if rel.startswith("merlin/contract"):
            dst = ws / "merlin/contract"
            if rel == "merlin/contract" and not dst.exists():
                shutil.copytree(src, dst, ignore=_ignore, symlinks=False)
                report["copied"].append(rel)
            continue
        # Allowed Merlin tool DIRS: copy minus deny-wins sub-paths (oracle-callable helpers).
        if rel.startswith("merlin/") and src.is_dir():
            dst = ws / rel
            if dst.exists():
                continue
            excl = set(_denied_subpaths_under(rel, bundle))
            excl_names = {Path(e.rstrip("/")).name for e in excl}

            def _tool_ignore(dirpath, names, _excl=excl, _excl_names=excl_names, _rel=rel):
                skip = {"__pycache__"}
                relroot = Path(dirpath).resolve().relative_to((C.REPO / _rel).resolve())
                for n in names:
                    cand = (relroot / n).as_posix()
                    if n in _excl_names or cand in _excl or any(cand == e.rstrip("/") for e in _excl):
                        skip.add(n)
                        report["tool_subpaths_excluded"].append(f"{_rel}/{cand}")
                return skip

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, ignore=_tool_ignore, symlinks=False)
            report["copied_minus"].append(rel)
            continue
        # everything else (toolchain, headers, task, single tool files, the vanilla repo) -> symlink at
        # `as:` if given, else its natural rel. src may be an absolute out-of-repo path (C.REPO / abs = abs).
        dst = ws / dst_rel
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
            report["symlinked"].append(dst_rel)
        except FileExistsError:
            pass
    return report


# The answer/grader/oracle path-fragment tokens the transcript audit flags as READS are DERIVED from the
# SAME declared registry + descriptor as the filesystem mask (merlin.targetgen.sandbox.answer_surfaces),
# so there is one source of truth (no parallel hand-list to drift):
#   answer  = goldens + hidden set + the reference/simulator/backend oracle modules + the descriptor's
#             prior_backends + grader-private (reading any is an answer/route leak, BOTH arms).
#   grader  = the decoder/grader/golden-gen module stems (reverse-engineering the grader, BOTH arms).
#   oracle_subpath = the oracle-callable helper subpaths inside otherwise-allowed tool dirs (the
#             merlin-arm leak — a callable route to the reference/simulator oracle, BOTH arms).
# Matched against READ commands only — a mere mention (an integrity self-scan, or the repo path
# containing "merlin") is NOT a violation.
_AUDIT_TOKENS = __import__("merlin.targetgen.sandbox", fromlist=["audit_tokens"]).audit_tokens(_te())
_ANSWER_TOKENS = _AUDIT_TOKENS["answer"]
_GRADER_TOKENS = _AUDIT_TOKENS["grader"]
_ORACLE_SUBPATH_TOKENS = _AUDIT_TOKENS["oracle_subpath"]
# Merlin authoring tool dirs: ALLOWED for merlin_assisted, DENIED for raw_baseline -> flag reads of
# them for raw ONLY (for merlin they are legitimate authoring inputs). Includes the target-agnostic
# compiler-modification SPINE exposed to the assisted arm: the CCA (extract), cca_compare (diff),
# cca_contract (bijection), action_catalog (route + seam map), microkernel (resolver), and the generic
# derivation-driven backend (targetgen/rtl_backend) that derives the routes/levers from RTL discovery.
# None import the oracle.
_MERLIN_TOOL_TOKENS = ("targetgen/contract", "targetgen/synthesize", "targetgen/generate",
                       "xdsl_dialects",
                       "kernels/cca", "kernels/action_catalog", "kernels/microkernel",
                       "targetgen/rtl_backend")
# NOTE: the raw command-trace decoder (targetgen/rocc_decode) is NOT exposed — it is a grader internal
# (see ALLOWED_MERLIN_TOOLS.md FORBIDDEN). rtl_backend's lifter consumes a PRE-DECODED trace, so it does
# not import the decoder; the agent gets the where/how spine, not the raw trace-decoding.
# Oracle USE in agent-authored code / inline python: an actual `from merlin.runtime import ...` or a
# call to the oracle. Flagged in Bash `python -c` and in Write/Edit of .py files (both arms — neither
# may self-grade against the true oracle; the redacted QA verdict is the only allowed feedback).
_ORACLE_CODE_TOKENS = ("from merlin.runtime", "import merlin.runtime", "reference_outputs(",
                       "pipeline.execute(", "outputs_match(")
# Commands that READ file content (vs. ls/test/grep-of-own-sources).
_READ_RE = re.compile(r"\b(cat|head|tail|less|more|sed|awk|cp|xxd|od|open\(|read_text|yaml\.safe_load|"
                      r"json\.load|np\.load|loadtxt|grep[^|]*?)\b")
_PYC_RE = re.compile(r"\bpython3?\b[^\n;|&]*\s-c\b")


def _path_tokens(arm: str) -> tuple:
    """Per-arm set of path tokens whose READ is a violation. The merlin arm legitimately reads its
    authoring tools, so those are excluded for it (but the oracle sub-paths remain flagged)."""
    toks = _ANSWER_TOKENS + _GRADER_TOKENS + _ORACLE_SUBPATH_TOKENS
    if arm != "merlin_assisted":
        toks = toks + _MERLIN_TOOL_TOKENS
    return toks


def _is_answer_read(cmd: str, path_tokens: tuple) -> str | None:
    """Return the token if some shell SEGMENT both names a withheld path AND reads its content.
    Operating per-segment (split on ; && | newline) avoids conflating a self-scan or an `ls` boundary
    probe with a real read elsewhere in the same compound command."""
    for seg in re.split(r"[;&|\n]+", cmd):
        for tok in path_tokens:
            if tok not in seg:
                continue
            if _READ_RE.search(seg) or f"< {tok}" in seg or f"<{tok}" in seg:
                return tok
    return None


def _is_oracle_code(text: str) -> str | None:
    """Return the oracle-code token if `text` imports/calls the reference/simulator oracle."""
    for tok in _ORACLE_CODE_TOKENS:
        if tok in text:
            return tok
    return None


_EMPTY_RESULT_MARKERS = ("(bash completed with no output)", "(no output)", "(no content)")
# Read-FAILED signatures: the mask blocks an answer path either by binding it to /dev/null (an EMPTY
# result) OR by leaving it absent / mode-000 (the tool returns an ERROR). In BOTH cases NO answer content
# reached the agent, so an attempt whose result carries one of these is a benign blocked_probe, not a leak.
# (A golden.yaml's own content never IS one of these error strings, so this cannot mask a true leak.)
_BLOCKED_READ_MARKERS = (
    "no such file", "enoent", "does not exist", "cannot open", "not found",
    "permission denied", "eacces", "operation not permitted", "eperm",
    "file not found", "is a directory", "error: enoent", "no content",
)


def _result_text_by_id(tpath: Path) -> dict:
    """Map each tool_use_id -> the tool_result text the agent actually received. Lets the audit tell a
    BLOCKED probe of a masked answer file (empty result — the mask binds it to /dev/null) apart from a real
    LEAK (content returned, i.e. the mask failed). Only the claude-CLI transcript emits tool_result events;
    a transcript without them (the Converse driver) yields an empty map, so those reads stay conservatively
    flagged as violations."""
    out = {}
    for line in tpath.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") != "user":
            continue
        tur = e.get("tool_use_result")
        stdout = tur.get("stdout") if isinstance(tur, dict) else None
        for b in e.get("message", {}).get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id"):
                if stdout is not None:
                    txt = stdout
                else:
                    c = b.get("content")
                    txt = c if isinstance(c, str) else json.dumps(c)
                out[b["tool_use_id"]] = txt or ""
    return out


def _read_was_blocked(result_text) -> bool:
    """True iff a withheld-path read returned NOTHING — the mask (/dev/null bind) blocked it, so no answer
    content reached the agent. Non-empty content means the mask FAILED and bytes actually leaked. A missing
    result (None) is treated conservatively as NOT blocked (a real read)."""
    if result_text is None:
        return False
    s = result_text.strip().lower()
    if s == "" or s in _EMPTY_RESULT_MARKERS:
        return True
    # a read that FAILED (masked-absent / mode-000) returned an error, not answer bytes -> still blocked.
    return any(m in s for m in _BLOCKED_READ_MARKERS)


# Path-LISTING searches (grep -l / find / ls / locate) output FILENAMES, not file content — so they cannot
# leak answer bytes. They only matter if a listed path is itself an answer file.
_PATHLIST_RE = re.compile(r"\b(find|locate|which|whereis|ls)\b|\bgrep\b[^|;&]*\s-\w*l\b")


def _classify_bash_read(cmd: str, result_text, answer_tokens) -> str:
    """Classify a flagged Bash read: 'blocked_probe' (masked -> empty result), 'recon_probe' (a path-listing
    search that surfaced no answer path — e.g. the agent locating its OWN granted tool), or 'path_read' (a
    content read that returned data, or a search that actually located an answer file). Only 'path_read' (and
    oracle_use) break `clean`; the mask remains the real enforcement, this is defence-in-depth."""
    if _read_was_blocked(result_text):
        return "blocked_probe"
    if _PATHLIST_RE.search(cmd) and not any(t in (result_text or "") for t in answer_tokens):
        return "recon_probe"
    return "path_read"


_ADVISORY_KINDS = frozenset({"blocked_probe", "recon_probe"})


def audit_transcript(tpath: Path, arm: str = "raw_baseline") -> dict:
    """Flag genuine READS of withheld answer/grader/oracle paths AND oracle USE in agent-authored
    code (defence-in-depth beyond the masked workspace). Self-scans of the submission and bare path
    mentions are NOT flagged. Arm-aware: the merlin arm's allowed tools are not treated as cheats.

    A withheld-path read whose RESULT was empty (the mask returned /dev/null) is recorded as an advisory
    ``blocked_probe`` — it does NOT break ``clean`` — since no answer content reached the agent. Only a read
    that actually returned content (a mask breach) or oracle USE breaks ``clean``. This stops a thorough
    model that merely *probes* a masked golden (and gets nothing) from being falsely marked answer-unclean."""
    hits = []
    if not tpath.exists():
        return {"clean": True, "hits": [], "note": "no transcript"}
    path_tokens = _path_tokens(arm)
    results = _result_text_by_id(tpath)
    for line in tpath.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") != "assistant":
            continue
        for b in e.get("message", {}).get("content", []):
            if b.get("type") != "tool_use":
                continue
            inp = b.get("input", {})
            name = b.get("name")
            if name == "Read":
                fp = inp.get("file_path") or ""
                tok = next((t for t in path_tokens if t in fp), None)
                if tok:
                    kind = "blocked_probe" if _read_was_blocked(results.get(b.get("id"))) else "path_read"
                    hits.append({"tool": name, "kind": kind, "token": tok, "input": fp[:200]})
            elif name == "Bash":
                cmd = inp.get("command") or ""
                tok = _is_answer_read(cmd, path_tokens)
                if tok:
                    kind = _classify_bash_read(cmd, results.get(b.get("id")), _ANSWER_TOKENS)
                    hits.append({"tool": name, "kind": kind, "token": tok, "input": cmd[:200]})
                # inline python that imports/calls the oracle (e.g. `python -c "from merlin.runtime..."`)
                if _PYC_RE.search(cmd):
                    otok = _is_oracle_code(cmd)
                    if otok:
                        hits.append({"tool": name, "kind": "oracle_use", "token": otok,
                                     "input": cmd[:200]})
            elif name in ("Write", "Edit", "MultiEdit"):
                fp = inp.get("file_path") or ""
                if not fp.endswith(".py"):
                    continue  # only executable sources can self-grade; prose mentions are not a cheat
                blobs = [inp.get("content") or "", inp.get("new_string") or ""]
                blobs += [ed.get("new_string") or "" for ed in (inp.get("edits") or [])]
                otok = next((_is_oracle_code(t) for t in blobs if _is_oracle_code(t)), None)
                if otok:
                    hits.append({"tool": name, "kind": "oracle_use", "token": otok,
                                 "input": f"{fp}: {otok}"})
    # Advisory hits (a masked/blocked read, or a path-listing search that surfaced no answer path) do NOT
    # break `clean`; only an actual content leak or oracle USE does. Keep every hit in `hits` (still visible).
    violations = [h for h in hits if h.get("kind") not in _ADVISORY_KINDS]
    return {"clean": len(violations) == 0, "hits": hits,
            "blocked_probes": sum(1 for h in hits if h.get("kind") == "blocked_probe"),
            "recon_probes": sum(1 for h in hits if h.get("kind") == "recon_probe")}


def claude_runtime_binds() -> list[str]:
    """RO-bind the `claude` CLI runtime into the sandbox (delegates to the shared sandbox module)."""
    from merlin.targetgen.sandbox import bwrap as _BW
    return _BW.claude_runtime_binds()


def _granted_merlin_tools(arm: str) -> set:
    """The merlin tool paths THIS arm's bundle grants (its ``allowed_files.txt``) — the authoritative,
    per-arm set the enforced-workflow prompt block is derived from, so arm-4's RTL-facts grant (via the
    rtlchecks bundle RX.ARM_BUNDLE was swapped to) distinguishes it from arm-3. Empty when the file is
    absent (render_prompt then falls back to its coarse arm-string gate). Target-agnostic."""
    bdir = C.BUNDLES / RX.ARM_BUNDLE[arm]
    f = bdir / "allowed_files.txt"
    if not f.is_file():
        return set()
    return {ln.strip() for ln in f.read_text().splitlines()
            if ln.strip().startswith(("merlin/", "experiments/"))}


def _resolved_tools() -> tuple:
    """The ARM-GATED tools this run actually carries — its ablation cell.

    Resolution order, most authoritative first:

    1. the bundle's generated ``tools.txt`` — a cell is self-describing, so a run launched with
       ``--bundle`` on a pre-generated cell gets the right brokers without repeating the flags;
    2. the bundle-id STEM (longest match against the ladder's stems).

    Not from ``_ARM``: the CIRCT driver runs under ``_ARM == "merlin_assisted"`` with a swapped bundle,
    so an arm-name test cannot see it. Not from the manifest's ``arm:`` field either — the legacy
    ``merlin_assisted_rtlchecks_public_v0`` declares ``arm: merlin_assisted``, so that field is stale on
    exactly the bundles where being wrong costs the most.

    The ``--with-tool``/``--without-tool`` flags apply last, so an ad-hoc cell needs no regeneration.
    """
    bundle_id = RX.ARM_BUNDLE[_ARM]
    bdir = C.BUNDLES / bundle_id
    tools_f = bdir / "tools.txt"
    if tools_f.is_file():
        base = tuple(ln.strip() for ln in tools_f.read_text().splitlines() if ln.strip())
    else:
        base = _TR.ARM_TOOLS.get(_arm_from_bundle_id(bundle_id), ())
        _warn_if_grants_disagree(bdir, base)
    drop = set(_DROP_TOOLS)
    out = [t for t in base if t not in drop]
    out += [t for t in _ADD_TOOLS if t not in out]
    return tuple(out)


def _arm_from_bundle_id(bundle_id: str) -> str:
    """The ladder rung a bundle id names, by LONGEST matching stem.

    Longest wins because the stems nest: ``merlin_assisted_rtlchecks_*`` also starts with
    ``merlin_assisted_``, and picking the shorter one silently downgrades the CIRCT arm to the xDSL arm.
    """
    from merlin.targetgen.generate_bundles import _ARMS
    best, best_len = "", -1
    for arm, stem in _ARMS.items():
        if bundle_id.startswith(stem + "_") and len(stem) > best_len:
            best, best_len = arm, len(stem)
    if not best:
        raise KeyError(f"bundle id {bundle_id!r} matches no ladder rung (stems: {sorted(_ARMS.values())})")
    return best


_GRANT_WARNED: set = set()


def _warn_if_grants_disagree(bdir: Path, tools: tuple) -> None:
    """Surface a bundle whose NAME, BINDINGS and PROMPT disagree, instead of running on the mismatch.

    Three surfaces must agree about which tools an arm has, and they are written by different code:

      * the rung (this bundle's stem)      -> decides which brokers the driver starts;
      * ``input_bundle_manifest.yaml``     -> what the sandbox actually binds (fully regenerated);
      * ``allowed_files.txt``              -> what the PROMPT tells the agent it has, via
        :func:`_granted_merlin_tools`. It is written only-when-absent, so on a hand-authored bundle it
        goes stale while the manifest moves on.

    When the prompt and the bindings disagree the arm is either told about a tool it cannot open, or
    silently handed one it is never told about — and in both cases the treatment is not the one the
    run's label claims. That is not hypothetical: an arm once ran a whole campaign without executing a
    single one of its generators because its prompt never mentioned them.

    Advisory, not fatal, and reported once per bundle: several legacy bundles predate the generator.
    """
    if bdir.name in _GRANT_WARNED:
        return
    _GRANT_WARNED.add(bdir.name)
    man_f, files_f = bdir / "input_bundle_manifest.yaml", bdir / "allowed_files.txt"
    if not man_f.is_file():
        return
    man = yaml.safe_load(man_f.read_text()) or {}
    bound = {e.get("path") for e in (man.get("allowed") or []) if isinstance(e, dict)}
    for name in _TR.known_tools():
        t = _TR.spec(name)
        if not t.bundle_paths:
            continue                                  # brokered: staged, never bound -- nothing to compare
        if (set(t.bundle_paths) <= bound) != (name in tools):
            state = "binds" if set(t.bundle_paths) <= bound else "does not bind"
            print(f"  note: bundle {bdir.name} {state} {name!r} but its rung says otherwise; "
                  f"the rung decides which brokers start", file=sys.stderr)
    if files_f.is_file():
        told = {ln.strip() for ln in files_f.read_text().splitlines() if ln.strip()}
        unadvertised = {p for p in bound - told if p.startswith("merlin/python/")}
        if unadvertised:
            print(f"  WARNING: bundle {bdir.name} BINDS {len(unadvertised)} merlin tool path(s) that its "
                  f"allowed_files.txt does not list, so the prompt will not mention them: "
                  f"{sorted(unadvertised)[:4]}{' ...' if len(unadvertised) > 4 else ''}", file=sys.stderr)


def bwrap_cmd(inner: str, ws: Path, bundle: dict, extra_binds: list[str] | None = None) -> str:
    """bwrap argv (deny-by-default) + claude runtime binds + TOOLCHAIN binds (the legit build+sim tools,
    bound back over the /scratch* masks) + the DERIVED answer-mask pass + toolchain env. The mask set now
    comes from the shared descriptor-driven answer surface (goldens/hidden/prior/oracle/grader/memory) and
    is coverage-proven by test_sandbox_isolation; masking is bundle-independent (a bind that re-exposes an
    answer surface is re-masked here, not left to the bundle's denied list)."""
    import sandbox_toolchain as TC
    from merlin.targetgen.sandbox import bwrap as _BW
    from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces as _surfaces
    parts = RX.bwrap_argv(ws, bundle) + claude_runtime_binds() + TC.toolchain_binds()
    # These point at operator-only snapshot storage for the out-of-sandbox grader. The agent receives
    # the declared bytes through read-only mounts and never needs (or gets) their host storage path.
    parts += ["--unsetenv", _MODEL_HOST_SNAPSHOT_ROOT_ENV,
              "--unsetenv", _MODEL_HOST_SNAPSHOT_REQUIRED_ENV]
    # Per-driver runtime binds (e.g. the Codex CLI's package dir + an isolated
    # CODEX_HOME) go in BEFORE the mask pass, so a bind can never re-expose an
    # answer surface: masking is applied last and therefore wins.
    parts += list(extra_binds or [])
    # A trusted toolchain bind may overlap a declared arm grant (LLVM is both).
    # Reassert the frozen per-run bundle after every such bind so no live
    # worktree/tool input can override the snapshot, then apply answer masks.
    parts = _BW.reapply_bundle_snapshot(parts, ws, bundle, repo=C.REPO)
    parts = _BW.apply_answer_masks(parts, _surfaces(_te()))
    payload = f"{TC.sandbox_env(ws)} {inner}"
    # Single-quote the whole payload for the OUTER `bash -c`, escaping any embedded single quotes (the
    # POSIX '\'' idiom). ``inner`` may itself be shlex-quoted by the caller (the opencode driver quotes
    # its prompt arg, e.g. "…(if present)…"), so a naive f"…'{inner}'" would let those quotes close the
    # wrapper early and expose a `(` to the outer shell (opencode arm died rc=2 on exactly this). The
    # INNER bash still re-parses the payload as a script, so $(…)/\( \) in mask_selftest keep working.
    return " ".join(parts) + " bash -c '" + payload.replace("'", "'\\''") + "'"


def _corpus_probe_paths() -> tuple[Path, Path]:
    """A (golden.yaml, capsule.interface.mlir) pair from the DECLARED capsule corpus for the masking
    self-test — derived from the descriptor, not a hardcoded capsule id, so any target's corpus works.
    Masking hides every golden.yaml, so any one is a valid leak probe. Falls back to the isa corpus."""
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        corpus = load_target_experiment(C.EXP / "target_experiment.yaml").capsule_corpus
        g = next(iter(sorted(corpus.rglob("golden.yaml"))), None)
        s = next(iter(sorted(corpus.rglob("capsule.interface.mlir"))), None)
        if g:
            return g, (s or g.parent / "capsule.interface.mlir")
    except Exception:  # noqa: BLE001 — no/invalid descriptor ⇒ fall back, never break the self-test
        pass
    base = C.REPO / "merlin/contract" / "capsules" / "isa" / "A4_acc_scale_i8"
    return base / "golden.yaml", base / "capsule.interface.mlir"


def mask_selftest(ws: Path, bundle: dict, sandbox: str) -> dict:
    """Confirm the agent view withholds goldens, model weights, and hidden capsules."""
    if sandbox == "bwrap":
        # Search the agent's VIEW for ANY readable golden/expected/weight file —
        # independent of golden_files(), so a golden the mask FAILED to enumerate is STILL caught (the past
        # leak was exactly such a miss). Masked answers appear as empty /dev/null overlays (test -s false);
        # a corpus root that is tmpfs-hidden yields nothing (no false leak).
        roots = {str(C.REPO / "merlin/contract/capsules")}
        try:
            from merlin.targetgen.target_experiment import load_target_experiment
            cc = load_target_experiment(C.EXP / "target_experiment.yaml").capsule_corpus
            if cc:
                roots.add(str(cc))
        except Exception:  # noqa: BLE001 — no/invalid descriptor ⇒ probe the default tree only
            pass
        roots_sh = " ".join(sorted(roots))
        # Probe the hidden capsule DIRS too, not only golden/expected files. Masking
        # goldens file-by-file left the hidden test set enumerable — its capsule.yaml
        # inputs readable — while this self-test still reported OK, because it only
        # ever looked for answer VALUES. A readable held-out test set is its own
        # integrity problem, so it is a LEAK here.
        script = (f'for f in $(find {roots_sh} \\( -name "golden.*" '
                  f'-o -name "expected_command_buffer*" '
                  f'-o -name "expected_instruction_coverage.yaml" '
                  f'-o -name "*.safetensors" '
                  f'-o -name "*.safetensors.manifest.json" \\) '
                  f'2>/dev/null); do test -s "$f" && echo "LEAK:$f"; done; '
                  f'for f in $(find {roots_sh} -path "*/hidden/*" -name "capsule.yaml" 2>/dev/null); '
                  f'do test -s "$f" && echo "LEAK:$f"; done; echo DONE')
        out = subprocess.run(["bash", "-c", bwrap_cmd(script, ws, bundle)],
                             capture_output=True, text=True).stdout
        leaked = [ln[len("LEAK:"):] for ln in out.splitlines() if ln.startswith("LEAK:")]
        return {"pilot_golden_visible_to_agent": "LEAK" if leaked else "OK",
                "n_answer_files_masked": len(answer_files()),
                "leaked_answer_files": leaked[:10]}
    # sandbox == none: assert no answer is reachable. (1) the bench_contract COPY (small) holds no
    # golden/weights/hidden; (2) no workspace symlink resolves into the real merlin/contract/capsules tree
    # (which would re-expose goldens). Avoid walking the symlinked 307MB toolchain.
    bc = ws / "merlin/contract"
    goldens = [str(p) for p in bc.rglob("golden.yaml")] if bc.exists() else []
    weights = ([str(p) for p in bc.rglob("*.safetensors")]
               + [str(p) for p in bc.rglob("*.safetensors.manifest.json")]
               if bc.exists() else [])
    hidden = [str(p) for p in bc.rglob("hidden") if "capsules" in str(p)] if bc.exists() else []
    real_caps = str((C.REPO / "merlin/contract" / "capsules").resolve())
    bad_links = []
    for root, dirs, _files in os.walk(ws):  # followlinks=False: inspect, don't descend, symlinks
        for d in dirs:
            p = Path(root) / d
            if p.is_symlink() and str(p.resolve()).startswith(real_caps):
                bad_links.append(str(p))
    leak = bool(goldens or weights or hidden or bad_links)
    _g, spec = _corpus_probe_paths()               # a real corpus spec (derived), for the presence flag
    return {"pilot_golden_visible_to_agent": "LEAK" if leak else "OK",
            "goldens_in_workspace": goldens[:10], "weights_in_workspace": weights[:10],
            "hidden_in_workspace": hidden[:10],
            "symlinks_into_capsules": bad_links[:10], "spec_present": spec.exists()}


def _task_runtime_scope(te, sandbox: str) -> dict:
    """Facts the served task states, derived from this launch's real inputs."""
    from merlin.targetgen.capsule_common import discover_capsules

    contract = C.REPO / "merlin" / "contract"
    public = discover_capsules(te.graded_roots(), labels={"public", "dev"}, contract=contract)
    excluded = set(getattr(te, "graded_exclude", ()) or ())
    public = [cap for cap in public if cap.get("name") not in excluded]
    hidden = discover_capsules(te.hidden_roots(), labels={"hidden"}, contract=contract)
    if not public:
        raise RuntimeError(
            f"{te.target}: descriptor-derived public/dev task scope is empty; refusing to serve a "
            "vacuous completion target")
    return {
        "target": te.target,
        "required_public_dev_capsules": len(public),
        "held_out_capsules": len(hidden),
        "sandbox": sandbox,
        "scope_source": "TargetExperiment.graded_roots + labels public,dev + graded_exclude",
    }


def _task_runtime_scope_block(te, sandbox: str) -> str:
    """Agent-facing launch facts; authoritative over static bundle prose."""
    scope = _task_runtime_scope(te, sandbox)
    if sandbox == "bwrap":
        isolation = ("deny-by-default bwrap; allowed inputs are the frozen run snapshot and answer "
                     "surfaces are masked")
    else:
        isolation = ("unsandboxed diagnostic override; this launch cannot support a trusted isolation "
                     "claim")
    return (
        "\n\n## Runtime scope (generated for this launch; authoritative)\n"
        f"- Required public/dev capsules: **{scope['required_public_dev_capsules']}**, derived from "
        "the descriptor's graded roots, label filter, and exclusions. Completion is non-vacuous only "
        "when every required member passes.\n"
        f"- Held-out capsules: **{scope['held_out_capsules']}**, derived from the descriptor's hidden "
        "roots; their contents remain sealed.\n"
        f"- Active sandbox: **`{sandbox}`** ({isolation}).\n"
        "- If an older bundled document states a fixed capsule count or a different isolation mode, "
        "this launch-generated block wins.\n")


def _build_task(arm: str, ws: Path, run_dir: Path, sandbox: str = "bwrap") -> None:
    """Stage the workspace TASK.md. Both arms get the IDENTICAL graded contract (TASK_pilot.md). The
    merlin arm appends TASK_ADDENDUM.md (merlin-specific tool guidance + provenance ask) and stages
    the merlin-only docs into the workspace so the agent can read them. The graded pilot contract is
    never altered — the addendum only adds allowances, so grading stays apples-to-apples."""
    if _EXPERIMENT == "realistic":
        # whole-repo + self-check tool + self-paced READY marker; TASK_realistic is self-contained,
        # so skip the full experiment's grading-tier addendum appended below.
        ws_task = ws / "TASK.md"
        # A target that ships a hand-authored realistic task uses it (gemmini); a descriptor-only target
        # (e.g. atlas) has none, so fall back to the GENERATED target-agnostic prompt — exactly what the
        # 'full' branch below already does. render_prompt is the COMPLETE per-arm task (incl. the seam menu
        # for the assisted/CIRCT arms), so the bundle STARTER_PROMPT is not re-appended in that case.
        _task_md = C.EXP / "task" / "TASK_realistic.md"
        _generated_task = not _task_md.is_file()
        if _generated_task:
            from merlin.targetgen.generate_prompt import render_prompt
            body = render_prompt(_te(), _manifest(), "realistic", arm,
                                 granted_tools=_granted_merlin_tools(arm))
        else:
            body = _task_md.read_text()
        if os.environ.get("PILOT_LANG", "").strip().lower() == "cpp":
            _opt = f"{C.TARGET}-opt"          # the OOT MLIR tool name (derived from the active target)
            body += (
                "\n\n## Language mandate: C++ out-of-tree MLIR (REQUIRED for this run)\n"
                "- `manifest.yaml` MUST declare `language: cpp` and a `build` block that builds a real "
                f"out-of-tree MLIR tool `mlir_oot/build/bin/{_opt}` against the provided LLVM/MLIR-23 "
                "(`-DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR`); the runner builds it before grading.\n"
                "- Implement the 4 entrypoints as real MLIR passes in a C++ OOT package (input dialect + "
                f"{C.TARGET} target dialect + conversions + the `{_opt}` tool). A Python tool is NOT "
                "acceptable for this run. All integrity rules still apply.\n")
        bdir = C.BUNDLES / RX.ARM_BUNDLE[arm]
        # STARTER_PROMPT.md carries the ARM-SPECIFIC guidance (CIRCT generators for the +CIRCT arm,
        # verified-IR/kit for merlin, C++ method for baseline). It MUST be delivered or the arm's whole
        # approach is invisible to the agent (abc8: the CIRCT arm never ran a single generator because this
        # was missing). Append it to TASK.md (the agent always reads TASK.md) for every arm.
        # When the task body was GENERATED (render_prompt above), it already carries the arm's full
        # approach — appending the bundle STARTER_PROMPT (also render_prompt) would just duplicate it.
        starter = "" if _generated_task else (
            (bdir / "STARTER_PROMPT.md").read_text() if (bdir / "STARTER_PROMPT.md").exists() else "")
        if starter:
            body += "\n\n---\n\n# Starter plan / approach for THIS arm (read this)\n\n" + starter
        body += _task_runtime_scope_block(_te(), sandbox)
        if arm == "merlin_assisted":
            add = (bdir / "TASK_ADDENDUM.md").read_text() if (bdir / "TASK_ADDENDUM.md").exists() else ""
            ws_task.write_text(body + ("\n\n---\n\n" + add if add else ""))
            for doc in MERLIN_WS_DOCS:
                src = bdir / doc
                if src.exists():
                    shutil.copy(src, ws / doc)
        else:
            ws_task.write_text(body)
        shutil.copy(ws_task, run_dir / "TASK.md")
        return
    # The graded contract is now the GENERATED (target-agnostic) prompt: ONE shared skeleton + slots
    # DERIVED from {descriptor + RTL fact bundle + endpoint}, so a per-target committed TASK_full.md is no
    # longer the source of truth (its content is covered by the generated body — proven by the dry-run
    # diff). The runtime-only operational blocks below (language mandate + descriptor-derived scope and
    # grading tiers) are still appended: they carry run-specific facts the target-agnostic template
    # deliberately omits.
    from merlin.targetgen.generate_prompt import render_prompt
    pilot = render_prompt(_te(), _manifest(), _EXPERIMENT, arm, granted_tools=_granted_merlin_tools(arm))
    # Language mandate (env PILOT_LANG=cpp|python). The C++ arms (baseline/cpp_merlininfra) are forced to
    # the status-quo C++ OOT MLIR; the merlin arms (arm-3 merlin_assisted + arm-4 merlin_assisted_rtlchecks,
    # both carry "merlin_assisted") DEFAULT to the xDSL/Python path — the whole point of those arms is to
    # build the dialect with the granted xDSL kit, NOT a hand C++/TableGen tool. Arm-driven so it holds
    # however the run is launched (a merlin arm never "chooses C++"). tool stem is target-agnostic.
    _stem = f"{_te().target}-opt"
    _lang = os.environ.get("PILOT_LANG", "").strip().lower() or ("python" if "merlin_assisted" in arm else "")
    if _lang == "cpp":
        pilot += (
            "\n\n## Language mandate: C++ out-of-tree MLIR (REQUIRED for this run)\n"
            "- `manifest.yaml` MUST declare `language: cpp` and a `build` block "
            "(`configure`/`command`/`tool_output`) that builds a real out-of-tree MLIR tool "
            f"`mlir_oot/build/bin/{_stem}` against the provided LLVM/MLIR-23 "
            "(`-DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR`); the runner builds it before grading.\n"
            "- Implement the 4 entrypoints as real MLIR passes in a C++ OOT package "
            f"(input dialect + target dialect + conversions + the `{_stem}` tool). "
            "A Python tool is NOT acceptable for this run.\n"
            "- Every integrity rule above still applies unchanged (no C compute kernels, no copied "
            "reference kernels, no high-level device libs, no hardcoded outputs). `integrity_exempt: false`.\n")
    elif _lang == "python":
        pilot += (
            "\n\n## Language mandate: xDSL / Python (REQUIRED for this arm)\n"
            "- Build the dialect + the 4 entrypoints with the granted **xDSL kit** "
            "(`oot_starterkit/` — dialect.py / transforms.py / verify.py — + `xdsl_dialects/`): define the "
            "target dialect as xDSL ops with verifiers, and the interface->target lowering as xDSL rewrite "
            "passes. This is the approach this arm exists to exercise.\n"
            f"- `manifest.yaml` MUST declare `language: python`; the tool is an executable Python `{_stem}` "
            "exposing the 4 entrypoints. Do NOT author a C++/TableGen tool or a `build` block that compiles "
            "one (no cmake/`mlir-tblgen`/`*-opt` C++ binary) — a hand C++ backend is NOT acceptable for this "
            "arm. All integrity rules still apply (`integrity_exempt: false`).\n")
    # Tier wording is TARGET-AGNOSTIC: name the target's own oracle tiers from the manifest, not the
    # gemmini spike/verilator literals (atlas's loop tier is the arc program-oracle, its checkpoint the
    # cycle-accurate RTL cosim/Verilator). `_loop`/`_ckpt` are the tier keys the runner resolves.
    from merlin.targetgen import capsule_runner as CR
    _loop = min(CR.qa_loop_adapters(_te().target, _te().sim_via,
                                    declared_tiers=_declared_loop_tiers()) or {"L3": 1})
    _ckpt = max(CR.qa_checkpoint_adapters(_te().target, _te().sim_via) or {"L3": 1})
    pilot += _task_runtime_scope_block(_te(), sandbox)
    pilot += (
        "\n\n## Grading tiers (READ THIS)\n"
        f"- Each round the QA gate runs **L0+L1+trace + your fast RTL oracle tier ({_loop})** and returns "
        "a redacted verdict (pass/fail + failure plane, never goldens). Use it to fix failures.\n"
        f"- When your public capsules pass the loop tier, the harness runs the **cycle-accurate RTL "
        f"checkpoint ({_ckpt})**. If any capsule fails only there, you get **up to {VERILATOR_ATTEMPTS} "
        "checkpoint attempts** (a fix round between each) to make it cycle-accurate-correct. Treat RTL "
        "checkpoint failures as real bugs to fix, not noise.\n")
    ws_task = ws / "TASK.md"
    if arm == "merlin_assisted":
        bdir = C.BUNDLES / RX.ARM_BUNDLE[arm]
        add = (bdir / "TASK_ADDENDUM.md").read_text() if (bdir / "TASK_ADDENDUM.md").exists() else ""
        ws_task.write_text(pilot + "\n\n---\n\n" + add)
        for doc in MERLIN_WS_DOCS:
            src = bdir / doc
            if src.exists():
                shutil.copy(src, ws / doc)
    else:
        ws_task.write_text(pilot)
    shutil.copy(ws_task, run_dir / "TASK.md")  # archive the exact task served, for the record


def _trust_cli_cost(model: str) -> bool:
    """False when the round reached its model through the bridge.

    A CLI prices what it believes it ran. Pointed at the proxy it bills a foreign model at its own
    catalogue's rates, which is both a wrong artifact and an active hazard: the inflated figure counts
    against --max-spend-usd and kills a run that has spent almost nothing.
    """
    try:
        import agent_bridge as _BR
        return not _BR.bridged_name(model, _driver_for(model))
    except Exception:
        return True


def _driver_for(model: str) -> str:
    """Resolve which agent driver handles ``model``. ``--driver`` (the ``_DRIVER`` global) is authoritative
    when set to a concrete driver; ``auto`` (the default, behavior-preserving) routes by model id — the
    Bedrock Converse loop for a non-Anthropic id, else the ``claude`` CLI."""
    import bedrock_agent as _BA
    if _DRIVER and _DRIVER != "auto":
        return _DRIVER
    return "converse" if _BA.is_converse_model(model) else "claudecode"


_DRIVER_MODULES = {"codex": "codex_agent", "opencode": "opencode_agent",
                   "converse": "bedrock_agent"}   # 'claudecode' drives the claude CLI directly


def _billing_mode(model: str) -> str:
    """How the run that ``model`` produces is BILLED — asked of the driver, never inferred from the
    model id. A driver module declares ``BILLING_MODE``; anything that does not is metered (an API
    key charged per token). This is what keeps a subscription-seat run from reporting a dollar spend:
    a Codex round once landed ``estimated_cost_usd: 17.2103`` in the ledger, priced at opus rates for
    a model no price table knows, on an account that is not billed per token at all."""
    # A driver's declared BILLING_MODE describes ITS OWN account. A bridged round does not use that
    # account: it is our Bedrock key, charged per token, whichever CLI happens to be driving it. The
    # codex driver declares subscription_notional because it normally runs on a ChatGPT seat, so a
    # bridged codex round was booking real Bedrock spend as notional -- money that is never counted
    # against --max-spend-usd or the campaign budget. This is the mirror of trusting a CLI's own
    # total_cost_usd for a model it does not bill, and it under-reports instead of over-reporting.
    drv = _driver_for(model)
    try:
        import agent_bridge as _BR
        if _BR.bridged_name(model, drv):
            return ET.METERED
    except Exception:
        pass
    mod_name = _DRIVER_MODULES.get(drv)
    if not mod_name:
        return ET.METERED
    try:
        import importlib
        return getattr(importlib.import_module(mod_name), "BILLING_MODE", ET.METERED)
    except ImportError:
        return ET.METERED


def launch_agent(ws: Path, run_dir: Path, model: str, effort: str, sandbox: str,
                 bundle: dict, rnd: int, timeout: int, arm: str = "raw_baseline") -> tuple[int, Path]:
    # TASK.md must live INSIDE the bound workspace: run_dir is under runs/ which bwrap tmpfs-masks,
    # so a stdin redirect from run_dir/TASK.md is invisible inside the sandbox (empty stdin).
    ws_task = ws / "TASK.md"
    # Setup stages and seals the task BEFORE writing environment.yaml.  Rebuilding here used to leave
    # environment provenance unable to say what prompt was served, and a resumed round zero could silently
    # pick up changed bundle prose.  Missing after setup is corruption, not an invitation to regenerate.
    if not ws_task.is_file():
        raise RuntimeError(f"sealed task is missing before agent launch: {ws_task}")
    # Under bwrap the oracle is masked, so the agent's self-check (agent_selfcheck.py) can't grade in-box.
    # Start the driver-side BROKER (oracle available, outside the sandbox) + stage the in-box shim, so the
    # agent gets a REDACTED on-demand self-check (numeric diff, no goldens) without the oracle ever entering
    # its sandbox. Started BEFORE the backend split so BOTH the claude CLI and the Bedrock Converse agent
    # get the identical mid-round feedback loop.
    broker = _start_selfcheck_broker(ws) if sandbox == "bwrap" else None
    try:
        # Route to the selected driver (explicit --driver, or auto-by-model-id). A non-Anthropic model can't
        # drive the claude CLI (Anthropic API only) → the Bedrock Converse backend runs the same masked-
        # sandbox agentic loop (incl. the self_check tool wired to the shim above) + a compatible transcript;
        # 'opencode' drives the provider-agnostic OpenCode CLI; 'claudecode'/Anthropic uses the claude CLI.
        drv = _driver_for(model)
        # A (model, harness) pairing that needs the bridge needs the proxy running. Started here rather
        # than by the launcher so EVERY entry point (launch_ab_batch, chia_ab_batch, watchdog resume, a
        # bare run) gets it, and idempotently so concurrent arms of one campaign share one instance.
        import agent_bridge as _BR
        if _BR.bridged_name(model, drv):
            _pi = _BR.start_proxy(run_dir / "logs" / "litellm_proxy.log")
            (run_dir / "bridge.json").write_text(json.dumps(
                {**_BR.record(model, harness=drv), "proxy_started": _pi}, indent=1))
        if drv == "converse":
            import bedrock_agent as _BA
            return _BA.run_round(ws, run_dir, model, bundle, _te(), sandbox, rnd, timeout,
                                 subagent_model=_SUBAGENT_MODEL, background_model=_BACKGROUND_MODEL)
        if drv == "opencode":
            try:
                import opencode_agent as _OA
            except ImportError as e:  # Phase 3 not landed yet
                raise SystemExit(f"--driver opencode is not available yet: {e}")
            # effort threads here for the same reason it does for codex below — see that comment.
            return _OA.run_round(ws, run_dir, model, bundle, _te(), sandbox, rnd, timeout,
                                 subagent_model=_SUBAGENT_MODEL, background_model=_BACKGROUND_MODEL,
                                 effort=effort)
        if drv == "codex":
            try:
                import codex_agent as _CA
            except ImportError as e:
                raise SystemExit(f"--driver codex is not available: {e}")
            # effort is threaded through: codex takes it as a config override, and an
            # arm that silently ran at a different reasoning effort is a different arm.
            return _CA.run_round(ws, run_dir, model, bundle, _te(), sandbox, rnd, timeout,
                                 subagent_model=_SUBAGENT_MODEL, background_model=_BACKGROUND_MODEL,
                                 effort=effort)
        # claudecode. The claude CLI speaks the Anthropic Messages API, so a NON-Anthropic model reaches
        # it only through the LiteLLM bridge (ANTHROPIC_BASE_URL -> our proxy -> Bedrock). This is what
        # makes the harness the experimental variable instead of a fixed property of the model: nemotron
        # and glm5 can now be driven by the same harness that drives opus. An Anthropic model returns an
        # empty env here and takes its existing native/Bedrock path unchanged.
        import agent_bridge as _BR
        _bridge_env = _BR.claude_env(model)
        _cli_model = _BR.claude_model_name(model)
        inner = (f'claude --print --model {_cli_model} --effort {effort} '
                 f'--permission-mode bypassPermissions --add-dir {ws} '
                 f'--output-format stream-json --verbose < {ws_task}')
        if _bridge_env:
            # Exported INSIDE the sandbox: bwrap keeps the network namespace (loopback reaches the
            # proxy) but not the environment.
            _exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in _bridge_env.items())
            inner = f"env {_exports} {inner}"
        cmd = bwrap_cmd(inner, ws, bundle) if sandbox == "bwrap" else inner
        tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
        tpath.parent.mkdir(parents=True, exist_ok=True)
        epath = run_dir / "rounds" / f"round_{rnd:02d}.stderr.log"
        with open(tpath, "w") as tf, open(epath, "w") as ef:
            # start_new_session + killpg on timeout: the command is `bash -c '<bwrap ... claude ...>'`, so a
            # plain subprocess timeout SIGKILLs only the outer bash and leaves the bwrap->claude tree alive
            # (bwrap --die-with-parent does not reliably cascade). Kill the whole process group instead.
            proc = subprocess.Popen(["bash", "-c", cmd], cwd=str(ws), stdout=tf, stderr=ef,
                                    start_new_session=True)
            try:
                proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait()
                raise
    finally:
        _stop_selfcheck_broker(ws, broker)
    return proc.returncode, tpath


def _stage_shim(ws: Path, src_name: str, dst_name: str) -> None:
    """Copy an in-box shim into the ws, replacing any bound symlink (copy-onto-symlink would clobber
    the real script — learned the hard way)."""
    dst = ws / dst_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy(SCRIPTS / src_name, dst)


def _start_selfcheck_broker(ws: Path):
    """Stage the in-box shims (sync self-check + async simjob) and launch BOTH driver-side brokers, which
    run the redacted grade OUTSIDE the sandbox so the oracle never enters the box. Returns a list of
    Popens. The async simjob broker lets the agent run slow verilator per-capsule without blocking a turn."""
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    (ch / "STOP").unlink(missing_ok=True)
    _stage_shim(ws, "selfcheck_shim.py", "agent_selfcheck.py")   # sync self-check (spike, fast)
    _stage_shim(ws, "simjob_shim.py", "simjob.py")               # async oracle (spike/verilator/vcs)
    broker_specs = [("selfcheck_broker.py", "broker.log"), ("simjob_broker.py", "simjob_broker.log")]
    # Brokered TOOLS (the ISA assembler/disassembler/linter; the two mandated CCA calls) are part of the
    # arm's TREATMENT, so which ones start is read from the bundle's resolved tool set rather than from
    # the arm's name. That distinction matters twice over: the CIRCT arm runs under _ARM ==
    # "merlin_assisted" with a swapped bundle, so a name test cannot see it; and an ablation cell differs
    # from its rung only in that tool set. Each broker gets its own channel dir + STOP, like the
    # self-check. All of them are oracle-free and read no golden.
    for bs in _TR.brokers_for(_resolved_tools()):
        ch_dir = ws / bs.channel
        ch_dir.mkdir(parents=True, exist_ok=True)
        (ch_dir / "STOP").unlink(missing_ok=True)
        for shim_src, staged_as in bs.shims:
            _stage_shim(ws, shim_src, staged_as)
        broker_specs.append((bs.module, bs.log))
    brokers = []
    for name, log in broker_specs:
        brokers.append(subprocess.Popen(
            [sys.executable, str(SCRIPTS / name), "--ws", str(ws)],
            stdout=open(ch / log, "w"), stderr=subprocess.STDOUT))
    return brokers


def _stop_selfcheck_broker(ws: Path, brokers) -> None:
    if not brokers:
        return
    (ws / ".qa_channel" / "STOP").write_text("stop")
    isa_ch = ws / ".isa_channel"
    if isa_ch.is_dir():                                  # assisted-arm ISA-tools broker (if it was started)
        (isa_ch / "STOP").write_text("stop")
    cca_ch = ws / ".cca_channel"
    if cca_ch.is_dir():                                  # assisted-arm CCA-contract broker (if it was started)
        (cca_ch / "STOP").write_text("stop")
    for b in (brokers if isinstance(brokers, list) else [brokers]):
        try:
            b.wait(timeout=15)
        except Exception:
            b.kill()


def _language_ok(submission_dir: Path) -> tuple[bool, str]:
    """Enforce the current arm's language mandate on the emitted submission (merlin arms => xDSL/Python,
    not a hand C++ backend). Delegates to tooling_readiness.submission_language_ok; degrades to OK if the
    check is unavailable so it never spuriously blocks a run."""
    try:
        import tooling_readiness
        return tooling_readiness.submission_language_ok(submission_dir, _ARM)
    except Exception:  # noqa: BLE001 — never let the compliance check itself break grading
        return True, "language check unavailable"


def qa_grade(ws: Path, run_dir: Path, rnd: int, no_oracle: bool, timeout: int) -> dict:
    """Copy the agent's submission to an operator-only scratch, grade it, return + persist the
    redacted verdict (into ws/qa/verdict.json for the next round, and archived per round)."""
    cand = run_dir / "_qa_work" / f"cand_{rnd:02d}" / "submission"
    if cand.exists():
        shutil.rmtree(cand.parent)
    _lok, _lwhy = _language_ok(ws / "submission")
    if not (ws / "submission" / "manifest.yaml").exists():
        verdict = {"all_pass": False, "n_passed": 0, "n_capsules": 4,
                   "package_failure": {"plane": "schema", "detail": "no submission/manifest.yaml"},
                   "per_capsule": [], "note": "write submission/manifest.yaml first."}
    elif not _lok:
        # ENFORCE the arm language mandate: a merlin arm must build with the xDSL kit, not a C++ tool.
        # A non-compliant submission gets a failing verdict with the fix reason (never silently graded).
        verdict = {"all_pass": False, "n_passed": 0, "n_capsules": 4,
                   "package_failure": {"plane": "language", "detail": _lwhy},
                   "per_capsule": [], "note": f"language mandate: {_lwhy}. Use the xDSL/Python kit "
                   "(oot_starterkit + xdsl_dialects), declare language: python, no C++/cmake build."}
    else:
        shutil.copytree(ws / "submission", cand,
                        ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
        _strip_build_state(cand)   # clean, relocatable build per grade (abc9 L3-build bug)
        out = run_dir / "qa_history" / f"verdict_round_{rnd:02d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        argv = ["--submission", str(cand), "--capsules-root", str(_pilot_subset()),
                "--out", str(out), "--runs-root", str(run_dir / "_qa_work" / f"runs_{rnd:02d}"),
                "--timeout", str(timeout)]
        if no_oracle:
            argv.append("--no-oracle")
        import qa_check
        verdict = qa_check.run(str(cand), str(_pilot_subset()),
                               run_dir / "_qa_work" / f"runs_{rnd:02d}",
                               {"public", "dev"}, no_oracle, timeout)
        out.write_text(json.dumps(verdict, indent=2))
        _write_stage_ledger(run_dir, rnd, cand, run_dir / "_qa_work" / f"runs_{rnd:02d}", verdict)
        _attach_shape_generalization(verdict, cand, run_dir, rnd, timeout=timeout)
    # PROMOTE off the round grade too. Promotion is hooked into both BROKERS, but a broker only sees a
    # verdict the agent ASKED for -- and a converged agent stops asking. Measured on the run that
    # motivated this: 24 self-checks in round 0, then ZERO in rounds 1 and 2 once it reached the corpus
    # ceiling, so the only verdict produced in those rounds was this one and promotion had nothing to fire
    # on. Three paths produce a verdict; all three must consider promotion, or the deeper tier is only
    # ever reached while the agent is still struggling -- which is exactly backwards, since a converged
    # submission is the one worth certifying.
    # Runs AFTER _attach_shape_generalization so promotion considers the completed verdict.
    try:
        import sys as _sys
        from tier_promote import promote as _promote, resolve_tiers as _resolve
        _loop, _cert, _cover = _resolve(ws)
        if _loop and _cert and isinstance(verdict, dict) and verdict.get("per_capsule"):
            _p = _promote(ws, ws / ".qa_channel", verdict, _loop, _cert, _cover, _sys.stderr)
            if _p:
                print(f"  [promote] round grade -> {_cert}: {_p}", flush=True)
    except Exception as _pe:  # noqa: BLE001 -- promotion is an optimisation, never a gate
        print(f"  [promote] skipped: {type(_pe).__name__}: {_pe}", flush=True)

    # hand the redacted verdict to the agent for the next round
    qa_dir = ws / "qa"
    qa_dir.mkdir(exist_ok=True)
    (qa_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    return verdict


# The public suite is the set of shapes an agent can see, so a backend that keys on those shapes passes
# it by construction and the loop has nothing left to say. Measured: a submission converged at 14/26 with
# every failure a shape it had never implemented, its self-check clean on everything it HAD, and no round
# feedback able to point at the gap -- the first instrument to see it was a post-freeze holdout.
#
# These probes are derived from the target's own declared capability closure and its DERIVED tile edge
# (never the corpus), so they leak nothing an agent could not compute itself, and they run at the cheap
# loop tier. Restricted to the multi-tile corners: the question is "does this backend generalize past ONE
# tile, and along WHICH axis" -- per-axis, because a backend that loops over K and N but not M passes two
# of the three, and only naming the axis makes the result actionable.
def _attach_shape_generalization(verdict: dict, cand, run_dir, rnd: int, *, timeout: int) -> None:
    """Probe whether this round's candidate LOWERS shapes past a single tile, and fold it into the gate.

    Structural, not numerical: it runs only the emit half of the contract and compares the size of the
    emitted artifact across shapes, so it costs no oracle, needs no golden, and works on an operand
    format that has no CPU reference. See :mod:`merlin.targetgen.lowering_coverage` for the invariant
    ("a program for a bigger problem cannot be smaller") and why it is per-axis.

    Failure to run is RECORDED, never treated as clean -- a probe that did not run reading as a pass is
    the same class of bug as an unavailable oracle scoring as one.
    """
    out = run_dir / "qa_history" / f"shape_coverage_round_{rnd:02d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from merlin.targetgen import lowering_coverage as LC
        cov = LC.sweep(cand, target=C.TARGET, contract=str(C.REPO / "merlin/contract"),
                       timeout=min(timeout, 300))
        out.write_text(json.dumps(cov, indent=2))
    except Exception as e:  # noqa: BLE001 -- record it; never let it read as clean
        verdict["shape_coverage"] = {
            "ran": False, "error": f"{type(e).__name__}: {e}",
            "note": "the shape-coverage probes did NOT run this round; this is NOT a pass."}
        return

    verdict["shape_coverage"] = {
        "ran": True,
        "tile_edge": cov.get("tile_edge"),
        "baseline_tile_lowered": cov.get("baseline_tile_lowered"),
        "per_corner": {c["corner"]: c["outcome"] for c in cov.get("corners", [])},
        "emitted_work": cov.get("emitted_work"),
        "multi_tile_axes_uncovered": cov.get("multi_tile_axes_uncovered") or [],
        "all_covered": bool(cov.get("all_covered")),
        "unmeasured": cov.get("unmeasured"),
        "detail": {c["corner"]: c.get("detail") for c in cov.get("corners", []) if c.get("detail")},
        "note": ("DERIVED shape probes, not corpus capsules: the SAME contraction at one tile and at two "
                 "tiles in each of M, K and N, at this target's derived tile edge. `emitted_work` is the "
                 "size of the program you emitted for each -- a bigger problem cannot need a SMALLER "
                 "program, so a corner marked `collapsed` is a shape you silently refused. "
                 "`multi_tile_axes_uncovered` names the axis your lowering does not loop over: fix the "
                 "loop, not the arithmetic. If you genuinely cannot lower a shape, DECLARE it "
                 "(`declined` on the command buffer) instead of emitting a terminator."),
    }
    # THE GATE. Passing every public capsule while lowering only the shapes they happen to use is exactly
    # the state that shipped at 14/26, so the loop must not call that converged.
    if verdict.get("all_pass") and not verdict["shape_coverage"]["all_covered"]:
        verdict["all_pass"] = False
        verdict["not_converged_reason"] = (
            "every public capsule passes, but the derived shape probes show the backend does not lower "
            + (f"past one tile on axis/axes {verdict['shape_coverage']['multi_tile_axes_uncovered']}"
               if verdict["shape_coverage"]["multi_tile_axes_uncovered"]
               else "the baseline tile itself (nothing about shape can be concluded yet)"))


def _write_stage_ledger(run_dir, rnd: int, cand, runs_root, verdict) -> None:
    """Record per-round artifact fingerprints beside the verdict — OUT OF BAND.

    Answers "did the agent's edit reach what was graded?" from bytes, so a plateau is one line instead of
    an investigation. Written into the run dir (a sandbox-DENIED path) and never into the agent's
    ``qa/verdict.json``: telling the agent its edit was inert is feedback, and feedback defines an arm.

    Never raises. This is diagnostics — it must not be able to fail a round that otherwise graded fine.
    """
    try:
        from merlin.targetgen import stage_ledger as SL

        led_dir = run_dir / "rounds"
        led_dir.mkdir(parents=True, exist_ok=True)
        prev_p = led_dir / f"round_{rnd - 1:02d}.stage_ledger.json"
        prev = json.loads(prev_p.read_text()) if rnd and prev_p.is_file() else None

        # Per-capsule emit dirs, found by SHAPE at any depth: a dir named for the emit output whose
        # PARENT is a graded capsule dir (it holds the capsule's own result/manifest). Depth and the
        # intervening dir names are never assumed -- no target, no suite, and no generated-root literal --
        # so a target that lays its run tree out differently is still discovered, and one that lays it out
        # unrecognizably simply yields none rather than a wrong answer.
        marks = ("capsule_result.json", "run_manifest.yaml")
        roots = {d.parent.name: d for d in Path(runs_root).rglob("generated")
                 if d.is_dir() and any((d.parent / m).is_file() for m in marks)}
        led = SL.build(submission_dir=cand, emitted_roots=roots, previous=prev)
        led["round"] = rnd
        led["failing_and_frozen"] = SL.failing_and_frozen(led, verdict)
        (led_dir / f"round_{rnd:02d}.stage_ledger.json").write_text(json.dumps(led, indent=2))
        print(f"  {SL.summarize(led)}", flush=True)
        if led["failing_and_frozen"]:
            print(f"  stage_ledger: FAILING AND FROZEN ({len(led['failing_and_frozen'])}): "
                  f"{', '.join(led['failing_and_frozen'][:10])}"
                  f"{' ...' if len(led['failing_and_frozen']) > 10 else ''}", flush=True)
    except Exception as e:                                    # noqa: BLE001 - diagnostics never gate
        print(f"  stage_ledger: unavailable ({type(e).__name__}: {e})", flush=True)


_PASS_LINE = ("Baseline pilot passes all required public/dev pilot capsules and is ready for "
              "hidden pilot grading.")
_STATUS_LINES = (
    _PASS_LINE,
    "Baseline pilot does not yet pass all required public/dev pilot capsules; remaining failures "
    "are listed by capsule and failure plane.",
    "Baseline pilot is not comparable because it violates the compiler/runtime/integrity boundary.")


def _stamp_report_status(report: Path, pass_line: str) -> bool:
    """Last-resort guarantee: ensure REPORT.md's final status line states the VERIFIED result. The
    multi-round relaunch grades AFTER the agent exits, so the converging round's agent never saw a
    passing verdict and may leave a stale 'not yet passing' line. Returns True if it had to rewrite."""
    if not report.exists():
        report.write_text(f"# REPORT\n\n## Final status line\n{pass_line}\n")
        return True
    txt = report.read_text()
    if pass_line in txt:
        return False
    new = txt
    for sl in _STATUS_LINES:
        if sl != pass_line and sl in new:
            new = new.replace(sl, pass_line)
    if new == txt:  # no known status line present -> append one
        new = txt.rstrip() + f"\n\n## Final status line (verified)\n{pass_line}\n"
    report.write_text(new)
    return True


def finalize_report(ws: Path, run_dir: Path, model: str, effort: str, sandbox: str, bundle: dict,
                    arm: str, verdict: dict, timeout: int) -> dict:
    """After convergence, give the agent ONE bounded turn — with the PASSING verdict — to finalize
    REPORT.md/docs to the verified result, WITHOUT touching code. Re-grade; if the agent regressed the
    package, restore the pre-finalize (passing) submission. Always guarantee the report's final status
    line matches the verdict (programmatic stamp as a last resort). Returns finalize telemetry."""
    snap = run_dir / "_qa_work" / "pre_finalize" / "submission"
    if snap.exists():
        shutil.rmtree(snap.parent)
    snap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ws / "submission", snap,
                    ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))

    (ws / "qa" / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (ws / "FINALIZE.md").write_text(
        "All required public/dev pilot capsules now PASS (see qa/verdict.json: all_pass=true).\n\n"
        "Do ONLY this, then stop:\n"
        "1. Update `submission/REPORT.md` so its final status line reads EXACTLY:\n"
        f"   \"{_PASS_LINE}\"\n"
        "   and its body honestly reflects the verified per-capsule pass + oracle tiers.\n"
        "2. Update `submission/docs/iteration_notes.md` with the final converged state.\n"
        "DO NOT modify manifest.yaml, mlir_oot/, or any code — the package is frozen-pending and must\n"
        "keep passing. Touch only REPORT.md and docs/.\n")
    tpath = run_dir / "rounds" / "finalize.transcript.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    epath = run_dir / "rounds" / "finalize.stderr.log"
    rc = 0
    if _driver_for(model) == "claudecode":
        inner = (f'claude --print --model {model} --effort {effort} '
                 f'--permission-mode bypassPermissions --add-dir {ws} '
                 f'--output-format stream-json --verbose < {ws / "FINALIZE.md"}')
        cmd = bwrap_cmd(inner, ws, bundle) if sandbox == "bwrap" else inner
        try:
            with open(tpath, "w") as tf, open(epath, "w") as ef:
                rc = subprocess.run(["bash", "-c", cmd], cwd=str(ws), stdout=tf, stderr=ef,
                                    timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    else:
        # Converse / OpenCode: the claude CLI can't drive these models, so skip the agent finalize turn —
        # the driver stamps REPORT.md's status line below (_stamp_report_status). An empty transcript keeps
        # audit_transcript happy (no tokens, no answer-access hits).
        tpath.write_text("")

    # re-grade: if the finalize turn broke the (passing) package, restore the snapshot
    regrade = qa_grade(ws, run_dir, 90, False, timeout)
    restored = False
    if not regrade.get("all_pass"):
        shutil.rmtree(ws / "submission")
        shutil.copytree(snap, ws / "submission")
        restored = True
    # guarantee the frozen report's status line matches the verified verdict
    stamped = _stamp_report_status(ws / "submission" / "REPORT.md", _PASS_LINE)
    audit = audit_transcript(tpath, arm)
    return {"agent_rc": rc, "regrade_all_pass": regrade.get("all_pass"),
            "restored_after_regression": restored, "status_line_stamped_by_driver": stamped,
            "answer_access_clean": audit["clean"], "audit_hits": audit["hits"],
            "transcript": str(tpath)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["raw_baseline", "merlin_assisted", "cpp_merlininfra"], default=ARM,
                    help="which arm/bundle to run (default raw_baseline; the QA loop is identical)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    # AGENT DRIVER (Claude-Code-like interfaces). auto (default) preserves today's behavior: route by model
    # id — the Bedrock Converse loop for a non-Anthropic id, else the claude CLI.
    ap.add_argument("--driver", choices=["auto", "converse", "claudecode", "opencode", "codex"],
                    default="auto",
                    help="agent driver: auto (route by model id), converse (Bedrock Converse loop), "
                         "claudecode (claude CLI; Bedrock via --provider bedrock), opencode (OpenCode CLI), "
                         "codex (Codex CLI; ChatGPT auth = subscription_notional cost, never metered)")
    ap.add_argument("--subagent-model", default="",
                    help="delegate/subagent model (alias or Bedrock id) for tier-within-agent; default per "
                         "driver (Anthropic: sonnet; non-Anthropic: qwen-coder)")
    ap.add_argument("--background-model", default="",
                    help="background/mechanical model (alias or Bedrock id) for chores; default per driver "
                         "(Anthropic: haiku; non-Anthropic: nova-lite)")
    # PROVIDER for the agent's `claude` CLI — experiments-only, so the interactive Claude Code keeps the
    # subscription. subscription (default) = the machine's ~/.claude creds; bedrock = Claude Code's own
    # Bedrock mode (CLAUDE_CODE_USE_BEDROCK=1 + AWS creds + a Bedrock inference-profile model id).
    ap.add_argument("--provider", choices=["subscription", "bedrock"], default="subscription",
                    help="model provider for the agent CLI (experiments-only; subscription keeps ~/.claude)")
    ap.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"),
                    help="AWS region for --provider bedrock")
    ap.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""),
                    help="AWS profile (~/.aws) for --provider bedrock; else the env-var cred chain")
    ap.add_argument("--schedule", choices=("rounds", "continuous"), default="rounds",
                    help="ROUNDS (default, byte-identical to before): the loop is bounded by "
                         "--max-rounds, and the run ends when that budget is spent whether or not the "
                         "submission was still improving. CONTINUOUS: the round COUNT stops being a "
                         "terminator — the run ends on convergence, on a plateau, or on a wall/spend "
                         "budget, i.e. on evidence about the submission rather than on an arithmetic "
                         "cap. Per-capsule promotion is unaffected by this flag: a capsule's cert tier "
                         "is enqueued the moment its loop tier passes (tier_promote fires on EVERY "
                         "verdict — both brokers and the round grade), never at a round boundary. What "
                         "continuous removes is the ARTIFICIAL end, not the grading cadence.\n"
                         "Rounds remain the unit of agent invocation and of the artifact layout in both "
                         "modes, so every downstream reader (round_NN transcripts, qa_history verdicts, "
                         "the cost rollup) is unchanged.")
    ap.add_argument("--max-wall-s", type=int, default=0,
                    help="continuous only (0 = no wall cap): stop after this much ACTIVE agent wall "
                         "time. With --schedule continuous and no wall cap and no plateau, the only "
                         "terminators left are convergence and the spend ceiling — which is what you "
                         "want for a run whose whole point is to finish the work, but say so on purpose.")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--min-rounds", type=int, default=0,
                    help="OPT-IN (0 = disabled, the default). Refuse the agent's READY_FOR_BARRIER "
                         "self-declaration before round N unless it is actually passing. A model that "
                         "writes the marker while scoring zero has not converged, it has given up: GLM-5 "
                         "did exactly that in four consecutive runs, once after only two rounds, leaving "
                         "ten of its twelve rounds unspent. Declining the marker deletes it (the agent "
                         "must re-declare) and returns the loop to the agent with the same failing "
                         "verdict. Never overrides a genuine all_pass, which always ends the loop.")
    ap.add_argument("--plateau-rounds", type=int, default=0,
                    help="OPT-IN (0 = disabled, the default — never cut a productive run). When set to N, "
                         "stop early (not converged) after N consecutive rounds with NO progress: neither "
                         "the pass count NOR the total numeric mismatch improved. The mismatch-aware metric "
                         "means a run making ANY numeric progress is never stopped (verified: it would not "
                         "have fired on the productive glm5 15/20 run, whose flat-pass stretches kept "
                         "reducing mismatch). Enable it only for a run you know is pathologically stuck "
                         "(re-sending its uncached context each round with zero movement), e.g. N=3-4.")
    ap.add_argument("--round-timeout", type=int, default=14400,
                    help="per-round agent wall cap (s). Default 4h (matches launch_ab_batch): a TIGHT "
                         "cap is net-detrimental — it doesn't cut the work (fixed by difficulty), it "
                         "just forces more rounds, each adding a full grading pass + context re-read + "
                         "rate-limit-boundary exposure, and can cut a productive round mid-fix (rc=124). "
                         "The original abc runs used 4h and converged in ~1 productive round.")
    ap.add_argument("--qa-timeout", type=int, default=900)
    ap.add_argument("--model-budget-s", type=int, default=None,
                    help="wall-clock ceiling for ONE whole-model capsule inside a round grade (s). "
                         "Default: --qa-timeout, i.e. the capstone may cost at most what this operator "
                         "already said one grading step may. --qa-timeout itself is a PER-STEP "
                         "subprocess cap and a whole-model grade makes many such calls, so it cannot "
                         "bound the capsule. MEASURED: with no ceiling the gemmini capstone ran 5h30m "
                         "past its round's own 4h timeout, wrote nothing, and the round never graded. "
                         "0 = no ceiling (an operator certification run wants that; a per-round gate "
                         "does not).")
    # NOT default-on, and deliberately so: this is the LEGACY single-session path. It keeps one agent
    # session and re-grades underneath it, but it does NOT run the post-freeze public+hidden L3 grade, so
    # it can report progress and can NEVER report a formal success (it returns 1 and hardcodes
    # formal_complete=False). The certified continuous path is `--schedule continuous`, where the round
    # COUNT is not a terminator: the run stops on EVIDENCE (converged, plateaued) or on a declared budget,
    # and `--max-rounds` is ignored. Pair it with a long `--round-timeout` (e.g. 43200) so each agent
    # session is long and barriers are rare.
    #
    # Measured 2026-09-01: launched with `--continuous`, both gemmini sessions ended after ~1.5h at 18/33
    # with `grades=2` and `formal_complete=False` -- the session closed when the agent stopped, well inside
    # a 12h --round-timeout, and no formal verdict was reachable.
    ap.add_argument("--continuous", action="store_true",
                    help="LEGACY single-session mode: one long-lived agent session with a background "
                         "grader every --grade-interval seconds, instead of round relaunches. Reports "
                         "progress only -- it does NOT run the post-freeze public+hidden L3 grade and so "
                         "can never report a formal success. For the certified continuous path use "
                         "`--schedule continuous` with a long --round-timeout.")
    ap.add_argument("--grade-interval", type=int, default=900,
                    help="seconds between background grades in --continuous mode (default 900)")
    # Default sandbox=bwrap: enforced FS isolation (the agent cannot read a masked answer surface at all),
    # AND the driver-side broker tools (async simjob oracle, self-check, arc-model isa_tools, CCA) only
    # start under bwrap. The earlier "bwrap crashes claude" blocker was a sandbox-config bug, now fixed in
    # sandbox/bwrap.base_argv (bind the systemd-resolved dir so DNS works; drop inherited CLAUDE_CODE_*
    # nesting markers; provide an XDG runtime dir) — validated end-to-end by smoke_agent_check. "none" stays
    # available as an escape hatch (golden-masked COPY workspace + post-run transcript audit), but it is
    # detection-not-prevention and must not be used for scored runs.
    ap.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap")
    ap.add_argument("--allow-unsandboxed", action="store_true",
                    help="explicitly permit a real (spending) run under --sandbox none; without it a "
                         "'none' run is refused (workspace assembly alone does not hide denied paths).")
    ap.add_argument("--no-oracle", action="store_true", help="QA = L0+trace only (fast dev)")
    ap.add_argument("--skip-hidden", action="store_true")
    ap.add_argument("--experiment", choices=["full", "realistic"], default="full",
                    help="'realistic' (abc2): whole-repo + self-check tool + READY_FOR_BARRIER self-pacing "
                         "+ TASK_realistic; 'full' (abc1, default): spike-loop then verilator checkpoint")
    ap.add_argument("--bundle", default="", help="override the arm's bundle id (e.g. *_realistic_v0)")
    # Rate-limit awareness: when the org five-hour session budget rejects a round (zero work), sleep
    # until resetsAt and RETRY the same round instead of burning it. Lets an n>=3 sweep span windows
    # unattended. --max-rate-limit-waits caps total resets waited across the run (then stop honestly).
    ap.add_argument("--max-rate-limit-waits", type=int, default=3)
    ap.add_argument("--rl-test-reset-epoch", type=int, default=0,
                    help="TEST ONLY: override the reset epoch to wait toward (verification, not 5h)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing run_dir (cross-window robustness) instead of refusing")
    # Use a SECOND subscription account (separate org five-hour budget) by pointing claude at a
    # different config dir. This is the only real way to run two agent arms truly concurrently: the
    # five-hour limit is org-wide and non-overage, so process parallelism on ONE account just
    # time-shares the same bucket; a second account = a second bucket. Log in once with
    # `CLAUDE_CONFIG_DIR=<dir> claude auth login`, then pass --account-config-dir <dir> here.
    ap.add_argument("--with-tool", action="append", default=[], metavar="NAME",
                    help=f"ABLATION: grant this arm-gated tool on top of the arm's rung (repeatable). "
                         f"Known: {', '.join(_TR.ablatable_tools())}")
    ap.add_argument("--without-tool", action="append", default=[], metavar="NAME",
                    help="ABLATION: withhold this arm-gated tool from the arm's rung (repeatable). Pair "
                         "with a bundle generated for the same cell so the file grants match the brokers.")
    ap.add_argument("--account-config-dir", default=os.environ.get("CLAUDE_CONFIG_DIR", ""),
                    help="CLAUDE_CONFIG_DIR for the agent's claude CLI (a different subscription account)")
    a = ap.parse_args(argv)
    # A real (spending) run MUST be sandboxed: without bwrap the agent can read any absolute path (incl.
    # denied /scratch* answer dirs), so the copy-workspace + post-hoc transcript audit alone do NOT isolate
    # it. Fail closed (parity with run_agent_experiment.py) — an unsandboxed run needs an explicit opt-in.
    if a.sandbox != "bwrap" and not a.allow_unsandboxed:
        print("REFUSING: a real run requires --sandbox bwrap (or explicit --allow-unsandboxed). "
              "Workspace assembly alone does not hide denied absolute paths.", file=sys.stderr)
        return 4
    # BOUND THE CAPSTONE. The round grade includes any whole-model capsule whose op-pass gate is met,
    # and that grade is unbounded by --qa-timeout (a per-step cap, not a per-capsule one). Set the
    # ceiling the runner reads before any grading happens; 0 clears it for an unlimited operator run.
    _mb = a.qa_timeout if a.model_budget_s is None else a.model_budget_s
    if _mb:
        os.environ["MERLIN_MODEL_BUDGET_S"] = str(_mb)
    else:
        os.environ.pop("MERLIN_MODEL_BUDGET_S", None)
    arm = a.arm
    global _EXPERIMENT, _ARM, _DRIVER, _SUBAGENT_MODEL, _BACKGROUND_MODEL, _ADD_TOOLS, _DROP_TOOLS
    _EXPERIMENT = a.experiment
    _ARM = arm
    _DRIVER = a.driver
    _SUBAGENT_MODEL = a.subagent_model
    _BACKGROUND_MODEL = a.background_model
    # Fail closed on an unknown tool name BEFORE any spend: a typo would otherwise ablate nothing and
    # the cell would silently be a duplicate of its own rung.
    for _n in (*a.with_tool, *a.without_tool):
        _TR.spec(_n)
    _ADD_TOOLS, _DROP_TOOLS = tuple(a.with_tool), tuple(a.without_tool)
    if _ADD_TOOLS or _DROP_TOOLS:
        print(f"ABLATION CELL: {arm}{_TR.cell_suffix(_ADD_TOOLS, _DROP_TOOLS)} "
              f"-> tools {list(_resolved_tools())}")

    # DRIVER-SIDE grade env: qa_grade/_verilator_grade run build_package (cmake) for C++ submissions in
    # THIS process's env. The conda cmake transitively needs libidn.so.11 (host has only .12) -> ensure
    # the .compat_lib shim + conda libs are on LD_LIBRARY_PATH, else the C++ build fails "libidn.so.11:
    # cannot open shared object file" and every grade is 0/0 (the abc8 rb blocker). Python arms have no
    # build step so this is a harmless no-op for them.
    _CE = str(ext_path("chipyard") / ".conda-env")   # from .env, like the sandbox toolchain binds
    _compat = str(C.REPO / ".compat_lib")
    os.environ["LD_LIBRARY_PATH"] = (f"{_compat}:{_CE}/lib:{_CE}/riscv-tools/lib:"
                                     + os.environ.get("LD_LIBRARY_PATH", ""))
    # --- provider (experiments-only): route the agent's claude CLI to Bedrock or the subscription ---
    # subprocess (rounds + finalize) inherits os.environ, so setting these here scopes the provider to
    # THIS experiment process only — the user's interactive Claude Code (a different process, no Bedrock
    # env) stays on the subscription. bedrock uses Claude Code's OWN Bedrock mode; --model must then be a
    # Bedrock inference-profile id (e.g. us.anthropic.claude-...-v1:0). ~/.aws is bound into the sandbox
    # by claude_runtime_binds when CLAUDE_CODE_USE_BEDROCK=1.
    if a.provider == "bedrock":
        os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
        os.environ["AWS_REGION"] = a.aws_region
        os.environ.setdefault("AWS_DEFAULT_REGION", a.aws_region)
        if a.aws_profile:
            os.environ["AWS_PROFILE"] = a.aws_profile
        # Bearer-token auth (AWS_BEARER_TOKEN_BEDROCK) lives in the gitignored .env; the read-only .env
        # loader never mutates os.environ, so surface it here (a pre-set env var / --aws-profile still
        # wins) — this is the cred the sandboxed claude inherits when no ~/.aws profile is used.
        if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            from merlin.common.paths import env as _dotenv
            _bearer = _dotenv("AWS_BEARER_TOKEN_BEDROCK")
            if _bearer:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = _bearer
        # Full Claude Code multi-model on Bedrock: the Opus PRIMARY (--model) ORCHESTRATES, delegates the
        # sub-tasks it spawns via the Task tool to Sonnet (CLAUDE_CODE_SUBAGENT_MODEL), and routes
        # lightweight background chores (titles/summaries) to Haiku. Claude Code has NO native task-
        # complexity auto-routing — delegation is via subagents, so pinning the subagent model is the
        # lever that gives "Opus delegates down when possible". Pin all three Bedrock profiles; each is
        # overridable via its env var. ⚠️ opus-4-8 / sonnet-5 / opus-5 are LISTED by list-inference-
        # profiles but NOT invocable on this account — the invocable set is opus-4-6-v1 / sonnet-4-6 /
        # haiku-4-5, so those are the defaults.
        # Tier overrides: --subagent-model / --background-model win over the defaults below (set BEFORE the
        # setdefaults, so those become no-ops when an override is supplied). Aliases resolve via model_tiers.
        from model_tiers import resolve as _rmodel
        if a.subagent_model:
            os.environ["CLAUDE_CODE_SUBAGENT_MODEL"] = _rmodel(a.subagent_model)
        if a.background_model:
            _bg = _rmodel(a.background_model)
            os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = _bg
            os.environ["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = _bg
        _haiku = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        os.environ.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", "us.anthropic.claude-opus-4-6-v1")
        os.environ.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", "us.anthropic.claude-sonnet-4-6")
        os.environ.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", _haiku)
        os.environ.setdefault("ANTHROPIC_SMALL_FAST_MODEL", _haiku)          # background chores -> Haiku
        os.environ.setdefault("CLAUDE_CODE_SUBAGENT_MODEL",                  # delegated subagents -> Sonnet
                              "us.anthropic.claude-sonnet-4-6")
        print(f"[provider] agent CLI -> Bedrock (region={a.aws_region}"
              f"{', profile=' + a.aws_profile if a.aws_profile else ', env-var creds'}); "
              f"primary={a.model}; subagent={os.environ['CLAUDE_CODE_SUBAGENT_MODEL']}; "
              f"background={os.environ['ANTHROPIC_SMALL_FAST_MODEL']}")
    else:
        # Belt-and-braces: never let a stray Bedrock env leak a 'subscription' run onto Bedrock.
        os.environ.pop("CLAUDE_CODE_USE_BEDROCK", None)

    if a.bundle:                                  # abc2 realistic: point the arm at its realistic bundle
        RX.ARM_BUNDLE[arm] = a.bundle

    # --- per-arm integrity policy (runtime override of the FROZEN oot_runner._FORBIDDEN; the file is
    # NOT edited) ---------------------------------------------------------------------------------
    # The merlin arms are supposed to BUILD WITH the Merlin xDSL framework — so importing
    # merlin.xdsl_dialects (ops/types/verifiers/rewrite patterns) at runtime is legitimate, NOT cheating.
    # The default frozen policy forbids ALL `import merlin`, which wrongly blocks the framework and forced
    # the agents into standalone regex emitters. For the realistic merlin arms we relax to ORACLE-ONLY:
    # still forbid the answer (reference/simulator/reference_outputs) + the oracle-routing lowering
    # pipeline, but ALLOW the framework. baseline keeps the full standalone policy (it's the control).
    if _EXPERIMENT == "realistic" and arm == "merlin_assisted":
        from merlin.targetgen import oot_runner as _OOT
        _OOT._FORBIDDEN = ("merlin.runtime.reference", "merlin.runtime.simulator", "reference_outputs",
                           "xdsl_dialects.lowering", "xdsl_dialects/lowering", "outputs_match")
        print(f"[integrity] merlin arm: framework imports ALLOWED; oracle still forbidden "
              f"({_OOT._FORBIDDEN})")
        # "xDSL MUST work": make xdsl + the merlin framework importable for the package's runtime
        # subprocess UNDER ANY python the agent's manifest invokes (even bare system python3). The
        # grader spawns the entrypoint inheriting this env, so prepend the .venv site-packages.
        _site = str(f"{_ROOT}/.venv/lib/python3.13/site-packages")
        os.environ["PYTHONPATH"] = _site + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
        # sanity: confirm xdsl is importable in this env right now (fail LOUD if the framework is broken)
        import importlib.util as _u
        if _u.find_spec("xdsl") is None:
            print("[FATAL] xdsl not importable for the merlin arm — xDSL approach cannot work; aborting",
                  file=sys.stderr); return 7
        print(f"[xdsl] framework importable via PYTHONPATH={_site} (xdsl present) ✓")

    # Route every claude subprocess (rounds + finalize) at the chosen account's config dir. subprocess
    # inherits os.environ, so setting it here propagates without touching launch_agent/finalize.
    account_info = {"config_dir": None, "email": None, "orgId": None}
    if a.account_config_dir:
        cfg = os.path.abspath(os.path.expanduser(a.account_config_dir))
        os.environ["CLAUDE_CONFIG_DIR"] = cfg
        account_info["config_dir"] = cfg
        try:
            st = json.loads(subprocess.run(["claude", "auth", "status"], capture_output=True,
                                           text=True, timeout=60).stdout)
            account_info.update({"email": st.get("email"), "orgId": st.get("orgId"),
                                 "loggedIn": st.get("loggedIn")})
            if not st.get("loggedIn"):
                print(f"ERROR: account-config-dir {cfg} is NOT logged in. Run: "
                      f"CLAUDE_CONFIG_DIR={cfg} claude auth login", file=sys.stderr)
                return 6
            print(f"[account] agent runs as {account_info['email']} (org {account_info['orgId']}) "
                  f"via {cfg}")
        except Exception as e:
            print(f"ERROR: could not read auth status for {cfg}: {e}", file=sys.stderr)
            return 6

    bundle = RX._load_bundle(arm)
    bundle_dir = C.BUNDLES / RX.ARM_BUNDLE[arm]
    from merlin.targetgen.sandbox import bwrap as _BWS
    run_dir = C.RUNS / arm / a.run_id
    _resuming = run_dir.exists() and a.resume
    if run_dir.exists() and not a.resume:
        print(f"run dir exists, refusing to overwrite: {run_dir}", file=sys.stderr)
        return 2
    # Stage the agent workspace OUTSIDE runs/ (a denied path): bwrap_argv tmpfs-masks every denied
    # dir, so a workspace under runs/ would be clobbered (bwrap can't chdir into it). _qa_ws is not
    # denied, so its --bind ws ws survives the deny-masking.
    ws_root = C.EXP / "_qa_ws" / a.run_id
    ws = ws_root / "workspace"
    # Resume reuses the existing workspace+submission (cross-window continuation); a fresh run wipes any
    # stale workspace and re-stages from the golden-masked bundle.
    _have_ws = _resuming and (ws / "submission").exists()
    if ws_root.exists() and not _have_ws:
        _BWS.remove_bundle_snapshot(ws)
        shutil.rmtree(ws_root)
    run_dir.mkdir(parents=True, exist_ok=_resuming)
    _archived_bundle_manifest = run_dir / "input_bundle_manifest.yaml"
    if _resuming:
        if not _archived_bundle_manifest.is_file():
            raise RuntimeError("resume refused: archived input bundle manifest is missing")
    else:
        shutil.copy(bundle_dir / "input_bundle_manifest.yaml", _archived_bundle_manifest)

    if _have_ws:
        print(f"[resume] reusing existing workspace + submission at {ws}")
        _BWS.verify_bundle_snapshot(ws, bundle, repo=C.REPO)
        denied_names = [Path(d["path"]).name for d in bundle.get("denied", [])]
        viol, copy_report = [], None
    elif a.sandbox == "bwrap":
        denied_names = RX.assemble_workspace(bundle, ws)
        viol = RX.assert_isolation(ws, bundle)
        copy_report = None
    else:
        copy_report = assemble_copy_workspace(bundle, ws)
        denied_names = [Path(d["path"]).name for d in bundle.get("denied", [])]
        viol = []  # copy workspace contains no symlinks into denied paths by construction
    _bundle_snapshot_record = None
    _model_host_lane_snapshot = None
    _hidden_snapshot_record = None
    _hidden_dir = None
    if a.sandbox == "bwrap":
        # Both fresh setup and resume arrive here. Verify before exporting the host-only pointer; every
        # in-process grade and every operator-side broker inherits it, while bwrap_cmd strips it from the
        # agent environment. The model grader independently re-verifies the aggregate before use.
        _BWS.verify_bundle_snapshot(ws, bundle, repo=C.REPO)
        _snapshot_root = _BWS.bundle_snapshot_root(ws).resolve(strict=True)
        _bundle_snapshot_record = _BWS.snapshot_record(ws)
        _te_setup = _te()
        _hidden_dir = _hidden_snapshot_dir(_snapshot_root, _te_setup, C.REPO)
        _hidden_snapshot_record = _subtree_snapshot_record(_hidden_dir)
        if _hidden_snapshot_record["n_capsules"] <= 0:
            raise RuntimeError(
                f"hidden capsule snapshot contains no capsules: {_hidden_snapshot_record['path']}")
        if _te_setup.host_lane is not None:
            _, _model_host_lane_snapshot = _te_setup.resolve_host_lane(
                root=_snapshot_root / "repo")
            _model_host_lane_snapshot["run_snapshot"] = _bundle_snapshot_record
        os.environ[_MODEL_HOST_SNAPSHOT_ROOT_ENV] = str(_snapshot_root)
        os.environ[_MODEL_HOST_SNAPSHOT_REQUIRED_ENV] = "1"
    else:
        # Do not let an inherited pointer bind an explicitly unsandboxed diagnostic to another run.
        os.environ.pop(_MODEL_HOST_SNAPSHOT_ROOT_ENV, None)
        os.environ.pop(_MODEL_HOST_SNAPSHOT_REQUIRED_ENV, None)

    # Stage every prompt/document before provenance is written.  On resume these bytes are NEVER rebuilt
    # from the current worktree; they must match the treatment record from the first invocation.
    if not _resuming:
        _build_task(arm, ws, run_dir, sandbox=a.sandbox)
    elif not (ws / "TASK.md").is_file():
        raise RuntimeError("resume refused: sealed workspace TASK.md is missing")

    _resolved_tool_ids = _resolved_tools()
    _task_scope_record = _task_runtime_scope(_te(), a.sandbox)
    if (_hidden_snapshot_record is not None
            and _hidden_snapshot_record["n_capsules"] != _task_scope_record["held_out_capsules"]):
        raise RuntimeError(
            "live descriptor scope and frozen hidden snapshot disagree at setup: "
            f"{_task_scope_record['held_out_capsules']} vs "
            f"{_hidden_snapshot_record['n_capsules']}")
    _environment_path = run_dir / "environment.yaml"
    if _resuming:
        try:
            _environment_record = yaml.safe_load(_environment_path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001 — provenance is a formal resume gate
            raise RuntimeError(f"resume refused: environment record unreadable: {exc}") from exc
        _identity = {
            "run_id": a.run_id, "arm": arm, "sandbox": a.sandbox,
            "bundle_id": bundle["bundle_id"],
        }
        _expected_hidden_dir = (_hidden_snapshot_dir(_snapshot_root, _te(), C.REPO)
                                if a.sandbox == "bwrap" else None)
        try:
            _hidden_dir = _verify_persisted_run_inputs(
                _environment_record, identity=_identity, task_scope=_task_scope_record,
                ws=ws, run_dir=run_dir, bundle_dir=bundle_dir,
                resolved_tools=_resolved_tool_ids, expected_hidden_dir=_expected_hidden_dir)
        except RuntimeError as exc:
            raise RuntimeError(f"resume refused: {exc}") from exc
        if _hidden_dir is not None:
            _hidden_snapshot_record = dict(_environment_record["hidden_capsule_snapshot"])
        # Preserve the original setup record verbatim.  In particular, do not replace its start time,
        # exact task hash, or account with whichever process happens to perform the resume.
    else:
        _treatment_snapshot = _treatment_snapshot_record(
            ws, run_dir, bundle_dir, _resolved_tool_ids)
    mask = mask_selftest(ws, bundle, a.sandbox)
    if not _resuming:
        _environment_record = {
            "run_id": a.run_id, "arm": arm, "model": a.model, "effort": a.effort,
            # driver + provider decide how a dollar figure must be READ later: a subscription run's cost
            # is notional, a bedrock run's is metered spend against the budget.
            "driver": _DRIVER, "provider": a.provider,
            "subagent_model": _SUBAGENT_MODEL or None, "background_model": _BACKGROUND_MODEL or None,
            "sandbox": a.sandbox, "qa_loop": True,
            "task_scope": _task_scope_record,
            "workspace_path": str(ws), "workspace_copy_report": copy_report,
            "bundle_input_snapshot": _bundle_snapshot_record,
            "hidden_capsule_snapshot": _hidden_snapshot_record,
            "model_host_lane_snapshot": _model_host_lane_snapshot,
            "repo_sha": C.repo_sha(), "bundle_id": bundle["bundle_id"],
            # The explicit list remains convenient for analysis; treatment_snapshot binds it to the
            # source declarations and to the exact task/docs that instructed the agent.
            "resolved_tools": list(_resolved_tool_ids),
            "treatment_snapshot": _treatment_snapshot,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "isolation_violations": viol, "denied_paths_checked": denied_names,
            "golden_mask_selftest": mask,
            "account": account_info,  # which subscription/org budget this arm drew from (provenance)
        }
        _environment_path.write_text(yaml.safe_dump(_environment_record, sort_keys=False))
    if viol:
        print(f"ISOLATION FAILURE: {viol}", file=sys.stderr)
        return 3
    if mask["pilot_golden_visible_to_agent"] != "OK":
        print(f"GOLDEN-MASK FAILURE: agent can see golden values: {mask}", file=sys.stderr)
        return 5
    print(f"[setup] isolation ok; golden-mask: {mask}")

    # --- oracle preflight: abort a GRADEABLE run BEFORE spending if its required oracle can't run ------
    # A run graded WITHOUT its numeric oracle can only ever emit `oracle_unavailable` (the atlas 0/11 at
    # ~$43 failure): the agent gets no actionable failure plane and thrashes to timeout. Compute up-front
    # whether THIS target's required oracle is actually runnable — for an external_backend target the mlc
    # arc cosim + the model venv, for arc/chipyard the arc model / sim binaries (all routed from the
    # contract by capsule_runner.oracle_available, no target literal). If it cannot run AND the operator
    # did NOT ask for an explicit `--no-oracle` structure-only smoke, STOP here having launched no agent
    # and spent zero tokens (this is strictly before the first launch_agent in the round loop below).
    from merlin.targetgen import capsule_runner as _CRpf
    _te_pf = _te()
    _ora_ok, _ora_why = _CRpf.oracle_available(_te_pf.target, _te_pf.sim_via)
    (run_dir / "oracle_preflight.yaml").write_text(yaml.safe_dump({
        "target": _te_pf.target, "sim_via": _te_pf.sim_via,
        "oracle_available": _ora_ok, "reason": _ora_why,
        "no_oracle": bool(a.no_oracle),
        "verdict": "GO" if (_ora_ok or a.no_oracle) else "NO_GO",
    }, sort_keys=False))
    if not _ora_ok and not a.no_oracle:
        print(f"NO_GO: {_ora_why} — refusing to launch a gradeable run with no numeric oracle "
              f"(zero tokens spent). Re-run with --no-oracle for an explicit structure-only smoke, or "
              f"set MERLIN_MLC_DIR + build the arc model / model venv.", file=sys.stderr)
        return 4
    if not _ora_ok:  # a.no_oracle is set — honest structure-only smoke (see qa_grade/no-oracle path)
        print(f"[preflight] oracle unavailable ({_ora_why}); proceeding as an EXPLICIT --no-oracle "
              f"structure-only smoke (NOT gradeable — structural tiers only).")
    else:
        print(f"[preflight] oracle GO: {_ora_why}")

    # oracle_available proves we can GRADE; codegen_smoke proves the target's OWN emit path can PRODUCE a
    # runnable kernel (fork-free build -> reference sim -> correct result). A broken emit path would
    # tool-crash on EVERY capsule of a paid run (the atlas 0/N mode from the PRODUCE side, not the grade
    # side). It returns n/a for targets whose emit path this smoke doesn't cover, and only a hard False for
    # a genuinely broken fork-free pipeline — so gate only on that explicit False (zero tokens spent).
    if not a.no_oracle:
        _cg_ok, _cg_why = _CRpf.codegen_smoke(_te_pf.target)
        (run_dir / "codegen_smoke.yaml").write_text(yaml.safe_dump(
            {"target": _te_pf.target, "codegen_ok": _cg_ok, "reason": _cg_why}, sort_keys=False))
        # `is False`, not falsy: None means the smoke DID NOT RUN (n/a for this emit path), which is
        # neither a pass nor a NO_GO. Gating on falsiness would refuse every target the smoke does not
        # cover; recording None as True is what let a doomed run look verified.
        if _cg_ok is False:
            print(f"NO_GO: codegen smoke failed — the target's emit path cannot produce a runnable "
                  f"kernel: {_cg_why}. Refusing to launch (zero tokens spent).", file=sys.stderr)
            return 4
        print(f"[preflight] codegen smoke: {_cg_why}")

    # realistic (abc2): the self-check tool the agent runs logs each invocation here (dev-trajectory /
    # soft-failure / tool-trigger record). T0 anchors wall-offset. Inherited by every claude subprocess.
    if _EXPERIMENT == "realistic":
        os.environ["SELFCHECK_LOG"] = str(run_dir / "selfcheck_log.jsonl")
        os.environ.setdefault("SELFCHECK_T0", str(time.time()))

    # --- durable checkpoint so the experiment AND its accumulated time survive a process death ---
    # (in-process backoff handles a quota hit while alive; this handles reboot / session-end / OOM).
    # cumulative timing is split into active work vs rate-limit waiting and persisted after EACH round
    # and EACH wait, so a fresh --resume invocation continues exactly where it stopped — not from 0.
    state_p = run_dir / "qa_loop_state.yaml"
    rounds_summary: list = []
    try:                             # endpoint_kind — drives the per-round dev-conformance flag (asm applies only to external_backend)
        from merlin.targetgen.generate_prompt import prompt_slots as _pslots
        _endpoint_kind = _pslots(_te(), _manifest()).get("endpoint_kind", "")
    except Exception:  # noqa: BLE001
        _endpoint_kind = ""
    _best_progress = None            # plateau early-stop: best (#passed, -total_mismatch) seen so far
    _plateau_stall = 0               # consecutive rounds with no progress
    rl_waits_used = 0

    def _progress_key(v: dict) -> tuple:
        """Round progress, higher is better: (#passed, -total residual numeric mismatch). A non-passing
        capsule with no numeric mismatch (a structural fail) counts as a large residual, so a structural
        stall never reads as 'solved'. Used only to detect a plateau — never to grade."""
        tot = 0
        for pc in (v.get("per_capsule") or []):
            if pc.get("status") == "pass":
                continue
            mc = pc.get("mismatch_count")
            tot += int(mc) if isinstance(mc, int) else 1_000_000
        return (v.get("n_passed") or 0, -tot)
    active_wall_s = 0.0              # cumulative time DOING work (launch+grade) across all invocations
    rate_limit_wait_s = 0.0         # cumulative time slept waiting for five-hour window resets
    started_at = datetime.now(timezone.utc).isoformat()
    verdict = {"all_pass": False}
    workflow_conformant = False
    rnd = 0
    # Survives a --resume where the numeric/workflow checkpoint is already complete and the normal round
    # loop therefore does not launch again.  ``finalize.transcript`` is deliberately excluded: it is a
    # docs-only turn, not the authoring workflow whose mandatory tools must be evidenced.
    _saved_authoring_transcripts = sorted((run_dir / "rounds").glob("round_*.transcript.jsonl"))
    _latest_authoring_tpath = _saved_authoring_transcripts[-1] if _saved_authoring_transcripts else None
    if _resuming and state_p.exists():
        st = yaml.safe_load(state_p.read_text()) or {}
        rounds_summary = st.get("rounds", []) or []
        rnd = int(st.get("next_round", 0))
        verdict = {"all_pass": bool(st.get("converged", False))}
        workflow_conformant = bool(st.get("workflow_conformant", False))
        cum = st.get("cumulative", {}) or {}
        active_wall_s = float(cum.get("active_wall_s", 0.0))
        rate_limit_wait_s = float(cum.get("rate_limit_wait_s", 0.0))
        rl_waits_used = int(cum.get("rl_waits_used", 0))
        started_at = cum.get("started_at", started_at)
        print(f"[resume] restored from checkpoint: next_round={rnd} converged={verdict['all_pass']} "
              f"workflow_conformant={workflow_conformant} "
              f"active={active_wall_s:.0f}s rate_limit_wait={rate_limit_wait_s:.0f}s "
              f"waits_used={rl_waits_used}")

    def _authoring_complete() -> bool:
        """Pre-freeze gate: enough evidence to stop editing and begin the official grade."""
        return _authoring_completion(bool(verdict.get("all_pass")), workflow_conformant)

    def _checkpoint(next_round: int) -> None:
        state_p.write_text(yaml.safe_dump({
            "run_id": a.run_id, "arm": arm, "model": a.model, "effort": a.effort,
            "rounds": rounds_summary, "next_round": next_round,
            "converged": _authoring_complete(),
            "numeric_all_pass": bool(verdict.get("all_pass", False)),
            "workflow_conformant": workflow_conformant,
            "cumulative": {"active_wall_s": round(active_wall_s, 3),
                           "rate_limit_wait_s": round(rate_limit_wait_s, 3),
                           "rl_waits_used": rl_waits_used, "started_at": started_at},
            "last_updated": datetime.now(timezone.utc).isoformat()}, sort_keys=False))

    if a.continuous:
        # CONTINUOUS MODE — one session, no round barrier.
        #
        # The round loop exists to (a) give the agent a fresh context and (b) deliver a graded verdict.
        # (b) does not need (a): qa_grade already grades a SNAPSHOT COPY of the workspace and writes the
        # redacted verdict to ws/qa/verdict.json, so it is safe to run while the agent is working. Here a
        # background grader does exactly that on an interval, and the agent — a single long-lived session
        # — sees its feedback refresh continuously instead of at a barrier. Per-capsule tier promotion is
        # unchanged and already immediate: capsule_runner runs each capsule's ladder cheapest-first, so a
        # capsule that clears the screen goes on to the certifying tier in the same grade, not next round.
        import threading
        stop = threading.Event()
        state = {"tick": 0, "verdict": verdict}

        def _grader() -> None:
            while not stop.wait(max(30, int(a.grade_interval))):
                t = state["tick"] + 1
                try:
                    v = qa_grade(ws, run_dir, t, a.no_oracle, a.qa_timeout)
                except Exception as e:  # noqa: BLE001 — a mid-write submission must never kill the run
                    print(f"[continuous] grade {t} skipped: {type(e).__name__}: {e}", flush=True)
                    continue
                state["tick"], state["verdict"] = t, v
                print(f"[continuous] grade {t}: {v.get('n_passed')}/{v.get('n_capsules')} "
                      f"all_pass={v.get('all_pass')}", flush=True)
                _checkpoint(t)
                if v.get("all_pass"):
                    stop.set()                     # converged: stop grading; the session is torn down below
                    return

        gt = threading.Thread(target=_grader, name="continuous-grader", daemon=True)
        gt.start()
        try:
            rc, tpath = launch_agent(ws, run_dir, a.model, a.effort, a.sandbox, bundle, 0,
                                     a.round_timeout, arm=arm)
        except subprocess.TimeoutExpired:
            rc, tpath = 124, run_dir / "rounds" / "round_00.transcript.jsonl"
            print("[continuous] agent session TIMEOUT (the session bound, not a round)")
        stop.set()
        gt.join(timeout=60)
        # FINAL AUTHORITATIVE GRADE: the background grades are progress reports on a moving workspace;
        # the run's verdict is a grade of the submission as the session left it.
        verdict = qa_grade(ws, run_dir, state["tick"] + 1, a.no_oracle, a.qa_timeout)
        conf, workflow_conformant = _workflow_conformance(
            tpath, ws / "submission", arm, _endpoint_kind, _resolved_tools())
        rounds_summary.append({"round": 0, "mode": "continuous", "agent_rc": rc,
                               "grades": state["tick"] + 1,
                               "all_pass": verdict.get("all_pass", False),
                               "workflow_conformant": workflow_conformant,
                               "authoring_complete": _authoring_complete(),
                               "conformance": conf,
                               "n_passed": verdict.get("n_passed"),
                               "n_capsules": verdict.get("n_capsules")})
        _checkpoint(state["tick"] + 1)
        print(f"\ncontinuous run complete: {run_dir}  numeric_all_pass="
              f"{verdict.get('all_pass', False)} workflow_conformant={workflow_conformant} "
              f"authoring_complete={_authoring_complete()} formal_complete=False "
              f"grades={state['tick'] + 1}")
        # This legacy single-session path does not run the common L3 barrier or the post-freeze hidden
        # grader.  It can report progress, but can never report a formal success.  Use
        # ``--schedule continuous`` for the continuous, fully certified path.
        return 1

    cost_capped = False             # set if the batch dollar ceiling (MERLIN_MAX_SPEND_USD) is reached

    def _keep_going() -> bool:
        """Should the loop run another agent invocation?

        ROUNDS: the historical condition, unchanged — bounded by --max-rounds.

        CONTINUOUS: the round COUNT is not a terminator. A run stops on EVIDENCE (converged, plateaued)
        or on a declared BUDGET (wall, spend), never because an arithmetic cap ran out while the
        submission was still improving. Measured on the v12 arm-4 run: it reached its ceiling in round 0
        and then spent two more rounds and 37.9M tokens going nowhere — the round budget was both too
        loose (it kept paying after convergence) and, on other runs, too tight (a productive round cut at
        the cap). Neither failure is about the submission, which is the only thing a stop should be about.
        A safety cap remains via --max-rounds only if the caller explicitly lowers it; the default 12 is
        ignored in continuous mode so it cannot silently reimpose the very bound this removes.
        """
        if _authoring_complete():
            return False
        if a.schedule == "rounds":
            return rnd < a.max_rounds
        if a.max_wall_s and active_wall_s >= a.max_wall_s:
            print(f"[continuous] wall budget reached ({active_wall_s:.0f}s >= {a.max_wall_s}s) — "
                  f"stopping honestly (not converged; reason=max_wall_s)")
            return False
        return True

    while _keep_going():
        print(f"\n===== ROUND {rnd} =====")
        _rstart = time.time()
        try:
            rc, tpath = launch_agent(ws, run_dir, a.model, a.effort, a.sandbox, bundle, rnd,
                                     a.round_timeout, arm=arm)
        except subprocess.TimeoutExpired:
            rc, tpath = 124, run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
            print(f"[round {rnd}] agent TIMEOUT")
        _latest_authoring_tpath = tpath

        # Daily-quota wall: a provider DAILY token limit (429 'too many tokens per day') has no short
        # window reset to sleep to, so retrying burns every remaining round producing empty results
        # (the ccb2 waste). Abort the run early + honestly instead.
        if RL.daily_limit_hit(tpath):
            active_wall_s += time.time() - _rstart
            print(f"[round {rnd}] DAILY TOKEN LIMIT hit (429 'too many tokens per day') — the provider's "
                  f"daily quota is exhausted; aborting early (not converged) rather than burning the "
                  f"remaining rounds. Relaunch after the daily quota resets.")
            _checkpoint(rnd)
            break

        # WEEKLY (seven-day) quota wall: the subscription budget is exhausted and resets DAYS later, so a
        # same-session sleep-and-retry (what the five-hour branch below does) is pointless — it would burn
        # every remaining round against the wall. Crucially this can fire AFTER real work (the agent built
        # a partial submission), so grading it as a normal round would score a misleading 0 and discard the
        # partial. Instead: checkpoint at THIS round (a later --resume continues it — the workspace
        # submission/ persists), write a DISTINCT status, and exit early with a distinct code.
        if ROQ.weekly_quota_hit(tpath):
            active_wall_s += time.time() - _rstart
            _checkpoint(rnd)  # next_round stays rnd: --resume after the weekly reset continues THIS round
            sp = ROQ.write_quota_status(run_dir, ROQ.REASON_WEEKLY, rnd, transcript=tpath)
            print(f"[round {rnd}] {ROQ.STATUS_WEEKLY}: weekly (seven-day) subscription budget exhausted "
                  f"~mid-round; partial submission checkpointed. Relaunch with --resume (same run_id) "
                  f"after the weekly reset — see {sp}. Exiting (not a converged/failed 0).", flush=True)
            return ROQ.QUOTA_WEEKLY_EXIT_CODE

        # Rate-limit backoff: if the org five-hour budget REJECTED this round (zero work), don't
        # consume it — sleep until the window resets and retry the SAME round index.
        if RL.round_rejected(tpath):
            active_wall_s += time.time() - _rstart  # the rejected attempt itself was active time
            if rl_waits_used >= a.max_rate_limit_waits:
                print(f"[round {rnd}] rate-limited and --max-rate-limit-waits "
                      f"({a.max_rate_limit_waits}) exhausted — stopping honestly (not converged)")
                _checkpoint(rnd)
                break
            reset_epoch = a.rl_test_reset_epoch or RL.rate_limit_reset_epoch(tpath) or 0
            sleep_s = max(0.0, reset_epoch - time.time()) + 20  # +jitter past the boundary
            rl_waits_used += 1
            _checkpoint(rnd)  # persist BEFORE the long sleep: a death during sleep resumes at this round
            print(f"[round {rnd}] RATE-LIMITED (five-hour) — wait #{rl_waits_used}; sleeping "
                  f"{sleep_s:.0f}s until window reset, then retrying this round", flush=True)
            time.sleep(sleep_s)
            rate_limit_wait_s += sleep_s
            _checkpoint(rnd)
            continue  # retry same rnd, do NOT append to rounds_summary

        verdict = qa_grade(ws, run_dir, rnd, a.no_oracle, a.qa_timeout)
        # Cross-round MEMORY: write the harness-built round brief (progress log across all graded rounds +
        # the agent's own notes + a stale-notes nudge) so the NEXT fresh session carries its progress
        # instead of re-deriving it. Best-effort — never let a brief build failure end a run.
        try:
            import round_brief
            round_brief.write(run_dir, ws, rnd)
        except Exception as _e:  # noqa: BLE001
            print(f"[round {rnd}] round_brief skipped: {type(_e).__name__}: {_e}")
        # If this round was CUT SHORT mid-work but is resumable in-budget (a wall-clock timeout, rc=124),
        # its partial submission/ persists in the workspace — do NOT treat it as a converged/failed final.
        # Prepend a RESUME banner to the next round's brief so the fresh session CONTINUES the partial
        # (finish manifest.yaml + the CLI + the target artifact first) instead of restarting from scratch.
        _cut, _cut_reason = ROQ.round_was_cut_short(run_dir, rc=rc, transcript=tpath)
        if _cut and ROQ.resume_policy(_cut_reason) == ROQ.RESUME_IN_BUDGET:
            try:
                ROQ.prepend_resume_note(ws, _cut_reason)
                print(f"[round {rnd}] cut short ({_cut_reason}); partial submission preserved — next round "
                      f"will RESUME it (finish manifest.yaml + CLI + target artifact first).")
            except Exception as _e:  # noqa: BLE001
                print(f"[round {rnd}] resume-note skipped: {type(_e).__name__}: {_e}")
        rsum = ET.parse_transcript(tpath, billing_mode=_billing_mode(a.model),
                                   trust_cli_cost=_trust_cli_cost(a.model))
        audit = audit_transcript(tpath, arm)
        # Dev-conformance GATE: numeric progress is still reported, but a nonconformant workflow cannot
        # advance to the official claim-bearing grade.
        conf, workflow_conformant = _workflow_conformance(
            tpath, ws / "submission", arm, _endpoint_kind, _resolved_tools())
        _bad = [k for k, v in (conf.get("checks") or {}).items() if v is False]
        if _bad:
            print(f"[round {rnd}] NOT CONFORMANT — failing: {', '.join(_bad)} "
                  "(numeric grade still reports; formal completion remains blocked)")
        rounds_summary.append({"round": rnd, "agent_rc": rc, "conformance": conf,
                               "all_pass": verdict.get("all_pass"),
                               "workflow_conformant": workflow_conformant,
                               "authoring_complete": _authoring_complete(),
                               "n_passed": verdict.get("n_passed"),
                               "n_capsules": verdict.get("n_capsules"),
                               "tool_calls": rsum.get("tool_calls"),
                               # per-round effort split (was only recorded whole-run before) — lets us
                               # plot tokens/cost/thinking PER round, not just totals.
                               "tokens_total": rsum.get("tokens_total"),
                               "tokens_output": rsum.get("tokens_output"),
                               "tokens_cached": rsum.get("tokens_cached"),
                               "tokens_input": rsum.get("tokens_input"),
                               "thinking_blocks": rsum.get("thinking_blocks"),
                               "tokens_reasoning": rsum.get("tokens_reasoning"),
                               "estimated_cost_usd": rsum.get("estimated_cost_usd"),
                               "billing_mode": rsum.get("billing_mode"),
                               "subscription_notional_usd": rsum.get("subscription_notional_usd"),
                               "answer_access_clean": audit["clean"],
                               "audit_hits": audit["hits"]})
        active_wall_s += time.time() - _rstart
        print(f"[round {rnd}] all_pass={verdict.get('all_pass')} "
              f"{verdict.get('n_passed')}/{verdict.get('n_capsules')} "
              f"answer_access_clean={audit['clean']}")
        # Enforce the batch DOLLAR ceiling: once the shared spend ledger crosses MERLIN_MAX_SPEND_USD, stop
        # before starting the next (paid) round. Uses the authoritative subagent-inclusive per-round cost.
        _over, _spent, _cap = _spend_over_cap(rsum.get("estimated_cost_usd"))
        if _over:
            print(f"[round {rnd}] COST CAP: batch spend ${_spent:.2f} >= ${_cap:.2f} "
                  f"(MERLIN_MAX_SPEND_USD) — stopping before the next round; zero further spend.")
            cost_capped = True
            rnd = rnd + 1
            _checkpoint(rnd)
            break
        rnd = rnd + 1
        _checkpoint(rnd)  # next_round advances only after a completed (non-rate-limited) round
        # Plateau early-stop: a stuck agent re-sends its (uncached) growing context every round for no
        # gain — bound that spend. Stop after N consecutive rounds with no progress (neither the pass
        # count nor the total numeric mismatch improved). Disabled with --plateau-rounds 0.
        if a.plateau_rounds and not verdict.get("all_pass"):
            _prog = _progress_key(verdict)
            if _best_progress is None or _prog > _best_progress:
                _best_progress, _plateau_stall = _prog, 0
            else:
                _plateau_stall += 1
                if _plateau_stall >= a.plateau_rounds:
                    print(f"[plateau] no progress (pass count + mismatch) for {a.plateau_rounds} "
                          f"consecutive rounds; stopping early (not converged) at "
                          f"{verdict.get('n_passed')}/{verdict.get('n_capsules')} to avoid burning tokens "
                          f"on a stuck loop. Raise/relax with --plateau-rounds.")
                    break
        # realistic (abc2): the agent self-paces — it self-checks via the tool and drops READY_FOR_BARRIER
        # when it believes it's done. Break to the verilator barrier on that marker OR on spike all-pass.
        ready = _EXPERIMENT == "realistic" and (ws / "submission" / READY_MARKER).exists()
        # A marker dropped at zero is a surrender, not a convergence, and it silently forfeits the rest
        # of the round budget. --min-rounds declines it: remove the marker so the next round must earn
        # it again, and hand the agent back the same failing verdict. An honest all_pass is never
        # touched -- only the self-declaration is, and only while the run is still failing.
        if ready and not verdict.get("all_pass") and rnd - 1 < a.min_rounds:
            (ws / "submission" / READY_MARKER).unlink(missing_ok=True)
            print(f"[round {rnd-1}] agent dropped {READY_MARKER} at "
                  f"{verdict.get('n_passed')}/{verdict.get('n_capsules')} — DECLINED "
                  f"(--min-rounds {a.min_rounds}); marker cleared, continuing", flush=True)
            ready = False
        if ready and not workflow_conformant:
            (ws / "submission" / READY_MARKER).unlink(missing_ok=True)
            print(f"[round {rnd-1}] agent dropped {READY_MARKER}, but mandatory tooling is not "
                  "successfully evidenced — marker cleared, continuing", flush=True)
            ready = False
        if _authoring_complete() or ready:
            if ready:
                print(f"[round {rnd-1}] agent dropped {READY_MARKER} — proceeding to verilator barrier")
            break

    # --- cycle-accurate L3 (verilator) checkpoint -------------------------------------------------
    # The loop gate is spike (L2) only; this is where kernels are validated on the real RTL. Runs only
    # when the agent is "ready" (loop-oracle-converged on the descriptor-derived public/dev set). Up to
    # VERILATOR_ATTEMPTS chances
    # with a fix-round between each; parallel (max_workers) so 20 capsules don't serialize. Each attempt
    # recorded to verilator_checkpoints.json (the cycle-accurate result the agg/plots report).
    def _verilator_grade(attempt: int) -> dict:
        import qa_check as _qc
        from merlin.targetgen import capsule_grade as _CG
        from merlin.targetgen import capsule_runner as _CR
        vcand = run_dir / "_qa_work" / f"vcand_{attempt}" / "submission"
        if vcand.parent.exists():
            shutil.rmtree(vcand.parent)
        if not (ws / "submission" / "manifest.yaml").exists():
            return {"attempt": attempt, "all_pass": False, "n_passed": 0, "n_capsules": 20, "per_capsule": []}
        shutil.copytree(ws / "submission", vcand,
                        ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
        _strip_build_state(vcand)   # clean, relocatable build for the L3 cert (abc9 L3-0/20 'build' bug)
        vruns = run_dir / "_qa_work" / f"vruns_{attempt}"
        # Cycle-accurate checkpoint = the target's FULL oracle ladder, resolved from the descriptor's
        # target+sim_via via the shared factory (gemmini/chipyard -> spike L2 + verilator L3; an arc/mlc
        # target -> its RTL-derived arc tier), so a new target's L3 cert needs no edit here.
        _te_ck = _te()
        adapters = _CR.qa_checkpoint_adapters(_te_ck.target, _te_ck.sim_via)
        # CIRCT arm only: wrap each sim adapter with the CIRCT structural screen. It is ADVISORY — it
        # records the screen verdict + wall to circt_gate_log.jsonl but NEVER skips the sim or fails the
        # tier (a structural reject can be a false-positive on a conformant-but-different kernel, and
        # skipping would both mis-fail it and bias this arm vs the ungated plain/baseline arms). Both arms
        # now run identical sims; the wrap only adds telemetry for the rtlchecks arm.
        if RX.ARM_BUNDLE.get(arm, "").find("rtlchecks") >= 0:
            try:
                from merlin.targetgen import circt_gate as _GATE
                _glog: list = []
                adapters = {t: _GATE.gated_adapter(adp, log=_glog, target=_te_ck.target)
                            for t, adp in adapters.items()}
                (run_dir / "circt_gate_log.jsonl").write_text("")  # reset; appended after grade
                print(f"[verilator attempt {attempt}] CIRCT screen ADVISORY (records verdict; sim always runs)")
            except Exception as e:
                print(f"[verilator attempt {attempt}] CIRCT gate unavailable ({e}); running ungated")
                _glog = None
        else:
            _glog = None
        try:
            _CG.grade(str(vcand), capsules_root=str(_pilot_subset()), runs_root=str(vruns),
                      labels={"public", "dev"}, contract=str(C.REPO / "merlin/contract"),
                      oracle_adapters=adapters, timeout=_verilator_per_capsule_timeout(),
                      max_workers=_CG.default_grade_workers(), target=_te().target)
        except Exception as e:
            print(f"[verilator attempt {attempt}] grade error: {str(e)[:200]}")
        if _glog is not None:  # record CIRCT gate decisions (skips + per-call wall)
            skipped = sum(1 for r in _glog if r.get("sim_skipped"))
            with open(run_dir / "circt_gate_log.jsonl", "a") as gf:
                for r in _glog:
                    gf.write(json.dumps({"attempt": attempt, **r}) + "\n")
            print(f"[verilator attempt {attempt}] CIRCT gate: {skipped}/{len(_glog)} sims skipped (reject)")
        red = _qc._per_capsule_from_results(vruns)
        per, npass = [], 0
        for name, info in sorted(red.items()):
            l3 = (info.get("tiers") or {}).get("L3")
            npass += int(l3 == "pass")
            per.append({"capsule": name, "l3_status": l3, "failure_plane": info.get("failure_plane")})
        nc = len(red) or 20
        # The Arm4 qa wrapper can derive the same answer-free RTL structural block from this attempt's
        # generated traces.  Carry it into the L3 fix verdict so that a fix round can obey (and prove) its
        # mandatory RTL-feedback readback instead of receiving a reduced verdict with that surface absent.
        _rtl_block = getattr(_qc, "_rtl_block", None)
        try:
            rtl_checks = _rtl_block(vruns) if callable(_rtl_block) else None
        except Exception as _e:  # noqa: BLE001 — advisory data stays non-numeric, but absence is explicit
            rtl_checks = {"error": f"{type(_e).__name__}: {_e}"}
        return {"attempt": attempt, "all_pass": npass == nc and nc > 0, "n_passed": npass,
                "n_capsules": nc, "per_capsule": per, "rtl_checks": rtl_checks}

    verilator_attempts: list = []
    _ready_marker = (ws / "submission" / READY_MARKER).exists()
    # The cycle-accurate RTL barrier is a pass-gate only when the target's corpus makes its RTL-cert tier
    # MANDATORY. A prototype target graded on its functional oracle (L3 optional) skips it, so a normal run
    # is not blocked on a slow/hanging verilator; convergence then rides the functional-tier (L2) verdict.
    _run_l3, _l3_reason = _cycle_accurate_checkpoint_enabled()
    if not _run_l3:
        print(f"[verilator] cycle-accurate RTL (L3) barrier SKIPPED — {_l3_reason}")
    if _run_l3 and workflow_conformant and (
            verdict.get("all_pass") or (_EXPERIMENT == "realistic" and _ready_marker)):
        # ready = spike-converged OR (realistic) the agent declared done -> run the L3 verilator barrier.
        # In realistic mode this barrier IS the definition of done; the agent already self-checked on the
        # tool, so this confirms on the operator side. Up to VERILATOR_ATTEMPTS with a fix-round between.
        # NON-TERMINAL L3 cert: re-grade L3, and on failure hand the redacted cycle-accurate verdict back
        # for a fix round, REPEATING until L3 passes OR the round budget (max_rounds) is exhausted. A
        # premature READY / a single L3 timeout is therefore survivable (it just costs a round) — never a
        # run-ending event (the abc7 failure mode). The only success exit is all-L3-pass.
        print(f"[verilator] L3 cert — NON-terminal ({'READY-marker' if _ready_marker else 'spike-converged'}; "
              f"iterate to L3-pass within {a.max_rounds - rnd} remaining rounds)")
        vatt = 0
        while True:
            vatt += 1
            _vs = time.time()
            vv = _verilator_grade(vatt)
            active_wall_s += time.time() - _vs
            verilator_attempts.append({k: vv[k] for k in ("attempt", "all_pass", "n_passed", "n_capsules")})
            (run_dir / "verilator_checkpoints.json").write_text(json.dumps(
                {"attempts": verilator_attempts, "final_all_pass": vv["all_pass"],
                 "last_per_capsule": vv["per_capsule"]}, indent=2))
            print(f"[verilator attempt {vatt}] {vv['n_passed']}/{vv['n_capsules']} pass L3 (all_pass={vv['all_pass']})")
            # In CONTINUOUS mode the round count is not a terminator anywhere, including here: passing an
            # effectively unbounded cap keeps the decision to 'done' or 'iterate', so an L3 that has not
            # passed yet keeps being worked instead of stopping because an arithmetic budget expired.
            _cap = a.max_rounds if a.schedule == "rounds" else (rnd + 1_000_000)
            _decision = _l3_barrier_decision(vv["all_pass"], rnd, _cap)
            if _decision == "done":
                break                                    # ONLY success exit
            if _decision == "budget":
                print(f"[verilator] round budget exhausted at L3 ({vv['n_passed']}/{vv['n_capsules']}) — "
                      f"stopping honestly (not converged; reason=max_rounds, NOT a barrier timeout)")
                break
            # 'iterate' — NON-terminal: redacted L3 failures back + clear a false READY (agent must earn it again) + fix round
            (ws / "qa").mkdir(exist_ok=True)
            (ws / "qa" / "verdict.json").write_text(json.dumps(
                _l3_fix_verdict(vv, vatt), indent=2))
            (ws / "submission" / READY_MARKER).unlink(missing_ok=True)
            _fr = time.time()
            _fix_rc = 0
            _fix_tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
            try:
                _fix_rc, _fix_tpath = launch_agent(
                    ws, run_dir, a.model, a.effort, a.sandbox, bundle, rnd, a.round_timeout, arm=arm)
            except subprocess.TimeoutExpired:
                _fix_rc = 124
                print("[verilator fix-round] agent TIMEOUT")
            active_wall_s += time.time() - _fr
            # A fix round may replace the compiler after the previously-conformant round.  Recompute from
            # THIS transcript + THIS submission immediately; retaining the old True would let prior tool
            # use mask a hand-authored/non-RTL final workflow.
            _latest_authoring_tpath = _fix_tpath
            conf, workflow_conformant = _workflow_conformance(
                _fix_tpath, ws / "submission", arm, _endpoint_kind, _resolved_tools())
            _fix_bad = [k for k, v in (conf.get("checks") or {}).items() if v is False]
            if _fix_bad:
                print(f"[verilator fix-round {rnd}] NOT CONFORMANT — failing: "
                      f"{', '.join(_fix_bad)} (formal completion remains blocked)")
            _fix_effort = ET.parse_transcript(
                _fix_tpath, billing_mode=_billing_mode(a.model),
                trust_cli_cost=_trust_cli_cost(a.model))
            rounds_summary.append({
                "round": rnd, "mode": "verilator_fix", "agent_rc": _fix_rc,
                "conformance": conf, "workflow_conformant": workflow_conformant,
                "authoring_complete": _authoring_complete(),
                "l3_attempt_repaired": vatt,
                "tokens_total": _fix_effort.get("tokens_total"),
                "tokens_output": _fix_effort.get("tokens_output"),
                "tokens_cached": _fix_effort.get("tokens_cached"),
                "tokens_input": _fix_effort.get("tokens_input"),
                "tokens_reasoning": _fix_effort.get("tokens_reasoning"),
                "tool_calls": _fix_effort.get("tool_calls"),
                "estimated_cost_usd": _fix_effort.get("estimated_cost_usd"),
            })
            rnd += 1
            _checkpoint(rnd)
        # realistic (abc2): the verilator barrier IS the definition of done. Make it drive `converged`
        # (so the watchdog stops correctly + finalize runs) rather than the spike gate.
        if _EXPERIMENT == "realistic":
            verdict["all_pass"] = bool(vv["all_pass"])
            verdict["n_passed"], verdict["n_capsules"] = vv["n_passed"], vv["n_capsules"]
            _checkpoint(rnd)

    # The authoring tree may have changed in an L3 fix round (or this process may have resumed from an
    # older checkpoint).  Refresh once more before treating it as converged.  No transcript means no
    # persisted tool evidence and therefore fails closed.
    if _latest_authoring_tpath is not None:
        final_conformance, workflow_conformant = _workflow_conformance(
            _latest_authoring_tpath, ws / "submission", arm, _endpoint_kind, _resolved_tools())
    else:
        final_conformance = {"conformant": False, "error": "no authoring transcript"}
        workflow_conformant = False
    _checkpoint(rnd)

    # On convergence, give the agent ONE bounded turn to finalize its own REPORT.md/docs to the
    # VERIFIED result (the relaunch design means the converging round's agent never saw the passing
    # verdict, so its self-reported status lags by one round). Guarantees the frozen report matches.
    finalize = None
    if _authoring_complete():
        print("[finalize] converged — running bounded report-finalize turn")
        _fin_start = time.time()
        finalize = finalize_report(ws, run_dir, a.model, a.effort, a.sandbox, bundle, arm, verdict,
                                   min(a.round_timeout, 900))
        active_wall_s += time.time() - _fin_start  # finalize is active work
        print(f"[finalize] regrade_all_pass={finalize['regrade_all_pass']} "
              f"restored={finalize['restored_after_regression']} "
              f"stamped={finalize['status_line_stamped_by_driver']}")
        # Finalize is instructed to edit docs only, but the submission is not trusted until checked.  Scan
        # the final bytes against the last authoring workflow so code changes/regressions cannot inherit a
        # pre-finalize conformance True.
        final_conformance, workflow_conformant = _workflow_conformance(
            _latest_authoring_tpath, ws / "submission", arm, _endpoint_kind, _resolved_tools())
        _checkpoint(rnd)
    # cumulative wall = active work (rounds + finalize, across ALL invocations) + rate-limit sleeps
    wall = round(active_wall_s + rate_limit_wait_s, 3)

    # combined telemetry across all rounds + finalize (total effort to produce the deliverable)
    combined = run_dir / "transcript.jsonl"
    with open(combined, "w") as out:
        for tp in sorted((run_dir / "rounds").glob("round_*.transcript.jsonl")):
            out.write(tp.read_text())
        ftp = run_dir / "rounds" / "finalize.transcript.jsonl"
        if ftp.exists():
            out.write(ftp.read_text())
    summ = ET.parse_transcript(combined, billing_mode=_billing_mode(a.model),
                               trust_cli_cost=_trust_cli_cost(a.model))
    # active-vs-waiting split (cumulative across resume-invocations): wall = active work + rate-limit
    # sleeps; active_wall_s is time actually DOING work (agent rounds + oracle grading + finalize).
    active_wall_s = round(active_wall_s, 3)
    timing = {"wall_seconds": wall, "active_wall_s": active_wall_s,
              "rate_limit_wait_s": round(rate_limit_wait_s, 3),
              "rate_limit_waits_used": rl_waits_used, "started_at": started_at,
              "resumed": _resuming}
    ET.write_cost_yaml(summ, run_dir / "cost_time_toolcalls.yaml",
                       wall_time_seconds=wall, model=a.model, exit_code=0)
    # append the active-vs-waiting split to the cost yaml (write_cost_yaml owns the rest)
    _cy = run_dir / "cost_time_toolcalls.yaml"
    _cd = yaml.safe_load(_cy.read_text()) or {}
    _cd.update({"active_wall_s": active_wall_s, "rate_limit_wait_s": round(rate_limit_wait_s, 3),
                "rate_limit_waits_used": rl_waits_used})
    _cy.write_text(yaml.safe_dump(_cd, sort_keys=False))
    qa_summary = {
        "rounds": rounds_summary,
        "authoring_complete": _authoring_complete(),
        "numeric_all_pass": bool(verdict.get("all_pass", False)),
        "workflow_conformant": workflow_conformant,
        "final_conformance": final_conformance,
        "cost_capped": cost_capped,   # stopped by the batch dollar ceiling, not convergence/max_rounds
        "n_rounds": len(rounds_summary), "wall_seconds": wall, "finalize": finalize,
        "timing": timing,
    }

    # Additionally sink this run's agentic telemetry into the shared aet store (opt-in,
    # MERLIN_AET_SINK=1) so it shows up in `aet spend` / `aet plot` across experiments. This is
    # purely additive — the existing experiment_tokens cost yaml above stays authoritative.
    from merlin.targetgen import aet_bridge as AB
    if AB.aet_sink_enabled():
        AB.emit_to_aet(run_dir=run_dir, run_id=a.run_id, method=arm, model=a.model,
                       target=_te().target, suite="capsule-bench",
                       transcript_paths=[combined],
                       billing_mode=_billing_mode(a.model))

    # capture the final submission and run the OFFICIAL public+hidden record
    wsub = ws / "submission"
    official_grade = {"complete": False, "grader_returncode": None,
                      "manifest": str(run_dir / "run_manifest.yaml"),
                      "failures": ["submission_missing"]}
    if wsub.exists():
        dst = run_dir / "submission"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(wsub, dst, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    if wsub.exists() and workflow_conformant:
        # Last gate before the official process performs its public grade + freeze + hidden grade.  The
        # authoring loop can last across worktree edits and quota-window resumes, so setup-time hashing is
        # not enough: recompute every treatment byte and the private snapshot content now, fail closed on
        # drift, and hand the grader only the already-frozen hidden path.
        _current_scope = _task_runtime_scope(_te(), a.sandbox)
        _official_hidden_dir = None
        if a.sandbox == "bwrap":
            _BWS.verify_bundle_snapshot(ws, bundle, repo=C.REPO)
            _verified_snapshot_root = _BWS.bundle_snapshot_root(ws).resolve(strict=True)
            _expected_hidden_dir = _hidden_snapshot_dir(
                _verified_snapshot_root, _te(), C.REPO)
        else:
            _expected_hidden_dir = None
        try:
            _official_hidden_dir = _verify_persisted_run_inputs(
                _environment_record,
                identity={"run_id": a.run_id, "arm": arm, "sandbox": a.sandbox,
                          "bundle_id": bundle["bundle_id"]},
                task_scope=_current_scope, ws=ws, run_dir=run_dir,
                bundle_dir=bundle_dir, resolved_tools=_resolved_tools(),
                expected_hidden_dir=_expected_hidden_dir)
        except RuntimeError as exc:
            raise RuntimeError(f"official grade refused: {exc}") from exc
        grade_cmd = [sys.executable, str(C.EXP / "scripts" / "grade_agent_run.py"),
                     "--run-dir", str(run_dir), "--arm", arm, "--model", a.model,
                     "--capsules", str(_pilot_subset())]
        # bwrap formal runs consume the immutable operator-only copy, never the live worktree/symlink.
        # The unsandboxed mode is an explicit untrusted diagnostic and retains its historical live path.
        if _official_hidden_dir is None:
            _cc = Path(_te().capsule_corpus)
            _official_hidden_dir = (_cc if _cc.is_absolute() else (C.REPO / _cc)).parent / "hidden"
        if _official_hidden_dir.is_dir():
            grade_cmd += ["--hidden-capsules", str(_official_hidden_dir)]
        if a.no_oracle:
            grade_cmd.append("--no-oracle")
        if a.skip_hidden:
            grade_cmd.append("--skip-hidden")
        grade_proc = subprocess.run(grade_cmd, cwd=str(C.REPO))
        official_grade = _official_grade_result(grade_proc.returncode, run_dir)
    elif wsub.exists():
        official_grade["failures"] = ["workflow_nonconformant"]
    formal_complete = _formal_completion(bool(verdict.get("all_pass")), workflow_conformant,
                                         official_grade["complete"])
    qa_summary.update({"converged": formal_complete, "formal_complete": formal_complete,
                       "official_grade": official_grade})
    (run_dir / "qa_loop_summary.yaml").write_text(yaml.safe_dump(qa_summary, sort_keys=False))
    # The detailed usage record is written before the potentially long outer grade so the grader can
    # include it in its manifest.  Correct its launcher exit code once the authoritative grade exists.
    _cd = yaml.safe_load(_cy.read_text()) or {}
    _cd["exit_code"] = 0 if formal_complete else 1
    _cy.write_text(yaml.safe_dump(_cd, sort_keys=False))
    # auto-emit a per-run timing decomposition (best-effort; the detailed cross-arm view is the analysis
    # script, but this gives each run its own think+gen / tool / sim split + CIRCT-gate skips at finish).
    try:
        _emit_run_timing(run_dir, rounds_summary)
    except Exception as e:
        print(f"[timing] decomposition skipped: {e}")
    print(f"\nrun complete: {run_dir}  numeric_all_pass={verdict.get('all_pass')} "
          f"workflow_conformant={workflow_conformant} "
          f"official_grade_complete={official_grade['complete']} formal_complete={formal_complete} "
          f"rounds={len(rounds_summary)}")
    return 0 if formal_complete else 1


def _emit_run_timing(run_dir: Path, rounds_summary: list) -> None:
    """Write run_dir/timing_detailed.json: think+gen (dedup'd result events) vs tool/sim, + CIRCT skips."""
    import glob as _glob
    api_ms = total_ms = 0
    # dedup result events per round (retried/resumed rounds emit multiple 'result's — take the LAST per file)
    for tp in sorted(_glob.glob(str(run_dir / "rounds" / "round_*.transcript.jsonl"))):
        last = None
        for ln in Path(tp).read_text(errors="ignore").splitlines():
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("type") == "result":
                last = o
        if last:
            api_ms += last.get("duration_api_ms", 0) or 0
            total_ms += last.get("duration_ms", 0) or 0
    # CIRCT gate skips (CIRCT arm)
    gate = run_dir / "circt_gate_log.jsonl"
    skips = ran = 0
    if gate.is_file():
        for ln in gate.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            skips += int(bool(r.get("sim_skipped"))); ran += int(not r.get("sim_skipped"))
    (run_dir / "timing_detailed.json").write_text(json.dumps({
        "think_generate_s": round(api_ms / 1000, 1),
        "tool_and_wait_s": round(max(0.0, total_ms - api_ms) / 1000, 1),
        "think_pct": round(100 * api_ms / max(total_ms, 1), 1),
        "circt_gate": {"sims_skipped": skips, "sims_run": ran},
        "note": "result events dedup'd per round (retries not double-counted). think+gen=duration_api_ms.",
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
