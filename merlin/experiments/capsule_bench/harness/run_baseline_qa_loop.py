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
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

import _common as C
import run_agent_experiment as RX  # reuse bundle/workspace/bwrap primitives
import _ratelimit as RL  # five-hour rate-limit detection + reset epoch (shared)

SCRIPTS = Path(__file__).resolve().parent  # this scripts/ dir (selfcheck_broker.py, selfcheck_shim.py)


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
from merlin.targetgen import experiment_tokens as ET  # noqa: E402

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


def _verilator_per_capsule_timeout() -> int:
    """Per-capsule verilator timeout from the readiness-measured T_obs (generous 2x), min 900s. The
    readiness gate (readiness_check.test_oracles_endtoend) writes scripts/.oracle_timing.json; the
    launcher refuses to start without it, so a fresh measurement always backs this number."""
    import math
    try:
        t = json.loads((C.EXP / "scripts" / ".oracle_timing.json").read_text())["verilator_per_capsule_s"]
        return max(900, int(math.ceil(2 * float(t))))
    except Exception:
        return 1200
_EXPERIMENT = "full"    # set in main(): 'full' (abc1) or 'realistic' (abc2, whole-repo + self-check)
_ARM = ""               # set in main(): the arm being run (enforces the per-arm language mandate)
READY_MARKER = "READY_FOR_BARRIER"  # realistic: agent drops submission/<this> to self-declare done
# Merlin-arm-only docs staged into the workspace alongside the (shared) graded task.
MERLIN_WS_DOCS = ("TASK_ADDENDUM.md", "ALLOWED_MERLIN_TOOLS.md", "MERLIN_PROVENANCE_TEMPLATE.md")


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


def _manifest():
    """This target's capability manifest (endpoint kind + sim tiers) — the second input render_prompt
    needs. Derived from the committed target_contract, so any target's runner works unchanged."""
    from merlin.targetgen.target_experiment import load_capability_manifest
    return load_capability_manifest(C.TARGET)


def answer_files() -> list[Path]:
    """Every answer-bearing GOLDEN file the agent must NOT see — DERIVED from the declared capsule corpus
    (+ its sibling corpora) via the shared sandbox module, not a hard-coded isa/layers/model_slices list.
    Hidden capsules / oracle / grader / memory are masked by the shared answer-mask pass (bwrap_cmd)."""
    from merlin.targetgen.sandbox import golden_files
    return golden_files(_te())


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
    (bwrap is unusable here: the Bun-based `claude` binary crashes under it — SIGILL / FailedToOpen-
    Socket — so filesystem isolation is by construction + a post-run transcript audit, not bwrap.)

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
        # so nested / non-.yaml / non-_g0 answers do not escape the copy drop (parity with the sandbox mask
        # in answer_surfaces.golden_files).
        return n.startswith("golden.") or n.startswith("expected_command_buffer")

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


def audit_transcript(tpath: Path, arm: str = "raw_baseline") -> dict:
    """Flag genuine READS of withheld answer/grader/oracle paths AND oracle USE in agent-authored
    code (defence-in-depth beyond the masked workspace). Self-scans of the submission and bare path
    mentions are NOT flagged. Arm-aware: the merlin arm's allowed tools are not treated as cheats."""
    hits = []
    if not tpath.exists():
        return {"clean": True, "hits": [], "note": "no transcript"}
    path_tokens = _path_tokens(arm)
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
                    hits.append({"tool": name, "kind": "path_read", "token": tok, "input": fp[:200]})
            elif name == "Bash":
                cmd = inp.get("command") or ""
                tok = _is_answer_read(cmd, path_tokens)
                if tok:
                    hits.append({"tool": name, "kind": "path_read", "token": tok, "input": cmd[:200]})
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
    return {"clean": len(hits) == 0, "hits": hits}


def claude_runtime_binds() -> list[str]:
    """RO-bind the `claude` CLI runtime into the sandbox (delegates to the shared sandbox module)."""
    from merlin.targetgen.sandbox import bwrap as _BW
    return _BW.claude_runtime_binds()


def bwrap_cmd(inner: str, ws: Path, bundle: dict) -> str:
    """bwrap argv (deny-by-default) + claude runtime binds + TOOLCHAIN binds (the legit build+sim tools,
    bound back over the /scratch* masks) + the DERIVED answer-mask pass + toolchain env. The mask set now
    comes from the shared descriptor-driven answer surface (goldens/hidden/prior/oracle/grader/memory) and
    is coverage-proven by test_sandbox_isolation; masking is bundle-independent (a bind that re-exposes an
    answer surface is re-masked here, not left to the bundle's denied list)."""
    import sandbox_toolchain as TC
    from merlin.targetgen.sandbox import bwrap as _BW
    from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces as _surfaces
    parts = RX.bwrap_argv(ws, bundle) + claude_runtime_binds() + TC.toolchain_binds()
    parts = _BW.apply_answer_masks(parts, _surfaces(_te()))
    return " ".join(parts) + f" bash -c '{TC.sandbox_env(ws)} {inner}'"


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
    """Confirm the agent's view withholds answers. bwrap: probe the masked golden. none: confirm the
    COPIED workspace has no golden.yaml and no hidden capsules anywhere under it."""
    if sandbox == "bwrap":
        # Search the agent's VIEW (the bound corpus trees) for ANY readable golden/expected file —
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
        script = (f'for f in $(find {roots_sh} \\( -name "golden.*" -o -name "expected_command_buffer*" \\) '
                  f'2>/dev/null); do test -s "$f" && echo "LEAK:$f"; done; echo DONE')
        out = subprocess.run(["bash", "-c", bwrap_cmd(script, ws, bundle)],
                             capture_output=True, text=True).stdout
        leaked = [ln[len("LEAK:"):] for ln in out.splitlines() if ln.startswith("LEAK:")]
        return {"pilot_golden_visible_to_agent": "LEAK" if leaked else "OK",
                "n_answer_files_masked": len(answer_files()),
                "leaked_answer_files": leaked[:10]}
    # sandbox == none: assert no answer is reachable. (1) the bench_contract COPY (small) holds no
    # golden/hidden; (2) no workspace symlink resolves into the real merlin/contract/capsules tree
    # (which would re-expose goldens). Avoid walking the symlinked 307MB toolchain.
    bc = ws / "merlin/contract"
    goldens = [str(p) for p in bc.rglob("golden.yaml")] if bc.exists() else []
    hidden = [str(p) for p in bc.rglob("hidden") if "capsules" in str(p)] if bc.exists() else []
    real_caps = str((C.REPO / "merlin/contract" / "capsules").resolve())
    bad_links = []
    for root, dirs, _files in os.walk(ws):  # followlinks=False: inspect, don't descend, symlinks
        for d in dirs:
            p = Path(root) / d
            if p.is_symlink() and str(p.resolve()).startswith(real_caps):
                bad_links.append(str(p))
    leak = bool(goldens or hidden or bad_links)
    _g, spec = _corpus_probe_paths()               # a real corpus spec (derived), for the presence flag
    return {"pilot_golden_visible_to_agent": "LEAK" if leak else "OK",
            "goldens_in_workspace": goldens[:10], "hidden_in_workspace": hidden[:10],
            "symlinks_into_capsules": bad_links[:10], "spec_present": spec.exists()}


def _build_task(arm: str, ws: Path, run_dir: Path) -> None:
    """Stage the workspace TASK.md. Both arms get the IDENTICAL graded contract (TASK_pilot.md). The
    merlin arm appends TASK_ADDENDUM.md (merlin-specific tool guidance + provenance ask) and stages
    the merlin-only docs into the workspace so the agent can read them. The graded pilot contract is
    never altered — the addendum only adds allowances, so grading stays apples-to-apples."""
    if _EXPERIMENT == "realistic":
        # whole-repo + self-check tool + self-paced READY marker; TASK_realistic is self-contained,
        # so skip the abc1 "20-public / verilator-checkpoint" addendum appended below.
        ws_task = ws / "TASK.md"
        # A target that ships a hand-authored realistic task uses it (gemmini); a descriptor-only target
        # (e.g. atlas) has none, so fall back to the GENERATED target-agnostic prompt — exactly what the
        # 'full' branch below already does. render_prompt is the COMPLETE per-arm task (incl. the seam menu
        # for the assisted/CIRCT arms), so the bundle STARTER_PROMPT is not re-appended in that case.
        _task_md = C.EXP / "task" / "TASK_realistic.md"
        _generated_task = not _task_md.is_file()
        if _generated_task:
            from merlin.targetgen.generate_prompt import render_prompt
            body = render_prompt(_te(), _manifest(), "realistic", arm)
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
    # diff). The runtime-only operational blocks below (language mandate + the 20-public/L3 scope note) are
    # still appended: they carry run-specific numbers the target-agnostic template deliberately omits.
    from merlin.targetgen.generate_prompt import render_prompt
    pilot = render_prompt(_te(), _manifest(), _EXPERIMENT, arm)
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
    _loop = min(CR.qa_loop_adapters(_te().target, _te().sim_via) or {"L3": 1})
    _ckpt = max(CR.qa_checkpoint_adapters(_te().target, _te().sim_via) or {"L3": 1})
    pilot += (
        "\n\n## Scope & grading (READ THIS)\n"
        "- You are graded on the **20 public capsules** (isa A0–A7, layers B0–B4, model_slices C0–C6). "
        "Iterate until ALL of them pass. There are also **5 hidden** capsules you cannot see; aim for a "
        "general, correct dialect so it generalizes to them too (the goal is all 25).\n"
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


def launch_agent(ws: Path, run_dir: Path, model: str, effort: str, sandbox: str,
                 bundle: dict, rnd: int, timeout: int, arm: str = "raw_baseline") -> tuple[int, Path]:
    # TASK.md must live INSIDE the bound workspace: run_dir is under runs/ which bwrap tmpfs-masks,
    # so a stdin redirect from run_dir/TASK.md is invisible inside the sandbox (empty stdin).
    ws_task = ws / "TASK.md"
    if rnd == 0:
        _build_task(arm, ws, run_dir)
    # A non-Anthropic model can't drive the claude CLI (Anthropic API only) — route it to the Bedrock
    # Converse agent backend, which runs the same masked-sandbox agentic loop + a compatible transcript so
    # grading/accounting/audit are unchanged. Anthropic models fall through to the claude CLI below.
    import bedrock_agent as _BA
    if _BA.is_converse_model(model):
        return _BA.run_round(ws, run_dir, model, bundle, _te(), sandbox, rnd, timeout)
    inner = (f'claude --print --model {model} --effort {effort} '
             f'--permission-mode bypassPermissions --add-dir {ws} '
             f'--output-format stream-json --verbose < {ws_task}')
    cmd = bwrap_cmd(inner, ws, bundle) if sandbox == "bwrap" else inner
    tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    epath = run_dir / "rounds" / f"round_{rnd:02d}.stderr.log"
    # Under bwrap the oracle is masked, so the agent's self-check (agent_selfcheck.py) can't grade in-box.
    # Start the driver-side BROKER (oracle available, outside the sandbox) + stage the in-box shim, so the
    # agent gets a REDACTED on-demand self-check without the oracle ever entering its sandbox.
    broker = _start_selfcheck_broker(ws) if sandbox == "bwrap" else None
    try:
        with open(tpath, "w") as tf, open(epath, "w") as ef:
            proc = subprocess.run(["bash", "-c", cmd], cwd=str(ws), stdout=tf, stderr=ef,
                                  timeout=timeout)
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
    brokers = []
    for name, log in (("selfcheck_broker.py", "broker.log"), ("simjob_broker.py", "simjob_broker.log")):
        brokers.append(subprocess.Popen(
            [sys.executable, str(SCRIPTS / name), "--ws", str(ws)],
            stdout=open(ch / log, "w"), stderr=subprocess.STDOUT))
    return brokers


def _stop_selfcheck_broker(ws: Path, brokers) -> None:
    if not brokers:
        return
    (ws / ".qa_channel" / "STOP").write_text("stop")
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
    # hand the redacted verdict to the agent for the next round
    qa_dir = ws / "qa"
    qa_dir.mkdir(exist_ok=True)
    (qa_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    return verdict


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
    inner = (f'claude --print --model {model} --effort {effort} '
             f'--permission-mode bypassPermissions --add-dir {ws} '
             f'--output-format stream-json --verbose < {ws / "FINALIZE.md"}')
    cmd = bwrap_cmd(inner, ws, bundle) if sandbox == "bwrap" else inner
    tpath = run_dir / "rounds" / "finalize.transcript.jsonl"
    epath = run_dir / "rounds" / "finalize.stderr.log"
    rc = 0
    try:
        with open(tpath, "w") as tf, open(epath, "w") as ef:
            rc = subprocess.run(["bash", "-c", cmd], cwd=str(ws), stdout=tf, stderr=ef,
                                timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        rc = 124

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
    # PROVIDER for the agent's `claude` CLI — experiments-only, so the interactive Claude Code keeps the
    # subscription. subscription (default) = the machine's ~/.claude creds; bedrock = Claude Code's own
    # Bedrock mode (CLAUDE_CODE_USE_BEDROCK=1 + AWS creds + a Bedrock inference-profile model id).
    ap.add_argument("--provider", choices=["subscription", "bedrock"], default="subscription",
                    help="model provider for the agent CLI (experiments-only; subscription keeps ~/.claude)")
    ap.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"),
                    help="AWS region for --provider bedrock")
    ap.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""),
                    help="AWS profile (~/.aws) for --provider bedrock; else the env-var cred chain")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--round-timeout", type=int, default=14400,
                    help="per-round agent wall cap (s). Default 4h (matches launch_ab_batch): a TIGHT "
                         "cap is net-detrimental — it doesn't cut the work (fixed by difficulty), it "
                         "just forces more rounds, each adding a full grading pass + context re-read + "
                         "rate-limit-boundary exposure, and can cut a productive round mid-fix (rc=124). "
                         "The original abc runs used 4h and converged in ~1 productive round.")
    ap.add_argument("--qa-timeout", type=int, default=900)
    # Default sandbox=none: bwrap crashes the Bun-based `claude` binary in this environment. Isolation
    # under "none" is enforced by a golden-masked COPIED workspace + a post-run transcript audit.
    ap.add_argument("--sandbox", choices=["bwrap", "none"], default="none")
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
    ap.add_argument("--account-config-dir", default=os.environ.get("CLAUDE_CONFIG_DIR", ""),
                    help="CLAUDE_CONFIG_DIR for the agent's claude CLI (a different subscription account)")
    a = ap.parse_args(argv)
    arm = a.arm
    global _EXPERIMENT, _ARM
    _EXPERIMENT = a.experiment
    _ARM = arm

    # DRIVER-SIDE grade env: qa_grade/_verilator_grade run build_package (cmake) for C++ submissions in
    # THIS process's env. The conda cmake transitively needs libidn.so.11 (host has only .12) -> ensure
    # the .compat_lib shim + conda libs are on LD_LIBRARY_PATH, else the C++ build fails "libidn.so.11:
    # cannot open shared object file" and every grade is 0/0 (the abc8 rb blocker). Python arms have no
    # build step so this is a harmless no-op for them.
    _CE = "/path/to/chipyard/.conda-env"
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
        shutil.rmtree(ws_root)
    run_dir.mkdir(parents=True, exist_ok=_resuming)
    shutil.copy(C.BUNDLES / RX.ARM_BUNDLE[arm] / "input_bundle_manifest.yaml",
                run_dir / "input_bundle_manifest.yaml")

    if _have_ws:
        print(f"[resume] reusing existing workspace + submission at {ws}")
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
    mask = mask_selftest(ws, bundle, a.sandbox)
    (run_dir / "environment.yaml").write_text(yaml.safe_dump({
        "run_id": a.run_id, "arm": arm, "model": a.model, "effort": a.effort,
        "sandbox": a.sandbox, "qa_loop": True, "pilot": ["A0", "A2", "A4", "B0"],
        "workspace_path": str(ws), "workspace_copy_report": copy_report,
        "repo_sha": C.repo_sha(), "bundle_id": bundle["bundle_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "isolation_violations": viol, "denied_paths_checked": denied_names,
        "golden_mask_selftest": mask,
        "account": account_info,  # which subscription/org budget this arm drew from (provenance)
    }, sort_keys=False))
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
    rl_waits_used = 0
    active_wall_s = 0.0              # cumulative time DOING work (launch+grade) across all invocations
    rate_limit_wait_s = 0.0         # cumulative time slept waiting for five-hour window resets
    started_at = datetime.now(timezone.utc).isoformat()
    verdict = {"all_pass": False}
    rnd = 0
    if _resuming and state_p.exists():
        st = yaml.safe_load(state_p.read_text()) or {}
        rounds_summary = st.get("rounds", []) or []
        rnd = int(st.get("next_round", 0))
        verdict = {"all_pass": bool(st.get("converged", False))}
        cum = st.get("cumulative", {}) or {}
        active_wall_s = float(cum.get("active_wall_s", 0.0))
        rate_limit_wait_s = float(cum.get("rate_limit_wait_s", 0.0))
        rl_waits_used = int(cum.get("rl_waits_used", 0))
        started_at = cum.get("started_at", started_at)
        print(f"[resume] restored from checkpoint: next_round={rnd} converged={verdict['all_pass']} "
              f"active={active_wall_s:.0f}s rate_limit_wait={rate_limit_wait_s:.0f}s "
              f"waits_used={rl_waits_used}")

    def _checkpoint(next_round: int) -> None:
        state_p.write_text(yaml.safe_dump({
            "run_id": a.run_id, "arm": arm, "model": a.model, "effort": a.effort,
            "rounds": rounds_summary, "next_round": next_round,
            "converged": verdict.get("all_pass", False),
            "cumulative": {"active_wall_s": round(active_wall_s, 3),
                           "rate_limit_wait_s": round(rate_limit_wait_s, 3),
                           "rl_waits_used": rl_waits_used, "started_at": started_at},
            "last_updated": datetime.now(timezone.utc).isoformat()}, sort_keys=False))

    while rnd < a.max_rounds and not verdict.get("all_pass"):
        print(f"\n===== ROUND {rnd} =====")
        _rstart = time.time()
        try:
            rc, tpath = launch_agent(ws, run_dir, a.model, a.effort, a.sandbox, bundle, rnd,
                                     a.round_timeout, arm=arm)
        except subprocess.TimeoutExpired:
            rc, tpath = 124, run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
            print(f"[round {rnd}] agent TIMEOUT")

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
        rsum = ET.parse_transcript(tpath)
        audit = audit_transcript(tpath, arm)
        rounds_summary.append({"round": rnd, "agent_rc": rc,
                               "all_pass": verdict.get("all_pass"),
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
                               "estimated_cost_usd": rsum.get("estimated_cost_usd"),
                               "answer_access_clean": audit["clean"],
                               "audit_hits": audit["hits"]})
        active_wall_s += time.time() - _rstart
        print(f"[round {rnd}] all_pass={verdict.get('all_pass')} "
              f"{verdict.get('n_passed')}/{verdict.get('n_capsules')} "
              f"answer_access_clean={audit['clean']}")
        rnd = rnd + 1
        _checkpoint(rnd)  # next_round advances only after a completed (non-rate-limited) round
        # realistic (abc2): the agent self-paces — it self-checks via the tool and drops READY_FOR_BARRIER
        # when it believes it's done. Break to the verilator barrier on that marker OR on spike all-pass.
        ready = _EXPERIMENT == "realistic" and (ws / "submission" / READY_MARKER).exists()
        if verdict.get("all_pass") or ready:
            if ready:
                print(f"[round {rnd-1}] agent dropped {READY_MARKER} — proceeding to verilator barrier")
            break

    # --- cycle-accurate L3 (verilator) checkpoint -------------------------------------------------
    # The loop gate is spike (L2) only; this is where kernels are validated on the real RTL. Runs only
    # when the agent is "ready" (spike-converged on the 20 public). Up to VERILATOR_ATTEMPTS chances
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
        # CIRCT arm only: wrap each sim adapter with the CIRCT pre-screen gate so a structural reject
        # SKIPS the ~222 s/capsule sim (catch the bug in ~7 ms). Records skips to circt_gate_log.jsonl.
        # The plain merlin / baseline arms run the sims ungated (no CIRCT) — preserves the C-B distinction.
        if RX.ARM_BUNDLE.get(arm, "").find("rtlchecks") >= 0:
            try:
                from merlin.targetgen import circt_gate as _GATE
                _glog: list = []
                adapters = {t: _GATE.gated_adapter(adp, log=_glog, target=_te_ck.target)
                            for t, adp in adapters.items()}
                (run_dir / "circt_gate_log.jsonl").write_text("")  # reset; appended after grade
                print(f"[verilator attempt {attempt}] CIRCT sim-skip gate ACTIVE (reject ⇒ skip sim)")
            except Exception as e:
                print(f"[verilator attempt {attempt}] CIRCT gate unavailable ({e}); running ungated")
                _glog = None
        else:
            _glog = None
        try:
            _CG.grade(str(vcand), capsules_root=str(_pilot_subset()), runs_root=str(vruns),
                      labels={"public", "dev"}, contract=str(C.REPO / "merlin/contract"),
                      oracle_adapters=adapters, timeout=_verilator_per_capsule_timeout(), max_workers=8,
                      target=_te().target)
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
        return {"attempt": attempt, "all_pass": npass == nc and nc > 0, "n_passed": npass,
                "n_capsules": nc, "per_capsule": per}

    verilator_attempts: list = []
    _ready_marker = (ws / "submission" / READY_MARKER).exists()
    if verdict.get("all_pass") or (_EXPERIMENT == "realistic" and _ready_marker):
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
            _decision = _l3_barrier_decision(vv["all_pass"], rnd, a.max_rounds)
            if _decision == "done":
                break                                    # ONLY success exit
            if _decision == "budget":
                print(f"[verilator] round budget exhausted at L3 ({vv['n_passed']}/{vv['n_capsules']}) — "
                      f"stopping honestly (not converged; reason=max_rounds, NOT a barrier timeout)")
                break
            # 'iterate' — NON-terminal: redacted L3 failures back + clear a false READY (agent must earn it again) + fix round
            (ws / "qa").mkdir(exist_ok=True)
            (ws / "qa" / "verdict.json").write_text(json.dumps(
                {"stage": "verilator_checkpoint", "attempt": vatt, "all_pass": False,
                 "n_passed": vv["n_passed"], "n_capsules": vv["n_capsules"],
                 "note": "CYCLE-ACCURATE (verilator/L3) results — fix the capsules failing at L3, then "
                         "re-run agent_selfcheck (spike) clean before declaring READY again.",
                 "per_capsule": vv["per_capsule"]}, indent=2))
            (ws / "submission" / READY_MARKER).unlink(missing_ok=True)
            _fr = time.time()
            try:
                launch_agent(ws, run_dir, a.model, a.effort, a.sandbox, bundle, rnd, a.round_timeout, arm=arm)
            except subprocess.TimeoutExpired:
                print("[verilator fix-round] agent TIMEOUT")
            active_wall_s += time.time() - _fr
            rnd += 1
            _checkpoint(rnd)
        # realistic (abc2): the verilator barrier IS the definition of done. Make it drive `converged`
        # (so the watchdog stops correctly + finalize runs) rather than the spike gate.
        if _EXPERIMENT == "realistic":
            verdict["all_pass"] = bool(vv["all_pass"])
            verdict["n_passed"], verdict["n_capsules"] = vv["n_passed"], vv["n_capsules"]
            _checkpoint(rnd)

    # On convergence, give the agent ONE bounded turn to finalize its own REPORT.md/docs to the
    # VERIFIED result (the relaunch design means the converging round's agent never saw the passing
    # verdict, so its self-reported status lags by one round). Guarantees the frozen report matches.
    finalize = None
    if verdict.get("all_pass"):
        print("[finalize] converged — running bounded report-finalize turn")
        _fin_start = time.time()
        finalize = finalize_report(ws, run_dir, a.model, a.effort, a.sandbox, bundle, arm, verdict,
                                   min(a.round_timeout, 900))
        active_wall_s += time.time() - _fin_start  # finalize is active work
        print(f"[finalize] regrade_all_pass={finalize['regrade_all_pass']} "
              f"restored={finalize['restored_after_regression']} "
              f"stamped={finalize['status_line_stamped_by_driver']}")
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
    summ = ET.parse_transcript(combined)
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
    (run_dir / "qa_loop_summary.yaml").write_text(yaml.safe_dump(
        {"rounds": rounds_summary, "converged": verdict.get("all_pass", False),
         "n_rounds": len(rounds_summary), "wall_seconds": wall, "finalize": finalize,
         "timing": timing}, sort_keys=False))

    # Additionally sink this run's agentic telemetry into the shared aet store (opt-in,
    # MERLIN_AET_SINK=1) so it shows up in `aet spend` / `aet plot` across experiments. This is
    # purely additive — the existing experiment_tokens cost yaml above stays authoritative.
    from merlin.targetgen import aet_bridge as AB
    if AB.aet_sink_enabled():
        AB.emit_to_aet(run_dir=run_dir, run_id=a.run_id, method=arm, model=a.model,
                       target=_te().target, suite="capsule-bench",
                       transcript_paths=[combined])

    # capture the final submission and run the OFFICIAL public+hidden record
    wsub = ws / "submission"
    if wsub.exists():
        dst = run_dir / "submission"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(wsub, dst, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
        grade_cmd = [sys.executable, str(C.EXP / "scripts" / "grade_agent_run.py"),
                     "--run-dir", str(run_dir), "--arm", arm, "--model", a.model,
                     "--capsules", str(_pilot_subset())]
        if a.no_oracle:
            grade_cmd.append("--no-oracle")
        if a.skip_hidden:
            grade_cmd.append("--skip-hidden")
        subprocess.run(grade_cmd, cwd=str(C.REPO))
    # auto-emit a per-run timing decomposition (best-effort; the detailed cross-arm view is the analysis
    # script, but this gives each run its own think+gen / tool / sim split + CIRCT-gate skips at finish).
    try:
        _emit_run_timing(run_dir, rounds_summary)
    except Exception as e:
        print(f"[timing] decomposition skipped: {e}")
    print(f"\nrun complete: {run_dir}  converged={verdict.get('all_pass')}  rounds={len(rounds_summary)}")
    return 0 if verdict.get("all_pass") else 1


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
