"""Agent-invokable SELF-CHECK runner — the realistic dev test loop (given to ALL arms, unlimited use).

This is the agent's own test harness: it builds the agent's package (its 4 CLI entrypoints), emits the
command buffer for each PUBLIC capsule, runs it on a chosen simulator, compares the output to the
harness's internally-computed golden, and prints a REDACTED verdict — pass/fail + mismatch_count + the
failing tier/plane, **never the golden values**. The agent may call this as many times as it wants, on
any subset, in parallel.

Iterate on spike (fast, same integer numerics), then clear the BARRIER on verilator/VCS (cycle-accurate
RTL). A dialect is "done" when every public capsule passes on verilator/VCS. Cycles are NOT a criterion
here — only functional/numerical correctness.

  # fast functional iteration (seconds/capsule):
  python agent_selfcheck.py --sim spike --capsules all
  # the real barrier (cycle-accurate RTL; minutes/capsule; runs in parallel):
  python agent_selfcheck.py --sim verilator --capsules all --workers 8
  # focus on the ones still failing:
  python agent_selfcheck.py --sim verilator --capsules B3_conv2d_im2col_i8,A1_mvin_mvout

What it does NOT reveal: golden output tensors, hidden capsules, or the reference oracle. It tells you
WHETHER and roughly WHERE you are wrong (mismatch_count, failing plane) — not the answer.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path


def _strip_build_state(root: Path) -> None:
    """Delete ALL cmake/ninja build state under `root` so a graded copy configures from scratch in its own
    absolute path. `ignore_patterns("build")` only drops a dir literally named 'build'; a stale CMakeCache /
    ninja state in any other location pins the ORIGINAL temp source dir and makes cmake error
    ('source does not match cache used to generate cache'). Stripping these guarantees a clean, relocatable
    build (the abc9 baseline 'L3 0/20 build' bug)."""
    for p in list(root.rglob("CMakeCache.txt")) + list(root.rglob("CMakeFiles")) \
            + list(root.rglob("build.ninja")) + list(root.rglob(".ninja_deps")) \
            + list(root.rglob(".ninja_log")) + list(root.rglob("cmake_install.cmake")):
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        except Exception:
            pass

# operator-side modules (this tool is NOT part of the graded submission; the submission must stay
# integrity-clean and never import merlin — but this harness may, exactly like a vendor's test rig)
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "merlin" / "python"))
import qa_check as _qc                                   # redaction helper (no golden leak)
from merlin.targetgen import capsule_grade as CG         # build+run+compare
from merlin.targetgen import capsule_runner as CR        # tier adapters

# the PUBLIC capsule set (with goldens — operator-side; output is redacted before the agent sees it),
# DERIVED per-target from the descriptor's capsule_corpus (atlas fp8/L3, gemmini i8/L2) — no committed
# gemmini leak. Falls back to the legacy committed set if the descriptor can't be resolved.
def _public_capsules() -> Path:
    try:
        import _common as _C
        from merlin.targetgen.contract.materialize import public_capsules_for
        from merlin.targetgen.target_experiment import load_target_experiment
        return public_capsules_for(load_target_experiment(_C.EXP / "target_experiment.yaml"))
    except Exception:  # noqa: BLE001 — keep the self-check usable without a resolvable descriptor
        return _HERE / "full_public_capsules"


PUBLIC_CAPSULES = _public_capsules()


def _target_sim_via() -> tuple[str, str | None]:
    """The (target, sim_via) being self-checked, from the descriptor (so CG.grade builds ITS RunnerConfig
    — e.g. an external_backend target — not a hardcoded default, and the oracle tiers resolve from the
    target's contract). Falls back to the active target (from _common) with an unknown sim_via if the
    descriptor can't be parsed."""
    import _common as _C
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        return te.target, te.sim_via
    except Exception:  # noqa: BLE001
        return _C.TARGET, None


def _target() -> str:
    return _target_sim_via()[0]


SIM_TIER = {"spike": "L2", "verilator": "L3", "vcs": "L4"}


def _adapters(sim: str, target: str, sim_via: str | None) -> tuple[dict, str]:
    """Resolve the self-check oracle tiers from the TARGET's contract (target-agnostic, mirrors the driver
    grade). A chipyard target (gemmini) exposes the spike/verilator/vcs ladder selectable via --sim; any
    other target grades on its OWN contract-derived RTL tier (atlas external_backend -> the program oracle;
    an arc target -> the RTL-derived arc cosim), where --sim is not applicable. Routing atlas here through
    the hardcoded spike/verilator adapters ran the gemmini/RVV lowering path and crashed (AW4)."""
    if sim_via == "chipyard":
        ad = {"L2": CR._spike_verilator_adapter("spike", target)}
        if sim in ("verilator", "vcs"):
            ad["L3"] = CR._spike_verilator_adapter("verilator", target)
        if sim == "vcs":
            try:
                from merlin.targetgen import vcs_adapter as VA  # optional, config-gated
                ad["L4"] = VA.adapter()
            except Exception:
                print("  (vcs adapter unavailable in this environment — falling back to verilator L3)", file=sys.stderr)
                sim = "verilator"
        return ad, sim
    # non-chipyard target: its contract-resolved tiers (arc / program oracle); --sim does not apply.
    return CR.oracle_adapters(target, sim_via), sim


def _log_telemetry(out: dict, capsules_arg: str) -> None:
    """Append ONE operator-side dev-trajectory line per self-check (set by the driver via $SELFCHECK_LOG).
    Records EVERY verdict — including the degenerate build_failed / no_results ones — so a blind or
    flat-lining loop is visible in the operator log, not only in the agent's transcript. Best-effort."""
    import os, time as _t
    log_path = os.environ.get("SELFCHECK_LOG")
    if not log_path:
        return
    try:
        t0 = float(os.environ.get("SELFCHECK_T0", "0") or 0)
        rows = out.get("per_capsule") or []
        line = {"wall_offset_s": round(_t.time() - t0, 1) if t0 else None,
                "sim": out.get("sim"), "barrier_tier": out.get("barrier_tier"),
                # The EFFECTIVE barrier each capsule was scored against, distinct from the declared one.
                # When they differ the declared tier never ran, and scoring against it silently fails
                # every capsule -- the atlas defect that hid behind 110 identical 0/11 lines. Recording
                # both makes that visible in the log instead of only in a per-row field nobody reads.
                "barrier_used": sorted({r.get("barrier_tier") for r in rows if r.get("barrier_tier")}) or None,
                "capsules": capsules_arg, "n_passed": out.get("n_passed"),
                "n_capsules": out.get("n_capsules"), "all_pass": out.get("all_pass"),
                "build_failed": out.get("build_failed", False), "no_results": out.get("no_results", False),
                "failing": [r["capsule"] for r in rows if not r.get("pass")]}
        with open(log_path, "a") as lf:
            lf.write(json.dumps(line) + "\n")
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agent self-check runner (redacted; spike/verilator/vcs).")
    ap.add_argument("--submission", default="submission", help="path to your package (with manifest.yaml)")
    ap.add_argument("--sim", choices=["spike", "verilator", "vcs"], default="spike")
    ap.add_argument("--capsules", default="all", help="'all' or comma-separated capsule names")
    ap.add_argument("--workers", type=int, default=8, help="parallel sim workers (verilator/vcs)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="", help="optional: also write the redacted JSON here")
    a = ap.parse_args(argv)

    sub = Path(a.submission)
    if not (sub / "manifest.yaml").exists():
        print(json.dumps({"error": f"no {sub}/manifest.yaml — build your package first"})); return 2
    if not PUBLIC_CAPSULES.is_dir():
        print(json.dumps({"error": "public capsule set not found"})); return 2

    # FAITHFUL FEEDBACK: grade an ISOLATED COPY of the submission (no bench_contract / repo files sitting
    # next to it), exactly like the real driver grade (cand_00). Otherwise the self-check runs the agent's
    # entrypoints from a workspace where bench_contract IS present, so an agent reading a package-relative
    # repo path (e.g. merlin/contract/schemas/...) PASSES self-check but the standalone grade CRASHES — the
    # self-check would lie. EXCLUDE build/ exactly like the driver grade (a copied CMakeCache has absolute
    # paths to the original location and would break) -> a from-scratch build, same as the grade (fast: the
    # LLVM/MLIR is prebuilt, the OOT project is small).
    import tempfile as _tf2
    _isodir = Path(_tf2.mkdtemp(prefix="selfcheck_iso_"))
    shutil.copytree(sub, _isodir / "submission",
                    ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    sub = _isodir / "submission"
    _strip_build_state(sub)   # every grade builds from scratch in its OWN path — see helper

    _tgt, _sim_via = _target_sim_via()
    adapters, sim = _adapters(a.sim, _tgt, _sim_via)
    # barrier = the deepest resolved RTL tier: chipyard maps from --sim; any other target uses its
    # single contract-derived tier (atlas -> L3 program oracle), so read it from the adapters.
    barrier_tier = SIM_TIER[sim] if _sim_via == "chipyard" else max(adapters) if adapters else "L3"
    # subset selection (operator-side capsule dirs)
    want = None if a.capsules == "all" else set(s.strip() for s in a.capsules.split(",") if s.strip())
    import tempfile
    runs_root = Path(tempfile.mkdtemp(prefix="selfcheck_"))

    # Subset filtering that ACTUALLY limits the grade: CG.grade runs EVERY capsule under capsules_root, so
    # the `want` output-filter below alone would still grade all 20 (a "1-capsule verilator" check = ~1hr).
    # Build a temp capsules_root symlinking only the requested capsule dirs -> grade runs just the subset.
    caps_root = PUBLIC_CAPSULES
    if want:
        caps_root = Path(tempfile.mkdtemp(prefix="selfcheck_caps_"))
        missing = [n for n in sorted(want) if not (PUBLIC_CAPSULES / n / "capsule.yaml").exists()]
        if missing:
            print(json.dumps({"error": f"unknown capsule(s): {missing}"})); return 2
        for n in sorted(want):                          # copy (not symlink): rglob won't recurse symlinked dirs
            shutil.copytree(PUBLIC_CAPSULES / n, caps_root / n)

    # build + run + compare (parallel); CG.grade handles the agent's 4 entrypoints + the tier ladder
    try:
        _score = CG.grade(str(sub), capsules_root=str(caps_root), runs_root=str(runs_root),
                          labels={"public", "dev"}, contract=str(_REPO / "merlin/contract"),
                          oracle_adapters=adapters, timeout=a.timeout, max_workers=a.workers,
                          target=_tgt)
    except Exception as e:
        print(json.dumps({"error": f"grade failed: {str(e)[:300]}"})); return 1

    # SURFACE a build / integrity failure that prevented ANY capsule from running. Without this the agent
    # only sees n_capsules=0 with no reason and (as happened) misreads it as a "stubbed grader". The build
    # stderr is compiler/cmake output — it contains no golden tensors, so it is safe to show.
    if isinstance(_score, dict) and _score.get("failure"):
        f = _score["failure"]
        out = {"sim": sim, "barrier_tier": barrier_tier, "n_passed": 0, "n_capsules": 0,
               "all_pass": False, "per_capsule": [],
               "build_failed": True, "failure_plane": f.get("plane"),
               "failure_category": f.get("category"), "detail": f.get("detail"),
               "note": "Your package did NOT build/scan, so ZERO capsules ran — this is NOT a stubbed "
                       "grader. Fix the build first (see 'detail' for the cmake/compiler stderr). NOTE: "
                       "the grade builds an ISOLATED COPY of your package WITHOUT the build/ tree, so your "
                       "manifest 'build' must support a CLEAN configure (provide a 'configure' step, e.g. "
                       "cmake -S <src> -B <build>, not only 'cmake --build'); build commands run from the "
                       "package root, so relative paths and $MLIR_DIR/$LLVM_DIR/$CM env vars are honored."}
        txt = json.dumps(out, indent=2); print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        _log_telemetry(out, a.capsules)
        return 1

    # FULL developer visibility: read each capsule_result.json directly (not the over-redacted helper)
    # and surface everything the agent's OWN run produced — its command buffer, decoded RoCC trace +
    # instruction counts, full numeric diagnostics (max_abs_diff, mismatch_count, first-mismatch INDEX
    # and the agent's OWN value), sim console, and the exact failure. The ONLY thing withheld is the
    # golden EXPECTED value (the answer key): we drop `first_mismatch.expected`. The agent's own
    # artifacts are copied to ./selfcheck_out/<capsule>/ so it can inspect/diff them like a real dev.
    out_dir = Path("selfcheck_out"); out_dir.mkdir(exist_ok=True)
    # Read results from the TARGET'S OWN suite dir. run_capsule writes under cfg.suite
    # (e.g. atlas-capsule-bench); globbing the gemmini SUITE literal here made every non-gemmini
    # self-check return n_capsules:0 with per_capsule:[] — the agent's feedback loop went blind while
    # the driver's in-memory grade was correct (the atlas 0/11 blind-loop bug).
    cb_root = runs_root / "runs" / CR.suite_for(_tgt)
    rows, npass = [], 0
    for cr in sorted(cb_root.glob("*/capsule_result.json")) if cb_root.exists() else []:
        try:
            d = json.loads(cr.read_text())
        except Exception:
            continue
        name = d.get("capsule", cr.parent.name)
        if want and name not in want:
            continue
        tiers = {t: (v or {}).get("status") for t, v in (d.get("tiers") or {}).items()}
        bar = tiers.get(barrier_tier)
        bar_used = barrier_tier
        if bar is None:
            # The declared barrier never ran for this target, so scoring against it marks EVERY capsule
            # failed no matter what the grade said. Measured on atlas: the barrier resolved to L4 (the
            # lexicographic max of a wider adapter set) while its results carry only {L0 skipped, L1
            # skipped, L2 pass} -- so 110 consecutive self-checks reported 0/11 while the operator grade
            # of the same submission was 10/11. An agent told it fails everything cannot converge; it
            # rewrites working code. Fall back to the deepest tier that actually produced a verdict.
            # This can only reclassify a capsule the grade RAN and PASSED: `status == "pass"` is still
            # required below, and a 'skipped' tier is never treated as a barrier.
            ran = [k for k, v in tiers.items() if v not in (None, "skipped")]
            if ran:
                bar_used = max(ran)
                bar = tiers.get(bar_used)
        passed = (d.get("status") == "pass") and (bar == "pass")
        npass += int(passed)
        # numeric diagnostics — keep ALL stats; redact ONLY the golden expected value
        num = dict(d.get("numeric") or {})
        fm = num.get("first_mismatch")
        if isinstance(fm, dict):
            num["first_mismatch"] = {"output": fm.get("output"), "index": fm.get("index"),
                                     "your_value": fm.get("observed")}  # 'expected' (golden) withheld
        # the agent's OWN emitted artifacts -> copy out for inspection + summarize the trace
        gen = cr.parent / "generated"
        own = {}
        for art in ("command_buffer.json", "instruction_trace.json", "lowered.llvm.mlir"):
            src = gen / art
            if src.exists():
                dst = out_dir / name / art
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst); own[art] = str(dst)
        tr = {}
        try:
            t = json.loads((gen / "instruction_trace.json").read_text())
            tr = t.get("summary") or {}
        except Exception:
            pass
        # sim console (the agent's own run output)
        console_tail = None
        for lg in (cr.parent / "artifacts").glob("*_console.log") if (cr.parent / "artifacts").is_dir() else []:
            console_tail = lg.read_text()[-800:]
        row = {"capsule": name, "pass": passed, "barrier_tier": bar_used,
               "barrier_declared": barrier_tier, "barrier_status": bar}
        if not passed:
            # FULL debug detail ONLY for a FAILING capsule (the one you are working). A passing capsule's
            # diff stats / trace dump / console tail are noise that re-inflates the agent's context every
            # round (the self_check output is re-fed each turn) — the pass flag is all that's needed for it.
            row.update({
                "tiers": tiers,
                "numeric": num,                               # full diff stats, reference value redacted
                "trace_summary": tr,                          # YOUR instruction counts (mvin/mvout/compute/…)
                "trace_check": d.get("trace_check"),
                "failure": d.get("failure"),                  # full plane + detail
                "your_artifacts": own,                        # copied to ./selfcheck_out/<capsule>/
                "sim_console_tail": console_tail,
            })
        rows.append(row)
    n = len(rows)
    # n==0 here means the grade returned WITHOUT a top-level build failure yet wrote no capsule_result
    # under cb_root — a harness/path problem (not "all clear", not a stubbed grader). Say so loudly with
    # the paths, so it can never again be silently misread as an empty-but-fine verdict.
    if n == 0:
        out = {"sim": sim, "barrier_tier": barrier_tier, "n_passed": 0, "n_capsules": 0,
               "all_pass": False, "per_capsule": [], "no_results": True,
               "caps_discovered": sum(1 for _ in PUBLIC_CAPSULES.glob("*/capsule.yaml")),
               "results_root": str(cb_root),
               "note": "HARNESS ISSUE: the build did not fail, but ZERO capsule results were found at "
                       "results_root. This is NOT 'all clear' and NOT a stubbed grader — it means the "
                       "grade wrote results somewhere this reader did not look. Report this; it is not "
                       "something your dialect can fix."}
        txt = json.dumps(out, indent=2); print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        _log_telemetry(out, a.capsules)
        return 1
    out = {"sim": sim, "barrier_tier": barrier_tier, "n_passed": npass, "n_capsules": n,
           "all_pass": npass == n and n > 0, "per_capsule": rows,
           "note": f"Self-check on {sim} ({barrier_tier}). You see EVERYTHING your dialect produced — "
                   "command buffer, decoded trace + instruction counts, sim console, and your artifacts "
                   "copied to ./selfcheck_out/. The diff stats (mismatch_count, magnitudes) are YOUR "
                   "output measured against the operation's own definition, which you can reproduce from "
                   "the declared inputs — there is no answer key; the reference output values are withheld "
                   "so you debug from your own intent, as in real bring-up. 'done' = all public pass on "
                   "verilator/VCS; cycles are not a criterion. (Movement ops legitimately have 0 matmuls.)"}
    txt = json.dumps(out, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    _log_telemetry(out, a.capsules)   # one operator-side dev-trajectory line (all verdicts, incl. degenerate)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
