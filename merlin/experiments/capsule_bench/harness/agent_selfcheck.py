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


def select_tiers(full: dict, default: dict, requested: str) -> tuple[dict, str | None]:
    """Which tiers this self-check runs: the target's cheap LOOP ladder by default, or exactly the tiers
    ``--tiers`` names, validated against everything the endpoint can REACH.

    The two maps are a deliberate distinction and were being conflated. Resolving the DEFAULT from the
    loop ladder is what keeps a sweep cheap -- a cert tier is minutes per capsule against seconds for the
    functional one, and paying it on every capsule of every sweep was 80% of an agent round. Validating
    the REQUEST against that same shrunken map is a different question, and answering it with the loop
    map made the flag useless for the one job it exists to do: on this repo's SIMT target the loop ladder
    is {L2}, so ``--tiers L2,L3`` was refused as "unreachable" by an endpoint whose full map is
    {L2, L3}.

    The same conflation silently disabled automatic promotion. ``tier_promote.resolve_tiers`` derives its
    cert tier as ``oracle_adapters - qa_loop_adapters`` -- i.e. precisely a tier the loop map never
    contains -- and the broker forwards it as ``--tiers <cert>``. So every promoted job asked for a tier
    this validation could only refuse. Promotion was not merely untested in production; it could not fire
    even in principle.

    Returns ``(adapters, error)``. An unreachable tier is still NAMED, never silently dropped: a
    self-check that quietly grades fewer tiers than it was asked for reads as a pass.
    """
    want = {t.strip().upper() for t in requested.split(",") if t.strip()}
    if not want:
        return default, None
    missing = sorted(want - set(full))
    if missing:
        return {}, (f"--tiers names {missing}, which this endpoint does not reach; "
                    f"reachable: {sorted(full)}")
    return {t: ad for t, ad in full.items() if t in want}, None


def _adapters(sim: str, target: str, sim_via: str | None) -> tuple[dict, dict, str]:
    """``(default_tiers, reachable_tiers, sim)`` from the TARGET's contract (target-agnostic, mirrors the
    driver grade).

    ``default`` is what a sweep runs when ``--tiers`` is absent: the fastest ladder the CORPUS declares.
    ``reachable`` is everything the endpoint can run, cert tiers included -- the set ``--tiers`` is
    validated against, and the set the broker's promotion draws its cert tier from. Returning only the
    first of these is what broke both (see :func:`select_tiers`).

    A chipyard target (gemmini) exposes the spike/verilator/vcs ladder selectable via --sim; any other
    target grades on its OWN contract-derived RTL tier (atlas external_backend -> the program oracle; an
    arc target -> the RTL-derived arc cosim), where --sim is not applicable. Routing atlas here through
    the hardcoded spike/verilator adapters ran the gemmini/RVV lowering path and crashed (AW4).
    """
    if sim_via == "chipyard":
        ad = {"L2": CR._spike_verilator_adapter("spike", target)}
        full = {"L2": ad["L2"], "L3": CR._spike_verilator_adapter("verilator", target)}
        if sim in ("verilator", "vcs"):
            ad["L3"] = full["L3"]
        if sim == "vcs":
            try:
                from merlin.targetgen import vcs_adapter as VA  # optional, config-gated
                ad["L4"] = full["L4"] = VA.adapter()
            except Exception:
                print("  (vcs adapter unavailable in this environment — falling back to verilator L3)", file=sys.stderr)
                sim = "verilator"
        return ad, full, sim
    # non-chipyard target: its contract-resolved tiers (arc / program oracle); --sim does not apply.
    #
    # The LOOP ladder, not the full one. `oracle_adapters` is the CHECKPOINT set -- everything the endpoint
    # can reach, cycle-accurate tiers included -- and returning it here made every self-check pay the
    # cert tier on every capsule. Measured on this repo's SIMT target that was 80% of an agent round's
    # wall clock (134 of 167 minutes across nine sweeps), of which 34% returned no verdict at all, for a
    # tier the score never reads: the DRIVER-side barrier already declines it via
    # `_cycle_accurate_checkpoint_enabled`, because no capsule in that corpus declares a cert tier
    # mandatory. Two code paths disagreeing, not a design choice.
    #
    # `qa_loop_adapters` keeps the fastest tier the CORPUS ITSELF declares, and fails closed (`{}`) rather
    # than substituting one it never asked for. `--tiers` is the way to go deeper, per capsule,
    # deliberately -- which requires the FULL map to be returned alongside, not discarded.
    _declared = None
    try:
        import _common as _C
        from merlin.targetgen.contract.materialize import declared_oracle_tiers
        from merlin.targetgen.target_experiment import load_target_experiment
        _te_ = load_target_experiment(_C.EXP / "target_experiment.yaml")
        _declared = declared_oracle_tiers(*_te_.graded_roots())
    except Exception:  # noqa: BLE001 -- no resolvable descriptor: fall back to the full ladder
        _declared = None
    full = CR.oracle_adapters(target, sim_via)
    loop = CR.qa_loop_adapters(target, sim_via, declared_tiers=_declared) if _declared else {}
    return (loop or full), full, sim


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
    ap.add_argument("--tiers", default="", metavar="L2,L3",
                    help="oracle tiers to run, comma-separated. Default is the LOOP tier this target's "
                         "corpus declares (the cheap one). Name a deeper tier to certify deliberately -- "
                         "a cert tier costs minutes per capsule against seconds for the loop tier, so it "
                         "is worth spending on a capsule that already passes, not on the whole corpus.")
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
    adapters, full, sim = _adapters(a.sim, _tgt, _sim_via)
    adapters, _err = select_tiers(full, adapters, a.tiers)
    if _err:
        print(json.dumps({"error": _err}))
        return 2
    # barrier = the deepest resolved RTL tier: chipyard maps from --sim; any other target uses its
    # single contract-derived tier (atlas -> L3 program oracle), so read it from the adapters.
    barrier_tier = SIM_TIER[sim] if _sim_via == "chipyard" else max(adapters) if adapters else "L3"
    # REPORT the engine that actually ran, not the --sim flag. --sim selects a tier only for a chipyard
    # target's spike/verilator/vcs ladder; every other target is contract-routed (oracle_adapters), so the
    # flag is inert and echoing it told the agent it was iterating on "spike" while a SIMT core's cyclotron
    # + RTL tiers did the work. A wrong engine name invites reasoning about the wrong machine (RoCC/RISC-V
    # semantics for a SIMT GPU), and SIM_TIER maps spike->L2 while the note printed the resolved L3 --
    # internally inconsistent on top of being wrong. Label only; tier SELECTION is unchanged.
    if _sim_via != "chipyard":
        sim = f"{_sim_via or 'contract-routed'} [{','.join(sorted(adapters))}]"
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
    _results = sorted(cb_root.glob("*/capsule_result.json")) if cb_root.exists() else []
    # Does the declared barrier EVER produce a verdict for this target? Two very different situations
    # look identical per-capsule, and only this whole-corpus view separates them:
    #   • the tier ran for some capsule but not this one -> this capsule genuinely fell short of the
    #     barrier and MUST fail; falling back would pass it on weaker evidence than the bar demands.
    #   • the tier ran for NO capsule -> the declared barrier is one the target cannot produce at all
    #     (a descriptor/adapter mismatch), and scoring against it fails everything regardless of merit.
    # Only the second case licenses a fallback.
    _declared_ran = False
    for _cr in _results:
        try:
            _d = json.loads(_cr.read_text())
        except Exception:
            continue
        _s = ((_d.get("tiers") or {}).get(barrier_tier) or {}).get("status")
        if _s not in (None, "skipped"):
            _declared_ran = True
            break
    for cr in _results:
        try:
            d = json.loads(cr.read_text())
        except Exception:
            continue
        name = d.get("capsule", cr.parent.name)
        if want and name not in want:
            continue
        tier_objs = {t: (v or {}) for t, v in (d.get("tiers") or {}).items()}
        tiers = {t: v.get("status") for t, v in tier_objs.items()}
        bar = tiers.get(barrier_tier)
        # Gate on the barrier tier ONLY when the capsule's own contract makes it mandatory, and only when
        # it actually produced a verdict. `barrier_tier` above is the DEEPEST resolved adapter tier, which
        # for a corpus that requires [L0,L1,L2] while also shipping an OPTIONAL cycle-accurate cert is a
        # tier the grade explicitly does not require: the driver derives exactly this distinction
        # (run_baseline_qa_loop._cycle_accurate_checkpoint_enabled) and SKIPS the barrier, so demanding it
        # here reported FAIL for capsules the authoritative grade counts as PASS. Measured: five attention
        # capsules numerically exact at the mandatory functional tier (zero mismatch) with the optional
        # cert tier `unavailable` were shown to the agent as failures, and it spent a round re-deriving
        # them. `status` already enforces every mandatory tier (not_run_is_not_pass), so dropping a
        # NON-mandatory conjunct cannot manufacture a pass -- and a non-mandatory tier that ran and FAILED
        # still gates, so an explicitly requested cert failure is never reported as success.
        # WHICH tier this capsule is held to. When the declared barrier ran for NO capsule in the corpus
        # it is not a real bar (a descriptor/adapter mismatch: atlas resolved L4 while every result
        # carries only L0/L1/L2), and scoring against it fails everything regardless of merit -- an agent
        # told it fails everything cannot converge, it rewrites working code. Fall back to the deepest
        # tier that actually produced a verdict. Guarded by `_declared_ran`, so where the target genuinely
        # produces the declared tier a capsule that never reached it still fails.
        bar_used = barrier_tier
        if bar is None and not _declared_ran:
            ran = [k for k, v in tiers.items() if v not in (None, "skipped")]
            if ran:
                bar_used = max(ran)
                bar = tiers.get(bar_used)
        # WHETHER a tier that did not run is allowed to pass: only if this capsule does not require it.
        # Keyed on the tier actually used, so the flag describes the bar the row reports.
        bar_mandatory = bool(tier_objs.get(bar_used, {}).get("mandatory"))
        bar_did_not_run = bar in (None, "unavailable", "skipped")
        passed = (d.get("status") == "pass") and (
            bar == "pass" or (bar_did_not_run and not bar_mandatory))
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
        row = {"capsule": name, "pass": passed, "barrier_tier": bar_used, "barrier_status": bar,
               # say WHETHER the barrier gates this capsule, so an `unavailable` optional cert tier beside
               # `pass: true` reads as "not required here", not as an unexplained contradiction to chase.
               "barrier_gates": bar_mandatory,
               "required_tiers": sorted(t for t, v in tier_objs.items() if v.get("mandatory"))}
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
                   "so you debug from your own intent, as in real bring-up. 'done' = every public capsule "
                   "passes its own MANDATORY tiers (see per_capsule.required_tiers); a tier reported "
                   "`unavailable` with barrier_gates=false is OPTIONAL for that capsule and does NOT hold "
                   "it back -- do not spend rounds on it. Cycles are not a criterion. (Movement ops "
                   "legitimately have 0 matmuls.)"}
    txt = json.dumps(out, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    _log_telemetry(out, a.capsules)   # one operator-side dev-trajectory line (all verdicts, incl. degenerate)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
