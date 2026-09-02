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

What it does NOT reveal: golden output tensors, private model weights, hidden capsules, or the reference
oracle. It tells you
WHETHER and roughly WHERE you are wrong (mismatch_count, failing plane) — not the answer.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path


# Restored during branch integration: the merge took this file whole from the other side, which
# dropped select_tiers and left the six tests that pin it failing on a missing attribute.
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


# gsim shares verilator's rung: both are elaborated-RTL, cycle-accurate engines over the SAME
# design (GemminiGsimMonitorsConfig is GemminiRocketConfig plus a harness-level clock
# instantiator, so the DUT is identical). It is the tier that differs by ENGINE, never by name.
SIM_TIER = {"spike": "L2", "verilator": "L3", "gsim": "L3", "vcs": "L4"}


def _adapters(sim: str, target: str, sim_via: str | None) -> tuple[dict, str]:
    """Resolve the self-check oracle tiers from the TARGET's contract (target-agnostic, mirrors the driver
    grade). A chipyard target (gemmini) exposes the spike/verilator/vcs ladder selectable via --sim; any
    other target grades on its OWN contract-derived RTL tier (atlas external_backend -> the program oracle;
    an arc target -> the RTL-derived arc cosim), where --sim is not applicable. Routing atlas here through
    the hardcoded spike/verilator adapters ran the gemmini/RVV lowering path and crashed (AW4)."""
    if sim_via == "chipyard":
        ad = {"L2": CR._spike_verilator_adapter("spike", target)}
        if sim in ("verilator", "vcs", "gsim"):
            # Build L3 from the engine the caller NAMED. This used to hardcode "verilator", which made
            # --sim gsim silently certify on verilator -- a result attributed to the wrong engine. The
            # adapter factory is engine-generic (it forwards simulator=sim and gates on
            # backend.available(sim)), so an absent engine fails closed here rather than substituting.
            _l3_engine = "gsim" if sim == "gsim" else "verilator"
            ad["L3"] = CR._spike_verilator_adapter(_l3_engine, target)
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


def _shape_coverage(sub: Path, out_path: str) -> int:
    """The agent-facing shape-coverage report (see ``--shape-coverage``).

    Leaks nothing: the probe shapes are multiples of the target's own derived tile edge, which the agent
    can read off its RTL facts, and the report contains no golden, no corpus capsule and no reference
    value -- only what the agent's own compiler emitted.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _common as _C
    from merlin.targetgen import lowering_coverage as LC
    try:
        cov = LC.sweep(sub, target=_C.TARGET, contract=str(_C.REPO / "merlin/contract"))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"shape-coverage probe failed: {type(e).__name__}: {e}"}))
        return 2
    cov["note"] = (
        "Each corner is the SAME contraction at a different extent. `emitted_work` is how many "
        "instructions YOUR compiler emitted for it. A larger problem cannot need a smaller program, so a "
        "corner marked `collapsed` is a shape you silently refused -- at the numeric tier that arrives as "
        "an output of zeros and is indistinguishable from wrong arithmetic. `multi_tile_axes_uncovered` "
        "names the axis your lowering does not loop over. If you truly cannot lower a shape, DECLARE it "
        "(set `declined` on the command buffer) rather than emitting a terminator.")
    txt = json.dumps(cov, indent=2)
    print(txt)
    if out_path:
        Path(out_path).write_text(txt)
    return 0 if cov.get("all_covered") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agent self-check runner (redacted; spike/verilator/vcs).")
    ap.add_argument("--submission", default="submission", help="path to your package (with manifest.yaml)")
    # DEFAULTS TO THE CERTIFYING SIM, not the screen. The capsules declare a cycle-accurate cert
    # tier as mandatory, and this ladder runs cheapest-measured-first with fail-fast, so the
    # screen still refutes a broken submission at screen cost -- what changes is that a capsule
    # which PASSES the screen goes on to certify instead of stopping there. Choosing "spike"
    # explicitly is a legitimate fast screen, but it CANNOT certify: the mandatory cert tier
    # reports unavailable and the capsule is not a pass.
    ap.add_argument("--sim", choices=["spike", "verilator", "gsim", "vcs"], default="verilator")
    ap.add_argument("--capsules", default="all", help="'all' or comma-separated capsule names")
    ap.add_argument("--workers", type=int, default=8, help="parallel sim workers (verilator/vcs)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="", help="optional: also write the redacted JSON here")
    # THE FLAG select_tiers WAS WRITTEN FOR. It existed as a function with six tests pinning it and no
    # way to reach it: argparse never accepted --tiers, so the broker's promotion jobs -- which always
    # forward `--tiers <cert_tier>` -- died on "unrecognized arguments: --tiers L3" before doing any
    # work, and the broker reported that as "no verdict produced". Measured on
    # merlincirct_arm4_func_20260901_v4: 19 promotion jobs, every one killed by argparse.
    ap.add_argument("--tiers", default="", help="comma-separated tiers to grade (default: this "
                                               "target's cheap loop ladder), e.g. L3 for a cert run")
    ap.add_argument("--shape-coverage", action="store_true",
                    help="INSTEAD of the capsule suite, probe whether your backend LOWERS the same "
                         "contraction at one tile and at two tiles in each of M, K and N. Costs no "
                         "simulator and no oracle (it only runs your emit path), so it is cheap to run "
                         "often. The public capsules are a fixed set of shapes; passing all of them "
                         "says nothing about whether you generalize past them, and a backend that "
                         "hardcodes those shapes scores 100%% here and fails the held-out grade.")
    a = ap.parse_args(argv)

    if a.shape_coverage:
        return _shape_coverage(Path(a.submission), a.out)

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
    if a.tiers:
        # Validate the REQUEST against everything the endpoint can reach, not against the cheap loop
        # ladder -- that conflation is exactly what select_tiers documents, and it is why a cert tier
        # (derived as oracle_adapters - qa_loop_adapters) could never be asked for.
        _full = dict(CR.oracle_adapters(_tgt, _sim_via))
        _full.update(adapters)                     # a --sim-selected tier is reachable by definition
        adapters, _tier_err = select_tiers(_full, adapters, a.tiers)
        if _tier_err:
            # Fail CLOSED and leave the reason on disk. Exiting without writing --out is what produced
            # the broker's contentless "no verdict produced".
            out = {"error": _tier_err, "all_pass": False, "sim": sim, "tiers_requested": a.tiers}
            txt = json.dumps(out, indent=2); print(txt)
            if a.out:
                Path(a.out).write_text(txt)
            _log_telemetry(out, a.capsules)
            return 2
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
    rows, npass, ncert, nscreened = [], 0, 0, 0
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
        _tier_results = d.get("tiers") or {}
        tiers = {t: (v or {}).get("status") for t, v in _tier_results.items()}
        bar = tiers.get(barrier_tier)
        bar_used = barrier_tier
        if bar is None:
            # A mixed suite can legitimately have different tier shapes. Whole-model capsules, for
            # example, do not produce the per-op command-buffer screen tier, but can produce a DEEPER
            # required model/RTL verdict. A corpus-wide `_declared_ran` must not let an unrelated op's
            # L2 result veto that stronger per-capsule L3 evidence. Derive this exception entirely from
            # the capsule result: the substitute must be mandatory, must actually have run, and must be
            # deeper than the missing barrier. Thus an op that stopped at L0/L1 still fails closed, and
            # an advisory tier cannot certify it.
            _ran_required = [t for t, v in _tier_results.items()
                             if bool((v or {}).get("mandatory"))
                             and (v or {}).get("status") not in
                             (None, "skipped", "unavailable")]
            _deeper_required = [t for t in _ran_required if t > barrier_tier]
            if _deeper_required:
                bar_used = max(_deeper_required)
                bar = tiers.get(bar_used)
            elif not _declared_ran:
                # The declared barrier never ran for ANY capsule, so scoring against it marks EVERY
                # capsule failed no matter what the grade said. Measured on atlas: the barrier resolved
                # to L4 while its results carry only {L0 skipped, L1 skipped, L2 pass}. Fall back to the
                # deepest tier that actually produced a verdict. `status == "pass"` is still required
                # below, and a skipped tier is never treated as a barrier.
                ran = [k for k, v in tiers.items() if v not in (None, "skipped")]
                if ran:
                    bar_used = max(ran)
                    bar = tiers.get(bar_used)
        # CLASSIFY UNDER THE ORACLE SELECTION IN FORCE.
        #
        # A capsule declares the tiers it requires; an oracle selection supplies adapters for some subset.
        # A required tier with no adapter here did not RUN, and a tier that did not run is absence of
        # evidence about the backend, not evidence against it. `NOT_RUN_IS_NOT_PASS` is the right rule for
        # a CERTIFYING grade -- it stops an unrun tier being read as a pass -- but applying it as a FAILURE
        # inside a screen scores the submission for the engine the caller picked.
        #
        # Measured: a submission whose capsule graded {L0 pass, L1 pass, L2 pass, L3 unavailable} scored
        # 0, and a run spent 325 consecutive self-checks at zero while the certifying grade of the same
        # submission was 22/25. An agent told it fails everything rewrites working code.
        #
        # So: `passed` = passed everything this selection could measure; `certified` = also cleared every
        # mandatory tier. `all_pass` keys on certification, so a screen can never certify -- it may only
        # eliminate. Derived from the capsule's own declared tiers and the adapters present; no engine or
        # target is named here.
        _f = d.get("failure") or {}
        _blocked_by_selection = (_f.get("category") == "NOT_RUN_IS_NOT_PASS"
                                 and _f.get("tier_status") == "unavailable")
        _ran = {t: v for t, v in tiers.items() if v not in (None, "skipped", "unavailable")}
        _ran_clean = bool(_ran) and all(v == "pass" for v in _ran.values())
        certified = (d.get("status") == "pass") and (bar == "pass")
        screened = (not certified) and _blocked_by_selection and _ran_clean and bar in ("pass", None)
        passed = certified or screened
        npass += int(passed)
        ncert += int(certified)
        nscreened += int(screened)
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
        row = {"capsule": name, "pass": passed,
               "execution_digest": _qc._execution_digest_from_result(cr),
               "barrier_tier": bar_used,
               "barrier_declared": barrier_tier, "barrier_status": bar}
        # A STATED DECLINE IS THE MOST ACTIONABLE THING THIS REPORT CAN CARRY, so it rides the row
        # whether or not the capsule passed, and ahead of the numeric block. Without it a declined
        # capsule reads exactly like one whose arithmetic is wrong -- an output of zeros and a
        # "does not compute the declared operation" -- and the fix for those two is not the same.
        if d.get("declined"):
            row["declined"] = d["declined"]
            row["declined_note"] = ("your backend DECLINED this capsule -- it never emitted a program. "
                                    "This is a coverage gap (a shape/op you do not lower), not a "
                                    "numeric bug; do not debug the arithmetic.")
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
    n_declined = sum(1 for r in rows if r.get("declined"))
    # A capsule can fail here for a reason the agent does not control: it declares a mandatory cert tier
    # that the SIM CHOSEN FOR THIS RUN supplies no adapter for. That is a screen being asked to certify,
    # not a defect in the submission, and it must say so -- being told to fix an oracle it cannot reach is
    # how a conformant run decays into ten rounds of effort against nothing.
    _unreached = sorted({(r.get("failure") or {}).get("tier") for r in rows
                         if (r.get("failure") or {}).get("category") == "NOT_RUN_IS_NOT_PASS"
                         and (r.get("failure") or {}).get("tier_status") == "unavailable"} - {None})
    out = {"sim": sim, "barrier_tier": barrier_tier, "n_passed": npass, "n_capsules": n,
           # `n_passed` is what this oracle selection could measure; `n_certified` is what cleared every
           # mandatory tier. They differ exactly when the selection cannot reach a required tier, and
           # `all_pass` keys on the second: a screen may eliminate, it may never certify.
           "n_certified": ncert, "n_screened_only": nscreened,
           "all_pass": ncert == n and n > 0, "per_capsule": rows,
           "n_declined": n_declined,
           "note": f"Self-check on {sim} ({barrier_tier}). You see EVERYTHING your dialect produced — "
                   "command buffer, decoded trace + instruction counts, sim console, and your artifacts "
                   "copied to ./selfcheck_out/. The diff stats (mismatch_count, magnitudes) are YOUR "
                   "output measured against the operation's own definition, which you can reproduce from "
                   "the declared inputs — there is no answer key; the reference output values are withheld "
                   "so you debug from your own intent, as in real bring-up. 'done' = all public pass on "
                   "verilator/VCS; cycles are not a criterion. (Movement ops legitimately have 0 matmuls.)"
                   + (f" {n_declined} capsule(s) were DECLINED by your backend -- see 'declined' on those "
                      f"rows. A decline is a shape/op you never lowered, NOT wrong arithmetic."
                      if n_declined else "")
                   + (f" ⚠ {len(_unreached)} mandatory cert tier(s) {_unreached} were NOT REACHABLE with "
                      f"--sim {sim}. {nscreened} capsule(s) passed EVERY tier this sim can run and are "
                      f"reported as screened, not certified: they are counted in n_passed and NOT in "
                      f"n_certified, and they are not failures — the tier could not run here, which is a "
                      f"property of the sim you selected, not of your backend. Re-run without --sim (the "
                      f"default certifies) to convert them; all_pass requires certification."
                      if _unreached else "")}
    txt = json.dumps(out, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    _log_telemetry(out, a.capsules)   # one operator-side dev-trajectory line (all verdicts, incl. degenerate)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
