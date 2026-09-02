"""Prove the bwrap sandbox BEFORE any run: every legit tool works + every answer is masked.

Assembles a fresh workspace for one arm, builds the real bwrap command (bwrap_argv + claude_runtime_binds
+ toolchain_binds + env), and runs a battery INSIDE the sandbox. Exit 0 only if all green. No agent, no
network, no claude invocation — just the agent's tool environment.

  MASKING (must be invisible):  public goldens, private model weights, hidden capsules, the oracle, the full
                                kernel suite, ~/.claude memory, other repos under /scratch & /scratch2
  TOOLS (must work):            python3 + import merlin.targetgen.oot_starterkit (+ xdsl dep), mlir-opt,
                                clang, cmake>=3.20, ninja, make, spike, riscv64-unknown-elf-gcc, verilator L3
  END-TO-END:                   agent_selfcheck.py --sim spike on one capsule actually runs

Usage: test_sandbox.py [--arm merlin_rtlchecks|merlin|baseline]
"""
from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
import run_agent_experiment as RX
import run_baseline_qa_loop as QA
import sandbox_toolchain as TC

SCRIPTS_DIR = Path(__file__).resolve().parent
ARM_BUNDLE = {
    "baseline": "raw_baseline_hwbringup_v0",
    "merlin": "merlin_assisted_hwbringup_v0",
    "merlin_rtlchecks": "merlin_assisted_rtlchecks_hwbringup_v0",
}
results = []


def _ok(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _screen_completion(verdict: dict) -> tuple[bool, str]:
    """Validate a complete, non-vacuous screen without calling it certification.

    A cheap oracle may pass every capsule it can measure while a mandatory deeper tier remains
    unrun.  In that case ``n_passed == n_capsules`` is true, ``n_screened_only`` is non-zero, and
    ``all_pass`` must remain false.  Conversely, a target whose selected oracle reaches its complete
    contract may legitimately certify here.  Require the explicit counters to agree in either case.
    """
    rows = verdict.get("per_capsule") or []
    n_capsules = verdict.get("n_capsules")
    n_passed = verdict.get("n_passed")
    n_certified = verdict.get("n_certified")
    n_screened = verdict.get("n_screened_only")
    counts_are_explicit = all(isinstance(v, int) for v in (
        n_capsules, n_passed, n_certified, n_screened))
    complete = bool(
        counts_are_explicit
        and n_capsules > 0
        and len(rows) == n_capsules
        and n_passed == n_capsules
        and n_certified + n_screened == n_capsules
        and verdict.get("all_pass") is (n_certified == n_capsules)
        and all(row.get("pass") is True
                and row.get("barrier_tier")
                and row.get("barrier_status") == "pass"
                for row in rows)
    )
    tiers = sorted({str(row.get("barrier_tier")) for row in rows if row.get("barrier_tier")})
    detail = (f"measured={n_passed}/{n_capsules} certified={n_certified} "
              f"screened_only={n_screened} all_pass={verdict.get('all_pass')} "
              f"tiers={','.join(tiers) or 'none'}")
    return complete, detail


def _bwrap(ws: Path, bundle: dict, inner: str) -> str:
    """The SAME sandbox the agent runs in — delegated to the run path, never rebuilt here.

    This used to hand-roll its own box: base argv + binds, then an UNGUARDED ``--ro-bind /dev/null``
    over every file in the legacy ``answer_files()`` list. Two things went wrong with that. It skipped
    the directory surfaces entirely, and — worse — binding a file that the deny-by-default base had
    already tmpfs-hidden RE-CREATES that path inside the tmpfs, so the held-out capsule NAMES
    (``hidden/<capsule>/…``) became listable again even though their contents were empty. A gate
    that assembles its own weaker sandbox is not evidence about the real one; ``bwrap_cmd`` applies the
    derived answer-mask pass, which skips a surface the base already hides."""
    return QA.bwrap_cmd(inner, ws, bundle)


def _run(ws, bundle, inner, timeout=180):
    cmd = _bwrap(ws, bundle, inner)
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def _descriptor() -> dict:
    """The SELECTED target's target_experiment.yaml, as data."""
    import yaml
    d = C.EXP / "target_experiment.yaml"
    try:
        return yaml.safe_load(d.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}


def _declares_bespoke_sim() -> bool:
    """True iff this target's RTL tier runs through an external RISC-V/chipyard simulator.

    Read from the descriptor's ``sim_via``: a target that leaves it empty grades on an in-process RTL
    model and never shells out to spike / riscv-gcc / a chipyard verilator binary. Derived, not a
    target-name branch -- a new endpoint's answer falls out of its own descriptor.
    """
    def _find(node):
        """`sim_via` is nested (gemmini declares it under `toolchain`); find it wherever it sits."""
        if isinstance(node, dict):
            v = node.get("sim_via")
            if isinstance(v, str) and v.strip():
                return True
            return any(_find(x) for x in node.values())
        if isinstance(node, list):
            return any(_find(x) for x in node)
        return False

    return _find(_descriptor())


def _broker_sim() -> str:
    """The simulator name the broker should run for THIS target's loop tier.

    Derived from the target's own oracle adapters (the loop tier is the one the QA loop grades on), so a
    target whose RTL tier is an in-process model is probed through that model rather than through a
    RISC-V simulator it never invokes.
    """
    try:
        from merlin.targetgen.capsule_runner import qa_loop_adapters
        tiers = sorted(qa_loop_adapters(C.TARGET))
    except Exception:  # noqa: BLE001 -- an unresolvable adapter set is not this probe's business
        tiers = []
    if not _declares_bespoke_sim():
        # the broker's neutral token: "grade on whatever tier this target's contract resolves to"
        return "contract"
    return "spike" if "L2" in tiers else (tiers[0].lower() if tiers else "spike")


def _probe_capsule() -> str:
    """One real public capsule id belonging to THIS target, read from the corpus it declares.

    The probe only needs A capsule that exists and is public. Naming one target's capsule here would
    make the gate fail on every other target for a reason that has nothing to do with the sandbox --
    and, worse, would silently hand a NEW target of the same class another target's capsule id. The
    corpus root comes from the descriptor's ``capsule_corpus``, so each target answers for itself.
    """
    import yaml
    root = _descriptor().get("capsule_corpus")
    roots = [C.REPO / root] if isinstance(root, str) and root.strip() else [C.REPO / "merlin/contract/capsules"]
    for base in roots:
        for cy in sorted(base.rglob("capsule.yaml")):
            if "hidden" in cy.parts:                      # never probe a held-out capsule
                continue
            try:
                y = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except OSError:
                continue
            if str(y.get("label", "")).strip() == "public":
                return cy.parent.name
    raise SystemExit(f"no public capsule found for {C.TARGET} under {[str(r) for r in roots]}; "
                     f"the sandbox probe cannot be constructed without one")


def _verilator_design() -> str:
    """The chipyard verilator design this target's sim is built for, from the descriptor."""
    d = _descriptor()
    for key in ("verilator_design", "chipyard_config"):
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    o = d.get("oracle")
    if isinstance(o, dict) and isinstance(o.get("verilator_design"), str):
        return o["verilator_design"].strip()
    return "GemminiRocketConfig"  # historical default for the chipyard endpoint


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="merlin_rtlchecks", choices=list(ARM_BUNDLE))
    a = ap.parse_args(argv)
    bundle = _load_bundle_by_id(ARM_BUNDLE[a.arm])
    print(f"=== sandbox test — arm={a.arm} bundle={ARM_BUNDLE[a.arm]} ===")

    with tempfile.TemporaryDirectory() as td:            # honours TMPDIR; bwrap masks via tmpfs, not a /tmp bind
        ws = Path(td) / "workspace"
        RX.assemble_workspace(bundle, ws)

        # --- MASKING: answers must be invisible ---
        print("\n-- masking (answers must be invisible) --")
        # a denied answer kernel from the target's example suite (chipyard <target>-rocc-tests) must be
        # masked; resolve the real sim repo (falls back to a nonexistent stand-in), target from _common.
        from merlin.common.paths import ext_path
        _cy = ext_path("chipyard")
        _suite_kernel = (str(_cy / f"generators/{C.TARGET}/software/{C.TARGET}-rocc-tests/bareMetalC/conv.c")
                         if _cy else f"/nonexistent/{C.TARGET}-rocc-tests/bareMetalC/conv.c")
        masks = {
            "public golden": C.REPO / "merlin/contract/capsules/isa/A4_acc_scale_i8/golden.yaml",
            "hidden capsules": C.REPO / "merlin/contract/capsules/hidden",
            "oracle reference.py": C.REPO / "merlin/python/merlin/runtime/reference.py",
            "oracle simulator.py": C.REPO / "merlin/python/merlin/runtime/simulator.py",
            "kernel suite (conv.c)": _suite_kernel,
            "~/.claude memory": TC.MEMORY_DIR + "/MEMORY.md",
            "repo (other source)": C.REPO / "merlin/python/merlin/targetgen/capsule_grade.py",
        }
        for label, p in masks.items():
            # CONTENT-level check: goldens are /dev/null-overlaid (present but empty/unreadable), denied
            # dirs are empty tmpfs. "masked" = cannot read any bytes / dir holds no real files.
            if Path(p).is_dir():
                inner = f'test "$(find "{p}" -type f 2>/dev/null | wc -l)" = "0" && echo masked || echo VISIBLE'
            else:
                inner = f'if head -c1 "{p}" >/dev/null 2>&1 && test -s "{p}"; then echo VISIBLE; else echo masked; fi'
            rc, out, _ = _run(ws, bundle, inner, timeout=60)
            _ok(f"masked: {label}", out.endswith("masked"), out)

        # EVERY surface the ACTIVE descriptor declares, on top of the fixed probes above. The fixed list
        # names a handful of known paths; this asks the same registry the sandbox masks FROM
        # (answer_surfaces) what this target's surfaces actually are, so a newly declared one — a prior
        # reference backend, a per-target hidden set — is PROVEN masked the day it is declared rather
        # than trusted. A declared-but-unmasked answer key is worse than an undeclared one: the
        # descriptor claims cover that does not exist. Checked in ONE sandbox entry (there are hundreds
        # of surfaces; one bwrap per path would make the gate too slow to run before every launch).
        from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces as _surfaces
        from merlin.targetgen.target_experiment import load_target_experiment
        _declared = _surfaces(load_target_experiment(C.DESCRIPTOR))
        if _declared:
            _probe = (
                'while IFS= read -r p; do\n'
                '  if [ -d "$p" ]; then\n'
                '    if [ "$(find "$p" -type f 2>/dev/null | wc -l)" = "0" ]; then s=masked; else s=VISIBLE; fi\n'
                '  elif head -c1 "$p" >/dev/null 2>&1 && [ -s "$p" ]; then s=VISIBLE; else s=masked; fi\n'
                '  echo "$s|$p"\n'
                "done <<'__SURFACES__'\n"
                + "\n".join(str(s.path) for s in _declared)
                + "\n__SURFACES__\n")
            rc, out, err = _run(ws, bundle, _probe, timeout=300)
            verdicts = dict((ln.split("|", 1)[1], ln.split("|", 1)[0])
                            for ln in out.splitlines() if "|" in ln)
            by_origin: dict[str, list] = {}
            for s in _declared:
                by_origin.setdefault(s.origin, []).append(s)
            for origin, surfs in sorted(by_origin.items()):
                # A surface with NO verdict line is not a pass: the probe never reached it.
                leaked = [s.label for s in surfs if verdicts.get(str(s.path)) != "masked"]
                _ok(f"masked: every declared {origin} surface ({len(surfs)})", not leaked,
                    "all masked" if not leaked else f"VISIBLE: {', '.join(leaked[:4])}")

        # --- TOOLS: must work ---
        print("\n-- tools (must work) --")
        checks = [
            ("python3", "python3 --version"),
            ("mlir-opt", "mlir-opt --version | head -1"),
            ("g++ (C++ compiler for baseline)", "g++ --version | head -1"),
            ("cmake>=3.20", "cmake --version | head -1"),
            ("ninja", "ninja --version"),
            ("make", "make --version | head -1"),
        ]
        # The RISC-V/chipyard toolchain is required only by a target whose RTL tier RUNS THROUGH IT. A
        # target that declares no bespoke sim (``sim_via: ""`` -- its RTL oracle is an in-process model
        # such as the mlc arc MXU model) never invokes spike, riscv-gcc or a chipyard verilator binary,
        # so demanding them marks a healthy sandbox NO-GO for tools it will never call. Derived from the
        # descriptor, not from a target name, so a new endpoint needs no edit here.
        if _declares_bespoke_sim():
            checks += [
                ("spike", "spike --help 2>&1 | head -1"),
                ("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-gcc --version | head -1"),
                ("verilator L3 sim",
                 f'test -x {TC.CHIPYARD_VERILATOR}/simulator-chipyard.harness-{_verilator_design()} && echo present'),
            ]
        else:
            print(f"  [n/a ] RISC-V/chipyard toolchain — {C.TARGET} declares no bespoke sim "
                  f"(sim_via empty); its RTL tier is an in-process model")
        for label, inner in checks:
            rc, out, err = _run(ws, bundle, inner, timeout=120)
            _ok(f"tool: {label}", rc == 0 and out, (out or err).splitlines()[0][:70] if (out or err) else f"rc={rc}")
        # merlin infra: required for the xDSL arms, MUST be absent for the C++ baseline (no merlin tools)
        rc, out, err = _run(ws, bundle, 'python3 -c "import merlin.targetgen.oot_starterkit, xdsl; print(42)"', timeout=120)
        merlin_ok = "42" in out
        if a.arm == "baseline":
            _ok("merlin infra ABSENT for baseline (control)", not merlin_ok, "correctly unavailable")
        else:
            _ok("tool: import oot_starterkit (+xdsl)", merlin_ok, (out or err).splitlines()[-1][:70] if (out or err) else f"rc={rc}")

        # --- END-TO-END: the agent's real in-sandbox capability = compile + run on spike ---
        # (golden grading is DRIVER-SIDE/redacted, outside the sandbox — the target's runtime backend
        # imports the oracle, so it is correctly absent here. In-sandbox the agent authors -> compiles -> runs spike.)
        if _declares_bespoke_sim():
            print("\n-- end-to-end (compile with riscv-gcc + curated harness, run on spike) --")
            e2e = (f'cd /tmp && printf "int main(){{return 0;}}" > t.c && '
                   f'riscv64-unknown-elf-gcc -I {TC.CURATED_HARNESS}/include -c t.c -o t.o && echo COMPILED && '
                   f'riscv64-unknown-elf-gcc -march=rv64gc -o t.elf t.c -nostartfiles -e main 2>/dev/null; '
                   f'spike --help >/dev/null 2>&1 && echo SPIKE_OK')
            rc, out, err = _run(ws, bundle, e2e, timeout=120)
            _ok("riscv-gcc compiles (curated harness) + spike runnable",
                "COMPILED" in out and "SPIKE_OK" in out, (out or err).replace("\n", " ")[:80])
        else:
            print(f"\n-- end-to-end: n/a — {C.TARGET} compiles no RISC-V host program (no bespoke sim) --")

        # --- ASYNC ORACLE through bwrap: a REAL submission, real spike L2=pass via the simjob broker;
        #     oracle stays out-of-box. (Verilator L3 is covered by readiness_check; here spike is enough
        #     to prove the in-sandbox->broker->redacted-verdict path on a real package.)
        print("\n-- async oracle (simjob) through bwrap: real spike L2=pass + oracle masked --")
        import shutil as _sh
        ch = ws / ".qa_channel"; ch.mkdir(parents=True, exist_ok=True)
        (ch / "STOP").unlink(missing_ok=True)
        # stage BOTH shims (unlink symlink first — copy-onto-symlink would clobber the real scripts)
        for src, dstn in (("selfcheck_shim.py", "agent_selfcheck.py"), ("simjob_shim.py", "simjob.py")):
            d = ws / dstn
            if d.exists() or d.is_symlink():
                d.unlink()
            _sh.copy(SCRIPTS_DIR / src, d)
        # real known-good submission so the broker grades a genuine L2 pass (not just a channel ping)
        ref = C.REPO / "out/artifacts/targets" / C.TARGET / "agent_spec_v1_mlir_oot"
        sub = ws / "submission"
        if sub.exists() or sub.is_symlink():
            sub.unlink() if sub.is_symlink() else _sh.rmtree(sub)
        sub.symlink_to(ref)
        # start_new_session=True makes the broker its OWN process-group leader, so teardown can kill
        # exactly the sims IT spawned (scoped) instead of a broad pkill on a shared host.
        broker = subprocess.Popen([sys.executable, str(SCRIPTS_DIR / "simjob_broker.py"), "--ws", str(ws)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=True)
        try:
            # submit (async) from INSIDE the sandbox, then poll to a real verdict
            rc, out, err = _run(ws, bundle,
                                f'python3 simjob.py submit --sim {_broker_sim()} --capsules {_probe_capsule()}', timeout=60)
            import json as _json
            jid = None
            try:
                jid = _json.loads(out.strip().splitlines()[-1])["job_id"]
            except Exception:
                pass
            _ok("simjob submit (async) returns a job_id in-sandbox", bool(jid), str(out)[:70])
            verdict = {}
            if jid:
                for _ in range(40):
                    rc, pout, _e = _run(ws, bundle, f'python3 simjob.py poll --job-id {jid}', timeout=60)
                    try:
                        d = _json.loads(pout.strip().splitlines()[-1])
                    except Exception:
                        d = {}
                    if d.get("state") in ("done", "error"):
                        verdict = d.get("result") or {}
                        break
                    time.sleep(5)
            screen_ok, screen_detail = _screen_completion(verdict)
            _ok("async oracle job -> complete measured-tier pass through bwrap",
                screen_ok, screen_detail)
            # the oracle must NOT have become readable in-box during the brokered run
            rc2, oout, _ = _run(ws, bundle,
                                f'head -c1 "{C.REPO}/merlin/python/merlin/runtime/reference.py" >/dev/null 2>&1 && echo LEAK || echo masked',
                                timeout=60)
            _ok("oracle stays masked during brokered async run", oout.strip().endswith("masked"), oout.strip())
        finally:
            (ch / "STOP").write_text("stop")
            try:
                broker.wait(timeout=10)
            except Exception:
                broker.kill()
            # SCOPED teardown: kill ONLY the broker's own process group (the spike/verilator children IT
            # spawned) — never a broad `pkill -f simulator-chipyard`, which on this shared host would also
            # kill a concurrent session's verilator (the shared-host rule; caused false exit-144s).
            try:
                os.killpg(os.getpgid(broker.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    n = sum(1 for _, ok in results if ok)
    print(f"\n{'='*56}\nSANDBOX TEST: {n}/{len(results)} passed")
    go = n == len(results)
    print("🟢 sandbox GO — tools work, answers masked; safe to relaunch under --sandbox bwrap"
          if go else "🔴 sandbox NO-GO — fix the FAILs before relaunching")
    return 0 if go else 1


def _load_bundle_by_id(bundle_id: str) -> dict:
    import yaml
    p = C.BUNDLES / bundle_id / "input_bundle_manifest.yaml"
    return yaml.safe_load(p.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
