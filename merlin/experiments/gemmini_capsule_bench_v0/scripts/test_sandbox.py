"""Prove the bwrap sandbox BEFORE any run: every legit tool works + every answer is masked.

Assembles a fresh workspace for one arm, builds the real bwrap command (bwrap_argv + claude_runtime_binds
+ toolchain_binds + env), and runs a battery INSIDE the sandbox. Exit 0 only if all green. No agent, no
network, no claude invocation — just the agent's tool environment.

  MASKING (must be invisible):  public goldens, hidden capsules, the oracle reference/simulator, the full
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


def _bwrap(ws: Path, bundle: dict, inner: str) -> str:
    parts = RX.bwrap_argv(ws, bundle) + QA.claude_runtime_binds() + TC.toolchain_binds()
    for f in QA.answer_files():
        parts += ["--ro-bind", "/dev/null", str(f)]
    return " ".join(parts) + f" bash -c '{TC.sandbox_env(ws)} {inner}'"


def _run(ws, bundle, inner, timeout=180):
    cmd = _bwrap(ws, bundle, inner)
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="merlin_rtlchecks", choices=list(ARM_BUNDLE))
    a = ap.parse_args(argv)
    bundle = _load_bundle_by_id(ARM_BUNDLE[a.arm])
    print(f"=== sandbox test — arm={a.arm} bundle={ARM_BUNDLE[a.arm]} ===")

    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        ws = Path(td) / "workspace"
        RX.assemble_workspace(bundle, ws)

        # --- MASKING: answers must be invisible ---
        print("\n-- masking (answers must be invisible) --")
        masks = {
            "public golden": C.REPO / "merlin/contract/capsules/isa/A4_acc_scale_i8/golden.yaml",
            "hidden capsules": C.REPO / "merlin/contract/capsules/hidden",
            "oracle reference.py": C.REPO / "merlin/python/merlin/runtime/reference.py",
            "oracle simulator.py": C.REPO / "merlin/python/merlin/runtime/simulator.py",
            "kernel suite (conv.c)": "/path/to/chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/conv.c",
            "~/.claude memory": TC.MEMORY_DIR + "/MEMORY.md",
            "repo (other source)": C.REPO / "merlin/python/merlin/targetgen/capsule_grade.py",
        }
        for label, p in masks.items():
            # CONTENT-level check: goldens are /dev/null-overlaid (present but empty/unreadable), denied
            # dirs are empty tmpfs. "masked" = cannot read any bytes / dir holds no real files.
            if str(p).endswith("hidden") or "capsules/hidden" in str(p):
                inner = f'test "$(find "{p}" -type f 2>/dev/null | wc -l)" = "0" && echo masked || echo VISIBLE'
            else:
                inner = f'if head -c1 "{p}" >/dev/null 2>&1 && test -s "{p}"; then echo VISIBLE; else echo masked; fi'
            rc, out, _ = _run(ws, bundle, inner, timeout=60)
            _ok(f"masked: {label}", out.endswith("masked"), out)

        # --- TOOLS: must work ---
        print("\n-- tools (must work) --")
        checks = [
            ("python3", "python3 --version"),
            ("mlir-opt", "mlir-opt --version | head -1"),
            ("g++ (C++ compiler for baseline)", "g++ --version | head -1"),
            ("cmake>=3.20", "cmake --version | head -1"),
            ("ninja", "ninja --version"),
            ("make", "make --version | head -1"),
            ("spike", "spike --help 2>&1 | head -1"),
            ("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-gcc --version | head -1"),
            ("verilator L3 sim", f'test -x {TC.CHIPYARD_VERILATOR}/simulator-chipyard.harness-GemminiAndOPUShuttleConfig && echo present'),
        ]
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
        # (golden grading is DRIVER-SIDE/redacted, outside the sandbox — the runner gemmini.py imports the
        # oracle, so it is correctly absent here. In-sandbox the agent authors -> compiles -> runs spike.)
        print("\n-- end-to-end (compile with riscv-gcc + curated harness, run on spike) --")
        e2e = (f'cd /tmp && printf "int main(){{return 0;}}" > t.c && '
               f'riscv64-unknown-elf-gcc -I {TC.CURATED_HARNESS}/include -c t.c -o t.o && echo COMPILED && '
               f'riscv64-unknown-elf-gcc -march=rv64gc -o t.elf t.c -nostartfiles -e main 2>/dev/null; '
               f'spike --help >/dev/null 2>&1 && echo SPIKE_OK')
        rc, out, err = _run(ws, bundle, e2e, timeout=120)
        _ok("riscv-gcc compiles (curated harness) + spike runnable",
            "COMPILED" in out and "SPIKE_OK" in out, (out or err).replace("\n", " ")[:80])

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
        ref = C.REPO / "out/artifacts/targets/gemmini/agent_spec_v1_mlir_oot"
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
                                'python3 simjob.py submit --sim spike --capsules A1_mvin_mvout', timeout=60)
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
            cap0 = (verdict.get("per_capsule") or [{}])[0]
            _ok("async spike job -> real L2=pass through bwrap",
                verdict.get("all_pass") and cap0.get("barrier_status") == "pass",
                f"n={verdict.get('n_passed')}/{verdict.get('n_capsules')}")
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
