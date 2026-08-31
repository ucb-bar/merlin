"""T1 — deterministic test of the async constrained oracle service (simjob shim + broker), no agent.

Proves: submit is ASYNC (returns a job_id immediately), the round-trip returns a REAL redacted spike
verdict, the runner is CONSTRAINED (bad capsule / bad sim rejected), output is redacted (no golden), and
the per-capsule capsule-filter actually limits the grade. Verilator is exercised as a submit+running check
(the full L3 run is covered by the readiness gate, since it takes minutes).

Usage: test_simjob.py  (uses the prebuilt known-good backend out/artifacts/targets/<target>/agent_spec_v1_mlir_oot)
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _common as C  # noqa: E402 — active target (descriptor-driven), bootstraps merlin/python
REPO = C.REPO
PY = sys.executable
REF = REPO / "out/artifacts/targets" / C.TARGET / "agent_spec_v1_mlir_oot"
CE = "/path/to/chipyard/.conda-env"
results = []


def _ok(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _shim(ws, *args, timeout=60):
    return subprocess.run([PY, str(ws / "simjob.py"), *args], cwd=str(ws),
                          capture_output=True, text=True, timeout=timeout)


def _screen_only_l2_completion(verdict: dict) -> tuple[bool, str]:
    """A1 requires L3, so a spike-only L2 pass is measured but not certified."""
    rows = verdict.get("per_capsule") or []
    n_capsules = verdict.get("n_capsules")
    n_passed = verdict.get("n_passed")
    n_certified = verdict.get("n_certified")
    n_screened = verdict.get("n_screened_only")
    counts_are_explicit = all(isinstance(v, int) for v in (
        n_capsules, n_passed, n_certified, n_screened))
    complete = bool(
        counts_are_explicit
        and n_capsules == 1
        and n_passed == 1
        and n_certified == 0
        and n_screened == 1
        and verdict.get("all_pass") is False
        and len(rows) == 1
        and rows[0].get("pass") is True
        and rows[0].get("barrier_tier") == "L2"
        and rows[0].get("barrier_status") == "pass"
    )
    detail = (f"measured={n_passed}/{n_capsules} certified={n_certified} "
              f"screened_only={n_screened} all_pass={verdict.get('all_pass')} "
              f"barrier={rows[0].get('barrier_tier') if rows else None}/"
              f"{rows[0].get('barrier_status') if rows else None}")
    return complete, detail


def main():
    if not (REF / "manifest.yaml").exists():
        print("FAIL: reference backend missing"); return 1
    env = dict(os.environ)
    env["PATH"] = f"{CE}/bin:{CE}/riscv-tools/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{CE}/lib:{CE}/riscv-tools/lib:" + env.get("LD_LIBRARY_PATH", "")
    with tempfile.TemporaryDirectory() as td:            # honours TMPDIR
        ws = Path(td)
        import shutil
        shutil.copy(HERE / "simjob_shim.py", ws / "simjob.py")
        (ws / "submission").symlink_to(REF)             # broker reads <ws>/submission via agent_selfcheck
        broker = subprocess.Popen([PY, str(HERE / "simjob_broker.py"), "--ws", str(ws), "--max-jobs", "2"],
                                  env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=True)
        try:
            # 1. async submit returns immediately
            t0 = time.time(); r = _shim(ws, "submit", "--sim", "spike", "--capsules", "A1_mvin_mvout")
            dt = time.time() - t0
            jid = (json.loads(r.stdout).get("job_id") if r.returncode == 0 else None)
            _ok("submit is async (returns job_id < 3s)", jid and dt < 3, f"{dt:.1f}s job={jid}")
            # 2. constraint: malicious capsule rejected
            rj = _shim(ws, "submit", "--sim", "spike", "--capsules", "../../etc/passwd")
            jbad = json.loads(rj.stdout).get("job_id")
            # 3. The untrusted shim accepts an opaque sim token; the trusted broker owns the derived,
            # closed allowlist and must reject the bad value visibly.
            rs = _shim(ws, "submit", "--sim", "evil", "--capsules", "A1_mvin_mvout")
            jevil = json.loads(rs.stdout).get("job_id") if rs.returncode == 0 else None
            # 4. poll spike job to completion (fast)
            st = None
            for _ in range(40):
                p = _shim(ws, "poll", "--job-id", jid)
                d = json.loads(p.stdout); st = d["state"]
                if st in ("done", "error"):
                    break
                time.sleep(5)
            res = (d.get("result") or {})
            screen_ok, screen_detail = _screen_only_l2_completion(res)
            _ok("spike job returns a complete screen-only L2 pass on known-good backend",
                st == "done" and screen_ok, f"state={st} {screen_detail}")
            # 5. malicious job ended as error (rejected), not run
            stbad = None
            for _ in range(6):
                pb = _shim(ws, "poll", "--job-id", jbad); stbad = json.loads(pb.stdout)["state"]
                if stbad in ("error", "done"):
                    break
                time.sleep(1)
            _ok("malicious capsule -> error (constrained, not executed)", stbad == "error", f"state={stbad}")
            evil_state = None
            evil_result = {}
            if jevil:
                for _ in range(6):
                    pe = _shim(ws, "poll", "--job-id", jevil)
                    ed = json.loads(pe.stdout)
                    evil_state = ed["state"]
                    evil_result = ed.get("result") or {}
                    if evil_state in ("error", "done"):
                        break
                    time.sleep(1)
            _ok("bad --sim rejected by the trusted broker",
                evil_state == "error" and evil_result.get("rejected_field") == "sim",
                f"state={evil_state} field={evil_result.get('rejected_field')}")
            # 6. redaction: no 'expected'/'golden' KEY anywhere in the responses (the descriptive `note`
            # text legitimately mentions the words — check structure, not substring).
            def _has_golden_key(o):
                if isinstance(o, dict):
                    return any(k in ("expected", "golden") for k in o) or any(_has_golden_key(v) for v in o.values())
                if isinstance(o, list):
                    return any(_has_golden_key(x) for x in o)
                return False
            leaked = []
            for f in (ws / ".qa_channel").glob("simresp_*.json"):
                try:
                    if _has_golden_key(json.loads(f.read_text())):
                        leaked.append(f.name)
                except Exception:
                    pass
            _ok("no golden/expected KEY in any simresp (redacted)", not leaked, str(leaked))
            # 7. verilator: submit + reaches running (full L3 covered by readiness gate)
            rv = _shim(ws, "submit", "--sim", "verilator", "--capsules", "A1_mvin_mvout")
            jv = json.loads(rv.stdout).get("job_id")
            vstate = None
            for _ in range(12):
                pv = _shim(ws, "poll", "--job-id", jv); vstate = json.loads(pv.stdout)["state"]
                if vstate in ("running", "done", "error"):
                    break
                time.sleep(2)
            # A shared host may have every global verilator slot occupied by another campaign. Queued is
            # therefore a valid accepted state; readiness separately proves an actual L3 execution.
            _ok("verilator job accepted + queued/scheduled without rejection",
                bool(jv) and vstate in ("queued", "running", "done"), f"state={vstate}")
        finally:
            (ws / ".qa_channel" / "STOP").write_text("stop")
            try:
                broker.wait(timeout=10)
            except Exception:
                broker.kill()
            # Kill only this test broker's process group and any simulator children it owns. A broad
            # pkill would terminate unrelated experiments on the shared host.
            try:
                os.killpg(os.getpgid(broker.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    n = sum(1 for _, ok in results if ok)
    print(f"\nT1 simjob: {n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
