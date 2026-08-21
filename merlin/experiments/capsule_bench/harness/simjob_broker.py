"""Driver-side ASYNC oracle broker — runs the agent's requested sims OUTSIDE the sandbox, redacted.

Companion to simjob_shim.py (the in-sandbox CLI). Watches <ws>/.qa_channel for `simreq_*.json`, runs each
as a SEPARATE PROCESS = the existing agent_selfcheck.py (the single grading+redaction authority — golden
expected values withheld), and writes back `simresp_<id>.json` + `simdone_<id>`. A timeout is a normal
redacted verdict, never a crash.

CONSTRAINED SIM-RUNNER (load-bearing isolation): a request may only name {sim in spike/verilator/vcs,
capsules that exist under the public capsule set, debug from a whitelist, workers (clamped)}. The broker
maps those to a FIXED agent_selfcheck.py argv — it NEVER execs anything the request names. So the agent
gets full sim power on its OWN submission + the harness capsules, but cannot read goldens/oracle or run
arbitrary shell outside the sandbox.

Bounded local pool (--max-jobs) + a GLOBAL cross-arm verilator slot cap (file semaphore in a shared dir)
so the 3 concurrent arms don't launch 3x the verilator load. Exits on the existing STOP sentinel.

Usage: simjob_broker.py --ws <workspace> [--max-jobs 4] [--veril-slots 2] [--poll 0.5]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELFCHECK = HERE / "agent_selfcheck.py"


def _public_capsules() -> Path:
    """Same per-target set agent_selfcheck validates against — DERIVED from the descriptor's
    capsule_corpus (no committed gemmini leak); legacy committed set as fallback."""
    try:
        import _common as _C
        from merlin.targetgen.contract.materialize import public_capsules_for
        from merlin.targetgen.target_experiment import load_target_experiment
        return public_capsules_for(load_target_experiment(_C.EXP / "target_experiment.yaml"))
    except Exception:  # noqa: BLE001
        return HERE / "full_public_capsules"


PUBLIC_CAPSULES = _public_capsules()
PY = sys.executable
# Driver-side sim toolchain env — resolve via ext_path('chipyard') (honors .env), NOT a hard-coded
# path. This is the host-side broker (runs sims OUTSIDE the sandbox), so it must find spike/riscv-gcc
# in the conda env itself; the previous '/path/to/...' placeholder left spike off PATH -> L2 n=0/1.
try:
    from merlin.common.paths import ext_path as _ext_path
    _CY = _ext_path("chipyard")
    CE = str(_CY / ".conda-env") if _CY else "/path/to/chipyard/.conda-env"
except Exception:  # noqa: BLE001 — keep the broker importable even if merlin isn't on the path
    CE = "/path/to/chipyard/.conda-env"
GLOBAL_VERIL_SLOTS = Path("/tmp/merlin_veril_slots")   # cross-arm verilator semaphore
_CAP_RE = re.compile(r"^[A-Za-z0-9_]+$")
# debug-flag whitelist: symbolic name -> (currently a no-op passthrough; real sim args wired later).
DEBUG_WHITELIST = {"trace", "cycles", "verbose"}


def _valid_capsules(spec: str) -> list[str] | None:
    if spec == "all":
        return ["all"]
    names = [c.strip() for c in spec.split(",") if c.strip()]
    out = []
    for n in names:
        if not _CAP_RE.match(n) or not (PUBLIC_CAPSULES / n).is_dir():
            return None                                # reject ../, globs, unknown names
        out.append(n)
    return out or None


def _sim_env() -> dict:
    e = dict(os.environ)
    e["PATH"] = f"{CE}/bin:{CE}/riscv-tools/bin:" + e.get("PATH", "")
    e["RISCV"] = f"{CE}/riscv-tools"
    # .compat_lib first: the conda cmake needs libidn.so.11 (host has only .12) during C++ build configure
    compat = str(HERE.parents[3] / ".compat_lib")   # scripts -> capsule_bench_v0 -> experiments -> <repo>/.compat_lib
    e["LD_LIBRARY_PATH"] = f"{compat}:{CE}/lib:{CE}/riscv-tools/lib:" + e.get("LD_LIBRARY_PATH", "")
    return e


def _strip_golden(obj):
    """Defence-in-depth: drop any 'expected'/'golden' keys before publishing (agent_selfcheck already
    redacts; this guarantees it even if a future change regresses)."""
    if isinstance(obj, dict):
        return {k: _strip_golden(v) for k, v in obj.items() if k not in ("expected", "golden")}
    if isinstance(obj, list):
        return [_strip_golden(x) for x in obj]
    return obj


def _veril_acquire(n_slots: int) -> Path | None:
    GLOBAL_VERIL_SLOTS.mkdir(parents=True, exist_ok=True)
    for i in range(n_slots):
        slot = GLOBAL_VERIL_SLOTS / f"slot_{i}"
        try:
            fd = os.open(str(slot), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            return slot
        except FileExistsError:
            continue
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--veril-slots", type=int, default=2)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--per-capsule-timeout", type=int, default=0, help="0=read .oracle_timing.json or default")
    a = ap.parse_args(argv)
    ws = Path(a.ws); ch = ws / ".qa_channel"; ch.mkdir(parents=True, exist_ok=True)

    # verilator per-capsule timeout: measured (Part 3) or a generous default
    vpc = a.per_capsule_timeout
    if vpc <= 0:
        tf = HERE / ".oracle_timing.json"
        try:
            vpc = max(900, int(2 * json.loads(tf.read_text())["verilator_per_capsule_s"]))
        except Exception:
            vpc = 1200

    running: dict[str, dict] = {}          # jid -> {proc, slot, resp_tmp, sim}
    claimed: set[str] = set()
    while True:
        if (ch / "STOP").exists():
            break
        # honor cancels FIRST — a stale full-suite verilator job otherwise holds its slot for hours
        # (the sole cycle-exact oracle wedges) with the agent unable to free it. Kills a RUNNING job
        # (releasing its slot) or voids a QUEUED one; the shim reports state 'canceled'.
        for csent in sorted(ch.glob("simcancel_*")):
            jid = csent.name[len("simcancel_"):]
            if jid in running:
                j = running.pop(jid)
                j["proc"].kill()
                if j["slot"]:
                    j["slot"].unlink(missing_ok=True)
            elif (ch / f"simdone_{jid}").exists() or (ch / f"simerr_{jid}").exists():
                csent.unlink(missing_ok=True)   # finished before the cancel landed — result stands
                continue
            else:
                claimed.add(jid)                # queued: never launch it
            (ch / f"simresp_{jid}.json").write_text(json.dumps(
                {"state": "canceled", "all_pass": False, "error": None,
                 "note": "canceled on request; the simulator slot is free for your next submit"}, indent=2))
            (ch / f"simcanceled_{jid}").write_text("ok")
            csent.unlink(missing_ok=True)
        # reap finished
        for jid, j in list(running.items()):
            if j["proc"].poll() is None:
                continue
            rc = j["proc"].returncode
            resp = ch / f"simresp_{jid}.json"
            tmp = j["resp_tmp"]
            try:
                out = _strip_golden(json.loads(Path(tmp).read_text())) if Path(tmp).exists() else \
                    {"error": "no verdict produced", "all_pass": False}
            except Exception as e:
                out = {"error": f"verdict parse: {e}", "all_pass": False}
            if rc == 124:                                  # timeout exit
                out = {"sim": j["sim"], "state": "timeout", "all_pass": False,
                       "error": f"{j['sim']} exceeded its time budget"}
            resp.write_text(json.dumps(out, indent=2))
            (ch / (f"simerr_{jid}" if out.get("error") else f"simdone_{jid}")).write_text("ok")
            if j["slot"]:
                j["slot"].unlink(missing_ok=True)
            running.pop(jid)
        # launch queued (respect local + global caps)
        if len(running) < a.max_jobs:
            for req in sorted(ch.glob("simreq_*.json")):
                jid = req.stem[len("simreq_"):]
                if jid in claimed:
                    continue
                if len(running) >= a.max_jobs:
                    break
                r = json.loads(req.read_text()) if req.exists() else {}
                sim = r.get("sim")
                caps = _valid_capsules(str(r.get("capsules", "all")))
                if sim not in ("spike", "verilator", "vcs") or caps is None:
                    (ch / f"simresp_{jid}.json").write_text(json.dumps(
                        {"error": "rejected: bad sim or capsule (constrained runner)", "all_pass": False}))
                    (ch / f"simerr_{jid}").write_text("rejected"); claimed.add(jid); continue
                slot = None
                if sim == "verilator":
                    slot = _veril_acquire(a.veril_slots)
                    if slot is None:
                        continue                            # global verilator budget full; try later
                workers = max(1, min(int(r.get("workers", 1)), 2 if sim == "verilator" else 8))
                capspec = "all" if caps == ["all"] else ",".join(caps)
                # 'all' = the ACTUAL public-set size, not a legacy literal: under-counting (20 vs a
                # 37-capsule corpus) under-budgets the wall-clock cap and kills the job mid-suite.
                ncaps = (sum(1 for _ in PUBLIC_CAPSULES.glob("*/capsule.yaml")) or 20) \
                    if caps == ["all"] else len(caps)
                to = (vpc * ncaps) if sim == "verilator" else 900
                resp_tmp = ch / f"simtmp_{jid}.json"
                argv2 = [PY, str(SELFCHECK), "--submission", str(ws / "submission"),
                         "--sim", sim, "--capsules", capspec, "--workers", str(workers),
                         "--timeout", str(to), "--out", str(resp_tmp)]
                (ch / f"simrun_{jid}").write_text("running")
                proc = subprocess.Popen(["timeout", str(to + 120)] + argv2, cwd=str(ws),
                                        env=_sim_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                running[jid] = {"proc": proc, "slot": slot, "resp_tmp": str(resp_tmp), "sim": sim}
                claimed.add(jid)
        time.sleep(a.poll)
    # drain on STOP
    for j in running.values():
        j["proc"].kill()
        if j["slot"]:
            j["slot"].unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
