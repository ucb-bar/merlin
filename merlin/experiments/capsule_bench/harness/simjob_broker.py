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
def _veril_slots_dir() -> Path:
    """The cross-arm verilator semaphore directory, PER USER.

    This was a fixed ``/tmp/merlin_veril_slots``. ``/tmp`` is world-writable with the sticky bit, so the
    FIRST user on the host to run a bench owns that directory at mode 0775 -- and every other user's
    broker is then locked out of creating a slot inside it. ``mkdir(exist_ok=True)`` hides the problem
    completely: the mkdir succeeds because the directory already exists, and the PermissionError only
    surfaces later, on the slot file, where it read as "the L3 infrastructure crashed".

    Measured on the live round merlincirct_arm4_func_20260901_v4: the directory belonged to a different
    user, so this host's agent could not acquire a verilator slot and therefore could not run L3 AT ALL
    for two entire rounds. Zero of ~600 agent commands mentioned verilator; GM0/GM1 sat failing at L3
    with no way for the agent to reproduce them; and the agent's own round report said "L3
    infrastructure crashed on permission to /tmp/merlin_veril_slots/slot_0".

    The semaphore only ever needed to keep THIS user's arms from oversubscribing verilator against each
    other, never to coordinate across users. So honour ``TMPDIR`` -- which this project sets per user and
    on the large filesystem, where working files are supposed to live -- and qualify by uid so the path
    is still unsquattable if TMPDIR is unset or shared.
    """
    base = os.environ.get("TMPDIR") or "/tmp"
    return Path(base) / f"merlin_veril_slots_{os.getuid()}"


GLOBAL_VERIL_SLOTS = _veril_slots_dir()                # cross-arm (same-user) verilator semaphore
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


import tier_promote as _TP
from tier_promote import promote as _promote, resolve_tiers as _resolve_tiers  # noqa: E402


def _strip_golden(obj):
    """Defence-in-depth: drop any 'expected'/'golden' keys before publishing (agent_selfcheck already
    redacts; this guarantees it even if a future change regresses)."""
    if isinstance(obj, dict):
        return {k: _strip_golden(v) for k, v in obj.items() if k not in ("expected", "golden")}
    if isinstance(obj, list):
        return [_strip_golden(x) for x in obj]
    return obj


class VerilSlotsUnusable(RuntimeError):
    """The slot directory cannot be used at all -- distinct from every slot being busy."""


def _veril_acquire(n_slots: int) -> Path | None:
    """A free slot, or None when they are all BUSY.

    "All busy" and "the directory is not usable" must not look alike. Only FileExistsError used to be
    caught, so a PermissionError propagated out of the broker and killed the job with a bare traceback --
    the agent read that as the oracle being broken, which is exactly what it looked like. A directory we
    cannot write is a configuration fault: name it, and name the remedy.
    """
    try:
        GLOBAL_VERIL_SLOTS.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise VerilSlotsUnusable(
            f"cannot create the verilator slot directory {GLOBAL_VERIL_SLOTS}: {e}") from e
    if not os.access(GLOBAL_VERIL_SLOTS, os.W_OK):
        import pwd
        try:
            owner = pwd.getpwuid(GLOBAL_VERIL_SLOTS.stat().st_uid).pw_name
        except Exception:  # noqa: BLE001
            owner = "another user"
        raise VerilSlotsUnusable(
            f"the verilator slot directory {GLOBAL_VERIL_SLOTS} exists but is not writable by "
            f"{os.getuid()} (owner: {owner}). No verilator slot can be acquired, so no L3 job can run. "
            f"Set TMPDIR to a directory you own (this project uses a per-user one under /scratch).")
    for i in range(n_slots):
        slot = GLOBAL_VERIL_SLOTS / f"slot_{i}"
        try:
            fd = os.open(str(slot), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            return slot
        except FileExistsError:
            continue                                   # busy: try the next one
        except PermissionError as e:
            raise VerilSlotsUnusable(
                f"cannot create {slot}: {e}. No L3 job can run until the slot directory is writable.") from e
    return None


def _sim_via() -> str:
    """This target's declared simulator route, from its own descriptor (empty = in-process RTL model)."""
    import yaml

    def _find(node):
        if isinstance(node, dict):
            v = node.get("sim_via")
            if isinstance(v, str):
                return v.strip()
            for x in node.values():
                r = _find(x)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for x in node:
                r = _find(x)
                if r is not None:
                    return r
        return None
    try:
        import _common as _C
        d = yaml.safe_load((_C.EXP / "target_experiment.yaml").read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 -- an unreadable descriptor means "no bespoke sim", fail closed
        return ""
    return _find(d) or ""


def _allowed_sims() -> tuple[str, ...]:
    """The sim names a request may name — STILL a closed allowlist (this is load-bearing isolation:
    an agent must not be able to make the broker run something arbitrary), but DERIVED rather than
    baked to one target's ladder.

    A chipyard target selects along the spike/verilator/vcs ladder. A target that declares no bespoke
    sim grades on its own contract-resolved tier, where ``--sim`` does not apply at all
    (``agent_selfcheck._adapters``) -- so the only accepted token is the neutral sentinel below, and it
    is NOT forwarded to the self-check. Without this, such a target's agent could reach no oracle from
    inside the sandbox while every gate reported the sandbox healthy.
    """
    if _sim_via() != "chipyard":
        return (_NEUTRAL_SIM,)
    # DERIVE the elaborated-RTL engines from the engine policy instead of restating one ladder here.
    # The policy is the single place that knows which engines exist and in what order; a second literal
    # tuple in this file is how `gsim` came to be unreachable from inside the sandbox after the backend
    # already supported it. The screen tier (spike) is not an elaborated-RTL engine, so it is named
    # separately. Fail CLOSED: if the policy cannot be imported, offer only what was always offered --
    # never widen the allowlist on an error path (this list is load-bearing isolation).
    try:
        from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY
    except Exception:  # noqa: BLE001 -- no policy module: keep the historical ladder, do not widen
        return ("spike", "verilator", "vcs")
    return ("spike",) + tuple(ENGINE_PRIORITY)


_NEUTRAL_SIM = "contract"
_LOOP_TIER = None      # resolved in main() from the target's ladder
_CERT_TIER = None
_COVER = None   # "grade on whatever tier this target's contract resolves to"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--veril-slots", type=int, default=2)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--per-capsule-timeout", type=int, default=0, help="0=read .oracle_timing.json or default")
    a = ap.parse_args(argv)
    ws = Path(a.ws); ch = ws / ".qa_channel"; ch.mkdir(parents=True, exist_ok=True)

    # Which tier is the cheap gate and which is the expensive cert, DERIVED from this target's own
    # adapter map -- never a literal, so a target with a different ladder gets the right split with no
    # edit here. loop = the fastest tier the corpus declares; cert = the deepest reachable above it.
    # Promotion is simply disabled (both None) when the endpoint exposes only one tier, rather than
    # inventing a second.
    global _LOOP_TIER, _CERT_TIER, _COVER
    _LOOP_TIER, _CERT_TIER, _COVER = _resolve_tiers(ws)
    print(f"[promote] loop={_LOOP_TIER} cert={_CERT_TIER} "
          f"cover={len(_COVER) if _COVER is not None else 'all'}", file=sys.stderr, flush=True)

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
    # STOP alone does not bound this broker's life: the sentinel is written by the driver, so if the
    # driver dies first nobody ever writes it and the broker polls forever -- while HOLDING sim slots and
    # child simulators. Three sibling brokers were found orphaned to init hours after their run ended,
    # spawned for a round that never started. Exit when the process that started us is gone.
    orig_ppid = os.getppid()
    while True:
        if (ch / "STOP").exists() or os.getppid() != orig_ppid:
            break
        # reap finished
        for jid, j in list(running.items()):
            if j["proc"].poll() is None:
                continue
            rc = j["proc"].returncode
            resp = ch / f"simresp_{jid}.json"
            tmp = j["resp_tmp"]
            try:
                out = _strip_golden(json.loads(Path(tmp).read_text())) if Path(tmp).exists() else \
                    {"error": f"no verdict produced (rc={rc}); child output in "
                              f"simlog_{jid}.txt", "all_pass": False}
            except Exception as e:
                out = {"error": f"verdict parse: {e}", "all_pass": False}
            if rc == 124:                                  # timeout exit
                out = {"sim": j["sim"], "state": "timeout", "all_pass": False,
                       "error": f"{j['sim']} exceeded its time budget"}
            resp.write_text(json.dumps(out, indent=2))
            (ch / (f"simerr_{jid}" if out.get("error") else f"simdone_{jid}")).write_text("ok")
            # PROMOTE: a capsule that just passed the loop tier earns the cert tier now, not at a round
            # boundary hours away. Skipped for a job that WAS a promotion, so a cert verdict cannot
            # re-enqueue itself.
            if not j.get("promoted") and not out.get("error"):
                try:
                    _promote(ws, ch, out, _LOOP_TIER, _CERT_TIER, _COVER, sys.stderr)
                except Exception as _pe:  # noqa: BLE001 -- promotion is an optimisation, never a gate
                    print(f"[promote] skipped: {type(_pe).__name__}: {_pe}", file=sys.stderr, flush=True)
            elif j.get("promoted") and not out.get("error"):
                # A promotion's own verdict: record it, so the cert this just PAID FOR on real RTL is
                # kept instead of discarded. Without this the capsule stays `pending` forever and the
                # same bytes are re-certified on the next loop verdict.
                try:
                    # Forward the identity the promotion was ENQUEUED for, so the result lands on the
                    # exact record it belongs to even when the verdict reader produced no per-capsule
                    # artifact identity. Absent (a request written before this field existed) it is
                    # None, and the recorder keeps its previous attribution rule.
                    _TP.record_cert(ws, out, _CERT_TIER, sys.stderr, identity=j.get("identity"))
                except Exception as _re:  # noqa: BLE001 -- recording must never gate a run either
                    print(f"[promote] record skipped: {type(_re).__name__}: {_re}",
                          file=sys.stderr, flush=True)
            if j.get("log"):
                try:
                    j["log"].close()
                except Exception:  # noqa: BLE001 -- closing a log must never break the reap
                    pass
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
                if sim not in _allowed_sims() or caps is None:
                    # Say WHICH of the two it was, and what would be accepted. The old message named
                    # neither, and a request rejected without a remedy reads as "the oracle is broken":
                    # measured on a live run, an agent submitted twice, was rejected twice with this
                    # text, and never used the async oracle again -- while the arm that DID find the
                    # async path used it 98 times in the round its score moved 17 -> 26. An unhelpful
                    # rejection costs more than the check it protects.
                    _allowed = _allowed_sims()
                    if sim not in _allowed:
                        _why = (f"--sim {sim!r} is not accepted for this target. Use "
                                f"{' or '.join(repr(s) for s in _allowed)}"
                                + (f"; this target's tier comes from its contract, so {_NEUTRAL_SIM!r} "
                                   f"means 'grade on whatever tier the contract resolves to' and the "
                                   f"tier itself is chosen with --tiers"
                                   if _allowed == (_NEUTRAL_SIM,) else ""))
                    else:
                        _named = str(r.get("capsules", "all"))
                        _why = (f"--capsules {_named!r} named something this runner will not run: a "
                                f"capsule must be an existing public capsule directory name (letters, "
                                f"digits, underscore), or 'all'")
                    (ch / f"simresp_{jid}.json").write_text(json.dumps(
                        {"error": f"rejected: {_why}", "all_pass": False,
                         "rejected_field": "sim" if sim not in _allowed else "capsules"}))
                    (ch / f"simerr_{jid}").write_text("rejected"); claimed.add(jid); continue
                slot = None
                if sim == "verilator":
                    slot = _veril_acquire(a.veril_slots)
                    if slot is None:
                        continue                            # global verilator budget full; try later
                workers = max(1, min(int(r.get("workers", 1)), 2 if sim == "verilator" else 8))
                capspec = "all" if caps == ["all"] else ",".join(caps)
                ncaps = len(_valid_capsules("all") or []) if caps == ["all"] else len(caps)
                to = (vpc * ncaps) if sim == "verilator" else 900
                resp_tmp = ch / f"simtmp_{jid}.json"
                argv2 = [PY, str(SELFCHECK), "--submission", str(ws / "submission"),
                         "--capsules", capspec, "--workers", str(workers),
                         "--timeout", str(to), "--out", str(resp_tmp)]
                if sim != _NEUTRAL_SIM:      # --sim is meaningless where the contract picks the tier
                    argv2[4:4] = ["--sim", sim]
                _tiers = str(r.get("tiers") or "").strip()
                if _tiers:                   # validated downstream against the RESOLVED adapter map
                    argv2 += ["--tiers", _tiers]
                (ch / f"simrun_{jid}").write_text("running")
                # KEEP THE CHILD'S OUTPUT. This was stdout/stderr=DEVNULL, so when a job exited without
                # writing its verdict file the only trace was the broker's own "no verdict produced" --
                # a job that failed and a job that produced nothing were indistinguishable, and the
                # REASON was gone. Measured on merlincirct_arm4_func_20260901_v4: 19 promotion jobs
                # answered "no verdict produced" with no diagnostic anywhere on disk. The log is
                # per-job, beside the response, so a failure can be read after the fact.
                job_log = (ch / f"simlog_{jid}.txt").open("wb")
                proc = subprocess.Popen(["timeout", str(to + 120)] + argv2, cwd=str(ws),
                                        env=_sim_env(), stdout=job_log, stderr=subprocess.STDOUT)
                running[jid] = {"proc": proc, "slot": slot, "resp_tmp": str(resp_tmp), "sim": sim,
                                "promoted": bool(r.get("promoted")), "log": job_log,
                                # which tier-state record this promotion was launched for (see the reap)
                                "identity": r.get("identity")}
                claimed.add(jid)
        time.sleep(a.poll)
    # drain on STOP
    for j in running.values():
        j["proc"].kill()
        if j["slot"]:
            j["slot"].unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
