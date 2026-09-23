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
import shutil
import subprocess
import sys
import time
from pathlib import Path

from merlin_experiments.phase1.context import InvocationContext, add_context_arguments, resolve_context
from merlin_experiments.phase1.feedback import promotion as _TP
from merlin_experiments.phase1.feedback.dispatch import allowed_sims as _allowed_sims
from merlin_experiments.phase1.feedback.promotion import promote as _promote  # noqa: E402
from merlin_experiments.phase1.feedback.promotion import resolve_tiers as _resolve_tiers
from merlin_experiments.phase1.feedback.selfcheck import _public_capsules, worker_command
from merlin_experiments.phase1.feedback.snapshots import recover_promotion_snapshot

PY = sys.executable


def _conda_env() -> str:
    # Resolve only after explicit invocation initialization sources experiment.env.
    try:
        from merlin.common.paths import ext_path

        chipyard = ext_path("chipyard")
        return str(chipyard / ".conda-env") if chipyard else "/path/to/chipyard/.conda-env"
    except Exception:
        return "/path/to/chipyard/.conda-env"


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


GLOBAL_VERIL_SLOTS = _veril_slots_dir()  # cross-arm (same-user) verilator semaphore
# debug-flag whitelist: symbolic name -> (currently a no-op passthrough; real sim args wired later).
DEBUG_WHITELIST = {"trace", "cycles", "verbose"}


def _valid_capsules(spec: str, capsules_root: Path) -> list[str] | None:
    if spec == "all":
        return ["all"]
    names = [c.strip() for c in spec.split(",") if c.strip()]
    out = []
    for n in names:
        if not all(c.isascii() and (c.isalnum() or c == "_") for c in n) or not (capsules_root / n).is_dir():
            return None  # reject ../, globs, unknown names
        out.append(n)
    return out or None


def _sim_env(context: InvocationContext) -> dict:
    e = dict(os.environ)
    CE = _conda_env()
    # THE SUBMISSION'S OWN INTERPRETER COMES FIRST. A submission entrypoint is a script with a
    # `#!/usr/bin/env python3` shebang, so whatever `python3` PATH resolves to is what compiles the
    # capsule. Prepending the sim toolchain's conda env put ITS python3 in front of the harness's, and
    # that interpreter has none of the compiler's dependencies -- so every cert job died in its FIRST
    # declared command with `ModuleNotFoundError`, before any RTL ran.
    #
    # Measured on merlincirct_g4p1_20260905: 41 of 41 promoted L3 jobs returned
    # `parse rc=1: ModuleNotFoundError: No module named 'xdsl'`, all with `barrier_status: None` and no
    # execution identity -- so every one was then discarded as unattributable, and the capsule went back
    # to `pending`. The sibling self-check broker launches the same child with the inherited environment
    # and works, which is why this was invisible: only the PROMOTED path was broken.
    #
    # The conda env still supplies spike / riscv-gcc / the RTL engines; it just no longer shadows the
    # interpreter the harness itself is running under.
    e["PATH"] = f"{os.path.dirname(PY)}:{CE}/bin:{CE}/riscv-tools/bin:" + e.get("PATH", "")
    e["RISCV"] = f"{CE}/riscv-tools"
    # .compat_lib first: the conda cmake needs libidn.so.11 (host has only .12) during C++ build configure
    compat = str(context.repo / ".compat_lib")  # scripts -> capsule_bench_v0 -> experiments -> <repo>/.compat_lib
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


def _completed_job_ids(ch: Path) -> set[str]:
    """Recover jobs whose durable response was already published by an earlier broker process."""
    prefix = "simresp_"
    return {p.stem[len(prefix) :] for p in ch.glob(f"{prefix}*.json")}


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
        raise VerilSlotsUnusable(f"cannot create the verilator slot directory {GLOBAL_VERIL_SLOTS}: {e}") from e
    if not os.access(GLOBAL_VERIL_SLOTS, os.W_OK):
        import pwd

        try:
            owner = pwd.getpwuid(GLOBAL_VERIL_SLOTS.stat().st_uid).pw_name
        except Exception:  # noqa: BLE001
            owner = "another user"
        raise VerilSlotsUnusable(
            f"the verilator slot directory {GLOBAL_VERIL_SLOTS} exists but is not writable by "
            f"{os.getuid()} (owner: {owner}). No verilator slot can be acquired, so no L3 job can run. "
            f"Set TMPDIR to a directory you own (this project uses a per-user one under /scratch)."
        )
    for i in range(n_slots):
        slot = GLOBAL_VERIL_SLOTS / f"slot_{i}"
        for _attempt in (0, 1):  # second pass runs only after a stale reclaim
            try:
                fd = os.open(str(slot), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return slot
            except FileExistsError:
                if _attempt == 0 and _reclaim_if_stale(slot):
                    continue  # holder is gone; retry this same slot once
                break  # genuinely busy: try the next one
            except PermissionError as e:
                raise VerilSlotsUnusable(
                    f"cannot create {slot}: {e}. No L3 job can run until the slot directory is writable."
                ) from e
    return None


def _reclaim_if_stale(slot: Path) -> bool:
    """Free ``slot`` if the process that wrote its pid is gone. True when it was reclaimed.

    A slot is released by the broker that took it (``slot.unlink``), so a broker that dies without
    running its cleanup leaks the slot FOREVER -- and the acquire loop reads a leaked slot as "busy".
    With the default two slots, two such deaths disable verilator L3 for every future run by this user,
    silently: the loop simply returns None and the agent sees L3 never happening. MEASURED 2026-09-08 --
    both slots were held by pid 2187933, dead since 2026-09-02, and nothing would ever have released
    them. The host's memory monitor killing a broker is enough to cause this, which is not a rare event.

    Deliberately narrow. It reclaims ONLY when the pid file is readable, parses as an integer, and that
    pid does not exist; an unreadable or malformed slot is left alone rather than guessed at, because
    stealing a slot from a LIVE verilator is worse than failing to acquire one -- it would oversubscribe
    the exact resource this semaphore exists to bound. pid reuse is possible in principle; a reused pid
    reads as live, so the slot stays held, which fails safe in the same direction.
    """
    try:
        holder = slot.read_text().strip()
    except OSError:
        return False
    if not holder.isdigit():
        return False
    try:
        os.kill(int(holder), 0)
    except ProcessLookupError:
        pass  # holder is gone -- the slot is stale
    except PermissionError:
        return False  # alive and owned by someone else: leave it
    except OSError:
        return False
    else:
        return False  # holder is alive: genuinely busy
    try:
        slot.unlink()
    except OSError:
        return False  # another broker reclaimed it first
    print(f"[simjob] reclaimed stale verilator slot {slot.name} (holder pid {holder} is gone)", flush=True)
    return True


_NEUTRAL_SIM = "contract"

# --- cert-tier time budget: DERIVED from the engine that will actually run --------------------------
# A cert job's wall clock has to come from the cost law of the engine that serves the cert tier. It used
# to come from `.oracle_timing.json::verilator_per_capsule_s` -- a VERILATOR measurement -- and the
# elaborated-RTL launch below did not even use that: every non-verilator engine got a bare 900 s. Since
# `rtl_engine_policy` ranks `gsim` above `verilator` on cost, and the two engines' laws are ~1000x apart
# per cycle (measured on this repo's largest corpus: 0.00024 s/cycle over a 41 s floor from 135 samples,
# against 0.229 s/cycle over a 55 s floor from 34), the mis-derivation was the NORMAL path, not an edge
# case: the six heaviest DRAM movers in a 97-capsule batch were abandoned at exactly 900 s in every arm,
# and an abandoned cert reads downstream like a numeric defect.
#
# Raising the number alone would be wrong -- that just lets a pathological program burn more budget --
# so the budget is SIZED from the fit, with both margins stated:
#   * cycles: predict at 2x the engine's measured maximum cycle count. Sizing past that is
#     extrapolation, and 2x is the same ceiling `cert_cost.max_cycles_within` already honours.
#   * seconds: x1.5, because the fit's own leave-one-out error is p90 31% / worst 51%, so a budget at
#     the predicted value alone abandons capsules that were in fact affordable.
_CERT_TIMEOUT_FLOOR_S = 900  # never size BELOW what the historical default allowed
_CERT_TIMEOUT_FALLBACK_S = 1200  # no fit and no measurement on disk: the historical generous default
_SCREEN_TIMEOUT_S = 900  # the functional screen / contract tier keeps its historical budget
_CERT_CYCLE_CEILING = 2.0
_CERT_FIT_MARGIN = 1.5


def _rtl_engines() -> tuple[str, ...]:
    """The elaborated-RTL engines in the policy's own cost order.

    Fail CLOSED to the historical pair when the policy cannot be imported, for the same reason
    `_allowed_sims` does: an error path must not invent an engine, and it must not silently drop the
    engines that were always there.
    """
    try:
        from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY
    except Exception:  # noqa: BLE001 -- no policy module: keep the historical ladder
        return ("vcs", "verilator")
    return tuple(ENGINE_PRIORITY)


def _cert_budget_s(target: str) -> tuple[int | None, str]:
    """``(seconds, why)`` for one capsule's cert tier, or ``(None, why)`` when nothing supports a number.

    NO AVAILABILITY PROBE HAPPENS HERE. `rtl_engine_policy.select` is the authority on which engine
    runs, but its probes locate/elaborate simulators and may raise, and this function is called by a
    broker that holds sim slots and must never die. So the engine is resolved from the two facts already
    on disk:

      1. ``MERLIN_REQUIRED_RTL_ENGINE`` -- when the experiment PINS an engine that IS the engine, and it
         is the same pin `_allowed_sims` already honours for the request surface.
      2. otherwise, the first engine in the policy's order that this target has measured cert samples
         for. An engine with no samples here has never certified this target, so it is not the engine to
         size against.

    `cert_cost` refuses rather than guessing, and that refusal is preserved: an unmeasured target
    returns None and the caller keeps its historical default rather than inventing a cost law.
    """
    try:
        from merlin.targetgen import cert_cost
    except Exception as exc:  # noqa: BLE001
        return None, f"cert_cost not importable ({type(exc).__name__}: {exc})"
    pinned = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
    try:
        if pinned:
            engine, why = pinned, "pinned by MERLIN_REQUIRED_RTL_ENGINE"
            fit = cert_cost.fit_cycles_for(str(target), engine=engine)
        else:
            fits = cert_cost.fits_cycles_for(str(target))
            engine = next((e for e in _rtl_engines() if e in fits), None)
            why = (
                "first policy-order engine with measured cert samples"
                if engine
                else "no measured cert samples on any policy engine"
            )
            fit = fits.get(engine) if engine else None
    except Exception as exc:  # noqa: BLE001 -- unreadable history is "no fit", never a broker crash
        return None, f"cert cost history unreadable ({type(exc).__name__}: {exc})"
    if fit is None:
        return None, f"{why}: no cycle-cost fit for engine {engine!r} on target {target!r}"
    cycles = int(fit.cycles_max * _CERT_CYCLE_CEILING)
    try:
        secs = cert_cost.predict_seconds_from_cycles(fit, cycles)
    except Exception as exc:  # noqa: BLE001
        return None, f"{engine} fit unusable ({type(exc).__name__}: {exc})"
    if not secs or secs <= 0:
        return None, f"{engine} fit predicted no positive cost for {cycles} cycles"
    return int(_CERT_FIT_MARGIN * float(secs)), (
        f"{engine} ({why}): {fit.intercept_s:.1f}s + {fit.per_cycle_s:.6f}s/cycle x {cycles} cycles "
        f"(={_CERT_CYCLE_CEILING:g}x the measured max, n={fit.n_samples}, r2={fit.r2:.2f}) "
        f"x {_CERT_FIT_MARGIN:g} margin"
    )


def _spawn_selfcheck(argv, *, cwd, env, job_log, timeout_s):
    """Launch one asynchronous simulator check without inheriting the campaign terminal.

    ``timeout`` creates a separate process group for the command it supervises. When that group is
    behind the campaign's foreground process group, any simulator probe that reads inherited stdin is
    stopped by the kernel with SIGTTIN. The broker then mistakes a job that never ran for an RTL
    timeout. Simulator checks are non-interactive, so an explicit EOF is the only truthful stdin.
    """
    from merlin_experiments.frozen_python import inherited_python_command

    return subprocess.Popen(
        ["timeout", str(timeout_s + 120)] + inherited_python_command(argv),
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=job_log,
        stderr=subprocess.STDOUT,
    )


def _per_capsule_timeout(
    requested: int, *, context: InvocationContext, timing_file: Path | None = None
) -> tuple[int, str]:
    """``(seconds, why)``: the per-capsule elaborated-RTL budget this broker will use.

    Order, and each step's reason for existing:

    1. an explicit ``--per-capsule-timeout`` -- an operator overrides derivation, always.
    2. the cost law of the engine that will actually run (:func:`_cert_budget_s`), FLOORED at
       :data:`_CERT_TIMEOUT_FLOOR_S` so a cheap fit can never shorten the budget below what the old
       default already allowed. Derivation is allowed to raise the budget, never to lower it.
    3. the recorded verilator measurement, i.e. exactly what this used to do -- kept so a target with
       no cert history behaves as it did rather than losing its budget to the new code path.
    4. the historical constant, when even that measurement is absent.
    """
    if requested and int(requested) > 0:
        return int(requested), "requested on the command line (--per-capsule-timeout)"
    derived, why = _cert_budget_s(str(context.target or ""))
    if derived:
        return max(_CERT_TIMEOUT_FLOOR_S, int(derived)), f"derived from {why}"
    tf = timing_file
    try:
        measured = int(2 * json.loads(tf.read_text())["verilator_per_capsule_s"])
    except Exception:  # noqa: BLE001 -- no measurement either: the historical constant
        return _CERT_TIMEOUT_FALLBACK_S, f"no engine cost law and no recorded measurement ({why})"
    return max(_CERT_TIMEOUT_FLOOR_S, measured), (
        f"no engine cost law ({why}); fell back to the recorded verilator measurement"
    )


def _job_timeout_s(sim: str, ncaps: int, vpc: int) -> int:
    """Wall clock for ONE sim job: the derived per-capsule budget for every elaborated-RTL engine.

    This read ``(vpc * ncaps) if sim == "verilator" else 900``, so the engine `rtl_engine_policy`
    actually prefers was handed a flat 900 s no matter how large the request or how expensive the
    engine -- the literal "timed out after 900 seconds" that abandoned the heaviest capsules. The
    functional screen and the contract sentinel are not elaborated RTL and keep their own budget.
    """
    return (max(1, int(vpc)) * max(1, int(ncaps))) if sim in _rtl_engines() else _SCREEN_TIMEOUT_S


def main(
    argv=None,
    *,
    context=None,
    capsules_root: Path | None = None,
    contract: Path | None = None,
    timing_file: Path | None = None,
    policy_capsules_root: Path | None = None,
):
    ap = argparse.ArgumentParser()
    add_context_arguments(ap)
    ap.add_argument("--capsules-root", type=Path)
    ap.add_argument("--contract", type=Path)
    ap.add_argument("--timing-file", type=Path)
    ap.add_argument(
        "--policy-capsules-root", type=Path, help="frozen descriptor corpus for promotion, not the QA subset"
    )
    ap.add_argument("--ws", required=True)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--veril-slots", type=int, default=2)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument(
        "--per-capsule-timeout", type=int, default=0, help="0=derive from the cert engine's own measured cost law"
    )
    a = ap.parse_args(argv)
    context = resolve_context(a, ap, context)
    capsules_root = a.capsules_root if a.capsules_root is not None else capsules_root
    promotion_capsules_root = a.policy_capsules_root if a.policy_capsules_root is not None else policy_capsules_root
    if capsules_root is None:
        capsules_root = _public_capsules(context)
    contract = a.contract if a.contract is not None else contract
    timing_file = a.timing_file if a.timing_file is not None else timing_file
    if timing_file is None and context.harness is not None:
        timing_file = context.harness / ".oracle_timing.json"
    ws = Path(a.ws)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)

    # Which tier is the cheap gate and which is the expensive cert, DERIVED from this target's own
    # adapter map -- never a literal, so a target with a different ladder gets the right split with no
    # edit here. loop = the fastest tier the corpus declares; cert = the deepest reachable above it.
    # Promotion is simply disabled (both None) when the endpoint exposes only one tier, rather than
    # inventing a second.
    _LOOP_TIER, _CERT_TIER, _COVER = _resolve_tiers(ws, context=context, capsules_root=promotion_capsules_root)
    print(
        f"[promote] loop={_LOOP_TIER} cert={_CERT_TIER} cover={len(_COVER) if _COVER is not None else 'all'}",
        file=sys.stderr,
        flush=True,
    )

    vpc, _budget_why = _per_capsule_timeout(a.per_capsule_timeout, context=context, timing_file=timing_file)
    print(f"[budget] per-capsule cert timeout {vpc}s: {_budget_why}", file=sys.stderr, flush=True)

    running: dict[str, dict] = {}  # jid -> {proc, slot, resp_tmp, sim}
    # One broker is created per round, but the channel and its queue survive across rounds.  Starting
    # from an empty set replays every historical request, including paid L3 work.  A response is the
    # durable commit record; a bare simrun marker is not (the previous broker may have died mid-job), so
    # only responses are recovered here and interrupted requests remain retryable.
    claimed: set[str] = _completed_job_ids(ch)
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
                out = (
                    _strip_golden(json.loads(Path(tmp).read_text()))
                    if Path(tmp).exists()
                    else {
                        "error": f"no verdict produced (rc={rc}); child output in simlog_{jid}.txt",
                        "all_pass": False,
                    }
                )
            except Exception as e:
                out = {"error": f"verdict parse: {e}", "all_pass": False}
            if rc == 124:  # timeout exit
                out = {
                    "sim": j["sim"],
                    "state": "timeout",
                    "all_pass": False,
                    "error": f"{j['sim']} exceeded its time budget",
                }
            resp.write_text(json.dumps(out, indent=2))
            (ch / (f"simerr_{jid}" if out.get("error") else f"simdone_{jid}")).write_text("ok")
            # PROMOTE: a capsule that just passed the loop tier earns the cert tier now, not at a round
            # boundary hours away. Skipped for a job that WAS a promotion, so a cert verdict cannot
            # re-enqueue itself.
            if not j.get("promoted") and not out.get("error"):
                try:
                    _promote(
                        ws,
                        ch,
                        out,
                        _LOOP_TIER,
                        _CERT_TIER,
                        _COVER,
                        sys.stderr,
                        context=context,
                        capsules_root=promotion_capsules_root,
                    )
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
                    _TP.record_cert(
                        ws,
                        out,
                        _CERT_TIER,
                        sys.stderr,
                        identity=j.get("identity"),
                        source_identity_verified=bool(j.get("source_identity_verified")),
                    )
                except Exception as _re:  # noqa: BLE001 -- recording must never gate a run either
                    print(f"[promote] record skipped: {type(_re).__name__}: {_re}", file=sys.stderr, flush=True)
            if j.get("log"):
                try:
                    j["log"].close()
                except Exception:  # noqa: BLE001 -- closing a log must never break the reap
                    pass
            if j["slot"]:
                j["slot"].unlink(missing_ok=True)
            if j.get("snapshot_root"):
                shutil.rmtree(j["snapshot_root"], ignore_errors=True)
            running.pop(jid)
        # launch queued (respect local + global caps)
        if len(running) < a.max_jobs:
            for req in sorted(ch.glob("simreq_*.json")):
                jid = req.stem[len("simreq_") :]
                if jid in claimed:
                    continue
                if len(running) >= a.max_jobs:
                    break
                r = json.loads(req.read_text()) if req.exists() else {}
                sim = r.get("sim")
                caps = _valid_capsules(str(r.get("capsules", "all")), capsules_root)
                if sim not in _allowed_sims(context) or caps is None:
                    # Say WHICH of the two it was, and what would be accepted. The old message named
                    # neither, and a request rejected without a remedy reads as "the oracle is broken":
                    # measured on a live run, an agent submitted twice, was rejected twice with this
                    # text, and never used the async oracle again -- while the arm that DID find the
                    # async path used it 98 times in the round its score moved 17 -> 26. An unhelpful
                    # rejection costs more than the check it protects.
                    _allowed = _allowed_sims(context)
                    if sim not in _allowed:
                        _why = (
                            f"--sim {sim!r} is not accepted for this target. Use "
                            f"{' or '.join(repr(s) for s in _allowed)}"
                            + (
                                f"; this target's tier comes from its contract, so {_NEUTRAL_SIM!r} "
                                f"means 'grade on whatever tier the contract resolves to' and the "
                                f"tier itself is chosen with --tiers"
                                if _allowed == (_NEUTRAL_SIM,)
                                else ""
                            )
                        )
                    else:
                        _named = str(r.get("capsules", "all"))
                        _why = (
                            f"--capsules {_named!r} named something this runner will not run: a "
                            f"capsule must be an existing public capsule directory name (letters, "
                            f"digits, underscore), or 'all'"
                        )
                    (ch / f"simresp_{jid}.json").write_text(
                        json.dumps(
                            {
                                "error": f"rejected: {_why}",
                                "all_pass": False,
                                "rejected_field": "sim" if sim not in _allowed else "capsules",
                            }
                        )
                    )
                    (ch / f"simerr_{jid}").write_text("rejected")
                    claimed.add(jid)
                    continue
                slot = None
                if sim == "verilator":
                    slot = _veril_acquire(a.veril_slots)
                    if slot is None:
                        continue  # global verilator budget full; try later
                submission = Path(ws) / "submission"
                snapshot_root = None
                source_identity_verified = False
                if bool(r.get("promoted")):
                    submission, snapshot_root, snapshot_error = recover_promotion_snapshot(
                        Path(ws),
                        r.get("submission_digest"),
                        r.get("identity"),
                        r.get("submission_snapshot"),
                        submission_digest=_TP._submission_digest,
                    )
                    if snapshot_error:
                        if slot:
                            slot.unlink(missing_ok=True)
                        (ch / f"simresp_{jid}.json").write_text(
                            json.dumps(
                                {"error": snapshot_error, "all_pass": False, "promotion_source_verified": False},
                                indent=2,
                            )
                        )
                        (ch / f"simerr_{jid}").write_text("source identity mismatch")
                        print(f"[promote] {jid} not launched: {snapshot_error}", file=sys.stderr, flush=True)
                        claimed.add(jid)
                        continue
                    source_identity_verified = True
                workers = max(1, min(int(r.get("workers", 1)), 2 if sim == "verilator" else 8))
                capspec = "all" if caps == ["all"] else ",".join(caps)
                ncaps = len(_valid_capsules("all", capsules_root) or []) if caps == ["all"] else len(caps)
                to = _job_timeout_s(sim, ncaps, vpc)
                resp_tmp = ch / f"simtmp_{jid}.json"
                argv2 = worker_command(context, capsules_root, contract) + [
                    "--submission",
                    str(submission),
                    "--capsules",
                    capspec,
                    "--workers",
                    str(workers),
                    "--timeout",
                    str(to),
                    "--out",
                    str(resp_tmp),
                ]
                if sim != _NEUTRAL_SIM:  # --sim is meaningless where the contract picks the tier
                    argv2 += ["--sim", sim]
                _tiers = str(r.get("tiers") or "").strip()
                if _tiers:  # validated downstream against the RESOLVED adapter map
                    argv2 += ["--tiers", _tiers]
                (ch / f"simrun_{jid}").write_text("running")
                # KEEP THE CHILD'S OUTPUT. This was stdout/stderr=DEVNULL, so when a job exited without
                # writing its verdict file the only trace was the broker's own "no verdict produced" --
                # a job that failed and a job that produced nothing were indistinguishable, and the
                # REASON was gone. Measured on merlincirct_arm4_func_20260901_v4: 19 promotion jobs
                # answered "no verdict produced" with no diagnostic anywhere on disk. The log is
                # per-job, beside the response, so a failure can be read after the fact.
                job_log = (ch / f"simlog_{jid}.txt").open("wb")
                try:
                    proc = _spawn_selfcheck(argv2, cwd=ws, env=_sim_env(context), job_log=job_log, timeout_s=to)
                except Exception:
                    job_log.close()
                    if slot:
                        slot.unlink(missing_ok=True)
                    if snapshot_root:
                        shutil.rmtree(snapshot_root, ignore_errors=True)
                    raise
                running[jid] = {
                    "proc": proc,
                    "slot": slot,
                    "resp_tmp": str(resp_tmp),
                    "sim": sim,
                    "promoted": bool(r.get("promoted")),
                    "log": job_log,
                    # which tier-state record this promotion was launched for (see the reap)
                    "identity": r.get("identity"),
                    "source_identity_verified": source_identity_verified,
                    "snapshot_root": snapshot_root,
                }
                claimed.add(jid)
        time.sleep(a.poll)
    # drain on STOP
    for j in running.values():
        j["proc"].kill()
        if j["slot"]:
            j["slot"].unlink(missing_ok=True)
        if j.get("snapshot_root"):
            shutil.rmtree(j["snapshot_root"], ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
