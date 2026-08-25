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


def _submission_digest(ws) -> str:
    """A content address for the submission the verdict was earned against.

    Verdicts are keyed by BYTES, not by round: unchanged bytes never need re-grading, and changed bytes
    invalidate exactly the capsules they touch. Without this the loop re-certifies thirty-odd capsules to
    learn about the one that moved.
    """
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(ws, "submission").rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(f.relative_to(ws).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _cert_cover(ws) -> set | None:
    """Which capsules are worth certifying at all. The hardware cannot tell two capsules in the same
    (family, dtype) cell apart, so certifying both spends minutes to learn nothing. `None` on any failure
    -> certify anything eligible, because a cover that silently comes back empty is indistinguishable
    from everything already being done."""
    try:
        from merlin.targetgen.contract.materialize import cert_capsule_cover
        from merlin.targetgen.target_experiment import load_target_experiment
        import _common as _C
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        # Pass the tile edge so the cover certifies PARTIAL tiles as their own cell. A cover built on
        # family and dtype alone can pick, per cell, the capsule whose extents happen to divide evenly and
        # then certify no ragged extent anywhere -- and a partial tile is exactly what a functional model
        # is least able to stand in for (a taped-out unit here got `n % 64 != 0` wrong while every
        # functional check passed).
        _td = None
        try:
            from merlin.targetgen.corpus_spec import _tile_dim
            from merlin.targetgen.target_experiment import load_capability_manifest
            _td = int(_tile_dim(te.target, load_capability_manifest(te.target).contract)) or None
        except Exception:  # noqa: BLE001 -- no derivable tile edge: cover without the alignment axis
            _td = None
        return set(cert_capsule_cover(te.graded_roots(), tile_dim=_td)["capsules"])
    except Exception:  # noqa: BLE001 -- no resolvable corpus: stay permissive, never silently empty
        return None


def _tier_state(ws) -> dict:
    import json as _j
    f = Path(ws, "qa", "tier_state.json")
    try:
        return _j.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_tier_state(ws, st) -> None:
    import json as _j
    f = Path(ws, "qa", "tier_state.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(_j.dumps(st, indent=2))


def _promote(ws, ch, verdict, loop_tier, cert_tier, cover, log):
    """Record what the loop tier just learned, and enqueue cert jobs for what it unlocked.

    Returns the capsule names promoted. Enqueues by writing a `simreq_` the broker's own queue picks up --
    the same path an agent request takes, so it inherits the constrained-runner validation rather than
    routing around it.
    """
    import json as _j
    digest = _submission_digest(ws)
    st = _tier_state(ws)
    promoted = []
    for row in (verdict.get("per_capsule") or []):
        name = row.get("capsule")
        if not name:
            continue
        entry = st.setdefault(name, {})
        entry[loop_tier] = {"status": "pass" if row.get("pass") else "fail", "digest": digest}
        if not row.get("pass"):
            continue                                   # a failed loop tier cannot be rescued by RTL
        if cover is not None and name not in cover:
            continue                                   # outside the representative cover
        known = entry.get(cert_tier) or {}
        if known.get("digest") == digest:
            continue                                   # already certified FOR THESE BYTES
        entry[cert_tier] = {"status": "pending", "digest": digest}
        jid = f"promo{len(promoted)}_{digest}_{name}"[:80]
        if not (ch / f"simreq_{jid}.json").exists():
            (ch / f"simreq_{jid}.json").write_text(_j.dumps(
                {"sim": _NEUTRAL_SIM, "capsules": name, "workers": 1, "tiers": cert_tier,
                 "promoted": True, "submitted_at": time.time()}))
            promoted.append(name)
    _save_tier_state(ws, st)
    if promoted:
        print(f"[promote] {loop_tier} pass -> {cert_tier}: {promoted}", file=log, flush=True)
    return promoted


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
    return ("spike", "verilator", "vcs") if _sim_via() == "chipyard" else (_NEUTRAL_SIM,)


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
    _LOOP_TIER = _CERT_TIER = None
    _COVER = None
    try:
        from merlin.targetgen import capsule_runner as _CR
        from merlin.targetgen.contract.materialize import declared_oracle_tiers
        from merlin.targetgen.target_experiment import load_target_experiment
        import _common as _C
        _te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        _decl = declared_oracle_tiers(*_te.graded_roots())
        _loop = _CR.qa_loop_adapters(_te.target, _te.sim_via, declared_tiers=_decl)
        _full = _CR.oracle_adapters(_te.target, _te.sim_via)
        _deeper = sorted(set(_full) - set(_loop))
        if _loop and _deeper:
            _LOOP_TIER, _CERT_TIER = sorted(_loop)[0], _deeper[-1]
            _COVER = _cert_cover(ws)
    except Exception as _e:  # noqa: BLE001 -- unresolvable ladder: no promotion, and SAY so
        print(f"[promote] disabled: {type(_e).__name__}: {_e}", file=sys.stderr, flush=True)
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
                    {"error": "no verdict produced", "all_pass": False}
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
                proc = subprocess.Popen(["timeout", str(to + 120)] + argv2, cwd=str(ws),
                                        env=_sim_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                running[jid] = {"proc": proc, "slot": slot, "resp_tmp": str(resp_tmp), "sim": sim,
                                "promoted": bool(r.get("promoted"))}
                claimed.add(jid)
        time.sleep(a.poll)
    # drain on STOP
    for j in running.values():
        j["proc"].kill()
        if j["slot"]:
            j["slot"].unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
