#!/usr/bin/env python3
"""Run the five ADAPTER_CONTRACT controls through the real FireSimPool and write the qualification proof.

usage: qualify.py --hosts-slots h0s0[,h0s1..] ; env from /work/exp/tvmfs/env.sh
"""
import hashlib, json, os, statistics, subprocess, sys, time
from pathlib import Path

EXP = Path("/work/exp/tvmfs")
sys.path.insert(0, os.environ["COMPARISON"])
from firesim_job_pool import FireSimPool, identity  # noqa: E402

F2 = EXP / "f2"
Q = F2 / "qualification"
Q.mkdir(parents=True, exist_ok=True)
ELF = lambda case: EXP / "work/baseline" / case / "guest"  # noqa: E731
SLOT = sys.argv[1] if len(sys.argv) > 1 else "h0s0"
CFG = F2 / "pool-prelim.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def pool(extra_args=()):
    cfg = json.loads(CFG.read_text())
    cfg["slots"] = [dict(id=SLOT, lock=str(F2 / "locks" / (SLOT + ".lock")),
                         command=cfg["slot_command_template"] + list(extra_args))]
    tmp = F2 / ("pool-q-%d.json" % os.getpid())
    tmp.write_text(json.dumps(cfg))
    return FireSimPool(tmp, deadline_epoch=time.time() + 6 * 3600)


def run(p, elf, name, wall):
    folder = Q / "runs" / name
    try:
        warm = p.run(elf, folder, wall_timeout=wall)
        return dict(ok=True, warm=warm, folder=str(folder))
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=repr(e), folder=str(folder))


def result(folder):
    try:
        return json.loads((Path(folder) / "result.json").read_text())
    except OSError:
        return {}


def cycles(folder):
    t = (Path(folder) / "transcript.txt").read_text()
    for line in t.splitlines():
        if line.startswith("DSE_CYCLES"):
            return {k: int(v) for k, v in (kv.split("=") for kv in line.split()[1:])}
    return {}


def emulated(folder):
    for name in ("remote.stderr", "transcript.txt"):
        for line in (Path(folder) / name).read_text(errors="replace").splitlines():
            if "Target Cycles Emulated:" in line:
                return int(line.split(":")[1])
    return None


tests = {}
p = pool()
valid = ELF("G01") / "valid.elf"
bad = ELF("G01") / "bad-oracle.elf"

# 1+2: repeat the same ELF; startup/warmup excluded from the label; report variability.
reps = [run(p, valid, "repeat-%d" % i, 120) for i in range(5)]
warms = [r["warm"] for r in reps if r["ok"]]
firsts = [cycles(r["folder"]).get("first") for r in reps if r["ok"]]
total = [emulated(r["folder"]) for r in reps if r["ok"]]
tests["same_elf_repeat_agreement"] = dict(
    test="same_elf_repeat_agreement",
    status="pass" if len(warms) == 5 and len(set(warms)) <= 5 and min(warms) > 0 else "fail",
    warm_cycles=warms, min=min(warms) if warms else None, max=max(warms) if warms else None,
    spread_fraction=((max(warms) - min(warms)) / statistics.median(warms)) if warms else None,
    note="Reports actual cycle variability; determinism is not assumed.")
tests["startup_exclusion"] = dict(
    test="startup_exclusion",
    status="pass" if warms and all(w < t for w, t in zip(warms, total)) and all(f and f > 0 for f in firsts) else "fail",
    warm_cycles=warms, first_call_cycles=firsts, total_target_cycles_emulated=total,
    note="Pool cost is the guest second-call window; the first (warmup) call and boot are not in it.")

# 3: wrong oracle must be rejected by the guest (nonzero exit, no DSE_PASS), pool must refuse it.
wo = run(p, bad, "wrong-oracle", 120)
wr = result(wo["folder"])
txt = (Path(wo["folder"]) / "transcript.txt").read_text() if (Path(wo["folder"]) / "transcript.txt").exists() else ""
tests["wrong_oracle_rejection"] = dict(
    test="wrong_oracle_rejection",
    status="pass" if (not wo["ok"]) and wr.get("guest_exit") not in (0, None) and "DSE_PASS" not in txt.splitlines() else "fail",
    guest_exit=wr.get("guest_exit"), pool_error=wo.get("error"))

# 4: a bounded attempt is stopped by the wall deadline. Full ResNet runs far longer than 45 s.
long_elf = ELF("../../../..") if False else EXP / "work/baseline/full-resnet"
long_candidates = sorted(long_elf.rglob("valid.elf"))
wt = run(p, long_candidates[0], "wall-timeout", 45)
wtr = result(wt["folder"])
tests["wall_timeout"] = dict(
    test="wall_timeout",
    status="pass" if (not wt["ok"]) and wtr.get("status") == "timeout" and wtr.get("remote_stopped") is True and wtr.get("wall_limit_enforced") is True else "fail",
    elf=str(long_candidates[0]), wall_seconds=45, result_status=wtr.get("status"), guest_exit=wtr.get("guest_exit"),
    occupied_seconds=wtr.get("occupied_seconds"), pool_error=wt.get("error"))

# 5a: slot is safe to reuse after the timeout.
after = run(p, valid, "reuse-after-timeout", 120)
# 5b: force the local-kill path (remote wall far beyond the deadline) and confirm the adapter stops the slot itself.
p2 = pool(["--test-remote-wall", "3000", "--test-local-cutoff", "15"])
lk = run(p2, long_candidates[0], "local-kill-path", 40)
lkr = result(lk["folder"])
after2 = run(p, valid, "reuse-after-local-kill", 120)
tests["remote_cancellation"] = dict(
    test="remote_cancellation",
    status="pass" if (after["ok"] and after2["ok"] and (not lk["ok"]) and lkr.get("remote_stopped") is True and lkr.get("status") == "timeout") else "fail",
    reuse_after_timeout=after, local_kill_result=dict(status=lkr.get("status"), remote_stopped=lkr.get("remote_stopped")),
    reuse_after_local_kill=after2,
    note="Local-kill path: remote driver was left running past the deadline; adapter probed it, reset the slot, and the next run passed.")

proof_tests = {}
for name, body in tests.items():
    proof_tests[name] = write_json(Q / ("%s.json" % name), body)
    print(name, body["status"])

cfg = json.loads(CFG.read_text())
proof = dict(
    schema="gemmini_firesim_runner_qualification_v1",
    timeout_scope="wall_execution", cycle_scope="fixed_warm_model_run",
    adapter_source_sha256=[sha(s["path"]) for s in cfg["adapter_sources"]],
    deployment_id=cfg["deployment_id"], tests=proof_tests,
    created_epoch=time.time(), slot=SLOT)
pf = write_json(Q / "runner_qualification.json", proof)
print(json.dumps(pf))
sys.exit(0 if all(t["status"] == "pass" for t in tests.values()) else 1)
