#!/usr/bin/env python3
"""After tuning: merge databases, build tuned ELFs (default plan + stock graph alternatives), confirm on F2.

Per unit (18 ResNet chunks, 16 kernel cases): baseline x3, tuned x3 (default plan), then up to 5 distinct
stock graph alternatives (fusion 0-3 x CSE) built WITH the tuned database, one run each. A tuned build that
finds zero structural database hits is recorded as a miss and the correct default is kept.
Final E2E (full-resnet baseline/tuned, 1800 s wall) is a separate step: stage2.py e2e
"""
import hashlib, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path("/work/exp/tvmfs")
WORK = EXP / "work"
C = Path(os.environ["COMPARISON"])
PY = os.environ["PY"]
sys.path.insert(0, str(C))
from firesim_job_pool import FireSimPool  # noqa: E402

OUT = EXP / "f2/results"
OUT.mkdir(parents=True, exist_ok=True)
pool = FireSimPool(Path(os.environ.get("POOL", EXP / "f2/pool.json")), deadline_epoch=time.time() + 6 * 3600)
lock = threading.Lock()
RESNET = sorted(p.name for p in (WORK / "resnet-inputs/resnet-release/resnet-chunks").iterdir() if p.is_dir())
KERNELS = subprocess.check_output([PY, str(C / "prepare_kernel_benchmarks.py"), "list"], text=True).split()
ALTS = [(3, False), (1, False), (0, False), (2, True), (3, True), (1, True), (0, True)]
MAX_ALTS = 5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan_name(fusion, cse):
    return "fuse%d%s" % (fusion, "-cse" if cse else "")


def run_elf(elf, label, wall=900):
    folder = OUT / "runs" / label
    cached = folder / "result.json"
    if cached.exists():
        r = json.loads(cached.read_text())
        if r.get("status") == "passed" and r.get("elf_sha256") == sha(elf) and r.get("remote_stopped") is True:
            for line in Path(r["transcript"]["path"]).read_text().splitlines():
                if line.startswith("DSE_CYCLES"):
                    return dict(ok=True, cycles=int(line.split("second=")[1]), elf_sha256=sha(elf), run=str(folder), reused=True)
    if folder.exists():
        folder = OUT / "runs" / ("%s.retry%d" % (label, int(time.time())))
    try:
        warm = pool.run(elf, folder, wall_timeout=wall)
        return dict(ok=True, cycles=warm, elf_sha256=sha(elf), run=str(folder))
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=repr(e), elf_sha256=sha(elf), run=str(folder))


def build(unit, kind, plan, db):
    fusion, cse = plan
    out = WORK / "tuned" / plan_name(*plan) / unit
    log = EXP / "logs" / ("tuned-%s-%s.log" % (plan_name(*plan), unit))
    if (out / "guest/valid.elf").exists():
        return dict(ok=True, elf=str(out / "guest/valid.elf"), log=str(log), tuned_miss=False, reused=True)
    if out.exists():
        return dict(ok=False, elf=str(out / "guest/valid.elf"), log=str(log), tuned_miss=False, error="partial earlier build; remove to retry")
    if kind == "kernel":
        cmd = [PY, str(C / "prepare_kernel_benchmarks.py"), "build", "--prepared", str(WORK / "kernel-inputs" / unit),
               "--output-dir", str(out), "--riscv-gcc", os.environ["RISCV_GCC"], "--adapter-receipt", os.environ["ADAPTER_RECEIPT"],
               "--graph-fusion-level", str(fusion), "--tuning-database", str(db)]
    else:
        cmd = [PY, str(C / "build_shared_chunk_elf.py"), "--tvm-source", os.environ["TVM_SOURCE"], "--restored-root",
               str(WORK / "resnet-inputs"), "--chunk", unit, "--output-dir", str(out), "--riscv-gcc", os.environ["RISCV_GCC"],
               "--adapter-receipt", os.environ["ADAPTER_RECEIPT"], "--host-math-policy", "native",
               "--tensor-memory-bytes", "268435456", "--graph-fusion-level", str(fusion), "--tuning-database", str(db)]
    if cse:
        cmd.append("--graph-cse")
    with log.open("w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elf = out / "guest/valid.elf"
    return dict(ok=r.returncode == 0 and elf.exists(), elf=str(elf), log=str(log),
                tuned_miss="zero" in log.read_text(errors="replace").lower() and r.returncode != 0)


def unit_pipeline(unit, kind, db):
    res = dict(unit=unit, kind=kind, database=str(db))
    base = WORK / "baseline" / unit / "guest/valid.elf"
    res["baseline"] = [run_elf(base, "%s-baseline-%d" % (unit, i)) for i in range(3)]
    b = build(unit, kind, (2, False), db)
    res["tuned_build"] = b
    seen = {sha(base): "baseline"}
    if b["ok"]:
        res["tuned"] = [run_elf(b["elf"], "%s-tuned-%d" % (unit, i)) for i in range(3)]
        seen[sha(b["elf"])] = "tuned-default"
        res["tuned_elf_identical_to_baseline"] = sha(b["elf"]) == sha(base)
    res["alternatives"] = []
    for plan in ALTS:
        if len([a for a in res["alternatives"] if a.get("distinct")]) >= MAX_ALTS:
            break
        ab = build(unit, kind, plan, db)
        row = dict(plan=plan_name(*plan), build=ab)
        if ab["ok"]:
            s = sha(ab["elf"])
            row["distinct"] = s not in seen
            if row["distinct"]:
                seen[s] = plan_name(*plan)
                row["run"] = run_elf(ab["elf"], "%s-%s-0" % (unit, plan_name(*plan)))
            else:
                row["same_program_as"] = seen[s]
        res["alternatives"].append(row)
    with lock:
        (OUT / ("unit-%s.json" % unit)).write_text(json.dumps(res, indent=1))
    return res


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "e2eplan":
        db, fusion, cse = Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "cse"
        b = build("full-resnet", "resnet", (fusion, cse), db)
        print("tuned full-resnet build:", plan_name(fusion, cse), b)
        if not b["ok"]:
            raise SystemExit("tuned build failed")
        r = run_elf(b["elf"], "full-resnet-e2e-tuned-" + plan_name(fusion, cse), 1800)
        (OUT / ("e2e-tuned-%s.json" % plan_name(fusion, cse))).write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "e2e":
        db = Path(sys.argv[2])
        b = build("full-resnet", "resnet", (2, False), db)
        print("tuned full-resnet build:", b)
        if not b["ok"]:
            raise SystemExit("tuned full-resnet build failed; baseline remains the default")
        with ThreadPoolExecutor(2) as ex:
            futs = {tag: ex.submit(run_elf, elf, "full-resnet-e2e-" + tag, 1800)
                    for tag, elf in (("baseline", WORK / "baseline/full-resnet/guest/valid.elf"),
                                     ("tuned", WORK / "tuned/fuse2/full-resnet/guest/valid.elf"))}
            results = {t: f.result() for t, f in futs.items()}
        bad = WORK / "baseline/full-resnet/guest/bad-oracle.elf"
        results["bad_oracle_control"] = run_elf(bad, "full-resnet-e2e-bad-oracle", 1800)
        (OUT / "e2e.json").write_text(json.dumps(results, indent=1))
        print(json.dumps(results, indent=1))
        return
    resnet_db = WORK / "db-resnet"
    jobs = [(u, "resnet", resnet_db) for u in RESNET] + \
           [(u, "kernel", WORK / "tune-kernel" / u / "validated-database") for u in KERNELS]
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda j: unit_pipeline(*j), jobs))
    (OUT / "confirm.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        bc = [x["cycles"] for x in r["baseline"] if x["ok"]]
        tc = [x["cycles"] for x in r.get("tuned", []) if x["ok"]]
        print(r["unit"], "base", bc, "tuned", tc, "build_ok", r["tuned_build"]["ok"])


if __name__ == "__main__":
    main()
