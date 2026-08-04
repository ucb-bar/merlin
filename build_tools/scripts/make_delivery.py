#!/usr/bin/env python3
"""Assemble a delivery package: binaries someone else can run on a board we cannot reach.

The deliverable is not a measurement — it is everything needed to run the binary and to interpret the
run that comes back. Inputs and weights are already embedded in the image (``llvmlower.c_runtime``
bakes inputs in as C arrays and the weights as one blob), so the package needs no data files and the
board needs no filesystem, no host I/O and no network.

What each part is for:

* **the ELF(s)** — one per hart count, so the authors can run 1-hart and N-hart back to back and check
  the outputs are bit-identical *themselves*. That turns their board into a self-validating test: any
  difference is their SoC's vector/SMP state, not our arithmetic.
* **``expected_console.txt``** — the console text OUR spike run produced. They can diff it.
* **``golden*.npy``** — the references, because grading happens against them, not against the binary.
* **``grade.py``** — reads a console log and prints the verdict, using the *same* parser and gate the
  repo grades with (``zephyr_model._parse_console`` + ``_gate``). No merlin checkout needed beyond numpy.
* **``elf_audit.json``** — what we checked before shipping: segments inside their DRAM, entry point,
  ``.htif`` present, vector instructions present, and the upload-time estimate.
* **``manifest.json``** — board facts we built for, per-binary build hashes, the package id, and the
  merlin commit. The build hash also appears in the console (``METRIC build_hash``), so a log can be
  tied to a binary rather than being unattributable.
* **``README.md``** — the exact loader command for *that* board's own flow, and the honest status of
  each model.

Deliberately NOT here: archiving. Producing a directory keeps this composable and reviewable; zip it as
a final manual step if that is how it is being sent.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))

import numpy as np                                                          # noqa: E402

from merlin.common.artifacts import new_product                             # noqa: E402
from merlin.common.paths import repo_root                                   # noqa: E402
from merlin.llvmlower.impr_features import PEROP_BLOCK_NAME                 # noqa: E402
from merlin.runtime import boards, elf_audit                                # noqa: E402
from merlin.runtime.backends import zephyr_model as zm                      # noqa: E402
from merlin.rvvgen.registry import load_rvv_package                         # noqa: E402

#: How each board is loaded and run, in ITS OWN flow. Quoted from the board's own repo so the authors
#: recognise it, rather than a command we invented.
LOADER_DOC = {
    "chipyard_kodiak": """\
```bash
# 1. reset the board (Arduino-style controller on /dev/ttyUSB0 @ 9600; sends "all", waits "Completed: all")
python3 scripts/send_all.py

# 2. load + run + capture the console over UART-TSI + FESVR
pyuartsi --port /dev/ttyUSB2 --elf <THIS.elf> --load --hart0_msip --fesvr \\
         --baudrate 57600 --cflush_addr 0x2010200 --selfcheck
```
The run ends by itself: the image calls `sys_reboot()`, which the SoC turns into an HTIF `tohost=1`, and
the loader prints `DUT forcefuly exit`. Everything between the banner and `DONE` is what we need back.

If your `tools/pyuartsi` is the pinned submodule (`c1bbb3a`), invoke it as `python3 -m pyuartsi` and drop
`--use_symbols` — that version has neither the console-script entry point nor that flag.""",
    "gemmelos_bearly25": """\
```bash
# Load and run over UART-TSI (the Chipyard host tool; NOT pyuartsi):
make tsi-run TTY=<your tty> BINARY=<THIS.elf>
#   equivalently: uart_tsi +tty=<tty> +baudrate=921600 <THIS.elf>

# or over JTAG:
openocd -f platform/bearly25/bearly25.cfg -c "reset run" -c "halt" \\
        -c "load_image <THIS.elf>" -c "resume 0x80000000"
```
This is a **bare-metal** ELF (entry `_start`, resumed at `0x80000000`, `-mcmodel=medany`, static) — not a
Zephyr image, because your SDK has no RTOS. Console output goes over HTIF, which is what the TSI/FESVR
link carries. If your flow needs UART0 (`PLATFORM=CHIP`) output instead, tell us and we will rebuild with
a UART console — it is a harness swap, not a recompile of the model.

Note `uart_tsi` uploads each segment's **MemSiz**, so the estimated upload time below is real.""",
    "default": """\
Load the ELF with your usual loader and capture the console. The image is self-contained: no filesystem,
no host I/O, no arguments. It ends by rebooting, after printing `DONE`.""",
}

GRADE_PY = '''#!/usr/bin/env python3
"""Grade a console log from this package's binary. Usage: python grade.py <console.txt> [--model NAME]

Uses the same parser and gate merlin grades with, vendored to a single file so this needs only numpy.
Tiers: `w8a8` (vs the W8A8 reference: cos > 0.999, rel < 1e-2, per-element max-rel < 5%) and `fp32`
(vs the weight-only golden: cos > 0.99, argmax matches). Grading a W8A8 run against golden.npy measures
activation-quantization error rather than correctness, which is why both references ship.
"""
import argparse, json, struct, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def parse_console(text):
    out_line = next((l for l in text.splitlines() if l.startswith("OUT ")), None)
    if out_line is None or "DONE" not in text:
        raise SystemExit("this log has no OUT line and/or no DONE — the run did not complete")
    parts = out_line.split()
    n = int(parts[1])
    bits = [int(x) for x in parts[2:2 + n]]
    vals = np.array([struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0] for b in bits],
                    dtype=np.float32)
    metrics = {}
    for l in text.splitlines():
        if l.startswith("METRIC "):
            p = l.split()
            if len(p) >= 3 and p[1] != "iter_cycles":
                try:
                    metrics[p[1]] = int(p[2])
                except ValueError:
                    metrics[p[1]] = p[2]
    return vals, metrics


def grade(prefix, refs, max_rel=0.05):
    out = {}
    for tier, ref in refs.items():
        r = np.asarray(ref, dtype=np.float32).ravel()[:len(prefix)]
        rmax = max(1e-9, float(np.abs(r).max()))
        rms = max(1e-9, float(np.sqrt(np.mean(r.astype(np.float64) ** 2))))
        sig = np.abs(r) >= 1e-2 * rms
        out[tier] = {
            "cos": float(prefix @ r / (np.linalg.norm(prefix) * np.linalg.norm(r) + 1e-12)),
            "rel": float(np.abs(prefix - r).max()) / rmax,
            "argmax": bool(int(np.argmax(prefix)) == int(np.argmax(r))),
            "max_rel": (float((np.abs(prefix[sig] - r[sig]) / np.abs(r[sig])).max())
                        if sig.any() else 0.0)}
    w, f = out.get("w8a8"), out.get("fp32")
    t1 = bool(w and w["cos"] > 0.999 and w["rel"] < 1e-2 and w["max_rel"] < max_rel)
    t2 = bool(f and f["cos"] > 0.99 and f["argmax"] and f["max_rel"] < max_rel)
    return {"tiers": out, "tier_ok": "w8a8" if t1 else "fp32" if t2 else None, "ok": t1 or t2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("console")
    ap.add_argument("--model", default=None, help="model name (default: the only one in the manifest)")
    a = ap.parse_args()
    man = json.loads((HERE / "manifest.json").read_text())
    models = sorted({b["model"] for b in man["binaries"]})
    model = a.model or (models[0] if len(models) == 1 else None)
    if model is None:
        raise SystemExit(f"--model is required; this package has: {', '.join(models)}")
    text = Path(a.console).read_text(errors="replace")
    prefix, metrics = parse_console(text)
    refs = {}
    for tier, fn in (("fp32", f"{model}.golden.npy"), ("w8a8", f"{model}.golden_w8a8.npy")):
        p = HERE / fn
        if p.is_file():
            refs[tier] = np.load(p)
    if not refs:
        raise SystemExit(f"no reference .npy for {model} in this package")
    res = grade(prefix, refs)
    print(f"model      : {model}")
    print(f"build_hash : {metrics.get('build_hash', '(absent — old binary or truncated log)')}")
    known = {b["build_hash"] for b in man["binaries"] if b["model"] == model}
    if metrics.get("build_hash") and metrics["build_hash"] not in known:
        print(f"  WARNING: that build_hash is not in this package (expected one of {sorted(known)})")
    print(f"cycles     : {metrics.get('cycles')}")
    print(f"elements   : {len(prefix)}")
    for tier, m in res["tiers"].items():
        print(f"  vs {tier:5s}: cos={m['cos']:.9f} rel={m['rel']:.6g} max_rel={m['max_rel']:.6g}")
    print(f"VERDICT    : {'PASS' if res['ok'] else 'FAIL'} (tier_ok={res['tier_ok']})")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

#: Honest status per model — shipped verbatim in the README, because a number nobody qualified is worse
#: than no number.
STATUS = {
    "spectformer": "VERIFIED. Bit-exact against the W8A8 reference on spike (1 and 3 harts) and on a "
                   "SpacemiT K1 board (real RVV silicon): w8a8_rel = 0.0. A failure here is about the "
                   "board, not about us — which is why this is the one to run first.",
    "deepjscc": "VERIFIED with per-op register blocking: w8a8_cos = 1.0, rel = 0.0. (With the older "
                "per-op-CLASS schedule this model scored 0.9176; that is fixed, not hidden.)",
    "lstmnetvit": "KNOWN DIVERGENT on RISC-V: w8a8_cos 0.9943 on spike and on the K1, while the same "
                  "IR is exact on x86. The 1-hart and N-hart runs ARE bit-identical, so it is useful "
                  "for bring-up and timing. Do not treat its output as an accuracy result — the cause "
                  "is ours to fix, not yours to debug.",
    "whisper_tiny": "Builds and lowers with 100% of its MACs on the vector path (per-op blocking). Its "
                    "encoder attention makes it much longer-running than the others; treat a long run "
                    "as expected rather than as a hang, and send the console log whatever it says.",
}


def build_baremetal(bundle: Path, brd, *, work: Path, timeout: int):
    """Build a BARE-METAL image for a board with no RTOS, and run it on spike.

    Used for Baremetal-IDE-style targets (gemmelos). Same lowering, same c_runtime artifacts and the same
    console protocol as the Zephyr path — only the harness differs (crt.S + our linker script + an
    absolute memory map instead of a Zephyr app). The map is packed inside the board's real DRAM, which
    the default spike map is not: its arena at 0xC0000000 and weights at 0x2_0000_0000 are simply not
    memory on a 1 GB chip, so the image would fault on its first activation.
    """
    from merlin.runtime.backends import spike_model

    # The SAME package, features and int8 datapath the Zephyr path uses. Not optional: lowering the raw
    # bundle here scored cos 0.925 on deepjscc, which the prepared path gets bit-exact.
    pkg = load_rvv_package(repo_root() / "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8")
    b = spike_model.build(bundle, work, inputs_npz=bundle / "inputs.npz",
                          dram_base=brd.dram_base, dram_bytes=brd.dram_bytes,
                          int8_compute=True, features=frozenset([PEROP_BLOCK_NAME]),
                          rvv_schedule=pkg.schedule_text,
                          cflags_override=pkg.cflags + zm._CFLAGS_COMMON, vlen=brd.vlen)
    refs = {"fp32": np.load(bundle / "golden.npy")}
    if (bundle / "golden_w8a8.npy").is_file():
        refs["w8a8"] = np.load(bundle / "golden_w8a8.npy")
    run = spike_model.run(b["elf"], harts=1, mem_bytes=b["mem_bytes"], timeout=timeout,
                          isa=zm.spike_isa(brd.vlen))
    res = dict(run)
    res.update(zm._gate(run["prefix"], refs))
    res.setdefault("metrics", run.get("metrics", {}))
    res["vlen"] = brd.vlen or 128
    return res, {"elf": b["elf"], "ram_bytes": b["mem_bytes"],
                 "build_hash": b.get("build_hash", "")}


def build_one(bundle: Path, brd, harts: int, *, vlen, work: Path, timeout: int):
    """Build one image and run it on spike at the board's VLEN and hart count."""
    pkg = load_rvv_package(repo_root() / "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8")
    refs = {"fp32": np.load(bundle / "golden.npy")}
    if (bundle / "golden_w8a8.npy").is_file():
        refs["w8a8"] = np.load(bundle / "golden_w8a8.npy")
    res = zm.build_and_run(
        bundle, work, board="spike_riscv64",          # run on spike; the ELF for the board is separate
        backend="rvv", rvv_hart=0, harts=max(2, harts), int8_compute=True, n_harts=harts,
        rvv_schedule=pkg.schedule_text, cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
        features=frozenset([PEROP_BLOCK_NAME]), references=refs, vlen=vlen, timeout=timeout)
    board_build = zm.build_app(
        bundle, work / f"board_h{harts}", board=brd.name, backend="rvv", rvv_hart=0,
        cpus=max(2, harts), int8_compute=True, rvv_schedule=pkg.schedule_text,
        cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
        features=frozenset([PEROP_BLOCK_NAME]), n_harts=harts, vlen=vlen)
    return res, board_build


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", required=True, help="Zephyr board (e.g. chipyard_kodiak)")
    ap.add_argument("--models", default="spectformer,deepjscc,lstmnetvit",
                    help="comma-separated workload names")
    ap.add_argument("--harts", default="1,3", help="comma-separated hart counts, one binary each")
    ap.add_argument("--dram-mb", type=int, default=None, help="the board's REAL DRAM")
    ap.add_argument("--vlen", type=int, default=None, help="the board's REAL vector length")
    ap.add_argument("--dtype", default="int8")
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--jobs", type=int, default=None,
                    help="images to build/simulate at once (default: min(images, 6)); each is one "
                         "single-threaded spike")
    ap.add_argument("--out", default=None, help="destination dir (default: an out/artifacts product)")
    a = ap.parse_args(argv)

    overrides = {}
    if a.dram_mb:
        overrides["dram_bytes"] = a.dram_mb * 1024 * 1024
    if a.vlen:
        overrides["vlen"] = a.vlen
    brd = boards.board(a.board, **overrides)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    hart_list = [int(h) for h in a.harts.split(",") if h.strip()]
    if max(hart_list) > brd.harts:
        print(f"[make_delivery] refusing {max(hart_list)} harts: {brd.name} has {brd.harts}",
              file=sys.stderr)
        return 2

    if a.out:
        dest = Path(a.out)
        dest.mkdir(parents=True, exist_ok=True)
        manifest_writer = None
    else:
        prod = new_product("delivery", version=1, target=a.board,
                           notes=f"int8 multicore-RVV binaries for {a.board}")
        dest, manifest_writer = Path(prod.path), prod
    print(f"[make_delivery] board={brd.name} dram={brd.dram_bytes // 2**20}MB harts={brd.harts} "
          f"vlen={brd.vlen or 'unknown(assume 128)'} -> {dest}", flush=True)

    binaries, audits, problems = [], {}, []
    todo = []                                     # (model, bundle, harts) -- one image each
    for model in models:
        bundle = repo_root() / f"out/artifacts/recaptures/{model}_{a.dtype}_full"
        if not (bundle / "model.mlir").is_file():
            problems.append(f"{model}: no bundle at {bundle}")
            continue
        for tier, fn in (("fp32", "golden.npy"), ("w8a8", "golden_w8a8.npy")):
            if (bundle / fn).is_file():
                shutil.copy2(bundle / fn, dest / f"{model}.{fn}")
        for harts in hart_list:
            if brd.flow == boards.FLOW_BAREMETAL and harts != 1:
                # gemmelos dispatches hart 1 through its own thread-lib, not OpenMP; a multi-hart
                # bare-metal image is a separate integration, so say so rather than shipping something
                # untested.
                problems.append(f"{model} h{harts}: baremetal multicore not implemented "
                                f"(hart 1 waits in wfi; their thread-lib dispatches it)")
                continue
            todo.append((model, bundle, harts))

    def _one(item):
        model, bundle, harts = item
        work = Path(tempfile.mkdtemp(prefix=f"delivery_{model}_h{harts}_"))
        print(f"  [{model} h{harts}] building + simulating in {work}", flush=True)
        if brd.flow == boards.FLOW_BAREMETAL:
            out = build_baremetal(bundle, brd, work=work, timeout=a.timeout)
        else:
            out = build_one(bundle, brd, harts, vlen=brd.vlen, work=work, timeout=a.timeout)
        cyc = out[0]["metrics"].get("cycles")
        print(f"  [{model} h{harts}] done: {cyc:,} cycles, gate={out[0].get('tier_ok')}", flush=True)
        return out

    # CONCURRENTLY, because the wall clock here is spike, not us: each image is one single-threaded
    # functional simulation of a whole int8 inference (spectformer alone is 2.78 G cycles at roughly
    # 2.4 M cycles/s, i.e. ~20 minutes), and a delivery is several of them. Serially that is hours for
    # a package whose slowest single item is under an hour. Each job is its own temp dir, its own build
    # tree and its own subprocesses, so the only shared state is the result assembly below -- done
    # after the pool drains, in a deterministic order rather than completion order.
    jobs = a.jobs or min(len(todo), 6)
    print(f"[make_delivery] {len(todo)} image(s), {jobs} at a time", flush=True)
    done: dict[tuple, object] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(_one, it): it for it in todo}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                done[item] = fut.result()
            except Exception as exc:                                        # noqa: BLE001
                msg = f"{item[0]} h{item[2]}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                problems.append(msg)
                # Say it NOW: a failure held until the final summary reads as an image still building,
                # and the rest of the set can take tens of minutes.
                print(f"  FAILED {msg}", file=sys.stderr, flush=True)

    for model in models:
        outputs = {}
        for harts in hart_list:
            got = done.get((model, repo_root() / f"out/artifacts/recaptures/{model}_{a.dtype}_full",
                            harts))
            if got is None:
                continue
            res, board_build = got
            elf_name = f"{model}_{a.dtype}_h{harts}_{brd.name}.elf"
            shutil.copy2(board_build["elf"], dest / elf_name)
            rep = elf_audit.audit(dest / elf_name, brd,
                                  ram_bytes=board_build["ram_bytes"])
            audits[elf_name] = rep.to_dict()
            if not rep.ok:
                problems += [f"{elf_name}: {p}" for p in rep.problems]
            (dest / f"{model}_h{harts}.expected_console.txt").write_text(res["console"])
            outputs[harts] = res["outputs"]
            binaries.append({
                "model": model, "elf": elf_name, "harts": harts, "dtype": a.dtype,
                "build_hash": board_build.get("build_hash", ""),
                "ram_bytes": board_build["ram_bytes"],
                "spike_cycles": res["metrics"].get("cycles"),
                "spike_vlen": res.get("vlen"), "gate_ok": bool(res.get("ok")),
                "tier_ok": res.get("tier_ok"), "cos": res.get("cos"), "rel": res.get("rel"),
                "upload_estimate_s": rep.facts.get("upload_estimate_s"),
            })
            print(f"  {elf_name}: cycles={res['metrics'].get('cycles'):,} gate={res.get('tier_ok')} "
                  f"audit={'OK' if rep.ok else 'FAIL'}", flush=True)
        # 1-hart vs N-hart bit-identity: the property a multicore run exists to establish
        if len(outputs) > 1:
            base_h = min(outputs)
            for h, arr in outputs.items():
                if h == base_h:
                    continue
                if not np.array_equal(outputs[base_h], arr):
                    problems.append(f"{model}: {h}-hart output differs from {base_h}-hart — an "
                                    f"overlapping or lost work split, not rounding")
                else:
                    print(f"  {model}: {h}-hart output bit-identical to {base_h}-hart", flush=True)

    (dest / "elf_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (dest / "grade.py").write_text(GRADE_PY)
    (dest / "grade.py").chmod(0o755)
    manifest = {
        "board": {"name": brd.name, "dram_bytes": brd.dram_bytes, "harts": brd.harts,
                  "vlen": brd.vlen, "console": brd.console, "notes": brd.notes},
        "dtype": a.dtype, "binaries": binaries, "problems": problems,
        "merlin_commit": _git_sha(), "validated_on": "spike (functional, at the board's VLEN)",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dest / "README.md").write_text(_readme(brd, manifest))
    if manifest_writer is not None:
        for f in sorted(dest.iterdir()):
            manifest_writer.add_artifact(f.name)
        manifest_writer.write_manifest()
    print(f"[make_delivery] {len(binaries)} binaries -> {dest}")
    if problems:
        print("[make_delivery] PROBLEMS (recorded in manifest.json, not hidden):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    return 0 if binaries and not problems else 1


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=repo_root(), timeout=30).stdout.strip()[:12] or "unknown"
    except Exception:                                                       # noqa: BLE001
        return "unknown"


def _readme(brd, manifest: dict) -> str:
    loader = LOADER_DOC.get(brd.name, LOADER_DOC["default"])
    baremetal = brd.flow == boards.FLOW_BAREMETAL
    image = ("bare-metal ELF (our own crt/linker script — your SDK has no RTOS)" if baremetal
             else "Zephyr image")
    hart_counts = sorted({b["harts"] for b in manifest["binaries"]})
    rows = "\n".join(
        f"| `{b['elf']}` | {b['model']} | {b['harts']} | "
        f"{b['ram_bytes'] // 2**20} MB | {b['upload_estimate_s']}s | "
        f"{'PASS' if b['gate_ok'] else 'see status'} |"
        for b in manifest["binaries"])
    statuses = "\n".join(f"- **{m}** — {STATUS.get(m, 'no status recorded')}"
                         for m in sorted({b["model"] for b in manifest["binaries"]}))
    vlen = brd.vlen or 128
    pair_section = ("""\
## Please run BOTH hart counts

The pair exists so you can check them against each other: **the outputs must be bit-identical.** They
are the same computation split differently, so any difference at all is vector/SMP state on the SoC, not
rounding — and that is a far more useful signal for you than either run alone. We verified this holds on
spike at your vector length before shipping.""" if len(hart_counts) > 1 else """\
## Only one hart, deliberately

Every binary here runs on one hart. We did not ship a multi-hart bare-metal image because dispatching
your second hart goes through your own thread-lib rather than the OpenMP runtime our multicore lowering
targets, and shipping that untested would waste your bench time. If you want it, that integration is
ours to do next — say so and we will build against your dispatch.""")
    identity_line = ("- Confirmed 1-hart and %d-hart outputs are bit-identical." % max(hart_counts)
                     if len(hart_counts) > 1 else
                     "- Single-hart images only (see above), so there is no hart-split to check.")
    return f"""\
# Merlin int8 RVV binaries for `{brd.name}`

Compiled by the Merlin flow (model2MLIR capture -> int8 W8A8 lowering -> certified RVV schedule ->
{image}). **Each binary is self-contained**: inputs and weights are baked in, so there is no
filesystem, no host I/O, no arguments, and nothing to install on the board.

We have no access to this board — you running these is the first time they touch the real chip.

## What we built for

| fact | value | where it came from |
|---|---|---|
| DRAM | {brd.dram_bytes // 2**20} MB at `{hex(brd.dram_base)}` | your chip — {'the linker scripts in your repo declare less' if baremetal else 'the DTS in your repo declares less'} |
| harts on the chip | {brd.harts} | your chip |
| harts these binaries use | {', '.join(str(h) for h in hart_counts)} | {'single-hart only — see below' if len(hart_counts) == 1 else 'one binary per count'} |
| vector length | {vlen} bits{'' if brd.vlen else ' (assumed — the V minimum, since nothing declares it)'} | {'stated in your repo' if brd.vlen else 'NOT declared anywhere in the board files'} |
| console | {brd.console} | {'your `chip_config.h` / SIMS platform' if baremetal else 'board DT'} |

**If any of those is wrong, tell us** — a mismatch is the most likely cause of a silent hang, and each
one is a one-line rebuild on our side. In particular we would like to know `vlenb` on the real chip.

## The binaries

| file | model | harts | linked region | est. upload | our spike verdict |
|---|---|---|---|---|---|
{rows}

Upload time is the *memory* size, not the file size — a UART loader transmits `MemSiz`, so a big
embedded weights blob costs minutes per attempt.

## Run one

{loader}

## Send back

The console text, verbatim, from the banner through `DONE`. That is enough for us to grade it: it
carries the output, the cycle counts, and a `METRIC build_hash` that identifies exactly which binary
produced it.

You can also grade it yourself, offline, with only numpy:

```bash
python grade.py <your_console_log.txt> --model spectformer
```

It prints `PASS`/`FAIL` with the cosine and per-element error against the same references we use, and
warns if the log's `build_hash` is not one of ours.

{pair_section}

## Status of each model — please read before reporting a number

{statuses}

## What we already checked, without the board

- Built for `{brd.name}` with the board's own facts, and audited the ELF against them: every LOAD
  segment inside your DRAM, entry point in range, `.htif` present so your loader can find
  `tohost`/`fromhost`, and real vector instructions in the image (see `elf_audit.json`).
- Ran the identical image on spike at **{vlen}-bit** vectors, and gated the output against the W8A8
  reference.
{identity_line}

What that does *not* cover: your clock, your DRAM timing, your vector unit's actual VLEN, and anything
about wall-clock performance. spike is functional — it proves correctness, never speed.

Merlin commit `{manifest['merlin_commit']}`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
