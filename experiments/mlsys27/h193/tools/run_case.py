#!/usr/bin/env python3
"""run_case.py --pkg PKGDIR --case CASEDIR --out OUTDIR [--policy exact|spec_int8|native_host_exact]
One verification-suite case (model2MLIR linalg-on-tensors capture + inputs + reference) through the candidate Phase-1/2 backend package:
  parse -> lower_interface_to_target -> (byte copy) -> emit_target_bundle   (the graded route, same seams as the F2 pipeline lead's mkcase.py)
  -> admission (declined / empty commands / lane) -> kernel ABI v2 pointer order -> mlir-translate -> clang-23 -> link (runner's bare-metal rt) with a generated harness
  -> Spike (rv64gc + libgemmini) with -l instruction trace; device instructions = retired RoCC (custom-3, opcode 0x7b) words -> compare outputs with the case reference.
Adapter: NONE for the IR (the capture's capsule.linalg.mlir is fed to the package unchanged).  The harness feeds inputs.json (positional, = func args) cast to the dtype the
command buffer declares for each input tensor, and reads outputs back as hex.  Writes OUTDIR/result.json; prints one JSON line."""
import argparse, json, os, re, struct, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bva_lib import *  # noqa

NP = None
def np():
    global NP
    if NP is None:
        import numpy; NP = numpy
    return NP

CT = {"i1": ("uint8_t", 1, "u1"), "i8": ("int8_t", 1, "i1"), "u8": ("uint8_t", 1, "u1"), "i16": ("int16_t", 2, "i2"), "i32": ("int32_t", 4, "i4"), "i64": ("int64_t", 8, "i8"),
      "f32": ("uint32_t", 4, "f4"), "f64": ("uint64_t", 8, "f8"), "f16": ("uint16_t", 2, "f2"), "bf16": ("uint16_t", 2, "u2")}
NPD = {"i1": "bool", "i8": "int8", "u8": "uint8", "i16": "int16", "i32": "int32", "i64": "int64", "f32": "float32", "f64": "float64", "f16": "float16"}


def flat(x):
    if isinstance(x, list):
        for y in x: yield from flat(y)
    else: yield x


def to_bytes(vals, dt):
    n = np()
    if dt == "bf16":
        a = n.array(list(vals), dtype=n.float32).view(n.uint32); return ((a >> 16).astype(n.uint16)).tobytes()
    a = n.array([0 if (isinstance(v, str) or v is None) else v for v in vals], dtype=object)
    if dt.startswith("f"): return n.array([float(v) for v in a], dtype=NPD[dt]).tobytes()
    ints = [int(round(float(v))) if not isinstance(v, bool) else int(v) for v in a]
    return n.array(ints, dtype=n.int64).astype(NPD[dt]).tobytes()


def classify_exc(msg):
    m = msg.lower()
    if any(k in m for k in ("notimplementederror", "no exact command-buffer lowering", "unsupported", "not supported", "refus", "declin", "cannot lower", "no lowering", "no executable payload")): return "declined"
    return "crashed"


def norm_cause(msg):
    m = re.sub(r"0x[0-9a-f]+|\d+", "N", msg.strip().splitlines()[-1] if msg.strip() else "")
    return m[:160]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pkg", required=True); ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="exact"); ap.add_argument("--entry-timeout", type=int, default=600); ap.add_argument("--spike-timeout", type=int, default=900); ap.add_argument("--no-spike", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    case = Path(a.case)
    log = open(out / "steps.log", "w")
    res = dict(case=str(case), pkg=str(a.pkg), outcome="crashed", stage="init", cause="", steps={})
    try:
        man = load_manifest(a.pkg)
        iface = case / "capsule.linalg.mlir"
        res["iface_sha256"] = hashlib.sha256(iface.read_bytes()).hexdigest()
        cmds = man["commands"]
        p, dt = run_entry(a.pkg, man, "parse", iface, None, out=out, log=log, timeout=a.entry_timeout)
        res["steps"]["parse"] = dict(rc=p.returncode, s=round(dt, 2))
        if p.returncode: raise Fail("parse", p.stderr.decode(errors="replace")[-600:])
        cbp, llp = out / "cb.json", out / "kernel.llvm.mlir"
        route = "graded" if "emit_target_bundle" in cmds else ("analysis" if "emit_analysis_bundle" in cmds else "split")
        res["route"] = route
        run_route(a.pkg, man, route, iface, out, cbp, llp, res, log, a.entry_timeout)
        cb = json.load(open(cbp))
        if cb.get("declined"):
            res["outcome"] = "declined"; res["cause"] = "declined-object: " + json.dumps(cb["declined"])[:200]
            if cb.get("commands"): res["outcome"] = "crashed"; res["cause"] = "BAD_PARTITION declined+commands"
            raise Fail("admission", res["cause"])
        if not cb.get("commands"):
            # An empty command buffer WITHOUT a declined object is a legitimate whole-program output: every region runs on the host and the
            # complete ELF (kernel_abi whole_program) is the only artifact (capsule_runner: "scalar-host regions live only in the submitted ELF").
            # Do not decline here; build and RUN it.  It still needs a runnable kernel ABI; classification (ok with 0 retired RoCC = lowered-on-host)
            # is aggregate193's job (DEVICE57 expects device insns > 0, HOST136 expects exactly zero + correct values).
            if (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
                res["outcome"] = "declined"; res["cause"] = "empty commands and no whole_program kernel_abi (nothing runnable)"; raise Fail("admission", res["cause"])
            res["host_only_program"] = True
        lanes = (cb.get("params") or {}).get("lane_placement") or []
        res["lane_placement"] = lanes
        res["opcodes"] = sorted({c["opcode"] for c in cb["commands"]})
        txt = llp.read_text()
        res["static_rocc_insns"] = len(re.findall(r"\.insn ", txt))
        args, outs, how = kernel_args(cb)
        res["abi"] = dict(args=[n for n, _ in args], outputs=outs)
        # --- reference / input data
        req = json.load(open(case / "request.json")); inputs = json.load(open(case / "inputs.json")); golden = json.load(open(case / "golden.json"))
        ins = [n for n, acc in args if acc == "read"]
        if isinstance(inputs, dict):          # name-keyed inputs (hand-built merlin_iface family cases)
            if set(inputs) == set(ins): inputs = [inputs[n] for n in ins]
            elif not (set(inputs) & set(cb["tensors"])) and len(inputs) == len(ins):
                # grader's POSITIONAL FALLBACK (capsule_runner.py: "Whole-program lowering may rename leaves positionally (A0 -> arg0)"):
                # when no canonical key is a cb tensor name, re-key by position (capsule arg_order == cb entry leaves, @forward-argument order)
                res["positional_rebind"] = dict(zip(inputs, ins)); inputs = list(inputs.values())
            else: raise Fail("abi", f"input names {sorted(inputs)} vs command buffer {ins}")
        if len(ins) != len(inputs):
            res["outcome"] = "crashed"; raise Fail("abi", f"command buffer has {len(ins)} input tensors, case has {len(inputs)}")
        L = ['#include <stdint.h>', '#include <stddef.h>',
             'void console_init(void); void htif_puts(const char *); void htif_putc(char); void htif_exit(int) __attribute__((noreturn));',
             f'extern void {ENTRY}({", ".join("void *" for _ in args)}); extern void {FENCE}(void);',
             'static inline uint64_t rd_instret(void){uint64_t c;__asm__ volatile("csrr %0, minstret":"=r"(c)::"memory");return c;}',
             'static void put_u64(unsigned long long v){char b[24];int i=0;do{b[i++]=(char)(\'0\'+v%10);v/=10;}while(v);while(i)htif_putc(b[--i]);}',
             'static void put_hex(const unsigned char *p, unsigned long n){static const char H[]="0123456789abcdef";for(unsigned long i=0;i<n;i++){htif_putc(H[p[i]>>4]);htif_putc(H[p[i]&15]);}}']
        casts = []; rawsz = {}
        for n, acc in args:
            t = cb["tensors"][n]; dtn = t["dtype"]
            if dtn not in CT: raise Fail("abi", f"tensor {n}: dtype {dtn} unsupported by harness")
            cnt = 1
            for d in t["shape"]: cnt *= d
            nb = cnt * CT[dtn][1]
            if nb > (4 << 20): raise Fail("abi", f"tensor {n}: {nb} bytes too large for the harness")
            if acc == "read":
                v = list(flat(inputs[ins.index(n)]))
                if len(v) != cnt: res["outcome"] = "crashed"; raise Fail("abi", f"input {n}: {len(v)} case elements vs command buffer {t['shape']}")
                b = to_bytes(v, dtn)
                if dtn[0] in "iu" and any(isinstance(x, float) and not float(x).is_integer() for x in v): casts.append(f"{n}:float->{dtn}")
                L.append(f'static const unsigned char raw_{n}[{max(nb,1)}] = {{{",".join(str(x) for x in b) or "0"}}};'); rawsz[n] = max(nb, 1)
                L.append(f'static {CT[dtn][0]} buf_{n}[{max(cnt,1)}] __attribute__((aligned(64)));')
            else:
                L.append(f'static {CT[dtn][0]} buf_{n}[{max(cnt,1)}] __attribute__((aligned(64)));')
        res["input_casts"] = casts
        L.append("int main(int hart){ if(hart!=0) for(;;); console_init();")
        for n, acc in args:
            if acc == "read":
                nb = rawsz[n]
                L.append(f'  for(unsigned long i=0;i<{nb}UL;i++) ((unsigned char*)buf_{n})[i]=raw_{n}[i];')
        for n in outs:
            t = cb["tensors"][n]; cnt = 1
            for d in t["shape"]: cnt *= d
            L.append(f'  for(unsigned long i=0;i<{cnt * CT[t["dtype"]][1]}UL;i++) ((volatile unsigned char*)buf_{n})[i]=0xA5;')
        L.append('  __asm__ volatile("fence":::"memory"); uint64_t i0=rd_instret();')
        L.append(f'  {ENTRY}({", ".join("(void*)buf_" + n for n, _ in args)}); {FENCE}(); __asm__ volatile("fence":::"memory"); uint64_t i1=rd_instret();')
        L.append('  htif_puts("INSTRET "); put_u64(i1-i0); htif_putc(10);')
        for k, n in enumerate(outs):
            t = cb["tensors"][n]; cnt = 1
            for d in t["shape"]: cnt *= d
            L.append(f'  htif_puts("OUT {k} {t["dtype"]} "); put_hex((const unsigned char*)buf_{n},{cnt * CT[t["dtype"]][1]}UL); htif_putc(10);')
        L.append('  htif_puts("DONE\\n"); htif_exit(0); return 0; }')
        (out / "main.c").write_text("\n".join(L) + "\n")
        # --- build
        tr = TOOLS["LLVM"] + "/bin/mlir-translate"; cl = TOOLS["LLVM"] + "/bin/clang-23"
        p, dt = sh([tr, "--mlir-to-llvmir", llp, "-o", out / "kernel.ll"], stage="mlir-translate", log=log)
        if p.returncode: raise Fail("translate", p.stderr.decode(errors="replace")[-800:])
        p, dt = sh([cl, "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin", "-c", out / "kernel.ll", "-o", out / "kernel.o"], stage="clang", log=log, timeout=900)
        if p.returncode: raise Fail("clang", p.stderr.decode(errors="replace")[-800:])
        rt = Path(TOOLS["RT"])
        p, dt = sh([TOOLS["GCC"], *CFLAGS, "-T", rt / "link.ld", "-I", rt, rt / "crt.S", out / "main.c", rt / "htif.c", rt / "libc_min.c", out / "kernel.o", "-o", out / "prog.elf", "-Wl,--start-group", "-lm", "-lgcc", "-Wl,--end-group"], stage="link", log=log, timeout=900)
        if p.returncode: raise Fail("link", p.stderr.decode(errors="replace")[-1500:])
        res["stage"] = "built"
        if a.no_spike: res["outcome"] = "built"; raise Fail("done", "no-spike")
        # --- Spike with trace; count retired RoCC words (opcode 0x7b)
        cmd = [TOOLS["SPIKE"], "-l", "--isa=rv64gc", "-p1", f"--extlib={TOOLS['EXTLIB']}", "--extension=gemmini", "-m0x80000000:2147483648", out / "prog.elf"]
        t0 = time.time()
        sp = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gp = subprocess.Popen(["grep", "-cE", r"\(0x[0-9a-f]{6}(7b|fb)\)"], stdin=sp.stderr, stdout=subprocess.PIPE)
        sp.stderr.close()
        try:
            console, _ = sp.communicate(timeout=a.spike_timeout)
        except subprocess.TimeoutExpired:
            sp.kill(); raise Fail("spike", f"timeout {a.spike_timeout}s")
        gout = gp.communicate()[0].decode().strip()
        res["device_insns_retired"] = int(gout or 0); res["spike_rc"] = sp.returncode; res["spike_s"] = round(time.time() - t0, 1)
        console = console.decode(errors="replace"); (out / "spike.console.txt").write_text(console[-200000:])
        if sp.returncode != 0 or "DONE" not in console: raise Fail("spike", f"rc={sp.returncode} no DONE")
        m = re.search(r"INSTRET (\d+)", console); res["window_instret"] = int(m.group(1)) if m else None
        # --- compare
        n = np()
        exp = req["expected"]
        # expected may be a dict (tensor / tuple), a list of tensor dicts (multi-output), or a bare python scalar (sym_* / _local_scalar_dense)
        is_multi = isinstance(exp, list) or (isinstance(exp, dict) and exp.get("kind") == "tuple")
        exps = (exp if isinstance(exp, list) else exp["items"]) if is_multi else [exp if isinstance(exp, dict) else {}]
        gl = golden if is_multi else [golden]
        metadata_only = req.get("comparison") == "metadata"   # empty / empty_strided: values are uninitialized memory, only shape+dtype are specified
        got = {}
        for l in console.splitlines():
            mm = re.match(r"OUT (\d+) (\S+) ([0-9a-f]*)$", l)
            if mm: got[int(mm.group(1))] = (mm.group(2), bytes.fromhex(mm.group(3)))
        if len(got) != len(outs) or len(gl) != len(outs):
            res["outcome"] = "wrong"; raise Fail("compare", f"{len(outs)} kernel outputs, {len(gl)} reference outputs")
        worst = 0.0; bad = []; exact_all = True
        for k, (dtn, bts) in sorted(got.items()):
            arr = n.frombuffer(bts, dtype=NPD.get(dtn, "uint16")).astype("float64") if dtn in NPD else None
            g = n.array(list(flat(gl[k])), dtype="float64")
            shp = cb["tensors"][outs[k]]["shape"]
            if arr is None or arr.size != g.size: bad.append(f"out{k}: size {None if arr is None else arr.size} vs ref {g.size}"); continue
            gdt = exps[k].get("dtype", "") if k < len(exps) and isinstance(exps[k], dict) else ""
            res.setdefault("out_dtypes", []).append(f"{dtn}<-ref:{gdt}")
            if metadata_only:       # size already checked above; the grader's comparison=metadata ignores values
                res.setdefault("metadata_only", True); continue
            with n.errstate(invalid="ignore"):
                same = n.array_equal(arr, g, equal_nan=True)
                if not same:
                    exact_all = False
                    fin = n.isfinite(arr) & n.isfinite(g)
                    err = n.where(fin, n.abs(arr - g), 0.0)
                    nanmis = bool(((~n.isfinite(arr)) != (~n.isfinite(g))).any() or (n.isinf(arr) & n.isinf(g) & (arr != g)).any())
                    me = float(err.max()) if err.size else 0.0; worst = max(worst, me)
                    if a.policy == "host_f32":     # host-path sanity: float outputs within f32 rounding of the PyTorch reference, integer/bool outputs exact
                        ok = (not nanmis) and dtn.startswith("f") and bool((err <= 1e-5 + 1e-4 * n.abs(n.where(fin, g, 0.0))).all())
                    elif a.policy != "spec_int8":
                        ok = False
                    else:
                        gm = float(n.abs(g[n.isfinite(g)]).max()) if n.isfinite(g).any() else 0.0
                        ok = (not nanmis) and bool((err <= 2.0 * gm / 127.0 + 0.05 * n.abs(n.where(fin, g, 0.0)) + 1e-6).all())
                    if not ok: bad.append(f"out{k}[{dtn}]: max_abs_err={me:.6g}" + (" nan/inf mismatch" if nanmis else ""))
        res["max_abs_err"] = worst; res["bit_exact"] = exact_all
        res["policy_applied"] = a.policy
        if bad: res["outcome"] = "wrong"; res["cause"] = "; ".join(bad)[:300]; raise Fail("compare", res["cause"])
        res["outcome"] = "ok"; res["stage"] = "compared"
    except Fail as f:
        res["stage"] = res["stage"] if f.stage == "done" else f.stage
        if res["outcome"] == "crashed":     # exception exits of the package are refusals when they say so
            res["outcome"] = classify_exc(f.msg) if f.stage in ("interface_to_target", "emit", "parse") else "crashed"
        res["cause"] = res["cause"] or norm_cause(f.msg); res["error"] = f.msg[-1500:]
    except Exception as e:  # noqa
        res["cause"] = f"{type(e).__name__}: {e}"[:300]; res["error"] = str(e)[-1500:]
    res["opcodes"] = res.get("opcodes", [])
    json.dump(res, open(out / "result.json", "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("case", "outcome", "stage", "device_insns_retired", "cause")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
