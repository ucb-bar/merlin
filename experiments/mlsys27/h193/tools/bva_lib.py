"""bva_lib.py -- seam code copied verbatim from /scratch2/loren/f2-cycles-backend/src/mkcase.py (F2 pipeline lead): entrypoint runner, graded route, kernel ABI pointer order."""
import hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
TOOLS = dict(
    LLVM=os.environ.get("MERLIN_CLANG_INSTALL", "/scratch/agustin/projects/oscar-merlin/third_party/llvm-install"),
    GCC=os.environ.get("RV_GCC", "/scratch2/loren/conda/pkgs/riscv-tools-1.0.6-0_h1234567_g56c29e0/riscv-tools/bin/riscv64-unknown-elf-gcc"),
    SPIKE=os.environ.get("SPIKE", "/scratch/loren/core-aten-snap/phase1-prep/toolchain/.conda-env/riscv-tools/bin/spike"),
    EXTLIB=os.environ.get("GEMMINI_EXTLIB", "/scratch/loren/s4rehearsal/extension/libgemmini.so"),
    RT=os.environ.get("F2B_RT", str(Path(__file__).resolve().parent / "rt")),
)
CFLAGS = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin", "-nostdlib", "-nostartfiles",
          "-fno-tree-loop-distribute-patterns", "-std=gnu11"]           # target_contract.yaml runner.chipyard_rocc.toolchain.cflags
ENTRY, FENCE = "kernel_entry", "kernel_complete"                         # target_contract.yaml logical_harness / abi: entry_symbol, fence_symbol
INPUT_ROLES = ("input", "weight", "bias", "scale")
PUBLISHED_BY = ("COMMIT", "BIAS_ADD", "RESIDUAL_ADD", "ATTENTION_QK", "ATTENTION_PV", "CONV2D", "BATCHED_MATMUL", "MOVEMENT")


class Fail(Exception):
    def __init__(self, stage, msg):
        super().__init__(f"{stage}: {msg}"); self.stage, self.msg = stage, msg


def sh(argv, *, stage, cwd=None, env=None, timeout=3600, stdout_path=None, log=None):
    t0 = time.time()
    with open(stdout_path, "wb") if stdout_path else open(os.devnull, "wb") as so:
        p = subprocess.run([str(a) for a in argv], cwd=cwd, env=env, stdout=so if stdout_path else subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    dt = time.time() - t0
    if log:
        log.write(f"[{stage}] rc={p.returncode} {dt:.2f}s :: {' '.join(str(a) for a in argv)[:300]}\n")
        if p.returncode: log.write((p.stderr or b"").decode(errors="replace")[-3000:] + "\n")
        log.flush()
    return p, dt


# ---------------------------------------------------------------------------------------------------------------------------------
def load_manifest(pkg):
    import yaml
    return yaml.safe_load(open(Path(pkg) / "manifest.yaml"))


def run_entry(pkg, man, name, input_mlir, output_json, *, out, log, timeout, stdout_path=None):
    cmd = man["commands"][name]["argv"]
    tool = str((Path(pkg) / man["entrypoints"]["tool"]).resolve()) if man.get("entrypoints", {}).get("tool") else None
    sub = {"{tool}": tool, "{input_mlir}": str(Path(input_mlir).resolve()), "{output_json}": str(output_json) if output_json else ""}
    argv = [sub.get(a, a) for a in cmd]
    argv = [re.sub(r"\{(tool|input_mlir|output_json)\}", lambda m: sub["{" + m.group(1) + "}"], a) for a in argv]
    if argv[0].endswith(".py"):
        argv = [sys.executable, "-P", *argv]
    env = dict(os.environ)
    # the sandbox puts the selected LLVM bin AND the merlin venv bin (python3 with xdsl) first on PATH (grader invocation.json:
    # export PATH=<merlin-wtC>/bin:<llvm>/bin:...); a package that imports xdsl declines with ModuleNotFoundError without it
    env["PATH"] = os.environ.get("F2B_PKG_PYBIN", "/scratch/loren/core-aten-snap/venvs/merlin-wtC/bin") + ":" + TOOLS["LLVM"] + "/bin:" + env["PATH"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"                      # running a package must never change its bytes
    if os.environ.get("F2B_PKG_PYTHONPATH"):                  # e.g. /perf-compiler-library/<sha>/python if the final package imports the shared library
        env["PYTHONPATH"] = os.environ["F2B_PKG_PYTHONPATH"]
    return sh(argv, stage=f"entry:{name}", cwd=str(Path(pkg).resolve()), env=env, timeout=timeout, stdout_path=stdout_path, log=log)


def kernel_args(cb):
    """Pointer order of the logical kernel ABI v2, derived from the command buffer (never from operand roles)."""
    tensors = cb["tensors"]
    ka = cb.get("kernel_abi") or {}
    if (cb.get("params") or {}).get("storage_encodings"):
        raise Fail("abi", "params.storage_encodings present: physical layout, the harness would have to pack it (not implemented, refusing)")
    if ka.get("kind") == "whole_program":
        args = [(a["tensor"], a.get("access", "read")) for a in ka["args"]]
        outs = list(ka.get("outputs") or [n for n, a in args if a == "write"])
        return args, outs, "kernel_abi.args"
    produced = {c["operands"].get("dst") for c in cb["commands"] if "dst" in c.get("operands", {})}
    ins = [n for n, t in tensors.items() if t.get("role") in INPUT_ROLES and n not in produced]
    outs = []
    for c in cb["commands"]:
        if c["opcode"] in PUBLISHED_BY and c["operands"].get("dst") and c["operands"]["dst"] not in outs:
            outs.append(c["operands"]["dst"])
    outs += [n for n, t in tensors.items() if t.get("role") == "output" and n not in outs]
    if cb.get("outputs"):
        outs = [n for n in outs if n in cb["outputs"]]
    return [(n, "read") for n in ins] + [(n, "write") for n in outs], outs, "tensors-order+result-order"


ELEM = {"i8": ("int8_t", 1), "u8": ("uint8_t", 1), "i16": ("int16_t", 2), "i32": ("int32_t", 4), "f32": ("uint32_t", 4), "i64": ("int64_t", 8)}


def run_route(pkg, man, route, iface, work, cbp, llp, res, log, tmo):
    """The sequence the grader runs (capsule_common.lower_interface). `graded` = shared-pass identity target: parse; lower_interface_to_target (stdout = lowered target IR);
    that IR is copied byte-for-byte to selected.target.mlir; emit_target_bundle on THAT file writes the command buffer at {output_json} and the artifact on stdout."""
    cmds = man["commands"]
    if route == "graded":
        p, dt = run_entry(pkg, man, "lower_interface_to_target", iface, None, out=work, log=log, timeout=tmo, stdout_path=work / "lowered.target.mlir")
        res["steps"]["lower_to_target"] = dict(rc=p.returncode, s=round(dt, 2))
        if p.returncode or not (work / "lowered.target.mlir").stat().st_size: raise Fail("interface_to_target", f"rc={p.returncode} " + p.stderr.decode(errors="replace")[-800:])
        shutil.copyfile(work / "lowered.target.mlir", work / "selected.target.mlir")
        p, dt = run_entry(pkg, man, "emit_target_bundle", work / "selected.target.mlir", cbp, out=work, log=log, timeout=tmo, stdout_path=llp)
        res["steps"]["emit"] = dict(rc=p.returncode, s=round(dt, 2))
        if p.returncode: raise Fail("emit", p.stderr.decode(errors="replace")[-800:])
        if not Path(cbp).exists(): raise Fail("emit", "emit_target_bundle exited 0 but wrote no file at {output_json}")
    elif route == "analysis":
        p, dt = run_entry(pkg, man, "emit_analysis_bundle", iface, cbp, out=work, log=log, timeout=tmo, stdout_path=llp)
        res["steps"]["emit"] = dict(rc=p.returncode, s=round(dt, 2))
        if p.returncode: raise Fail("emit", p.stderr.decode(errors="replace")[-800:])
    elif route == "bundle":      # raw interface straight into emit_target_bundle (NOT what the grader does; kept for diagnosis)
        p, dt = run_entry(pkg, man, "emit_target_bundle", iface, cbp, out=work, log=log, timeout=tmo, stdout_path=llp)
        res["steps"]["emit"] = dict(rc=p.returncode, s=round(dt, 2))
        if p.returncode: raise Fail("emit", p.stderr.decode(errors="replace")[-800:])
    else:
        p, dt = run_entry(pkg, man, "emit_command_buffer", iface, cbp, out=work, log=log, timeout=tmo)
        p2, dt2 = run_entry(pkg, man, "lower_target_to_llvm" if "lower_target_to_llvm" in cmds else "emit_target_artifact", iface, None, out=work, log=log, timeout=tmo, stdout_path=llp)
        res["steps"]["emit"] = dict(rc=p.returncode, s=round(dt, 2)); res["steps"]["artifact"] = dict(rc=p2.returncode, s=round(dt2, 2))
        if p.returncode or p2.returncode: raise Fail("emit", (p.stderr + p2.stderr).decode(errors="replace")[-800:])


