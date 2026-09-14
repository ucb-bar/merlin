"""Run bridged Voyager conv schedules (and merlin's own lowering of the same layer) on Gemmini, bare metal.

A :class:`merlin.baselines.voyager_schedule.Schedule` is abstract block ops. This script packs them with
the certified reference package's own encoders -- exactly as ``package_template/voyager_replay.py``
packs a GEMM schedule (a CONFIG_LD before every MVIN, the reference's leading FENCE/FLUSH, no trailing
fence of its own) -- emits the kernel as LLVM-dialect MLIR in the package's ``llvm_artifact`` form
(pointer-derived operands, one ``.insn`` per command, one closing fence), and links it with a
runner-style harness through the target's own build recipe. The same ELF then runs on Spike and on the
cycle-accurate Verilator binary the capsule runner uses.

Translations beyond the GEMM template, each derived from the RTL or its generated header:

* a halo (``Mvin`` role ``zero``) is an MVIN from DRAM address 0: Gemmini's LoadController marks a read
  at vaddr 0 ``all_zeros`` (``LoadController.scala`` ``val all_zeros = vaddr === 0.U``) and the
  scratchpad routes it to its ``ZeroWriter`` (``Scratchpad.scala``), so no DRAM is read;
* an ``AccMvin`` (bias) is an MVIN to an accumulator address with a stride-0 broadcast, which the
  LoadController serves as one DRAM row repeated (``actual_rows_read = 1`` when ``stride === 0``);
* a scaled accumulator MVIN is refused unless the target's generated header scales accumulator loads:
  ``gemmini_params.h`` defines ``MVIN_SCALE_ACC(x, scale)`` and on an identity definition a scale
  would be silently dropped.

Exactness: the harness prints a 64-bit FNV-1a digest of the output buffer, compared against the digest
of ``execute()`` (for the Voyager arm) or of a direct convolution (for merlin's arm) -- a digest, not the
values, because a Verilator console is the dominant cost of a cycle-accurate run.

The merlin arm is the certified reference package's own ``build_trace`` of a ``merlin_iface.conv2d``
program for the same layer, emitted through the same emitter and harness, so the arms differ only in
the command stream.

Usage (merlin venv):
    python merlin/experiments/voyager_h2h/scripts/emit_gemmini_c.py --model-dir <voyager export dir> \
        [--model-dir ...] --simulators spike verilator
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from merlin.baselines import voyager_schedule as vs
from merlin.baselines.voyager import geometry_for
from merlin.baselines.voyager_ir import UnsupportedConstruct, load_model, replay
from merlin.common import provenance
from merlin.common.artifacts import new_product
from merlin.common.paths import artifacts_dir, merlin_dir, runs_dir

TARGET = "gemmini"
DEFAULT_PACKAGE = artifacts_dir() / "targets" / TARGET / "gemmini_xdsl_rtl_v0"
_FNV_OFFSET, _FNV_PRIME = 0xCBF29CE484222325, 0x100000001B3
#: Schedule dtype names -> the package's tensor dtype names.
_DTYPES = {"int32": "i32", "i32": "i32", "int8": "i8", "i8": "i8"}


# ---------------------------------------------------------------------------------------------------
# the package's encoders
# ---------------------------------------------------------------------------------------------------
def load_package(package: Path):
    """(isa, ir_ingest) modules of an OOT package, imported the way its own tool imports them."""
    root = str(Path(package) / "mlir_oot")
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("lowering.isa"), importlib.import_module("ir_ingest")


def acc_mvin_scales(params_h: Path) -> bool:
    """Whether the generated header's accumulator-load scale does anything (parsed, not assumed)."""
    for line in params_h.read_text().splitlines():
        head, _, body = line.partition(")")
        if head.replace(" ", "").startswith("#defineMVIN_SCALE_ACC(x,scale"):
            return body.strip() not in ("(x)", "x")
    return False


def schedule_instructions(schedule: vs.Schedule, isa, *, names: dict, strides: dict, elems: dict,
                          out_dtype: str, acc_scale_ok: bool) -> list:
    """Pack ``schedule`` with the package encoders (the replay template's packing, extended to conv)."""
    I, A, word = isa.Instruction, isa.Address, isa._tile_word
    readout = isa.ACC_BASE | (isa.ACC_FULL if out_dtype == "i32" else 0)

    def store_config(op) -> tuple:
        scale = getattr(op, "scale", None)
        relu = bool(getattr(op, "relu", False))
        dtype = _DTYPES.get(getattr(op, "out_dtype", None) or out_dtype)
        if dtype != out_dtype:
            raise UnsupportedConstruct(f"a store of {dtype} into the {out_dtype} output buffer")
        return (1.0 if scale is None else float(scale), relu)

    def config_st(key: tuple):
        scale, relu = key
        return isa._config_st(strides["out"], {"epilogue": ["relu"] if relu else [],
                                               "acc_scale": scale}, out_dtype)

    stores = [op for op in schedule.ops if isinstance(op, vs.Mvout)]
    current = store_config(stores[0]) if stores else (1.0, False)
    trace = [I("FENCE"), I("FLUSH", 0, 0), isa._config_ex(weight_stationary=True), config_st(current)]
    for op in schedule.ops:
        kind = type(op).__name__
        if kind == "Mvin":
            if op.role == "zero":
                # DRAM address 0 is the RTL's zero-writer path; the stride is unused there.
                trace += [isa._config_ld(0, channel=0),
                          I("MVIN", 0, word(op.spad_row, op.cols, op.rows))]
                continue
            trace += [isa._config_ld(strides[op.role] * op.row_step, channel=0),
                      I("MVIN", A(names[op.role], op.dram_row * strides[op.role]
                                  + op.dram_col * elems[op.role]),
                        word(op.spad_row, op.cols, op.rows))]
        elif kind == "AccMvin":
            scale = getattr(op, "scale", None)
            if scale not in (None, 1, 1.0) and not acc_scale_ok:
                raise UnsupportedConstruct(f"a {op.role} accumulator load scaled by {scale}: this "
                                           "target's MVIN_SCALE_ACC is the identity")
            if scale not in (None, 1, 1.0):
                raise UnsupportedConstruct("scaled accumulator loads are not packed yet")
            addr = isa.ACC_BASE | op.acc_row | (isa.ACC_ACCUMULATE if getattr(op, "accumulate", False)
                                                else 0)
            trace += [isa._config_ld(strides[op.role] * op.row_step, shrunk=elems[op.role] == 1,
                                     channel=0),
                      I("MVIN", A(names[op.role], op.dram_row * strides[op.role]
                                  + op.dram_col * elems[op.role]),
                        word(addr, op.cols, op.rows))]
        elif kind == "Preload":
            b = word(isa.GARBAGE_ADDR if op.weight_row is None else op.weight_row, isa.DIM, isa.DIM)
            c = readout | op.acc_row | (isa.ACC_ACCUMULATE if op.accumulate else 0)
            trace.append(I("PRELOAD", b, word(c, op.cols, op.rows)))
        elif kind == "Compute":
            trace.append(I("COMPUTE_PRELOADED" if op.fresh_weights else "COMPUTE_ACCUMULATE",
                           word(op.input_row, isa.DIM, op.rows),
                           word(isa.GARBAGE_ADDR, isa.DIM, isa.DIM)))
        elif kind == "Mvout":
            key = store_config(op)
            if key != current:
                trace.append(config_st(key))
                current = key
            trace.append(I("MVOUT", A(names["out"], op.dram_row * strides["out"]
                                      + op.dram_col * elems["out"]),
                           word(readout | isa.ACC_ACCUMULATE | op.acc_row, op.cols, op.rows)))
        else:
            raise UnsupportedConstruct(f"{kind} has no device packing (a host op needs a host lane)")
    return trace


def emit_llvm(tensor_order: list[str], trace: list, isa) -> str:
    """LLVM-dialect MLIR in the package's ``llvm_artifact`` form."""
    args = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(len(tensor_order)))
    lines = ["module {", f"  llvm.func @gemmini_kernel({args}) {{"]
    bases: dict[str, str] = {}
    counter = [0]

    def new(prefix: str = "c") -> str:        # the package's numbering, so the texts match
        name = f"%{prefix}{counter[0]}"
        counter[0] += 1
        return name

    for index, name in enumerate(tensor_order):
        value = new("p")
        lines.append(f"    {value} = llvm.ptrtoint %arg{index} : !llvm.ptr to i64")
        bases[name] = value

    def operand(value) -> str:
        if isinstance(value, isa.Address):
            if value.offset == 0:
                return bases[value.tensor]
            const, result = new(), new("a")
            lines.append(f"    {const} = llvm.mlir.constant({value.offset} : i64) : i64")
            lines.append(f"    {result} = llvm.add {bases[value.tensor]}, {const} : i64")
            return result
        const = new()
        lines.append(f"    {const} = llvm.mlir.constant({int(value or 0)} : i64) : i64")
        return const

    for ins in trace:
        if ins.name == "FENCE":
            lines.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
            continue
        rs1, rs2 = operand(ins.rs1), operand(ins.rs2)
        lines.append(f'    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 0x{ins.funct:x}, x0, '
                     f'$0, $1", "r,r" {rs1}, {rs2} : (i64, i64) -> ()')
    lines += ['    llvm.inline_asm has_side_effects "fence", "" : () -> ()', "    llvm.return", "  }",
              "}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------
# harness, build, run
# ---------------------------------------------------------------------------------------------------
_C_TYPES = {"i8": ("elem_t", "row_align(1)"), "i32": ("int32_t", "row_align_acc(1)")}


def fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def render_harness(tensors: list[tuple[str, str, np.ndarray | None, tuple]]) -> str:
    """``tensors``: (name, dtype, values or None for the output, shape), in kernel-argument order."""
    out = [t for t in tensors if t[2] is None]
    if len(out) != 1:
        raise ValueError("exactly one output buffer")
    lines = ["#include <stdint.h>", "#include <stdio.h>", '#include "include/gemmini_testutils.h"',
             "extern void gemmini_kernel();"]
    for name, dtype, values, shape in tensors:
        ctype, align = _C_TYPES[dtype]
        count = int(np.prod(shape))
        if values is None:
            lines.append(f"static {ctype} T_{name}[{count}] {align};")
        else:
            body = ",".join(str(int(v)) for v in np.asarray(values).reshape(-1))
            lines.append(f"static const {ctype} T_{name}[{count}] {align} = {{{body}}};")
    oname = out[0][0]
    call = ", ".join(f"(void*)T_{name}" for name, *_ in tensors)
    lines += [
        "static uint64_t fnv1a64(const uint8_t *p, unsigned long n) {",
        f"  uint64_t h = {_FNV_OFFSET}ULL;",
        f"  for (unsigned long i = 0; i < n; i++) {{ h ^= p[i]; h *= {_FNV_PRIME}ULL; }}",
        "  return h;", "}",
        "int main() {",
        "  uint64_t c0 = read_cycles();",
        f"  gemmini_kernel({call});",
        "  gemmini_fence();",
        "  uint64_t c1 = read_cycles();",
        '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
        f'  printf("DIGEST {oname} %lu\\n", (unsigned long)fnv1a64((const uint8_t *)T_{oname}, '
        f"sizeof(T_{oname})));",
        '  printf("DONE\\n");', "  return 0;", "}"]
    return "\n".join(lines) + "\n"


def build_elf(workdir: Path, mlir_text: str, harness_text: str) -> Path:
    from merlin.runtime.backends import base as backends
    from merlin.targetgen.contract.compile import llvm_mlir_to_object
    from merlin.targetgen.runtime_build import derived_link_script
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "kernel.llvm.mlir").write_text(mlir_text)
    obj = llvm_mlir_to_object(mlir_text, workdir, target=TARGET)
    recipe = backends.harness_build_recipe(TARGET)
    (workdir / "harness.c").write_text(harness_text)
    objects = []
    for source in [workdir / "harness.c", obj, *recipe.support_sources]:
        source = Path(source)
        if source.suffix not in (".c", ".S", ".s"):
            objects.append(source)
            continue
        unit = workdir / f"{source.stem}.o"
        step = subprocess.run(recipe.compile_command(source=source, output=unit), capture_output=True,
                              text=True)
        if step.returncode:
            raise RuntimeError(f"compile {source.name}: {step.stderr[-1500:]}")
        objects.append(unit)
    ld = derived_link_script(recipe.load_address, recipe.link_script, workdir)
    elf = workdir / "kernel.elf"
    step = subprocess.run(recipe.link_command(objects=objects, output=elf, link_script=ld),
                          capture_output=True, text=True)
    if step.returncode:
        raise RuntimeError(f"link: {step.stderr[-1500:]}")
    return elf


def run_elf(elf: Path, simulator: str, timeout: int) -> tuple[str, float]:
    from merlin.runtime.backends import base as backends
    backend = backends.get_backend(TARGET)
    t0 = time.monotonic()
    console = backend.run_elf(elf, simulator=simulator, timeout=timeout)
    return console, time.monotonic() - t0


def parse_console(console: str) -> dict:
    got: dict = {}
    for line in console.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["METRIC", "cycles"]:
            got["cycles"] = int(parts[2])
        elif len(parts) == 3 and parts[0] == "DIGEST":
            got["digest"] = int(parts[2])
        elif parts == ["DONE"]:
            got["done"] = True
    return got


# ---------------------------------------------------------------------------------------------------
# the two arms of one conv layer
# ---------------------------------------------------------------------------------------------------
def conv_case(workload: dict, seed: int = 0):
    """Operands (NHWC input, HWIO weight, bias or None) and the exact int32 convolution plus bias."""
    k, cin, cout = workload["k"], workload["Cin"], workload["Cout"]
    stride, pad = workload.get("stride", 1), workload.get("padding", k // 2)
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, size=(1, workload["H"], workload["W"], cin), dtype=np.int64)
    w = rng.integers(-128, 128, size=(k, k, cin, cout), dtype=np.int64)
    bias = (rng.integers(-2**20, 2**20, size=cout, dtype=np.int64)
            if workload.get("bias", True) else None)
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    oh = (padded.shape[1] - k) // stride + 1
    ow = (padded.shape[2] - k) // stride + 1
    out = np.zeros((1, oh, ow, cout), dtype=np.int64)
    for fy in range(k):
        for fx in range(k):
            window = padded[:, fy:fy + stride * oh:stride, fx:fx + stride * ow:stride]
            out += np.einsum("nhwc,co->nhwo", window, w[fy, fx])
    if bias is not None:
        out = out + bias
    return x, w, bias, out.reshape(-1, cout).astype(np.int32), (oh, ow, stride, pad)


def voyager_arm(model_dir: Path, workload: dict, isa, acc_scale_ok: bool):
    schedule = vs.lower_conv(replay(load_model(model_dir / "model.json")), geometry_for(TARGET))
    x, w, bias, reference, _ = conv_case(workload)
    cin, cout = workload["Cin"], workload["Cout"]
    lhs, weight = x.reshape(-1, cin), w.reshape(-1, cout)
    operands = {"bias": bias[None, :]} if bias is not None else {}
    expected = vs.execute(schedule, lhs, weight, **operands)
    if not np.array_equal(expected, reference):
        raise AssertionError("execute() disagrees with the direct convolution")
    names = {"lhs": "lhs", "weight": "weight", "bias": "bias", "out": "out"}
    strides = {"lhs": cin, "weight": cout, "bias": cout * 4, "out": cout * 4}
    elems = {"lhs": 1, "weight": 1, "bias": 4, "out": 4}
    trace = schedule_instructions(schedule, isa, names=names, strides=strides, elems=elems,
                                  out_dtype="i32", acc_scale_ok=acc_scale_ok)
    tensors = [("lhs", "i8", lhs, lhs.shape), ("weight", "i8", weight, weight.shape)]
    if bias is not None:
        tensors.append(("bias", "i32", bias, bias.shape))
    tensors.append(("out", "i32", None, expected.shape))
    counts = {k.__name__: schedule.count(k) for k in (vs.Mvin, vs.AccMvin, vs.Preload, vs.Compute,
                                                      vs.Mvout)}
    return trace, tensors, expected, counts


def merlin_iface(workload: dict, oh: int, ow: int, stride: int, pad: int) -> str:
    h, w, k, cin, cout = workload["H"], workload["W"], workload["k"], workload["Cin"], workload["Cout"]
    ifm, wt, out = f"tensor<1x{h}x{w}x{cin}xi8>", f"tensor<{k * k * cin}x{cout}xi8>", \
        f"tensor<{oh * ow}x{cout}xi32>"
    return "\n".join([
        'module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", '
        'merlin_iface.abi_version = "0.1"} {',
        f'  %IFM = merlin_iface.tensor {{name = "IFM", role = "input"}} : {ifm}',
        f'  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : {wt}',
        f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_conv_rhs"}} : ({wt}) -> '
        "!merlin_iface.resident",
        f'  %Y0 = merlin_iface.conv2d %IFM, %W_res {{kernel = [{k}, {k}, {cin}, {cout}], stride = '
        f'[{stride}, {stride}], padding = [{pad}, {pad}, {pad}, {pad}], dilation = [1, 1], name = '
        f'"Y0", epilogue = [], output_dtype = "i32", layout = "nhwc"}} : ({ifm}, '
        f"!merlin_iface.resident) -> {out}",
        "  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()", "}", ""])


def merlin_arm(workload: dict, isa, ingest, workdir: Path):
    if workload.get("bias", True):
        raise UnsupportedConstruct("merlin_iface.conv2d has no bias operand")
    x, w, _, reference, (oh, ow, stride, pad) = conv_case(workload)
    workdir.mkdir(parents=True, exist_ok=True)
    iface = workdir / "conv.interface.mlir"
    iface.write_text(merlin_iface(workload, oh, ow, stride, pad))
    _, module = ingest.parse_verified(iface)
    program = ingest.extract_program(module)
    trace = isa.build_trace(program)          # raises the package's own refusal on capacity
    order = [n for n, spec in program.tensors.items() if spec.role in ("input", "weight", "bias",
                                                                          "output")]
    values = {"IFM": x.reshape(1, -1), "W": w.reshape(-1, workload["Cout"])}
    tensors = [(n, "i32" if n == "Y0" else "i8", values.get(n), (reference.shape if n == "Y0"
                                                                  else values[n].shape))
               for n in order]
    tensors = [(n, d, (None if n == "Y0" else v), s) for n, d, v, s in tensors]
    counts = {name: sum(1 for i in trace if i.name == name)
              for name in ("MVIN", "PRELOAD", "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE", "MVOUT")}
    return trace, tensors, reference, counts


# ---------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", action="append", required=True, type=Path,
                        help="a voyager_export.py output dir (model.json + manifest.json)")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="the reference package whose encoders pack both arms")
    parser.add_argument("--arms", nargs="+", default=["voyager", "merlin"])
    parser.add_argument("--simulators", nargs="+", default=["spike", "verilator"])
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    isa, ingest = load_package(args.package)
    from merlin.runtime.backends import base as backends
    recipe = backends.harness_build_recipe(TARGET)
    params_h = next(Path(root) / "include" / "gemmini_params.h" for root in recipe.include_roots
                    if (Path(root) / "include" / "gemmini_params.h").is_file())
    acc_scale_ok = acc_mvin_scales(params_h)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = runs_dir() / TARGET / "voyager-h2h" / "emit_c" / (stamp + (f"_{args.tag}" if args.tag
                                                                     else ""))
    builds = []
    for model_dir in args.model_dir:
        manifest = json.loads((model_dir / "manifest.json").read_text())
        workload = manifest["workload"]
        probe = model_dir.name
        for arm in args.arms:
            work = root / probe / arm
            record = {"probe": probe, "arm": arm, "workload": workload,
                      "voyager_commit": manifest.get("voyager_commit"), "workdir": str(work)}
            try:
                if arm == "voyager":
                    trace, tensors, expected, counts = voyager_arm(model_dir, workload, isa,
                                                                   acc_scale_ok)
                else:
                    trace, tensors, expected, counts = merlin_arm(workload, isa, ingest, work)
                record["op_counts"] = counts
                record["rocc_commands"] = sum(1 for i in trace if i.name != "FENCE")
                elf = build_elf(work, emit_llvm([t[0] for t in tensors], trace, isa),
                                render_harness(tensors))
                record["elf"] = str(elf)
                record["elf_sha256"] = hashlib.sha256(elf.read_bytes()).hexdigest()
                record["expected_digest"] = fnv1a64(np.ascontiguousarray(expected, "<i4").tobytes())
                builds.append(record)
            except (UnsupportedConstruct, ValueError) as exc:
                record["status"] = "refused"
                record["reason"] = f"{type(exc).__name__}: {exc}"
                builds.append(record)
            except AssertionError as exc:
                # The abstract schedule itself is wrong: no program is built from it.
                record["status"] = "inexact_lowering"
                record["reason"] = str(exc)
                builds.append(record)
            print(json.dumps({k: record.get(k) for k in ("probe", "arm", "status", "reason",
                                                          "rocc_commands")}), flush=True)

    jobs = [(b, sim) for b in builds if "elf" in b for sim in args.simulators]

    def run(job):
        build, sim = job
        try:
            console, wall = run_elf(Path(build["elf"]), sim, args.timeout)
        except Exception as exc:  # noqa: BLE001 -- recorded per run, never fatal to the table
            return {"probe": build["probe"], "arm": build["arm"], "simulator": sim,
                    "status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:600]}"}
        (Path(build["workdir"]) / f"console.{sim}.log").write_text(console)
        got = parse_console(console)
        exact = got.get("digest") == build["expected_digest"]
        return {"probe": build["probe"], "arm": build["arm"], "simulator": sim,
                "status": "pass" if exact and got.get("done") else "fail", "exact": exact,
                "cycles": got.get("cycles"), "wall_s": round(wall, 1),
                "cycle_accurate": sim != "spike"}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        runs = []
        for result in pool.map(run, jobs):
            print(json.dumps(result), flush=True)
            runs.append(result)

    os.environ.setdefault("MERLIN_EXT_VOYAGER_COMPILER",
                          str(merlin_dir().parent / "out" / "build" / "external" / "voyager-compiler"))
    pins = {}
    for name in ("gemmini_rtl", "voyager_compiler"):
        try:
            pins[name] = provenance.verify(name)
        except Exception as exc:  # noqa: BLE001 -- drift is recorded, not fatal
            print(f"pin {name}: {exc}")
    from merlin.runtime.backends import base as _b
    backend = _b.get_backend(TARGET)
    sims = {}
    for sim in args.simulators:
        locate = getattr(backend, f"{sim}_path", None)
        if callable(locate):
            sims[sim] = Path(locate())
    product = new_product("compare", version=1, target=TARGET,
                          notes="voyager_h2h conv schedules as bare-metal Gemmini programs"
                                + (f" [{args.tag}]" if args.tag else ""))
    doc = {"target": TARGET, "package": str(args.package), "acc_mvin_scale_supported": acc_scale_ok,
           "builds": builds, "runs": runs,
           "provenance": provenance.record(pins=pins, artifacts=sims)}
    (product.path / "results.json").write_text(json.dumps(doc, indent=1, default=str))
    lines = ["# Conv probes as bare-metal Gemmini programs", "",
             "| probe | arm | simulator | status | exact | cycles | RoCC commands |",
             "|---|---|---|---|---|---|---|"]
    commands = {(b["probe"], b["arm"]): b.get("rocc_commands") for b in builds}
    for r in runs:
        lines.append(f"| {r['probe']} | {r['arm']} | {r['simulator']} | {r['status']} | "
                     f"{r.get('exact')} | {r.get('cycles')} | {commands.get((r['probe'], r['arm']))} |")
    for b in builds:
        if b.get("status") == "refused":
            lines.append(f"| {b['probe']} | {b['arm']} | - | refused | - | - | {b['reason']} |")
    (product.path / "table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"product": str(product.path)}))
    return 0 if all(r["status"] == "pass" for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
