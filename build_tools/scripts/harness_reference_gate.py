#!/usr/bin/env python3
"""Gate: the logical runner-owned harness grades correctly, independent of any compiler.

For every capsule in the selected trees (and a few synthetic buffers covering the whole-program path)
this gate:

(a) renders the generic logical harness (``merlin.targetgen.contract.harness_render``) and links it
    against a NAIVE C reference kernel written straight from the capsule's semantics
    (``merlin.targetgen.contract.reference_kernel``: plain loops, no accelerator instruction, no target
    library), runs the ELF on spike, and requires the printed results to equal
    ``merlin.runtime.reference.reference_outputs`` EXACTLY -- cold and warm;
(b) links the same harness against deliberately wrong kernels (an off-by-one index, a wrong scale, a
    transpose, an output never written, two inputs swapped) and requires every one to FAIL the same
    comparison. A mutation whose output is provably identical to the correct one for this capsule
    (e.g. a transpose of a 1-row result) is reported as vacuous, not as caught;
(c) requires the harness to expose no target library: it includes only the logical ABI's standard
    headers, its object references no symbol beyond the kernel entry and the console routines, and
    (with ``--forbidden-header``) none of that header's routines is named in the harness or linked.

The build uses Merlin's own bare-metal spike environment (``merlin/runtime/baremetal/spike``), not a
target's vendor tree, so nothing here depends on a target support package.

Usage::

    python build_tools/scripts/harness_reference_gate.py --target <t> \\
        [--trees isa layers model_slices _model_layers] [--forbidden-header <vendor header>] \\
        [--jobs 16] [--report out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import prod
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "merlin" / "python"))
_DEFAULT_TREES = ("isa", "layers", "model_slices", "_model_layers")
_ALLOWED_INCLUDES = ("#include <stdint.h>", "#include <stdio.h>")
_CONSOLE_SYMBOLS = frozenset({"printf", "memset", "memcpy", "puts", "putchar", "sprintf"})

_SHIM = r"""
#include "htif.h"
int merlin_harness_main(void);
int main(long hart) {
  if (hart != 0) for (;;) {}
  console_init();
  htif_exit(merlin_harness_main());
}
"""


# --------------------------------------------------------------------------------------------------
# build + run on spike
# --------------------------------------------------------------------------------------------------
def _toolchain():
    from merlin.common.paths import runtime_dir
    from merlin.runtime.backends import spike

    env = runtime_dir() / "baremetal" / "spike"
    return spike.gcc_path(), spike.spike_path(), env


_CFLAGS = ("-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin-printf")


def _compile(gcc: Path, source: Path, obj: Path, *extra: str) -> None:
    proc = subprocess.run(
        [str(gcc), *_CFLAGS, *extra, "-c", str(source), "-o", str(obj)], capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        raise RuntimeError(f"compile {source.name} failed:\n{proc.stderr[-3000:]}")


def _runtime_objects(work: Path) -> list[Path]:
    gcc, _spike, env = _toolchain()
    objs = []
    for name, extra in (
        ("crt.S", ()),
        ("htif.c", ()),
        ("printf_min.c", ()),
        ("libc_min.c", ("-Dputs=merlin_libc_puts", "-Dabort=merlin_libc_abort")),
    ):
        obj = work / f"rt_{name}.o"
        if not obj.exists():
            _compile(gcc, env / name, obj, "-I", str(env), *extra)
        objs.append(obj)
    shim_c = work / "rt_shim.c"
    shim_c.write_text(_SHIM, encoding="utf-8")
    shim_o = work / "rt_shim.o"
    _compile(gcc, shim_c, shim_o, "-I", str(env))
    return objs + [shim_o]


def _link_and_run(work: Path, harness_o: Path, kernel_o: Path, runtime: list[Path], tag: str, timeout: int) -> str:
    gcc, spike, env = _toolchain()
    elf = work / f"{tag}.elf"
    proc = subprocess.run(
        [
            str(gcc),
            *_CFLAGS,
            "-nostdlib",
            "-nostartfiles",
            "-T",
            str(env / "link.ld"),
            *map(str, runtime),
            str(harness_o),
            str(kernel_o),
            "-lgcc",
            "-o",
            str(elf),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"link {tag} failed:\n{proc.stderr[-3000:]}")
    run = subprocess.run(
        [str(spike), "--isa=rv64gc_zicntr", "-p1", str(elf)], capture_output=True, text=True, timeout=timeout
    )
    if run.returncode != 0:
        raise RuntimeError(f"spike exited {run.returncode}: {run.stdout[-500:]}{run.stderr[-500:]}")
    return run.stdout


def _flatten(value) -> list:
    if isinstance(value, list):
        return [x for item in value for x in _flatten(item)]
    return [value]


def _compare(console: str, reference: dict) -> tuple[bool, str]:
    from merlin.runtime.backends.base import parse_console

    try:
        outputs, _metrics = parse_console(console)
    except Exception as exc:  # noqa: BLE001 -- an unparseable console is a failed comparison
        return False, f"console: {exc}"[:300]
    if set(outputs) != set(reference):
        return False, f"output roster {sorted(outputs)} != reference {sorted(reference)}"
    for name, expected in reference.items():
        got, want = _flatten(outputs[name]), _flatten(expected)
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            return False, f"{name}: {len(got)} vs {len(want)} values, first difference at {first}"
    return True, "exact"


# --------------------------------------------------------------------------------------------------
# corpus + classification
# --------------------------------------------------------------------------------------------------
def command_shape(cb: dict) -> str:
    """A label for reporting only (the logical ABI itself has one rule for every non-whole-program buffer)."""
    if (cb.get("kernel_abi") or {}).get("kind") == "whole_program":
        return "whole_program"
    commands = cb.get("commands") or []
    if not commands:
        return "host_lane"
    ops = [c.get("opcode") for c in commands]
    whole = [o for o in ops if o in ("ATTENTION_QK", "ATTENTION_PV", "BATCHED_MATMUL", "CONV2D")]
    if whole:
        return "native_whole_op:" + "+".join(sorted(set(whole)))
    if "RES_PACK" not in ops and any(
        c.get("opcode") == "MOVEMENT"
        or (c.get("opcode") == "VECTOR_MAP" and (c.get("attributes") or {}).get("combine") == "identity")
        for c in commands
    ):
        return "movement"
    return "resident_matmul"


def _as_whole_program(cb: dict) -> dict:
    """The same computation behind an explicit whole-program pointer boundary."""
    from merlin.targetgen.contract.harness_render import explicit_whole_program, logical_abi

    return explicit_whole_program(cb, logical_abi())


def _corpus(args) -> list[tuple[str, dict]]:
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    out = []
    root = Path(args.capsule_root)
    for tree in args.trees:
        for capsule in sorted((root / tree).iterdir()):
            source = capsule / "capsule.interface.mlir"
            if not source.is_file():
                continue
            try:
                cb = parse_interface_mlir(source.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                out.append((f"{tree}/{capsule.name}", {"__error__": f"parse: {exc}"}))
                continue
            out.append((f"{tree}/{capsule.name}", cb))
    # The kernel-ABI gate's probe buffers cover command shapes no capsule in a tree may exercise.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_kernel_abi_gate", _HERE.parent / "check_kernel_abi_arg_order.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    probes = [
        ("whole_program", gate._probe_whole_program()),
        ("movement", gate._probe_movement()),
        ("resident_matmul", gate._probe_resident_matmul()),
    ]
    probes += [
        (f"native_{op}", gate._probe_native_whole_op(op))
        for op in ("CONV2D", "ATTENTION_QK", "ATTENTION_PV", "BATCHED_MATMUL")
    ]
    out += [(f"probe/{name}", {"target": args.target, **cb}) for name, cb in probes]
    # The whole-program path, exercised with the same semantics as a capsule that has a matmul + bias.
    for label, cb in list(out):
        if label.endswith("B0_quantized_linear_i8") or label.endswith("A2_single_tile_matmul"):
            out.append((f"synthetic/whole_program[{label}]", _as_whole_program(cb)))
    return out


def _vacuous(mutation: str, cb: dict, reference: dict) -> bool:
    """True when the mutation provably cannot change this capsule's printed results."""
    from merlin.runtime.reference import reference_outputs
    from merlin.targetgen.contract.harness_render import logical_abi, logical_interface

    buffers = [b for b in logical_interface(cb, logical_abi()) if b.kind == "output"]
    if mutation == "uninitialized_output":
        return False
    if mutation == "swapped_inputs":
        from merlin.runtime.commandbuffer import materialize_inputs

        inputs = [b for b in logical_interface(cb, logical_abi()) if b.kind == "input"]
        pair = next(
            (
                (x, y)
                for i, x in enumerate(inputs)
                for y in inputs[i + 1 :]
                if x.elements == y.elements and x.dtype == y.dtype
            ),
            None,
        )
        leaves = materialize_inputs(cb)
        swapped = {
            pair[0].name: _nest(leaves[pair[1].name].data, pair[0].shape),
            pair[1].name: _nest(leaves[pair[0].name].data, pair[1].shape),
        }
        return reference_outputs(cb, swapped) == reference
    for buf in buffers:
        flat = _flatten(reference[buf.name])
        n, (rows, cols) = len(flat), buf.matrix
        if mutation == "off_by_one":
            mutated = [flat[(e + 1) % n] for e in range(n)]
        elif mutation == "wrong_scale":
            mutated = [_wrap(2 * v, buf.dtype) for v in flat]
        else:  # transposed
            mutated = [flat[(e % cols) * rows + e // cols] for e in range(n)]
        if mutated != flat:
            return False
    return True


def _wrap(value: int, dtype: str) -> int:
    bits = int(dtype[1:])
    value &= (1 << bits) - 1
    return value - (1 << bits) if dtype.startswith("i") and value >= 1 << (bits - 1) else value


def _nest(flat, shape):
    if len(shape) <= 1:
        return list(flat)
    step = len(flat) // shape[0]
    return [_nest(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


# --------------------------------------------------------------------------------------------------
# library-exposure check (c)
# --------------------------------------------------------------------------------------------------
def _identifiers(text: str) -> set[str]:
    out, word = set(), []
    for ch in text + " ":
        if ch.isalnum() or ch == "_":
            word.append(ch)
        elif word:
            out.add("".join(word))
            word = []
    return out


def forbidden_routines(header: Path) -> set[str]:
    """Routine and function-like macro names a vendor header defines (structural scan, no regex)."""
    names = set()
    for raw in header.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("#define"):
            head = line[len("#define") :].strip()
            name, paren, _ = head.partition("(")
            if paren and name and all(c.isalnum() or c == "_" for c in name):
                names.add(name)
            continue
        if raw[:1].isspace() or "(" not in line or line.startswith(("//", "/*", "*")):
            continue
        before = line.partition("(")[0].split()
        if len(before) >= 2 and all(c.isalnum() or c == "_" or c == "*" for c in before[-1]):
            names.add(before[-1].lstrip("*"))
    return {n for n in names if n and not n[0].isdigit()}


def _exposure(text: str, harness_o: Path, entry: str, forbidden: set[str]) -> list[str]:
    problems = []
    includes = [line.strip() for line in text.splitlines() if line.strip().startswith("#include")]
    stray = [line for line in includes if line not in _ALLOWED_INCLUDES]
    if stray:
        problems.append(f"harness includes {stray}")
    named = sorted(_identifiers(text) & forbidden)
    if named:
        problems.append(f"harness names vendor routines {named[:8]}")
    nm = Path(str(_toolchain()[0]).replace("gcc", "nm"))
    proc = subprocess.run([str(nm), str(harness_o)], capture_output=True, text=True, timeout=60)
    undefined = {line.split()[-1] for line in proc.stdout.splitlines() if line.split()[:1] == ["U"]}
    extra = sorted(undefined - _CONSOLE_SYMBOLS - {entry})
    if extra:
        problems.append(f"harness object references {extra}")
    return problems


# --------------------------------------------------------------------------------------------------
# one capsule
# --------------------------------------------------------------------------------------------------
def _check(label: str, cb: dict, args, work_root: Path, runtime: list[Path], forbidden: set[str]) -> dict:
    from merlin.runtime.reference import reference_outputs
    from merlin.targetgen.contract import harness_render as hr
    from merlin.targetgen.contract.reference_kernel import (
        MUTATIONS,
        ReferenceKernelUnsupported,
        render_reference_kernel,
    )

    row = {"capsule": label, "shape": command_shape(cb) if "__error__" not in cb else "unparsed"}
    if "__error__" in cb:
        return {**row, "status": "skipped", "why": cb["__error__"]}
    try:
        abi, hooks = hr.resolve(args.target)
        buffers = hr.logical_interface(cb, abi)
        macs = _macs(cb)
        if macs > args.max_macs or sum(b.elements for b in buffers) > args.max_elements:
            return {
                **row,
                "status": "skipped",
                "why": f"{macs} MACs / {sum(b.elements for b in buffers)} elements over the gate budget",
            }
        reference = reference_outputs(cb)
        kernel_c = render_reference_kernel(cb, symbol=hooks.entry_symbol)
    except (ReferenceKernelUnsupported, hr.HarnessRenderError) as exc:
        return {**row, "status": "skipped", "why": f"{type(exc).__name__}: {exc}"[:300]}
    except Exception as exc:  # noqa: BLE001 -- the reference itself refused (unmodeled op, float policy)
        return {**row, "status": "skipped", "why": f"reference: {type(exc).__name__}: {exc}"[:300]}
    if any(not all(isinstance(v, int) for v in _flatten(values)) for values in reference.values()):
        return {**row, "status": "skipped", "why": "non-integer reference"}
    gcc = _toolchain()[0]
    work = work_root / label.replace("/", "__").replace("[", "_").replace("]", "_")
    work.mkdir(parents=True, exist_ok=True)
    result = {**row, "status": "pass", "checks": {}, "mutations": {}, "exposure": []}
    try:
        for state in ("cold", "warm"):
            text = hr.render_with(cb, abi=abi, hooks=hooks, cache_state=state)
            harness_c = work / f"harness_{state}.c"
            harness_c.write_text(text, encoding="utf-8")
            harness_o = work / f"harness_{state}.o"
            _compile(gcc, harness_c, harness_o, "-Dmain=merlin_harness_main")
            if state == "cold":
                result["exposure"] = _exposure(text, harness_o, hooks.entry_symbol, forbidden)
            for mutation in (None, *MUTATIONS):
                tag = f"{state}_{mutation or 'reference'}"
                try:
                    source = render_reference_kernel(cb, symbol=hooks.entry_symbol, mutation=mutation)
                except ReferenceKernelUnsupported as exc:
                    result["mutations"][tag] = f"n/a: {exc}"
                    continue
                if mutation is not None and _vacuous(mutation, cb, reference):
                    result["mutations"][tag] = "vacuous"
                    continue
                kernel_c = work / f"kernel_{tag}.c"
                kernel_c.write_text(source, encoding="utf-8")
                kernel_o = work / f"kernel_{tag}.o"
                _compile(gcc, kernel_c, kernel_o)
                try:
                    console = _link_and_run(work, harness_o, kernel_o, runtime, tag, args.timeout)
                    ok, why = _compare(console, reference)
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    ok, why = False, f"run: {exc}"[:300]
                if mutation is None:
                    result["checks"][state] = why if ok else f"MISMATCH {why}"
                    if not ok:
                        result["status"] = "fail"
                else:
                    result["mutations"][tag] = "caught" if not ok else "MISSED"
                    if ok:
                        result["status"] = "fail"
        if result["exposure"]:
            result["status"] = "fail"
    except Exception as exc:  # noqa: BLE001
        result.update(status="error", why=f"{type(exc).__name__}: {exc}"[:800])
    return result


def _macs(cb: dict) -> int:
    tensors = cb.get("tensors") or {}
    total = 0
    resident = {}
    for cmd in cb.get("commands") or []:
        ops, attrs = cmd.get("operands") or {}, cmd.get("attributes") or {}
        if cmd["opcode"] == "RES_PACK":
            resident[ops["dst"]] = ops["src"]
        elif cmd["opcode"] in ("MATMUL", "MATMUL_RESIDENT"):
            lhs = (tensors.get(ops["lhs"]) or {}).get("shape") or [0, 0]
            rhs = (tensors.get(resident.get(ops["rhs"], ops["rhs"])) or {}).get("shape") or [0, 0]
            total += prod(lhs) * rhs[-1]
        elif cmd["opcode"] == "CONV2D":
            kernel = attrs.get("kernel") or [0, 0, 0, 0]
            dst = (tensors.get(ops["dst"]) or {}).get("shape")
            ifm = (tensors.get(ops["ifm"]) or {}).get("shape") or [0]
            total += (prod(dst) if dst else prod(ifm)) * prod(kernel[:3])
        elif cmd["opcode"] in ("BATCHED_MATMUL", "ATTENTION_QK", "ATTENTION_PV"):
            shapes = [prod((tensors.get(v) or {}).get("shape") or [1]) for v in ops.values()]
            total += max(shapes) * 64
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True)
    ap.add_argument("--capsule-root", default=str(_ROOT / "merlin" / "contract" / "capsules"))
    ap.add_argument("--trees", nargs="+", default=list(_DEFAULT_TREES))
    ap.add_argument(
        "--forbidden-header",
        action="append",
        default=[],
        help="a vendor library header none of whose routines may be visible to candidate code",
    )
    ap.add_argument("--max-macs", type=int, default=4_000_000)
    ap.add_argument("--max-elements", type=int, default=400_000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="harness_reference_gate_", dir=os.environ.get("TMPDIR")))
    work.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_objects(work)
    forbidden: set[str] = set()
    for header in args.forbidden_header:
        forbidden |= forbidden_routines(Path(header))
    corpus = _corpus(args)
    print(f"[gate] {len(corpus)} buffers; forbidden vendor routines: {len(forbidden)}; work dir {work}")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda item: _check(item[0], item[1], args, work, runtime, forbidden), corpus))
    by_shape: dict[str, Counter] = defaultdict(Counter)
    mutations: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        by_shape[r["shape"]][r["status"]] += 1
        for tag, verdict in (r.get("mutations") or {}).items():
            mutations[tag.split("_", 1)[1]][verdict.split(":")[0]] += 1
    print("\n[gate] per command shape (pass = reference kernel exact cold+warm, every mutation caught or vacuous):")
    for shape, counts in sorted(by_shape.items()):
        print(f"  {shape:42s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("\n[gate] mutations:")
    for mutation, counts in sorted(mutations.items()):
        print(f"  {mutation:22s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    bad = [r for r in results if r["status"] in ("fail", "error")]
    for r in bad[:30]:
        print(
            f"  ! {r['status']} {r['capsule']} [{r['shape']}] {r.get('why', '')} {r.get('checks')} "
            f"{ {k: v for k, v in (r.get('mutations') or {}).items() if v == 'MISSED'} } {r.get('exposure')}"
        )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(
                {"schema": "harness_reference_gate_v1", "target": args.target, "trees": args.trees, "results": results},
                indent=2,
            )
            + "\n"
        )
        print(f"[gate] report: {args.report}")
    passed = sum(1 for r in results if r["status"] == "pass")
    if not passed:
        print("[gate] no buffer was exercised end to end -- NOT a pass")
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
