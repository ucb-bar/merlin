#!/usr/bin/env python3
"""One convolution, our own schedule, with the strided-load bit flipped and nothing else.

The three ResNet-50 projection shortcuts are 1x1 stride-2 convolutions, and the descriptors this repo
emits for them carry ``downsample = 0``. The loader therefore walks the input window one pixel at a
time and the mesh reads every other row of what it staged. The device has a bit for this
(``LoopConvLdInput``: the mvin's DRAM row stride doubles, the row iterator steps by two, the row count
halves), and it is only correct together with ``a_stride = stride >> downsample``, because the loader
having applied the stride is exactly why the execute unit must stop applying it.

This measures what that bit is worth, by taking the C the schedule ALREADY emits and rewriting those
two fields in place. Everything else -- the tiling, the loop nest, the pointers, the pads, the number
of descriptors -- is byte for byte the same function, so the difference between the two timed regions
is the bit and its obligation and nothing else. The rewrite is structural (balanced-paren argument
splitting over the two call sites, with every other argument asserted unchanged), not a text
substitution that could quietly hit something else.

Four timed regions: the schedule's own C, that C with the bit off (the control, which must match the
first), that C with the bit on, and the vendor call. All four outputs are compared element for element
against the vendor's, so a faster wrong answer is not mistaken for a result.

    conv_downsample_ab.py --target <target> --name g18 --in-dim 56 --in-channels 256 \
        --out-channels 512 --kernel 1 --stride 2 --padding 0 --emulator <gsim> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import sched_vendor_ab as AB

from merlin.common import provenance

RTL_PIN = "gemmini_rtl"

#: The two call sites the mode touches, and which argument of each carries it. The positions are the
#: header macro's own parameter order; they are CHECKED against the header rather than trusted, because
#: a position written here would be a claim about a signature this repo does not own.
_CONV_MACRO = "gemmini_loop_conv_ws"
_CONFIG_EX_MACRO = "gemmini_extended_config_ex"
_CONV_FIELD = "downsample"
_CONFIG_EX_FIELD = "A_stride"


def _provider_components(target: str):
    """This target-specific experiment requires its selected backend's predicate.

    Its scheduling implementation is the provider-owned ``gemmini_sched`` sibling
    of that backend package, not a module inferred from the requested target name.
    A provider lacking either capability cannot borrow a native implementation.
    """
    backend = AB.base.get_backend(target)
    predicate = getattr(backend, "conv_downsample_flag", None)
    if not callable(predicate):
        raise SystemExit(f"selected backend {target!r} lacks conv_downsample_flag required by this experiment")
    try:
        sched = importlib.import_module(f"{backend.__name__}.gemmini_sched")
    except ImportError as exc:
        raise SystemExit(f"selected backend {target!r} lacks its required scheduling module: {exc}") from exc
    return backend, sched, predicate


def _macro_parameters(header_text: str, macro: str) -> list[str]:
    """The macro's parameter names, in order, from its ``#define``.

    Read structurally: find the define, take the balanced parameter list, split it at the top level.
    """
    needle = "#define " + macro + "("
    index = header_text.find(needle)
    if index < 0:
        raise SystemExit(f"UNKNOWN macro {macro}: not defined in the target's header")
    start = index + len(needle) - 1
    end = _balanced_end(header_text, start)
    return [part.strip() for part in _split_top_level(header_text[start + 1 : end - 1], ",")]


def _balanced_end(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    raise SystemExit("unbalanced parentheses while reading a call")


def _split_top_level(text: str, sep: str) -> list[str]:
    out, depth, current = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
    out.append("".join(current))
    return out


def rewrite(definition: str, *, header_text: str, downsample: int, stride: int) -> str:
    """The same function with only ``downsample`` and ``A_stride`` changed.

    Returns the rewritten C. The two positions are looked up by NAME in the header's own parameter
    list and every other argument is carried through untouched, which is the property that makes this
    an A/B of one thing.
    """
    conv_params = _macro_parameters(header_text, _CONV_MACRO)
    ex_params = _macro_parameters(header_text, _CONFIG_EX_MACRO)
    if _CONV_FIELD not in conv_params:
        raise SystemExit(f"UNKNOWN field: {_CONV_MACRO} has no {_CONV_FIELD} parameter")
    if _CONFIG_EX_FIELD not in ex_params:
        raise SystemExit(f"UNKNOWN field: {_CONFIG_EX_MACRO} has no {_CONFIG_EX_FIELD} parameter")
    conv_at = conv_params.index(_CONV_FIELD)
    ex_at = ex_params.index(_CONFIG_EX_FIELD)

    def edit(text: str, macro: str, at: int, value: str, arity: int) -> str:
        out, rest, hits = [], text, 0
        while True:
            index = rest.find(macro + "(")
            if index < 0:
                break
            open_at = index + len(macro)
            end = _balanced_end(rest, open_at)
            args = _split_top_level(rest[open_at + 1 : end - 1], ",")
            if len(args) != arity:
                raise SystemExit(f"{macro} called with {len(args)} arguments, the header declares {arity}")
            args[at] = " " + value
            out.append(rest[:index] + macro + "(" + ",".join(args) + ")")
            rest = rest[end:]
            hits += 1
        if hits == 0:
            raise SystemExit(f"the emitted schedule never calls {macro}")
        return "".join(out) + rest

    body = edit(definition, _CONV_MACRO, conv_at, str(downsample), len(conv_params))
    body = edit(body, _CONFIG_EX_MACRO, ex_at, str(stride >> downsample), len(ex_params))
    return body


PROGRAM = """/* GENERATED by conv_downsample_ab.py -- one convolution, the strided-load bit flipped. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/gemmini_counter.h"

#define BLOB(sym, file, type) \\
    __asm__(".section .rodata\\n.balign 64\\n.global " #sym "\\n" #sym ":\\n.incbin \\"" file "\\"\\n.previous\\n"); \\
    extern const type sym[];
{blobs}

static {ctype} OUT_V[{elements}] row_align(1);
static {ctype} OUT_A[{elements}] row_align(1);
static {ctype} OUT_B[{elements}] row_align(1);
static {ctype} OUT_C[{elements}] row_align(1);

#define NSLOTS {nslots}
static const int SLOT_CODE[NSLOTS] = {{{codes}}};
static const char *SLOT_NAME[NSLOTS] = {{{names}}};

static void probe_arm(void) {{
    counter_reset();
    for (int i = 0; i < NSLOTS; i++) counter_configure(i, SLOT_CODE[i]);
}}

static void probe_report(const char *tag, uint64_t cycles) {{
    printf("PROBE_CYCLES %s %llu\\n", tag, (unsigned long long)cycles);
    for (int i = 0; i < NSLOTS; i++)
        printf("PROBE_COUNTER %s %s %u\\n", tag, SLOT_NAME[i], counter_read(i));
}}

static void compare(const char *tag, const {ctype} *got) {{
    long long diffs = 0, first = -1, sum = 0;
    for (size_t i = 0; i < {elements}; i++) {{
        sum += (long long)got[i];
        if (got[i] != OUT_V[i]) {{ if (first < 0) first = (long long)i; diffs++; }}
    }}
    printf("PROBE_DIFFS %s %lld first=%lld sum=%lld verdict=%s\\n",
           tag, diffs, first, sum, diffs == 0 ? "BIT_EXACT" : "DIFFERS");
}}

{definition}
{definition_ds0}
{definition_ds1}

int main(void) {{
    gemmini_flush(0);
    uint64_t t0, cyc;
    printf("MERLIN_PROFILE warmup begin\\n");
    {vendor}
    {call_a}
    {call_b}
    {call_c}
    printf("MERLIN_PROFILE warmup end rc=0\\n");
    printf("MERLIN_PROFILE measured begin\\n");
    probe_arm(); t0 = read_cycles(); {vendor} cyc = read_cycles() - t0; probe_report("vendor", cyc);
    probe_arm(); t0 = read_cycles(); {call_a} cyc = read_cycles() - t0; probe_report("ours", cyc);
    probe_arm(); t0 = read_cycles(); {call_b} cyc = read_cycles() - t0; probe_report("hand_ds0", cyc);
    probe_arm(); t0 = read_cycles(); {call_c} cyc = read_cycles() - t0; probe_report("hand_ds1", cyc);
    compare("ours", OUT_A);
    compare("hand_ds0", OUT_B);
    compare("hand_ds1", OUT_C);
    printf("PROBE_GROUP {name} elements={elements}\\n");
    printf("MERLIN_PROFILE measured end rc=0\\n");
    return 0;
}}
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--in-dim", required=True, type=int)
    parser.add_argument("--in-channels", required=True, type=int)
    parser.add_argument("--out-channels", required=True, type=int)
    parser.add_argument("--kernel", required=True, type=int)
    parser.add_argument("--stride", required=True, type=int)
    parser.add_argument("--padding", required=True, type=int)
    parser.add_argument("--scale", default=0.0234375, type=float)
    parser.add_argument("--seed", default=17, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--emulator", type=Path)
    parser.add_argument("--max-cycles", default="400000000")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)

    probe = importlib.import_module("conv_shape_counter_probe")

    backend, sched, downsample_flag = _provider_components(args.target)
    iset = backend.sched_instruction_set()
    harness = Path(backend.rocc_tests_dir())
    compiler = backend.gcc_path()
    header_text = (harness / "include" / "gemmini.h").read_text(encoding="utf-8")

    flag = downsample_flag(
        kernel=args.kernel,
        stride=args.stride,
        padding=args.padding,
        in_rows=args.in_dim,
        in_cols=args.in_dim,
        pooled=False,
        header_text=header_text,
    )
    if not flag:
        raise SystemExit(f"{args.name} is not eligible for the strided load; there is nothing to A/B")

    slots = list(probe.PARTITION) + ["RDMA_BYTES_REC"]
    codes = probe.counter_codes(harness / "include" / "gemmini_counter.h", slots)

    case = AB.conv_case(
        sched, iset.facts, iset,
        name=args.name, in_dim=args.in_dim, ci=args.in_channels, co=args.out_channels,
        kernel=args.kernel, stride=args.stride, padding=args.padding, pool=(0, 0, 0),
        scale=args.scale, relu=True, seed=args.seed,
    )  # fmt: skip
    symbol = f"{args.name}_ours"
    ds0 = rewrite(
        case["definition"].replace(symbol, symbol + "_ds0"),
        header_text=header_text, downsample=0, stride=args.stride,
    )  # fmt: skip
    ds1 = rewrite(
        case["definition"].replace(symbol, symbol + "_ds1"),
        header_text=header_text, downsample=1, stride=args.stride,
    )  # fmt: skip
    # The control must be the schedule's own C with nothing but the symbol changed: if it is not, the
    # rewrite moved something and the third region is not an A/B of one field.
    if ds0 != case["definition"].replace(symbol, symbol + "_ds0"):
        raise SystemExit("the rewrite changed the control; the fields are not where the header says")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, array in case["arrays"].items():
        (args.out / f"{name}.bin").write_bytes(AB.np.ascontiguousarray(array).tobytes())
    blobs = "\n".join(f'BLOB({s}, "{s}.bin", {"acc_t" if s == "BS" else "elem_t"})' for s in case["arrays"])
    actuals = "(elem_t *)IN, (elem_t *)WT, (acc_t *)BS"
    source = args.out / "probe.c"
    source.write_text(
        PROGRAM.format(
            blobs=blobs,
            ctype=case["ctype"],
            elements=case["elements"],
            definition=case["definition"],
            definition_ds0=ds0,
            definition_ds1=ds1,
            vendor=case["vendor"],
            call_a=f"{symbol}({actuals}, OUT_A);",
            call_b=f"{symbol}_ds0({actuals}, OUT_B);",
            call_c=f"{symbol}_ds1({actuals}, OUT_C);",
            name=args.name,
            nslots=len(slots),
            codes=", ".join(str(codes[s]) for s in slots),
            names=", ".join(f'"{s}"' for s in slots),
        ),
        encoding="utf-8",
    )

    common = harness / "riscv-tests" / "benchmarks" / "common"
    includes = [
        f"-I{harness / 'riscv-tests'}",
        f"-I{harness / 'riscv-tests' / 'env'}",
        f"-I{harness}",
        f"-I{common}",
    ]
    objects, log = [], []
    for src in (source, common / "syscalls.c", common / "crt.S"):
        unit = args.out / f"{src.stem}.o"
        done = subprocess.run(
            [str(compiler), *AB.CFLAGS, *includes, f"-Wa,-I{args.out}", "-c", str(src), "-o", str(unit)],
            capture_output=True, text=True, cwd=args.out,
        )  # fmt: skip
        log.append(done.stdout + done.stderr)
        if done.returncode:
            (args.out / "build.log").write_text("".join(log), encoding="utf-8")
            raise SystemExit(f"compile of {src.name} failed:\n{done.stderr[-3000:]}")
        objects.append(str(unit))
    elf = args.out / "probe.elf"
    done = subprocess.run(
        [str(compiler), *AB.CFLAGS, "-T", str(common / "test.ld"), *objects, "-o", str(elf)],
        capture_output=True, text=True, cwd=args.out,
    )  # fmt: skip
    (args.out / "build.log").write_text("".join(log) + done.stdout + done.stderr, encoding="utf-8")
    if done.returncode:
        raise SystemExit(f"link failed:\n{done.stderr[-3000:]}")

    try:
        pins = {RTL_PIN: provenance.verify(RTL_PIN)}
        note = None
    except Exception as why:
        pins, note = {}, f"UNKNOWN({type(why).__name__}: {why})"
    (args.out / "case.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "target": args.target,
                "downsample_eligible": flag,
                "attrs": case["attrs"],
                "provenance": provenance.record(
                    pins=pins,
                    sources=[str(harness / "include" / "gemmini.h")],
                    artifacts={"emulator": str(args.emulator)} if args.emulator else None,
                    extra={"pin_note": note} if note else None,
                ),
                "elf": str(elf),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"built {elf}: {case['attrs']}")
    if args.build_only or args.emulator is None:
        return 0
    done = subprocess.run(
        [str(args.emulator), str(elf), f"+max-cycles={args.max_cycles}", f"+loadmem={elf}"],
        capture_output=True, text=True, cwd=args.out,
    )  # fmt: skip
    (args.out / "run.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    for line in (done.stdout + done.stderr).splitlines():
        if line.startswith("PROBE_"):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
