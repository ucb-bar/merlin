#!/usr/bin/env python3
"""Schedule → affinity-directive sidecar generator.

Reads MOSEK's schedule.json and the source MLIR (post-flow phase), then
emits a new MLIR with `stream.affinity = #hal.device.affinity<@cpu | @qnn_gpu
| @qnn_hta>` attached to every `flow.dispatch` site according to the
schedule's `hardware_target` field. Also rewrites the module header to
declare three device globals so the downstream compile can fan dispatches
out across CPU + QNN_GPU + QNN_HTA.

Inputs:
  --schedule    /tmp/yolov8n_mosek/breakdowns/schedule.json
  --flow-mlir   /tmp/yolov8n_het/yolov8n_flow.mlir  (post-flow IR snapshot)
  --out         /tmp/yolov8n_het/yolov8n_pinned.mlir
  --default-target CPU_P
                Target for any flow.dispatch whose name is absent from the
                schedule (e.g., initializers, late-pass dispatches).

Output:
  A pinned MLIR ready for: iree-compile --compile-from=flow ... with three
  --iree-hal-target-device flags (CPU/QNN_GPU/QNN_HTA).
"""

import argparse
import json
import pathlib
import re
import sys

TARGET_DEVICE_NAME = {
    "CPU_P": "cpu",
    "QNN_GPU": "qnn_gpu",
    "QNN_HTA": "qnn_hta",
    # Aliases that may appear in older schedules.
    "CPU": "cpu",
    "CPU_E": "cpu",
}

# Device-target attributes IREE recognizes for each backend.
DEVICE_TARGET_ATTR = {
    "cpu": (
        '#hal.device.target<"local", '
        '[#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", '
        '{cpu = "cortex-a77", cpu_features = "+v8.2a,+fullfp16,+dotprod,+i8mm", '
        'data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", '
        "native_vector_size = 16 : i64, "
        'target_abi = "lp64", '
        'target_triple = "aarch64-unknown-unknown-eabi-elf", '
        'ukernels = "all"}>]> : !hal.device'
    ),
    "qnn_gpu": (
        '#hal.device.target<"qnn", '
        '[#hal.executable.target<"qnn", "qnn-context-binary", '
        '{qnn_backend = "gpu", opaque_binary = true}>]> : !hal.device'
    ),
    "qnn_hta": (
        '#hal.device.target<"qnn", '
        '[#hal.executable.target<"qnn", "qnn-context-binary", '
        '{qnn_backend = "hta", opaque_binary = true}>]> : !hal.device'
    ),
}

# `flow.dispatch @<exec>::@<export>(...)` — capture the executable name as
# group 1 so we can look it up in schedule.json.
DISPATCH_RE = re.compile(r"flow\.dispatch\s+@([A-Za-z0-9_$]+)::")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", type=pathlib.Path, required=True)
    p.add_argument("--flow-mlir", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument(
        "--default-target",
        default="CPU_P",
        help="hardware_target used for any flow.dispatch not listed in the " "schedule (default: CPU_P).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sched = json.loads(args.schedule.read_text())
    name_to_dev = {}
    for name, info in sched["dispatches"].items():
        tgt = info.get("hardware_target") or info.get("machine") or ""
        if tgt not in TARGET_DEVICE_NAME:
            print(
                f"WARN: dispatch {name} has unknown target {tgt!r}; " f"falling back to {args.default_target}",
                file=sys.stderr,
            )
            tgt = args.default_target
        name_to_dev[name] = TARGET_DEVICE_NAME[tgt]

    default_dev = TARGET_DEVICE_NAME[args.default_target]
    text = args.flow_mlir.read_text()

    # Strip the existing single-device module header — we'll replace it.
    # The original module header looks like:
    #   module attributes {stream.affinity.default = #hal.device.affinity<@"..."">} {
    # plus a util.global private @"..." = #hal.device.target<...> : !hal.device
    text = re.sub(
        r"module attributes \{stream\.affinity\.default = #hal\.device\.affinity<@\"[^\"]+\">\} \{\n",
        "module attributes {stream.affinity.default = " f"#hal.device.affinity<@{default_dev}>}} {{\n",
        text,
        count=1,
    )
    # Remove the existing ugly-named device util.global line.
    text = re.sub(
        r"^\s*util\.global private @\"#hal\.device\.target<[^\n]+\n",
        "",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    # Insert three new device globals + the default affinity reference at the
    # top of the module body.
    decl_block = "\n".join(f"  util.global private @{n} = {a}" for n, a in DEVICE_TARGET_ATTR.items())
    text = text.replace(
        f"#hal.device.affinity<@{default_dev}>}} {{\n",
        f"#hal.device.affinity<@{default_dev}>}} {{\n{decl_block}\n",
        1,
    )

    # Rewrite every stream.affinity.default reference to the ugly device name
    # so other globals (hoisted constants) point at our default device.
    text = re.sub(
        r"#hal\.device\.affinity<@\"#hal\.device\.target<[^\"]+\">",
        f"#hal.device.affinity<@{default_dev}>",
        text,
    )

    # Inject per-dispatch affinity. Line-by-line approach is simpler than a
    # multi-clause regex because flow.dispatch lines can carry an existing
    # attr block (e.g., `{iree.dispatch_id = N : i64}`) that we have to
    # merge into rather than duplicate.
    pinned = unknown = 0
    out_lines = []
    for line in text.splitlines(keepends=True):
        m = DISPATCH_RE.search(line)
        if m is None:
            out_lines.append(line)
            continue
        name = m.group(1)
        dev = name_to_dev.get(name)
        if dev is None:
            unknown += 1
            dev = default_dev
        else:
            pinned += 1
        aff = f"stream.affinity = #hal.device.affinity<@{dev}>"

        # Find the part of the line *after* the closing `)` of the dispatch
        # call. Everything up to and including that `)` is the prefix. Past
        # it we either have ` {existing attrs}` then ` : ...` or just ` : ...`.
        paren_open = line.find("(", m.end())
        if paren_open < 0:
            out_lines.append(line)
            continue
        # Match the closing paren for THIS open paren (depth-balanced).
        depth = 1
        i = paren_open + 1
        while i < len(line) and depth > 0:
            if line[i] == "(":
                depth += 1
            elif line[i] == ")":
                depth -= 1
            i += 1
        if depth != 0:
            out_lines.append(line)
            continue
        prefix = line[:i]  # through and including the matching `)`
        tail = line[i:]

        # Trim leading whitespace from tail; identify an optional `{...}` block.
        stripped = tail.lstrip()
        ws = tail[: len(tail) - len(stripped)]
        if stripped.startswith("{"):
            close = stripped.find("}")
            if close < 0:
                out_lines.append(line)
                continue
            inner = stripped[1:close].strip().rstrip(",").strip()
            rest = stripped[close + 1 :]
            merged = f"{{{inner}, {aff}}}" if inner else f"{{{aff}}}"
            out_lines.append(f"{prefix}{ws}{merged}{rest}")
        else:
            out_lines.append(f"{prefix} {{{aff}}}{tail}")
    text = "".join(out_lines)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)

    # Summary.
    by_dev = {}
    for d in name_to_dev.values():
        by_dev[d] = by_dev.get(d, 0) + 1
    print(f"Wrote pinned MLIR: {args.out}")
    print(f"  Schedule entries: {len(name_to_dev)}")
    for d, n in sorted(by_dev.items()):
        print(f"    {d:<9}: {n} dispatches")
    print(f"  Flow.dispatch sites pinned-from-schedule: {pinned}")
    print(f"  Flow.dispatch sites fallback→{default_dev}: {unknown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
