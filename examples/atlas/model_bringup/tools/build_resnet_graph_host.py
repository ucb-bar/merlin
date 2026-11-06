#!/usr/bin/env python3
"""Link a bound 54-kernel host graph ELF; linking does not execute the ELF."""

import kernel_library  # noqa: E402,F401
from atlas_tooling import Admission, sha, write_json, run_command

import argparse
import json
import os
import re
import sys
from pathlib import Path
import struct
import subprocess
from atlas_host_build import START, LINK, TEMPLATES, TEMPLATE_BYTES, audit_elf, validate_graph_program
from atlas_host_layout import ATLAS_DRAM_BASE, HEADER as LAYOUT_HEADER, HEADER_SHA256
from atlas_path_guard import checked
from chipyard import result_transport

for directory in ("chipyard", "host_iss"):
    sys.path.insert(0, str(Path(__file__).absolute().parent / directory))
import arena
from firesim_target import package as package_firesim

HOST = r"""

#include "assets.h"
#include "atlas_result_transport.h"
#include <stddef.h>
#include <stdint.h>
__attribute__((section(".tohost"), aligned(64))) volatile uint64_t tohost;
__attribute__((section(".tohost"), aligned(64))) volatile uint64_t fromhost;
typedef struct {
  uint32_t magic, version, status, graph_result;
  uint32_t completed, failing_index, device_result, runtime_result, first_bad_word, last_sequence;
  uint32_t sequence[54], cycles[54];
  int activation_scale[54], rescale_exponent[54];
  uint64_t trap_mcause, trap_mepc, trap_mtval;
} atlas_graph_mailbox;
__attribute__((section(".handoff"), aligned(64))) volatile atlas_graph_mailbox atlas_workload_mailbox;
__attribute__((aligned(64))) atlas_graph_device_context atlas_resnet_state;
static atlas_host host;
static atlas_result result_report;
static atlas_result_uart result_uart;
__attribute__((aligned(64))) static uint32_t program_buffer[ATLAS_GRAPH_PROGRAM_WORDS];
__attribute__((aligned(64))) static float logits[1000];
int atlas_resnet_graph_run(const float *, float *, void *,
                           int (*)(void *, unsigned, const uint8_t *, size_t, int, uint8_t *, size_t));
void *memset(void *p, int value, size_t n) {
  unsigned char *b = p;
  for (size_t i = 0; i < n; ++i)
    b[i] = (unsigned char)value;
  return p;
}
void *memcpy(void *p, const void *q, size_t n) {
  unsigned char *a = p;
  const unsigned char *b = q;
  for (size_t i = 0; i < n; ++i)
    a[i] = b[i];
  return p;
}
static void finish(uint32_t result) {
  if (!result && !atlas_result_pass(&result_report)) result = 3;
  if (result && !result_report.failures) atlas_result_fail(&result_report, "host-result", result, 0);
  atlas_workload_mailbox.status = result;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  tohost = ((uint64_t)result << 1) | 1;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  for (;;)
    __asm__ volatile("wfi");
}
void atlas_block_trap_report(uint64_t cause, uint64_t pc, uint64_t value) {
  if (!result_report.putchar) {
    atlas_result_uart_init(&result_uart);
    atlas_result_begin(&result_report, atlas_result_uart_putchar, &result_uart, "resnet50-full", 54);
  }
  atlas_result_fail(&result_report, "trap-mcause", cause, 0);
  atlas_workload_mailbox.magic = 0x41544c53;
  atlas_workload_mailbox.trap_mcause = cause;
  atlas_workload_mailbox.trap_mepc = pc;
  atlas_workload_mailbox.trap_mtval = value;
  finish(0xff);
}
static int launch(void *context, unsigned index, const uint8_t *a, size_t a_bytes, int scale,
                  uint8_t *out, size_t out_bytes) {
  int result = atlas_resnet_graph_device(context, index, a, a_bytes, scale, out, out_bytes);
  if (!result) atlas_result_progress(&result_report, atlas_resnet_state.completed, "contraction");
  else atlas_result_fail(&result_report, "device-launch", result, 0);
  return result;
}
void atlas_workload_main(void) {
  atlas_result_uart_init(&result_uart);
  atlas_result_begin(&result_report, atlas_result_uart_putchar, &result_uart, "resnet50-full", 54);
  atlas_workload_mailbox.magic = 0x41544c53;
  atlas_workload_mailbox.version = 3;
  atlas_workload_mailbox.status = UINT32_MAX;
  atlas_workload_mailbox.first_bad_word = UINT32_MAX;
  atlas_resnet_state = (atlas_graph_device_context){.host = &host,
                                                    .kernels = graph_kernels,
                                                    .rescale_words = graph_rescale,
                                                    .dram = (uint8_t *)(uintptr_t)ATLAS_DRAM_BASE,
                                                    .dram_capacity = ATLAS_GRAPH_DRAM_BYTES,
                                                    .max_polls = ATLAS_GRAPH_MAX_POLLS,
                                                    .program_buffer = program_buffer,
                                                    .program_capacity_words = ATLAS_GRAPH_PROGRAM_WORDS};
  atlas_host_result r = atlas_host_init(&host, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_BYTES},
                                        (atlas_host_mmio){atlas_host_direct_read32, atlas_host_direct_write32, 0},
                                        (atlas_host_memory){1, 0, 0});
  if (r != ATLAS_HOST_OK) {
    atlas_workload_mailbox.runtime_result = r;
    atlas_result_fail(&result_report, "host-init", r, ATLAS_HOST_OK);
    finish(0xfe);
  }
  int result = atlas_resnet_graph_run(graph_entry, logits, &atlas_resnet_state, launch);
  atlas_workload_mailbox.graph_result = (uint32_t)result;
  atlas_workload_mailbox.completed = atlas_resnet_state.completed;
  atlas_workload_mailbox.failing_index = atlas_resnet_state.failing_index;
  atlas_workload_mailbox.device_result = atlas_resnet_state.result;
  atlas_workload_mailbox.runtime_result = atlas_resnet_state.runtime_result;
  atlas_workload_mailbox.last_sequence = host.invocation_sequence;
  for (unsigned i = 0; i < 54; ++i) {
    atlas_workload_mailbox.sequence[i] = atlas_resnet_state.sequence[i];
    atlas_workload_mailbox.cycles[i] = atlas_resnet_state.cycles[i];
    atlas_workload_mailbox.activation_scale[i] = atlas_resnet_state.activation_scale[i];
    atlas_workload_mailbox.rescale_exponent[i] = atlas_resnet_state.rescale_exponent[i];
  }
  if (result || atlas_resnet_state.completed != 54) {
    if (result) atlas_result_fail(&result_report, "graph-result", result, 0);
    atlas_result_check(&result_report, "launch-count", atlas_resnet_state.completed, 54);
    finish(1);
  }
  for (unsigned i = 0; i < 1000; ++i) {
    union {
      float f;
      uint32_t u;
    } v = {.f = logits[i]};
    if (v.u != graph_expected[i]) {
      atlas_workload_mailbox.first_bad_word = i;
      atlas_result_fail(&result_report, "logit-word", v.u, graph_expected[i]);
      finish(2);
    }
  }
  finish(0);
}
"""


def relocate_program(raw, patch_word, arena_base):
    """Relocate DMA operands and track the one runtime rescale patch."""
    validate_graph_program(raw)
    if type(patch_word) is not int or not 0 <= patch_word < len(raw) // 4:
        raise ValueError("rescale patch outside raw program")
    offset = patch_word * 4
    word = struct.unpack_from("<I", raw, offset)[0]
    if word & 0x7fffff != 0x25f:
        raise ValueError("rescale patch must be VLI_ALL m4")
    relocated = arena.relocate(raw, ATLAS_DRAM_BASE, arena_base)
    validate_graph_program(relocated)
    if arena_base == ATLAS_DRAM_BASE:
        return relocated, patch_word
    # A second exponent follows exactly the same audited instruction mapping.
    probe = bytearray(raw)
    struct.pack_into("<I", probe, offset, word ^ (1 << 23))
    changed = arena.relocate(bytes(probe), ATLAS_DRAM_BASE, arena_base)
    indices = [i // 4 for i in range(0, len(relocated), 4) if relocated[i:i + 4] != changed[i:i + 4]]
    if len(changed) != len(relocated) or len(indices) != 1:
        raise ValueError("rescale relocation is not unique")
    patch_word = indices[0]
    if struct.unpack_from("<I", relocated, patch_word * 4)[0] != word:
        raise ValueError("relocation changed rescale instruction")
    return relocated, patch_word


def placed_layout(raw, arena_base):
    text, count = re.subn(r"^#define ATLAS_DRAM_BASE 0x[0-9a-fA-F]+$",
        f"#define ATLAS_DRAM_BASE {hex(arena_base)}", raw.decode(), flags=re.M)
    if count != 1:
        raise ValueError("runtime DMA arena constant is not unique")
    return text.encode()


def placed_link(dram_bytes, arena_base):
    arena.window(arena_base, dram_bytes)
    text = LINK.replace("@DRAM_BYTES@", str(dram_bytes))
    text = re.sub(r"(\.atlas_dram )0x[0-9a-fA-F]+", lambda m: m[1] + hex(arena_base), text)
    return re.sub(r"(__atlas_dram_end <= )0x[0-9a-fA-F]+",
        lambda m: m[1] + hex(((arena_base >> 32) + 1) << 32), text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generated", "replay-gate", "reference", "templates", "rescale-table", "atlas-root", "clang", "lld", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-polls", type=int, default=100000000)
    parser.add_argument("--skip-bss-clear", action="store_true",
                        help="omit the startup BSS clear; only valid when the loader zero-fills memory (Chipyard LOADMEM)")
    parser.add_argument("--target", choices=("sim", "firesim"), default="sim")
    arena.add_arguments(parser)
    result_transport.add_arguments(parser)
    args = parser.parse_args()
    arena_base = ATLAS_DRAM_BASE if args.arena_base is None else args.arena_base
    for value in vars(args).values():
        if isinstance(value, Path):
            checked(value)
    if args.out.exists() or not 0 < args.max_polls <= 100000000:
        raise ValueError("fresh output/bounded polls required")
    if any(c in str(args.out) for c in ('"', "\\", "\n", "\r")):
        raise ValueError("unsafe assembly path")
    ad = Admission()
    ad.read(LAYOUT_HEADER, HEADER_SHA256)
    gate = json.loads(ad.read(args.replay_gate))
    if (gate.get("status") != "PASS" or not gate.get("source_bookends_equal")
        or gate.get("all_54_device_inputs_and_scales_exact") is not True
        or gate.get("final_1000_logit_words_exact") is not True):
        raise ValueError("full 54 native graph gate required")
    for path, digest in gate["inputs_sha256"].items():
        ad.read(path, digest)
    graphpath = args.generated / "graph.json"
    if str(graphpath) not in gate["inputs_sha256"]:
        raise ValueError("gate is not bound to selected graph")
    graph = json.loads(ad.read(graphpath, gate["inputs_sha256"][str(graphpath)]))
    for path, digest in graph["input_sha256"].items():
        ad.read(path, digest)
    for name, digest in graph["generated_sha256"].items():
        ad.read(args.generated / name, digest)
    for name, digest in graph["params_sha256"].items():
        ad.read(args.generated / name, digest)
    template = json.loads(ad.read(args.templates, graph["input_sha256"].get(str(args.templates))))
    if str(args.templates) not in graph["input_sha256"] or len(template["kernels"]) != 54:
        raise ValueError("template identity differs")
    if graph.get("geometry_admission") != (
        "all operators/shapes/arity/topology/54 contractions/groups 1 proven before generation"):
        raise ValueError("unvalidated geometry")
    reference = json.loads(ad.read(args.reference / "receipt.json",
        gate["inputs_sha256"].get(str(args.reference / "receipt.json"))))
    if str(args.reference / "receipt.json") not in gate["inputs_sha256"]:
        raise ValueError("reference differs")
    assets = {"graph_entry": ad.read(reference["nodes"][0]["payload"]["path"], reference["nodes"][0]["payload"]["sha256"]),
        "graph_expected": ad.read(args.reference / "target-logits.f32.bin", reference["final_logits"]["sha256"]),
        "graph_rescale": ad.read(args.rescale_table, template["rescale_table_sha256"])}
    if len(assets["graph_entry"]) != 3 * 224 * 224 * 4 or len(assets["graph_expected"]) != 4000:
        raise ValueError("entry/final extent differs")
    words = struct.unpack("<254I", assets["graph_rescale"])
    if list(words) != [((e + 127) << 23) | 0x25F for e in range(-126, 128)]:
        raise ValueError("rescale table is not qualified VLI_ALL m4")
    rows = []
    for i, (k, t) in enumerate(zip(graph["kernels"], template["kernels"], strict=True)):
        if k["template"] != t or k["kernel_index"] != i or t["base_rescale_exponent"] != 0:
            raise ValueError("template graph mismatch")
        raw = ad.read(t["program"]["path"], t["program"]["sha256"])
        validate_graph_program(raw, t["program_words"])
        if struct.unpack_from("<I", raw, t["rescale_patch_word_index"] * 4)[0] != words[126]:
            raise ValueError("base patch not canonical")
        weight = ad.read(t["weight"]["fp8_tiled"]["path"], t["weight"]["fp8_tiled"]["sha256"])
        mp, kp, np = t["padded_mkn"]
        if len(weight) != kp * np:
            raise ValueError("weight extent differs")
        raw, patch_word = relocate_program(raw, t["rescale_patch_word_index"], arena_base)
        t = dict(t, program_words=len(raw) // 4, rescale_patch_word_index=patch_word, dram_base=arena_base)
        assets["graph_program_" + str(i)] = raw
        assets["graph_weight_" + str(i)] = weight
        rows.append(t)
    sources = ["atlas_host.c", "atlas_host.h", "atlas_host_layout.h", "atlas_resnet_graph_math.c", "atlas_numeric.h",
        "atlas_resnet_graph_device.c", "atlas_resnet_graph_device.h", "atlas_result.h"]
    mathpath = args.atlas_root / "runtime/atlas_resnet_graph_math.c"
    if str(mathpath) not in gate["inputs_sha256"]:
        raise ValueError("math gate must bind current exact math source")
    for path in [args.clang, args.lld, Path(__file__).absolute(), Path(checked.__code__.co_filename).absolute(),
        Path(Admission.read.__code__.co_filename).absolute(), Path(audit_elf.__code__.co_filename).absolute(),
        Path(arena.__file__).absolute(), Path(arena.Image.__init__.__code__.co_filename).absolute(),
        Path(package_firesim.__code__.co_filename).absolute(), Path(result_transport.__file__).absolute()
        ] + [args.atlas_root / "runtime" / name for name in sources]:
        ad.read(path)
    for name, data in TEMPLATE_BYTES.items():
        ad.read(TEMPLATES / name, sha(data))
    dram = max(t["total_dram_bytes"] for t in rows)
    link = placed_link(dram, arena_base)
    if args.dram_base < 0 or args.dram_bytes <= 0 or args.dram_base + args.dram_bytes > 1 << 64:
        raise ValueError("invalid physical DRAM range")
    if not args.dram_base <= arena_base <= args.dram_base + args.dram_bytes - dram:
        raise ValueError("DMA arena outside physical DRAM range")
    args.out.mkdir(parents=True)
    uart = result_transport.write_config(args.out, args)
    snapshot = args.out / "runtime"
    snapshot.mkdir()
    for name in sources:
        data = ad.read(args.atlas_root / "runtime" / name)
        (snapshot / name).write_bytes(placed_layout(data, arena_base) if name == "atlas_host_layout.h" else data)
    # Copy graph/parameters into a self-contained bundle and rewrite incbin paths.
    graphdir = args.out / "graph"
    graphdir.mkdir()
    for name, digest in graph["params_sha256"].items():
        data = ad.read(args.generated / name, digest)
        (graphdir / name).write_bytes(data)
        node = name.removeprefix("params-").removesuffix(".bin")
        assets["params_" + node] = data
    (graphdir / "graph.c").write_bytes(ad.read(args.generated / "graph.c"))
    constants = ad.read(args.generated / "constants.S").decode().replace(str(args.generated), str(graphdir))
    (graphdir / "constants.S").write_text(constants)
    (args.out / "start.S").write_text(START)
    (args.out / "link.ld").write_text(link)
    (args.out / "host.c").write_text(HOST)
    assembly = ['.section .rodata,"a",@progbits']
    definitions = ['#include "assets.h"']
    declarations = ['#include "atlas_resnet_graph_device.h"', f"#define ATLAS_GRAPH_DRAM_BYTES {dram}",
        f"#define ATLAS_GRAPH_MAX_POLLS {args.max_polls}"]
    for label, data in assets.items():
        if label.startswith("params_"):
            continue
        filename = args.out / (label + ".bin")
        filename.write_bytes(data)
        assembly += [".balign 64", f".global {label}", label + ":", f'.incbin "{filename}"', f".size {label}, .-{label}"]
        kind = "float" if label == "graph_entry" else "uint8_t" if label.startswith("graph_weight_") else "uint32_t"
        declarations.append(f"extern const {kind} {label}[];")
    definitions += ["const atlas_graph_kernel graph_kernels[54]={"]
    for i, t in enumerate(rows):
        m, k, n = t["padded_mkn"]
        definitions.append("{"
            f'.program=graph_program_{i},.weights=graph_weight_{i},.program_words={t["program_words"]},'
            f'.patch_word={t["rescale_patch_word_index"]},.activation_bytes={m*k},.weight_bytes={k*n},'
            f'.output_bytes={t["output_bytes"]},.output_offset={t["output_offset"]},'
            f'.dram_bytes={t["total_dram_bytes"]},.weight_scale={t["b_scale_exponent"]}' + "},")
    definitions += ["};"]
    declarations += ["extern const atlas_graph_kernel graph_kernels[54];"]
    (args.out / "images.S").write_text("\n".join(assembly) + "\n")
    (args.out / "assets.h").write_text("\n".join(declarations) + "\n")
    (args.out / "assets.c").write_text("\n".join(definitions) + "\n")
    generated = ([args.out / n for n in ("start.S", "link.ld", "host.c", "images.S", "assets.h",
        "assets.c", "atlas_result_transport.h", "result-transport.json")] + [snapshot / n for n in sources]
        + [graphdir / "graph.c",
        graphdir / "constants.S"] + [graphdir / n for n in graph["params_sha256"]]
        + [args.out / (label + ".bin") for label in assets if not label.startswith("params_")])
    sourcepins = {str(path): sha(path.read_bytes()) for path in generated}
    skip_bss = ["-DATLAS_SKIP_BSS_CLEAR"] if args.skip_bss_clear else []
    command = ([str(args.clang), "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany",
        "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-fno-fast-math", "-ffreestanding",
        "-fno-builtin", "-fno-stack-protector", "-fno-pic", "-fno-vectorize", "-fno-slp-vectorize", "-nostdlib",
        "--ld-path=" + str(args.lld), "-I" + str(snapshot),
        "-I" + str(args.out)] + skip_bss + [str(args.out / n) for n in ("start.S", "host.c", "assets.c",
        "images.S")] + [str(graphdir / n) for n in ("graph.c", "constants.S")] + [str(snapshot / n) for n in ("atlas_host.c",
        "atlas_resnet_graph_math.c", "atlas_resnet_graph_device.c")] + ["-Wl,-T," + str(args.out / "link.ld"),
        "-Wl,--no-relax", "-Wl,-Map," + str(args.out / "host.map"), "-o", str(args.out / "host.elf")])
    result = run_command(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=dict(os.environ, TMPDIR=str(args.out)))
    (args.out / "build.log").write_text(result.stdout)
    bookends = ad.recheck() and all(sha(checked(Path(path)).read_bytes()) == digest for path, digest in sourcepins.items())
    report = {"scope": "Complete 54-kernel live C host graph ELF linked; no target execution", "command": command,
        "returncode": result.returncode or int(not bookends), "source_bookends_equal": bookends, "inputs_sha256": ad.inputs,
        "generated_inputs_sha256": sourcepins, "host_elf_executed": False, "full_soc_executed": False,
        "firmpolicy": (
        "FS enabled, FRM RNE, FP32 separate multiply/add; sequential average "
        "policy; no cohort quality equivalence claim"), "kernel_count": 54, "captured_successor_A_embedded": False,
        "dram_bytes": dram, "arena_base": arena_base,
        "placement": {"arena_base": arena_base, "dram_base": args.dram_base, "dram_bytes": args.dram_bytes},
        "physical_dram": {"base": args.dram_base, "bytes": args.dram_bytes}, "target": args.target,
        "uart": uart, "result_name": "resnet50-full", "result_progress_launches": list(range(1, 55))}
    if not report["returncode"]:
        raw = (args.out / "host.elf").read_bytes()
        report["elf_sha256"] = sha(raw)
        report["elf_audit"] = audit_elf(raw, assets, dram, arena_base=arena_base)
        report["physical_audit"] = arena.audit(args.out, args.dram_base, args.dram_bytes)
        if args.target == "firesim":
            report["firesim_contract"] = package_firesim(args.out, "atlas-resnet50-full",
                args.dram_base, args.dram_bytes, uart=uart)
    write_json(args.out / "build.json", report)
    print(json.dumps({key: report[key] for key in ("returncode", "source_bookends_equal", "kernel_count")}))
    raise SystemExit(report["returncode"])


if __name__ == "__main__":
    main()
