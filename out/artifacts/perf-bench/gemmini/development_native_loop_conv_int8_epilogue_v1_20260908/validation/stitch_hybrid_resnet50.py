#!/usr/bin/env python3
"""Link/run a ResNet hybrid: Merlin's 53 conv kernels, Jack TVM graph orchestration.

No FPGA or queue action is performed.  The TVM-generated host functions remain responsible for
residual rescaling/add/ReLU, pooling orchestration, global average pool, and dense.  Each original
``tiled_conv_auto`` body is replaced by one uniquely named Merlin-emitted LOOP_CONV kernel.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
ARTIFACT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).parent), str(ARTIFACT / "compiler"),
                str(ROOT / "merlin" / "python")]
os.environ["MERLIN_TARGET_PATH"] = str(ARTIFACT / "runtime_target")

from build_jack_native_aligned_conv_set import REFERENCE_C, TVM, balanced_call, calls
from merlin.runtime.backends import base


PROJECT = TVM / "project/src"
BUILD = ARTIFACT / "validation/native_aligned_resnet50_hybrid"
OBJECTS = ARTIFACT / "validation/native_aligned_resnet50_layer_build"
REFERENCE = TVM / "host_reference.i8.bin"
OBJCOPY = ROOT / "third_party/llvm-install/bin/llvm-objcopy"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError("command failed:\n" + " ".join(command) + "\n" + proc.stdout)


def modified_tvm_conv_source() -> tuple[str, list[dict]]:
    text = REFERENCE_C.read_text()
    replacements = []
    records = []
    for match in re.finditer(r"\btiled_conv_auto\s*\(", text):
        payload, end = balanced_call(text, match.start())
        # Reuse the converter's nesting-aware argument splitter through its parsed record.
        # Match by the enclosing function name, which is TVM's conv2d_N topological index.
        name_at = text.rfind("TVM_DLL int32_t ", 0, match.start())
        name = re.match(
            r"TVM_DLL int32_t (tvmgen_default_fused_contrib_gemmini_conv2d(?:_(\d+))?)\(",
            text[name_at:])
        if name is None:
            raise ValueError("could not bind TVM convolution call")
        index = int(name.group(2) or 0)
        from build_jack_native_aligned_conv_set import split_args
        args = split_args(payload)
        semi = text.index(";", end) + 1
        if index == 0:
            replacement = f'''/* Merlin LOOP_CONV conv2d_0; compact-C input is padded once here. */
  for (int merlin_p = 0; merlin_p < 224 * 224; ++merlin_p) {{
    for (int merlin_c = 0; merlin_c < 16; ++merlin_c)
      merlin_conv0_input[merlin_p * 16 + merlin_c] =
          merlin_c < 3 ? {args[17]}[merlin_p * 3 + merlin_c] : 0;
  }}
  merlin_conv_0((void*)merlin_conv0_input, (void*){args[18]},
                (void*){args[19]}, (void*)merlin_conv0_unpooled);
  /* TVM's first call fuses 3x3/stride-2/pad-1 maxpool.  Keep that exact graph
     operation runner-owned until Merlin's LOOP_CONV tiler supports pooled tiles. */
  for (int merlin_oy = 0; merlin_oy < 56; ++merlin_oy)
    for (int merlin_ox = 0; merlin_ox < 56; ++merlin_ox)
      for (int merlin_c = 0; merlin_c < 64; ++merlin_c) {{
        int8_t merlin_best = 0;
        for (int merlin_ky = 0; merlin_ky < 3; ++merlin_ky)
          for (int merlin_kx = 0; merlin_kx < 3; ++merlin_kx) {{
            int merlin_iy = merlin_oy * 2 + merlin_ky - 1;
            int merlin_ix = merlin_ox * 2 + merlin_kx - 1;
            if (merlin_iy >= 0 && merlin_iy < 112 && merlin_ix >= 0 && merlin_ix < 112) {{
              int8_t merlin_v = merlin_conv0_unpooled[
                  (merlin_iy * 112 + merlin_ix) * 64 + merlin_c];
              if (merlin_v > merlin_best) merlin_best = merlin_v;
            }}
          }}
        {args[20]}[(merlin_oy * 56 + merlin_ox) * 64 + merlin_c] = merlin_best;
      }}'''
            split = "host_maxpool_and_input_channel_padding"
        else:
            replacement = (f"merlin_conv_{index}((void*){args[17]}, (void*){args[18]}, "
                           f"(void*){args[19]}, (void*){args[20]});")
            split = None
        replacements.append((match.start(), semi, replacement))
        records.append({"index": index, "replaced": True, "runner_owned_split": split})
    for begin, end, replacement in reversed(replacements):
        text = text[:begin] + replacement + text[end:]
    declarations = "\n".join(
        f"extern void merlin_conv_{i}(void*, void*, void*, void*);" for i in range(53))
    declarations += "\nstatic elem_t merlin_conv0_input[224 * 224 * 16] row_align(1);\n"
    declarations += "static elem_t merlin_conv0_unpooled[112 * 112 * 64] row_align(1);\n"
    anchor = '#include "gemmini_testutils.h"\n'
    text = text.replace(anchor, anchor + declarations + "\n", 1)
    runtime = PROJECT / "standalone_crt/include/tvm/runtime"
    text = text.replace('#include "../../src/standalone_crt/include/tvm/runtime/c_runtime_api.h"',
                        f'#include "{runtime / "c_runtime_api.h"}"')
    text = text.replace('#include "../../src/standalone_crt/include/tvm/runtime/c_backend_api.h"',
                        f'#include "{runtime / "c_backend_api.h"}"')
    return text, sorted(records, key=lambda row: row["index"])


def warm_measured_driver_source() -> str:
    text = (PROJECT / "dense.c").read_text()
    text = text.replace("tvm-current-gemmini-18801f4eab0cd2e8",
                        "merlin53-loopconv-tvmhost-warm-v1")
    anchor = "  uint64_t cycle_start;\n"
    warm = '''  /* One complete untimed warm invocation; then restore the immutable input. */
  if (tvmgen_default_run(&inputs, &outputs) != 0) return 3;
  for (int i = 0; i < 150528; ++i)
    tvmgen_default_input0[i] = (int8_t)universal_input_begin[i];
'''
    if text.count(anchor) != 1:
        raise ValueError("could not locate measurement boundary in Jack driver")
    return text.replace(anchor, warm + anchor)


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    # Rebuild all objects after any compiler edit, then rename their public entry symbols.
    subprocess.run([sys.executable, str(Path(__file__).parent /
                    "build_jack_native_aligned_conv_set.py")], check=True)
    source, replacements = modified_tvm_conv_source()
    modified = BUILD / "default_lib1.merlin_hybrid.c"
    modified.write_text(source)
    driver = BUILD / "dense.merlin_hybrid.c"
    driver.write_text(warm_measured_driver_source())
    if source.count("tiled_conv_auto(") != 0 or len(replacements) != 53:
        raise RuntimeError("not all 53 TVM convolution calls were replaced")

    renamed = []
    for index in range(53):
        src = OBJECTS / f"conv2d_{index:02d}/kernel.o"
        dst = BUILD / f"merlin_conv_{index:02d}.o"
        shutil.copyfile(src, dst)
        run([str(OBJCOPY), "--redefine-sym", f"gemmini_kernel=merlin_conv_{index}", str(dst)])
        renamed.append(dst)

    backend = base.get_backend("gemmini")
    recipe = base.harness_build_recipe("gemmini")
    gcc = str(recipe.compiler)
    include = [f"-I{path}" for path in (
        PROJECT, PROJECT / "include", PROJECT / "../include/tvm", *recipe.include_roots)]
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany", "-std=gnu99",
             "-O2", "-ffast-math", "-fno-common", "-fno-builtin", "-fno-builtin-printf",
             "-march=rv64gc", "-Wa,-march=rv64gc", "-DID_STRING=", "-DPRINT_TILE=0",
             "-DBAREMETAL=1", *include]
    sources = {
        "dense.o": driver,
        "default_lib0.o": PROJECT / "model/default_lib0.c",
        "default_lib1.o": modified,
        "syscalls.o": recipe.support_sources[0],
        "crt.o": recipe.support_sources[1],
    }
    for name, source_path in sources.items():
        run([gcc, *flags, "-c", str(source_path), "-o", str(BUILD / name)])
    elf = BUILD / "resnet50_merlin53_tvmhost.elf"
    # Match Jack's TVM project link: it supplies its own CRT but retains newlib,
    # whose libm square-root path owns __errno.
    run([gcc, "-nostartfiles", "-static", "-mcmodel=medany", "-march=rv64gc",
         "-T", str(recipe.link_script),
         *(str(BUILD / name) for name in sources), *(str(path) for path in renamed),
         "-lm", "-lgcc", "-o", str(elf)])

    console = backend.run_elf(elf, simulator="spike", timeout=900)
    (BUILD / "spike.log").write_text(console)
    # Jack's frozen harness uses ``OUT Y0 i8 1000 ...`` while Merlin's generic
    # parser expects ``OUT name rows cols ...``.  Parse the declared legacy ABI
    # directly; the value count is checked below.
    out_line = next(line for line in console.splitlines() if line.startswith("OUT Y0 i8 "))
    words = out_line.split()
    declared_outputs = int(words[3])
    actual = np.asarray([int(value) for value in words[4:]], dtype=np.int8)
    if actual.size != declared_outputs:
        raise RuntimeError(f"OUT declares {declared_outputs} values but printed {actual.size}")
    metrics = {}
    for line in console.splitlines():
        if line.startswith("METRIC "):
            _, name, value = line.split()
            metrics[name] = int(value)
    expected = np.fromfile(REFERENCE, dtype=np.int8)
    mismatches = int(np.count_nonzero(actual != expected))
    hybrid_cycles = int(metrics["cycles"])
    tvm_spike_cycles = 338_090_625
    q534_firesim_cycles = 1_537_416_019
    receipt = {
        "schema": "merlin53_loopconv_tvm_host_hybrid_spike_v1",
        "status": "passed" if mismatches == 0 and actual.size == 1000 else "failed",
        "qualification": "local_spike_functional_not_firesim_or_fpga",
        "measurement_protocol": "one_complete_untimed_warm_then_one_complete_measured_inference",
        "merlin_compiled": {"conv2d": 53, "loop_conv_descriptors": 3550,
                            "load3_bias_descriptors": 2316},
        "tvm_runner_owned": {"residual_rescale_add_relu": 20, "first_maxpool": 1,
                             "global_avgpool_flatten": 1, "dense": 1,
                             "activation_arena_and_call_graph": 1,
                             "first_input_channel_padding": 1},
        "convolution_replacements": replacements,
        "logits": int(actual.size), "mismatches": mismatches,
        "max_abs_diff_i8": int(np.max(np.abs(
            actual.astype(np.int16) - expected.astype(np.int16)))),
        "nonfinite": 0, "top1": int(np.argmax(actual)),
        "reference_top1": int(np.argmax(expected)),
        "reference_sha256": sha(REFERENCE), "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "elf": str(elf), "elf_sha256": sha(elf), "metrics": metrics,
        "comparisons": {
            "jack_tvm_local_spike": {
                "cycles": tvm_spike_cycles,
                "evidence": str(TVM.parent / "spike_gate/stdout.log"),
                "hybrid_over_tvm": hybrid_cycles / tvm_spike_cycles,
                "hybrid_percent_slower":
                    (hybrid_cycles / tvm_spike_cycles - 1.0) * 100.0,
                "comparable": False,
                "why_not_comparable": (
                    "same local Spike oracle, but Jack TVM is a cold single run while "
                    "the hybrid window follows one complete untimed warm inference"
                ),
            },
            "merlin_q534_firesim": {
                "cycles": q534_firesim_cycles,
                "evidence": str(ROOT / "out/artifacts/perf-bench/gemmini/"
                                "q534_whole_model_headroom_roofline_20260908/receipt.json"),
                "q534_over_hybrid_arithmetic_ratio": q534_firesim_cycles / hybrid_cycles,
                "comparable": False,
                "why_not_comparable": "q534 is FireSim; hybrid is functional local Spike",
            },
        },
        "firesim": "not_run",
    }
    (BUILD / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "status", "qualification", "logits", "mismatches", "top1", "reference_top1",
        "metrics")}, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
