from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from merlin.analysis.perf import functional_gate as FG

PASS_TRANSCRIPT = "MERLIN_MODEL resnet50_w8a8_torchao_pt2e\nMERLIN_COMPILER phase2_reduction_resident_kernel\nMERLIN_INPUT universal_resnet50_dog_sha256_92994d4d2219bc51\nMERLIN_INVOCATIONS warmup=1 measured=1 batch=1\nMERLIN_INPUT_PROLOGUE timed=1 owner=phase2_reduction_resident_kernel kind=static_pt2e_w8a8\nMERLIN_PROFILE warmup begin\nMERLIN_PROFILE warmup end rc=0\nMERLIN_WARM_RESET timed=0 kind=zero_mutable_arena bytes=90083520\nMERLIN_PROFILE measured begin\nMERLIN_METRIC cycles=691647813\nMERLIN_METRIC instret=691647817\nMERLIN_METRIC main_ld_cycles=5841166\nMERLIN_METRIC main_st_cycles=5862797\nMERLIN_METRIC main_ex_cycles=5813796\nMERLIN_METRIC exe_active_cycle=5830844\nMERLIN_METRIC loop_matmul_active_cycles=5836613\nMERLIN_METRIC rdma_bytes_rec=13\nMERLIN_METRIC wdma_bytes_sent=11\nMERLIN_METRIC reservation_station_active_cycles=5860767\nMERLIN_RESULT checksum_fnv1a64=c6e777c3fe0aae90\nMERLIN_RESULT max_abs_bits=00000000 max_rel_bits=00000000 qdq_max_abs_bits=3e31e2aa\nMERLIN_RESULT logits_checked=1000 bad=0 nonfinite=0 top1=258 expected_top1=258\nMERLIN_PROFILE measured end rc=0\nPASS: warm-then-measured Merlin W8A8 ResNet-50 and all-logit check\nAccelerator extension configured with:\n    dim = 16\n"
GATE = FG.FunctionalGateSpec(
    {"checksum_fnv1a64": "c6e777c3fe0aae90", "max_abs_bits": "00000000", "bad": 0, "top1": 258}
)


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\nset -u\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_bundle(tmp_path: Path, *, link_exit: int = 0, elf_count: int = 1) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "payload").mkdir(parents=True)
    (bundle / "payload" / "single_run_harness.c").write_text("int main(void){return 0;}\n")
    (bundle / "runtime" / "include").mkdir(parents=True)
    (bundle / "runtime" / "crt.S").write_text("\n")
    _script(
        bundle / "build_elf.sh",
        f"""here=$(cd "$(dirname "$0")" && pwd -P)\n[[ -f "$here/compiler/kernel.o" ]] || {{ echo "missing required file: kernel.o" >&2; exit 2; }}\n[[ -f "$here/payload/single_run_harness.c" ]] || exit 2\necho "link args: $*" > "$here/link_args.txt"\nfor i in $(seq 1 {elf_count}); do printf 'ELF%s' "$i" > "$here/model_$i.elf"; done\nexit {link_exit}\n""",
    )
    return bundle


def _fake_toolchain(
    tmp_path: Path, *, transcript: str, translate_exit: int = 0, clang_exit: int = 0, simulator_body: str | None = None
) -> FG.FunctionalGateToolchain:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "transcript.txt").write_text(transcript, encoding="utf-8")
    _script(
        tools / "mlir-translate", f'[[ "$1" == "--mlir-to-llvmir" ]] || exit 9\ncp "$2" "$4"\nexit {translate_exit}\n'
    )
    _script(
        tools / "clang",
        f'out=""; while [[ $# -gt 0 ]]; do if [[ "$1" == "-o" ]]; then out="$2"; shift; fi; shift; done\n[[ -n "$out" ]] && echo "obj" > "$out"\nexit {clang_exit}\n',
    )
    _script(
        tools / "spike",
        simulator_body
        or f"""printf '%s' "$*" > "$(dirname "$0")/spike_args.txt"\ncat "$(dirname "$0")/transcript.txt"\nexit 0\n""",
    )
    return FG.FunctionalGateToolchain(
        mlir_translate=tools / "mlir-translate",
        clang=tools / "clang",
        simulator=tools / "spike",
        clang_target="riscv64-unknown-elf",
        march="rv64gc",
        mabi="lp64d",
        simulator_isa="rv64gc_zicntr",
        simulator_extension="accel",
        link_script_args=("/opt/sdk",),
        env={"MERLIN_FAKE": "1"},
    )


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "FG",
    "GATE",
    "PASS_TRANSCRIPT",
    "Path",
    "_fake_bundle",
    "_fake_toolchain",
    "_script",
    "json",
    "pytest",
    "stat",
]
