#!/usr/bin/env python3
"""Per-chunk QNN context-binary compiler.

Drives the QNN SDK toolchain to produce a `.qnn-ctx` per (chunk, backend)
that the IREE QNN HAL driver can load via QnnContext_createFromBinary.

Pipeline per (chunk, backend):

    chunk MLIR / ONNX (from chunk_extractor.py + tools/breakdown_vmfb.py)
                │
                ▼
        ONNX subgraph  (extracted via onnx.utils.extract_model when chunk
                        is a slice of the parent model; passed through
                        directly when chunk_manifest.json carries an
                        explicit `onnx_path` field)
                │
                ▼  qairt-converter
        QNN model.cpp + model.bin
                │
                ▼  qnn-model-lib-generator (cross-compile)
        libqnn_model_<chunk>.so   (aarch64-ubuntu-gcc9.4 against board sysroot)
                │
                ▼  qnn-context-binary-generator
        chunk_<id>.<backend>.qnn-ctx

Outputs land at <output_dir>/chunk_<id>.<backend>.qnn-ctx.

CLI:
    tools/compile_qnn.py
        --chunk-manifest eval/qrb5165/dronet/breakdowns/manifest.json
        --chunks 10                   # comma-list or "all"
        --backends qnn-gpu,qnn-hta
        --output-dir build/qnn_chunks
        [--qnn-sdk-root /scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326]
        [--board-sysroot /scratch2/agustin/qrb5165_sysroot]
        [--verbose]

The default qnn-sdk-root + board-sysroot match what was validated this
session at /tmp/qnn_aarch64_build/. Cross-compile flags follow the manual
build in tmp/qnn_iree_integration_plan.md §1 step 4.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

# Locations for the QNN SDK + board sysroot + aarch64 cross-toolchain.
# All overridable via env vars; the defaults below match the validated
# environment on the merlin development host but are not portable across
# users or machines — override before invoking on a different host.
DEFAULT_QNN_SDK_ROOT = os.environ.get("QNN_SDK_ROOT", "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326")
DEFAULT_BOARD_SYSROOT = os.environ.get("QNN_BOARD_SYSROOT", "/scratch2/agustin/qrb5165_sysroot")
DEFAULT_TOOLCHAIN = os.environ.get(
    "QNN_CROSS_TOOLCHAIN",
    "/ecad/tools/xilinx/Vitis/2023.1/gnu/aarch64/lin/aarch64-linux",
)

BACKEND_LIB = {
    "qnn-cpu": "libQnnCpu.so",
    "qnn-gpu": "libQnnGpu.so",
    "qnn-hta": "libQnnHta.so",  # NPU on Snapdragon 865/QRB5165
    "qnn-htp": "libQnnHtp.so",  # 8 Gen 1+ NPU (not on QRB5165)
}


def _run(cmd: list[str], cwd: pathlib.Path | None = None, env: dict | None = None, verbose: bool = False) -> None:
    """Run a subprocess, surface failures with context."""
    if verbose:
        print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=not verbose, text=True)
    if r.returncode != 0:
        if not verbose:
            sys.stderr.write(r.stdout or "")
            sys.stderr.write(r.stderr or "")
        raise RuntimeError(f"command failed (rc={r.returncode}): " f"{' '.join(cmd)}")


def _onnx_for_chunk(chunk_entry: dict, model_onnx: pathlib.Path, out_path: pathlib.Path) -> None:
    """Extract the chunk's ONNX subgraph from the parent model.

    chunk_entry is a manifest.json["dispatches"][name] record. If it has an
    `onnx_path` field (set by chunk_extractor when the chunk has been
    pre-extracted), copy that. Otherwise use onnx.utils.extract_model with
    the chunk's parent_dispatch_ids → ONNX node-name mapping. The mapping
    requires the original model's node naming convention.

    For PR-4 first-cut we support the explicit onnx_path mode; the
    auto-extract path is a follow-on.
    """
    explicit = chunk_entry.get("onnx_path")
    if explicit:
        shutil.copy(pathlib.Path(explicit), out_path)
        return

    # Auto-extract: requires onnx package + parent_dispatch_ids → node names.
    # Implemented as a best-effort copy of the whole parent model when
    # parent_dispatch_ids cover the entire model (e.g. for single-chunk
    # workloads). For partial slices, the user must pre-extract.
    parent_ids = chunk_entry.get("parent_dispatch_ids", [])
    if not parent_ids:
        raise RuntimeError(
            "chunk has no `onnx_path` or `parent_dispatch_ids`; pre-extract "
            "the chunk's ONNX subgraph and add an `onnx_path` field to the "
            "chunk_manifest entry, or supply a single-chunk workload that "
            "covers the whole model"
        )
    shutil.copy(model_onnx, out_path)


def _qairt_convert(
    qnn_sdk: pathlib.Path, onnx_path: pathlib.Path, out_dir: pathlib.Path, verbose: bool
) -> tuple[pathlib.Path, pathlib.Path]:
    """ONNX → QNN model.cpp + model.bin via qairt-converter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = out_dir / "model.cpp"
    bin_path = out_dir / "model.bin"
    cmd = [
        str(qnn_sdk / "bin" / "x86_64-linux-clang" / "qairt-converter"),
        "--input_network",
        str(onnx_path),
        "--output_path",
        str(cpp_path),
    ]
    _run(cmd, verbose=verbose)
    if not cpp_path.exists():
        raise RuntimeError(f"qairt-converter did not produce {cpp_path}")
    return cpp_path, bin_path


def _build_model_so(
    qnn_sdk: pathlib.Path,
    board_sysroot: pathlib.Path,
    toolchain: pathlib.Path,
    cpp_path: pathlib.Path,
    bin_path: pathlib.Path,
    out_so: pathlib.Path,
    verbose: bool,
) -> None:
    """Cross-compile model.cpp + model.bin → model.so for aarch64.

    Mirrors the manual build validated this session at /tmp/qnn_aarch64_build/.
    """
    out_dir = out_so.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract the .bin tar archive (qairt-converter packs raw weights into it).
    obj_dir = out_dir / "obj_binary"
    obj_dir.mkdir(parents=True, exist_ok=True)
    if bin_path.exists():
        _run(["tar", "xf", str(bin_path), "-C", str(obj_dir)], verbose=verbose)
        # objcopy each .raw → .o with path-prefixed symbol naming.
        for raw in sorted(obj_dir.glob("*.raw")):
            o_path = raw.with_suffix(".o")
            _run(
                [
                    str(toolchain / "bin" / "aarch64-linux-gnu-objcopy"),
                    "-I",
                    "binary",
                    "-O",
                    "elf64-littleaarch64",
                    "-B",
                    "aarch64",
                    str(raw),
                    str(o_path),
                ],
                cwd=out_dir,
                verbose=verbose,
            )

    o_files = sorted(str(p) for p in obj_dir.glob("*.o"))

    # Cross-compile.
    gxx = str(toolchain / "bin" / "aarch64-linux-gnu-g++")
    sysroot_inc_cpp9 = board_sysroot / "usr" / "include" / "c++" / "9"
    sysroot_inc_arch_cpp9 = sysroot_inc_cpp9 / "aarch64-linux-gnu"
    sysroot_inc_arch = board_sysroot / "usr" / "include" / "aarch64-linux-gnu"
    sysroot_inc = board_sysroot / "usr" / "include"
    sysroot_lib_arch = board_sysroot / "usr" / "lib" / "aarch64-linux-gnu"
    sysroot_lib_gcc = board_sysroot / "usr" / "lib" / "gcc" / "aarch64-linux-gnu" / "9"
    sysroot_lib = board_sysroot / "lib" / "aarch64-linux-gnu"
    qnn_jni = qnn_sdk / "share" / "QNN" / "converter" / "jni"

    cmd = [
        gxx,
        f"--sysroot={board_sysroot}",
        f"-isystem{sysroot_inc_cpp9}",
        f"-isystem{sysroot_inc_arch_cpp9}",
        f"-isystem{sysroot_inc_arch}",
        f"-isystem{sysroot_inc}",
        f"-I{qnn_sdk / 'include' / 'QNN'}",
        f"-I{qnn_jni}",
        f"-B{sysroot_lib_arch}",
        f"-B{sysroot_lib_gcc}",
        f"-L{sysroot_lib_arch}",
        f"-L{sysroot_lib_gcc}",
        f"-L{sysroot_lib}",
        "-Wl,--dynamic-linker=/lib/ld-linux-aarch64.so.1",
        f"-Wl,-rpath-link={sysroot_lib_arch}:{sysroot_lib}",
        "-O2",
        "-std=c++14",
        "-fPIC",
        "-shared",
        "-w",
        '-DQNN_API=__attribute__((visibility("default")))',
        str(qnn_jni / "QnnModel.cpp"),
        str(qnn_jni / "linux" / "QnnModelPal.cpp"),
        str(qnn_jni / "QnnWrapperUtils.cpp"),
        str(cpp_path),
        *o_files,
        "-o",
        str(out_so),
    ]
    _run(cmd, verbose=verbose)
    if not out_so.exists():
        raise RuntimeError(f"cross-build did not produce {out_so}")


def _qnn_context_binary(
    qnn_sdk: pathlib.Path, model_so: pathlib.Path, backend: str, out_qnn_ctx: pathlib.Path, verbose: bool
) -> None:
    """model.so + libQnn{Backend}.so → .qnn-ctx via qnn-context-binary-generator."""
    backend_lib = BACKEND_LIB[backend]
    backend_path = qnn_sdk / "lib" / "x86_64-linux-clang" / backend_lib
    if not backend_path.exists():
        # qnn-context-binary-generator is host-side; on host we may only
        # have specific backend libs. The generator usually accepts a
        # board-side libQnn*.so since it just wants the offline graph
        # specialization. Fall back to the aarch64 lib if x86_64 is missing.
        backend_path = qnn_sdk / "lib" / "aarch64-ubuntu-gcc9.4" / backend_lib
    if not backend_path.exists():
        raise RuntimeError(f"backend lib '{backend_lib}' not found in QNN SDK")

    out_dir = out_qnn_ctx.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(qnn_sdk / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"),
        "--model",
        str(model_so),
        "--backend",
        str(backend_path),
        "--binary_file",
        out_qnn_ctx.stem,
        "--output_dir",
        str(out_dir),
    ]
    _run(cmd, verbose=verbose)
    # The tool emits <stem>.bin in the output dir; rename to our expected
    # .qnn-ctx so the rest of the pipeline can match by extension.
    produced = out_dir / f"{out_qnn_ctx.stem}.bin"
    if produced.exists():
        produced.rename(out_qnn_ctx)
    if not out_qnn_ctx.exists():
        raise RuntimeError(f"context binary not produced at {out_qnn_ctx}")


def compile_chunk(
    chunk_id: int,
    chunk_entry: dict,
    model_onnx: pathlib.Path,
    qnn_sdk: pathlib.Path,
    board_sysroot: pathlib.Path,
    toolchain: pathlib.Path,
    backend: str,
    output_dir: pathlib.Path,
    verbose: bool,
) -> pathlib.Path:
    """Compile one (chunk, backend) → .qnn-ctx. Returns the output path."""
    work_dir = output_dir / f"chunk_{chunk_id}_{backend}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / f"chunk_{chunk_id}.onnx"
    _onnx_for_chunk(chunk_entry, model_onnx, onnx_path)
    cpp_path, bin_path = _qairt_convert(qnn_sdk, onnx_path, work_dir, verbose)
    model_so = work_dir / f"libqnn_chunk_{chunk_id}.so"
    _build_model_so(qnn_sdk, board_sysroot, toolchain, cpp_path, bin_path, model_so, verbose)
    out_qnn_ctx = output_dir / f"chunk_{chunk_id}.{backend}.qnn-ctx"
    _qnn_context_binary(qnn_sdk, model_so, backend, out_qnn_ctx, verbose)
    return out_qnn_ctx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunk-manifest", type=pathlib.Path, required=True, help="Path to chunk_manifest.json")
    p.add_argument("--chunks", default="all", help="Comma-separated chunk IDs (or 'all'). Default: all")
    p.add_argument(
        "--backends",
        default="qnn-gpu",
        help="Comma-separated backends from %s. Default: qnn-gpu" % ",".join(BACKEND_LIB),
    )
    p.add_argument("--model-onnx", type=pathlib.Path, help="Parent model ONNX (used when chunk has no onnx_path)")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--qnn-sdk-root", type=pathlib.Path, default=pathlib.Path(DEFAULT_QNN_SDK_ROOT))
    p.add_argument("--board-sysroot", type=pathlib.Path, default=pathlib.Path(DEFAULT_BOARD_SYSROOT))
    p.add_argument("--toolchain", type=pathlib.Path, default=pathlib.Path(DEFAULT_TOOLCHAIN))
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    manifest = json.loads(args.chunk_manifest.read_text())
    dispatches = manifest.get("dispatches", manifest.get("chunks", {}))
    if not dispatches:
        print("no chunks in manifest", file=sys.stderr)
        return 1

    if args.chunks == "all":
        chunk_ids = sorted(int(d.get("id", -1)) for d in dispatches.values() if d.get("id", -1) >= 0)
    else:
        chunk_ids = [int(s) for s in args.chunks.split(",") if s.strip()]

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKEND_LIB:
            print(f"unknown backend '{b}'; expected one of {list(BACKEND_LIB)}", file=sys.stderr)
            return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Optional XPU-RT feedback overlay (inert if breakdowns/feedback.json
    # is absent — see tools/compile/feedback_overlay.py).
    from compile.feedback_overlay import (
        load_feedback_overlay,
        merlin_dir_from_chunk_manifest,
    )

    merlin_dir = merlin_dir_from_chunk_manifest(args.chunk_manifest)
    overlay = load_feedback_overlay(merlin_dir) if merlin_dir else None
    if overlay and not overlay.is_empty:
        print(f"feedback overlay: {overlay.source_path} " f"(run_id={overlay.run_id}, hints={overlay.summary()})")

    # Reverse lookup chunk_id → dispatch_id (dict key) so we can match
    # the feedback overlay (which is keyed by operation_name).
    id_to_name: dict[int, str] = {}
    for name, e in dispatches.items():
        cid = e.get("id")
        if isinstance(cid, int):
            id_to_name[cid] = name

    summary: list[dict] = []
    for chunk_id in chunk_ids:
        # Find the entry with matching id.
        entry = next((d for d in dispatches.values() if d.get("id") == chunk_id), None)
        if entry is None:
            print(f"chunk_id {chunk_id} not in manifest, skipping", file=sys.stderr)
            continue

        # Per-chunk backend selection: feedback's pin_target overrides
        # the default --backends list when it names a known QNN backend.
        # Other hints are advisory only — their actionability lives in
        # tools/chunk_extractor.py (split/fuse). We log them here so the
        # operator sees them in the build output.
        chunk_backends = list(backends)
        chunk_advisory: list[str] = []
        if overlay is not None:
            decision = overlay.for_dispatch(id_to_name.get(chunk_id), f"chunk_{chunk_id}", str(chunk_id))
            if decision.pin_target and decision.pin_target in BACKEND_LIB:
                if decision.pin_target not in chunk_backends:
                    chunk_backends = [decision.pin_target]
                else:
                    chunk_backends = [decision.pin_target]
                print(f"  chunk_{chunk_id}: pin_target={decision.pin_target} " f"({decision.rationale})")
            elif decision.pin_target:
                # pin_target names something that isn't a QNN backend
                # (e.g. CPU_E from a heterogeneous schedule). Surface it
                # as advisory rather than silently ignoring.
                chunk_advisory.append(f"pin_target={decision.pin_target} (no QNN match)")
            for h in decision.advisory:
                chunk_advisory.append(h)
            if chunk_advisory:
                print(f"  chunk_{chunk_id}: advisory hints " f"{chunk_advisory} ({decision.rationale})")

        for backend in chunk_backends:
            try:
                out_path = compile_chunk(
                    chunk_id,
                    entry,
                    args.model_onnx,
                    args.qnn_sdk_root,
                    args.board_sysroot,
                    args.toolchain,
                    backend,
                    args.output_dir,
                    args.verbose,
                )
                rec = {"chunk": chunk_id, "backend": backend, "path": str(out_path), "size": out_path.stat().st_size}
                if chunk_advisory:
                    rec["feedback_advisory"] = list(chunk_advisory)
                summary.append(rec)
                print(f"OK chunk_{chunk_id} {backend} -> {out_path} " f"({out_path.stat().st_size} bytes)")
            except Exception as e:
                rec = {"chunk": chunk_id, "backend": backend, "error": str(e)}
                if chunk_advisory:
                    rec["feedback_advisory"] = list(chunk_advisory)
                summary.append(rec)
                print(f"FAIL chunk_{chunk_id} {backend}: {e}", file=sys.stderr)

    summary_path = args.output_dir / "compile_qnn_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    fail_count = sum(1 for s in summary if "error" in s)
    print(f"\n{len(summary) - fail_count}/{len(summary)} succeeded; " f"summary at {summary_path}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
