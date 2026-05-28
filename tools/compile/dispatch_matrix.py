#!/usr/bin/env python3
"""Compile a model for each (dispatch, target) cell and emit matrix.json.

Phase B of the heterogeneous-scheduling pipeline (model-agnostic). Drives
`iree-compile` once per target on the source MLIR, with the new
`--iree-hal-dump-executable-dispatch-modules-to=<dir>` flag enabled. Each
target's dump dir gives one MLIR + manifest per (executable, variant). The
canonical dispatch identity is the executable name (set in dispatch creation,
stable across targets); we cross-reference per-target dump dirs to record
feasibility per (canonical_dispatch, target).

Usage:
  python tools/compile/dispatch_matrix.py \
      --source <path/to/model.mlir> \
      --targets cpu,qnn_gpu,qnn_hta \
      --out-dir <matrix_output_dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_IREE_COMPILE = REPO_ROOT / "build/host-merlin-release-qrb/tools/iree-compile"


@dataclasses.dataclass
class TargetSpec:
    name: str
    device_target_arg: str
    plugin: str | None = None
    extra_flags: tuple[str, ...] = ()

    def iree_args(self) -> list[str]:
        out: list[str] = []
        if self.plugin:
            out.append(f"--iree-plugin={self.plugin}")
        out.append(f"--iree-hal-target-device={self.device_target_arg}")
        out.extend(self.extra_flags)
        return out


_QNN_GPU_DEVICE = (
    '#hal.device.target<"qnn", [#hal.executable.target<"qnn", '
    '"qnn-context-binary", {qnn_backend = "gpu", opaque_binary = true}>]>'
)
_QNN_HTA_DEVICE = (
    '#hal.device.target<"qnn", [#hal.executable.target<"qnn", '
    '"qnn-context-binary", {qnn_backend = "hta", opaque_binary = true}>]>'
)
_CPU_DEVICE = (
    '#hal.device.target<"local", [#hal.executable.target<"llvm-cpu", '
    '"embedded-elf-arm_64", {target_triple = "aarch64-linux-gnu", '
    'cpu = "generic", cpu_features = "+fullfp16", native_vector_size = 16 : i64}>]>'
)

_TARGETS: dict[str, TargetSpec] = {
    "cpu": TargetSpec("cpu", _CPU_DEVICE),
    "qnn_gpu": TargetSpec(
        "qnn_gpu",
        _QNN_GPU_DEVICE,
        plugin="hal_target_qnn",
        # Phase 0: failures must be loud. We deliberately do NOT pass
        # --iree-hal-qnn-allow-placeholder so missing patterns surface as
        # compile errors instead of silently-broken placeholder VMFBs.
        extra_flags=(),
    ),
    "qnn_hta": TargetSpec(
        "qnn_hta",
        _QNN_HTA_DEVICE,
        plugin="hal_target_qnn",
        # Phase 0: failures must be loud. We deliberately do NOT pass
        # --iree-hal-qnn-allow-placeholder so missing patterns surface as
        # compile errors instead of silently-broken placeholder VMFBs.
        extra_flags=(),
    ),
}


_DISPATCH_ID_RE = re.compile(r"_dispatch_(\d+)$")


def _preprocess_for_qnn(iree_opt: pathlib.Path, source: pathlib.Path, log_out: pathlib.Path) -> pathlib.Path | None:
    """For QNN targets, run iree-opt to put the source into a form that
    `merlin-convert-linalg-to-qnn` can match: NCHW convs are flipped to
    NHWC (legalize-layout pass), then named convs are generalized so the
    linalg.generic-form QNN patterns can match them.

    No im2col/decompose: that path explodes (one tensor.insert per output
    spatial position — 691,200 inserts on the yolov8 stem conv). The
    direct legalize+generalize path lowers `linalg.conv_2d_nhwc_hwcf`
    into a single `qnn.conv2d` op without scalar gather expansion.

    Returns the rewritten MLIR path on success, None on failure.
    """
    if not iree_opt.is_file():
        return None
    out_mlir = log_out.with_suffix(".qnn-pre.mlir")
    # IREE-imported modules use util.func; hand-written fixtures use
    # func.func. Apply legalize to both so both forms are supported.
    inner_passes = "merlin-qnn-legalize-layout-to-nhwc,linalg-generalize-named-ops"
    cmd = [
        str(iree_opt),
        "--iree-plugin=hal_target_qnn",
        f"--pass-pipeline=builtin.module(func.func({inner_passes})," f"util.func({inner_passes}))",
        str(source),
        "-o",
        str(out_mlir),
    ]
    pre_log = log_out.with_suffix(".im2col.log")
    pre_log.parent.mkdir(parents=True, exist_ok=True)
    with pre_log.open("w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    return out_mlir if rc == 0 and out_mlir.is_file() else None


def _run_compile(
    iree_compile: pathlib.Path,
    source_mlir: pathlib.Path,
    target: TargetSpec,
    dump_dir: pathlib.Path,
    vmfb_out: pathlib.Path,
    log_out: pathlib.Path,
    phase_dump_dir: pathlib.Path | None = None,
    compile_to: str | None = None,
) -> tuple[bool, str]:
    """Run iree-compile with the dispatch-modules dump enabled."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    actual_source = source_mlir
    if target.name.startswith("qnn_"):
        iree_opt = iree_compile.parent / "iree-opt"
        pre = _preprocess_for_qnn(iree_opt, source_mlir, log_out)
        if pre is not None:
            actual_source = pre
    cmd = [
        str(iree_compile),
        *target.iree_args(),
        f"--iree-hal-dump-executable-dispatch-modules-to={dump_dir}",
        "-o",
        str(vmfb_out),
        str(actual_source),
    ]
    if phase_dump_dir is not None:
        phase_dump_dir.mkdir(parents=True, exist_ok=True)
        cmd.append(f"--dump-compilation-phases-to={phase_dump_dir}")
    if compile_to:
        cmd.append(f"--compile-to={compile_to}")
    with log_out.open("w") as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    return proc.returncode == 0, " ".join(cmd)


def _parse_dump_dir(dump_dir: pathlib.Path) -> dict[str, dict]:
    """Index every emitted dispatch by the canonical executable name.

    The canonical key is derived from the manifest's `entry_point` field
    (`<exec>::<variant>::<export>`); the executable name is stable across
    targets while filename and variant names are not.

    Returns map: canonical_exec_name -> {
       "func": str, "binding_byte_sizes": [...], "workload": [...],
       "mlir": <abs_path>, "manifest": <abs_path>, "variant": str,
       "dispatch_id": int | None,
    }
    """
    out: dict[str, dict] = {}
    for mlir_path in sorted(dump_dir.glob("*_dispatches.mlir")):
        manifest_path = mlir_path.with_name(mlir_path.name.replace(".mlir", ".manifest.json"))
        if not manifest_path.exists():
            continue
        try:
            entries = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if not entries:
            continue
        e = entries[0]
        ep = e.get("entry_point", "")
        if "::" not in ep:
            continue
        exec_name, variant_name, _ = ep.split("::", 2)
        m = _DISPATCH_ID_RE.search(exec_name)
        dispatch_id = int(m.group(1)) if m else None
        out[exec_name] = {
            "func": e.get("func", ""),
            "binding_byte_sizes": list(e.get("binding_byte_sizes", [])),
            "workload": list(e.get("workload", [])),
            "mlir": str(mlir_path),
            "manifest": str(manifest_path),
            "variant": variant_name,
            "dispatch_id": dispatch_id,
            "all_entries": entries,
        }
    return out


_QNN_FAIL_RE = re.compile(r"export '(?P<exp>[^']+)' has no QNN graph", re.MULTILINE)
_QNN_CODEGEN_ERR_RE = re.compile(r"error:.*qnn-codegen:", re.MULTILINE)
_QNN_REAL_PAYLOAD_RE = re.compile(r"embedded \d+ bytes of in-compiler qnn-graph", re.MULTILINE)
_QNN_PLACEHOLDER_RE = re.compile(r"embedding placeholder", re.MULTILINE)


def _parse_qnn_failed_exports(log_path: pathlib.Path) -> set[str]:
    """Scan an iree-compile log for `[qnn-target] export 'X' has no QNN
    graph` lines and return the set of failing export symbol names.

    These names are the *canonical executable name* (matches the
    manifest's entry_point first segment), so the matrix builder can mark
    those cells infeasible without dropping the rest.
    """
    if not log_path.is_file():
        return set()
    out: set[str] = set()
    text = log_path.read_text(errors="replace")
    for m in _QNN_FAIL_RE.finditer(text):
        out.add(m.group("exp"))
    return out


def _probe_per_dispatch(
    iree_compile: pathlib.Path,
    target: TargetSpec,
    dispatch_mlir: pathlib.Path,
) -> dict:
    """Standalone-compile one per-dispatch dump MLIR to determine real
    feasibility for `target`. Returns a dict with:
      ok        : True if iree-compile produced a VMFB
      real      : True if the QNN target embedded a real qnn-graph payload
                  (False for slow_memcpy/passthrough placeholders, which
                  would fail at runtime even though compile succeeds)
      vmfb_size : bytes of the produced VMFB or 0
      stderr    : last 4 KB of stderr (for diagnostics)
    For non-QNN targets `real` is always True when `ok` is True.
    """
    out = dispatch_mlir.with_suffix(".probe.vmfb")
    cmd = [str(iree_compile)]
    if target.plugin:
        cmd.append(f"--iree-plugin={target.plugin}")
    cmd.extend(["-o", str(out), str(dispatch_mlir)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0 and out.is_file()
    err_tail = (result.stderr or "")[-4096:]
    is_qnn = target.name.startswith("qnn_")
    if ok and is_qnn:
        real = bool(_QNN_REAL_PAYLOAD_RE.search(err_tail))
        if not real and _QNN_PLACEHOLDER_RE.search(err_tail):
            real = False
    else:
        real = ok
    size = out.stat().st_size if out.is_file() else 0
    return {"ok": ok, "real": real, "vmfb_size": size, "stderr_tail": err_tail}


def _log_has_qnn_codegen_errors(log_path: pathlib.Path) -> bool:
    """Returns True if the compile log contains any `qnn-codegen:` errors —
    a broader signal than the Phase-0 "no QNN graph" diagnostic. Used to
    conservatively mark all of a target's dispatches infeasible when the
    module-level compile failed: without a VMFB the per-dispatch dumps
    cannot actually run on that target.
    """
    if not log_path.is_file():
        return False
    text = log_path.read_text(errors="replace")
    return _QNN_CODEGEN_ERR_RE.search(text) is not None


_EXPORT_TO_EXEC_RE = re.compile(r"^(?P<exec>.+_dispatch_\d+)(?:_.*)?$")


def _canonical_exec_from_export(export: str) -> str:
    """Strip the trailing `_<dispatch_op_summary>` from an export symbol
    to get the canonical executable name (matches our manifest keys).
    e.g. 'foo_dispatch_3_slow_memcpy' -> 'foo_dispatch_3'.
    """
    m = _EXPORT_TO_EXEC_RE.match(export)
    return m.group("exec") if m else export


def _build_matrix(
    per_target: dict[str, dict[str, dict]],
    target_compile_status: dict[str, dict],
    target_qnn_failed_exports: dict[str, set[str]],
    target_probes: dict[str, dict[str, dict]] | None = None,
) -> dict:
    """Cross-reference per-target dumps into a unified matrix.

    A "row" is a canonical dispatch (executable name); a "column" is a target.
    A cell is feasible if the target's dump produced an MLIR for that
    dispatch AND the iree-compile log did not list its export among
    QNN-pattern-unmatched exports (Phase 0 fail-fast diagnostic).
    """
    all_dispatches: set[str] = set()
    for d in per_target.values():
        all_dispatches.update(d.keys())

    matrix = {}
    for canonical in sorted(all_dispatches):
        row = {}
        for target_name, dispatches in per_target.items():
            entry = dispatches.get(canonical)
            failed_exports = target_qnn_failed_exports.get(target_name, set())
            # An export NAME like `<canonical>_<dispatch_op_summary>` is what
            # appears in the QNN-target error. We mark the cell infeasible
            # if any export sharing this canonical's prefix is in the
            # failure list.
            qnn_failed = any(_canonical_exec_from_export(e) == canonical for e in failed_exports)
            # Standalone per-dispatch probe is the source of truth.
            # When available, it tells us exactly whether THIS dispatch
            # compiles to a real VMFB on this target (not just whether the
            # whole-module compile succeeded). A QNN target's probe also
            # distinguishes a real qnn-graph payload from a slow_memcpy
            # placeholder that would fail at runtime.
            probe = (target_probes or {}).get(target_name, {}).get(canonical)
            if entry is None:
                row[target_name] = {"feasible": False, "reason": "missing"}
            elif qnn_failed:
                row[target_name] = {
                    "feasible": False,
                    "reason": "qnn-pattern-unmatched",
                    "mlir": entry["mlir"],
                    "manifest": entry["manifest"],
                }
            elif probe is not None:
                if probe["real"]:
                    row[target_name] = {
                        "feasible": True,
                        "func": entry["func"],
                        "binding_byte_sizes": entry["binding_byte_sizes"],
                        "workload": entry["workload"],
                        "mlir": entry["mlir"],
                        "manifest": entry["manifest"],
                        "variant": entry["variant"],
                        "probe_vmfb_size": probe["vmfb_size"],
                    }
                else:
                    reason = "qnn-placeholder-only" if probe["ok"] else "standalone-compile-failed"
                    row[target_name] = {
                        "feasible": False,
                        "reason": reason,
                        "mlir": entry["mlir"],
                        "manifest": entry["manifest"],
                    }
            else:
                # No probe data; fall back to dump-presence heuristic
                # (legacy behavior — only used if --no-probe is passed).
                row[target_name] = {
                    "feasible": True,
                    "func": entry["func"],
                    "binding_byte_sizes": entry["binding_byte_sizes"],
                    "workload": entry["workload"],
                    "mlir": entry["mlir"],
                    "manifest": entry["manifest"],
                    "variant": entry["variant"],
                }
        matrix[canonical] = row

    return {
        "targets": list(per_target.keys()),
        "compile_status": target_compile_status,
        "dispatches": matrix,
    }


def _summary(matrix: dict) -> str:
    targets = matrix["targets"]
    counts = {t: 0 for t in targets}
    total = 0
    for cells in matrix["dispatches"].values():
        total += 1
        for t in targets:
            if cells.get(t, {}).get("feasible"):
                counts[t] += 1
    lines = [f"Total canonical dispatches: {total}"]
    for t in targets:
        cs = matrix["compile_status"][t]
        status = "OK" if cs["ok"] else "compile-FAIL"
        lines.append(f"  {t:10s} {status:14s} feasible={counts[t]}/{total}")
    return "\n".join(lines)


def _load_dispatch_graph(path: pathlib.Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if "dispatch_graph" in payload and isinstance(payload["dispatch_graph"], dict):
        return payload["dispatch_graph"]
    dispatches = payload.get("dispatches")
    if isinstance(dispatches, dict):
        graph: dict[str, dict] = {}
        for name, entry in dispatches.items():
            if not isinstance(entry, dict):
                continue
            graph[name] = {
                "dependencies": list(entry.get("dependencies", [])),
                "id": entry.get("id"),
                "subid": entry.get("subid"),
                "ordinal": entry.get("ordinal"),
                "total": entry.get("total"),
                "op_summary": entry.get("op_summary"),
            }
        return graph
    raise ValueError(f"unsupported dispatch-graph schema: {path}")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=pathlib.Path, required=True, help="Source MLIR file (e.g., yolov8.mlir)")
    p.add_argument(
        "--targets", default="cpu,qnn_gpu,qnn_hta", help="Comma-separated targets from: " + ",".join(_TARGETS)
    )
    p.add_argument(
        "--out-dir", type=pathlib.Path, required=True, help="Output dir for per-target dump dirs + matrix.json"
    )
    p.add_argument(
        "--iree-compile",
        type=pathlib.Path,
        default=DEFAULT_IREE_COMPILE,
        help=f"Path to iree-compile (default: {DEFAULT_IREE_COMPILE})",
    )
    p.add_argument(
        "--dispatch-graph-json",
        type=pathlib.Path,
        default=None,
        help="Optional dependency manifest to carry into matrix.json. "
        "Accepts either a top-level dispatch_graph mapping or "
        "breakdown_vmfb.py's dispatches schema.",
    )
    p.add_argument(
        "--dump-compilation-phases",
        action="store_true",
        help="Also dump IREE phase MLIR under each target dir. This is "
        "used to materialize the exact constant arena matching "
        "the per-dispatch wrappers.",
    )
    p.add_argument(
        "--compile-to",
        default=None,
        help="Forward --compile-to=<phase> to iree-compile. Useful "
        "with --dump-compilation-phases when the final target "
        "link/codegen is not needed for wrapper capture.",
    )
    p.add_argument("--clean", action="store_true", help="Remove out-dir before running")
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the per-dispatch standalone-compile probe. "
        "Without probes the matrix falls back to dump-"
        "presence as the feasibility heuristic, which can "
        "false-positive on QNN targets that reject specific "
        "ops at codegen time. Use only for fast iteration.",
    )
    args = p.parse_args(argv)

    if not args.source.is_file():
        print(f"error: source MLIR not found: {args.source}", file=sys.stderr)
        return 2
    if not args.iree_compile.is_file():
        print(f"error: iree-compile not found: {args.iree_compile}", file=sys.stderr)
        return 2
    if args.dispatch_graph_json and not args.dispatch_graph_json.is_file():
        print(f"error: dispatch-graph JSON not found: {args.dispatch_graph_json}", file=sys.stderr)
        return 2

    targets = []
    for t in args.targets.split(","):
        t = t.strip()
        if t not in _TARGETS:
            print(f"error: unknown target {t!r}; known: {list(_TARGETS)}", file=sys.stderr)
            return 2
        targets.append(_TARGETS[t])

    out_dir: pathlib.Path = args.out_dir
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_target: dict[str, dict[str, dict]] = {}
    target_compile_status: dict[str, dict] = {}
    target_qnn_failed_exports: dict[str, set[str]] = {}

    for target in targets:
        target_dir = out_dir / target.name
        dump_dir = target_dir / "dispatches"
        target_dir.mkdir(parents=True, exist_ok=True)
        vmfb = target_dir / "full.vmfb"
        log = target_dir / "compile.log"
        phase_dump_dir = target_dir / "phases" if args.dump_compilation_phases else None
        ok, cmd = _run_compile(
            args.iree_compile,
            args.source,
            target,
            dump_dir,
            vmfb,
            log,
            phase_dump_dir=phase_dump_dir,
            compile_to=args.compile_to,
        )
        # Even when the full-model link step fails (because some QNN
        # dispatches lack patterns), the per-dispatch dumps written
        # earlier are valid. Parse the log to learn WHICH exports
        # failed; remaining dispatches are still feasible for the target.
        qnn_failed = _parse_qnn_failed_exports(log)
        target_qnn_failed_exports[target.name] = qnn_failed
        target_compile_status[target.name] = {
            "ok": ok,
            "cmd": cmd,
            "vmfb": str(vmfb) if ok else None,
            "log": str(log),
            "qnn_unmatched_exports": sorted(qnn_failed),
            "qnn_codegen_errors": _log_has_qnn_codegen_errors(log),
        }
        per_target[target.name] = _parse_dump_dir(dump_dir)
        n = len(per_target[target.name])
        suffix = f", {len(qnn_failed)} unmatched" if qnn_failed else ""
        print(f"[{target.name:10s}] compile={'OK' if ok else 'FAIL'} " f"dispatches_dumped={n}{suffix}")

    target_probes: dict[str, dict[str, dict]] = {}
    if not args.no_probe:
        for target in targets:
            probes: dict[str, dict] = {}
            for canonical, entry in per_target[target.name].items():
                mlir_path = pathlib.Path(entry["mlir"])
                probes[canonical] = _probe_per_dispatch(args.iree_compile, target, mlir_path)
            target_probes[target.name] = probes
            real = sum(1 for p in probes.values() if p["real"])
            ph = sum(1 for p in probes.values() if p["ok"] and not p["real"])
            failed = sum(1 for p in probes.values() if not p["ok"])
            print(f"[{target.name:10s}] probe real={real} " f"placeholder={ph} failed={failed} " f"(of {len(probes)})")
    matrix = _build_matrix(
        per_target, target_compile_status, target_qnn_failed_exports, target_probes=target_probes or None
    )
    dispatch_graph = _load_dispatch_graph(args.dispatch_graph_json)
    if dispatch_graph:
        matrix["dispatch_graph"] = dispatch_graph
    matrix_path = out_dir / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2))
    print(f"\nwrote {matrix_path}\n")
    print(_summary(matrix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
