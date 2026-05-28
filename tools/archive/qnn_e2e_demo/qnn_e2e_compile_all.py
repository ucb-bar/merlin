"""Multi-target QNN compile orchestrator with full phase tracing.

Takes one MLIR fixture and drives the FULL compile pipeline against every
backend in scope on QRB5165 (CPU via IREE LLVM, GPU via QNN passthrough).
For each backend it:

  1. Loads + filters the kernel manifest to that target.
  2. Precompiles each kernel (board-side build for QNN, native for CPU).
  3. Runs spec_gen to emit transform_spec.mlir + sidecar JSON.
  4. Invokes iree-compile with `--mlir-print-ir-after-all` and dumps every
     pass output to its own file under `<bundle>/<target>/phases/<NN>_<pass>.mlir`.
  5. Saves the final .vmfb plus a phase index for navigation.

Bundle layout under `<out>`:

    summary.md
    input.mlir
    targets/
      cpu/
        compile.flags
        compile.stderr
        compile.stdout
        artifact.vmfb
        phases/00_iree-auto-input-conversion.mlir
        phases/01_iree-import-public.mlir
        ...
        phase_index.txt
      qnn_gpu/
        kernel_manifest_filtered.json
        kernels_cache/<kernel>.qnn-ctx (one per matched kernel)
        transform_spec.mlir
        transform_spec.qnn_manifest.json
        compile.flags
        compile.stderr
        compile.stdout
        artifact.vmfb
        phases/...
        phase_index.txt

Usage:

    QNN_USE_BOARD_BUILD=1 QNN_BOARD_HOST=qdev \\
        QNN_BOARD_QAIRT_ROOT=/tmp/qnn_probe \\
        conda run -n merlin-dev uv run python \\
        tools/kernels/qnn_e2e_compile_all.py \\
        benchmarks/QRB5165/mlir/heterogeneous_smoke.mlir \\
        --kernel-manifest benchmarks/QRB5165/kernels/manifest.json \\
        --out build/qnn_e2e_all
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from kernels import manifest as _kmanifest  # noqa: E402
from kernels import precompile as _kprecompile  # noqa: E402
from kernels import spec_gen as _kspec_gen  # noqa: E402

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TargetSpec:
    name: str  # human-readable target id (cpu / qnn_gpu / qnn_hta)
    iree_flags: tuple[str, ...]  # iree-compile flags that select this backend
    manifest_target: str | None  # which manifest target to filter to (None for CPU)
    needs_board_build: bool  # True if kernel ctxbins must be built on the QRB5165


_TARGETS = {
    "cpu": TargetSpec(
        name="cpu",
        iree_flags=(
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-triple=aarch64-linux-gnu",
            "--iree-llvmcpu-target-cpu=cortex-a77",
            "--iree-execution-model=async-external",
        ),
        manifest_target=None,
        needs_board_build=False,
    ),
    "qnn_gpu": TargetSpec(
        name="qnn_gpu",
        iree_flags=(
            "--iree-hal-target-device=qnn",
            "--iree-hal-qnn-backend=gpu",
            "--iree-execution-model=async-external",
            # When a manifest entry doesn't match (e.g., chained activation
            # not yet supported by the matcher), embed a placeholder so the
            # rest of the compile completes and we still get a viewable
            # VMFB. Runtime would fail to load such a placeholder, but we
            # surface that explicitly in the per-backend report.
            "--iree-hal-qnn-allow-placeholder",
        ),
        manifest_target="qnn-gpu",
        needs_board_build=True,
    ),
    "qnn_hta": TargetSpec(
        name="qnn_hta",
        iree_flags=(
            "--iree-hal-target-device=qnn",
            "--iree-hal-qnn-backend=hta",
            "--iree-execution-model=async-external",
            "--iree-hal-qnn-allow-placeholder",
        ),
        manifest_target="qnn-hta",
        needs_board_build=True,
    ),
}


def _setup_kernels(
    target: TargetSpec,
    kernel_manifest: pathlib.Path,
    out_dir: pathlib.Path,
) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    """Run precompile + spec_gen for `target`. Returns (transform_spec_path,
    qnn_manifest_path) or (None, None) when the target doesn't use the
    kernel-embedding path (e.g., plain CPU)."""
    if target.manifest_target is None:
        return None, None

    full = _kmanifest.load(kernel_manifest)
    selected = [k for k in full.kernels if target.manifest_target in k.targets]
    if not selected:
        _LOG.warning("manifest has no kernels with target=%s", target.manifest_target)
        return None, None
    m = _kmanifest.Manifest(
        path=full.path,
        schema_version=full.schema_version,
        kernels=tuple(selected),
        select=full.select,
    )

    # Mirror the filtered manifest for visibility.
    filtered_path = out_dir / "kernel_manifest_filtered.json"
    filtered_path.write_text(
        json.dumps(
            {
                "schema_version": full.schema_version,
                "target_filter": target.manifest_target,
                "kernels": [k.name for k in selected],
            },
            indent=2,
        )
        + "\n"
    )

    cache_dir = out_dir / "kernels_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    objects = _kprecompile.precompile(
        m,
        cache_dir,
        targets_filter=(target.manifest_target,),
    )

    spec_path = out_dir / "transform_spec.mlir"
    gen = _kspec_gen.emit(m, objects, spec_path, object_search_path=cache_dir)

    qnn_manifest = gen.qnn_manifest_path
    if qnn_manifest is not None and qnn_manifest != out_dir / qnn_manifest.name:
        # spec_gen writes the sidecar next to the spec; ensure it's in our
        # bundle root (it is, since we passed spec_path under out_dir).
        pass
    return spec_path, qnn_manifest


def _iree_compile_path() -> pathlib.Path:
    """Resolve the in-tree iree-compile binary."""
    candidate = REPO / "build" / "host-vanilla-release" / "tools" / "iree-compile"
    if not candidate.exists():
        raise FileNotFoundError(
            f"iree-compile not found at {candidate} — build it first via "
            f"./merlin build --profile vanilla --cmake-target iree-compile"
        )
    return candidate


_PASS_DUMP_RE = re.compile(r"^// -----// IR Dump After (\S+) (?:\(([^)]+)\))? //----- //")


def _split_ir_dumps(stderr_text: str, phases_dir: pathlib.Path) -> list[str]:
    """Walk the iree-compile stderr (with --mlir-print-ir-after-all) and
    write each pass's IR snapshot to its own file. Returns a chronological
    list of pass names (for the phase_index)."""
    phases_dir.mkdir(parents=True, exist_ok=True)
    lines = stderr_text.splitlines()
    pass_index: list[str] = []
    cur_buf: list[str] = []
    cur_pass: str | None = None

    def flush() -> None:
        nonlocal cur_buf, cur_pass
        if cur_pass is None:
            cur_buf.clear()
            return
        idx = len(pass_index)
        # Sanitize pass name for filesystem.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", cur_pass)
        path = phases_dir / f"{idx:03d}_{safe}.mlir"
        path.write_text("\n".join(cur_buf) + "\n")
        pass_index.append(cur_pass)
        cur_buf = []

    for line in lines:
        m = _PASS_DUMP_RE.match(line)
        if m:
            flush()
            cur_pass = m.group(2) or m.group(1)  # prefer the cli flag
            cur_buf = []
            continue
        if cur_pass is not None:
            cur_buf.append(line)
    flush()

    # Write the index for navigation.
    idx_path = phases_dir.parent / "phase_index.txt"
    idx_path.write_text("\n".join(f"{i:03d}\t{name}" for i, name in enumerate(pass_index)) + "\n")
    return pass_index


def _compile_one(
    mlir: pathlib.Path,
    target: TargetSpec,
    kernel_manifest: pathlib.Path | None,
    target_dir: pathlib.Path,
) -> dict:
    """Drive iree-compile for one target. Returns a metadata dict for the
    summary.md report."""
    target_dir.mkdir(parents=True, exist_ok=True)

    spec_path: pathlib.Path | None = None
    qnn_sidecar: pathlib.Path | None = None
    if kernel_manifest is not None:
        spec_path, qnn_sidecar = _setup_kernels(target, kernel_manifest, target_dir)

    flags: list[str] = list(target.iree_flags)
    if spec_path is not None:
        flags.append(f"--iree-preprocessing-transform-spec-filename={spec_path}")
        cache_dir = target_dir / "kernels_cache"
        flags.append(f"--iree-hal-executable-object-search-path={cache_dir}")
    if qnn_sidecar is not None:
        flags.append(f"--iree-hal-qnn-manifest={qnn_sidecar}")

    # Curated set of "key transition" passes — the ones that materially
    # change IR shape. Dumping every pass on a real model produces GB of
    # near-duplicate IR; this list captures the moments worth examining
    # without exploding disk usage.
    # Curated key passes — names verified against actual iree-compile dump.
    KEY_PASSES = (
        "iree-import-public",
        "iree-abi-wrap-entry-points",
        "iree-preprocessing-transform-interpreter",
        "iree-flow-outline-dispatch-regions",
        "iree-flow-deduplicate-executables",
        "iree-stream-conversion",
        "iree-stream-schedule-execution",
        "iree-hal-materialize-target-devices",
        "iree-hal-materialize-interfaces",
        "iree-hal-translate-target-executable-variants",
        "iree-hal-translate-all-executables",
        "iree-hal-conversion",
        "iree-hal-prune-executables",
        "iree-vm-conversion",
    )
    print_after = ",".join(KEY_PASSES)

    artifact = target_dir / "artifact.vmfb"
    cmd = [
        str(_iree_compile_path()),
        str(mlir),
        "-o",
        str(artifact),
        *flags,
        f"--mlir-print-ir-after={print_after}",
        "--mlir-disable-threading",
    ]
    (target_dir / "compile.flags").write_text(" \\\n  ".join(cmd) + "\n")

    env = os.environ.copy()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)

    (target_dir / "compile.stdout").write_text(res.stdout)
    (target_dir / "compile.stderr").write_text(res.stderr)

    phases_dir = target_dir / "phases"
    pass_names = _split_ir_dumps(res.stderr, phases_dir)

    success = res.returncode == 0 and artifact.exists()
    embedded_bytes = sum(
        int(m.group(1)) for m in re.finditer(r"\[qnn-target\] embedded (\d+) bytes", res.stderr + res.stdout)
    )
    placeholders = res.stderr.count("embedding placeholder")

    # Count dispatches in the post-outline phase (most representative of the
    # final dispatch breakdown; pre-codegen).
    dispatch_count = 0
    matched_qnn = 0
    for f in sorted(phases_dir.glob("*iree-flow-outline-dispatch-regions.mlir")):
        text = f.read_text()
        # Each outline produces one `flow.dispatch @<sym>` per dispatch.
        dispatches = re.findall(r"flow\.dispatch @[A-Za-z_$][A-Za-z0-9_$]*", text)
        if len(dispatches) > dispatch_count:
            dispatch_count = len(dispatches)
            matched_qnn = sum(1 for d in dispatches if "@kb_qnn_" in d)

    return {
        "target": target.name,
        "ok": success,
        "exit_code": res.returncode,
        "artifact": str(artifact) if artifact.exists() else None,
        "artifact_bytes": artifact.stat().st_size if artifact.exists() else 0,
        "phase_count": len(pass_names),
        "dispatch_count": dispatch_count,
        "matched_qnn_dispatches": matched_qnn,
        "qnn_embedded_bytes": embedded_bytes,
        "qnn_placeholder_count": placeholders,
        "spec_path": str(spec_path) if spec_path else None,
        "qnn_sidecar_path": str(qnn_sidecar) if qnn_sidecar else None,
    }


def _write_summary(out: pathlib.Path, mlir: pathlib.Path, results: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Multi-target QNN e2e compilation bundle\n")
    lines.append(f"- input MLIR: `{mlir}`")
    lines.append(f"- targets: {len(results)} ({', '.join(r['target'] for r in results)})\n")

    lines.append("## Per-target outcome\n")
    lines.append("| Target | Result | VMFB | Phases | Dispatches | QNN-matched | QNN bytes | Placeholders |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        ok = "✅" if r["ok"] else "❌"
        vmfb = f"`{pathlib.Path(r['artifact']).name}` " f"({r['artifact_bytes']:,} B)" if r["artifact"] else "—"
        lines.append(
            f"| `{r['target']}` | {ok} (exit {r['exit_code']}) | "
            f"{vmfb} | {r['phase_count']} | "
            f"{r['dispatch_count']} | {r['matched_qnn_dispatches']} | "
            f"{r['qnn_embedded_bytes']:,} B | {r['qnn_placeholder_count']} |"
        )
    lines.append("")

    lines.append("## Per-target layout\n")
    for r in results:
        d = r["target"]
        lines.append(f"### `{d}`\n")
        lines.append(f"- flags: `targets/{d}/compile.flags`")
        lines.append(f"- stderr / stdout: `targets/{d}/compile.{{stderr,stdout}}`")
        lines.append(f"- per-pass IR dumps: `targets/{d}/phases/<NNN>_<pass>.mlir`")
        lines.append(f"- pass index: `targets/{d}/phase_index.txt`")
        lines.append(
            f"- final VMFB: `targets/{d}/artifact.vmfb`"
            if r["artifact"]
            else "- final VMFB: (compile failed; see stderr)"
        )
        if r["spec_path"]:
            lines.append(f"- transform spec: `targets/{d}/transform_spec.mlir`")
        if r["qnn_sidecar_path"]:
            lines.append(f"- QNN sidecar manifest: `targets/{d}/transform_spec.qnn_manifest.json`")
            lines.append(f"- per-kernel ctxbins: `targets/{d}/kernels_cache/board_*/<kernel>.qnn-ctx`")
        lines.append("")

    lines.append("## How to read a phase dump\n")
    lines.append(
        "Each `phases/<NNN>_<pass>.mlir` file is the IR *immediately after* "
        "the named IREE pass ran. They're numbered chronologically; "
        "`phase_index.txt` is the navigation index. To diff back-to-back "
        "passes:\n\n"
        "    diff -u targets/qnn_gpu/phases/042_*.mlir targets/qnn_gpu/phases/043_*.mlir\n\n"
        "Notable phases to look for:\n"
        "- `iree-preprocessing-transform-interpreter` — kernel embedding "
        "applies here; matched dispatches become `flow.dispatch @kb_<name>`.\n"
        "- `iree-flow-form-dispatch-regions` / `iree-flow-clone-into-dispatch` — "
        "remaining ops fuse into dispatches.\n"
        "- `iree-stream-conversion` — tensors become `stream.resource`s.\n"
        "- `iree-hal-conversion` — final lowering to HAL ops; QNN executable "
        "binaries are embedded by the `serializeExecutable` callback in "
        "`compiler/plugins/target/QNN/QNNTarget.cpp`."
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mlir", type=pathlib.Path)
    p.add_argument("--kernel-manifest", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument(
        "--targets",
        default="cpu,qnn_gpu",
        help="Comma-separated target ids (cpu, qnn_gpu, qnn_hta).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.mlir, out / "input.mlir")

    target_ids = [t.strip() for t in args.targets.split(",") if t.strip()]
    results: list[dict] = []
    for tid in target_ids:
        if tid not in _TARGETS:
            _LOG.error("unknown target '%s' (known: %s)", tid, sorted(_TARGETS))
            continue
        target = _TARGETS[tid]
        target_dir = out / "targets" / tid
        _LOG.info("=== compiling for target=%s ===", tid)
        manifest = args.kernel_manifest if target.manifest_target else None
        r = _compile_one(args.mlir, target, manifest, target_dir)
        _LOG.info("  result: ok=%s phases=%d qnn_bytes=%d", r["ok"], r["phase_count"], r["qnn_embedded_bytes"])
        results.append(r)

    _write_summary(out, args.mlir, results)
    print(f"\n  bundle:  {out}")
    print(f"  summary: {out / 'summary.md'}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
