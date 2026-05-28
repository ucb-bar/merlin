"""End-to-end QNN heterogeneous-compile inspector.

Drives the upper half of the QNN compile flow — kernel manifest load,
on-board precompile, transform-spec generation — and dumps every artifact
to a single inspection directory so you can see *what gets generated and
how the matches are wired*. Stops short of `iree-compile` because the
host-side IREE-pass crash in HAL conversion is tracked separately
(task #106 / #107).

What the inspector emits under `<out_dir>/`:

    inspect/
      summary.md                     — table of contents + match wiring
      input.mlir                     — copy of the user MLIR
      transform_spec.mlir            — generated cast-and-call spec
      transform_spec.qnn_manifest.json — symbol → .qnn-ctx mapping
      kernels/
        <kernel_name>/
          source.qnn.cpp             — the kernel source we built
          match.mlir                 — the linalg-DAG match pattern
          ctxbin.qnn-ctx             — the on-board-built blob
          ctxbin.size                — size in bytes (gut-check)

Run with:

    QNN_USE_BOARD_BUILD=1 QNN_BOARD_HOST=qdev QNN_BOARD_QAIRT_ROOT=/tmp/qnn_probe \\
        ./merlin compile-tool inspect \\
            benchmarks/QRB5165/mlir/heterogeneous_smoke.mlir \\
            --target qnn-gpu \\
            --kernel-manifest benchmarks/QRB5165/kernels/manifest.json \\
            --out build/qnn_e2e_inspect

(or directly: `uv run tools/kernels/qnn_e2e_inspect.py ...`)
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from kernels import manifest as _kmanifest  # noqa: E402
from kernels import precompile as _kprecompile  # noqa: E402
from kernels import spec_gen as _kspec_gen  # noqa: E402

_LOG = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mlir", type=pathlib.Path, help="Input model MLIR.")
    p.add_argument(
        "--target",
        default="qnn-gpu",
        choices=["qnn-gpu", "qnn-hta"],
        help="QNN backend target to filter the manifest with.",
    )
    p.add_argument("--kernel-manifest", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path, help="Output directory for the inspection bundle.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    kernels_dir = out / "kernels"
    kernels_dir.mkdir(exist_ok=True)
    cache_dir = out / "_cache"
    cache_dir.mkdir(exist_ok=True)

    # 1. Mirror the input MLIR.
    shutil.copy2(args.mlir, out / "input.mlir")

    # 2. Load + filter manifest.
    full = _kmanifest.load(args.kernel_manifest.resolve())
    selected = [k for k in full.kernels if args.target in k.targets]
    if not selected:
        _LOG.error("manifest at %s has no kernels with target=%s", args.kernel_manifest, args.target)
        return 1
    m = _kmanifest.Manifest(
        path=full.path,
        schema_version=full.schema_version,
        kernels=tuple(selected),
        select=full.select,
    )
    _LOG.info("manifest: %d kernels selected for target=%s", len(selected), args.target)

    # 3. Precompile (board-side iff QNN_USE_BOARD_BUILD=1).
    objects = _kprecompile.precompile(m, cache_dir, targets_filter=(args.target,))
    _LOG.info("precompile: produced %d ctxbin(s)", len(objects))

    # 4. Generate transform spec + sidecar JSON manifest.
    spec_path = out / "transform_spec.mlir"
    gen = _kspec_gen.emit(m, objects, spec_path, object_search_path=cache_dir)
    sidecar = gen.qnn_manifest_path
    if sidecar is not None and sidecar != out / sidecar.name:
        shutil.copy2(sidecar, out / sidecar.name)

    # 5. Per-kernel snapshot directory.
    for kernel in selected:
        kdir = kernels_dir / kernel.name
        kdir.mkdir(exist_ok=True)
        if kernel.source.exists():
            shutil.copy2(kernel.source, kdir / f"source{kernel.source.suffix}")
        if kernel.match.kind == "linalg_dag" and kernel.match.spec_path.exists():
            shutil.copy2(kernel.match.spec_path, kdir / "match.mlir")
        art = objects.get((kernel.name, args.target))
        if art is not None and art.path.exists():
            ctx = kdir / "ctxbin.qnn-ctx"
            shutil.copy2(art.path, ctx)
            (kdir / "ctxbin.size").write_text(f"{ctx.stat().st_size} bytes\n")

    # 6. summary.md — table of contents + match wiring.
    qmap: dict[str, str] = {}
    if sidecar is not None and sidecar.exists():
        qmap = json.loads(sidecar.read_text())
    summary_lines: list[str] = []
    summary_lines.append("# QNN heterogeneous-compile inspection bundle\n")
    summary_lines.append(f"- input MLIR: `{args.mlir.name}`")
    summary_lines.append(f"- target: `{args.target}`")
    summary_lines.append(f"- manifest: `{args.kernel_manifest}`")
    summary_lines.append("- transform spec: `transform_spec.mlir`")
    summary_lines.append("- QNN sidecar manifest: " "`transform_spec.qnn_manifest.json`\n")
    summary_lines.append("## Kernel-by-kernel wiring\n")
    summary_lines.append("| Kernel | match pattern | export symbol → ctxbin path |")
    summary_lines.append("|---|---|---|")
    for kernel in selected:
        export = _kspec_gen._export_name(kernel)  # type: ignore[attr-defined]
        ctxbin_rel = qmap.get(export, "?")
        match_kind = kernel.match.kind
        summary_lines.append(
            f"| `{kernel.name}` | `{match_kind}` "
            f"({kernel.match.spec_path.name if kernel.match.kind == 'linalg_dag' else '-'}) "
            f"| `{export}` → `{ctxbin_rel}` |"
        )
    summary_lines.append("\n## How matching works\n")
    summary_lines.append(
        "1. `iree-compile` runs the `transform_spec.mlir` interpreter via "
        "`--iree-preprocessing-transform-spec-filename=`. \n"
        "2. For each kernel, the `match_<name>` sequence applies "
        "`transform.iree.match.cast_compatible_dag_from_root` against the "
        "linalg-DAG body in `kernels/<kernel>/match.mlir`. Matches are "
        "structural — op chain, indexing maps, iterator types, arith body. \n"
        "3. The matcher returns `(%ins, %out)` to the `cast_and_call_<name>` "
        "sequence, which inserts a `transform.util.cast_and_call` to the "
        "wrapper `util.func @call_<name>` (also emitted into the spec). \n"
        "4. The wrapper's body is `flow.dispatch @kb_<name>::@<name>_dispatch` — "
        "the dispatch points at a `hal.executable.source` whose `objects` "
        "attribute names the `.qnn-ctx` blob you can find in "
        "`kernels/<kernel>/ctxbin.qnn-ctx`. \n"
        "5. The QNN target plugin (`compiler/plugins/target/QNN/QNNTarget.cpp`) "
        "reads `transform_spec.qnn_manifest.json` during `serializeExecutable` "
        "to embed the prebuilt blob as the executable binary."
    )
    (out / "summary.md").write_text("\n".join(summary_lines) + "\n")

    print(f"\n  inspection bundle: {out}")
    print(f"  summary:           {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
