#!/usr/bin/env python3
"""Per-breakdown VMFB emitter.

Compiles a model with `--dump-compilation-phases-to` (so we get the
post-DispatchCreation MLIR with `iree.dispatch_id` stamps) and
`--iree-hal-dump-executable-benchmarks-to` (so IREE produces a self-contained
per-dispatch benchmark MLIR), then for each dispatch:

    1. Re-compiles the per-dispatch benchmark MLIR into its own .vmfb (so the
       scheduler can hand it to runtime without going through the full
       compile pipeline at scheduling time).
    2. Extracts input + output tensor shapes by parsing the dispatch's
       `iree_tensor_ext.dispatch.tensor<...>` binding signatures.
    3. Records dependencies by walking SSA edges between
       `flow.dispatch.workgroups` ops in the dispatch-creation phase.

Output layout:

    <out>/breakdowns/manifest.json        # XPU-RT-compatible dispatch_deps schema
    <out>/breakdowns/<name>.vmfb          # one VMFB per dispatch (or split-half)
    <out>/breakdowns/<name>.shapes.json   # input/output tensor types

The manifest is structurally compatible with XPU-RT's
`<model>_dispatch_deps.json` schema, plus an `executable` block per dispatch
that points at the per-dispatch VMFB and the shape file. This lets the
XPU-RT data-generation scripts pick up realistic tensor shapes without
parsing MLIR.

Model-agnostic, target-agnostic.

Usage:

    ./merlin compile <model.mlir> --target <yaml> [--with-schedule <s.json>] \
        --output-dir <out> --build-dir <bd> --dump-phases \
        --iree-compile-arg=--iree-hal-dump-executable-benchmarks-to=<out>/benchmarks
    python tools/compile/breakdown_vmfb.py \
        --output-dir <out> \
        --target <yaml> \
        --build-dir <bd>

Or just `tools/compile/breakdown_vmfb.py --orchestrate ...` to do both steps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable

_LOG = logging.getLogger("breakdown_vmfb")

# Regexes for parsing the post-DispatchCreation MLIR. We deliberately don't
# bind to a real MLIR parser — the surface we need (op name, shape, id,
# subid, SSA operands/result) is stable text and the alternative is loading
# the MLIR Python bindings which adds a heavy dep for a thin orchestrator.

_RE_WORKGROUPS = re.compile(
    r"%(?P<result>\S+)\s*=\s*flow\.dispatch\.workgroups"
    r"\((?P<operands>[^)]*)\)\s*:\s*\((?P<intypes>[^)]*)\)\s*->\s*"
    r"(?P<outtypes>[^\n{]+?)(?:\s+attributes\s*\{(?P<attrs>[^}]*)\})?\s*=",
    re.MULTILINE,
)
_RE_REGION = re.compile(
    r"%(?P<result>\S+)\s*=\s*flow\.dispatch\.region\s*->\s*\((?P<outtypes>[^)]+)\)"
    r"\s*attributes\s*\{(?P<attrs>[^}]*)\}",
    re.MULTILINE,
)
_RE_DISPATCH_ID = re.compile(r"iree\.dispatch_id\s*=\s*(\d+)\s*:\s*i64")
_RE_DISPATCH_SUBID = re.compile(r"iree\.dispatch_subid\s*=\s*(\d+)\s*:\s*i64")
_RE_TENSOR = re.compile(r"tensor<([^>]+)>")
# Generic SSA def: `%name = op_name ...`. Captures the LHS name and the rest
# of the line so we can pull operand SSA names from arbitrary ops (used to
# walk back through flow.tensor.reshape / flow.tensor.encode / iree_encoding
# passthrough ops to find the originating dispatch).
_RE_SSA_DEF = re.compile(
    r"^\s*%(?P<name>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.*)$",
    re.MULTILINE,
)
_RE_OPERAND_REF = re.compile(r"%([A-Za-z0-9_]+)")
# Benchmark-MLIR filename produced by --iree-hal-dump-executable-benchmarks-to:
#   module_<func>$async_dispatch_<N>_<variant>_benchmark.mlir
_RE_BENCH_FILENAME = re.compile(
    r"^module_(?P<func>[^_]+(?:_[^_]+)*?)\$async_dispatch_(?P<id>\d+)"
    r"(?:_(?P<variant>[^_]+(?:_[^_]+)*))?_benchmark\.mlir$"
)


@dataclasses.dataclass
class DispatchEntry:
    name: str  # "dispatch_<id>" or "dispatch_<id>_<subid>"
    id: int
    subid: int | None
    op_summary: str  # e.g. "elementwise_3x112x112_f32xi8" lifted from the export name
    module_name: str  # full export name, e.g. "dronet$async_dispatch_0_elementwise_3x112x112_f32xi8"
    inputs: list[str]  # tensor type strings like "tensor<3x112x112xf32>"
    outputs: list[str]
    operand_ssa: list[str]  # SSA names of inputs (for dep graph)
    result_ssa: list[str]
    deps: list[str]  # populated after the SSA→name resolution pass
    benchmark_mlir: pathlib.Path | None
    vmfb: pathlib.Path | None


def _parse_attrs(attrs: str) -> tuple[int | None, int | None]:
    id_match = _RE_DISPATCH_ID.search(attrs)
    sub_match = _RE_DISPATCH_SUBID.search(attrs)
    return (
        int(id_match.group(1)) if id_match else None,
        int(sub_match.group(1)) if sub_match else None,
    )


def parse_dispatch_creation(mlir_path: pathlib.Path) -> dict[str, DispatchEntry]:
    """Walk the post-DispatchCreation MLIR text. Indexes by SSA result name
    so the dependency-resolution pass can map producer SSA -> consumer
    DispatchEntry directly.

    NOTE: we read the *region* form rather than the *workgroups* form
    because TagDispatchIdsPass stamps regions and the conversion-pass patch
    propagates onto workgroups; either is fine, but at the dispatch-creation
    phase boundary IREE produces workgroups. We accept either by regex.
    """
    text = mlir_path.read_text()
    out: dict[str, DispatchEntry] = {}

    # Workgroups form (post ConvertDispatchRegionsToWorkgroups).
    for m in _RE_WORKGROUPS.finditer(text):
        attrs_block = m.group("attrs") or ""
        did, sub = _parse_attrs(attrs_block)
        if did is None:
            continue
        name = f"dispatch_{did}" + (f"_{sub}" if sub is not None else "")
        operand_list = [tok.strip() for tok in m.group("operands").split(",") if tok.strip()]
        in_types = [t.strip() for t in m.group("intypes").split(",") if t.strip()]
        # The output field uses one of two forms:
        #   -> tensor<...>            # fresh result tensor
        #   -> %N                     # tied result (in-place; reuses operand
        #                              # %N's type)
        # We normalize tied results back to their operand's tensor type so
        # the manifest's `outputs` field is uniformly tensor types.
        out_field = m.group("outtypes").strip().rstrip(", ")
        out_types_raw = [t.strip() for t in out_field.split(",") if t.strip()]
        operand_to_type: dict[str, str] = {}
        if len(operand_list) == len(in_types):
            for op_ssa, ty in zip(operand_list, in_types):
                operand_to_type[op_ssa] = ty
        out_types: list[str] = []
        for ot in out_types_raw:
            ot = ot.split("=")[-1].strip()  # drop `%foo =` if present
            if ot.startswith("%"):
                ot = operand_to_type.get(ot, ot)
            out_types.append(ot)
        result_ssa = ["%" + m.group("result")]
        out[name] = DispatchEntry(
            name=name,
            id=did,
            subid=sub,
            op_summary="",  # filled later from benchmark filename
            module_name="",  # filled later from benchmark export name
            inputs=in_types,
            outputs=out_types,
            operand_ssa=operand_list,
            result_ssa=result_ssa,
            deps=[],
            benchmark_mlir=None,
            vmfb=None,
        )
    return out


def _build_ssa_def_graph(
    mlir_text: str,
) -> dict[str, list[str]]:
    """Map every func-body SSA value name to the list of SSA operand names
    referenced on its defining RHS. We restrict to the function-body
    indentation level (4 leading spaces) so we don't conflate inner
    `flow.dispatch.workgroups` region SSA names (6+ space indent) with the
    outer values they happen to share names with.

    For passthrough ops like `flow.tensor.reshape`, `flow.tensor.encode`,
    and `iree_encoding.*`, this records the operand chain that lets the
    dependency resolver walk back to the originating dispatch.
    """
    defs: dict[str, list[str]] = {}
    func_body_prefix = "    %"  # exactly 4 leading spaces
    for line in mlir_text.splitlines():
        if not line.startswith(func_body_prefix):
            continue
        m = _RE_SSA_DEF.match(line)
        if not m:
            continue
        name = m.group("name")
        rhs = m.group("rhs")
        rhs_no_attrs = rhs.split("attributes")[0]
        refs = _RE_OPERAND_REF.findall(rhs_no_attrs)
        refs = [r for r in refs if r != name]
        # Don't overwrite — first def at func-body level wins. (SSA is
        # single-assignment so duplicate matches would only happen if we
        # accidentally captured an inner-region line, which the indent
        # filter already excludes.)
        defs.setdefault(name, refs)
    return defs


def resolve_dependencies(entries: dict[str, DispatchEntry], mlir_text: str | None = None) -> None:
    """Populate `deps`: for each dispatch, find which other dispatches
    produce its operands. Operands flowing through non-dispatch passthrough
    ops (flow.tensor.reshape, flow.tensor.encode, iree_encoding.*, etc.)
    are recursively traced back to the originating dispatch.

    `mlir_text`, when provided, is the post-DispatchCreation MLIR used to
    build the SSA def graph for the recursive walk. Without it, we fall
    back to the legacy direct-edge resolution (kept for compatibility but
    will under-report deps when reshape/encode ops sit between dispatches).
    """
    ssa_to_name: dict[str, str] = {}
    for entry in entries.values():
        for ssa in entry.result_ssa:
            # `entry.result_ssa` items look like "%22"; strip the leading %.
            key = ssa.lstrip("%")
            ssa_to_name[key] = entry.name

    ssa_defs: dict[str, list[str]] = _build_ssa_def_graph(mlir_text) if mlir_text else {}

    def find_producer(name: str, visited: set[str]) -> str | None:
        if name in visited:
            return None
        visited.add(name)
        if name in ssa_to_name:
            return ssa_to_name[name]
        # Walk back through the def graph: this %name was defined by some
        # passthrough op; recurse into each of its operands until we hit a
        # dispatch result or run out of edges (func arg / global / cst).
        for ref in ssa_defs.get(name, []):
            producer = find_producer(ref, visited)
            if producer is not None:
                return producer
        return None

    for entry in entries.values():
        seen: set[str] = set()
        for ssa in entry.operand_ssa:
            ssa = ssa.strip()
            ssa_root = ssa.split()[0]
            if not ssa_root.startswith("%"):
                continue
            ssa_name = ssa_root.lstrip("%")
            producer = find_producer(ssa_name, set())
            if producer and producer != entry.name and producer not in seen:
                entry.deps.append(producer)
                seen.add(producer)


def attach_benchmark_mlirs(entries: dict[str, DispatchEntry], benchmarks_dir: pathlib.Path) -> None:
    """Match per-dispatch benchmark.mlir files to DispatchEntry by id."""
    if not benchmarks_dir.exists():
        return
    for path in sorted(benchmarks_dir.iterdir()):
        m = _RE_BENCH_FILENAME.match(path.name)
        if not m:
            continue
        did = int(m.group("id"))
        # Find matching entry. If there are subids (split case) the
        # benchmark file currently doesn't carry the subid; pick the entry
        # without subid first, else first subid match.
        candidates = [e for e in entries.values() if e.id == did]
        if not candidates:
            continue
        primary = next((e for e in candidates if e.subid is None), candidates[0])
        primary.benchmark_mlir = path
        # Lift op summary from the filename (everything after the variant).
        text = path.read_text()
        export_match = re.search(r"hal\.executable\.export public @(\S+) ordinal", text)
        if export_match:
            full = export_match.group(1)
            primary.module_name = full
            # "main_graph$async_dispatch_0_elementwise_3x112x112_f32xi8" →
            # "elementwise_3x112x112_f32xi8"
            primary.op_summary = re.sub(r"^[^$]+\$async_dispatch_\d+_", "", full)


def compile_per_dispatch_vmfbs(
    entries: dict[str, DispatchEntry],
    target_flags: list[str],
    iree_compile: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """Take each per-dispatch benchmark.mlir and compile it independently
    into its own .vmfb. The benchmark MLIR is already a self-contained
    module with the executable inlined and a top-level `util.func` to drive
    it, so we just need to pass it back through iree-compile."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries.values():
        if entry.benchmark_mlir is None:
            continue
        vmfb = out_dir / f"{entry.name}.vmfb"
        cmd = [str(iree_compile), str(entry.benchmark_mlir), "-o", str(vmfb), *target_flags]
        _LOG.info("compiling %s -> %s", entry.name, vmfb)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            _LOG.warning("compile of %s failed: %s", entry.name, res.stderr.strip())
            continue
        entry.vmfb = vmfb


def write_shape_files(entries: Iterable[DispatchEntry], out_dir: pathlib.Path) -> None:
    """One shapes.json per dispatch — input/output tensor types in MLIR
    notation so a downstream data-generator script knows what to fabricate.

    Format:
        {
          "name": "dispatch_<id>",
          "id": 0,
          "subid": null,
          "op_summary": "elementwise_3x112x112_f32xi8",
          "inputs":  ["tensor<3x112x112xf32>"],
          "outputs": ["tensor<3x114x114xi8>"]
        }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = out_dir / f"{entry.name}.shapes.json"
        path.write_text(
            json.dumps(
                {
                    "name": entry.name,
                    "id": entry.id,
                    "subid": entry.subid,
                    "op_summary": entry.op_summary,
                    "inputs": entry.inputs,
                    "outputs": entry.outputs,
                },
                indent=2,
            )
            + "\n"
        )


def write_manifest(entries: dict[str, DispatchEntry], out_path: pathlib.Path) -> None:
    """XPU-RT-compatible dispatch_deps + per-dispatch executable pointers.

    Mirrors XPU-RT/xpu-rt/pytorch_workload/samples/<m>_dispatch_deps.json's
    `dispatches: { name: { id, ordinal, total, dependencies } }` shape, plus
    extra fields the scheduler can use directly: `executable` (vmfb path),
    `shapes` (shapes.json path), `op_summary` (lifted from the IREE export
    name).
    """
    # Group by id so we can compute ordinal/total per (split) family.
    by_id: dict[int, list[DispatchEntry]] = {}
    for e in entries.values():
        by_id.setdefault(e.id, []).append(e)
    for fam in by_id.values():
        fam.sort(key=lambda e: (e.subid is None, e.subid or 0))

    dispatches_block = {}
    for fam in by_id.values():
        total = len(fam)
        for ordinal, e in enumerate(fam, start=1):
            dispatches_block[e.name] = {
                "id": e.id,
                "subid": e.subid,
                "ordinal": ordinal,
                "total": total,
                "dependencies": e.deps,
                "op_summary": e.op_summary,
                "module_name": e.module_name,
                "inputs": e.inputs,
                "outputs": e.outputs,
                "executable": str(e.vmfb) if e.vmfb else None,
                "shapes": (str(out_path.parent / f"{e.name}.shapes.json") if e.vmfb else None),
            }
    payload = {
        "schema_version": 1,
        "source": "merlin/tools/breakdown_vmfb.py",
        "num_dispatches": len(dispatches_block),
        "dispatches": dispatches_block,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


_SOURCE_LANG_EXT = {
    "c": "c",
    "cc": "cc",
    "cuda": "cu",
    "glsl": "comp",
    "cl": "cl",
    "spirv": "spv",
    "hexagon-c": "c",
    "qnn-context-binary": "bin",
}


def _kernel_signature(entry: DispatchEntry) -> dict:
    """Build a manifest signature {operands: [{role, tensor}, ...]} from
    a DispatchEntry's inputs/outputs lists."""
    operands = [{"role": "in", "tensor": t} for t in entry.inputs]
    operands += [{"role": "out", "tensor": t} for t in entry.outputs]
    return {"operands": operands}


def _emit_match_mlir(entry: DispatchEntry, out_path: pathlib.Path) -> None:
    """Emit a SKELETON match.mlir fragment for `entry`. The user fills in
    the `transform.iree.match.cast_compatible_dag_from_root` body once they
    decide what part of the dispatch they're matching (single linalg op,
    a fused chain, a tile, etc.). The skeleton encodes the input/output
    types so the cast_compatible match has the right operand schema."""
    in_types = ", ".join(entry.inputs) if entry.inputs else ""
    out_types = ", ".join(entry.outputs) if entry.outputs else ""
    body = (
        "// Auto-scaffolded by tools/breakdown_vmfb.py --emit-kernel-manifest.\n"
        "// Fill in the dag-from-root body below to match the dispatch's\n"
        f"// root linalg op. Inputs: {in_types}.  Outputs: {out_types}.\n"
        "transform.named_sequence @match_KERNEL_NAME(\n"
        "    %root: !transform.any_op {transform.readonly}\n"
        ") -> (!transform.any_value, !transform.any_value) {\n"
        "  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root\n"
        "      %root {\n"
        "    // TODO: replace with the matched linalg dag.\n"
        "    // Example for a 2D matmul:\n"
        "    //   ^bb0(%lhs: tensor<MxKxf32>, %rhs: tensor<KxNxf32>,\n"
        "    //        %init: tensor<MxNxf32>):\n"
        "    //     %0 = linalg.matmul ins(%lhs, %rhs : ...)\n"
        "    //                        outs(%init : tensor<MxNxf32>)\n"
        "    //                        -> tensor<MxNxf32>\n"
        "    //     transform.yield %0 : tensor<MxNxf32>\n"
        "  } : !transform.any_op -> (!transform.any_value, !transform.any_value)\n"
        "  transform.yield %ins, %outs : !transform.any_value, !transform.any_value\n"
        "}\n"
    )
    out_path.write_text(body)


def emit_kernel_manifest(
    entries: dict[str, DispatchEntry],
    out_dir: pathlib.Path,
    source_lang: str,
    targets: list[str],
) -> pathlib.Path:
    """Auto-scaffold a kernel-manifest tree under `out_dir`. Returns the
    manifest.json path. Does NOT overwrite existing files (if a user has
    started filling in kernels, we preserve their edits and only top-up
    the manifest entries themselves)."""
    out_dir = out_dir.resolve()
    src_dir = out_dir / "src"
    match_dir = out_dir / "match"
    src_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    ext = _SOURCE_LANG_EXT.get(source_lang, source_lang)
    kernels: list[dict] = []
    for name, entry in sorted(entries.items()):
        src_path = src_dir / f"{name}.{ext}"
        match_path = match_dir / f"{name}.match.mlir"
        if not src_path.exists():
            placeholder = (
                f"// {name} ({entry.op_summary})\n"
                f"// Auto-scaffolded by breakdown_vmfb.py --emit-kernel-manifest.\n"
                f"// Inputs:  {entry.inputs}\n"
                f"// Outputs: {entry.outputs}\n"
                f"// Dispatch id: {entry.id}"
                + (f", subid: {entry.subid}\n" if entry.subid is not None else "\n")
                + "// Replace this stub with the actual kernel body.\n"
            )
            src_path.write_text(placeholder)
        if not match_path.exists():
            _emit_match_mlir(entry, match_path)
        kernels.append(
            {
                "name": name,
                "source": f"src/{src_path.name}",
                "source_lang": source_lang,
                "entry_symbol": name,
                "signature": _kernel_signature(entry),
                "match": {
                    "kind": "linalg_dag",
                    "spec_path": f"match/{match_path.name}",
                },
                "targets": list(targets),
                "merlin_dispatch": {
                    "id": entry.id,
                    "subid": entry.subid,
                    "op_summary": entry.op_summary,
                    "module_name": entry.module_name,
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "kernels": kernels,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# Auto-scaffolded kernel manifest\n\n"
            "This directory was generated by `tools/breakdown_vmfb.py "
            "--emit-kernel-manifest <dir>`. It contains:\n\n"
            "- `manifest.json` — one entry per dispatch in the source model.\n"
            "- `match/<name>.match.mlir` — placeholder match-bodies (fill\n"
            "  the `cast_compatible_dag_from_root` block to identify which\n"
            "  region of the dispatch you're substituting).\n"
            "- `src/<name>.<ext>` — empty source stubs.\n\n"
            "Workflow:\n"
            "1. Open the source stub for the dispatch you want to override.\n"
            "2. Implement the kernel body (or paste in a Claude/KernelBlaster "
            "output).\n"
            "3. Fill in the match.mlir's `cast_compatible_dag_from_root` "
            "body to match the dispatch's root linalg op.\n"
            "4. Pass `--kernels-dir <this-dir>` to `./merlin compile` to "
            "embed.\n"
        )

    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit one VMFB + shape JSON per dispatch (incl. splits) " "from a compiled model output directory."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="The same --output-dir passed to `./merlin compile`. Must "
        "already contain phases/ and benchmarks/ directories.",
    )
    parser.add_argument(
        "--iree-compile",
        type=pathlib.Path,
        help="Path to iree-compile binary (defaults to " "build/host-vanilla-debug/tools/iree-compile if unset).",
    )
    parser.add_argument(
        "--target-flag",
        action="append",
        default=[],
        help="Extra flag forwarded to iree-compile per-dispatch. Repeatable. "
        "If omitted, the per-dispatch benchmarks compile with their "
        "embedded #hal.device.target attribute and need no extra "
        "flags.",
    )
    parser.add_argument(
        "--phase-stem",
        default=None,
        help="Override the phase MLIR stem (default: " "auto-detect from phases/*.5.dispatch-creation.mlir).",
    )
    parser.add_argument(
        "--emit-kernel-manifest",
        type=pathlib.Path,
        default=None,
        help="Auto-scaffold a kernel-manifest tree for "
        "external-kernel embedding. For every dispatch, "
        "emits manifest.json + match/<name>.match.mlir "
        "+ src/<name>.<ext> stubs the user fills in.",
    )
    parser.add_argument(
        "--kernel-source-lang",
        default="c",
        help="source_lang for emitted manifest entries "
        "(c|cuda|glsl|cl|qnn-context-binary). "
        "Determines the source-stub extension.",
    )
    parser.add_argument(
        "--kernel-targets",
        action="append",
        default=[],
        help="HAL target string(s) to attach to each " "manifest entry (e.g. llvm-cpu-x86_64). " "Repeatable.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    out_dir = args.output_dir.resolve()
    phases_dir = out_dir / "phases"
    benchmarks_dir = out_dir / "benchmarks"
    breakdowns_dir = out_dir / "breakdowns"

    if not phases_dir.exists():
        _LOG.error("missing %s — re-run merlin compile with --dump-phases", phases_dir)
        return 1

    if args.phase_stem:
        dispatch_phase = phases_dir / args.phase_stem
    else:
        candidates = sorted(phases_dir.glob("*.5.dispatch-creation.mlir"))
        if not candidates:
            _LOG.error("no *.5.dispatch-creation.mlir found in %s", phases_dir)
            return 1
        dispatch_phase = candidates[0]
    _LOG.info("parsing %s", dispatch_phase)
    mlir_text = dispatch_phase.read_text()
    entries = parse_dispatch_creation(dispatch_phase)
    if not entries:
        _LOG.error("no dispatches found in %s — was iree.dispatch_id stamped?", dispatch_phase)
        return 1
    resolve_dependencies(entries, mlir_text=mlir_text)
    attach_benchmark_mlirs(entries, benchmarks_dir)
    _LOG.info(
        "indexed %d dispatches; %d have benchmark wrappers",
        len(entries),
        sum(1 for e in entries.values() if e.benchmark_mlir),
    )

    iree_compile = (
        args.iree_compile
        or pathlib.Path(__file__).resolve().parent.parent / "build" / "host-vanilla-debug" / "tools" / "iree-compile"
    )
    if not iree_compile.exists():
        _LOG.error("iree-compile not found at %s — pass --iree-compile", iree_compile)
        return 1
    compile_per_dispatch_vmfbs(entries, args.target_flag, iree_compile, breakdowns_dir)
    write_shape_files(entries.values(), breakdowns_dir)
    manifest_path = breakdowns_dir / "manifest.json"
    write_manifest(entries, manifest_path)
    _LOG.info("manifest -> %s", manifest_path)
    _LOG.info("VMFBs -> %s", breakdowns_dir)
    n_built = sum(1 for e in entries.values() if e.vmfb)
    _LOG.info("compiled %d/%d per-dispatch VMFBs", n_built, len(entries))

    if args.emit_kernel_manifest is not None:
        targets = list(args.kernel_targets)
        if not targets:
            # Best-guess default: derive from --target-flag values that
            # look like HAL targets. Otherwise leave empty so the user
            # is forced to set them when invoking ./merlin compile.
            targets = []
        kmanifest = emit_kernel_manifest(
            entries,
            args.emit_kernel_manifest,
            source_lang=args.kernel_source_lang,
            targets=targets,
        )
        _LOG.info(
            "kernel manifest -> %s (source_lang=%s, %d entries)", kmanifest, args.kernel_source_lang, len(entries)
        )

    return 0 if n_built > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
