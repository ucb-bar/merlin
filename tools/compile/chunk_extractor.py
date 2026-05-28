#!/usr/bin/env python3
"""Multi-granularity chunk extractor.

Generalises tools/compile/breakdown_vmfb.py from "one VMFB per
flow.dispatch.region" to "one chunk per (granularity-level) unit" where
unit ∈ {DISPATCH, LAYER, MEGAKERNEL, TILE}. The compiler-side TILE /
MEGAKERNEL transforms (split-by-id and grow-merged-region directives) live
in compiler/src/iree/compiler/DispatchCreation/ApplyScheduleDirectives.cpp
and are invoked here as a separate compile pass; this script only owns the
DISPATCH and LAYER aggregations on the post-dispatch-creation MLIR.

Model-agnostic.

Architecture:

    *.5.dispatch-creation.mlir
            │
            ▼
    parse_dispatch_creation()      # reused from breakdown_vmfb.py
    resolve_dependencies()
            │
            ▼
    aggregate_<level>(entries)     # this file: dispatch / layer aggregators
            │
            ▼
    chunk_manifest.json            # id, deps (chunk-ids), parent dispatch ids,
                                    # input/output tensor types, op_summary
            │
            ▼
    chunk_<id>.mlir per chunk      # MLIR fragment for downstream chunk_compile.py

Usage:
    python tools/compile/chunk_extractor.py extract \\
        --level=layer \\
        --input <phases_dir>/<model>.5.dispatch-creation.mlir \\
        --out  build/chunks/<model>/layer/

    python tools/compile/chunk_extractor.py verify \\
        build/chunks/<model>/layer/    # round-trip md5 check
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import re
import sys

# Reuse the SSA walker + dependency builder from breakdown_vmfb.py (sibling
# module inside the tools.compile package).
from compile.breakdown_vmfb import (  # noqa: E402
    DispatchEntry,
    parse_dispatch_creation,
    resolve_dependencies,
)

# Fallback regex for MLIRs that pre-date TagDispatchIdsPass. Captures both
# `flow.dispatch.region` (pre-conversion) and `flow.dispatch.workgroups`
# (post-conversion) forms. The id is assigned by parse order.
_RE_UNSTAMPED_DISPATCH_REGION = re.compile(
    r"%(?P<result>\S+)\s*=\s*flow\.dispatch\.region\b" r"(?P<head>[^{]*)" r"\{",
    re.MULTILINE,
)
_RE_UNSTAMPED_DISPATCH_WORKGROUPS = re.compile(
    r"%(?P<result>\S+)\s*=\s*flow\.dispatch\.workgroups"
    r"\((?P<operands>[^)]*)\)\s*:\s*"
    r"\((?P<intypes>.*?)\)\s*->\s*"
    r"(?P<outtypes>[^=]+?)\s*=",
    re.DOTALL,
)


def _entries_from_manifest(manifest_path: pathlib.Path) -> dict[str, DispatchEntry]:
    """Synthesise DispatchEntry records from a breakdown_vmfb-emitted
    manifest.json. The manifest already has op_summary, dependencies (as
    dispatch-name strings), inputs and outputs — everything LAYER
    aggregation needs."""
    payload = json.loads(manifest_path.read_text())
    src = payload.get("dispatches", {})
    out: dict[str, DispatchEntry] = {}
    for name, e in src.items():
        out[name] = DispatchEntry(
            name=name,
            id=int(e["id"]),
            subid=e.get("subid"),
            op_summary=e.get("op_summary", ""),
            module_name=e.get("module_name", ""),
            inputs=list(e.get("inputs", [])),
            outputs=list(e.get("outputs", [])),
            operand_ssa=[],  # not present in manifest; deps already resolved
            result_ssa=[],
            deps=list(e.get("dependencies", [])),
            benchmark_mlir=None,
            vmfb=None,
        )
    return out


def _parse_unstamped(mlir_path: pathlib.Path) -> dict[str, DispatchEntry]:
    """Parse a dispatch-creation MLIR that has no iree.dispatch_id markers.
    Assigns ids by parse order (mirrors how breakdown_vmfb manifest entries
    look on QRB5165 already)."""
    text = mlir_path.read_text()
    out: dict[str, DispatchEntry] = {}
    next_id = 0

    # Try workgroups form first (post-conversion).
    for m in _RE_UNSTAMPED_DISPATCH_WORKGROUPS.finditer(text):
        operand_list = [tok.strip() for tok in m.group("operands").split(",") if tok.strip()]
        in_types = [t.strip() for t in m.group("intypes").split(",") if t.strip()]
        out_field = m.group("outtypes").strip().rstrip(", ")
        out_types_raw = [t.strip() for t in out_field.split(",") if t.strip()]
        operand_to_type: dict[str, str] = {}
        if len(operand_list) == len(in_types):
            for op_ssa, ty in zip(operand_list, in_types):
                operand_to_type[op_ssa] = ty
        out_types: list[str] = []
        for ot in out_types_raw:
            ot = ot.split("=")[-1].strip()
            if ot.startswith("%"):
                ot = operand_to_type.get(ot, ot)
            out_types.append(ot)
        result_ssa = ["%" + m.group("result")]
        name = f"dispatch_{next_id}"
        out[name] = DispatchEntry(
            name=name,
            id=next_id,
            subid=None,
            op_summary="",
            module_name="",
            inputs=in_types,
            outputs=out_types,
            operand_ssa=operand_list,
            result_ssa=result_ssa,
            deps=[],
            benchmark_mlir=None,
            vmfb=None,
        )
        next_id += 1
    if out:
        return out

    # Fall back to region form (pre-conversion). The region form's
    # signature is harder to parse precisely (the body is a brace-balanced
    # block) so we just count regions and synthesise minimal entries with
    # operand/output ssa lifted from the head clause.
    for m in _RE_UNSTAMPED_DISPATCH_REGION.finditer(text):
        result_ssa = ["%" + m.group("result")]
        name = f"dispatch_{next_id}"
        out[name] = DispatchEntry(
            name=name,
            id=next_id,
            subid=None,
            op_summary="",
            module_name="",
            inputs=[],
            outputs=[],
            operand_ssa=[],
            result_ssa=result_ssa,
            deps=[],
            benchmark_mlir=None,
            vmfb=None,
        )
        next_id += 1
    return out


_LOG = logging.getLogger("chunk_extractor")

# A linalg op_summary lifted from IREE's executable-export naming convention.
# The exporter encodes the root op kind in the first underscore-delimited
# token (e.g. "matmul_512x512x512_f32", "conv_32x14x14x32x3x3_i8xi8xi32",
# "elementwise_3x112x112_f32xi8", "reduction_32x14x14x32_i8xi8xi32"). For
# LAYER aggregation we treat the op_summary's leading token as the family.
_FAMILY_ROOT = {"matmul", "conv", "reduction", "softmax"}
_FAMILY_FUSEABLE = {"elementwise", "fill", "reshape", "broadcast", "cast"}


def _family(op_summary: str) -> str:
    """Return the op family from an op_summary string."""
    if not op_summary:
        return "unknown"
    return op_summary.split("_", 1)[0].lower()


@dataclasses.dataclass
class ChunkEntry:
    """A schedulable unit at any granularity level. The `parent_dispatch_ids`
    field is the bridge back to the original dispatch graph — every chunk,
    no matter how aggregated, retains the list of dispatch ids it covers."""

    chunk_id: int
    op_summary: str
    parent_dispatch_ids: list[int]
    parent_dispatch_names: list[str]
    inputs: list[str]  # tensor types
    outputs: list[str]  # tensor types
    deps: list[int]  # other chunk ids this one depends on


def aggregate_dispatch(entries: dict[str, DispatchEntry]) -> list[ChunkEntry]:
    """Trivial aggregator: one chunk per dispatch."""
    name_order = sorted(entries.keys(), key=lambda n: entries[n].id)
    name_to_chunk: dict[str, int] = {n: i for i, n in enumerate(name_order)}
    chunks: list[ChunkEntry] = []
    for name in name_order:
        e = entries[name]
        deps_chunks = sorted({name_to_chunk[d] for d in e.deps if d in name_to_chunk})
        chunks.append(
            ChunkEntry(
                chunk_id=name_to_chunk[name],
                op_summary=e.op_summary or _family_summary_fallback(e),
                parent_dispatch_ids=[e.id],
                parent_dispatch_names=[name],
                inputs=list(e.inputs),
                outputs=list(e.outputs),
                deps=deps_chunks,
            )
        )
    return chunks


def _family_summary_fallback(e: DispatchEntry) -> str:
    """When op_summary wasn't filled in (no benchmark MLIR attached yet),
    synthesise something usable from the dispatch name + first input type."""
    if e.inputs:
        shape = e.inputs[0]
        return f"dispatch_{e.id}_{shape}"
    return f"dispatch_{e.id}"


def aggregate_megakernel(entries: dict[str, DispatchEntry]) -> list[ChunkEntry]:
    """MEGAKERNEL: a chunk is a maximal chain of dispatches with single-
    producer / single-consumer edges, regardless of family. Boundaries are
    cuts in the dependency DAG — a node with multiple successors or
    predecessors becomes a boundary.

    This is more aggressive than LAYER (which is rooted at one linalg op
    + its elementwise fusions). MEGAKERNEL collapses entire pipeline
    segments into one schedulable unit, trading scheduler flexibility for
    intra-megakernel cache locality."""
    name_order = _topo_sort(entries)
    consumers: dict[str, list[str]] = {n: [] for n in entries}
    for n, e in entries.items():
        for d in e.deps:
            if d in consumers:
                consumers[d].append(n)

    name_to_chunk: dict[str, int] = {}
    chunks: list[list[str]] = []
    for name in name_order:
        if name in name_to_chunk:
            continue
        # Start a new megakernel anchored at this node.
        cidx = len(chunks)
        chunks.append([name])
        name_to_chunk[name] = cidx
        # Greedily absorb downstream nodes as long as the chain has
        # in-degree 1 (single producer == us) and out-degree 1 (single
        # consumer that we'll absorb next iteration).
        cursor = name
        while True:
            succs = consumers[cursor]
            if len(succs) != 1:
                break
            succ = succs[0]
            if succ in name_to_chunk:
                break
            # Predecessor of succ must be exactly cursor (single
            # producer) — otherwise it's a join and we can't absorb.
            if entries[succ].deps != [cursor]:
                break
            chunks[cidx].append(succ)
            name_to_chunk[succ] = cidx
            cursor = succ

    return _materialise_chunks(entries, chunks, name_to_chunk)


def aggregate_tile(entries: dict[str, DispatchEntry], factor: int = 2) -> list[ChunkEntry]:
    """TILE granularity (cost-model only — no compiler-side splits).

    Each DISPATCH chunk becomes `factor` peer tiles. Each tile inherits the
    parent's deps + has cost = parent_cost / factor on each machine. The
    runtime can NOT actually run these tiles independently (no compiler
    splits exist), but the resulting schedule shows what the optimal would
    look like at finer-than-DISPATCH granularity. Useful as a what-if for
    motivating the compiler-side TILE work (PR 2 in the plan).

    The expand step (layer_schedule.py expand) collapses tile peers back
    to one DISPATCH execution on whichever machine the FIRST peer was
    assigned, so the on-board run still produces correct numerics.
    """
    parent_chunks = aggregate_dispatch(entries)
    tile_chunks: list[ChunkEntry] = []
    parent_tile_ids: dict[int, list[int]] = {}
    for parent in parent_chunks:
        ids = []
        for k in range(factor):
            new_id = len(tile_chunks)
            ids.append(new_id)
            tile_chunks.append(
                ChunkEntry(
                    chunk_id=new_id,
                    op_summary=f"{parent.op_summary}_tile{k}/{factor}",
                    parent_dispatch_ids=list(parent.parent_dispatch_ids),
                    parent_dispatch_names=list(parent.parent_dispatch_names),
                    inputs=list(parent.inputs),
                    outputs=list(parent.outputs),
                    deps=[],  # filled below in two passes
                )
            )
        parent_tile_ids[parent.chunk_id] = ids
    # Pass 2: cross-tile deps. For each parent's k-th tile, deps are the
    # k-th tiles of each parent dep (mirrors data flow: tile k of dispatch
    # X feeds tile k of dispatch Y).
    for parent in parent_chunks:
        for k, tile_id in enumerate(parent_tile_ids[parent.chunk_id]):
            tile_chunks[tile_id].deps = sorted({parent_tile_ids[d][k] for d in parent.deps if d in parent_tile_ids})
    return tile_chunks


def aggregate_layer(entries: dict[str, DispatchEntry]) -> list[ChunkEntry]:
    """LAYER granularity: a chunk is a single root linalg op + its
    elementwise / fill / reshape / broadcast / cast pre/post fusions.

    Algorithm:
      1. Topologically sort dispatches.
      2. Walk in topo order. For each unassigned dispatch:
         - If its family is a `_FAMILY_ROOT` (matmul/conv/reduction): start
           a new chunk, then greedily absorb downstream dispatches whose
           family is in `_FAMILY_FUSEABLE` AND whose only producer is the
           current chunk. Stop at any boundary (another root, multi-producer,
           reshape that re-types, or a dispatch consumed by something
           outside the chunk-being-built).
         - If its family is in `_FAMILY_FUSEABLE` and it has exactly one
           producer that's already in a chunk, attach it to that chunk
           (post-fusion).
         - Otherwise (orphan elementwise / unknown): give it its own chunk.

    Boundaries:
      - Another root linalg op (matmul/conv/reduction).
      - A dispatch that has multiple consumers — we don't replicate work.
      - A reshape / cast that changes the element type (reshape between
        compatible types is fine; a quant cast f32→i8 is a hard boundary).
      - A dispatch outside the chunk's transitive descendants.
    """
    name_order = _topo_sort(entries)
    name_to_chunk: dict[str, int] = {}
    chunks: list[list[str]] = []  # list of dispatch names per chunk

    # Pre-compute consumers per dispatch so we can detect multi-consumers.
    consumers: dict[str, list[str]] = {n: [] for n in entries}
    for n, e in entries.items():
        for d in e.deps:
            if d in consumers:
                consumers[d].append(n)

    for name in name_order:
        if name in name_to_chunk:
            continue
        e = entries[name]
        fam = _family(e.op_summary or "")
        if fam in _FAMILY_ROOT:
            # Start a fresh chunk anchored at this root.
            cidx = len(chunks)
            chunks.append([name])
            name_to_chunk[name] = cidx
            # Greedily absorb fuseable downstream dispatches.
            queue = list(consumers[name])
            while queue:
                cand = queue.pop(0)
                if cand in name_to_chunk:
                    continue
                ce = entries[cand]
                cfam = _family(ce.op_summary or "")
                if cfam not in _FAMILY_FUSEABLE:
                    continue
                # All deps must be in THIS chunk; otherwise it's a join
                # point with another logical layer and we stop.
                if any(name_to_chunk.get(d) != cidx for d in ce.deps):
                    continue
                # Cast that changes element type is a layer boundary.
                if cfam == "cast" and _changes_element_type(ce):
                    continue
                chunks[cidx].append(cand)
                name_to_chunk[cand] = cidx
                queue.extend(consumers[cand])
        elif fam in _FAMILY_FUSEABLE and len(e.deps) == 1 and e.deps[0] in name_to_chunk:
            # Post-fusion onto an already-built chunk.
            cidx = name_to_chunk[e.deps[0]]
            chunks[cidx].append(name)
            name_to_chunk[name] = cidx
        else:
            # Orphan — own chunk.
            cidx = len(chunks)
            chunks.append([name])
            name_to_chunk[name] = cidx

    return _materialise_chunks(entries, chunks, name_to_chunk)


def _changes_element_type(e: DispatchEntry) -> bool:
    """Heuristic: a cast dispatch whose input tensor type's element-type
    differs from its output tensor type's. Strings are like
    `tensor<3x114x114xi8>` → element type is the trailing token after `x`."""

    def _elt(t: str) -> str:
        # tensor<3x114x114xi8>  ->  i8
        if "<" in t and ">" in t:
            inside = t.split("<", 1)[1].rsplit(">", 1)[0]
            tail = inside.split("x")[-1]
            return tail
        return t

    if not e.inputs or not e.outputs:
        return False
    return _elt(e.inputs[0]) != _elt(e.outputs[0])


def _topo_sort(entries: dict[str, DispatchEntry]) -> list[str]:
    """Deterministic topo sort using dispatch ids as the secondary key."""
    in_deg = {n: 0 for n in entries}
    for n, e in entries.items():
        for d in e.deps:
            if d in entries:
                in_deg[n] += 1
    ready = sorted([n for n, d in in_deg.items() if d == 0], key=lambda n: entries[n].id)
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for c, ce in entries.items():
            if n in ce.deps:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    ready.append(c)
                    ready.sort(key=lambda x: entries[x].id)
    if len(out) != len(entries):
        raise RuntimeError(f"cycle detected: {len(entries) - len(out)} unsorted")
    return out


def _materialise_chunks(
    entries: dict[str, DispatchEntry],
    chunks: list[list[str]],
    name_to_chunk: dict[str, int],
) -> list[ChunkEntry]:
    """Convert per-chunk dispatch-name lists into ChunkEntry objects with
    correct cross-chunk deps, inputs, outputs."""
    out: list[ChunkEntry] = []
    for cidx, names in enumerate(chunks):
        member_set = set(names)
        # Inputs to the chunk = inputs of any member dispatch that come
        # from a non-member producer (or that have no in-graph producer).
        chunk_inputs: list[str] = []
        chunk_input_set: set[str] = set()
        for name in names:
            e = entries[name]
            for op_ssa, ty in zip(e.operand_ssa, e.inputs):
                op_root = op_ssa.strip().split()[0]
                # Find the producer dispatch (if any).
                producer = None
                for cand_name, cand_e in entries.items():
                    if op_root in cand_e.result_ssa:
                        producer = cand_name
                        break
                if producer is None or producer not in member_set:
                    key = f"{op_root}:{ty}"
                    if key not in chunk_input_set:
                        chunk_input_set.add(key)
                        chunk_inputs.append(ty)
        # Outputs = outputs of any member whose result is consumed outside
        # the chunk (or escapes to a return).
        chunk_outputs: list[str] = []
        for name in names:
            e = entries[name]
            for r_ssa, ty in zip(e.result_ssa, e.outputs):
                consumers_outside = []
                for cand_name, cand_e in entries.items():
                    if cand_name == name:
                        continue
                    for op_ssa in cand_e.operand_ssa:
                        if op_ssa.strip().split()[0] == r_ssa:
                            if cand_name not in member_set:
                                consumers_outside.append(cand_name)
                            break
                # If no in-graph consumer at all, treat as escaping (return value).
                if (not consumers_outside) or any(c not in member_set for c in consumers_outside):
                    chunk_outputs.append(ty)
        # Cross-chunk deps.
        deps: list[int] = []
        seen_deps: set[int] = set()
        for name in names:
            e = entries[name]
            for d in e.deps:
                d_chunk = name_to_chunk.get(d)
                if d_chunk is not None and d_chunk != cidx and d_chunk not in seen_deps:
                    seen_deps.add(d_chunk)
                    deps.append(d_chunk)
        # op_summary of the LAYER chunk: the root linalg's summary if
        # present, otherwise concat of members.
        summary = "+".join(
            entries[n].op_summary or _family_summary_fallback(entries[n])
            for n in names
            if _family(entries[n].op_summary or "") in _FAMILY_ROOT
        )
        if not summary:
            summary = "+".join(entries[n].op_summary or _family_summary_fallback(entries[n]) for n in names)

        out.append(
            ChunkEntry(
                chunk_id=cidx,
                op_summary=summary[:120],
                parent_dispatch_ids=[entries[n].id for n in names],
                parent_dispatch_names=list(names),
                inputs=chunk_inputs,
                outputs=chunk_outputs,
                deps=sorted(deps),
            )
        )
    return out


def write_chunk_manifest(
    chunks: list[ChunkEntry], out_path: pathlib.Path, source_mlir: pathlib.Path, level: str
) -> None:
    payload = {
        "schema_version": 1,
        "source": "tools/chunk_extractor.py",
        "source_mlir": str(source_mlir),
        "level": level,
        "num_chunks": len(chunks),
        "chunks": {f"chunk_{c.chunk_id}": dataclasses.asdict(c) for c in chunks},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def cmd_extract(args: argparse.Namespace) -> int:
    if args.from_manifest:
        # Build entries from an existing breakdown manifest.json (which
        # already has op_summary, deps as dispatch-name strings, inputs,
        # outputs). This is the right source for LAYER aggregation since
        # op_summary is the family discriminator.
        entries = _entries_from_manifest(args.from_manifest)
    else:
        entries = parse_dispatch_creation(args.input)
        if not entries and args.assign_ids_by_order:
            entries = _parse_unstamped(args.input)
        if entries:
            resolve_dependencies(entries)

    if not entries:
        _LOG.error(
            "no dispatches available. Pass either --input <dispatch-creation.mlir> "
            "(with iree.dispatch_id stamped, or with --assign-ids-by-order) "
            "or --from-manifest <breakdowns/manifest.json>."
        )
        return 2

    if args.level == "dispatch":
        chunks = aggregate_dispatch(entries)
    elif args.level == "layer":
        chunks = aggregate_layer(entries)
    elif args.level == "megakernel":
        chunks = aggregate_megakernel(entries)
    elif args.level == "tile":
        # PR 2-lite: cost-model only, no compiler-side splits. Each
        # dispatch becomes `factor` tile peers in the cost matrix.
        chunks = aggregate_tile(entries, factor=args.tile_factor)
    else:
        _LOG.error("unknown granularity %s", args.level)
        return 2

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_chunk_manifest(chunks, out_dir / "chunk_manifest.json", source_mlir=args.input, level=args.level)
    _LOG.info("level=%s: %d dispatches -> %d chunks", args.level, len(entries), len(chunks))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verification gate §A: confirm the chunk_manifest.json is internally
    consistent. The byte-equal-stitched-back round-trip lives in
    chunk_compile.py (PR 3) since that's where actual VMFBs are produced;
    here we only verify the manifest's graph properties."""
    mp = args.chunk_dir / "chunk_manifest.json"
    if not mp.exists():
        _LOG.error("missing %s", mp)
        return 2
    payload = json.loads(mp.read_text())
    chunks = payload["chunks"]
    # Every chunk's deps must reference an existing chunk id.
    ids = {int(name.split("_", 1)[1]) for name in chunks.keys()}
    failed = 0
    for name, c in chunks.items():
        for d in c["deps"]:
            if d not in ids:
                _LOG.error("%s: dep %d not in chunk set", name, d)
                failed += 1
    # Every parent dispatch id must be unique across chunks (no double-cover).
    seen_dispatch: dict[int, str] = {}
    for name, c in chunks.items():
        for did in c["parent_dispatch_ids"]:
            if did in seen_dispatch:
                _LOG.error("dispatch id %d in both %s and %s — double-cover!", did, seen_dispatch[did], name)
                failed += 1
            else:
                seen_dispatch[did] = name
    # Every chunk must have at least one input or no deps (root chunks).
    for name, c in chunks.items():
        if not c["deps"] and not c["inputs"]:
            _LOG.warning("%s has neither deps nor inputs — orphan chunk?", name)
    # Cycle detection in chunk-deps.
    in_deg = {int(n.split("_", 1)[1]): 0 for n in chunks}
    for name, c in chunks.items():
        in_deg[int(name.split("_", 1)[1])] = len(c["deps"])
    ready = [k for k, v in in_deg.items() if v == 0]
    visited = 0
    while ready:
        cur = ready.pop(0)
        visited += 1
        for name, c in chunks.items():
            cid = int(name.split("_", 1)[1])
            if cur in c["deps"]:
                in_deg[cid] -= 1
                if in_deg[cid] == 0:
                    ready.append(cid)
    if visited != len(chunks):
        _LOG.error("cycle detected in chunk deps (%d/%d visited)", visited, len(chunks))
        failed += 1
    _LOG.info("verify: %d chunks, %d errors", len(chunks), failed)
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s_ext = sub.add_parser("extract", help="extract chunks from a " "dispatch-creation MLIR")
    s_ext.add_argument("--level", choices=["dispatch", "layer", "megakernel", "tile"], default="dispatch")
    s_ext.add_argument(
        "--input",
        type=pathlib.Path,
        default=None,
        help="post-DispatchCreation MLIR (the *.5." "dispatch-creation.mlir output of merlin compile).",
    )
    s_ext.add_argument(
        "--from-manifest",
        type=pathlib.Path,
        default=None,
        help="Use an existing breakdown manifest.json as the "
        "entries source instead of re-parsing MLIR. "
        "Right for LAYER aggregation since op_summary "
        "is already lifted from benchmark filenames.",
    )
    s_ext.add_argument("--out", required=True, type=pathlib.Path)
    s_ext.add_argument("--tile-factor", type=int, default=2, help="Tile peers per dispatch when --level=tile.")
    s_ext.add_argument(
        "--assign-ids-by-order",
        action="store_true",
        default=True,
        help="When the input MLIR has no iree.dispatch_id "
        "(i.e. the model was compiled before "
        "TagDispatchIdsPass landed), assign chunk ids by "
        "parse order. Default ON since most existing "
        "compiled artefacts predate the pass.",
    )
    s_ext.add_argument("-v", "--verbose", action="store_true")
    s_ext.set_defaults(func=cmd_extract)

    s_ver = sub.add_parser("verify", help="sanity-check a generated chunk " "manifest")
    s_ver.add_argument("chunk_dir", type=pathlib.Path)
    s_ver.add_argument("-v", "--verbose", action="store_true")
    s_ver.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
