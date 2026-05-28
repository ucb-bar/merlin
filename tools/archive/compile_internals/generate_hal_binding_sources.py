#!/usr/bin/env python3
"""Generate HAL-aware binding-sources for the dispatch-flow runner.

The flow-phase extractor (tools/extract_flow_dispatch_call_graph.py) maps
flow.dispatch operands 1:1 to HAL bindings. That breaks after the HAL
phase packs multiple flow operands into shared storage_buffer arenas:
flow operand 1 may end up at HAL binding 2 (or share a binding with
operand 0). To get a binding-source mapping that matches the manifest's
actual HAL binding_byte_sizes, we classify each binding by size instead:

  - size == constant-arena file size   -> file:<remote-constant-arena>
  - input-image size on the FIRST       -> input
    dispatch (root of the topology)
  - any other non-constant binding,     -> pred:<latest-dep>:0
    not the last one
  - the LAST non-constant binding       -> zero  (dispatch output)

`<latest-dep>` is the dependency with the largest dispatch index among
the call's `dependencies` list (i.e., the most-recent upstream node).
For yolov8 this matches the HAL phase's actual arena reuse.

Outputs binding_sources.json + a copy alongside schedule.json/manifest.json
in --out-dir; the dispatch-flow runner consumes the JSON via
run_on_board_flow.py --binding-sources-json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

_CALL_INDEX_RE = re.compile(r"dispatch_(\d+)_call_")


def _call_index(name: str) -> int:
    m = _CALL_INDEX_RE.search(name)
    return int(m.group(1)) if m else -1


def _latest_dep(deps: list[str]) -> str | None:
    if not deps:
        return None
    return max(deps, key=_call_index)


def derive(
    *,
    manifest: dict,
    call_graph: dict,
    constant_arena_size: int,
    input_size: int,
    remote_constant_arena: str,
    root_dispatch: str,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    dispatch_graph = manifest.get("dispatch_graph", {})
    for name, entry in manifest["dispatches"].items():
        cell = entry.get("cpu") or next(iter(entry.values()))
        sizes = cell["binding_byte_sizes"]
        deps = list(dispatch_graph.get(name, {}).get("dependencies", []))
        latest = _latest_dep(deps)
        # First pass: mark every constant-sized binding as file:.
        sources: list[str | None] = [None] * len(sizes)
        for i, sz in enumerate(sizes):
            if sz == constant_arena_size:
                sources[i] = f"file:{remote_constant_arena}"
        # Locate the input + output slots among the non-constant bindings.
        non_const_slots = [i for i, s in enumerate(sources) if s is None]
        if not non_const_slots:
            out[name] = [s for s in sources if s is not None]  # all constants? shouldn't happen
            continue
        first_slot = non_const_slots[0]
        last_slot = non_const_slots[-1]
        # Input source for the first non-constant binding.
        if name == root_dispatch and sizes[first_slot] == input_size:
            sources[first_slot] = "input"
        elif latest:
            sources[first_slot] = f"pred:{latest}:0"
        else:
            # No dependency available — fall back to "input" if size matches.
            sources[first_slot] = "input" if sizes[first_slot] == input_size else "zero"
        # Output binding.
        if first_slot != last_slot:
            sources[last_slot] = "zero"
        out[name] = [s for s in sources if s is not None]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=pathlib.Path, required=True)
    p.add_argument(
        "--call-graph", type=pathlib.Path, required=True, help="JSON from tools/extract_flow_dispatch_call_graph.py"
    )
    p.add_argument(
        "--constant-arena",
        type=pathlib.Path,
        required=True,
        help="Local constant arena file (used only to read its size)",
    )
    p.add_argument("--remote-constant-arena", required=True, help="Path on the board where the constant arena lives")
    p.add_argument(
        "--input-from", type=pathlib.Path, required=True, help="Local input file (used only to read its size)"
    )
    p.add_argument(
        "--root-dispatch", default="dispatch_0_call_000", help="The dispatch whose first binding is the model input"
    )
    p.add_argument("--out", type=pathlib.Path, required=True)
    args = p.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    call_graph = json.loads(args.call_graph.read_text())
    constant_arena_size = args.constant_arena.stat().st_size
    input_size = args.input_from.stat().st_size
    sources = derive(
        manifest=manifest,
        call_graph=call_graph,
        constant_arena_size=constant_arena_size,
        input_size=input_size,
        remote_constant_arena=args.remote_constant_arena,
        root_dispatch=args.root_dispatch,
    )
    args.out.write_text(json.dumps(sources, indent=2))
    print(f"wrote {len(sources)} entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
