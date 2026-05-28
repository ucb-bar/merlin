#!/usr/bin/env python3
"""Extract a call-site dispatch graph from an IREE flow phase dump.

The compile/profile matrix is keyed by executable wrapper
(`main_graph$async_dispatch_N`), but a whole-network execution must be
call-site aware because IREE can call the same executable multiple times with
different tensors. This tool reads the flow-phase MLIR and emits:

* unique call nodes (`dispatch_N_call_XXX`),
* the canonical wrapper each call uses (`dispatch_N`),
* data dependencies between call nodes, and
* explicit binding-source specs for the dispatch-flow runner.

The binding-source specs are derived from the real flow SSA graph. Constants
become `file:<remote-constant-arena>`, splats/empties become `zero`, model
inputs become `input`, dispatch results become `pred:<call>:0`, and simple
`flow.tensor.update` concatenation chains become `concat:<pred...>`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

from iree.compiler import ir

_DISPATCH_RE = re.compile(r"@main_graph\$async_dispatch_(\d+)::@([^\]\s]+)")


@dataclass(frozen=True)
class SourceExpr:
    kind: str
    args: tuple[SourceExpr, ...] = ()
    value: str = ""

    def dependencies(self) -> list[str]:
        if self.kind == "pred":
            return [self.value]
        deps: list[str] = []
        for arg in self.args:
            deps.extend(arg.dependencies())
        return deps

    def flatten(self) -> list[SourceExpr]:
        if self.kind == "concat":
            out: list[SourceExpr] = []
            for arg in self.args:
                out.extend(arg.flatten())
            return out
        return [self]

    def spec(self, remote_constant_arena: str) -> str:
        if self.kind == "file":
            return f"file:{remote_constant_arena}"
        if self.kind == "pred":
            return f"pred:{self.value}:0"
        if self.kind in {"input", "zero"}:
            return self.kind
        if self.kind == "concat":
            parts = []
            for arg in self.flatten():
                if arg.kind != "pred":
                    raise ValueError("concat binding currently supports predecessor " f"outputs only, got {arg.kind}")
                parts.append(f"pred:{arg.value}:0")
            return "concat:" + ",".join(parts)
        raise ValueError(f"unsupported source kind: {self.kind}")


def _entry_point(op: ir.Operation) -> tuple[str, str, str]:
    attr = str(op.attributes["entry_points"])
    match = _DISPATCH_RE.search(attr)
    if not match:
        raise ValueError(f"cannot parse flow.dispatch entry_points: {attr}")
    dispatch_id = int(match.group(1))
    executable = f"main_graph$async_dispatch_{dispatch_id}"
    export = match.group(2)
    return f"dispatch_{dispatch_id}", executable, export


def _call_name(canonical: str, index: int) -> str:
    return f"{canonical}_call_{index:03d}"


def _owner_name(value: ir.Value) -> str | None:
    owner = getattr(value, "owner", None)
    return getattr(owner, "name", None)


def _owner_text(value: ir.Value) -> str:
    owner = getattr(value, "owner", None)
    return str(owner) if owner is not None else ""


class Extractor:
    def __init__(self, module: ir.Module):
        self.module = module
        self.call_by_dispatch_text: dict[str, str] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.binding_sources: dict[str, list[str]] = {}

    def source_expr(self, value: ir.Value) -> SourceExpr:
        owner_name = _owner_name(value)
        owner_text = _owner_text(value)
        if owner_name is None:
            return SourceExpr("input")
        if owner_name == "flow.dispatch":
            call = self.call_by_dispatch_text.get(owner_text)
            if call is None:
                raise ValueError(
                    "flow.dispatch operand was not emitted by an earlier " f"call site: {owner_text[:200]}"
                )
            return SourceExpr("pred", value=call)
        if owner_name == "util.global.load":
            return SourceExpr("file")
        if owner_name in {"flow.tensor.splat", "flow.tensor.empty"}:
            return SourceExpr("zero")
        owner = value.owner
        operands = list(owner.operands)
        if owner_name in {"flow.tensor.reshape", "tensor.cast", "flow.tensor.clone"}:
            if len(operands) != 1:
                raise ValueError(f"{owner_name} has {len(operands)} operands")
            return self.source_expr(operands[0])
        if owner_name == "flow.tensor.update":
            # Operand segments are dest, mixed-offsets, source for the static
            # update form emitted here. For YOLOv8 these are channel-concat
            # chains over contiguous row-major blocks; preserve the real data
            # dependency by concatenating predecessor outputs at runtime.
            if len(operands) < 2:
                raise ValueError(f"malformed flow.tensor.update: {owner_text}")
            dest = self.source_expr(operands[0])
            src = self.source_expr(operands[-1])
            if dest.kind == "zero":
                return src
            return SourceExpr("concat", args=(dest, src))
        if owner_name == "hal.tensor.import":
            return SourceExpr("input")
        if owner_name == "arith.constant":
            return SourceExpr("zero")
        if not operands:
            return SourceExpr("input")
        exprs = [self.source_expr(operand) for operand in operands]
        flat: list[SourceExpr] = []
        for expr in exprs:
            if expr.kind not in {"zero", "file"}:
                flat.extend(expr.flatten())
        if len(flat) == 1:
            return flat[0]
        if flat:
            return SourceExpr("concat", args=tuple(flat))
        return exprs[0]

    def run(self, remote_constant_arena: str) -> None:
        index = 0

        def visit(op: ir.Operation) -> ir.WalkResult:
            nonlocal index
            if op.name != "flow.dispatch":
                return ir.WalkResult.ADVANCE
            canonical, executable, export = _entry_point(op)
            call = _call_name(canonical, index)
            index += 1
            sources = [self.source_expr(operand) for operand in op.operands]
            deps: list[str] = []
            for source in sources:
                for dep in source.dependencies():
                    if dep not in deps:
                        deps.append(dep)
            # flow.dispatch has N operands (inputs) and M results (outputs).
            # The HAL pipeline_layout exposes BOTH as bindings; the runtime
            # needs an allocated buffer for each output binding. Emit a
            # "zero" source per result so strict mode passes without
            # treating the dispatch's own output as an externally sourced
            # value.
            output_sources = [SourceExpr("zero") for _ in op.results]
            all_sources = sources + output_sources
            self.nodes[call] = {
                "canonical_dispatch": canonical,
                "canonical_key": f"main_graph$async_{canonical}",
                "executable": executable,
                "export": export,
                "dependencies": deps,
                "binding_sources": [source.spec(remote_constant_arena) for source in all_sources],
            }
            self.binding_sources[call] = self.nodes[call]["binding_sources"]
            self.call_by_dispatch_text[str(op)] = call
            return ir.WalkResult.ADVANCE

        self.module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-mlir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--remote-constant-arena",
        default="/root/dispatch_flow_constant_arena.bin",
    )
    args = parser.parse_args(argv)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(args.flow_mlir.read_text(), ctx)
    extractor = Extractor(module)
    extractor.run(args.remote_constant_arena)

    dispatch_graph = {
        name: {
            "dependencies": node["dependencies"],
            "canonical_dispatch": node["canonical_dispatch"],
            "canonical_key": node["canonical_key"],
            "op_summary": node["export"],
        }
        for name, node in extractor.nodes.items()
    }
    payload = {
        "metadata": {
            "source": str(args.flow_mlir),
            "node_count": len(extractor.nodes),
            "remote_constant_arena": args.remote_constant_arena,
            "note": "Call-site graph extracted from flow.dispatch SSA operands.",
        },
        "dispatch_graph": dispatch_graph,
        "dispatch_calls": extractor.nodes,
        "binding_sources": extractor.binding_sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
