"""Emit original and heavy-pass schedules through one unchanged effect callback.

    python counted_emission_recipe.py [--shape M,K,N] [--output DIR] [--library DIR]

This is an optional emitter probe, not a new launch gate. Actual counted LLVM
contains external effect calls and the explicit original completion calls.
Replace those callbacks with your existing instruction encoder and real device
completion. Neutral effect declarations and test-only admissions confer no
hardware support, numerical qualification or performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path


def _support():
    if "recipe_support" not in sys.modules:
        spec = importlib.util.spec_from_file_location("recipe_support", Path(__file__).with_name("recipe_support.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["recipe_support"] = module
        spec.loader.exec_module(module)
    return sys.modules["recipe_support"]


def emit(result):
    """The SAME encoder/control/completion path for every checked owner type."""
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp, i64

    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    from merlin.llvmlower.host_builder import FnBuilder

    builder, declarations, emission_order = FnBuilder([]), {}, []
    control = FnBuilderControlAdapter(builder)

    def lower_effect(instruction, fields):
        if not instruction.isidentifier():
            raise ValueError("neutral effect declaration requires an identifier")
        symbol = "effect_" + instruction
        declarations[symbol] = len(fields)
        builder.add(llvm.CallOp(symbol, *(control.value(v) for v in fields.values())))
        emission_order.append(symbol)

    def complete_epoch():
        declarations["original_full_completion"] = 0
        builder.add(llvm.CallOp("original_full_completion"))
        emission_order.append("original_full_completion")

    receipt = result.emit_index_program(control, lower_effect, complete_epoch=complete_epoch)
    function = llvm.FuncOp("kernel", llvm.LLVMFunctionType([], llvm.LLVMVoidType()),
                          linkage=llvm.LinkageAttr("external"), body=builder.finish())
    module = ModuleOp([*(llvm.FuncOp(name, llvm.LLVMFunctionType([i64]*arity, llvm.LLVMVoidType()),
                                   linkage=llvm.LinkageAttr("external"))
                        for name, arity in sorted(declarations.items())), function])
    module.verify()
    calls = [op.callee.root_reference.data for op in function.walk() if op.name == "llvm.call"]
    if (not emission_order or emission_order[-1] != "original_full_completion"
            or calls.count("original_full_completion") != receipt["completion_mapping"]["emitted_static_drains"]):
        raise ValueError("actual emitted schedule lost its required closing completion")
    text = str(module)
    return text, {"emitted_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "emitted_operations": sum(1 for _ in module.walk()),
                  "retained_loop_blocks": len(function.body.blocks),
                  "completion_sites": calls.count("original_full_completion"),
                  "same_effect_encoder": "counted_emission_recipe.emit.lower_effect",
                  "last_effect": emission_order[-1], "receipt": receipt,
                  "closing_basis": "checked source order and actual LLVM completion sites; target binding separate"}


def recipe(facts=None, shape=None, output=None):
    S = _support()
    caps = S.capacities(facts)
    edge_k, edge = caps["array"]
    shape = tuple(shape or (2*edge-1, 3*edge_k, (caps["accumulator"]["rows"]//(2*edge)+1)*edge-1))
    if shape[0]*shape[1]*shape[2] > 1_000_000:
        raise ValueError("optional host-numerical probe budget exceeded; use your ordinary numerical owner")
    p, storage, resources = S.dense_view(shape, caps)
    adapter = S.adapter()
    window = storage.accumulator.capacity//2
    block = S.pick(S.blocks(p, storage, replace(resources, max_k_blocks=p.tiles[1]-1)),
                   lambda b: b.k < p.tiles[1] and b.m*b.n*p.tile_m <= window and p.tiles[2]//b.n > 1)
    variants = {}
    for name, batch, mode in (("original", False, "separate"), ("batched", True, "separate"),
                              ("ready", True, "ready"), ("disjoint", True, "disjoint")):
        result, _ = S.segmented_pipeline(p, storage, resources, adapter, block, name=name,
                                        batch=batch, readout_mode=mode)
        numeric = S.execute_dense(result.kernel, adapter, p)
        text, emitted = emit(result)
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (name + ".llvm.mlir")
            path.write_text(text + "\n")
            emitted["output"] = str(path)
        variants[name] = {**emitted, "source_check": numeric,
                          "dynamic_epoch_drains": result.realization["dynamic_epoch_drains"]}
    if not variants["batched"]["dynamic_epoch_drains"] < variants["original"]["dynamic_epoch_drains"]:
        raise ValueError("selected geometry exposes no readout batching effect")
    if len({r["emitted_sha256"] for r in variants.values()}) != len(variants):
        raise ValueError("a selected transformation did not reach actual LLVM emission")
    return {"facts": caps["source"], "shape": shape, "block": (block.m, block.k, block.n),
            "variants": variants, "verified": "every counted effect stream reproduced the source product",
            "binding": "neutral external effect/completion declarations; replace with selected target callbacks",
            "execution_qualified": False, "performance_claim": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--facts")
    parser.add_argument("--shape", help="M,K,N; default derived from selected store capacity")
    parser.add_argument("--library", help="editable generic compiler library")
    parser.add_argument("--output", help="write actual counted LLVM for both sides")
    args = parser.parse_args(argv)
    _support().use_library(args.library)
    shape = None if args.shape is None else tuple(int(v) for v in args.shape.split(","))
    if shape is not None and (len(shape) != 3 or any(v < 1 for v in shape)):
        parser.error("shape must have three positive dimensions")
    print(json.dumps(recipe(args.facts, shape, args.output), indent=2))


if __name__ == "__main__":
    main()
