#!/usr/bin/env python3
"""Compare independent and row-stacked products with identical integer work.

This is a kernel diagnostic with synthetic operands and source-derived geometry.
It excludes host quantization, packing and reconstruction, and is not a model
latency measurement. Both variants use the same unchanged device package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def selected_output(output, *, compiler, host_package, repetitions):
    """Keep every generated file outside the selected immutable packages."""
    output = Path(output).resolve()
    roots = [Path(p).resolve(strict=True) for p in (compiler, host_package)]
    if (output.exists() or not 2 <= repetitions <= 32
            or any(output.is_relative_to(root) for root in roots)):
        raise ValueError("fresh output outside compiler/host package and 2..32 repetitions required")
    return output


def products(*, groups, heads, rows, reduction, columns, seed):
    """Return equivalent bounded signed products in the two physical shapes."""
    if any(type(v) is not int or v <= 0 for v in (groups, heads, rows, reduction, columns)):
        raise ValueError("positive integer matrix dimensions are required")
    if reduction * 31 * 31 > np.iinfo(np.int32).max:
        raise ValueError("selected stimulus exceeds the exact signed accumulator range")
    rng = np.random.default_rng(seed)
    left = rng.integers(-31, 32, (groups, heads, rows, reduction), dtype=np.int8)
    right = rng.integers(-31, 32, (groups, reduction, columns), dtype=np.int8)
    a = left.reshape(groups * heads, rows, reduction)
    b = np.repeat(right, heads, axis=0)
    ar = left.reshape(groups, heads * rows, reduction)
    reference = a.astype(np.int32) @ b.astype(np.int32)
    regrouped = ar.astype(np.int32) @ right.astype(np.int32)
    np.testing.assert_array_equal(reference, regrouped.reshape(reference.shape))
    return ((a, b, reference), (ar, right, regrouped))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "host-package", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--groups", type=int, required=True)
    p.add_argument("--heads", type=int, required=True)
    p.add_argument("--rows", type=int, nargs="+", required=True)
    p.add_argument("--reduction", type=int, required=True)
    p.add_argument("--columns", type=int, required=True)
    p.add_argument("--seed", type=int, default=711)
    p.add_argument("--repetitions", type=int, default=6)
    p.add_argument("--dram-base", type=lambda s: int(s, 0), required=True)
    p.add_argument("--dram-bytes", type=int, required=True)
    args = p.parse_args(argv)
    args.output = selected_output(args.output, compiler=args.compiler,
                                  host_package=args.host_package, repetitions=args.repetitions)
    args.output.mkdir(parents=True)

    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.llvmlower.device_shim import logical_buffers
    from merlin.model_check.catalog import ProgramCatalog
    from merlin.model_check.commands import _replay_cases
    from merlin.model_check.replay import ReplayCase, digest
    from merlin.targetgen.corpus_spec import build

    binding = _mesh_tile_binding(args.target, "int8", "i32", numeric_policy=None)
    sidecar = {key: {} for key in ("signatures", "entries", "dtypes", "call_buffers", "programs")}
    sidecar["routed"] = []
    interfaces, cases, plan, comparisons = {}, [], [], []
    numeric_policy = None
    for rows in args.rows:
        pair = products(groups=args.groups, heads=args.heads, rows=rows,
                        reduction=args.reduction, columns=args.columns, seed=args.seed + rows)
        names = []
        for kind, (a, b, expected) in zip(("independent", "row_stacked"), pair, strict=True):
            symbol = f"batch_{kind}_m{rows}"
            names.append(symbol)
            entry = {"name": symbol, "kind": "op", "source_role": "controlled_integer_study",
                     "source_reference": "same dot products, repeated RHS, exact row mapping",
                     "op": "gemv_batched", "B": a.shape[0], "M": a.shape[1],
                     "K": args.reduction, "N": args.columns, "epilogue": []}
            requirement, interface = build(entry, binding)
            if numeric_policy is not None and requirement["numeric_policy"] != numeric_policy:
                raise ValueError("matched variants require the same arithmetic policy")
            numeric_policy = requirement["numeric_policy"]
            buffers, why = logical_buffers(interface)
            if why or buffers is None:
                raise ValueError(why)
            sidecar["signatures"][symbol] = [a.shape[1], args.reduction, args.columns]
            sidecar["entries"][symbol] = entry
            sidecar["dtypes"][symbol] = ["i8", "i8", "i32"]
            sidecar["call_buffers"][symbol] = [
                {"role": v.role, "shape": list(v.shape), "dtype": v.dtype} for v in buffers]
            sidecar["programs"][symbol] = {"entry": entry}
            interfaces[symbol] = interface
            (args.output / (symbol + ".mlir")).write_text(interface)
        comparisons.append({"rows_per_original_head": rows, "symbols": names,
                            "logical_mac_count": args.groups * args.heads * rows * args.reduction * args.columns})
        for repetition in range(args.repetitions):
            # Alternate order. Repetition zero is retained separately as cold.
            order = (0, 1) if repetition % 2 == 0 else (1, 0)
            for index in order:
                a, b, expected = pair[index]
                symbol = names[index]
                roles = tuple(v["role"] for v in sidecar["call_buffers"][symbol])
                label = f"rows={rows}/repeat={repetition}/" + ("independent" if index == 0 else "row_stacked")
                cases.append(ReplayCase(symbol, label, roles, (a, b, np.zeros_like(expected)), (expected,)))
                plan.append({"case": len(plan), "symbol": symbol, "label": label,
                             "outputs": [{"role": "out_0", "digest": digest(expected.tobytes())}]})
    routing = {"operand_dtype": "int8", "accum_dtype": "i32", "numeric_policy": numeric_policy}
    catalog = ProgramCatalog(args.target, args.output, args.output, args.output, routing, sidecar, interfaces)
    (args.output / "catalog.json").write_text(json.dumps(catalog.record(), indent=2) + "\n")
    (args.output / "sidecar.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    options = SimpleNamespace(submission=args.compiler, host_package=args.host_package, no_cache=True,
                              engine="spike", mode="full", dram_base=args.dram_base,
                              dram_bytes=args.dram_bytes, timeout_s=180, jobs=1)
    work = args.output / "work"
    work.mkdir()
    result = _replay_cases(options, catalog, cases, work)
    record = {"schema": "merlin.model_study.attention_batching.v1", "comparison": comparisons,
              "scope": __doc__, "seed": args.seed, "repetitions": args.repetitions,
              "stimulus": "synthetic signed-byte values in [-31,31], repeated RHS within each head group",
              "arithmetic": "identical exact i32 dots and original head coordinate order",
              "excluded": ["host quantization", "host packing", "floating reconstruction", "model state"],
              "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "replay": result}
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    ok = not result["refused"] and len(result["kernels"]) == len(interfaces) and all(
        row.get("status") == "pass" and row.get("cases") == args.repetitions for row in result["kernels"])
    print(json.dumps({"passed": ok, "kernels": len(result["kernels"]), "cases": len(cases),
                      "output": str(args.output)}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
