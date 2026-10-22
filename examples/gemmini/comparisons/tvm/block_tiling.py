"""Explicit shape-only macro-tile selection for isolated TVM experiments.

Uses the pinned backend's existing 1/2/4-tile schedules and tile_choices API.
The preference for maximum area is a declared heuristic, not measured tuning.
"""

import hashlib
import json
from math import prod
from pathlib import Path


SUPPORTED_TILES = (1, 2, 4)  # Existing pinned backend schedule domain.


def hardware_facts(receipt_path, header_path=None):
    receipt = json.loads(Path(receipt_path).read_text())
    expected = receipt["headers"]["include/gemmini_params.h"]["sha256"]
    if header_path is None:
        candidates = [Path(name) for name, sha in receipt["resolved_headers"].items()
                      if Path(name).name == "gemmini_params.h" and sha == expected and Path(name).is_file()]
        if len(candidates) != 1:
            raise ValueError("Supply one explicit generated Gemmini parameter header")
        header_path = candidates[0]
    path = Path(header_path).resolve(strict=True)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError("Parameter header does not match the bound adapter receipt")
    values = {}
    for line in payload.decode().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "#define" and parts[1] in ("DIM", "BANK_NUM", "BANK_ROWS", "ACC_ROWS"):
            if parts[1] in values:
                raise ValueError("Duplicate hardware parameter")
            values[parts[1]] = int(parts[2], 0)
    if set(values) != {"DIM", "BANK_NUM", "BANK_ROWS", "ACC_ROWS"} or any(value <= 0 for value in values.values()):
        raise ValueError("Missing positive literal hardware parameters")
    return {"dim": values["DIM"], "spad_rows": values["BANK_NUM"] * values["BANK_ROWS"],
            "acc_rows": values["ACC_ROWS"], "header": {"path": str(path), "sha256": actual}}


def choose_tiles(shape, facts, minimum_fill_percent=90):
    """Maximize supported macro area with at least the declared grid fill.

    Fill counts occupied micro-tile slots relative to allocated macro slots.
    Intrinsic partial rows/columns inside a micro tile are reported separately.
    The backend emits only valid tail blocks; larger groups add no compute calls.
    """
    if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("Expected positive static integer (M,N,K)")
    if type(minimum_fill_percent) is not int or not 1 <= minimum_fill_percent <= 100:
        raise ValueError("Minimum macro fill must be an integer percentage in 1..100")
    if any(type(facts.get(k)) is not int or facts[k] <= 0 for k in ("dim", "spad_rows", "acc_rows")):
        raise ValueError("Expected explicit positive hardware capacities")
    m, n, k = shape
    dim = facts["dim"]
    mt, nt, kt = [(v + dim - 1) // dim for v in shape]
    candidates = []
    for ti in SUPPORTED_TILES:
        for tj in SUPPORTED_TILES:
            spad, acc = dim * (ti + tj), dim * ti * tj
            if spad > facts["spad_rows"] or acc > facts["acc_rows"]:
                continue
            groups_i, groups_j = (mt + ti - 1) // ti, (nt + tj - 1) // tj
            slots = groups_i * ti * groups_j * tj
            if 100 * mt * nt < minimum_fill_percent * slots:
                continue
            counts = {"loads_a": mt * groups_j * kt, "loads_b": nt * groups_i * kt,
                      "computes": mt * nt * kt, "stores": mt * nt, "admissions": 1}
            candidates.append({"tiles": [ti, tj], "macro_grid_fill": mt * nt / slots,
                               "spad_rows": spad, "acc_rows": acc, "primitive_API_counts": counts})
    if not candidates:
        raise ValueError("No supported tile grouping fits hardware and fill requirements")
    selected = max(candidates, key=lambda c: (prod(c["tiles"]),
                   -c["primitive_API_counts"]["loads_a"] - c["primitive_API_counts"]["loads_b"],
                   c["tiles"][1], c["tiles"][0]))
    return {"shape": list(shape), **selected, "micro_tile_output_fill": m * n / (mt * nt * dim * dim),
            "minimum_macro_fill_percent": minimum_fill_percent, "candidates": candidates}


def select_from_relax(mod, tvm, facts, minimum_fill_percent=90):
    """Read static signed INT8→INT32 matmuls from semantic Relax, never names."""
    shapes = set()

    def static_shape(info, dtype):
        if not isinstance(info, tvm.relax.TensorStructInfo) or info.dtype != dtype or not isinstance(info.shape, tvm.relax.ShapeExpr):
            return None
        if any(not isinstance(v, tvm.tir.IntImm) or int(v) <= 0 for v in info.shape):
            return None
        return tuple(int(v) for v in info.shape)

    def visit(call):
        if not isinstance(call, tvm.relax.Call) or not isinstance(call.op, tvm.ir.Op) or call.op.name != "relax.matmul":
            return
        left = static_shape(call.args[0].struct_info, "int8")
        right = static_shape(call.args[1].struct_info, "int8")
        output = static_shape(call.struct_info, "int32")
        if left and right and len(right) == 2 and left[-1] == right[0] and output == left[:-1] + (right[1],):
            shapes.add((prod(left[:-1]), right[1], right[0]))

    for function in mod.functions.values():
        if isinstance(function, tvm.relax.Function):
            tvm.relax.analysis.post_order_visit(function.body, visit)
    rows = [choose_tiles(shape, facts, minimum_fill_percent) for shape in sorted(shapes)]
    return {tuple(row["shape"]): tuple(row["tiles"]) for row in rows}, rows
