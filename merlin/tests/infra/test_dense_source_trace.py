"""Concrete inverse checks of observed neutral primitive arguments.

No selected schedule supplies the observations. This interpreter establishes
complete ordered source coordinates and byte ownership for this declared C API;
it does not interpret arbitrary C or machine instructions.
"""

from merlin.sched.ir import Call, Const, Kernel

NAMES = {
    "lhs_load": ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
    "weight_load": ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
    "preload": ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth"),
    "compute": ("lhs", "m", "k", "rows", "depth"),
    "store": ("row", "col", "rows", "cols", "acc"),
    "store_i8": ("row", "col", "rows", "cols", "acc"),
}
KINDS = tuple(NAMES)


def check_trace(records, problem, storage, arguments, *, output_dtype="i32"):
    """Reconstruct every actual effect, including cross-view physical aliases."""
    memory, outputs, computations, calls = {}, set(), set(), []
    pending, loads, stores = None, 0, 0
    assert output_dtype in ("i32", "i8"), "unsupported final output type"
    output_bytes = 4 if output_dtype == "i32" else 1
    mt, kt, nt = problem.tiles

    def extent(start, size, tile):
        assert 0 <= start < size and start % tile == 0, "wrong source address"
        return min(tile, size - start)

    def addresses(view, row, rows, cols, width):
        assert row >= view.base and row + rows <= view.base + view.capacity, "wrong local address"
        assert row % view.alignment == 0 and 0 < cols * width <= view.unit_bytes, "wrong local row layout"
        return [
            (i, j, byte, (view.backing, (row + i) * view.unit_bytes + j * width + byte))
            for i in range(rows)
            for j in range(cols)
            for byte in range(width)
        ]

    for record in records:
        assert len(record) >= 2
        opcode, count, *values = record
        assert 0 <= opcode < len(KINDS), "unknown primitive"
        name = KINDS[opcode]
        assert count == len(NAMES[name]) and len(values) == count, "unknown primitive operands"
        assert all(type(value) is int and value >= 0 for value in values), "unbounded primitive operand"
        a = dict(zip(NAMES[name], values, strict=True))
        calls.append(Call(name, tuple((key, Const(value)) for key, value in a.items())))
        assert pending is None or name == "compute", "preload must complete at its adjacent compute"
        if name in ("lhs_load", "weight_load"):
            left = name == "lhs_load"
            view = storage.lhs if left else storage.rhs
            width = problem.lhs_bytes if left else problem.rhs_bytes
            shape = (problem.m, problem.k) if left else (problem.k, problem.n)
            tile = (problem.tile_m, problem.tile_k) if left else (problem.tile_k, problem.tile_n)
            assert a["src_row"] == a["a"] * tile[0] and a["src_col"] == a["b"] * tile[1], "wrong tile address"
            assert a["rows"] == extent(a["src_row"], shape[0], tile[0]), "wrong load tail"
            assert a["cols"] == extent(a["src_col"], shape[1], tile[1]) and a["step"] == 1, "wrong load tail/stride"
            for i, j, byte, address in addresses(view, a["dst"], a["rows"], a["cols"], width):
                memory[address] = ("lhs" if left else "rhs", a["src_row"] + i, a["src_col"] + j, byte)
            loads += a["rows"] * a["cols"] * width
        elif name == "preload":
            assert a["continue"] in (0, 1), "unknown continuation"
            pending = a
        elif name == "compute":
            assert pending is not None, "compute lacks preload"
            p = pending
            assert a["k"] == p["k"] and a["rows"] == p["rows"] and a["depth"] == p["depth"], "wrong preload binding"
            assert 0 <= a["m"] < mt and 0 <= a["k"] < kt and 0 <= p["n"] < nt, "wrong reduction tile"
            assert a["rows"] == extent(a["m"] * problem.tile_m, problem.m, problem.tile_m), "wrong compute tail"
            assert a["depth"] == extent(a["k"] * problem.tile_k, problem.k, problem.tile_k), "wrong reduction tail"
            assert p["cols"] == extent(p["n"] * problem.tile_n, problem.n, problem.tile_n), "wrong output tail"
            assert p["continue"] == int(a["k"] != 0), "wrong accumulator continuation"
            key = a["m"], p["n"], a["k"]
            assert key not in computations, "duplicate reduction tile"
            computations.add(key)
            for i, t, byte, address in addresses(storage.lhs, a["lhs"], a["rows"], a["depth"], problem.lhs_bytes):
                assert memory.get(address) == ("lhs", a["m"] * problem.tile_m + i, a["k"] * problem.tile_k + t, byte), (
                    "wrong/overwritten lhs address"
                )
            for t, j, byte, address in addresses(storage.rhs, p["weight"], a["depth"], p["cols"], problem.rhs_bytes):
                assert memory.get(address) == ("rhs", a["k"] * problem.tile_k + t, p["n"] * problem.tile_n + j, byte), (
                    "wrong/overwritten weight address"
                )
            for i, j, byte, address in addresses(storage.accumulator, p["acc"], a["rows"], p["cols"], 4):
                owner = ("acc", a["m"] * problem.tile_m + i, p["n"] * problem.tile_n + j)
                if p["continue"]:
                    assert memory.get(address) == (*owner, a["k"] * problem.tile_k, byte), (
                        "incomplete/overwritten reduction state"
                    )
                memory[address] = (*owner, a["k"] * problem.tile_k + a["depth"], byte)
            pending = None
        else:
            assert name == ("store" if output_bytes == 4 else "store_i8"), "wrong actual readout mode"
            assert a["rows"] == extent(a["row"], problem.m, problem.tile_m), "wrong store tail"
            assert a["cols"] == extent(a["col"], problem.n, problem.tile_n), "wrong store tail"
            for i, j, byte, address in addresses(storage.accumulator, a["acc"], a["rows"], a["cols"], 4):
                owner = (a["row"] + i, a["col"] + j, byte)
                assert memory.get(address) == ("acc", owner[0], owner[1], problem.k, byte), (
                    "incomplete/overwritten output"
                )
            for i in range(a["rows"]):
                for j in range(a["cols"]):
                    for byte in range(output_bytes):
                        owner = (a["row"] + i, a["col"] + j, byte)
                        assert owner not in outputs, "duplicate output"
                        outputs.add(owner)
            stores += a["rows"] * a["cols"] * output_bytes
    assert pending is None and len(outputs) == problem.m * problem.n * output_bytes, "incomplete output coverage"
    assert len(computations) == mt * nt * kt, "incomplete reduction coverage"
    kernel = Kernel("observed_dense", arguments, tuple(calls))
    return dict(
        commands=len(calls),
        requested_read_bytes=loads,
        requested_write_bytes=stores,
        observed_kernel_sha256=kernel.digest(),
        scope="concrete observed primitive effects; no arbitrary machine-code interpretation",
    )
