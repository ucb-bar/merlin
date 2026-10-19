"""Complete exact integer references from bounded original-seed panels.

Reuse the independent integer matmul owner, including its checked exact-F64
bound. Never allocate the original full arange/int64 RHS or full im2col matrix.
This changes reference preparation only, not a source, compiler or objective.
"""

from __future__ import annotations

import time
from math import prod

import numpy as np

from merlin.model_check.kernel_reference import _int_matmul

SUPPORTED = {
    "matmul",
    "matmul_rhs_nk",
    "conv_nhwc",
    "conv_transition_nhwc",
    "resident_rhs_reuse",
    "grouped_head_integer_mix",
    "parameter_projection_chain",
}


def _seed(flat, ordinal):
    return (((flat * (1 + 2 * ordinal % 16) + ordinal) % 17) - 8).astype(np.int8)


def seed_matrix(shape, ordinal, rows, columns):
    return _seed(rows[:, None] * shape[1] + columns[None, :], ordinal)


def plan(description, *, block, max_workspace_bytes, max_output_bytes, max_dense_source_bytes):
    if (
        not isinstance(block, dict)
        or set(block) != {"M", "N", "K"}
        or any(type(value) is not int or value < 1 for value in block.values())
        or any(
            type(value) is not int or value < 1
            for value in (max_workspace_bytes, max_output_bytes, max_dense_source_bytes)
        )
    ):
        raise ValueError("blocked reference requires finite positive panel and memory budgets")
    m, n, k = (block[x] for x in ("M", "N", "K"))
    # Includes seed coordinate temporaries, int64/F64 owner conversions, dot
    # temporaries, accumulator, and a streaming hash/IO margin. Output maps
    # and the Python runtime are reported separately, never hidden here.
    workspace = 64 * (m * k + k * n + m * n) + 4 * 1024**2
    output_bytes = sum(prod(row["shape"]) * np.dtype(row["dtype"]).itemsize for row in description["outputs"])
    # Two i64 temporaries coexist in the unchanged Torch seed expression.
    dense_lower = 16 * max(prod(row["shape"]) for row in [*description["inputs"], *description.get("parameters", [])])
    return {
        "block": block,
        "workspace_upper_bytes": workspace,
        "output_bytes": output_bytes,
        "reference_memory_admitted": workspace <= max_workspace_bytes and output_bytes <= max_output_bytes,
        "output_mapping_residency_upper_bytes": output_bytes,
        "dense_source_seed_peak_lower_bytes": dense_lower,
        "dense_source_payload_budget_bytes": max_dense_source_bytes,
        "dense_source_memory": (
            "REFUSED: unchanged eager seed construction exceeds separate payload budget"
            if dense_lower > max_dense_source_bytes
            else "UNKNOWN: seed lower bound fits; full eager allocation and capture still unmeasured"
        ),
        "python_runtime_and_blas_overhead": "separate measured RSS; not included in workspace bound",
    }


class _Engine:
    def __init__(self, directory, description, block, timeout_s):
        self.description, self.block = description, block
        self.deadline = time.monotonic() + timeout_s
        self.computed_work = 0
        self.paths = []
        self.outputs = []
        for ordinal, row in enumerate(description["outputs"]):
            path = directory / f"output-{ordinal}.npy"
            self.paths.append(path)
            self.outputs.append(
                np.lib.format.open_memmap(path, mode="w+", dtype=row["dtype"], shape=tuple(row["shape"]))
            )

    def mm(self, m, k, n, lhs, rhs, output, narrowed=None, shift=None):
        bm, bk, bn = (self.block[x] for x in ("M", "K", "N"))
        for i in range(0, m, bm):
            rows = np.arange(i, min(i + bm, m), dtype=np.int64)
            for j in range(0, n, bn):
                columns = np.arange(j, min(j + bn, n), dtype=np.int64)
                acc = np.zeros((len(rows), len(columns)), dtype=np.int64)
                for p in range(0, k, bk):
                    if time.monotonic() > self.deadline:
                        raise TimeoutError("complete reference exceeded its admitted per-member wall budget")
                    reduction = np.arange(p, min(p + bk, k), dtype=np.int64)
                    acc += _int_matmul(lhs(rows, reduction), rhs(reduction, columns))
                    self.computed_work += len(rows) * len(reduction) * len(columns)
                if np.any(acc < -(2**31)) or np.any(acc > 2**31 - 1):
                    raise ValueError("blocked exact reference exceeds the original i32 result contract")
                output[i : i + len(rows), j : j + len(columns)] = acc.astype(np.int32)
                if narrowed is not None:
                    narrowed[i : i + len(rows), j : j + len(columns)] = np.clip(
                        (acc + (1 << (shift - 1))) // (1 << shift), -128, 127
                    ).astype(np.int8)


def materialize(declaration, description, directory, *, block, timeout_s):
    """Write every original output, with all declared scalar products evaluated."""
    pattern, g = declaration["pattern"], declaration["geometry"]
    if pattern not in SUPPORTED or type(timeout_s) not in (int, float) or not 0 < timeout_s <= 120:
        raise ValueError("blocked reference supports explicit bounded integer source patterns only")
    if directory.exists() or not directory.parent.is_dir():
        raise ValueError("blocked reference member destination must be fresh")
    directory.mkdir()
    engine = _Engine(directory, description, block, timeout_s)
    if pattern.startswith("matmul"):
        m, k, n = (g[x] for x in ("M", "K", "N"))
        rhs = (
            (lambda p, j: seed_matrix((n, k), 1, j, p).T)
            if pattern.endswith("nk")
            else (lambda p, j: seed_matrix((k, n), 1, p, j))
        )
        engine.mm(m, k, n, lambda i, p: seed_matrix((m, k), 0, i, p), rhs, engine.outputs[0])
    elif pattern == "parameter_projection_chain":
        width, depth, shared = (g[x] for x in ("width", "depth", "shared"))
        state = ((np.arange(width, dtype=np.int64) % 17) - 8).astype(np.int8).reshape(1, width)
        for stage in range(depth):
            # The output map is reused between stages. Snapshot its *complete*
            # previous value before any N panel is written, preserving the SSA
            # state carried by the unchanged source program.
            previous = np.array(state, copy=True)
            parameter = 0 if shared else stage
            engine.mm(
                1,
                width,
                width,
                lambda i, p: previous[np.ix_(i, p)],
                lambda p, j, parameter=parameter: (
                    ((p[:, None] * width + j[None, :] + 3 * parameter + 1) % 17) - 8
                ).astype(np.int8),
                engine.outputs[0],
                narrowed=engine.outputs[1],
                shift=declaration["arithmetic"]["readout_shift"],
            )
            state = engine.outputs[1]
    elif pattern == "resident_rhs_reuse":
        k, n, count = g["K"], g["N"], len(g["M"])
        for ordinal, m in enumerate(g["M"]):
            engine.mm(
                m,
                k,
                n,
                lambda i, p, ordinal=ordinal, m=m: seed_matrix((m, k), ordinal, i, p),
                lambda p, j: seed_matrix((k, n), count, p, j),
                engine.outputs[ordinal],
            )
    elif pattern == "grouped_head_integer_mix":
        heads, kv, width, tokens, prefix, steps = (
            g[x] for x in ("query_heads", "kv_heads", "head_dim", "query_tokens", "prefix_tokens", "steps")
        )
        previous_k = previous_v = None
        cursor = 0
        for step in range(steps):
            length = prefix + (step + 1) * tokens
            cache_k, cache_v = engine.outputs[cursor + 2 * heads : cursor + 2 * heads + 2]
            before = length - tokens
            for group in range(kv):
                for col in range(0, width, block["N"]):
                    columns = np.arange(col, min(col + block["N"], width), dtype=np.int64)
                    for start in range(0, before, block["K"]):
                        rows = np.arange(start, min(start + block["K"], before), dtype=np.int64)
                        dest = (group, slice(start, start + len(rows)), slice(col, col + len(columns)))
                        if step:
                            cache_k[dest], cache_v[dest] = previous_k[dest], previous_v[dest]
                        else:
                            flat = (group * prefix + rows[:, None]) * width + columns[None, :]
                            cache_k[dest], cache_v[dest] = _seed(flat, 0), _seed(flat, 1)
                    for start in range(0, tokens, block["K"]):
                        rows = np.arange(start, min(start + block["K"], tokens), dtype=np.int64)
                        flat = (group * tokens + rows[:, None]) * width + columns[None, :]
                        dest = (
                            group,
                            slice(before + start, before + start + len(rows)),
                            slice(col, col + len(columns)),
                        )
                        cache_k[dest], cache_v[dest] = _seed(flat, 3 + step * 3), _seed(flat, 4 + step * 3)
            for head in range(heads):
                group = head // (heads // kv)
                score, mixed = engine.outputs[cursor : cursor + 2]
                engine.mm(
                    tokens,
                    width,
                    length,
                    lambda i, p, head=head, step=step: _seed(
                        (head * tokens + i[:, None]) * width + p[None, :], 2 + step * 3
                    ),
                    lambda p, j, group=group: cache_k[group][np.ix_(j, p)].T,
                    score,
                )
                shift = declaration["arithmetic"]["readout_shift"]
                engine.mm(
                    tokens,
                    length,
                    width,
                    lambda i, p: np.clip(
                        (score[np.ix_(i, p)].astype(np.int64) + (1 << (shift - 1))) // (1 << shift), -128, 127
                    ).astype(np.int8),
                    lambda p, j, group=group: cache_v[group][np.ix_(p, j)],
                    mixed,
                )
                cursor += 2
            previous_k, previous_v = cache_k, cache_v
            cursor += 2
    else:
        oh = (g["H"] + 2 * g["PH"] - g["KH"]) // g["SH"] + 1
        ow = (g["W"] + 2 * g["PW"] - g["KW"]) // g["SW"] + 1
        m, k, n = oh * ow, g["KH"] * g["KW"] * g["CI"], g["CO"]

        def spatial(rows, reduction):
            kh, kw = reduction // (g["KW"] * g["CI"]), (reduction // g["CI"]) % g["KW"]
            y = (rows // ow)[:, None] * g["SH"] - g["PH"] + kh[None, :]
            x = (rows % ow)[:, None] * g["SW"] - g["PW"] + kw[None, :]
            valid = (y >= 0) & (y < g["H"]) & (x >= 0) & (x < g["W"])
            flat = (y * g["W"] + x) * g["CI"] + (reduction % g["CI"])[None, :]
            return np.where(valid, _seed(flat, 0), np.int8(0))

        state = engine.outputs[1].reshape(m, n) if pattern == "conv_transition_nhwc" else None
        engine.mm(
            m,
            k,
            n,
            spatial,
            lambda p, j: seed_matrix((k, n), 1, p, j),
            engine.outputs[0],
            narrowed=state,
            shift=declaration["arithmetic"]["readout_shift"],
        )
        if state is not None:
            engine.mm(
                m,
                n,
                g["OUT"],
                lambda i, p: state[np.ix_(i, p)],
                lambda p, j: seed_matrix((n, g["OUT"]), 2, p, j),
                engine.outputs[2],
            )
    if engine.computed_work != description["scalar_products"]:
        raise ValueError("blocked reference omitted required scalar products")
    for value in engine.outputs:
        value.flush()
    return {
        "outputs": [str(path) for path in engine.paths],
        "computed_scalar_products": engine.computed_work,
        "complete_output_elements": sum(value.size for value in engine.outputs),
        "arithmetic_owner": "merlin.model_check.kernel_reference._int_matmul exact-bound implementation",
    }
