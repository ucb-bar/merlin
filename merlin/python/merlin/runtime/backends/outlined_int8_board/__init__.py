"""Reusable K1 RVV kernel for Merlin's W8A8 rank-2 contractions.

This is an execution backend, not an attribution shim.  It replaces the repeated
``i8 x i8 -> i32`` contraction bodies in a prepared whole model with calls to one
small C/RVV implementation.  One thin symbol is emitted per MLIR type signature,
but all symbols call the same descriptor-driven kernel body.

The route is deliberately rank-2 only.  Weight matmuls have an AOT-transposed,
row-major ``K x N`` RHS and are the 155 repeated bodies that dominate TinyLlama's
compile time.  Batched attention contractions stay in Merlin's normal lowering.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ....llvmlower import passes_opu

_HERE = Path(__file__).resolve().parent
_SHIM_SRC = _HERE / "outlined_gemm_i8_rvv.c"

SYMBOL_PREFIX = "merlin_outlined_gemm_i8"
SIDECAR_NAME = "outlined_int8_signatures.json"


class OutlinedInt8Unavailable(RuntimeError):
    pass


def _rank2(shape) -> bool:
    return len(shape.parallel) == 2


def rewrite_prepared_file(prepared: str | Path, work: str | Path) -> passes_opu.OpuRewrite:
    """Route every legal rank-2 W8A8 contraction and record its ABI signatures."""
    return passes_opu.rewrite_prepared_file(
        prepared, work, select=_rank2, symbol_prefix=SYMBOL_PREFIX,
        sidecar_name=SIDECAR_NAME)


def load_signatures(work: str | Path) -> dict[str, tuple[int, ...]]:
    return passes_opu.load_sidecar(work, SIDECAR_NAME)


def build_object(cc: Path, cflags: list[str], signatures: dict[str, tuple[int, ...]],
                 work: Path, *, parallel: bool = False) -> Path:
    """Compile one RVV body plus the monomorphic wrappers required by MLIR."""
    if not _SHIM_SRC.is_file():
        raise OutlinedInt8Unavailable(f"outlined INT8 RVV shim not found at {_SHIM_SRC}")
    rank2 = {sym: dims for sym, dims in signatures.items() if len(dims) == 3}
    if len(rank2) != len(signatures):
        raise OutlinedInt8Unavailable(
            "outlined INT8 CPU route only supports rank-2 (M,N,K) signatures")
    work.mkdir(parents=True, exist_ok=True)

    wrappers = []
    for sym, (m, n, k) in sorted(rank2.items()):
        wrappers.append(f"""
/* {sym}: M={m}, N={n}, K={k}; extents are re-read from the descriptors. */
merlin_memref_2d_i32 {sym}(
    int8_t*a0,int8_t*a1,intptr_t a2,intptr_t a3,intptr_t a4,intptr_t a5,intptr_t a6,
    int8_t*b0,int8_t*b1,intptr_t b2,intptr_t b3,intptr_t b4,intptr_t b5,intptr_t b6,
    int32_t*c0,int32_t*c1,intptr_t c2,intptr_t c3,intptr_t c4,intptr_t c5,intptr_t c6) {{
  return merlin_outlined_gemm_i8_body(
      a0,a1,a2,a3,a4,a5,a6,b0,b1,b2,b3,b4,b5,b6,c0,c1,c2,c3,c4,c5,c6);
}}
""")

    combined = work / "outlined_gemm_i8_combined.c"
    combined.write_text(_SHIM_SRC.read_text(encoding="utf-8") + "\n" + "".join(wrappers),
                        encoding="utf-8")
    obj = work / "outlined_gemm_i8_rvv.o"
    flags = [*cflags]
    if parallel:
        flags += ["-fopenmp", "-DMERLIN_OUTLINED_PARALLEL=1"]
    cmd = [str(cc), *flags, "-c", str(combined), "-o", str(obj)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not obj.is_file():
        raise OutlinedInt8Unavailable(
            f"outlined INT8 RVV compile failed:\ncmd: {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    (work / "build.json").write_text(json.dumps({
        "command": cmd,
        "parallel": bool(parallel),
        "signatures": {k: list(v) for k, v in sorted(rank2.items())},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return obj


def is_available() -> bool:
    return _SHIM_SRC.is_file()
