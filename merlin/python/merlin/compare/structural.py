"""LAYER 2 — STRUCTURAL: per-config CCA (the Common Compute Abstraction of each config's matmul).

REUSES ``kernels.cca`` (the CCA dataclasses + ``lift_asm``) and ``kernels.decode.rvv`` (the decoder).

v1 ingests the cached per-(kernel, shape) decode from ``artifacts/ceiling/
kernel_breakdown_decode.json`` and lifts each row into a real ``cca.CCA`` (so the comparator and
action catalog consume genuine CCA objects, identical to the live path). The live path —
``decode(model_o)`` then ``cca.lift_asm(stream, ...)`` over a host-clang-built .o — is wired behind
``decode_o()`` for the rebuild seam (no board: host clang -> model.o -> objdump).

The cached decode is the authoritative structural fingerprint that ``kernel_breakdown.md`` was built
from; lifting it back into CCA is honest (same fields, same source), deterministic, and avoids a
redundant per-shape rebuild while keeping the rebuild reachable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from merlin.kernels.cca import CCA, ComputeFacet, VectorFacet, lift_asm

from .spec import Config, Workload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


_DECODE_SOURCE = "out/artifacts/ceiling/kernel_breakdown_decode.json"

# spec config name -> kernel-breakdown-decode "kernel" field value.
_DECODE_KERNEL = {
    "xnnpack": "xnnpack",
    "openblas": "openblas",
    "ours_wholemodel": "ours_wholemodel",
    "ours_wholemodel_vf": "ours_v3",   # the .vf wholemodel carries the v3 register-blocked decode
    "ours_v3": "ours_v3",
    "ours_tiled": "ours_vfmacc_tiled",
}

# workload name -> the decode "shape" string that represents that workload's dominant matmul.
# (The breakdown decoded one representative matmul per model; the cube_64 / ukernel rows stand for
# the isolated-shape and shape-independent expert micro-kernels.)
_WORKLOAD_SHAPE = {
    "openvla": "openvla_proj_17x192x576",
    "rdt2": "rdt2_attn_28x1024x1024",
    "bitvla": "cube_64",          # bitvla's dominant projection decodes like the cube micro-kernel
}


def _row_to_cca(row: dict, *, source: str) -> CCA:
    """Lift a cached kernel_breakdown_decode row into a real CCA (same fields the live lift fills)."""
    lmul = row.get("lmul")
    sew = row.get("sew")
    vl_strategy = row.get("vl_strategy")
    mr = row.get("MR")
    # register_block matches the live lift's (mr, ("vsetvlmax", lmul)) shape when both are known.
    reg_block = (mr, ("vsetvlmax", lmul)) if (mr and sew and lmul) else None
    return CCA(
        op="matmul", backend=["rvv"],
        compute=ComputeFacet(
            op="matmul",
            contraction_form=row.get("contraction_form"),
            register_block=reg_block,
            accumulator_resident=row.get("accumulator_resident"),
            nr_is_vsetvlmax=row.get("nr_is_vsetvlmax"),
        ),
        vector=VectorFacet(sew=sew, lmul=lmul, vl_strategy=vl_strategy),
        provenance={
            "level": "asm", "source": source, "confidence": "high",
            "decode_kernel": row.get("kernel"), "decode_shape": row.get("shape"),
            # carry the .vf-vs-.vv counts so attribution can cite them (kernel_breakdown.md evidence).
            "fma_loop_vfmacc_vf": row.get("fma_loop_vfmacc_vf"),
            "fma_loop_vfmacc_vv": row.get("fma_loop_vfmacc_vv"),
            "fma_loop_acc_spills": row.get("fma_loop_acc_spills"),
            "n_insns": row.get("n_insns"),
        },
    )


def _load_decode(root: Path) -> list[dict]:
    p = root / _DECODE_SOURCE
    if not p.is_file():
        return []
    return json.loads(p.read_text())


def _pick_row(rows: list[dict], kernel: str, shape: str | None) -> dict | None:
    cand = [r for r in rows if r.get("kernel") == kernel]
    if not cand:
        return None
    if shape is not None:
        exact = [r for r in cand if r.get("shape") == shape]
        if exact:
            return exact[0]
    # shape-independent micro-kernels (xnnpack/openblas) have one row.
    return cand[0]


def cca_for(cfg: Config, wl: Workload | None = None, *, root: Path | None = None) -> CCA | None:
    """Per-config CCA. ``baseline`` has no vectorized matmul micro-kernel decode (scalar/hand_v0) so
    returns None — surfaced honestly, not faked. For a workload, picks the representative shape row;
    without one, picks the config's shape-independent / cube row."""
    root = root or _repo_root()
    if cfg.kind == "baseline":
        return None
    kernel = _DECODE_KERNEL.get(cfg.name)
    if kernel is None:
        return None
    rows = _load_decode(root)
    shape = None
    if wl is not None and wl.kind == "model":
        shape = _WORKLOAD_SHAPE.get(wl.name)
    row = _pick_row(rows, kernel, shape)
    if row is None and shape is not None:
        row = _pick_row(rows, kernel, None)
    if row is None:
        return None
    return _row_to_cca(row, source=_DECODE_SOURCE)


def cca_table(spec, *, root: Path | None = None) -> dict:
    """{config_name: CCA|None} — one representative CCA per config (cached so it is not redone).

    Uses the first model workload as the representative shape; falls back to shape-independent."""
    root = root or _repo_root()
    model_wls = [w for w in spec.workloads if w.kind == "model"]
    rep = model_wls[0] if model_wls else None
    out: dict[str, CCA | None] = {}
    for cfg in spec.configs:
        out[cfg.name] = cca_for(cfg, rep, root=root)
    return out


# ---- live rebuild seam (host clang -> model.o -> decode -> lift_asm). NOT used by v1 ingest. ----
def decode_o(model_o: str | Path, *, op: str = "matmul", source: str | None = None) -> CCA:
    """Decode a host-built .o and lift it to a CCA (the live structural path). REUSES
    ``decode.rvv.decode`` + ``cca.lift_asm`` exactly as the ceiling drivers do. v1 does not call this
    (it ingests the cached decode); it is the seam for re-decoding a freshly rebuilt kernel."""
    from merlin.kernels.decode.rvv import decode
    stream = decode(str(model_o))
    return lift_asm(stream, op=op, source=source or str(model_o), backend="rvv")
