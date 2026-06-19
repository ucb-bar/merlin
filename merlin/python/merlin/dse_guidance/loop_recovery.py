"""Loop-preserving capture recovery (P21-S1).

When a capture is taken through a ``torch.while_loop`` wrapper, model2MLIR's
CompGen FXImporter lowers the K-step denoise/decode loop to an ``scf.for(0,K,1)``
(see model2MLIR ``m2m/ir/import_fx.py``). This module parses that structure back
out of the flat ``model.mlir`` so the DSE front-end can report — *from the IR,
not from a config sidecar or an fqn heuristic* —:

  * **K**  : the loop trip count (the ``scf.for`` upper bound, a constant tagged
             ``prov.op = "while_loop"``) → ``recovered_from_ir`` (drop ``assumed``).
  * **loop-carried state** : the ``iter_args`` (the loop counter, the denoise
             latent, and/or the KV cache) with shapes + byte sizes.
  * **repeated region**    : the ``scf.for`` body = the structurally-recovered
             ``repeated_head`` (no fqn heuristic needed).

This is the evidence that flips the assumed-K / region-roles / KV-state /
loop-carried-state caveats to resolved-with-IR. Structural only — no timing,
no speedup. Bytes are shape×dtype (weight VALUES are irrelevant to the contract).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_DTYPE_BYTES = {
    "f64": 8, "f32": 4, "f16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1,
    "ui8": 1, "si64": 8, "si32": 4,
}

# %933 = arith.constant {prov.op = "while_loop", ...} 7 : index
_WL_CONST = re.compile(
    r'(%\S+)\s*=\s*arith\.constant\s*\{[^}]*prov\.op\s*=\s*"while_loop"[^}]*\}\s*(\d+)\s*:\s*index')
# %936, ... = scf.for %iv = %lb to %ub step %st iter_args(...) -> (T1, T2, ...) {
_SCF_FOR = re.compile(
    r'scf\.for\s+(%\S+)\s*=\s*(%\S+)\s+to\s+(%\S+)\s+step\s+(%\S+)\s+iter_args\((.*?)\)\s*->\s*\((.*?)\)')
_TENSOR = re.compile(r'tensor<([^>]*)>')


@dataclass
class CarriedState:
    index: int
    shape: list[int]
    dtype: str
    bytes: int | None
    role: str          # counter | latent | kv_cache | token_buffer | other

    def to_dict(self) -> dict:
        return {"index": self.index, "shape": self.shape, "dtype": self.dtype,
                "bytes": self.bytes, "role": self.role}


@dataclass
class LoopRecovery:
    workload: str
    present: bool
    K: int | None = None
    K_source: str = "recovered_from_ir"
    n_iter_args: int = 0
    carried_state: list[CarriedState] = field(default_factory=list)
    repeated_region_op_count: int = 0
    kv_cache_bytes: int | None = None
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "workload": self.workload,
            "loop_preserved": self.present,
            "K": self.K,
            "K_source": self.K_source if self.present else "assumed_or_config",
            "n_loop_carried_iter_args": self.n_iter_args,
            "loop_carried_state": [c.to_dict() for c in self.carried_state],
            "repeated_region_op_count": self.repeated_region_op_count,
            "kv_cache_bytes": self.kv_cache_bytes,
            "evidence": self.evidence,
        }


def _parse_tensor_type(t: str) -> tuple[list[int], str]:
    """'2x1x4x27x128xf32' -> ([2,1,4,27,128], 'f32'); 'i64' -> ([], 'i64')."""
    parts = t.split("x")
    dtype = parts[-1]
    dims: list[int] = []
    for p in parts[:-1]:
        try:
            dims.append(int(p))
        except ValueError:
            # dynamic dim '?' or symbolic — record as -1 (unknown)
            dims.append(-1)
    return dims, dtype


def _bytes_of(dims: list[int], dtype: str) -> int | None:
    if any(d < 0 for d in dims):
        return None
    n = 1
    for d in dims:
        n *= d
    eb = _DTYPE_BYTES.get(dtype)
    return n * eb if eb is not None else None


def _classify(idx: int, dims: list[int], dtype: str) -> str:
    is_int = dtype.startswith(("i", "si", "ui"))
    numel = 1
    for d in dims:
        numel *= max(d, 1)
    if is_int and numel <= 64:
        # scalar counter (tensor<i64>/[]) vs a small token buffer ([1xN])
        return "counter" if not dims or numel == 1 else "token_buffer"
    if not is_int and len(dims) >= 4:
        return "kv_cache"          # rank>=4 float carry = static KV cache
    if not is_int and len(dims) == 3:
        return "latent"            # action latent [B, horizon, dim]
    return "other"


def recover_loop(model_mlir_path: str | Path, workload: str = "") -> LoopRecovery:
    """Parse the ``scf.for`` lowered from ``torch.while_loop`` out of ``model.mlir``."""
    path = Path(model_mlir_path)
    if not path.exists():
        return LoopRecovery(workload=workload, present=False)
    text = path.read_text()

    # 1) the while_loop bound constants (ssa -> integer value)
    wl_consts = {m.group(1): int(m.group(2)) for m in _WL_CONST.finditer(text)}
    if not wl_consts:
        return LoopRecovery(workload=workload, present=False)

    lines = text.splitlines()
    # 2) the scf.for whose lb/ub/step are while_loop constants
    for li, line in enumerate(lines):
        m = _SCF_FOR.search(line)
        if not m:
            continue
        _iv, lb, ub, st, _iargs, rtypes = m.groups()
        if not ({lb, ub, st} & set(wl_consts)):
            continue
        # K = the upper-bound constant (lb=0, step=1 are <= K)
        K = wl_consts.get(ub)
        if K is None:
            K = max(wl_consts.values())

        carried: list[CarriedState] = []
        for i, tt in enumerate(_TENSOR.findall(rtypes)):
            dims, dtype = _parse_tensor_type(tt)
            b = _bytes_of(dims, dtype)
            carried.append(CarriedState(i, dims, dtype, b, _classify(i, dims, dtype)))
        kv_bytes = sum((c.bytes or 0) for c in carried if c.role == "kv_cache") or None

        # 3) repeated region op count: SSA-defining lines between scf.for and its scf.yield
        body_ops = 0
        depth = 0
        for bl in lines[li:]:
            depth += bl.count("{") - bl.count("}")
            if "scf.yield" in bl:
                break
            if " = " in bl and "scf.for" not in bl:
                body_ops += 1
        return LoopRecovery(
            workload=workload, present=True, K=K, K_source="recovered_from_ir",
            n_iter_args=len(carried), carried_state=carried,
            repeated_region_op_count=body_ops, kv_cache_bytes=kv_bytes,
            evidence=f"scf.for(0,{K},1) lowered from torch.while_loop "
                     f"(bounds tagged prov.op=while_loop); {len(carried)} iter_args",
        )
    return LoopRecovery(workload=workload, present=False)
