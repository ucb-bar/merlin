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

from dataclasses import dataclass, field
from pathlib import Path

from merlin.common import mlir_query

_DTYPE_BYTES = {
    "f64": 8, "f32": 4, "f16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1,
    "ui8": 1, "si64": 8, "si32": 4,
}


def _while_loop_for(module):
    """The ``scf.for`` (xDSL ForOp) lowered from ``torch.while_loop``: its upper-bound operand is an
    ``arith.constant`` tagged ``prov.op = "while_loop"``. Returns ``(forop, K)`` or ``(None, None)``.
    Read structurally from the parsed IR — the loop bounds/iter_args/body are real IR, not text."""
    for op in mlir_query.walk(module, "scf.for"):
        ub_owner = op.operands[1].owner
        if (mlir_query.op_name(ub_owner) == "arith.constant"
                and mlir_query.attr_str(ub_owner, "prov.op") == "while_loop"):
            return op, ub_owner.value.value.data
    return None, None


def _for_body(forop):
    return forop.body.blocks[0]


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


@dataclass
class ResidencyCert:
    """IR-proven residency split for a loop-preserving capture (P21 GAP-C)."""
    workload: str
    present: bool
    K: int | None = None
    n_loop_invariant_operands: int = 0     # referenced in body, defined OUTSIDE -> resident-eligible
    n_loop_carried: int = 0                # iter_args -> genuinely per-step state
    n_body_defs: int = 0
    resident_proof: str = ""
    evidence: str = "recovered_from_ir"

    def to_dict(self) -> dict:
        return {
            "workload": self.workload,
            "loop_preserved": self.present,
            "K": self.K,
            "n_loop_invariant_operands": self.n_loop_invariant_operands,
            "n_loop_carried_iter_args": self.n_loop_carried,
            "n_body_defs": self.n_body_defs,
            "resident_proof": self.resident_proof,
            "evidence": self.evidence,
        }


def residency_from_ir(model_mlir_path: str | Path, workload: str = "") -> ResidencyCert:
    """Classify the scf.for body's value references into loop-invariant (defined OUTSIDE the
    region -> reused every iteration -> resident-eligible) vs loop-carried (iter_args -> genuinely
    per-step state). This PROVES residency from the IR region boundary, replacing the prior
    'weights are loop-invariant (assumed) + configured K' framing. Structural; no bytes/timing."""
    path = Path(model_mlir_path)
    if not path.exists():
        return ResidencyCert(workload=workload, present=False)
    forop, K = _while_loop_for(mlir_query.parse(path))
    if forop is None:
        return ResidencyCert(workload=workload, present=False)
    body = _for_body(forop)
    block_args = set(body.args)                       # iv + iter_args (the loop-carried bound values)
    # Walk the body region: defined = every SSA result produced INSIDE the region; used = every
    # operand referenced. Loop-invariant = used but neither defined in-region nor a carried block arg
    # (i.e. defined OUTSIDE -> reused read-only every iteration -> resident-eligible).
    defined: set = set()
    used: set = set()

    def _walk(region):
        for blk in region.blocks:
            for op in blk.ops:
                defined.update(op.results)
                used.update(op.operands)
                for r in op.regions:
                    _walk(r)

    _walk(forop.body)
    invariant = used - defined - block_args
    return ResidencyCert(
        workload=workload, present=True, K=K,
        n_loop_invariant_operands=len(invariant),
        n_loop_carried=len(block_args) - 1,           # exclude the IV
        n_body_defs=len(defined),
        resident_proof=(f"{len(invariant)} operands referenced read-only in the scf.for body but "
                        f"defined outside the region -> loop-invariant across K={K} iterations "
                        "(resident-eligible); avoidable reload = resident_bytes x (K-1)"),
    )


def recover_loop(model_mlir_path: str | Path, workload: str = "") -> LoopRecovery:
    """Parse the ``scf.for`` lowered from ``torch.while_loop`` out of ``model.mlir``."""
    path = Path(model_mlir_path)
    if not path.exists():
        return LoopRecovery(workload=workload, present=False)
    forop, K = _while_loop_for(mlir_query.parse(path))
    if forop is None:
        return LoopRecovery(workload=workload, present=False)

    # iter_args are the scf.for operands after (lb, ub, step) — their types are the loop-carried state.
    iter_args = forop.operands[3:]
    carried: list[CarriedState] = []
    for i, val in enumerate(iter_args):
        dims, dtype = mlir_query.type_shape_dtype(val.type)
        carried.append(CarriedState(i, dims, dtype, _bytes_of(dims, dtype), _classify(i, dims, dtype)))
    kv_bytes = sum((c.bytes or 0) for c in carried if c.role == "kv_cache") or None

    # repeated-region op count: every result-defining op in the body region (recursively), the ops
    # that re-execute each of the K iterations.
    body_ops = 0

    def _count(region):
        nonlocal body_ops
        for blk in region.blocks:
            for op in blk.ops:
                if op.results and mlir_query.op_name(op) not in ("scf.for", "scf.yield"):
                    body_ops += 1
                for r in op.regions:
                    _count(r)

    _count(forop.body)
    return LoopRecovery(
        workload=workload, present=True, K=K, K_source="recovered_from_ir",
        n_iter_args=len(carried), carried_state=carried,
        repeated_region_op_count=body_ops, kv_cache_bytes=kv_bytes,
        evidence=f"scf.for(0,{K},1) lowered from torch.while_loop "
                 f"(bounds tagged prov.op=while_loop); {len(carried)} iter_args",
    )
