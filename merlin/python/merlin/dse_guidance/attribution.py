"""Level-1 topology recovery: attribute real IR facts to VLA topology phases.

Level 0 (``topology.py``) consumes a hand-authored sidecar — every interesting fact (K, roles,
loop-invariant state) is asserted by a human. This module is the first step away from that: it
reads the *real* captured ``model.mlir`` and attributes per-matmul facts (count, MACs, weight /
activation bytes, epilogue) to topology phases, recording **explicit provenance** for every
assignment.

What the captures actually carry (verified): per-op ``prov.*`` provenance — ``prov.region_id``
(sequential ``matmul_N``), ``prov.op`` (``matmul`` vs ``addmm`` = epilogue), ``prov.aten``,
shapes/dtypes. What they do NOT carry: a backbone-vs-head marker (``prov.module`` / ``prov.level``
are uniform; the forward args are positional, so there are no weight names). Therefore:

  * per-region **facts** are recovered exactly from the IR (``source: exact_from_ir``);
  * the **role** of a region (backbone_once vs repeated_head) cannot be auto-recovered from a
    flattened capture — it comes from an explicit operator mapping (``source: explicit_mapping``)
    or a low-confidence shape-cluster heuristic, and otherwise stays ``unknown``.

That boundary is the honest Level-1 result, and the persistent ``unknown`` is the motivation for
Level 2 (loop-preserving capture). Nothing is fabricated: an unmapped region reports ``unknown``
and leaves quantification blocked.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from merlin.design_pressure import region as R
from merlin.design_pressure.ingest import mlir_m2m
from merlin.dse_guidance.topology import VlaRuntimeTopology

# Roles a region can be attributed to (mirrors topology / temporal roles).
ROLE_BACKBONE = "backbone_once"
ROLE_REPEATED_HEAD = "repeated_head"
ROLE_PREFIX_BUILDER = "prefix_builder"
ROLE_UNKNOWN = "unknown"


def _read_attr(op, key: str) -> str | None:
    """Read a provenance attribute, trying the real ``prov.*`` namespace then ``m2m.*``."""
    for ns in ("prov.", "m2m."):
        v = op.attributes.get(ns + key)
        if v is not None:
            return str(v).strip().strip('"')
    return None


@dataclass
class MatmulRecord:
    index: int
    region_id: str | None
    op: str | None             # "matmul" | "addmm" | ...
    epilogue: bool             # addmm => matmul+bias => epilogue present
    M: int
    K: int
    N: int
    weight_bytes: int
    activation_bytes: int
    dtype: str | None

    @property
    def macs(self) -> int:
        return self.M * self.K * self.N

    @property
    def signature(self) -> tuple[int, int, int]:
        return (self.M, self.K, self.N)


@lru_cache(maxsize=None)
def extract_matmuls(capture_dir: str) -> tuple[MatmulRecord, ...]:
    """Per-matmul records read from a capture's ``model.mlir`` (empty tuple if it won't parse)."""
    path = f"{capture_dir}/model.mlir"
    try:
        module = mlir_m2m._parse_module(open(path, encoding="utf-8").read())
    except Exception:
        return ()
    out: list[MatmulRecord] = []
    for i, op in enumerate(o for o in module.walk() if o.name == "linalg.matmul"):
        ls, ld = mlir_m2m._shape_dtype(str(op.operands[0].type))
        rs, rd = mlir_m2m._shape_dtype(str(op.operands[1].type))
        if not (ls and rs and len(ls) == 2 and len(rs) == 2):
            continue
        M, K = ls
        N = rs[-1]
        op_kind = _read_attr(op, "op")
        out.append(MatmulRecord(
            index=i,
            region_id=_read_attr(op, "region_id"),
            op=op_kind,
            epilogue=(op_kind == "addmm"),
            M=M, K=K, N=N,
            weight_bytes=K * N * R.dtype_bytes(rd),
            activation_bytes=(M * K + M * N) * R.dtype_bytes(ld),
            dtype=rd,
        ))
    return tuple(out)


@dataclass
class RoleFacts:
    matmul_count: int = 0
    macs: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0
    has_epilogue: bool = False

    def add(self, rec: MatmulRecord) -> None:
        self.matmul_count += 1
        self.macs += rec.macs
        self.weight_bytes += rec.weight_bytes
        self.activation_bytes += rec.activation_bytes
        self.has_epilogue = self.has_epilogue or rec.epilogue

    def scaled(self, k: int) -> dict:
        """Per-invocation facts (as in IR) plus a K-scaled total for repeated regions."""
        return {
            "matmul_count": self.matmul_count,
            "macs_per_invocation": self.macs,
            "weight_bytes": self.weight_bytes,           # weights are reused, not ×K
            "activation_bytes_per_invocation": self.activation_bytes,
            "invocations": k,
            "macs_total": self.macs * k,
            "activation_bytes_total": self.activation_bytes * k,
        }


@dataclass
class RegionRoleAttribution:
    role: str
    attribution_status: str           # "attributed" | "unknown"
    confidence: float
    source: str                       # explicit_mapping | shape_cluster | unknown
    invocations: int
    facts: dict
    matmul_indices: list[int] = field(default_factory=list)
    reason: str = ""


@dataclass
class RegionAttribution:
    workload: str
    attribution_status: str           # "attributed" | "partial" | "unknown"
    regions: list[RegionRoleAttribution]
    repeated_signatures: list[dict] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    parsed: bool = True

    def role(self, role: str) -> RegionRoleAttribution | None:
        return next((r for r in self.regions if r.role == role), None)


def _match(rule: dict, rec: MatmulRecord) -> bool:
    m = rule.get("match", {})
    if "region_ids" in m and rec.region_id in set(m["region_ids"]):
        return True
    if "shape_signature" in m and list(rec.signature) == list(m["shape_signature"]):
        return True
    rng = m.get("region_id_range")
    if rng and rec.region_id and rec.region_id.startswith("matmul_"):
        try:
            n = int(rec.region_id.split("_")[-1])
            return rng[0] <= n <= rng[1]
        except ValueError:
            return False
    return False


def _repeated_signatures(records, min_count: int = 3) -> list[dict]:
    """Shape signatures that recur (the repeated transformer/denoise layers)."""
    counts = Counter(r.signature for r in records)
    out = []
    for sig, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        if c >= min_count:
            out.append({"signature": list(sig), "count": c})
    return out


def attribute(capture_dir: str, topo: VlaRuntimeTopology,
              mapping_rules: dict | None = None) -> RegionAttribution:
    """Attribute IR matmul facts to topology phases (explicit mapping > shape heuristic > unknown).

    Facts are always exact-from-IR; only the *role assignment* may be operator-supplied. With no
    mapping and no usable heuristic, every region is ``unknown`` and quantification stays blocked.
    """
    return attribute_records(extract_matmuls(capture_dir), topo, mapping_rules)


def attribute_records(records, topo: VlaRuntimeTopology,
                      mapping_rules: dict | None = None) -> RegionAttribution:
    """Pure attribution over already-extracted matmul records (unit-testable core)."""
    if not records:
        return RegionAttribution(workload=topo.workload, attribution_status="unknown",
                                 regions=[], parsed=False,
                                 unresolved=["capture did not parse; no IR facts extractable"])

    K = topo.K
    rules = (mapping_rules or {}).get("rules", [])
    assigned: dict[str, RoleFacts] = {}
    assigned_idx: dict[str, list[int]] = {}
    sources: dict[str, str] = {}
    unknown = RoleFacts()
    unknown_idx: list[int] = []

    for rec in records:
        role = None
        source = None
        for rule in rules:                       # 1) explicit operator mapping (highest trust)
            if _match(rule, rec):
                role, source = rule["role"], "explicit_mapping"
                break
        if role is None:                          # 2..n) auto-recovery unavailable from flat IR
            unknown.add(rec)
            unknown_idx.append(rec.index)
            continue
        assigned.setdefault(role, RoleFacts()).add(rec)
        assigned_idx.setdefault(role, []).append(rec.index)
        sources[role] = source

    regions: list[RegionRoleAttribution] = []
    for role, facts in assigned.items():
        invs = K if role == ROLE_REPEATED_HEAD else 1   # head ×K; backbone ×1 — never ×K backbone
        regions.append(RegionRoleAttribution(
            role=role, attribution_status="attributed", confidence=0.9,
            source=sources[role], invocations=invs, facts=facts.scaled(invs),
            matmul_indices=assigned_idx[role],
            reason=f"{facts.matmul_count} matmuls assigned by {sources[role]}"))
    if unknown.matmul_count:
        regions.append(RegionRoleAttribution(
            role=ROLE_UNKNOWN, attribution_status="unknown", confidence=0.0, source="unknown",
            invocations=1, facts=unknown.scaled(1), matmul_indices=unknown_idx,
            reason="no explicit mapping rule matched; role not recoverable from the flat capture"))

    status = ("attributed" if assigned and not unknown.matmul_count
              else "partial" if assigned else "unknown")
    unresolved = []
    if unknown.matmul_count:
        unresolved.append(
            f"{unknown.matmul_count}/{len(records)} matmuls unattributed: backbone-vs-head role "
            "is not encoded in the capture (prov.module/level uniform; args positional)")
    if not rules:
        unresolved.append("no mapping rules supplied; supply region_ids / shape_signature rules "
                          "to attribute roles, or add loop-preserving capture (Level 2)")
    return RegionAttribution(
        workload=topo.workload, attribution_status=status, regions=regions,
        repeated_signatures=_repeated_signatures(records), unresolved=unresolved)


def to_yaml_obj(attr: RegionAttribution) -> dict:
    return {
        "workload": attr.workload,
        "topology_recovery": {
            "level": 1,
            "attribution_status": attr.attribution_status,
            "regions": [
                {"role": r.role, "attribution_status": r.attribution_status,
                 "confidence": r.confidence, "source": r.source,
                 "invocations": r.invocations, "facts": r.facts,
                 "matmul_count_attributed": len(r.matmul_indices), "reason": r.reason}
                for r in attr.regions
            ],
            "repeated_shape_signatures": attr.repeated_signatures,
            "unresolved": attr.unresolved,
        },
    }
