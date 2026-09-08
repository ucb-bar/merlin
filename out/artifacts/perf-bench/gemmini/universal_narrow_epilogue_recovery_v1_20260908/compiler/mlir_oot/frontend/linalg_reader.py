"""Reader + lane router for the `linalg-on-tensors` capsule grammar.

The float / whole-model families arrive as `func.func @forward` over `linalg`/`tensor`/`math`
ops instead of `merlin_iface`.  Every op in that form carries the producer's own region
annotation (`prov.region_id`, `prov.family`, `prov.op`, `prov.orig_dtype`), so the module can be
partitioned into regions structurally — by reading typed attributes off real IR, with no text
matching anywhere.

Each region is then placed on a lane against the DATAPATH this target actually has (from the RTL
facts: an i8 x i8 -> i32 systolic mesh).  A region whose family or dtype the mesh does not admit
is placed on the host lane; that is a routing decision, not a refusal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType
from xdsl.ir import Operation
from ..tables import rtl_facts as F
from ..xdsl_utils import is_integer_matmul
from .direct_conv import recognize as recognize_direct_conv

#: The contraction families the mesh can take, and the operand dtype it can take them at.
MESH_FAMILIES = frozenset({"contraction"})
MESH_OPERAND_DTYPES = frozenset({F.OPERAND_DTYPE, "int8", "i8"})

#: The routing plan's own lane keys, as `capsule.schema.json:lanes` names them: a capsule's
#: `lanes.require` / `lanes.forbid` are matched against these, so the plan must speak them.
MESH_LANE = "on_mesh"
HOST_LANE = "scalar_rvv_lane"

#: prov.op -> the ABI whole-op command that expresses the same computation.
OP_TO_OPCODE = {
    "softmax": "SOFTMAX",
    "layernorm": "LAYERNORM",
    "rmsnorm": "RMSNORM",
    "gelu": "GELU",
    "geglu": "GEGLU",
    "reduce_sum": "VREDUCE",
    "attention": "ATTENTION_FULL",
    "sdpa": "ATTENTION_FULL",
    "rope": "ROPE",
}


@dataclass
class Region:
    """One provenance region of the input module."""

    region_id: str
    family: str = ""
    op: str = ""
    dtype: str = ""
    lane: str = "scalar_rvv_lane"
    reason: str = ""
    n_ops: int = 0


@dataclass
class LinalgWorkload:
    entry: str = "forward"
    args: list[tuple[list[int], str]] = field(default_factory=list)
    results: list[tuple[list[int], str]] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    op_names: list[str] = field(default_factory=list)
    #: Number of source operations at the entry block's ownership granularity. A top-level op
    #: owns the operations in its regions (for example a linalg generic's scalar body), matching
    #: the mixed program planner rather than double-counting nested implementation detail.
    source_op_count: int = 0
    #: True when the module is a WHOLE MODEL rather than a slice of one -- it names the parameter
    #: file its non-argument operands come from (`prov.weights_file`).  A whole model is graded by
    #: the model engine (`merlin-compile model --run mesh --verify`), which verifies the compiled
    #: program directly, rather than through the runner's `OUT <name> ... <integers>` readback --
    #: so the readback's integer-only encoding is not a reason to refuse one.
    whole_model: bool = False

    @property
    def mesh_regions(self) -> list[Region]:
        return [r for r in self.regions if r.lane == MESH_LANE]

    @property
    def host_regions(self) -> list[Region]:
        return [r for r in self.regions if r.lane == HOST_LANE]


def _s(op: Operation, key: str) -> str:
    attr = op.attributes.get(key)
    return attr.data if isinstance(attr, StringAttr) else ""


def _shape_dtype(t) -> tuple[list[int], str]:
    if isinstance(t, TensorType):
        return [int(d) for d in t.get_shape()], str(t.get_element_type())
    return [], str(t)


def _normalise_dtype(name: str) -> str:
    return {"float32": "f32", "bfloat16": "bf16", "float16": "f16",
            "int8": "i8", "int32": "i32", "int64": "i64"}.get(name, name)


def place(region: Region) -> Region:
    """Assign `region` to the mesh or to the host lane, from the RTL-derived datapath."""
    if region.family not in MESH_FAMILIES:
        region.lane = HOST_LANE
        region.reason = (f"family {region.family!r} has no datapath on a "
                         f"{F.DIM}x{F.DIM} {F.OPERAND_DTYPE} systolic mesh")
    elif region.dtype not in MESH_OPERAND_DTYPES:
        region.lane = HOST_LANE
        region.reason = (f"the mesh contracts {F.OPERAND_DTYPE} operands; this region is "
                         f"{region.dtype}")
    else:
        region.lane = MESH_LANE
        region.reason = "contraction at the mesh operand dtype"
    return region


def read(module: ModuleOp) -> LinalgWorkload:
    """Partition a `linalg-on-tensors` module into placed regions."""
    wl = LinalgWorkload()
    wl.whole_model = "prov.weights_file" in module.attributes
    seen: dict[str, Region] = {}
    for op in module.walk():
        if not isinstance(op, Operation):
            continue
        if op.name == "func.func":
            sym = op.attributes.get("sym_name")
            wl.entry = getattr(sym, "data", wl.entry) or wl.entry
            ftype = op.attributes.get("function_type") or op.properties.get("function_type")
            if ftype is not None:
                wl.args = [_shape_dtype(t) for t in ftype.inputs.data]
                wl.results = [_shape_dtype(t) for t in ftype.outputs.data]
            if op.regions and op.regions[0].blocks:
                wl.source_op_count = sum(
                    child.name != "func.return" for child in op.regions[0].blocks[0].ops
                )
            continue
        family = op.name.split(".", 1)[0]
        if family in ("linalg", "math", "tensor", "arith"):
            wl.op_names.append(op.name)
        rid = _s(op, "prov.region_id")
        if not rid:
            continue
        region = seen.get(rid)
        if region is None:
            structural_matmul = is_integer_matmul(op)
            structural_conv = recognize_direct_conv(op) is not None
            dtype = _normalise_dtype(_s(op, "prov.orig_dtype"))
            if structural_matmul or structural_conv:
                dtype = _shape_dtype(op.operands[0].type)[1]
            region = Region(rid,
                            family="contraction" if structural_matmul or structural_conv
                            else _s(op, "prov.family"),
                            op=_s(op, "prov.op"),
                            dtype=dtype)
            seen[rid] = region
            wl.regions.append(region)
        elif is_integer_matmul(op) or recognize_direct_conv(op) is not None:
            # Padding/quantization/transpose furniture can be the first op carrying a contraction
            # region id. Do not let that incidental f32 producer permanently route a later canonical
            # i8 contraction to the host lane: placement is decided after the entire region is read.
            region.family = "contraction"
            region.dtype = _shape_dtype(op.operands[0].type)[1]
        region.n_ops += 1
    if not wl.regions:
        # An unannotated module is one anonymous region; its family/dtype come from the entry
        # signature so the placement rule still applies.
        dtype = wl.results[0][1] if wl.results else (wl.args[0][1] if wl.args else "")
        wl.regions.append(Region("region_0", family="unknown", op="", dtype=dtype))
    wl.regions = [place(region) for region in wl.regions]
    return wl
