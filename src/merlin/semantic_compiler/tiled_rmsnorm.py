"""Decompose a selected RMSNorm policy into ordinary typed tile computations.

The reduction policy rounds after each physical row reduction and recursively
after groups of column tiles. It is a named precision choice, not a floating
point reassociation. The target supplies its existing primitive descriptors.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import struct

from .model import ConstantBinding, KernelRequest, SemanticNode, TensorType


@dataclass(frozen=True)
class RMSNormRow:
    row_start: int
    valid_rows: int
    request: KernelRequest


@dataclass(frozen=True)
class RMSNormTiling:
    source: KernelRequest
    tile_type: TensorType
    regions: tuple[RMSNormRow, ...]
    policy: str
    descriptor_records: tuple[tuple[str, str], ...]

    @property
    def shape(self):
        return self.source.nodes[0].type.shape

    @property
    def column_tiles(self):
        return self.shape[1] // self.tile_type.shape[1]

    def record(self):
        return dict(schema="merlin.rmsnorm_tiling.v1", source_request_digest=self.source.digest(),
                    policy=self.policy, shape=list(self.shape), tile_type=self.tile_type.record(),
                    column_tiles=self.column_tiles, reduction="recursive groups of tile width; adjacent-fp32-tree then bf16-rne",
                    descriptors=dict(self.descriptor_records),
                    regions=[dict(row_start=r.row_start, valid_rows=r.valid_rows, request_digest=r.request.digest()) for r in self.regions])


def decompose_rmsnorm(source, *, tile_type, operations, policy):
    roles = {"multiply": 2, "add": 2, "row_sum": 1, "sqrt": 1, "reciprocal": 1}
    if (len(source.nodes) != 3 or source.constants or source.physical_extents or source.boundary_layouts
            or source.output_layouts or set(operations) != set(roles) or tile_type.dtype != "bf16"):
        raise ValueError("RMSNorm requires two plain boundaries and its complete primitive catalog")
    data, gamma, result = source.nodes
    tm, tn = tile_type.shape
    if (data.effect != data.op or gamma.effect != gamma.op or data.op != "input" or gamma.op != "input"
            or data.inputs or gamma.inputs or data.index_maps or gamma.index_maps
            or data.attrs or gamma.attrs or len(data.type.shape) != 2
            or gamma.type.shape != (data.type.shape[1],) or any(node.type.dtype != "bf16" for node in (data, gamma))
            or any(node.type.numerical_policy != "raw-bit-pattern" for node in (data, gamma))
            or result.op != "rmsnorm" or result.effect != "pure" or result.inputs != (data.id, gamma.id) or result.index_maps
            or result.type != TensorType(data.type.shape, "bf16", policy)
            or source.outputs != (result.id,) or source.output_storages != ("external",)
            or source.input_storages != ((data.id, "external"), (gamma.id, "external"))
            or data.type.shape[1] % tn or tm != tn):
        raise ValueError("RMSNorm shapes, boundaries, or selected policy differ")
    attrs = dict(result.attrs)
    if set(attrs) != {"axis", "epsilon_bf16_bits"} or attrs["axis"] != 1:
        raise ValueError("RMSNorm axis or epsilon fields differ")
    eps = attrs["epsilon_bf16_bits"]
    if type(eps) is not int or not 0x2f80 <= eps <= 0x4780:
        raise ValueError("RMSNorm epsilon must lie in [2**-32,2**16]")
    for role, descriptor in operations.items():
        if (len(descriptor.input_storages) != roles[role] or descriptor.output_dtype != "bf16"
                or descriptor.parameter_attrs or descriptor.bundle_name or not descriptor.index_maps):
            raise ValueError("RMSNorm primitive signature differs")
    descriptors = tuple((role, sha256(json.dumps(d.record(), sort_keys=True).encode()).hexdigest()) for role, d in sorted(operations.items()))
    m, n = data.type.shape
    columns = n // tn
    if columns > tn * tn:
        raise ValueError("RMSNorm exceeds two tile-group reduction levels")
    inverse_bits = struct.unpack("<I", struct.pack("<f", 1.0 / n))[0]
    inverse = (inverse_bits + 0x7fff + ((inverse_bits >> 16) & 1)) >> 16
    regions = []
    for row_start in range(0, m, tm):
        nodes, constants = [], []

        def input_(name):
            nodes.append(SemanticNode(name, "input", (), tile_type, (operations["multiply"].required_attrs[0],), "input"))
            return name

        inputs = tuple(input_(f"data_{i}") for i in range(columns))
        gammas = tuple(input_(f"gamma_{i}") for i in range(columns))

        def literal(name, codes, role="multiply"):
            nodes.append(SemanticNode(name, "constant", (), tile_type, (operations[role].required_attrs[0],), "constant"))
            constants.append(ConstantBinding(name, "bf16-le-raw", struct.pack("<" + "H" * len(codes), *codes).hex()))
            return name

        masks = tuple(literal(f"column_mask_{i}", [0x3f80 if col == i else 0 for _ in range(tm) for col in range(tn)]) for i in range(min(columns, tn) if columns > 1 else 0))
        inverse_name = literal("inverse_width", [inverse] * (tm * tn))
        epsilon_name = literal("epsilon", [eps] * (tm * tn), "add")

        def operation(role, children):
            descriptor = operations[role]
            name = f"v{len(nodes)}"
            nodes.append(SemanticNode(name, descriptor.computation, tuple(children), tile_type,
                                      descriptor.required_attrs, index_maps=descriptor.index_maps))
            return name

        def grouped_sum(values):
            if len(values) == 1:
                return values[0]
            groups = []
            for start in range(0, len(values), tn):
                slots = [operation("multiply", (value, masks[i])) for i, value in enumerate(values[start:start + tn])]
                merged = slots[0]
                for slot in slots[1:]:
                    # The selected add joins disjoint lanes exactly. Only the
                    # following row reduction changes numerical precision.
                    merged = operation("add", (merged, slot))
                groups.append(operation("row_sum", (merged,)))
            return grouped_sum(groups)

        partials = [operation("row_sum", (operation("multiply", (value, value)),)) for value in inputs]
        total = grouped_sum(partials)
        mean = operation("multiply", (total, inverse_name))
        shifted = operation("add", (mean, epsilon_name))
        reciprocal = operation("reciprocal", (operation("sqrt", (shifted,)),))
        outputs = tuple(operation("multiply", (operation("multiply", (value, reciprocal)), gamma)) for value, gamma in zip(inputs, gammas, strict=True))
        identity = sha256(json.dumps(dict(source=source.digest(), row=row_start, descriptors=descriptors), sort_keys=True).encode()).hexdigest()
        request = KernelRequest(tuple(nodes), outputs, ("external",) * columns,
                                tuple((key, "external") for key in (*inputs, *gammas)),
                                source.target_identity, source.lowering_policy, constants=tuple(constants), source_identity=identity)
        regions.append(RMSNormRow(row_start, min(tm, m - row_start), request))
    return RMSNormTiling(source, tile_type, tuple(regions), policy, descriptors)
