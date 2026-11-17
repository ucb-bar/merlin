"""Consumer-owned runtime logical view proofs from original standard SSA.

Reconstruct source axes and mixed coordinates independently of the producer.
LLVM a47bddcc TensorOps.cpp/ViewLikeInterface.cpp define the mixed slice and
nonempty extraction contracts. Source records cannot supply executable facts.
"""

from xdsl.dialects import arith, builtin, cf, scf, tensor
from xdsl.ir import Operation

from .execution_call_contracts import CallContracts
from .execution_runtime_views import RuntimeViewContracts

POLICY = "runtime-logical-coordinates-v1"
PROFILE = "t2m.runtime-source-tensor-views.v1"
BINDING_PROFILE = "t2m.execution-checked-call-views.v3"
SCHEMA = "t2m.runtime-source-coordinates.v1"


def _axes(value):
    typ = value.type
    if (
        not isinstance(typ, builtin.TensorType)
        or not isinstance(typ.encoding, builtin.NoneAttr)
        or len(typ.get_shape()) > 8
        or any(n < 0 and n != builtin.DYNAMIC_INDEX for n in typ.get_shape())
    ):
        raise ValueError("runtime source view requires an unencoded bounded tensor rank")
    return tuple(typ.get_shape())


def _field(operation, name):
    field = operation.properties.get(name, operation.attributes.get(name))
    if not isinstance(field, builtin.DenseArrayBase) or field.elt_type != builtin.i64:
        raise ValueError("runtime source coordinates lack a complete typed field")
    return list(field.iter_values())


def _extraction(operation):
    if isinstance(operation, tensor.ExtractSliceOp):
        return operation
    if not isinstance(operation, scf.IfOp):
        return None
    branches = []
    for region in (operation.true_region, operation.false_region):
        if len(region.blocks) != 1 or region.block.args:
            raise ValueError("runtime view has an open slice branch")
        operations = list(region.block.ops)
        if len(operations) != 2 or not isinstance(operations[-1], scf.YieldOp):
            raise ValueError("runtime view branch is not a pure slice refinement")
        if tuple(operations[-1].operands) != tuple(operations[0].results):
            raise ValueError("runtime view branch returns a different value")
        branches.append(operations[0])
    extract, empty = branches
    if (
        not isinstance(extract, tensor.ExtractSliceOp)
        or not isinstance(empty, tensor.EmptyOp)
        or len(operation.results) != 1
        or extract.result.type != operation.results[0].type
        or empty.tensor.type != extract.result.type
        or tuple(empty.operands) != tuple(extract.sizes)
        or any(operation.is_ancestor(value.owner) for value in extract.operands)
    ):
        raise ValueError("runtime view needs exact extraction/empty shapes and dominating coordinates")
    return extract


def _chain(value, root):
    _axes(root)
    chain = []
    while value is not root:
        _axes(value)
        if value.type.element_type != root.type.element_type or len(chain) >= 32:
            raise ValueError("runtime view changes storage or exceeds the source path budget")
        operation = value.owner
        if not isinstance(operation, Operation) or len(operation.results) != 1 or not operation.operands:
            raise ValueError("runtime view has no executable source root")
        extract = _extraction(operation)
        parent = extract.source if extract is not None else operation.operands[0]
        source, target = _axes(parent), _axes(value)
        if len(source) != len(target):
            raise ValueError("runtime source view changes rank")
        if extract is not None:
            for name in ("offsets", "sizes", "strides"):
                coordinates = _field(extract, "static_" + name)
                if (
                    len(coordinates) != len(source)
                    or coordinates.count(builtin.DYNAMIC_INDEX) != len(getattr(extract, name))
                    or any(n < 0 and n != builtin.DYNAMIC_INDEX for n in coordinates)
                ):
                    raise ValueError("runtime source slice has inconsistent mixed operands")
        elif operation.name == "linalg.transpose":
            permutation = _field(operation, "permutation")
            if (
                len(operation.operands) != 2
                or sorted(permutation) != list(range(len(source)))
                or len(operation.regions) != 1
                or len(operation.regions[0].blocks) != 1
            ):
                raise ValueError("runtime source transpose has an incomplete axis map")
            body = operation.regions[0].block
            if (
                len(body.args) not in (1, 2)
                or len(tuple(body.ops)) != 1
                or body.last_op.name != "linalg.yield"
                or tuple(body.last_op.operands) != (body.args[0],)
            ):
                raise ValueError("runtime source transpose changes element words")
        elif isinstance(operation, tensor.CastOp):
            if len(operation.operands) != 1 or any(
                a >= 0 and b >= 0 and a != b for a, b in zip(source, target, strict=True)
            ):
                raise ValueError("runtime source cast changes a known axis")
        else:
            raise ValueError("runtime view needs standard transpose/slice/cast computation")
        chain.append((operation, extract))
        value = parent
    return list(reversed(chain))


class SourceCoordinates(RuntimeViewContracts):
    def require(self, predicate, message):
        self.emit(cf.AssertOp(predicate, "Triton executed call: source view " + message), "source-view-contract")

    def equal(self, left, right, message):
        self.require(self.emit(arith.CmpiOp(left, right, "eq"), "source-view-equality"), message)

    def add(self, left, right):
        value = self.emit(arith.AddiOp(left, right), "source-view-offset-add")
        self.require(
            self.emit(arith.CmpiOp(value, self.constant(0), "sge"), "source-view-add-fits"),
            "coordinate addition overflow",
        )
        return value

    def extents(self, value):
        return [
            self.constant(n)
            if n >= 0
            else self.emit(tensor.DimOp(value, self.constant(i)), "runtime-view-actual-dimension")
            for i, n in enumerate(_axes(value))
        ]

    def mixed(self, operation, field):
        values = iter(getattr(operation, field))
        return [
            next(values) if n == builtin.DYNAMIC_INDEX else self.constant(n)
            for n in _field(operation, "static_" + field)
        ]

    def geometry(self, view, root):
        chain = _chain(view, root)
        dimensions = self.extents(root)
        self.elements(dimensions)
        strides = [self.elements(dimensions[i + 1 :]) for i in range(len(dimensions))]
        offset, steps = self.constant(0), []
        for operation, extraction in chain:
            actual = self.extents(operation.results[0])
            if operation.name == "linalg.transpose":
                permutation = _field(operation, "permutation")
                expected = [dimensions[i] for i in permutation]
                strides = [strides[i] for i in permutation]
                steps.append(
                    {"op": operation.name, "permutation": permutation, "result_type": str(operation.results[0].type)}
                )
            elif extraction is not None:
                origins, sizes, increments = [
                    self.mixed(extraction, field) for field in ("offsets", "sizes", "strides")
                ]
                for coordinates in (origins, sizes, increments):
                    self.elements(coordinates)
                if isinstance(operation, scf.IfOp):
                    nonempty = self.emit(arith.ConstantOp.from_int_and_width(1, builtin.i1), "source-view-nonempty")
                    for size in sizes:
                        positive = self.emit(arith.CmpiOp(size, self.constant(0), "sgt"), "source-view-positive-size")
                        nonempty = self.emit(arith.AndIOp(nonempty, positive), "source-view-nonempty")
                    self.equal(operation.cond, nonempty, "empty-slice branch differs from actual dimensions")
                for origin, size, increment, bound in zip(origins, sizes, increments, dimensions, strict=True):
                    zero, one = self.constant(0), self.constant(1)
                    self.require(
                        self.emit(arith.CmpiOp(increment, zero, "sgt"), "source-view-positive-stride"),
                        "slice stride is not positive",
                    )
                    self.require(
                        self.emit(arith.CmpiOp(origin, bound, "sle"), "source-view-origin-bound"),
                        "slice origin exceeds its source dimension",
                    )
                    nonempty = self.emit(arith.CmpiOp(size, zero, "sgt"), "source-view-nonempty-axis")
                    prior = self.emit(arith.SubiOp(size, one), "source-view-last-element")
                    count = self.emit(arith.SelectOp(nonempty, prior, zero), "source-view-safe-last-element")
                    last = self.add(origin, self.elements([count, increment]))
                    within = self.emit(arith.CmpiOp(last, bound, "slt"), "source-view-last-in-bounds")
                    empty = self.emit(arith.CmpiOp(size, zero, "eq"), "source-view-empty-axis")
                    self.require(
                        self.emit(arith.OrIOp(empty, within), "source-view-slice-in-bounds"),
                        "slice exceeds its source dimension",
                    )
                for origin, stride in zip(origins, strides, strict=True):
                    offset = self.add(offset, self.elements([origin, stride]))
                strides = [
                    self.elements([stride, increment]) for stride, increment in zip(strides, increments, strict=True)
                ]
                expected = sizes
                steps.append(
                    {
                        "op": operation.name,
                        "result_type": str(operation.results[0].type),
                        **{field: _field(extraction, "static_" + field) for field in ("offsets", "sizes", "strides")},
                    }
                )
            else:
                expected = dimensions
                steps.append({"op": operation.name, "result_type": str(operation.results[0].type)})
            for observed, required in zip(actual, expected, strict=True):
                self.equal(observed, required, "operation result dimension differs from source coordinates")
            dimensions = actual
        return dimensions, strides, offset, {"schema": SCHEMA, "root_order": "logical-dense-row-major", "steps": steps}

    def check_views(self, records, mapping, by_name, scalars, provenance):
        if not isinstance(records, dict) or not records or len(records) > 64:
            raise ValueError("runtime source views require complete bounded records")
        pointers = provenance.get("source_argument_binding", {}).get("bindings", [])
        checked = {}
        for name in sorted(records):
            record = records[name]
            config = record.get("binding") if isinstance(record, dict) else None
            if (
                not isinstance(name, str)
                or name not in by_name
                or name in mapping.values()
                or not isinstance(config, dict)
                or set(config) != {"root", "source_arg", "shape", "strides", "offset"}
                or not isinstance(config["root"], str)
                or config["root"] not in mapping
                or type(config["source_arg"]) is not int
                or not 0 <= config["source_arg"] < len(pointers)
                or not isinstance(config["shape"], list)
                or not isinstance(config["strides"], list)
                or len(config["shape"]) != len(config["strides"])
            ):
                raise ValueError("runtime source view loses its original allocation/coordinate binding")
            fields = [*config["shape"], *config["strides"], config["offset"]]
            if any(not isinstance(field, str) or field not in scalars for field in fields):
                raise ValueError("runtime source view lacks an actual typed kernel scalar")
            if pointers[config["source_arg"]] != {
                "kind": "pointer",
                "allocation": config["root"],
                "offset": config["offset"],
            }:
                raise ValueError("runtime source view changes the source pointer root or offset binding")
            root_name = mapping[config["root"]]
            root, view = by_name[root_name], by_name[name]
            if len(_axes(view)) != len(config["shape"]):
                raise ValueError("runtime source view loses a source axis")
            dimensions, strides, offset, layout = self.geometry(view, root)
            for actual, field in zip([*dimensions, *strides, offset], fields, strict=True):
                value = scalars[field]
                if not isinstance(value.type, builtin.IndexType):
                    if not isinstance(value.type, builtin.IntegerType) or value.type.bitwidth not in (8, 16, 32, 64):
                        raise ValueError("runtime source coordinates need signed integer/index operands")
                    value = CallContracts(self.emit).index(value, builtin.IndexType())
                self.equal(actual, value, "kernel argument disagrees with original view coordinates: " + field)
            checked[name] = {
                "binding": config,
                "root_schema_operand": root_name,
                "root_type": str(root.type),
                "view_type": str(view.type),
                "layout": layout,
            }
        if checked != records:
            raise ValueError("runtime source view record disagrees with the original standard SSA")
        return checked
