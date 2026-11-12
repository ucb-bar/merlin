"""Independently check original model view coordinates and source root bindings.

Only parent SSA and captured kernel argument constants supply these facts.
Logical element relationships grant no physical alias or mutation permission;
the actual kernel allocation capsule retains its executable pointer sharing.
"""

import math

from xdsl.dialects.builtin import DenseArrayBase, NoneAttr, TensorType
from xdsl.ir import Operation

PROFILE = "t2m.captured-source-tensor-views.v1"


def _dimensions(value):
    tensor = value.type
    if (not isinstance(tensor, TensorType) or not isinstance(tensor.encoding, NoneAttr)
            or len(tensor.get_shape()) > 8 or any(size < 0 for size in tensor.get_shape())
            or math.prod(tensor.get_shape()) >= 1 << 63):
        raise ValueError("derived source views require static unencoded tensor values")
    return list(tensor.get_shape())


def derive(view, root):
    root_shape = _dimensions(root)
    chain, current = [], view
    while current is not root:
        _dimensions(current)
        op = current.owner
        if (len(chain) == 32 or not isinstance(op, Operation) or len(op.results) != 1 or not op.operands
                or op.name not in ("linalg.transpose", "tensor.extract_slice", "tensor.cast")
                or current.type.element_type != root.type.element_type):
            raise ValueError("derived source view has no bounded executable path to its declared root")
        chain.append(op)
        current = op.operands[0]
    strides = [math.prod(root_shape[i + 1:]) for i in range(len(root_shape))]
    shape, offset, steps = root_shape, 0, []
    for op in reversed(chain):
        attrs = {**op.attributes, **op.properties}
        target_shape = _dimensions(op.results[0])

        def static(name):
            value = attrs.get(name)
            if not isinstance(value, DenseArrayBase):
                raise ValueError("derived source view lacks static coordinate fields")
            return list(value.iter_values())

        if op.name == "tensor.cast":
            if len(op.operands) != 1 or op.operands[0].type != op.results[0].type:
                raise ValueError("derived source view cast must retain its actual tensor type")
            continue
        if op.name == "linalg.transpose":
            permutation = static("permutation")
            if (len(op.operands) != 2 or sorted(permutation) != list(range(len(shape)))
                    or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
                raise ValueError("derived source view needs a complete transpose")
            body = op.regions[0].block
            if (len(body.args) not in (1, 2) or len(tuple(body.ops)) != 1
                    or body.last_op.name != "linalg.yield" or tuple(body.last_op.operands) != (body.args[0],)):
                raise ValueError("derived source transpose changes element words")
            if target_shape != [shape[axis] for axis in permutation]:
                raise ValueError("derived source transpose dimensions disagree")
            strides = [strides[axis] for axis in permutation]
            shape = target_shape
            steps.append({"op": op.name, "permutation": permutation})
        else:
            origins, sizes, increments = (static(key) for key in ("static_offsets", "static_sizes", "static_strides"))
            if (len(op.operands) != 1 or len(shape) != len(target_shape) or sizes != target_shape
                    or len(origins) != len(shape) or len(increments) != len(shape)):
                raise ValueError("derived source slice needs static rank-preserving coordinates")
            for origin, size, increment, bound in zip(origins, sizes, increments, shape, strict=True):
                if origin < 0 or increment <= 0 or origin > bound or size and origin + (size - 1) * increment >= bound:
                    raise ValueError("derived source slice exceeds its actual parent domain")
            offset += sum(origin * stride for origin, stride in zip(origins, strides, strict=True))
            strides = [stride * increment for stride, increment in zip(strides, increments, strict=True)]
            shape = target_shape
            steps.append({"op": op.name, "offsets": origins, "sizes": sizes, "strides": increments})
    return {"shape": shape, "strides": strides, "offset": offset, "steps": steps}


def checked_source_views(records, mapping, by_name, values, provenance):
    if not isinstance(records, dict) or not records or len(records) > 64:
        raise ValueError("derived source views need a complete bounded binding")
    pointers = (provenance or {}).get("source_argument_binding", {}).get("bindings", [])
    for name, record in records.items():
        if not isinstance(record, dict) or set(record) != {"binding", "root_schema_operand", "root_type", "view_type", "layout"}:
            raise ValueError("derived source view record is incomplete")
        config = record["binding"]
        if (not isinstance(name, str) or name not in by_name or name in mapping.values()
                or not isinstance(config, dict) or set(config) != {"root", "source_arg", "shape", "strides", "offset"}
                or not isinstance(config["root"], str) or config["root"] not in mapping
                or type(config["source_arg"]) is not int or not 0 <= config["source_arg"] < len(pointers)
                or not isinstance(config["shape"], list) or not isinstance(config["strides"], list)
                or len(config["shape"]) != len(config["strides"])):
            raise ValueError("derived source view binding is ambiguous")
        fields = [*config["shape"], *config["strides"], config["offset"]]
        if any(not isinstance(field, str) or field not in values or type(values[field]) is not int for field in fields):
            raise ValueError("derived source view fields must be actual captured integer kernel arguments")
        if pointers[config["source_arg"]] != {"kind": "pointer", "allocation": config["root"], "offset": config["offset"]}:
            raise ValueError("derived source view differs from its executable source allocation binding")
        root_name = mapping[config["root"]]
        root, view = by_name[root_name], by_name[name]
        actual = derive(view, root)
        declared = {"shape": [values[field] for field in config["shape"]],
                    "strides": [values[field] for field in config["strides"]], "offset": values[config["offset"]]}
        if (record["root_schema_operand"] != root_name or record["root_type"] != str(root.type)
                or record["view_type"] != str(view.type) or record["layout"] != actual
                or any(actual[key] != value for key, value in declared.items())):
            raise ValueError("derived source view disagrees with original parent dataflow and captured arguments")
