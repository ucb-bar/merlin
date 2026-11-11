"""Exact static floating Linalg graphs with explicit, unresolved domain premises.

Reuse the semantic graph boundary and per-operation recognizers. Every source
operation and SSA edge is accounted for, including retained intermediate outputs.
Recognition proves source structure, never target equivalence or runtime domains.
"""

from hashlib import sha256

from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg.ops import FillOp, GenericOp

from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.transforms.ordered_float_dense import recognize_ordered_float_dense

from .linalg_bf16_add import recognize_bf16_f32_chop_add, recognize_bf16_f32_chop_sub
from .linalg_bf16_mul import recognize_bf16_product_rne
from .linalg_bf16_relu import recognize_bf16_sign_clamp_relu
from .linalg_fp8_cast import recognize_bf16_fp8_truncation
from .linalg_text import parse_mlir_text
from .linalg_errors import InvalidLinalgPattern


def _require(condition, reason):
    if not condition:
        raise ValueError("floating graph: " + reason)


def _type(value, policy):
    ty = value.type
    _require(type(ty) is builtin.TensorType and type(ty.encoding) is builtin.NoneAttr,
             "default-encoded tensor required")
    _require(len(ty.get_shape()) == 2, "rank-two tensor required")
    return TensorType(tuple(ty.get_shape()), str(ty.element_type), policy)


def read_float_graph(source, *, entry, target_identity, ordered_policies):
    """Read closed ordered dots, BF16 pointwise operations and declared FP8 casts.

    Each node retains its numeric policy and input-domain obligation. Targets
    must validate these obligations before executing; annotations are premises.
    No implicit conversion, fusion, reassociation or padding is performed.
    """
    module = parse_mlir_text(source)
    module.verify()
    functions = tuple(module.body.block.ops)
    _require(not module.attributes and len(functions) == 1 and type(functions[0]) is func.FuncOp,
             "one function without module attributes required")
    function = functions[0]
    _require(function.sym_name.data == entry and len(function.body.blocks) == 1,
             "entry differs or contains control flow")
    _require(not function.attributes and set(function.properties) <= {
        "sym_name", "function_type", "arg_attrs", "res_attrs", "sym_visibility"},
        "unknown function attributes")
    _require(function.res_attrs is None or all(not a.data for a in function.res_attrs), "result attributes")
    block = function.body.block
    operations = tuple(block.ops)
    _require(operations and type(operations[-1]) is func.ReturnOp, "return required")
    ret = operations[-1]
    _require(not ret.attributes and not ret.properties and ret.operands, "invalid return")
    consumed = {ret}
    nodes, values = [], {}
    arg_attrs = function.arg_attrs
    for index, arg in enumerate(block.args):
        attrs = {} if arg_attrs is None else arg_attrs.data[index].data
        _require(set(attrs) <= {"merlin.input_domain"}, "unknown input attributes")
        domain = attrs.get("merlin.input_domain")
        _require(domain is None or type(domain) is builtin.StringAttr and domain.data, "invalid input domain")
        node = SemanticNode(f"input{index}", "input", (), _type(arg, "source-input"),
                            () if domain is None else (("input_domain", domain.data),), effect="input")
        nodes.append(node)
        values[arg] = node.id
    policies = {p.identity: p for p in ordered_policies}
    _require(len(policies) == len(ordered_policies), "duplicate ordered policy")
    for operation in operations[:-1]:
        if type(operation) in (arith.ConstantOp, tensor.EmptyOp, FillOp):
            continue  # Admitted only if an exact consumer accounts for it below.
        _require(type(operation) is GenericOp, "unsupported source operation")
        declared = operation.attributes.get("merlin.numeric_semantics")
        _require(type(declared) is builtin.StringAttr and declared.data, "explicit numeric policy required")
        attrs = {}
        if declared.data in policies:
            policy = policies[declared.data]
            m, k, n, transposed = recognize_ordered_float_dense(operation, policy=policy)
            kind = "ordered_dot"
            attrs.update(m=m, k=k, n=n, rhs_transposed=transposed, reduction_order="k-ascending")
            fill = operation.outputs[0].owner
            consumed.update((fill, fill.inputs[0].owner, fill.outputs[0].owner))
        else:
            recognizers = (
                ("relu", recognize_bf16_sign_clamp_relu),
                ("cast", recognize_bf16_fp8_truncation),
                ("add", recognize_bf16_f32_chop_add),
                ("sub", recognize_bf16_f32_chop_sub),
                ("mul", recognize_bf16_product_rne),
            )
            for kind, recognize in recognizers:
                try:
                    recognized = recognize(operation)
                    break
                except InvalidLinalgPattern:
                    continue
            else:
                raise ValueError("floating graph: unsupported scalar body or numeric policy")
            empty = operation.outputs[0].owner
            _require(type(empty) is tensor.EmptyOp and not empty.operands and
                     not empty.attributes and not empty.properties, "pointwise output needs a static empty")
            consumed.add(empty)
            if kind == "cast":
                attrs["scale_code"] = recognized[-1]
        attrs["input_domain"] = operation.attributes["merlin.input_domain"].data
        _require(all(value in values for value in operation.inputs), "unknown tensor producer")
        node = SemanticNode(f"node{len(nodes)}", kind, tuple(values[v] for v in operation.inputs),
                            _type(operation.results[0], declared.data), tuple(sorted(attrs.items())))
        nodes.append(node)
        values[operation.results[0]] = node.id
        consumed.add(operation)
    _require(consumed == set(operations), "unconsumed initializer or source operation")
    _require(all(value in values for value in ret.operands), "unknown returned tensor")
    outputs = tuple(values[value] for value in ret.operands)
    live = set(outputs)
    for node in reversed(nodes):
        if node.id in live:
            live.update(node.inputs)
    _require(live == {node.id for node in nodes}, "unused input or dead computation")
    return KernelRequest(tuple(nodes), outputs, tuple(f"output:{i}" for i in range(len(outputs))),
                         tuple((node.id, f"input:{i}") for i, node in enumerate(nodes) if node.effect == "input"),
                         target_identity, source_identity=sha256(source.encode()).hexdigest())
