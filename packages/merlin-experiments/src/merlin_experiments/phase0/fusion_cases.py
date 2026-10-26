"""Freeze finite fusion cases from enumerated graphs, never the chain catalog.

This is operator-side validation, separate from the shared release capsule budget.
All source programs use the existing composition builder and ordinary Linalg IR.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from merlin.common.paths import artifacts_dir, merlin_dir, repo_root
from merlin.targetgen import composition_corpus as corpus
from merlin.targetgen.composition_capsules import build_composition

from .fusion_reference import exact_reference, stimuli, unfused_reference


def canonical(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(document):
    return hashlib.sha256(canonical(document)).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_frozen(path, raw):
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"frozen member changed: {path.name}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent) as stream:
        stream.write(raw)
        stream.flush()
        try:
            os.link(stream.name, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError(f"frozen member changed: {path.name}") from None


def facts_for_cases(manifest, raw_facts):
    from merlin.targetgen.capability_manifests import _mesh_from_facts

    fields = manifest["normalized_facts"]["fields"]
    facts = {key: row["value"] for key, row in fields.items() if row["status"] == "derived"}
    facts.update(output_path=manifest["normalized_facts"]["output_path"])
    facts["operand_signed_bits"] = facts["operand_bits"]
    # Geometry comes from the enumerator's normalized mesh evidence.
    mesh = _mesh_from_facts(raw_facts.get("facts", raw_facts))
    if mesh is None:
        raise ValueError("UNKNOWN: selected mesh geometry is absent")
    if mesh["rows"] != mesh["cols"]:
        raise ValueError("composition builder requires a square selected mesh")
    facts["tile_dim"] = mesh["rows"]
    scale = next(stage for stage in facts["output_path"]["readout"] if stage["kind"] == "scale")
    facts["columns_per_scale"] = scale["columns_per_scale"]
    return facts


def chain_for_node(node, dataflow):
    producer = "movement" if node["form"] == "window_movement" else node["form"]
    chain = dict(
        id=digest(node),
        producer=producer,
        exits=[node["readout"]["exit"]],
        accumulator_stages={"preload": "optional"} if node["seed"] else {},
        readout_stages="from_output_path",
        constraints=["accumulator_exit_applies_no_readout_stage"],
        quantization_schemes={"primary": [node["scheme"]]},
    )
    if dataflow == "OS":
        chain["dataflows"] = "from_facts"
        chain["constraints"].append(dict(os_dataflow_requires="mesh_partial_sum_bound_from_constant_operands"))
    if node["residual"]:
        chain["extension"] = "constant_integer_operand"
        chain["constraints"].extend(["coefficient_integer_exact", dict(k_extension_within="from_facts")])
    if producer == "operand_sum":
        chain["constraints"].extend([dict(operand_multiplier_in="(0, 1]"), "load_minimum_not_negated"])
    if node["realization"] == "mean":
        chain["reduction_kinds"] = ["sum"]
    elif any(stage["kind"] == "pool" for stage in node["readout"]["stages"]):
        chain["reduction_kinds"] = ["max"]
        chain["constraints"].extend(
            [
                {"max_requires_store_path_pooling": "from_facts"},
                "scratchpad_source_pooling_fenced",
                {"padded_max_requires": "producer_lower_bound_or_derived_pad_value"},
            ]
        )
    return chain


def parameters(candidate):
    labels = ["aligned", "partial", "rne_saturation", "conversion_boundary"]
    if any(node["form"] == "operand_sum" for node in candidate["semantic"]["nodes"]):
        labels.append("sum_widening")
    result = [(label, None) for label in labels]
    bounds = [n["residual"]["coefficient_bound"] for n in candidate["semantic"]["nodes"] if n["residual"]]
    if bounds:
        result.extend(("coefficient_extreme", sign * min(bounds)) for sign in (-1, 1))
    if any(
        node["residual"] and node["readout"]["exit"] == "requant" and node["regime"]["scale_value"] == "arbitrary_f32"
        for node in candidate["semantic"]["nodes"]
    ):
        result.append(("residual_rounding", 1))
    return result


def node_entry(node, scheme, facts, label, coefficient, dataflow, offset=0):
    from .runtime_scale_reference import effective_scheme

    scheme = effective_scheme(scheme, node)
    chain = chain_for_node(node, dataflow)
    shape_label = "partial" if label == "partial" else "aligned"
    if node["realization"] == "mean":
        shape_label = "window_mean"
    elif any(stage["kind"] == "pool" for stage in node["readout"]["stages"]):
        shape_label = "standalone_pool" if node["realization"] == "pool" else "fused_pool"
    shape = corpus._case(chain, scheme, shape_label, facts)
    case = shape["composition_instance"]
    if label == "residual_rounding":
        shape["K"] = max(7, shape["K"])
    if node["realization"] == "mean" and any(stage["kind"] == "pool" for stage in node["readout"]["stages"]):
        spatial = corpus._case(chain, scheme, "fused_pool", facts)
        shape["M"] = spatial["M"]
        for key in ("pool_in_dims", "pool_window", "pool_stride", "padding"):
            case[key] = spatial["composition_instance"][key]
    if "pool_in_dims" in case and label == "partial":
        shape["N"] += 1
    if node["realization"] == "mean" and label == "partial":
        if "pool_in_dims" not in case:
            shape["M"] += 1
            shape["N"] += 1
        case["constant_weights"] = [[1] * shape["N"] for _ in range(shape["K"])]
    if offset:
        for key in ("K", "N") if "pool_in_dims" in case else ("M", "K", "N"):
            shape[key] += offset * facts["tile_dim"]
        if case.get("constant_weights") is not None:
            case["constant_weights"] = [[1] * shape["N"] for _ in range(shape["K"])]
    m, k, n = (shape[key] for key in ("M", "K", "N"))
    stages = [s["kind"] for s in node["readout"]["stages"]]
    case.update(
        exit=node["readout"]["exit"],
        readout_stages=stages,
        accumulator_stages=["preload"] if node["seed"] else [],
        dataflow=dataflow,
        output_columns=n,
        shape_class=label,
    )
    if "pool" in stages:
        case["padding"] = [1, 1] if node["readout"].get("pool_padding_class") == "padded" else [0, 0]
        case["members"] = ["producer", "readout", "pool"]
    if node["seed"]:
        case["seed_range"] = [-(2**24) - 1, 2**24 + 1]
        case["seed_source"] = node.get("seed_source", "runtime_argument")
    width = scheme["columns_per_scale"] if scheme["weight_scale_granularity"] == "column_block" else n
    grid, _ = corpus.scheme_scales(scheme)
    if label == "rne_saturation":
        grid = [0.5]  # A realizable half tie for both declared scale domains.
    if "scale" not in stages or node["realization"] == "pool":
        grid = [1.0]
    if node["realization"] == "mean":
        grid = [1.0 / k] if "scale" in stages else [1.0]
    if label == "residual_rounding" and node["residual"] and scheme["scale_value"] == "arbitrary_f32":
        grid = [float(np.float32(1.0 / 6))]
    if label == "sum_widening" and node["form"] == "operand_sum":
        case["operand_multipliers"] = [1.0, 1.0]
        grid = [1.0]
    case["multipliers"] = [grid[i % len(grid)] for i in range((n + width - 1) // width)]
    if node["residual"]:
        case.update(
            coefficient=1 if coefficient is None else coefficient,
            source_k=k,
            residual_width=n,
            source_partial_bound=k * (1 << (facts["operand_signed_bits"] - 1)) ** 2,
        )
    if dataflow == "OS" and case["producer"] == "contraction":
        case.update(constant_weights=[[1] * n for _ in range(k)], seed_range=[0, 0])
    entry = dict(
        name="fusion_case",
        kind="layer",
        op="composition_program",
        source_role="derived_sweep",
        source_reference="bounded enumerated fusion graph",
        label="public",
        **shape,
        generalization=dict(quantization_scheme=scheme),
    )
    return entry


def _merge_sources(nodes, shared):
    """Inline original node bodies with SSA cloning and an explicit shared argument."""
    from xdsl.dialects.builtin import ModuleOp, StringAttr
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from merlin.common import mlir_query as mq

    functions = [next(op for op in mq.parse(source).body.block.ops if op.name == "func.func") for _, source in nodes]
    inputs, types, arguments = [], [], {}
    for index, ((cap, _), function) in enumerate(zip(nodes, functions, strict=True)):
        for spec, arg in zip(cap["inputs"], function.body.block.args, strict=True):
            name = "A" if shared and spec["name"] == "A" else f"node{index}_{spec['name']}"
            if name not in arguments:
                arguments[name] = len(types)
                inputs.append({**spec, "name": name})
                types.append(arg.type)
    block, outputs = Block(arg_types=types), []
    for index, ((cap, _), function) in enumerate(zip(nodes, functions, strict=True)):
        mapper = {}
        for spec, arg in zip(cap["inputs"], function.body.block.args, strict=True):
            name = "A" if shared and spec["name"] == "A" else f"node{index}_{spec['name']}"
            mapper[arg] = block.args[arguments[name]]
        for op in function.body.block.ops:
            if op.name == "func.return":
                outputs.extend(mapper[value] for value in op.operands)
            else:
                cloned = op.clone(value_mapper=mapper)
                marker = cloned.attributes.get("prov.composition_stage")
                if marker is not None:
                    del cloned.attributes["prov.composition_stage"]
                    cloned.attributes["prov.trace_role"] = StringAttr(f"node{index}_{marker.data}")
                block.add_op(cloned)
    block.add_op(ReturnOp(*outputs))
    module = ModuleOp([FuncOp.from_region("forward", types, [value.type for value in outputs], Region(block))])
    if shared:
        _order_siblings(block)
    module.verify()
    return _print_source(module), inputs


def _print_source(module):
    # The custom reduce printer omits its provenance attributes.
    from io import StringIO

    from xdsl.printer import Printer

    stream = StringIO()
    Printer(stream=stream, print_generic_format=any(op.name == "linalg.reduce" for op in module.walk())).print_op(
        module
    )
    return stream.getvalue() + "\n"


def _order_siblings(block):
    """Place sibling producers before their independent readout consumers."""
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    operations = tuple(block.ops)
    selected, ordered = set(), []

    def append(op):
        if op in selected or op not in operations:
            return
        for operand in op.operands:
            append(operand.owner)
        selected.add(op)
        ordered.append(op)

    # Immutable tensor constants precede all contiguous initializer/MAC groups.
    for op in operations:
        if op.name == "arith.constant" and hasattr(op.results[0].type, "get_shape"):
            append(op)
    for op in operations:
        stage = CG.classify(op)
        if stage is not None and stage.kind == CG.CONTRACTION:
            append(op)
    for op in operations:
        append(op)
    for op in operations:
        block.detach_op(op)
    block.add_ops(ordered)


def _accumulator_seed(cap, bias, mac):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    from merlin.compile.fusion_enumeration import seed_source

    case = cap["operation"]["attributes"]["case"]
    value = bias.operands[1]
    if case["seed_source"] == "constant":
        values = stimuli(cap, case["shape_class"])["Bias"]
        case["constant_seed"] = values.tolist()
        constant = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(value.type, values.flatten().tolist()))
        mac.parent.insert_op_before(constant, mac)
        value.replace_all_uses_with(constant.results[0])
        mac.parent.erase_arg(value)
        cap["inputs"] = [spec for spec in cap["inputs"] if spec["name"] != "Bias"]
        value = constant.results[0]
    if seed_source(value) != case["seed_source"]:
        raise ValueError("UNKNOWN: accumulator seed source differs from the enumerated class")
    if case["seed_source"] == "runtime_argument" and "coefficient" not in case:
        empty, guard = _seed_guard(value, case["seed_range"])
        mac.parent.insert_ops_before([empty, guard], mac)
        value = guard.results[0]
    return value


def _compile_source(cap, source):
    """Use ordinary initializer and stage boundaries, as the frontend does."""
    from xdsl.dialects import tensor
    from xdsl.rewriter import Rewriter

    from merlin.common import mlir_query as mq

    module = mq.parse(source)
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    block = function.body.block
    case = cap["operation"]["attributes"]["case"]
    if case.get("reduction_kind") == "sum":
        from xdsl.dialects.builtin import StringAttr

        mean = mq.parse(_mean_source(cap))
        next(op for op in mean.walk() if op.name == "linalg.reduce").attributes["prov.composition_stage"] = StringAttr(
            "mean_mac"
        )
        return cap, _print_source(mean)
    for op in tuple(block.ops):
        if op.name == "linalg.fill":
            Rewriter.replace_op(op, tensor.SplatOp(op.operands[0], [], op.results[0].type))
    if "preload" in case["accumulator_stages"]:
        bias = next(
            op for op in block.ops if getattr(op.attributes.get("prov.composition_stage"), "data", "") == "bias"
        )
        mac = bias.operands[0].owner
        seed = _accumulator_seed(cap, bias, mac)
        if "coefficient" in case:
            bias.operands = [bias.operands[0], seed, *bias.operands[2:]]
        else:
            mac.operands = [*mac.operands[:2], seed]
            bias.results[0].replace_all_uses_with(mac.results[0])
            block.erase_op(bias)
    if case["producer"] == "movement" and "pool" in case["readout_stages"]:
        return cap, _movement_pool_source(cap)
    split = case["producer"] == "contraction" or (
        case["producer"] == "operand_sum" and "pool" in case["readout_stages"]
    )
    if split and case["exit"] == "requant":
        if "coefficient" in case:
            extension = next(
                op
                for op in block.ops
                if getattr(op.attributes.get("prov.composition_stage"), "data", "") == "extension"
            )
            cap["_residual_argument"] = extension.operands[1].index
            extension.results[0].replace_all_uses_with(extension.operands[0])
            block.erase_op(extension)
        reader = next(
            op for op in block.ops if getattr(op.attributes.get("prov.composition_stage"), "data", "") == "readout"
        )
        _split_readout(cap, reader)
    # Remove only unused pure source scaffolding, retaining contiguous siblings.
    for op in reversed(tuple(block.ops)):
        if op.name in {"tensor.empty", "tensor.splat", "arith.constant"} and all(not r.uses for r in op.results):
            block.erase_op(op)
    if "pool" in case["readout_stages"] or case.get("seed_source") == "constant":
        from xdsl.dialects.builtin import FunctionType

        function.properties["function_type"] = FunctionType.from_lists(
            [arg.type for arg in block.args], [value.type for value in block.last_op.operands]
        )
    module.verify()
    return cap, str(module) + "\n"


def _mean_source(cap):
    """Retain the source window average; the backend selects its ones MAC."""
    attrs = cap["operation"]["attributes"]
    m, k, n = (attrs[key] for key in ("M", "K", "N"))
    attrs["case"]["window_mean"] = True
    cap["inputs"] = [dict(name="Image", shape=[m, n, k], dtype="i8", role="input")]
    full, part = f"tensor<{m}x{n}x{k}x", f"tensor<{m}x{n}x"
    ident = "affine_map<(d0,d1)->(d0,d1)>"
    activation = ""
    if "activation" in attrs["case"]["readout_stages"]:
        activation = f"""%ae = tensor.empty() : {part}f32>
          %activated = linalg.generic {{indexing_maps=[{ident}, {ident}], iterator_types=["parallel", "parallel"]}}
            ins(%mean: {part}f32>) outs(%ae: {part}f32>) attrs = {{prov.composition_stage="activation"}} {{
            ^bb0(%v: f32, %o: f32): %z = arith.constant 0.0 : f32
              %r = arith.maximumf %v, %z : f32 linalg.yield %r : f32
          }} -> {part}f32>"""
    value, value_type, pool = "activated" if activation else "mean", f"{part}f32>", ""
    if "pool" in attrs["case"]["readout_stages"]:
        from merlin.targetgen.composition_capsules import _Source

        source = _Source()
        source.types[value] = value_type
        value = _float_pool(source, value, attrs["case"], n)
        value_type, pool = source.types[value], "\n".join(source.lines)
    output_type = value_type.replace("f32", "i8")
    return f"""builtin.module {{ func.func @forward(%Image: {full}i8>) -> {output_type} {{
      %scale = arith.constant dense<1.0> : tensor<f32>
      %zp = arith.constant dense<0> : tensor<i64>
      %x = "quant_ext.dequantize_per_tensor"(%Image, %scale, %zp)
        <{{quant_min=-128:i64, quant_max=127:i64}}> {{prov.composition_stage="input"}} :
        ({full}i8>, tensor<f32>, tensor<i64>) -> {full}f32>
      %zero = arith.constant 0.0 : f32
      %init = tensor.splat %zero : {part}f32>
      %total = linalg.reduce ins(%x: {full}f32>) outs(%init: {part}f32>) dimensions = [2]
        (%a: f32, %b: f32) {{ %s = arith.addf %a, %b : f32 linalg.yield %s : f32 }}
      %count = arith.constant {float(k)} : f32
      %divisor = tensor.splat %count : {part}f32>
      %empty = tensor.empty() : {part}f32>
      %mean = linalg.generic {{indexing_maps=[{ident}, {ident}, {ident}], iterator_types=["parallel", "parallel"]}}
        ins(%total, %divisor: {part}f32>, {part}f32>) outs(%empty: {part}f32>)
        attrs = {{prov.composition_stage="mean"}} {{
        ^bb0(%v: f32, %d: f32, %o: f32): %r = arith.divf %v, %d : f32 linalg.yield %r : f32
      }} -> {part}f32>
      {activation}
      {pool}
      %q = "quant_ext.quantize_per_tensor"(%{value}, %scale, %zp)
        <{{quant_min=-128:i64, quant_max=127:i64, output_dtype="int8"}}> {{prov.composition_stage="readout"}} :
        ({value_type}, tensor<f32>, tensor<i64>) -> {output_type}
      func.return %q : {output_type}
    }} }}
    """


def _seed_guard(value, bounds):
    from merlin.common import mlir_query as mq
    from merlin.targetgen.composition_capsules import _Source

    source = _Source()
    shape = list(value.type.get_shape())
    source.input("Bias", shape, "i32")
    source.generic(
        "bias",
        ["Bias"],
        shape,
        "i32",
        [
            "%wide = arith.extsi %x0 : i32 to i64",
            f"%low = arith.constant {bounds[0]} : i64",
            "%ge = arith.cmpi sge, %wide, %low : i64",
            f"%high = arith.constant {bounds[1]} : i64",
            "%le = arith.cmpi sle, %wide, %high : i64",
            "%both = arith.andi %ge, %le : i1",
            'cf.assert %both, "declared seed range"',
            "linalg.yield %x0 : i32",
        ],
    )
    function = next(op for op in mq.parse(source.render(["bias"])).body.block.ops if op.name == "func.func")
    mapper = {function.body.block.args[0]: value}
    # Clone the destination as well; it is only the generic's unused output.
    empty, guard = [op.clone(value_mapper=mapper) for op in list(function.body.block.ops)[:-1]]
    # The retained assertion is an input-domain check, not an arithmetic fusion member.
    del guard.attributes["prov.composition_stage"]
    return empty, guard


def _split_readout(cap, reader):
    from .runtime_scale_reference import needs_runtime_scale, split_runtime_readout

    if needs_runtime_scale(cap):
        return split_runtime_readout(cap, reader)
    from merlin.common import mlir_query as mq
    from merlin.targetgen.composition_capsules import _Source

    attrs = cap["operation"]["attributes"]
    case, scheme = attrs["case"], attrs["quantization_scheme"]
    m, n = attrs["M"], attrs["N"]
    source = _Source()
    source.input("acc", [m, n], "i32")
    value = source.generic(
        "convert", ["acc"], [m, n], "f32", ["%f = arith.sitofp %x0 : i32 to f32", "linalg.yield %f : f32"]
    )
    width = scheme["columns_per_scale"] if scheme["weight_scale_granularity"] == "column_block" else n
    gains = [case["multipliers"][j // width] for j in range(n)]
    gain = source.constant("gains", [n], "f32", gains)
    value = source.generic(
        "scale",
        [value, gain],
        [m, n],
        "f32",
        ["%p = arith.mulf %x0, %x1 : f32", "linalg.yield %p : f32"],
        maps=["affine_map<(d0,d1)->(d0,d1)>", "affine_map<(d0,d1)->(d1)>", "affine_map<(d0,d1)->(d0,d1)>"],
    )
    if "coefficient" in case:
        source.input("residual", [m, n], "i8")
        scales = [float(np.float32(case["coefficient"] * gain)) for gain in gains]
        if len(set(scales)) == 1:
            source.lines.extend(
                [
                    f"%rs = arith.constant dense<{scales[0]}> : tensor<f32>",
                    "%rz = arith.constant dense<0> : tensor<i64>",
                    f'%rd = "quant_ext.dequantize_per_tensor"(%residual, %rs, %rz) '
                    f"<{{quant_min=-128:i64, quant_max=127:i64}}> : "
                    f"(tensor<{m}x{n}xi8>, tensor<f32>, tensor<i64>) -> tensor<{m}x{n}xf32>",
                ]
            )
        else:
            source.constant("rs", [n], "f32", scales)
            source.constant("rz", [n], "i64", [0] * n)
            source.lines.append(
                f'%rd = "quant_ext.dequantize_per_channel"(%residual, %rs, %rz) '
                f"<{{axis=1:i64, quant_min=-128:i64, quant_max=127:i64}}> : "
                f"(tensor<{m}x{n}xi8>, tensor<{n}xf32>, tensor<{n}xi64>) -> tensor<{m}x{n}xf32>"
            )
        source.types["rd"] = f"tensor<{m}x{n}xf32>"
        value = source.generic(
            "extension",
            [value, "rd"],
            [m, n],
            "f32",
            ["%v = arith.addf %x0, %x1 : f32", "linalg.yield %v : f32"],
        )
    if "activation" in case["readout_stages"]:
        value = source.generic(
            "activation",
            [value],
            [m, n],
            "f32",
            ["%z = arith.constant 0.0 : f32", "%v = arith.maximumf %x0, %z : f32", "linalg.yield %v : f32"],
        )
    if "pool" in case["readout_stages"]:
        value = _float_pool(source, value, case, n)
    source.lines.extend(
        ["%qs = arith.constant dense<1.0> : tensor<f32>", "%qz = arith.constant dense<0> : tensor<i64>"]
    )
    source.types["readout"] = source.types[value].replace("f32", "i8")
    source.lines.append(f"""%readout = "quant_ext.quantize_per_tensor"(%{value}, %qs, %qz)
        <{{quant_min=-128:i64, quant_max=127:i64, output_dtype="int8"}}>
        {{prov.composition_stage="readout"}} :
        ({source.types[value]}, tensor<f32>, tensor<i64>) -> {source.types["readout"]}""")
    result = "readout"
    function = next(op for op in mq.parse(source.render([result])).body.block.ops if op.name == "func.func")
    mapper = {function.body.block.args[0]: reader.operands[0]}
    if "coefficient" in case:
        mapper[function.body.block.args[1]] = reader.parent.args[cap.pop("_residual_argument")]
    old_pool = next(
        (op for op in reader.parent.ops if getattr(op.attributes.get("prov.composition_stage"), "data", "") == "pool"),
        None,
    )
    for op in list(function.body.block.ops)[:-1]:
        reader.parent.insert_op_before(op.clone(value_mapper=mapper), reader)
    target = old_pool if old_pool is not None else reader
    target.results[0].replace_all_uses_with(mapper[list(function.body.block.ops)[-1].operands[0]])
    if old_pool is not None:
        reader.parent.erase_op(old_pool)
        for op in reversed(tuple(reader.parent.ops)):
            if op.name in {"tensor.expand_shape", "tensor.collapse_shape", "tensor.pad"} and all(
                not result.uses for result in op.results
            ):
                reader.parent.erase_op(op)
    reader.parent.erase_op(reader)


def _float_pool(source, value, case, channels):
    """Ordinary NHWC max reduction, with explicit source-semantic padding."""
    h, w = case["pool_in_dims"]
    kh, kw = case["pool_window"]
    sh, sw = case["pool_stride"]
    py, px = case["padding"]
    image = f"tensor<1x{h}x{w}x{channels}xf32>"
    if source.types[value] != image:
        source.lines.append(
            f"%image = tensor.expand_shape %{value} [[0,1,2],[3]] "
            f"output_shape [1,{h},{w},{channels}] : {source.types[value]} into {image}"
        )
        value = "image"
    if py or px:
        framed = f"tensor<1x{h + 2 * py}x{w + 2 * px}x{channels}xf32>"
        source.lines.extend(
            [
                "%pz = arith.constant 0xFF800000 : f32",
                f"%border = tensor.splat %pz : {framed}",
                # Generic form: every supported xDSL release parses it.
                f'%framed = "tensor.insert_slice"(%{value}, %border) <{{'
                f"static_offsets = array<i64: 0, {py}, {px}, 0>, "
                f"static_sizes = array<i64: 1, {h}, {w}, {channels}>, "
                "static_strides = array<i64: 1, 1, 1, 1>, "
                f"operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}}> : ({image}, {framed}) -> {framed}",
            ]
        )
        value, image = "framed", framed
    oh, ow = (h + 2 * py - kh) // sh + 1, (w + 2 * px - kw) // sw + 1
    result = f"tensor<1x{oh}x{ow}x{channels}xf32>"
    source.types["pooled"] = result
    source.lines.extend(
        [
            "%ninf = arith.constant 0xFF800000 : f32",
            f"%pi = tensor.splat %ninf : {result}",
            f"%win = tensor.empty() : tensor<{kh}x{kw}xf32>",
            f"%pooled = linalg.generic"
            f" {{indexing_maps=[affine_map<(d0,d1,d2,d3,d4,d5)->(d0,d1*{sh}+d4,d2*{sw}+d5,d3)>,"
            f" affine_map<(d0,d1,d2,d3,d4,d5)->(d4,d5)>, affine_map<(d0,d1,d2,d3,d4,d5)->(d0,d1,d2,d3)>],"
            f' iterator_types=["parallel","parallel","parallel","parallel","reduction","reduction"]}}'
            f" ins(%{value},%win : {image},tensor<{kh}x{kw}xf32>) outs(%pi : {result}) attrs ="
            f' {{prov.composition_stage="pool"}} {{ ^bb0(%v:f32,%w:f32,%o:f32): %r=arith.maximumf %v,%o:f32'
            f" linalg.yield %r:f32 }} -> {result}",
        ]
    )
    return "pooled"


def _movement_pool_source(cap):
    from merlin.targetgen.composition_capsules import _Source

    attrs = cap["operation"]["attributes"]
    case, n = attrs["case"], attrs["N"]
    source = _Source()
    h, w = case["pool_in_dims"]
    raw = f"tensor<1x{h}x{w}x{n}xi8>"
    real = raw.replace("i8", "f32")
    source.input("Image", [1, h, w, n], "i8")
    cap["inputs"][0]["shape"] = [1, h, w, n]
    source.lines.extend(
        [
            "%s = arith.constant dense<1.0> : tensor<f32>",
            "%z = arith.constant dense<0> : tensor<i64>",
            f'%dq = "quant_ext.dequantize_per_tensor"(%Image,%s,%z) '
            f"<{{quant_min=-128:i64,quant_max=127:i64}}> : ({raw},tensor<f32>,tensor<i64>) -> {real}",
        ]
    )
    source.types["dq"] = real
    value = "dq"
    if "activation" in case["readout_stages"]:
        identity = "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>"
        source.lines.extend(
            [
                f"%ae = tensor.empty() : {real}",
                f"%activated = linalg.generic {{indexing_maps=[{identity},{identity}], "
                'iterator_types=["parallel","parallel","parallel","parallel"]} '
                f'ins(%dq : {real}) outs(%ae : {real}) attrs = {{prov.composition_stage="activation"}} {{'
                " ^bb0(%v:f32,%o:f32): %az=arith.constant 0.0:f32 %r=arith.maximumf %v,%az:f32"
                f" linalg.yield %r:f32 }} -> {real}",
            ]
        )
        value = "activated"
        source.types[value] = real
    value = _float_pool(source, value, case, n)
    source.types["q"] = source.types[value].replace("f32", "i8")
    source.lines.append(
        f'%q = "quant_ext.quantize_per_tensor"(%{value},%s,%z) '
        '<{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}> '
        f'{{prov.composition_stage="readout"}} : '
        f"({source.types[value]},tensor<f32>,tensor<i64>) -> {source.types['q']}"
    )
    return source.render(["q"])


def exclusion_context():
    document = yaml.safe_load((merlin_dir() / "contract/capsules/MANIFEST.yaml").read_bytes())
    roots = [
        str((merlin_dir() / "contract/capsules" / name).relative_to(repo_root()))
        for name in document["generated"]
        if name.startswith("_model_layers/")
    ]
    if not roots:
        raise ValueError("held-out publication inventory is empty")
    return SimpleNamespace(source_root=repo_root(), held_out_inputs=roots, workload_spec={})


def producer_views(nodes, facts):
    """Every sibling and physical extended-K realization gets its own carrier."""
    from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

    views = []
    for cap, _ in nodes:
        attrs = cap["operation"]["attributes"]
        case, m, k, n = attrs["case"], attrs["M"], attrs["K"], attrs["N"]
        geometries = [k]
        if "coefficient" in case:
            try:
                proof = lower_k_extension(
                    case["coefficient"],
                    source_k=k,
                    residual_width=n,
                    digit_bits=facts["operand_signed_bits"],
                    max_extended_k=facts["k_extension_within"]["max_k"],
                    accumulator_bits=facts["k_extension_within"]["partial_sum_bits"],
                    source_bound=case["source_partial_bound"],
                    residual_range=case["residual_range"],
                )
                geometries.append(proof["extended_k"])
            except ValueError:
                # Even a refused physical realization is screened using its declared extent.
                bits = facts["operand_signed_bits"]
                digits = (abs(case["coefficient"]) + (1 << (bits - 1)) - 2) // ((1 << (bits - 1)) - 1)
                geometries.append(k + digits * n)
        for extent in geometries:
            view = dict(
                name="fusion_producer_view",
                operation=dict(op="matmul"),
                inputs=[
                    dict(name="A", role="input", shape=[m, extent]),
                    dict(name="W", role="weight", shape=[extent, n]),
                ],
            )
            views.append(view)
    return views


def _public(nodes, facts, excluded, context, scratch):
    from merlin_experiments.corpus.publication import require_public_capsules
    from merlin_experiments.spec import SpecError

    views = producer_views(nodes, facts)
    for view in views:
        a, w = view["inputs"]
        corpus.assert_generic_shape(dict(M=a["shape"][0], K=a["shape"][1], N=w["shape"][1]), excluded)
    for index, view in enumerate([*(cap for cap, _ in nodes), *views]):
        root = scratch / str(index)
        root.mkdir(parents=True, exist_ok=True)
        (root / "capsule.yaml").write_text(yaml.safe_dump(view))
    try:
        require_public_capsules(context, [scratch])
    except SpecError as error:
        raise ValueError("held-out publication exclusion") from error


class ReferenceDisagreement(ValueError):
    """This candidate cannot be validated by the independent numeric references."""


def build_payload(candidate, schemes, facts, label, coefficient, excluded, context, scratch):
    binding = SimpleNamespace(
        operand_dtype="i8", accum_dtype="i32", cap_dtype=lambda value: value, classes_for=lambda **_: [], tiers=["L2"]
    )
    for offset in range(5):
        try:
            entries = [
                node_entry(
                    node, schemes[node["scheme"]], facts, label, coefficient, candidate["semantic"]["dataflow"], offset
                )
                for node in candidate["semantic"]["nodes"]
            ]
            if candidate["semantic"]["structure"] == "shared_input":
                _share_input_geometry(entries, facts)
                from .runtime_scale_reference import runtime_scale_geometry

                runtime_scale_geometry(entries)
            nodes = [build_composition(entry, binding, excluded_shapes=excluded) for entry in entries]
            nodes = [_compile_source(cap, source) for cap, source in nodes]
            _public(nodes, facts, excluded, context, scratch)
            break
        except ValueError as error:
            if offset == 4:
                raise ValueError(f"pending_shape_exclusion: {error}") from error
    source, inputs = _merge_sources(nodes, candidate["semantic"]["structure"] == "shared_input")
    values, references, witnesses = {}, [], []
    environments = [stimuli(cap, label, index) for index, (cap, _) in enumerate(nodes)]
    if candidate["semantic"]["structure"] == "shared_input":
        witness = next(
            (
                index
                for index, (cap, _) in enumerate(nodes)
                if label == "residual_rounding" and "coefficient" in cap["operation"]["attributes"]["case"]
            ),
            0,
        )
        shared = environments[witness]["A"].copy()
        for env in environments:
            env["A"] = shared
    for index, (cap, _) in enumerate(nodes):
        env = environments[index]
        for spec in cap["inputs"]:
            name = (
                "A"
                if candidate["semantic"]["structure"] == "shared_input" and spec["name"] == "A"
                else f"node{index}_{spec['name']}"
            )
            previous = values.setdefault(name, env[spec["name"]].tolist())
            if previous != env[spec["name"]].tolist():
                raise ValueError("shared input stimuli disagree")
        exact, events = exact_reference(cap, env)
        if exact != unfused_reference(cap, env):
            raise ReferenceDisagreement("independent unfused references disagree")
        for name, value in exact.items():
            from .runtime_scale_reference import typed_result_dtype

            dtype = typed_result_dtype(cap, name)
            references.append(dict(name=f"node{index}_{name}", dtype=dtype, shape=list(np.shape(value)), values=value))
        witnesses.append(events)
    return dict(
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        inputs=inputs,
        input_values=values,
        inputs_sha256=digest(dict(inputs=inputs, values=values)),
        references=references,
        references_sha256=digest(references),
        nodes=[cap for cap, _ in nodes],
        producer_views=producer_views(nodes, facts),
        witnesses=witnesses,
        required_witnesses=required_witnesses(nodes, label),
    )


def _share_input_geometry(entries, facts):
    """Build sibling contractions on one input, including spatial readouts."""
    m, k = (max(entry[axis] for entry in entries) for axis in ("M", "K"))
    for entry in entries:
        entry.update(M=m, K=k)
        case, n = entry["composition_instance"], entry["N"]
        if "pool_in_dims" in case and np.prod(case["pool_in_dims"]) != m:
            raise ValueError("shared input pool geometry differs from its source rows")
        if "coefficient" in case:
            case.update(source_k=k, source_partial_bound=k * (1 << (facts["operand_signed_bits"] - 1)) ** 2)
        if case.get("constant_weights") is not None:
            case["constant_weights"] = [[1] * n for _ in range(k)]


def required_witnesses(nodes, label):
    result = []
    for index, (cap, _) in enumerate(nodes):
        case = cap["operation"]["attributes"]["case"]
        narrow = case["exit"] == "requant"
        if label == "rne_saturation" and narrow and "scale" in case["readout_stages"]:
            result.append(dict(node=index, event="rne_tie"))
        seed_min, seed_max = case.get("seed_range", [0, 0])
        boundary_in_range = any(seed_min <= v <= seed_max for v in (-(2**24) - 1, 2**24 + 1))
        if label == "conversion_boundary" and narrow and "preload" in case["accumulator_stages"] and boundary_in_range:
            result.append(dict(node=index, event="conversion_boundary"))
        if label == "sum_widening" and case["producer"] == "operand_sum":
            result.append(dict(node=index, event="sum_widening"))
        if label == "residual_rounding" and narrow and "coefficient" in case:
            if cap["operation"]["attributes"]["quantization_scheme"]["scale_value"] == "arbitrary_f32":
                result.append(dict(node=index, event="residual_rounding_difference"))
    return result


def _generate_candidate(candidate, schemes, facts, excluded, context, destination):
    import tempfile

    rows = []
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        for label, coefficient in parameters(candidate):
            selection = dict(candidate_id=candidate["candidate_id"], shape_class=label, coefficient=coefficient)
            payload = build_payload(candidate, schemes, facts, label, coefficient, excluded, context, Path(temporary))
            raw = canonical(payload)
            pin = hashlib.sha256(raw).hexdigest()
            relative = f"fusion-members/{pin}.json"
            write_frozen(destination / relative, raw)
            rows.append(
                dict(
                    case_id=digest(selection),
                    **selection,
                    member=relative,
                    sha256=pin,
                    bytes=len(raw),
                    required_engines=["spike"],
                    positive=True,
                )
            )
    return rows


def generate(candidate_path, facts_path, destination=None, *, spec, workers=1):
    """Generate the fixed suite; the campaign is re-derived from the selected ``spec``."""
    from .fusion_validation import worker_pool

    manifest = json.loads(Path(candidate_path).read_bytes())
    from merlin.compile.fusion_enumeration import verify_stratification

    if not isinstance(spec, dict):
        raise ValueError("fusion case generation requires the selected software spec")
    verify_stratification(manifest, spec, required=True)
    destination = Path(destination or artifacts_dir() / "verification" / manifest["target"])
    raw_facts = json.loads(Path(facts_path).read_bytes())
    from merlin.compile.fusion_enumeration import _unordered_inputs, canonical_sha256

    if canonical_sha256(_unordered_inputs(raw_facts)) != manifest["identity"]["facts_sha256"]:
        raise ValueError("candidate facts binding changed")
    destination.mkdir(parents=True, exist_ok=True)
    facts, excluded, context = facts_for_cases(manifest, raw_facts), corpus.held_out_shapes(), exclusion_context()
    schemes = {s["id"]: s for s in manifest["scheme_resolution"]["schemes"]}
    rows = []
    refused = []
    candidates = sorted((r for r in manifest["rows"] if r["state"] == "candidate"), key=lambda r: r["candidate_id"])
    with worker_pool(workers) as pool:
        futures = [
            pool.submit(_generate_candidate, candidate, schemes, facts, excluded, context, destination)
            for candidate in candidates
        ]
        for index, future in enumerate(futures):
            try:
                rows.extend(future.result())
            except ReferenceDisagreement as error:
                refused.append(dict(candidate_id=candidates[index]["candidate_id"], reason=str(error)))
            if index % 100 == 0:
                print(json.dumps(dict(generated_candidates=index + 1, generated_cases=len(rows))), flush=True)
    identity = dict(
        candidate_manifest_sha256=file_digest(candidate_path),
        **manifest["identity"],
        private_exclusion_sha256=digest(sorted(excluded)),
        reference_implementation_sha256=reference_identity(),
    )
    suite = dict(
        schema="merlin.fusion_cases.v1",
        target=manifest["target"],
        identity=identity,
        profile=dict(
            version=1,
            alternate_geometry_offsets=list(range(5)),
            parameters="aligned, tail, edge stimuli, signed coefficient bounds",
            refused_candidates=refused,
        ),
        candidate_count=manifest["candidate_count"],
        candidate_case_count=len({r["candidate_id"] for r in rows}),
        case_count=len(rows),
        rows=sorted(rows, key=lambda row: row["case_id"]),
    )
    suite["seal"] = dict(canonicalization="merlin.canonical_json.v1", payload_sha256=digest(suite))
    write_frozen(destination / "fusion-cases.json", canonical(suite))
    return suite


def reference_identity():
    from merlin.common.paths import module_source_path

    modules = [
        __name__,
        "merlin_experiments.phase0.fusion_reference",
        "merlin_experiments.phase0.runtime_scale_reference",
        "merlin.targetgen.composition_golden",
        "merlin.targetgen.composition_capsules",
        "merlin.targetgen.composition_corpus",
        "merlin.targetgen.exact_quantized_reference",
        "merlin.targetgen.exact_arithmetic",
        "merlin.runtime.exact_readout",
    ]
    return digest({name: file_digest(module_source_path(name)) for name in modules})
