"""A declared floating zero-threshold activation executes on signed byte codes."""

import hashlib
from fractions import Fraction


def prepare(directory, report, selection, provider, package, module):
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.ir.affine import AffineMap

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.quantized_native import parameters
    from merlin.targetgen.exact_arithmetic import round_binary
    from merlin.targetgen.floating_contract import requires_quantized_model
    from merlin.targetgen.native_byte_activation import compile_activation
    from merlin.targetgen.operator_review import review_record
    from merlin.targetgen.quantized_literal_reference import qualify
    from merlin.xdsl_dialects.lowering.contraction_coverage import is_compare_select_relu

    records = [r for r in report.get("cases", ()) if r.get("status") == "bundled"]
    if len(records) != 1:
        return None
    if not requires_quantized_model(module)["quantized"]:
        return None
    rule = selection["spec"].get("numerical_semantics", {}).get("quantized_native_fallback")
    if not rule or not review_record(rule)["operator_accepted"]:
        return None
    original = str(module)
    from merlin.llvmlower.quantized_zero_threshold import normalize

    module = module.clone()
    normalize(module)
    functions = [op for op in module.walk() if op.name == "func.func"]
    if len(functions) != 1:
        return None
    function = functions[0]
    if function.sym_name.data != "forward" or len(function.body.blocks) != 1 or len(function.body.block.args) != 1:
        return None
    argument = function.body.block.args[0]
    element = str(getattr(argument.type, "element_type", ""))
    if element not in {"f16", "bf16", "f32"} or any(n <= 0 for n in argument.type.get_shape()):
        return None
    roots = [op for op in function.body.block.ops if op.name == "linalg.generic"]
    if len(roots) != 1 or not is_compare_select_relu(roots[0]):
        return None
    root = roots[0]
    if root.inputs[0] is not argument or root.results[0].type != argument.type:
        return None
    rank = len(argument.type.get_shape())
    identity = AffineMap.identity(rank)
    if any(it.data.value != "parallel" for it in root.get_iterator_types()) or any(
        m.data != identity for m in root.get_indexing_maps()
    ):
        return None
    if any(op.name not in {"tensor.empty", "linalg.generic", "func.return"} for op in function.body.block.ops):
        return None
    returned = function.body.block.last_op
    if returned.name != "func.return" or root.results[0] not in returned.operands:
        return None
    if any(v is not argument and v is not root.results[0] for v in returned.operands):
        return None
    contract = parameters(selection["spec"])
    scale = Fraction(contract["scale_numerator"], contract["scale_denominator"])
    # Every output code must survive the final storage narrowing exactly.
    # This exhausts the declared code domain, never the evaluation inputs.
    for code in range(-128, 128):
        q = code * scale
        try:
            rounded = round_binary(q, 32 if element == "f32" else 16, bfloat16=element == "bf16")
        except (ValueError, OverflowError):
            return None
        if Fraction(rounded) != q:
            return None
    shape = argument.type.get_shape()
    prefix = "x".join(map(str, shape))
    prefix = prefix + "x" if prefix else ""

    def typ(token):
        return "tensor<" + prefix + token + ">"

    maps = "[" + ",".join(["affine_map<" + str(identity) + ">"] * 2) + "]"
    iterators = "[" + ",".join(['"parallel"'] * rank) + "]"
    numeric = f'''prov.quant.storage_dtype="int8",prov.quant.quant_min=-128:i64,prov.quant.quant_max=127:i64,
      prov.quant.granularity="tensor",prov.quant.scale_numerator=["{scale.numerator}"],
      prov.quant.scale_denominator=["{scale.denominator}"],prov.quant.scale={float(scale):.17e}:f32,
      prov.quant.zero_point=0:i64'''
    result_types = ",".join(str(v.type) for v in returned.operands)
    results = ",".join("%out" if v is root.results[0] else "%x" for v in returned.operands)
    narrow = (
        f"""%n=tensor.empty() : {typ(element)}
      %out=linalg.generic {{indexing_maps={maps},iterator_types={iterators}}}
        ins(%dq:{typ("f32")}) outs(%n:{typ(element)}) {{
        ^bb2(%wide:f32,%unusedn:{element}):
        %narrow=arith.truncf %wide : f32 to {element}
        linalg.yield %narrow : {element}
      }} -> {typ(element)}"""
        if element != "f32"
        else ""
    )
    dqresult = "%out" if element == "f32" else "%dq"
    text = f"""module {{func.func @forward(%x:{typ(element)}) -> ({result_types}) {{
      %s=arith.constant dense<{float(scale):.17e}> : tensor<f64>
      %z=arith.constant dense<0> : tensor<i64>
      %e=tensor.empty() : {typ(element)}
      %safe=linalg.generic {{indexing_maps={maps},iterator_types={iterators}}}
        ins(%x:{typ(element)}) outs(%e:{typ(element)}) {{
        ^bb0(%a:{element},%unused:{element}):
        %p=arith.cmpf uno,%a,%a : {element}
        %zero=arith.constant 0.0 : {element}
        %v=arith.select %p,%zero,%a : {element}
        linalg.yield %v : {element}
      }} -> {typ(element)}
      %q="quant_ext.quantize_per_tensor"(%safe,%s,%z)
        <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>
        {{{numeric},prov.quant.rounding="half_to_even",merlin.quantizer_nan_storage_zero}}
        : ({typ(element)},tensor<f64>,tensor<i64>) -> {typ("i8")}
      %b=tensor.empty() : {typ("i8")}
      %r=linalg.generic {{indexing_maps={maps},iterator_types={iterators}}}
        ins(%q:{typ("i8")}) outs(%b:{typ("i8")}) {{
        ^bb1(%byte:i8,%unusedbyte:i8):
        %zbyte=arith.constant 0 : i8
        %negative=arith.cmpi slt,%byte,%zbyte : i8
        %positive=arith.select %negative,%zbyte,%byte : i8
        linalg.yield %positive : i8
      }} -> {typ("i8")}
      {dqresult}="quant_ext.dequantize_per_tensor"(%r,%s,%z)
        <{{quant_min=-128:i64,quant_max=127:i64,input_dtype="int8"}}>
        {{{numeric}}} : ({typ("i8")},tensor<f64>,tensor<i64>) -> {typ("f32")}
      {narrow}
      return {results} : {result_types}
    }} }}"""
    prepared = parse_mlir_text(text)
    prepared.attributes.update(module.attributes)
    generated = next(op for op in prepared.walk() if is_compare_select_relu(op))
    generated.attributes.update(root.attributes)
    generated.attributes["merlin.native_byte_activation"] = UnitAttr()
    prepared.verify()
    operations = list(prepared.walk())
    data = generated.inputs[0]
    plan = compile_activation(
        directory,
        report,
        selection,
        provider,
        package,
        prepared,
        generated,
        data,
        dict(op_ordinal=operations.index(data.owner), result=0),
    )
    if plan is None:
        return None
    plan.pop("input_bound_proof", None)
    plan["native_semantic_proof"] = dict(
        status="proven",
        method="declared_quantized_zero_threshold",
        full_code_domain=[-128, 127],
        storage_narrowing="all_256_codes_exact",
        source_sha256=hashlib.sha256(original.encode()).hexdigest(),
    )
    plan["calibration"] = dict(source="declared_observer_domain", evaluation_inputs="never_read", performed=False)
    return qualify(directory, report, plan, contract)
