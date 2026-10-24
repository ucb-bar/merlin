"""Signed word activation as the same native unit on sign-replicated digits.

Seven magnitude bits and the original sign occupy each signed byte. Native
ReLU zeros every digit of a negative word; nonnegative words retain every
magnitude bit. The host codecs only select and assemble storage bits.
"""

import hashlib


def prepare(directory, report, selection, provider, package, module):
    from xdsl.ir.affine import AffineMap

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.native_byte_activation import compile_activation
    from merlin.xdsl_dialects.lowering.contraction_coverage import is_compare_select_relu

    functions = [op for op in module.walk() if op.name == "func.func"]
    if len(functions) != 1:
        return None
    function = functions[0]
    if function.sym_name.data != "forward" or len(function.body.blocks) != 1 or len(function.body.block.args) != 1:
        return None
    argument = function.body.block.args[0]
    element = str(getattr(argument.type, "element_type", ""))
    if element not in {"i16", "i32", "i64"} or any(n <= 0 for n in argument.type.get_shape()):
        return None
    roots = [op for op in function.body.block.ops if op.name == "linalg.generic"]
    if len(roots) != 1 or not is_compare_select_relu(roots[0]):
        return None
    root = roots[0]
    if root.inputs[0] is not argument or root.results[0].type != argument.type:
        return None
    shape = argument.type.get_shape()
    rank = len(shape)
    if any(it.data.value != "parallel" for it in root.get_iterator_types()) or any(
        m.data != AffineMap.identity(rank) for m in root.get_indexing_maps()
    ):
        return None
    if any(op.name not in {"tensor.empty", "linalg.generic", "func.return"} for op in function.body.block.ops):
        return None
    returned = function.body.block.last_op
    if returned.name != "func.return" or root.results[0] not in returned.operands:
        return None
    if any(v is not argument and v is not root.results[0] for v in returned.operands):
        return None
    width = int(element[1:])
    digits = (width - 1 + 6) // 7
    packed = "tensor<" + "x".join(map(str, (digits, *shape))) + "xi8>"
    original = str(argument.type)
    dims = [f"d{i}" for i in range(rank)]
    domain = "(" + ",".join(["j", *dims]) + ")"
    project = "affine_map<" + domain + "->(" + ",".join(dims) + ")>"
    fullmap = "affine_map<" + str(AffineMap.identity(rank + 1)) + ">"
    outmap = "affine_map<" + str(AffineMap.identity(rank)) + ">"
    slices = ["affine_map<(" + ",".join(dims) + ")->(" + ",".join([str(j), *dims]) + ")>" for j in range(digits)]
    decodes = []
    last = None
    for j in range(digits):
        decodes.append(f"%u{j}=arith.extui %b{j} : i8 to {element}")
        decodes.append(f"%s{j}=arith.constant {7 * j} : {element}")
        decodes.append(f"%v{j}=arith.shli %u{j},%s{j} : {element}")
        if last is None:
            last = f"%v{j}"
        else:
            decodes.append(f"%o{j}=arith.ori {last},%v{j} : {element}")
            last = f"%o{j}"
    results = ",".join("%out" if v is root.results[0] else "%x" for v in returned.operands)
    result_types = ",".join(str(v.type) for v in returned.operands)
    one_digit = "tensor<" + "x".join(map(str, (1, *shape))) + "xi8>"
    encodings = []
    for j in range(digits):
        encodings.append(f"""      %e{j}=tensor.empty() : {one_digit}
      %codes{j}=linalg.generic {{indexing_maps=[{project},{fullmap}],iterator_types={["parallel"] * (rank + 1)!r}}}
        ins(%x:{original}) outs(%e{j}:{one_digit}) {{
        ^bb0(%word:{element},%unused:i8):
        %shift=arith.constant {7 * j} : {element}
        %signshift=arith.constant {width - 1} : {element}
        %sign=arith.shrui %word,%signshift : {element}
        %signbyte=arith.trunci %sign : {element} to i8
        %bit7=arith.constant 7 : i8
        %high=arith.shli %signbyte,%bit7 : i8
        %lowword=arith.shrui %word,%shift : {element}
        %lowbyte=arith.trunci %lowword : {element} to i8
        %mask=arith.constant 127 : i8
        %low=arith.andi %lowbyte,%mask : i8
        %code=arith.ori %low,%high : i8
        linalg.yield %code : i8
      }} -> {one_digit}
""")
    text = f"""module {{func.func @forward(%x:{original}) -> ({result_types}) {{
      {chr(10).join(encodings)}
      %codes=tensor.concat dim(0) {",".join(f"%codes{j}" for j in range(digits))}
        : ({",".join([one_digit] * digits)}) -> {packed}
      %a=tensor.empty() : {packed}
      %activated=linalg.generic {{indexing_maps=[{fullmap},{fullmap}],iterator_types={["parallel"] * (rank + 1)!r}}}
        ins(%codes:{packed}) outs(%a:{packed}) {{
        ^bb1(%byte:i8,%unuseda:i8):
        %zero=arith.constant 0 : i8
        %negative=arith.cmpi slt,%byte,%zero : i8
        %positive=arith.select %negative,%zero,%byte : i8
        linalg.yield %positive : i8
      }} -> {packed}
      %d=tensor.empty() : {original}
      %out=linalg.generic {{indexing_maps=[{",".join([*slices, outmap])}],iterator_types={["parallel"] * rank!r}}}
        ins({",".join(["%activated"] * digits)}:{",".join([packed] * digits)}) outs(%d:{original}) {{
        ^bb2({",".join([f"%b{j}:i8" for j in range(digits)])},%unusedd:{element}):
        {chr(10).join(decodes)}
        linalg.yield {last} : {element}
      }} -> {original}
      return {results} : {result_types}
    }} }}""".replace("'parallel'", '"parallel"')
    prepared = parse_mlir_text(text)
    prepared.attributes.update(module.attributes)
    activation = next(op for op in prepared.walk() if is_compare_select_relu(op))
    activation.attributes.update(root.attributes)
    prepared.verify()
    from merlin.targetgen.native_unit_policy import construction_evidence

    if construction_evidence(prepared)["status"] != "eligible":
        return None
    operations = list(prepared.walk())
    data = activation.inputs[0]
    plan = compile_activation(
        directory,
        report,
        selection,
        provider,
        package,
        prepared,
        activation,
        data,
        dict(op_ordinal=operations.index(data.owner), result=0),
        storage_movements=(data.owner,),
    )
    if plan is None:
        return None
    plan["input_bound_proof"] = dict(
        method="full_dtype_storage_domain", dtypes=[element], absolute_bounds=[2 ** (width - 1)]
    )
    plan["native_semantic_proof"] = dict(
        status="proven",
        method="sign_replicated_seven_bit_digits",
        width=width,
        digits=digits,
        input_domain="all_storage_bit_patterns",
        host_operations="bit_selection_and_assembly_only",
        source_sha256=hashlib.sha256(str(module).encode()).hexdigest(),
    )
    return plan
