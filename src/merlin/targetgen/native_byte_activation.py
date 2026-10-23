"""Authenticate a complete signed-byte activation through the public catalog.

The public numeric-schedule dispatcher currently misses raw byte activation.
Its public catalog still supplies the source-bound native unit. Authenticate
that unit's emitted instructions, rather than insert a quantization marker or
call an internal provider admission function.
"""

import hashlib
import json
import re
import subprocess


def audit_program(text, symbol):
    """Check actual emitted RoCC operands and the identity weight bytes."""
    if f"define void @{symbol}(" not in text:
        raise ValueError("activation kernel symbol missing from emitted program")
    body = text.split(f"define void @{symbol}(", 1)[1].split("\n}", 1)[0]
    constant = re.search(r"@" + re.escape(symbol) + r'__relu_identity = .*?c"([^\"]*)"', text)
    if constant is None:
        raise ValueError("missing activation identity weights")
    encoded = constant.group(1)
    if re.fullmatch(r"(?:\\[0-9A-Fa-f]{2}){256}", encoded) is None:
        raise ValueError("unknown activation weight encoding")
    weights = bytes(int(x, 16) for x in re.findall(r"\\([0-9A-Fa-f]{2})", encoded))
    if weights != bytes(int(i == j) for i in range(16) for j in range(16)):
        raise ValueError("activation weights are not the proved identity")
    seen = dict(activated=0, unactivated=0, computes=0, readouts=0, input_loads=0, weight_loads=0)
    calls = 0
    loads = set()
    store = False
    for line in body.splitlines():
        if "call void asm sideeffect" not in line or '"fence"' in line:
            continue
        match = re.search(r'"\.insn r 0x7b, 0x3, 0x([0-9a-f]+), x0, \$0, \$1".*\(i64 (.*), i64 (-?\d+)\)', line)
        if match is None:
            raise ValueError("unrecognized activation instruction")
        funct, a, b = int(match[1], 16), match[2], int(match[3]) & ((1 << 64) - 1)
        calls += 1
        if funct not in {0, 1, 2, 3, 4, 6, 7}:
            raise ValueError("activation uses an unproved native opcode")
        if funct == 0:
            try:
                a = int(a) & ((1 << 64) - 1)
            except ValueError as exc:
                raise ValueError("dynamic activation configuration") from exc
            role = a & 3
            if role == 0:
                activation = (a >> 3) & 3
                if not a & 4 or a >> 32 != 0x3F800000 or b & 0xFFFFFFFF or activation not in {0, 1}:
                    raise ValueError("activation requires WS, unit scale, and zero rounding shift")
                seen["activated" if activation else "unactivated"] += 1
            elif role == 1:
                if a >> 32 != 0x3F800000 or a & 4:
                    raise ValueError("activation requires unit signed-byte mvin")
                loads.add((a >> 3) & 3)
            elif role == 2:
                if a != 2 or b >> 32 != 0x3F800000:
                    raise ValueError("activation requires raw, unactivated unit-scale readout")
                store = True
            else:
                raise ValueError("unproved activation configuration role")
        elif funct in {1, 2}:
            if (1 if funct == 1 else 0) not in loads:
                raise ValueError("activation DMA has no proved unit load configuration")
            if b & (1 << 31):
                raise ValueError("activation operand DMA must load byte scratchpad storage")
            if funct == 1 and f"@{symbol}__relu_identity" not in a:
                raise ValueError("activation weight DMA is not bound to the identity constant")
            seen["weight_loads" if funct == 1 else "input_loads"] += 1
        elif funct == 3:
            if not store:
                raise ValueError("activation has no proved unit store configuration")
            if b & 0xE0000000 != 0xA0000000:
                raise ValueError("activation must read raw full i32 ACC")
            seen["readouts"] += 1
        elif funct == 4:
            a = int(a) & ((1 << 64) - 1)
            k = (a >> 32) & 0xFFFF
            if not 1 <= k <= 16 or b & 0xFFFFFFFF != 0xFFFFFFFF:
                raise ValueError("activation MAC exceeds 20-bit-safe K or uses mesh D")
            seen["computes"] += 1
    if not all(seen.values()) or seen["computes"] != 2 * seen["readouts"]:
        raise ValueError("activation lacks its native ReLU and raw-widening work")
    return dict(
        method="emitted_identity_signed_byte_execute_relu_then_raw_i32",
        **seen,
        native_calls=calls,
        llvm_sha256=hashlib.sha256(text.encode()).hexdigest(),
        mesh_partial_absolute_bound=16 * 16384,
        mesh_signed_bits=20,
    )


def prepare(directory, report, selection, provider, package, module):
    from xdsl.ir.affine import AffineMap

    from merlin.xdsl_dialects.lowering.contraction_coverage import is_compare_select_relu

    if not callable(getattr(provider, "core_aten_catalog", None)):
        return None
    mesh = selection["spec"].get("numerical_semantics", {}).get("native_unit_contract", {}).get("mesh_mac", {})
    if mesh.get("signed_bits") != 20:
        return None
    functions = [op for op in module.walk() if op.name == "func.func"]
    if len(functions) != 1:
        return None
    function = functions[0]
    if function.sym_name.data != "forward" or len(function.body.blocks) != 1 or len(function.body.block.args) != 1:
        return None
    argument = function.body.block.args[0]
    if str(getattr(argument.type, "element_type", "")) != "i8" or any(n <= 0 for n in argument.type.get_shape()):
        return None
    roots = [op for op in function.body.block.ops if op.name == "linalg.generic"]
    if len(roots) != 1 or not is_compare_select_relu(roots[0]):
        return None
    root = roots[0]
    if root.inputs[0] is not argument or root.results[0].type != argument.type:
        return None
    if any(it.data.value != "parallel" for it in root.get_iterator_types()):
        return None
    identity = AffineMap.identity(len(argument.type.get_shape()))
    if any(m.data != identity for m in root.get_indexing_maps()):
        return None
    allowed = {"tensor.empty", "linalg.generic", "func.return"}
    if any(op.name not in allowed for op in function.body.block.ops):
        return None
    returned = function.body.block.last_op
    if returned.name != "func.return" or root.results[0] not in returned.operands:
        return None
    if any(v is not argument and v is not root.results[0] for v in returned.operands):
        return None
    return compile_activation(directory, report, selection, provider, package, module, root, argument, {"func_arg": 0})


def compile_activation(
    directory, report, selection, provider, package, module, root, argument, operand_binding, *, storage_movements=()
):
    """Authenticate one unit in a caller-proved representation program."""
    from pathlib import Path

    from merlin.targetgen.software_spec import admit_operation
    from merlin.targetgen.transfer_contracts import screen_transfer_contract

    mesh = selection["spec"].get("numerical_semantics", {}).get("native_unit_contract", {}).get("mesh_mac", {})
    if mesh.get("signed_bits") != 20:
        return None
    signature = dict(
        family="elementwise_map",
        numerical_family="F4",
        operand_dtype="i8",
        rank=len(argument.type.get_shape()),
        expression="affine",
        tensor_operands=1,
        scale_granularity="tensor",
        broadcasting="none",
        layout="row_major_contiguous",
        aliasing="disjoint_inputs_outputs",
    )
    decision = admit_operation(selection["spec"], "elementwise_map", signature, "accelerator")
    if decision["status"] != "admitted":
        return None
    for origin, destination, a, b in [("host", "accelerator", "i8", "i8"), ("accelerator", "host", "i32", "i32")]:
        if (
            screen_transfer_contract(
                selection["spec"],
                source_placement=origin,
                destination_placement=destination,
                operand_dtype=a,
                result_dtype=b,
                operand_layout="row_major_contiguous",
                result_layout="row_major_contiguous",
            )["status"]
            != "admitted"
        ):
            return None
    stage = directory / "native-byte-activation"
    stage.mkdir(exist_ok=True)
    source = str(module)
    (stage / "model.mlir").write_text(source)
    try:
        path, obj = provider.core_aten_catalog(stage / "model.mlir", stage, package=package)
        manifest = json.loads(Path(path).read_text())
        if obj is None or manifest.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("activation lacks a compiled source-bound catalog")
        operations = list(module.walk())
        ordinal = operations.index(root)
        bindings = {b["root_ordinal"]: b for b in manifest["bindings"]}
        kernels = {k["symbol"]: k for k in manifest["kernels"]}
        if len(bindings) != len(manifest["bindings"]) or set(bindings) != {
            ordinal,
            *(operations.index(op) for op in storage_movements),
        }:
            raise ValueError("activation requires exactly its proved native unit and storage movements")
        if set(kernels) != {b["symbol"] for b in bindings.values()}:
            raise ValueError("activation catalog has missing or extra kernels")
        for movement in storage_movements:
            moved = bindings[operations.index(movement)]
            numeric = moved.get("numeric", {})
            meta = kernels[moved["symbol"]].get("meta", {})
            if (
                movement.name != "tensor.concat"
                or numeric.get("family") != "F1"
                or numeric.get("quantization") != "none"
                or numeric.get("exact_arithmetic") is not True
                or numeric.get("byte_order") != "little"
                or meta.get("proof") != "closed_static_element_selectors"
                or meta.get("readout") != "scratchpad_bytes"
            ):
                raise ValueError("activation packing has no exact byte-selector proof")
            expected_inputs = {
                f"x{i}": dict(op_ordinal=operations.index(v.owner), result=v.index)
                for i, v in enumerate(movement.operands)
            }
            if moved.get("operands") != expected_inputs or moved.get("result") != {
                "y": dict(op_ordinal=operations.index(movement), result=0)
            }:
                raise ValueError("activation packing is bound to different storage")
        binding = bindings[ordinal]
        kernel = kernels[binding["symbol"]]
        if binding["root_ordinal"] != ordinal or binding["symbol"] != kernel["symbol"]:
            raise ValueError("activation provider bound a different source result")
        if binding.get("operands") != {"x": operand_binding} or binding.get("result") != {
            "y": {"op_ordinal": ordinal, "result": 0}
        }:
            raise ValueError("activation kernel arguments do not bind the complete source values")
        meta = kernel["meta"]
        expected = dict(
            operation="identity_signed_byte_then_ReLU",
            input_range=[-128, 127],
            output_range=[0, 127],
            shift=0,
            clipping_is_identity=True,
        )
        if (
            meta.get("form") != "exact_linear_i32"
            or meta.get("device_relu") is not True
            or meta.get("execute_activation_proof") != expected
            or meta.get("arithmetic") != "signed_byte_execute_relu"
            or meta.get("readout") != "full_i32"
            or meta.get("integer_weights") != ["1"]
            or meta.get("device_constant") is not False
        ):
            raise ValueError("activation provider retained or changed the arithmetic")
        from merlin.llvmlower.toolchain import llvm_install

        # The ordinary compiler's O2 simplification resolves static native
        # tile-loop operands. No runtime tensor values enter this analysis.
        argv = [
            str(llvm_install() / "bin/opt"),
            "-S",
            "-passes=default<O2>",
            str(stage / "core_aten.ll"),
            "-o",
            str(stage / "core_aten.audit.ll"),
        ]
        subprocess.run(argv, check=True, capture_output=True, text=True, timeout=60)
        audit = audit_program((stage / "core_aten.audit.ll").read_text(), kernel["symbol"])
        audit.update(argv=argv, original_llvm_sha256=hashlib.sha256((stage / "core_aten.ll").read_bytes()).hexdigest())
        readouts = kernel["requantization"]
        if len(readouts) != 1 or readouts[0].get("relu") is not False or readouts[0].get("rounding") != "half_to_even":
            raise ValueError("activation must not retain its threshold in host readout")
        from fractions import Fraction

        def exact_words(value):
            if isinstance(value, list):
                return [word for child in value for word in exact_words(child)]
            if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
                raise ValueError("activation readout has an unproved coefficient")
            return [Fraction(int(value["numerator"]), int(value["denominator"]))]

        readout = readouts[0]
        terms = readout.get("terms", [])
        if len(terms) != 1 or terms[0][0] != "acc_combined":
            raise ValueError("activation readout does not use its native accumulator")
        factors, offsets = exact_words(terms[0][1]), exact_words(readout.get("constant"))
        if not factors or not offsets or any(v != 1 for v in factors) or any(v != 0 for v in offsets):
            raise ValueError("activation retains numerical work in host readout")
        if (
            readout.get("zero_point") != 0
            or readout.get("quant_min") != -128
            or readout.get("quant_max") != 127
            or any(readout.get(k) is not None for k in ("permutations", "broadcasts", "after", "placement"))
        ):
            raise ValueError("activation readout changes the signed byte result")
    except (KeyError, IndexError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        (stage / "refusal.json").write_text(json.dumps(dict(reason=str(exc))) + "\n")
        return None
    plan = dict(
        selection=selection,
        status="admitted",
        numerical_family="F4",
        exact_dtype_sites=1,
        native_contractions=0,
        quantized_contractions=0,
        quantized_non_contraction_sites=0,
        sites=[dict(source_operation_ordinal=ordinal, family="elementwise_map", status="admitted", software=decision)],
        provider_catalog=manifest,
        native_activation_instruction_audit=audit,
        native_semantic_proof=dict(
            status="proven",
            method="complete_signed_byte_zero_threshold_domain",
            input_domain="all_storage_bit_patterns",
        ),
        input_bound_proof=dict(method="full_dtype_storage_domain", dtypes=["i8"], absolute_bounds=[128]),
        host_floor=dict(status="required", comparison="native_eager_bits_every_boundary_result"),
    )
    # Seal the same normalized bytes the public catalog authenticated.
    (directory / "model.mlir").write_text(source + "\n")
    (directory / "spec-int8-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
    for record in report["cases"]:
        if record.get("status") in {None, "bundled"}:
            record["numeric_admission"] = plan
    return plan
