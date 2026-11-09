from xdsl.parser import Parser


def llvm_artifact(image, *, wrong_pointer=False):
    """The real emitted function uses the exact exported command image."""
    values = ", ".join((str(byte if byte < 128 else byte - 256) for byte in image))
    name = "unselected_image" if wrong_pointer else "fixture_commands"
    lines = [
        "module {",
        f"llvm.mlir.global constant @fixture_commands(dense<[{values}]> : tensor<{len(image)}xi8>) : !llvm.array<{len(image)} x i8>",
    ]
    if wrong_pointer:
        lines.append(
            f"llvm.mlir.global constant @unselected_image(dense<[{values}]> : tensor<{len(image)}xi8>) : !llvm.array<{len(image)} x i8>"
        )
    lines.extend(
        [
            "llvm.func @execute(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32",
            "llvm.func @fixture_kernel(%spaces: !llvm.ptr, %caps: !llvm.ptr, %counts: !llvm.ptr, %fault: i32) -> i32 {",
            f"%image = llvm.mlir.addressof @{name} : !llvm.ptr",
            f"%length = llvm.mlir.constant({len(image)} : i64) : i64",
            "%status = llvm.call @execute(%image, %length, %spaces, %caps, %counts, %fault) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32",
            "llvm.return %status : i32",
            "}",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def verify_artifact_image(expected, artifact):
    """Independently join actual function operand DAG and actual encoded calls."""
    from xdsl.dialects import llvm

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    module = Parser(make_llvm_context(), artifact).parse_module()
    module.verify()
    globals_ = [op for op in module.body.block.ops if isinstance(op, llvm.GlobalOp)]
    definitions = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.body.blocks]
    assert len(globals_) == 1 and len(definitions) == 1, "unrecognized emitted globals or function bodies"
    global_ = globals_[0]
    assert global_.sym_name.data == "fixture_commands"
    assert bytes((x & 255 for x in global_.value.get_values())) == expected, (
        "actual emitted call operands differ from selected IR"
    )
    definition = definitions[0]
    assert definition.sym_name.data == "fixture_kernel" and len(definition.body.blocks) == 1
    block = definition.body.block
    values = dict(zip(block.args, ("spaces", "caps", "counts", "fault"), strict=True))
    calls = []
    for op in block.ops:
        if isinstance(op, llvm.AddressOfOp):
            values[op.result] = ("global", op.global_name.root_reference.data)
        elif isinstance(op, llvm.ConstantOp):
            values[op.result] = op.value.value.data
        elif isinstance(op, llvm.CallOp):
            assert op.callee is not None
            calls.append((op.callee.root_reference.data, tuple((values[v] for v in op.operands))))
            assert len(op.results) == 1
            values[op.results[0]] = "status"
        else:
            assert isinstance(op, llvm.ReturnOp) and [values[v] for v in op.operands] == ["status"]
    assert calls == [("execute", (("global", "fixture_commands"), len(expected), "spaces", "caps", "counts", "fault"))]
    return expected


def artifact_image(artifact):
    from xdsl.dialects import llvm

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    module = Parser(make_llvm_context(), artifact).parse_module()
    module.verify()
    globals_ = [op for op in module.body.block.ops if isinstance(op, llvm.GlobalOp)]
    assert len(globals_) == 1 and globals_[0].sym_name.data == "fixture_commands"
    return bytes((x & 255 for x in globals_[0].value.get_values()))
