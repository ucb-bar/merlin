"""Neutral generated-ISA adapter; all execution premises belong to this fixture."""

from dataclasses import replace

from test_readout_model_fusion import _EXECUTOR, KEYS, WORD, command, graph, params
from test_readout_model_fusion import SPACES as SPACES
from test_selected_fusion_contracts import compile_case, inputs
from xdsl.context import Context
from xdsl.dialects import arith, builtin
from xdsl.parser import Parser

from merlin.targetgen.isa_dialect_gen import build_dialect, encode_instruction


def fixture(directory, shape, width, separate):
    return place_source_model(
        inputs(directory, shape=shape, width=width, separate=separate),
        shape,
        width,
        separate,
    )


def place_source_model(model, shape, width, separate):
    """Make the original neutral source executable before optional allocation."""
    selected, kernel, objects, adapter, cap, codec, limits, options = model
    banks = {bank.name: bank for bank in graph(*shape, width, separate)[2]}
    cursors = {bank.backing: bank.start for bank in banks.values()}
    placed = []
    for obj in objects:
        bank = banks[obj.bank]
        alignment = max(obj.alignment, bank.alignment)
        physical = (max(cursors[bank.backing], bank.start) + alignment - 1) // alignment * alignment
        base = physical
        assert bank.start <= base and base + obj.size <= bank.capacity, (
            "original source slots exceed original declared capacity"
        )
        placed.append(replace(obj, source_base=base))
        cursors[bank.backing] = physical + obj.size
    calls = []
    for call in kernel.body:
        effect = adapter.project(call)
        assert effect is not None
        accesses = []
        for access in effect.accesses:
            found = [
                i
                for i, obj in enumerate(objects)
                if obj.bank == access.space
                and obj.source_base <= access.start
                and access.start + access.size <= obj.source_base + obj.size
            ]
            if access.space in banks:
                assert len(found) == 1
                i = found[0]
                access = replace(
                    access,
                    start=placed[i].source_base + access.start - objects[i].source_base,
                )
            accesses.append(access)
        calls.append(adapter.rewrite(call, tuple(accesses)))
    return (
        selected,
        replace(kernel, body=tuple(calls)),
        tuple(placed),
        adapter,
        cap,
        codec,
        limits,
        options,
    )


def target_text(kernel, selected):
    import io

    from xdsl.printer import Printer

    from merlin.llvmlower.target_schedule import instruction_module

    closed = replace(kernel, name="target_program", args=(), attrs=())
    module = instruction_module(closed, selected)
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def read_target(text, selected, original):
    """Read actual operand SSA; external role mapping is original fixture ABI."""
    selected.revalidate()
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, build_dialect(selected.isa)):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    values, calls = {}, []
    for op in module.body.block.ops:
        if isinstance(op, arith.ConstantOp):
            assert isinstance(op.value, builtin.IntegerAttr) and str(op.result.type) == "i32"
            values[op.result] = op.value.value.data
        else:
            prefix = selected.isa.namespace + "."
            assert op.name.startswith(prefix) and not op.results and not op.regions
            assert len(op.operands) == len(KEYS)
            calls.append(
                command(
                    op.name[len(prefix) :],
                    **dict(zip(KEYS, (values[v] for v in op.operands), strict=True)),
                )
            )
    selected.revalidate()
    return replace(original, body=tuple(calls))


def transform(text, model, *, enabled=True, physical=True):
    selected, original, *_ = model
    actual = read_target(text, selected, original)
    current = (model[0], actual, *model[2:])
    result = compile_case(current, enabled=enabled, emitters=physical)
    assert len(result.program.segments) == 1
    return target_text(result.kernel, selected), result


def image_from_target(text, model):
    selected, original, *_ = model
    kernel = read_target(text, selected, original)
    image = b"".join(
        encode_instruction(selected.isa, c.instr, params(c)).to_bytes(WORD.size, "little") for c in kernel.body
    )
    return image


def llvm_artifact(image, *, wrong_pointer=False):
    """The real emitted function uses the exact exported command image."""
    values = ", ".join(str(byte if byte < 128 else byte - 256) for byte in image)
    name = "unselected_image" if wrong_pointer else "fixture_commands"
    lines = [
        "module {",
        f"llvm.mlir.global constant @fixture_commands(dense<[{values}]> : tensor<{len(image)}xi8>) "
        f": !llvm.array<{len(image)} x i8>",
    ]
    if wrong_pointer:
        lines.append(
            f"llvm.mlir.global constant @unselected_image(dense<[{values}]> : tensor<{len(image)}xi8>) "
            f": !llvm.array<{len(image)} x i8>"
        )
    lines.extend(
        [
            "llvm.func @execute(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32",
            "llvm.func @fixture_kernel(%spaces: !llvm.ptr, %caps: !llvm.ptr, %counts: !llvm.ptr, %fault: i32) -> i32 {",
            f"%image = llvm.mlir.addressof @{name} : !llvm.ptr",
            f"%length = llvm.mlir.constant({len(image)} : i64) : i64",
            "%status = llvm.call @execute(%image, %length, %spaces, %caps, %counts, %fault) "
            ": (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32",
            "llvm.return %status : i32",
            "}",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def verify_artifact(selected_text, artifact, model):
    """Independently join actual function operand DAG and actual encoded calls."""
    from xdsl.dialects import llvm

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    expected = image_from_target(selected_text, model)
    module = Parser(make_llvm_context(), artifact).parse_module()
    module.verify()
    globals_ = [op for op in module.body.block.ops if isinstance(op, llvm.GlobalOp)]
    definitions = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.body.blocks]
    assert len(globals_) == 1 and len(definitions) == 1, "unrecognized emitted globals or function bodies"
    global_ = globals_[0]
    assert global_.sym_name.data == "fixture_commands"
    assert bytes(x & 255 for x in global_.value.get_values()) == expected, (
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
            calls.append((op.callee.root_reference.data, tuple(values[v] for v in op.operands)))
            assert len(op.results) == 1
            values[op.results[0]] = "status"
        else:
            assert isinstance(op, llvm.ReturnOp) and [values[v] for v in op.operands] == ["status"]
    assert calls == [
        (
            "execute",
            (
                ("global", "fixture_commands"),
                len(expected),
                "spaces",
                "caps",
                "counts",
                "fault",
            ),
        )
    ]
    return expected


# Reuse the existing independently implemented C instruction executor verbatim.
RUNTIME = _EXECUTOR


def recurrent_runtime(model):
    """Host recurrence surrounds the unchanged instruction implementation."""
    args = model[1].args
    xbytes, ybytes, hbytes = args[0].nbytes, args[-1].nbytes, args[-2].nbytes
    width = 1 if args[0].dtype == "i8" else 2
    return (
        RUNTIME
        + f"""
extern int fixture_kernel(uint8_t **, const uint64_t *, uint64_t *, int);
int fixture_model(uint8_t **spaces,const uint64_t *caps,uint8_t *state,uint32_t *position,
                  uint8_t *hidden_history,uint8_t *output_history,uint64_t *totals,int fault) {{
 uint8_t mixed[{xbytes}];uint8_t *original=spaces[3];
 for(int j=0;j<3;j++)totals[j]=0;
 for(int epoch=0;epoch<3;epoch++) {{
  for(unsigned i=0;i<{xbytes // width};i++) {{
   uint32_t value=get(original+{width}*i,{width})+state[i%{ybytes}]+*position;
   for(unsigned b=0;b<{width};b++)mixed[{width}*i+b]=(uint8_t)(value>>(8*b));
  }}
  spaces[3]=mixed;uint64_t counts[3];
  int status=fixture_kernel(spaces,caps,counts,fault);
  spaces[3]=original;if(status)return status;
  for(int j=0;j<3;j++)totals[j]+=counts[j];
  memcpy(hidden_history+epoch*{hbytes},spaces[8],{hbytes});
  memcpy(output_history+epoch*{ybytes},spaces[9],{ybytes});
  memcpy(state,spaces[9],{ybytes});(*position)++;
 }}
 return 0;
}}
"""
    )


def artifact_image(artifact):
    from xdsl.dialects import llvm

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    module = Parser(make_llvm_context(), artifact).parse_module()
    module.verify()
    globals_ = [op for op in module.body.block.ops if isinstance(op, llvm.GlobalOp)]
    assert len(globals_) == 1 and globals_[0].sym_name.data == "fixture_commands"
    return bytes(x & 255 for x in globals_[0].value.get_values())
