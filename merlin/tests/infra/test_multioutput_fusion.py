"""Actual byte projection, one producer generation and complete fan-out outputs."""

import hashlib
import json
import struct
from dataclasses import replace

import pytest
import yaml
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.parser import Parser

from merlin.compile.scheduling.fusion_semantics import ElementwiseSemantics, UnsupportedElementwise
from merlin.compile.scheduling.multioutput_fusion import (
    MultiOutputBindings,
    MultiOutputCompositionLaw,
    compile_source_multioutput,
    read_source_multioutput,
)
from merlin.compile.scheduling.multioutput_fusion_check import _arguments, verify_multioutput_fusion
from merlin.compile.scheduling.program_limits import (
    AddressWindow,
    DecodedInstruction,
    EncodedAccess,
    ProgramDecoder,
    ProgramLimits,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, ScheduleEncoding
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.compile.scheduling.temporary_storage import StorageAdapter, TemporaryObject
from merlin.sched.ir import Call, Const, Kernel, evaluate
from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction

KEYS = ("src0", "src1", "dst", "r0", "r1", "w", "elements")
SPACES = ("private", "alias", "input0", "input1", "output0", "output1")
OPS = ("produce", "clip", "rectify")
WORD = struct.Struct("<B7I")


def scalar_bodies(width):
    dtype = f"i{width}"
    low, high = (-7, 9) if width == 8 else (-128, 127)
    return (
        f"%v=arith.addi %a,%b:{dtype}",
        f"%lo=arith.constant {low}:{dtype}\n%hi=arith.constant {high}:{dtype}\n"
        f"%l=arith.maxsi %a,%lo:{dtype}\n%h=arith.minsi %l,%hi:{dtype}\n"
        + ("%v=arith.trunci %h:i16 to i8" if width == 16 else f"%v=arith.addi %h,%lo:{dtype}"),
        f"%z=arith.constant 0:{dtype}\n%v=arith.maxsi %a,%z:{dtype}",
    )


def scalar_sources(width):
    dtype = f"i{width}"
    return tuple(
        f"builtin.module {{func.func @stage({args})->{result} {{\n{body}\nfunc.return %v:{result}\n}}}}".encode()
        for args, result, body in zip(
            (f"%a:{dtype},%b:{dtype}", f"%a:{dtype}", f"%a:{dtype}"),
            (dtype, "i8", dtype),
            scalar_bodies(width),
            strict=True,
        )
    )


def source_program(width=16, shape=(3, 5)):
    shape = "x".join(map(str, shape))
    dtype = f"i{width}"
    bodies = scalar_bodies(width)
    text = f"""builtin.module {{func.func @forward(%x:tensor<{shape}x{dtype}>,%bias:tensor<{shape}x{dtype}>)
    ->(tensor<{shape}xi8>,tensor<{shape}x{dtype}>) {{
    %pinit=tensor.empty():tensor<{shape}x{dtype}>
    %ainit=tensor.empty():tensor<{shape}xi8>
    %binit=tensor.empty():tensor<{shape}x{dtype}>
    """
    for i, (inputs, output, input_types, result) in enumerate(
        (
            ("%x,%bias", "%pinit", f"tensor<{shape}x{dtype}>,tensor<{shape}x{dtype}>", dtype),
            ("%r0", "%ainit", f"tensor<{shape}x{dtype}>", "i8"),
            ("%r0", "%binit", f"tensor<{shape}x{dtype}>", dtype),
        )
    ):
        types = [dtype, dtype] if i == 0 else [dtype]
        args = ",".join(f"%{name}:{typ}" for name, typ in zip(("a", "b"), types, strict=False))
        maps = ",".join(["affine_map<(d0,d1)->(d0,d1)>"] * (len(types) + 1))
        text += f"""%r{i}="linalg.generic"({inputs},{output}) <{{indexing_maps=[{maps}],
        iterator_types=[#linalg.iterator_type<parallel>,#linalg.iterator_type<parallel>],
        operandSegmentSizes=array<i32:{len(types)},1>}}> ({{^bb0({args},%init:{result}):
        {bodies[i]}
        "linalg.yield"(%v):({result})->()}}):({input_types},tensor<{shape}x{result}>)->tensor<{shape}x{result}>\n"""
    text += f"func.return %r1,%r2:tensor<{shape}xi8>,tensor<{shape}x{dtype}> }} }}"
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, func.Func, linalg.Linalg, tensor.Tensor):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return module, tuple(op for op in module.walk() if op.name == "linalg.generic")


def model(tmp_path, width=16, shape=(3, 5), unified=False, capacity=512):
    bits, byte = f"i{width}", width // 8
    source = scalar_sources(width)
    count = shape[0] * shape[1]

    def call(name, *, src0=0, src1=0, dst=0, r0=0, r1=0, w=0, elements=count):
        return Call(
            name,
            tuple((key, Const(value)) for key, value in zip(KEYS, (src0, src1, dst, r0, r1, w, elements), strict=True)),
        )

    def values(call):
        return {key: evaluate(call.arg(key), {}) for key in KEYS}

    def effects(call):
        if call.instr not in OPS:
            return None
        v, role = values(call), OPS.index(call.instr)
        reads = (EncodedAccess(SPACES[v["src0"]], v["r0"], v["elements"] * byte),)
        if role == 0:
            reads += (EncodedAccess(SPACES[v["src1"]], v["r1"], v["elements"] * byte),)
        output_width = 1 if role == 1 else byte
        return StorageEffects(
            call.instr,
            (*reads, EncodedAccess(SPACES[v["dst"]], v["w"], v["elements"] * output_width, True)),
            "synchronous",
            elementwise=ElementwiseSemantics(
                source[role], v["elements"], (call.instr,), ("little",) * (len(reads) + 1)
            ),
        )

    def rewrite(call, accesses):
        positions = {"r0": accesses[0].start, "w": accesses[-1].start}
        if call.instr == "produce":
            positions["r1"] = accesses[1].start
        return replace(call, args=tuple((key, Const(positions.get(key, value.value))) for key, value in call.args))

    from merlin.compile.scheduling.temporary_storage import StorageEffects

    facts = dict(
        facts=dict(
            target="neutral",
            word_bits=WORD.size * 8,
            arg_bits=32,
            opcode_bits=8,
            offsets=[8 + i * 32 for i in range(7)],
            codes=[0, 1, 2],
            selectors=list(range(len(SPACES))),
            zero=0,
            capacity=capacity,
            width=width,
            byte=8,
            alignment=byte,
        )
    )

    def ref(*parts):
        return {"fact": ["facts", *parts]}

    def arg(name):
        return {"argument": name}

    def selector(name):
        return dict(
            argument=name, choices=[dict(value=ref("selectors", i), resource=space) for i, space in enumerate(SPACES)]
        )

    instructions, declarations, elementwise = [], {}, {}
    for role, name in enumerate(OPS):
        fields = [dict(offset=ref("zero"), width=ref("opcode_bits"), fixed=ref("codes", role))] + [
            dict(offset=ref("offsets", i), width=ref("arg_bits"), argument=key) for i, key in enumerate(KEYS)
        ]
        instructions.append(
            dict(
                name=name,
                operands=[
                    dict(name=key, type=dict(kind="integer", bits=ref("arg_bits"), signed=False), unit="integer")
                    for key in KEYS
                ],
                results=[],
                immediates=[],
                encoding=fields,
                semantics=dict(primitive="neutral." + name, bindings={key: key for key in KEYS}),
                effects=dict(reads=list(SPACES), writes=list(SPACES)),
            )
        )

        def access(sel, address, size_bits, writes=False):
            return dict(
                space=selector(sel),
                start=arg(address),
                size=dict(elements=[arg("elements")], element_bits=ref(size_bits)),
                writes=writes,
            )

        accesses = [access("src0", "r0", "width")]
        if role == 0:
            accesses.append(access("src1", "r1", "width"))
        accesses.append(access("dst", "w", "byte" if role == 1 else "width", True))
        declarations[name] = dict(accesses=accesses, completion="synchronous", token=None, waits=[], exact_copy=False)
        elementwise[name] = dict(
            scalar_ir=source[role].decode(),
            elements=[arg("elements")],
            element_bits=[ref("width")] * (2 if role == 0 else 1) + [ref("byte" if role == 1 else "width")],
            stages=[name],
            byte_orders=["little"] * len(accesses),
            readout=None,
            execution="contiguous_read_then_write",
        )
    storage = dict(
        schema="merlin.temporary_storage_contract.v1",
        banks={
            name: dict(
                backing="shared" if unified else name,
                start=ref("zero"),
                size=ref("capacity"),
                alignment=ref("alignment"),
                ownership="invocation_private",
            )
            for name in SPACES[:2]
        },
        external={
            name: dict(backing=name, start=ref("zero"), size=ref("capacity"), ownership="runtime_managed")
            for name in SPACES[2:]
        },
        reservations=[],
        instructions=declarations,
    )
    software = dict(
        schema="merlin.software_spec.v1",
        target="neutral",
        status="reviewed",
        isa=dict(
            schema="merlin.isa_spec.v1",
            namespace="multioutput",
            facts_sha256="0" * 64,
            word_bits=ref("word_bits"),
            resources=[dict(name=name, kind="memory", unit="bytes", capacity=ref("capacity")) for name in SPACES],
            instructions=instructions,
        ),
        scheduling=dict(
            schema="merlin.schedule_effects.v1",
            numerical_policy="modular-" + bits,
            instruction_byteorder="little",
            configuration={},
            storage=storage,
            fusion=dict(schema="merlin.physical_fusion_contract.v1", elementwise=elementwise, readouts={}),
        ),
    )
    fp, sp = tmp_path / "facts.json", tmp_path / "software.yaml"
    fp.write_text(json.dumps(facts))
    software["isa"]["facts_sha256"] = hashlib.sha256(fp.read_bytes()).hexdigest()
    sp.write_text(yaml.safe_dump(software))
    selected = read_scheduling_contracts(fp, sp, target="neutral")
    adapter = StorageAdapter(
        effects, rewrite, selected.hardware_identity, selected.numerical_policy, "neutral-explicit-v1"
    )

    def encode(kernel):
        return b"".join(
            encode_instruction(selected.isa, c.instr, values(c)).to_bytes(WORD.size, "little") for c in kernel.body
        )

    def decode(image):
        rows = []
        for i in range(0, len(image), WORD.size):
            name, v = decode_instruction(selected.isa, int.from_bytes(image[i : i + WORD.size], "little"))
            rows.append(call(name, **v))
        return tuple(rows)

    def instructions(image):
        return tuple(
            DecodedInstruction(
                i * WORD.size, WORD.size, effects(c).accesses, completion="synchronous", relocatable=True
            )
            for i, c in enumerate(decode(image))
        )

    codec = ScheduleEncoding(
        encode,
        decode,
        ProgramDecoder(instructions, adapter.configuration_identity, "native-neutral"),
        adapter.numerical_policy_identity,
        "native-neutral",
    )
    module, operations = source_program(width, shape)
    logical = read_source_multioutput(module, operations)
    kernel = Kernel(
        "fanout",
        _arguments(logical),
        (
            call("produce", src0=2, src1=3, dst=0, w=3072),
            call("clip", src0=1 if unified else 0, dst=4, r0=3072),
            call("rectify", src0=0, dst=5, r0=3072),
        ),
    )
    law = MultiOutputCompositionLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
        "original-neutral-private-fanout-v1",
        ("produce", "produce"),
        (("clip", "clip"), ("rectify", "rectify")),
        complete_projected_effects=True,
        synchronous_private_composition=True,
        suppressed_intermediate_transfers_no_faults=True,
        suppressed_intermediate_transfers_unobserved=True,
    )
    bindings = MultiOutputBindings(
        (AddressWindow("input0", 0, count * byte), AddressWindow("input1", 0, count * byte)),
        (AddressWindow("output0", 0, count), AddressWindow("output1", 0, count * byte)),
        ("little",) * 4,
        True,
    )
    return (
        module,
        operations,
        dict(
            kernel=kernel,
            objects=(TemporaryObject("P", "private", 3072, count * byte, byte),),
            adapter=adapter,
            encoding=codec,
            limits=ProgramLimits(16 * WORD.size, 16, 1, tuple(AddressWindow(s, 0, 512) for s in SPACES), ()),
            selected_contracts=selected,
            bindings=bindings,
            composition_law=law,
            budgets=ScheduleBudgets(max_fusions=0, prefetch_distance=0),
            private_ownership=True,
        ),
    )


@pytest.mark.parametrize("width,shape,unified", [(8, (3, 5), False), (16, (2, 11), True)])
def test_actual_final_bytes_keep_one_producer_and_both_outputs(tmp_path, width, shape, unified):
    module, operations, options = model(tmp_path, width, shape, unified)
    result = compile_source_multioutput(module, operations, **options)
    assert len(result.images()) == 1
    assert result.conformance["address_scope"] == "actual_physical"
    assert result.conformance["producer_evaluations"] == shape[0] * shape[1]
    assert result.conformance["producer_instructions"] == 1
    assert result.conformance["epilogue_instructions"] == [1, 1]
    assert result.conformance["actual_commands"] == 3
    result.revalidate()
    execute_actual_bytes(result, width, unified, tmp_path)


def direct_case(options):
    selected, raw = options["selected_contracts"], options["adapter"]
    banks, _, external, effective = selected.storage.bind(
        selected.isa, raw, banks=None, reservations=(), external_windows=None, private_ownership=True
    )
    effective, _, _ = selected.fusion.bind(
        effective,
        emitters={},
        qualification_identity=options["composition_law"].qualification_identity,
        max_operations=128,
        max_ir_bytes=65536,
    )
    kernel = options["kernel"]
    changed = tuple(
        raw.rewrite(
            call,
            tuple(
                replace(access, start=access.start - 3008) if access.space in SPACES[:2] else access
                for access in raw.project(call).accesses
            ),
        )
        for call in kernel.body
    )
    return dict(
        kernel=replace(kernel, body=changed),
        objects=tuple(replace(obj, source_base=64) for obj in options["objects"]),
        banks=banks,
        adapter=effective,
        role_adapter=raw,
        bindings=options["bindings"],
        composition_law=options["composition_law"],
        external_windows=external,
        private_ownership=True,
        budgets=options["budgets"],
    )


def change(call, **values):
    return replace(call, args=tuple((key, Const(values.get(key, value.value))) for key, value in call.args))


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-producer",
        "missing-output",
        "swapped-outputs",
        "wrong-P-address",
        "wrong-input",
        "partial-producer",
        "wrong-output-offset",
    ],
)
def test_actual_instruction_mutations_cannot_publish_a_valid_pair(tmp_path, mutation):
    module, operations, options = model(tmp_path, 8)
    source = read_source_multioutput(module, operations)
    case = direct_case(options)
    producer, e0, e1 = case["kernel"].body
    body = {
        "duplicate-producer": (producer, producer, e0, e1),
        "missing-output": (producer, e0),
        "swapped-outputs": (producer, change(e0, dst=5), change(e1, dst=4)),
        "wrong-P-address": (producer, change(e0, r0=65), e1),
        "wrong-input": (change(producer, src0=3), e0, e1),
        "partial-producer": (change(producer, elements=14), e0, e1),
        "wrong-output-offset": (producer, change(e0, w=1), e1),
    }[mutation]
    case["kernel"] = replace(case["kernel"], body=body)
    with pytest.raises(ValueError):
        verify_multioutput_fusion(source, **case)


@pytest.mark.parametrize(
    "flag",
    [
        "complete_projected_effects",
        "synchronous_private_composition",
        "suppressed_intermediate_transfers_no_faults",
        "suppressed_intermediate_transfers_unobserved",
    ],
)
def test_missing_original_effect_law_declines_before_emission(tmp_path, flag):
    module, operations, options = model(tmp_path)
    options["composition_law"] = replace(options["composition_law"], **{flag: False})
    with pytest.raises(UnsupportedElementwise, match="unavailable"):
        compile_source_multioutput(module, operations, **options)
    with pytest.raises(ValueError, match="Boolean"):
        replace(options["composition_law"], **{flag: 1})


@pytest.mark.parametrize("mutation", ["unknown", "async", "wrong-rounding", "wrong-byte-order"])
def test_effect_or_scalar_mutations_are_not_inferred_legal(tmp_path, mutation):
    from merlin.compile.scheduling.temporary_storage import StorageEffects

    module, operations, options = model(tmp_path)
    case = direct_case(options)
    raw = options["adapter"]
    original = raw.project

    def wrong(call):
        effect = original(call)
        if call.instr != "clip":
            return effect
        if mutation == "unknown":
            return None
        if mutation == "async":
            return replace(effect, completion="asynchronous", token="pending")
        if mutation == "wrong-rounding":
            return replace(
                effect,
                elementwise=replace(
                    effect.elementwise,
                    function_ir=effect.elementwise.function_ir.replace(b"arith.maxsi", b"arith.minsi"),
                ),
            )
        return replace(effect, elementwise=replace(effect.elementwise, byte_orders=("big", "little")))

    assert type(original(case["kernel"].body[0])) is StorageEffects
    case["adapter"] = case["role_adapter"] = replace(raw, project=wrong)
    with pytest.raises(ValueError):
        verify_multioutput_fusion(read_source_multioutput(module, operations), **case)


def test_observed_producer_and_unknown_intervening_effect_preserve_original(tmp_path):
    from xdsl.dialects.builtin import UnregisteredOp

    from merlin.transforms.certificates import ir_digest

    module, operations = source_program()
    observer = func.CallOp("observer", [operations[0].results[0]], [])
    module.body.block.add_op(func.FuncOp.external("observer", [operations[0].results[0].type], []))
    operations[-1].parent.insert_op_after(observer, operations[-1])
    module.verify()
    before = ir_digest(module)
    with pytest.raises(UnsupportedElementwise, match="observed or escaping"):
        read_source_multioutput(module, operations)
    assert ir_digest(module) == before
    module, operations = source_program()
    operations[0].parent.insert_op_after(UnregisteredOp.with_name("unknown.effect").create(), operations[0])
    before = ir_digest(module)
    with pytest.raises(UnsupportedElementwise, match="crosses"):
        read_source_multioutput(module, operations)
    assert ir_digest(module) == before


def test_exact_capacity_exhaustion_refuses_without_an_incomplete_candidate(tmp_path):
    from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable

    module, operations, options = model(tmp_path, capacity=8)
    with pytest.raises(StorageOptimizationUnavailable):
        compile_source_multioutput(module, operations, **options)


@pytest.mark.parametrize(
    "helper", ["_elementwise", "_problem", "_source_state", "_checker_functions", "_same_callback", "memory-read"]
)
def test_projector_cannot_replace_the_independent_semantic_checker(tmp_path, monkeypatch, helper):
    from merlin.compile.scheduling import multioutput_fusion_check as checker

    module, operations, options = model(tmp_path)
    case = direct_case(options)
    raw = options["adapter"]
    target, name = (checker._Memory, "read") if helper == "memory-read" else (checker, helper)
    original = getattr(target, name)
    calls = []

    def project(call):
        monkeypatch.setattr(target, name, lambda *args, **kwargs: calls.append(args))
        return raw.project(call)

    case["adapter"] = case["role_adapter"] = replace(raw, project=project)
    try:
        with pytest.raises(ValueError, match="checker implementation"):
            verify_multioutput_fusion(read_source_multioutput(module, operations), **case)
        assert calls == []
    finally:
        monkeypatch.setattr(target, name, original)


@pytest.mark.parametrize("mutation", ["law", "source", "selected", "reader"])
def test_first_original_owner_cannot_rebase_frozen_inputs(tmp_path, monkeypatch, mutation):
    from merlin.compile.scheduling import multioutput_fusion as core
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts

    module, operations, options = model(tmp_path)
    selected, law = options["selected_contracts"], options["composition_law"]
    original = SelectedSchedulingContracts.revalidate
    reader = core._functions

    def changed(owner):
        if owner is selected:
            if mutation == "law":
                object.__setattr__(law, "suppressed_intermediate_transfers_unobserved", False)
            elif mutation == "source":
                operations[0].attributes["unknown"] = builtin.StringAttr("changed")
            elif mutation == "selected":
                object.__setattr__(selected, "numerical_policy", "different")
            else:
                monkeypatch.setattr(core, "_functions", lambda: ())
        return original(owner)

    monkeypatch.setattr(SelectedSchedulingContracts, "revalidate", changed)
    try:
        with pytest.raises(ValueError, match="changed; replan"):
            compile_source_multioutput(module, operations, **options)
    finally:
        monkeypatch.setattr(core, "_functions", reader)


def test_emitter_ignoring_selected_final_consumer_is_rejected(tmp_path):
    module, operations, options = model(tmp_path)
    codec = options["encoding"]
    options["encoding"] = replace(codec, encode=lambda k: codec.encode(replace(k, body=k.body[:-1])))
    with pytest.raises(ValueError):
        compile_source_multioutput(module, operations, **options)


def test_issued_program_and_provenance_mutations_cannot_reuse_a_proof(tmp_path):
    module, operations, options = model(tmp_path)
    result = compile_source_multioutput(module, operations, **options)
    for field, value in (
        ("operation_ordinals", (0, 1, 2)),
        ("source_passes", ()),
        ("outputs", result.outputs[::-1]),
        ("_guard", lambda candidate: None),
    ):
        before = getattr(result, field)
        assert value != before
        try:
            object.__setattr__(result, field, value)
            with pytest.raises(ValueError):
                result.revalidate()
        finally:
            object.__setattr__(result, field, before)
        result.revalidate()
    with pytest.raises(ValueError, match="issuance"):
        replace(result).revalidate()


@pytest.mark.parametrize("field", ["graph", "scalar_ir"])
def test_projector_cannot_rebase_detached_expected_source(tmp_path, field):
    module, operations, options = model(tmp_path)
    source = read_source_multioutput(module, operations)
    case = direct_case(options)
    raw = options["adapter"]
    before = getattr(source.producer, field)

    def project(call):
        replacement = source.epilogues[1].graph if field == "graph" else source.epilogues[1].scalar_ir
        object.__setattr__(source.producer, field, replacement)
        return raw.project(call)

    case["adapter"] = case["role_adapter"] = replace(raw, project=project)
    try:
        with pytest.raises(ValueError, match="kernel or law changed"):
            verify_multioutput_fusion(source, **case)
    finally:
        object.__setattr__(source.producer, field, before)


def execute_actual_bytes(result, width, unified, directory):
    """Independent scalar byte machine; no selected projector/reference calls."""
    image = b"".join(result.images())
    count = result.conformance["producer_evaluations"]
    byte, half = width // 8, 1 << (width - 1)
    left = ([-half, half - 1, half - 1, 0, -1, 1, -7, 127] * count)[:count]
    right = ([-1, 1, half - 1, -half, -half, 0, 32, -128] * count)[:count]
    memories = [bytearray([193] * 520) for _ in SPACES]
    if unified:
        memories[1] = memories[0]
    for i, values in zip((2, 3), (left, right), strict=True):
        memories[i][: count * byte] = b"".join(value.to_bytes(byte, "little", signed=True) for value in values)
    before = tuple(bytes(memory) for memory in memories)
    counters = [0, 0, 0]
    for opcode, src0, src1, dst, r0, r1, out, elements in WORD.iter_unpack(image):
        assert 0 <= opcode < 3
        for i in range(elements):
            value = int.from_bytes(memories[src0][r0 + i * byte : r0 + (i + 1) * byte], "little", signed=True)
            if opcode == 0:
                other = int.from_bytes(memories[src1][r1 + i * byte : r1 + (i + 1) * byte], "little", signed=True)
                value = (value + other + half) % (2 * half) - half
            elif opcode == 1:
                value = max(-7 if width == 8 else -128, min(9 if width == 8 else 127, value))
                if width == 8:
                    value -= 7
            else:
                value = max(value, 0)
            size = 1 if opcode == 1 else byte
            memories[dst][out + i * size : out + (i + 1) * size] = value.to_bytes(size, "little", signed=True)
            counters[opcode] += 1
    expected0, expected1 = [], []
    for a, b in zip(left, right, strict=True):
        p = (a + b + half) % (2 * half) - half
        expected0.append(max(-7 if width == 8 else -128, min(9 if width == 8 else 127, p)) - (7 if width == 8 else 0))
        expected1.append(max(p, 0))
    actual = (bytes(memories[4][:count]), bytes(memories[5][: count * byte]))
    assert actual == (
        bytes(value % 256 for value in expected0),
        b"".join(value.to_bytes(byte, "little", signed=True) for value in expected1),
    )
    assert counters == [count, count, count]
    assert tuple(bytes(memory) for memory in memories[2:4]) == before[2:4]
    assert all(bytes(memory)[512:] == before[i][512:] for i, memory in enumerate(memories))
    assert bytes(memories[4])[count:] == before[4][count:]
    assert bytes(memories[5])[count * byte :] == before[5][count * byte :]
    (directory / "actual-program.bin").write_bytes(image)
    (directory / "scalar-byte-replay.json").write_text(
        json.dumps(
            dict(
                scope="neutral Python byte execution, not native/target qualification",
                width=width,
                input_hex=[before[2].hex(), before[3].hex()],
                actual_outputs=[value.hex() for value in actual],
                expected_values=[expected0, expected1],
                actual_counters=counters,
                guards_unchanged=True,
            )
        )
    )


def test_copied_P_cannot_replace_the_retained_private_generation(tmp_path):
    from merlin.compile.scheduling.temporary_storage import StorageEffects

    module, operations, options = model(tmp_path)
    case = direct_case(options)
    raw, body = options["adapter"], case["kernel"].body
    count = read_source_multioutput(module, operations).shape[0] * 5
    copied = change(body[0], src0=0, dst=1, r0=64, w=128)
    copied = replace(copied, instr="copy")

    def project(call):
        if call.instr != "copy":
            return raw.project(call)
        return StorageEffects(
            "exact-bit-copy",
            (EncodedAccess("private", 64, 2 * count), EncodedAccess("alias", 128, 2 * count, True)),
            "synchronous",
            exact_copy=True,
        )

    case["adapter"] = case["role_adapter"] = replace(raw, project=project)
    case["kernel"] = replace(case["kernel"], body=(body[0], copied, change(body[1], src0=1, r0=128), body[2]))
    case["objects"] += (TemporaryObject("copied-P", "alias", 128, 2 * count, 2),)
    with pytest.raises(UnsupportedElementwise, match="only stage unchanged original inputs"):
        verify_multioutput_fusion(read_source_multioutput(module, operations), **case)


def test_unavailable_runtime_disjointness_preserves_original_route(tmp_path):
    module, operations, options = model(tmp_path)
    options["bindings"] = replace(options["bindings"], external_buffers_disjoint=False)
    with pytest.raises(UnsupportedElementwise, match="ownership"):
        compile_source_multioutput(module, operations, **options)


def test_direct_final_checker_does_not_accept_virtual_addresses_as_physical(tmp_path):
    module, operations, options = model(tmp_path)
    case = direct_case(options)
    case["kernel"], case["objects"] = options["kernel"], options["objects"]
    with pytest.raises(ValueError, match="physical bank bounds"):
        verify_multioutput_fusion(read_source_multioutput(module, operations), **case)


def test_equal_text_function_clone_is_rejected_before_encoding(tmp_path, monkeypatch):
    from merlin.transforms.certificates import ir_digest

    module, operations, options = model(tmp_path)
    selected, codec = options["selected_contracts"], options["encoding"]
    original_record = type(selected).record
    before = ir_digest(module)
    encoded = []

    def encode(kernel):
        encoded.append("encode")
        return codec.encode(kernel)

    def decode(image):
        encoded.append("decode")
        return codec.decode_calls(image)

    def record(owner):
        if owner is selected:
            function = tuple(module.body.block.ops)[0]
            replacement = function.clone()
            module.body.block.detach_op(function)
            module.body.block.add_op(replacement)
            module.verify()
            assert ir_digest(module) == before
            assert operations[0] not in tuple(module.walk())
        return original_record(owner)

    options["encoding"] = replace(codec, encode=encode, decode_calls=decode)
    monkeypatch.setattr(type(selected), "record", record)
    with pytest.raises(ValueError, match="original source, implementation or selection changed"):
        compile_source_multioutput(module, operations, **options)
    assert encoded == []
