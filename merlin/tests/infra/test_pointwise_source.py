"""Current tensor source to actual fusion bytes; neutral functional execution."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from test_readout_model_fusion import _EXECUTOR, EPILOGUE, SPACES, WORD, command
from test_selected_fusion_contracts import inputs
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg
from xdsl.parser import Parser

from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise
from merlin.compile.scheduling.pointwise_source import (
    PointwiseBindings,
    compile_source_pointwise,
    read_pointwise_source,
)
from merlin.compile.scheduling.program_limits import AddressWindow, DecodedInstruction, ProgramLimits
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.compile.scheduling.temporary_storage import TemporaryObject
from merlin.sched.ir import Kernel, TensorArg


def source_text(m=3, n=5):
    shape = f"{m}x{n}"
    return f"""builtin.module {{
 func.func @forward(%x:tensor<{shape}xi32>,%bias:tensor<{shape}xi32>,%init:tensor<{shape}xi8>)
 ->tensor<{shape}xi8> {{
 %out="linalg.generic"(%x,%bias,%init) <{{
 indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],
 iterator_types=[#linalg.iterator_type<parallel>,#linalg.iterator_type<parallel>],
 operandSegmentSizes=array<i32:2,1>}}> ({{
 ^bb0(%a:i32,%b:i32,%initial:i8):
 %sum=arith.addi %a,%b:i32
 %low=arith.constant -128:i32
 %high=arith.constant 127:i32
 %lower=arith.maxsi %sum,%low:i32
 %bounded=arith.minsi %lower,%high:i32
 %result=arith.trunci %bounded:i32 to i8
 "linalg.yield"(%result):(i8)->()
 }}) : (tensor<{shape}xi32>,tensor<{shape}xi32>,tensor<{shape}xi8>)->tensor<{shape}xi8>
 func.return %out:tensor<{shape}xi8>
 }} }}"""


def source_program(text=None, *, m=3, n=5):
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, func.Func, linalg.Linalg):
        context.load_dialect(dialect)
    module = Parser(context, source_text(m, n) if text is None else text).parse_module()
    module.verify()
    return module, next(op for op in module.walk() if op.name == "linalg.generic")


def case(tmp_path, *, m=3, n=5, separate=False, mutate=None):
    selected, _, _, adapter, cap, codec, _, _ = inputs(tmp_path, separate=separate, mutate=mutate)
    kernel = Kernel(
        "source_epilogue",
        (
            TensorArg("x", (m, n), "i32", "read"),
            TensorArg("b0", (m, n), "i32", "read"),
            TensorArg("y", (m, n), "i8", "write"),
        ),
        (
            command("readout", m=m, n=n, src=3, dst=1, source=0, destination=3072),
            command("finish", m=m, n=n, src=1, dst=9, source=3072, bias=6),
        ),
    )
    bindings = PointwiseBindings(
        (AddressWindow("x", 0, 4 * m * n), AddressWindow("b0", 0, 4 * m * n)),
        AddressWindow("y", 0, m * n),
        ("little",) * 3,
        True,
    )
    limits = ProgramLimits(16 * WORD.size, 16, 1, tuple(AddressWindow(s, 0, 1024) for s in SPACES), ())
    return dict(
        kernel=kernel,
        objects=(TemporaryObject("stage", "stage", 3072, 4 * m * n, 4),),
        adapter=adapter,
        encoding=codec,
        limits=limits,
        selected_contracts=selected,
        bindings=bindings,
        private_ownership=True,
        fusion_emitters={cap.readout.selector: cap.emit},
        fusion_qualification_identity="independent-neutral-two-layer-v1",
    )


@pytest.fixture
def execute(tmp_path, request):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    source, library = tmp_path / "execute.c", tmp_path / "execute.so"
    source.write_text(getattr(request, "param", _EXECUTOR))
    subprocess.run(
        [compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True
    )
    bytep = ctypes.POINTER(ctypes.c_uint8)
    fn = ctypes.CDLL(str(library)).execute
    fn.argtypes = [
        bytep,
        ctypes.c_size_t,
        ctypes.POINTER(bytep),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
    ]

    def run(images, data, *, separate=False, wrong=False):
        memory = [(ctypes.c_uint8 * 1032)(*([193] * 1032)) for _ in SPACES]
        if not separate:
            memory[2] = memory[1]
        for name, payload in data.items():
            ctypes.memmove(memory[SPACES.index(name)], payload, len(payload))
        before = [bytes(m) for m in memory]
        pointers = (bytep * len(SPACES))(*(ctypes.cast(m, bytep) for m in memory))
        capacities = (ctypes.c_uint64 * len(SPACES))(*([1024] * len(SPACES)))
        image = b"".join(images)
        raw = (ctypes.c_uint8 * len(image)).from_buffer_copy(image)
        counters = (ctypes.c_uint64 * 3)()
        assert fn(raw, len(image), pointers, capacities, counters, wrong) == 0
        for i, name in enumerate(SPACES):
            assert bytes(memory[i])[1024:] == before[i][1024:]
            if name in SPACES[3:-1]:
                assert bytes(memory[i]) == before[i]
        return bytes(memory[-1]), tuple(counters)

    return run


@pytest.mark.parametrize("m,n,separate", [(3, 5, False), (2, 11, True), (1, 64, False)])
def test_source_arithmetic_survives_fusion_and_allocation(tmp_path, execute, m, n, separate):
    model, operation = source_program(m=m, n=n)
    kwargs = case(tmp_path, m=m, n=n, separate=separate)
    data = np.arange(m * n, dtype=np.int32) * 13 - 100
    bias = np.full(m * n, (1 << 31) - 2, dtype=np.int32)
    wrapped = (data.astype(np.int64) + bias.astype(np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)
    expected = np.clip(wrapped, -128, 127).astype("i1").tobytes()
    results = []
    for enabled in (False, True):
        result = compile_source_pointwise(
            model, operation, budgets=ScheduleBudgets(max_fusions=8 if enabled else 0), **kwargs
        )
        output, counters = execute(result.images(), {"x": data.tobytes(), "b0": bias.tobytes()}, separate=separate)
        assert output[: m * n] == expected and output[m * n :] == bytes([193] * (1032 - m * n))
        assert counters == tuple(
            result.conformance[key] for key in ("actual_commands", "requested_read_bytes", "requested_write_bytes")
        )
        assert result.source_passes[-1].name == "source-conformance"
        results.append((result, counters))
    assert results[0][1][0] == 2 and results[1][1][0] == 1
    assert results[0][1][1] - results[1][1][1] == 4 * m * n
    assert results[0][1][2] - results[1][1][2] == 4 * m * n
    wrong, _ = execute(results[1][0].images(), {"x": data.tobytes(), "b0": bias.tobytes()}, wrong=True)
    assert wrong[: m * n] != expected


@pytest.mark.parametrize(
    "before,after",
    [
        ("%sum=arith.addi %a,%b:i32", "%sum=arith.subi %a,%b:i32"),
        ("%sum=arith.addi %a,%b:i32", "%sum=arith.addi %b,%a:i32"),
        ("%low=arith.constant -128:i32", "%low=arith.constant -127:i32"),
        ("%lower=arith.maxsi", "%lower=arith.maxui"),
    ],
)
def test_changed_source_cannot_reuse_provider_arithmetic(tmp_path, before, after):
    module, operation = source_program(source_text().replace(before, after))
    with pytest.raises(ValueError, match="differs from the complete typed source"):
        compile_source_pointwise(module, operation, **case(tmp_path))


@pytest.mark.parametrize(
    "before,after,message",
    [
        ("%sum=arith.addi %a,%b:i32", "%sum=arith.addi %a,%b overflow<nsw>:i32", "unmodeled|non-total"),
        ('"linalg.yield"(%result)', '"linalg.yield"(%initial)', "initializer"),
        ("#linalg.iterator_type<parallel>", "#linalg.iterator_type<reduction>", "parallel"),
        ("affine_map<(d0,d1)->(d0,d1)>", "affine_map<(d0,d1)->(d1,d0)>", "identity"),
    ],
)
def test_unknown_source_stays_unoptimized(before, after, message):
    _, operation = source_program(source_text(3, 3).replace(before, after, 1))
    with pytest.raises(UnsupportedElementwise, match=message):
        read_pointwise_source(operation, max_elements=1024, max_operations=128, max_ir_bytes=65536)


def test_source_and_actual_bytes_cannot_be_reused_after_mutation(tmp_path):
    module, operation = source_program()
    kwargs = case(tmp_path)
    result = compile_source_pointwise(module, operation, **kwargs)
    changed = replace(result.compiled.program, segments=(bytes(len(result.images()[0])),))
    with pytest.raises(ValueError, match="changed; replan"):
        replace(result, compiled=replace(result.compiled, program=changed)).images()
    constant = next(op for op in operation.walk() if op.name == "arith.constant")
    constant.properties["value"] = builtin.IntegerAttr(-127, builtin.i32)
    with pytest.raises(ValueError, match="changed; replan"):
        result.images()


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda b: replace(b, external_buffers_disjoint=False), "ownership"),
        (lambda b: replace(b, byte_orders=("big", "little", "little")), "byte layout"),
        (lambda b: replace(b, output=AddressWindow("x", 0, 15)), "aliases"),
        (lambda b: replace(b, inputs=(AddressWindow("x", 1, 60), b.inputs[1])), "undeclared external"),
    ],
)
def test_alias_and_layout_premises_are_not_inferred(tmp_path, change, message):
    module, operation = source_program()
    kwargs = case(tmp_path)
    kwargs["bindings"] = change(kwargs["bindings"])
    with pytest.raises(ValueError, match=message):
        compile_source_pointwise(module, operation, **kwargs)


def test_invalid_order_is_rejected_before_emission(tmp_path):
    module, operation = source_program()
    kwargs = case(tmp_path)
    called = []
    kwargs["encoding"] = replace(kwargs["encoding"], encode=lambda kernel: called.append(kernel))
    with pytest.raises(ValueError, match="prerequisites|ordering"):
        compile_source_pointwise(
            module, operation, pass_order=("schedule-lowering", "source-admission", "source-conformance"), **kwargs
        )
    assert not called


def test_final_source_checker_reconstructs_bytes_instead_of_trusting_the_schedule(tmp_path, monkeypatch):
    from merlin.compile.scheduling import pointwise_source

    module, operation = source_program()
    kwargs = case(tmp_path)
    valid = compile_source_pointwise(module, operation, **kwargs)
    # Simulate a later emitter changing the bias to the other legal input.
    bad_call = command("fused", m=3, n=5, src=3, dst=9, bias=3)
    image = kwargs["encoding"].encode(replace(valid.compiled.kernel, body=(bad_call,)))
    false_result = replace(valid.compiled, program=replace(valid.compiled.program, segments=(image,)))
    monkeypatch.setattr(pointwise_source, "compile_schedule", lambda *args, **kwargs: false_result)
    with pytest.raises(ValueError, match="differs from the complete typed source"):
        compile_source_pointwise(module, operation, **kwargs)


@pytest.mark.parametrize("separate", [False, True])
def test_private_views_follow_physical_backings(tmp_path, separate):
    module, operation = source_program()
    kwargs = case(tmp_path, separate=separate)
    original = kwargs["kernel"]
    kwargs["kernel"] = replace(
        original, body=(original.body[0], command("finish", m=3, n=5, src=2, dst=9, source=3072, bias=6))
    )
    if separate:
        with pytest.raises(ValueError, match="undefined|overwritten"):
            compile_source_pointwise(module, operation, **kwargs)
    else:
        result = compile_source_pointwise(module, operation, **kwargs)
        assert result.conformance["actual_commands"] == 1


def test_exact_same_window_input_aliases_are_explicit(tmp_path):
    module, operation = source_program()
    kwargs = case(tmp_path)
    b = kwargs["bindings"]
    kwargs["bindings"] = replace(b, inputs=(b.inputs[0], b.inputs[0]))
    kernel = kwargs["kernel"]
    kwargs["kernel"] = replace(
        kernel, body=(kernel.body[0], command("finish", m=3, n=5, src=1, dst=9, source=3072, bias=3))
    )
    result = compile_source_pointwise(module, operation, **kwargs)
    assert result.conformance["actual_commands"] == 1


@pytest.mark.parametrize("bound,commands", [(14, 2), (15, 1)])
def test_rtl_composition_boundary_keeps_legal_unfused_route(tmp_path, bound, commands):
    module, operation = source_program()
    kwargs = case(tmp_path, mutate=lambda facts, software, fusion: facts.update(max_elements=bound))
    result = compile_source_pointwise(module, operation, **kwargs)
    assert result.conformance["actual_commands"] == commands


def test_unknown_execution_qualification_keeps_source_available(tmp_path):
    module, operation = source_program()
    before = str(module)
    kwargs = case(tmp_path)
    kwargs["fusion_qualification_identity"] = None
    with pytest.raises(UnsupportedElementwise, match="unmodeled arithmetic"):
        compile_source_pointwise(module, operation, **kwargs)
    assert str(module) == before


@pytest.mark.parametrize("fact", [True, False])
def test_selected_inputs_cannot_change_after_source_compilation(tmp_path, fact):
    module, operation = source_program()
    kwargs = case(tmp_path)
    result = compile_source_pointwise(module, operation, **kwargs)
    path = kwargs["selected_contracts"].facts_path if fact else kwargs["selected_contracts"].software_path
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        result.images()


@pytest.mark.parametrize(
    "elements,operations,ir_bytes,message",
    [
        (44, 128, 65536, "logical element budget"),
        (45, 5, 65536, "operation budget"),
        (45, 128, 100, "source-byte budget"),
    ],
)
def test_source_admission_respects_compile_budgets(elements, operations, ir_bytes, message):
    _, operation = source_program()
    with pytest.raises(UnsupportedElementwise, match=message):
        read_pointwise_source(operation, max_elements=elements, max_operations=operations, max_ir_bytes=ir_bytes)


def test_overwrite_through_a_private_view_cannot_preserve_old_arithmetic(tmp_path):
    module, operation = source_program()
    kwargs = case(tmp_path)
    kernel = kwargs["kernel"]
    overwritten = command("readout", m=3, n=5, src=6, dst=2, destination=3072)
    kwargs["kernel"] = replace(kernel, body=(kernel.body[0], overwritten, kernel.body[1]))
    with pytest.raises(ValueError, match="differs from the complete typed source"):
        compile_source_pointwise(module, operation, **kwargs)


def test_shifted_input_alias_is_not_an_exact_same_window_alias(tmp_path):
    module, operation = source_program()
    kwargs = case(tmp_path)
    bindings = kwargs["bindings"]
    kwargs["bindings"] = replace(bindings, inputs=(bindings.inputs[0], AddressWindow("x", 4, 60)))
    with pytest.raises(UnsupportedElementwise, match="shifted"):
        compile_source_pointwise(module, operation, **kwargs)


@pytest.mark.parametrize("capacity,enabled", [(60, False), (59, False), (59, True)])
def test_capacity_boundary_is_replanned_after_fusion(tmp_path, capacity, enabled):
    from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable

    module, operation = source_program()
    kwargs = case(tmp_path, mutate=lambda facts, software, fusion: facts.update(local_capacity=capacity))
    budgets = ScheduleBudgets(max_fusions=8 if enabled else 0)
    if capacity < 60 and not enabled:
        with pytest.raises(StorageOptimizationUnavailable, match="no legal temporary allocation"):
            compile_source_pointwise(module, operation, budgets=budgets, **kwargs)
    else:
        result = compile_source_pointwise(module, operation, budgets=budgets, **kwargs)
        assert result.conformance["actual_commands"] == (1 if enabled else 2)


def test_changed_shared_library_invalidates_the_source_result(tmp_path, monkeypatch):
    from merlin.compile.scheduling import pointwise_source

    module, operation = source_program()
    result = compile_source_pointwise(module, operation, **case(tmp_path))
    library = dict(result.library_sources)
    library["merlin.compile.scheduling.pointwise_source"] = "0" * 64
    monkeypatch.setattr(pointwise_source, "shared_library_sources", lambda: library)
    with pytest.raises(ValueError, match="changed; replan"):
        result.images()


_EXECUTOR16 = (
    _EXECUTOR.replace("dst+(op==1?4:1)*elements", "dst+(op==1?4:2)*elements")
    .replace("sum < -128 ? -128 : sum > 127 ? 127 : sum", "sum < -32768 ? -32768 : sum > 32767 ? 32767 : sum")
    .replace(
        "spaces[w][dst+j]=(uint8_t)bounded;writes++;",
        """
 for(int byte=0;byte<2;byte++){
  spaces[w][dst+2*j+byte]=(uint8_t)((uint32_t)bounded>>(8*byte));writes++;
 }
""",
    )
)


@pytest.mark.parametrize("execute", [_EXECUTOR16], indirect=True, ids=["16-bit-output-machine"])
def test_source_fusion_supports_a_different_physical_element_width(tmp_path, execute):
    m, n = 2, 11
    source = source_text(m, n).replace("i8", "i16").replace("-128", "-32768").replace("127", "32767")
    module, operation = source_program(source)
    scalar = EPILOGUE.replace(b"i8", b"i16").replace(b"-128", b"-32768").replace(b"127", b"32767")

    def declaration(facts, software, fusion):
        facts["byte_bits"] = 16
        for name in ("finish", "fused"):
            fusion["elementwise"][name]["scalar_ir"] = scalar.decode()

    kwargs = case(tmp_path, m=m, n=n, separate=True, mutate=declaration)
    original = kwargs["adapter"]

    def project(call):
        effect = original.project(call)
        if call.instr not in ("finish", "fused"):
            return effect
        return replace(
            effect,
            accesses=(*effect.accesses[:-1], replace(effect.accesses[-1], size=2 * m * n)),
            elementwise=replace(effect.elementwise, function_ir=scalar),
        )

    kwargs["adapter"] = replace(original, project=project)
    codec = kwargs["encoding"]
    kwargs["encoding"] = replace(
        codec,
        program_decoder=replace(
            codec.program_decoder,
            decode=lambda image: tuple(
                DecodedInstruction(
                    i * WORD.size, WORD.size, project(call).accesses, completion="synchronous", relocatable=True
                )
                for i, call in enumerate(codec.decode_calls(image))
            ),
        ),
    )
    kernel = kwargs["kernel"]
    kwargs["kernel"] = replace(kernel, args=(*kernel.args[:-1], replace(kernel.args[-1], dtype="i16")))
    kwargs["bindings"] = replace(kwargs["bindings"], output=AddressWindow("y", 0, 2 * m * n))
    values = (np.arange(m * n, dtype=np.int32) * 13001 - 100000).reshape(m, n)
    bias = np.full((m, n), (1 << 31) - 2, "<i4")
    wide = values.astype(np.int64) + bias.astype(np.int64)
    expected = np.clip((wide + (1 << 31)) % (1 << 32) - (1 << 31), -32768, 32767).astype("<i2").tobytes()
    for enabled in (False, True):
        result = compile_source_pointwise(
            module, operation, budgets=ScheduleBudgets(max_fusions=8 if enabled else 0), **kwargs
        )
        actual, counts = execute(result.images(), {"x": values.tobytes(), "b0": bias.tobytes()}, separate=True)
        assert actual[: 2 * m * n] == expected and actual[2 * m * n :] == bytes([193] * (1032 - 2 * m * n))
        assert counts[0] == (1 if enabled else 2)
