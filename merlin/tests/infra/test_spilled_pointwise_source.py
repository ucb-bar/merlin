"""Source-bound device repair, complete tails and independent native execution."""

from dataclasses import replace

import pytest
from test_pointwise_source import (
    execute,  # noqa: F401
    source_program,
)
from test_private_copy_elimination import SPACES, command
from test_selected_spill_contracts import emit_copy, inputs
from test_selected_storage_contracts import native  # noqa: F401

from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise
from merlin.compile.scheduling.pointwise_source import PointwiseBindings, compile_source_pointwise
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable, TemporaryObject
from merlin.sched.ir import Kernel, TensorArg

INCREMENT = b"""builtin.module {func.func @step(%x:i8)->i8 {
 %one=arith.constant 1:i8
 %result=arith.addi %x,%one:i8
 func.return %result:i8
}}"""


def source_text(m=1, n=47):
    shape = f"{m}x{n}"
    return f"""builtin.module {{
 func.func @forward(%x:tensor<{shape}xi8>,%init:tensor<{shape}xi8>)->tensor<{shape}xi8> {{
 %out="linalg.generic"(%x,%init) <{{
 indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],
 iterator_types=[#linalg.iterator_type<parallel>,#linalg.iterator_type<parallel>],
 operandSegmentSizes=array<i32:1,1>}}> ({{
 ^bb0(%a:i8,%initial:i8):
 %one=arith.constant 1:i8
 %result=arith.addi %a,%one:i8
 "linalg.yield"(%result):(i8)->()
 }}) : (tensor<{shape}xi8>,tensor<{shape}xi8>)->tensor<{shape}xi8>
 func.return %out:tensor<{shape}xi8>
 }} }}"""


def case(tmp_path, *, m=1, n=47, width=16, register_slots=1, word_format="<BBBQQQ", manual=False, mutate=None):
    """All three chunks contribute; compare automatic repair to a legal manual one.

    The source schedule keeps A live while B and the tail execute. An earlier
    store of A could avoid pressure; this fixture tests checked repair, not an
    optimal scheduling claim. Both arms use exactly the same selected hardware.
    """
    count = m * n
    assert 2 * width < count <= 3 * width

    def declare(facts, software, storage):
        facts["byte_bits"] = 8
        software["scheduling"]["fusion"] = dict(
            schema="merlin.physical_fusion_contract.v1",
            elementwise={"increment": dict(
                scalar_ir=INCREMENT.decode(), elements=[{"argument": "size"}],
                element_bits=[{"fact": ["facts", "byte_bits"]}] * 2,
                stages=["increment"], byte_orders=["little"] * 2,
                readout=None, execution="contiguous_read_then_write",
            )}, readouts={},
        )
        if mutate:
            mutate(facts, software, storage)

    selected, original, _, adapter, codec, limits, _, _ = inputs(
        tmp_path, width=width, register_slots=register_slots, word_format=word_format, mutate=declare
    )
    a, b, c = 32, 96, 160
    tail = count - 2 * width
    calls = (
        command("copy", 0, a, width, 2, 0), command("increment", a, a, width),
        command("copy", width, b, width, 2, 0), command("increment", b, b, width),
        command("copy", b, width, width, 0, 3), command("copy", 2 * width, c, tail, 2, 0),
        command("increment", c, c, tail), command("copy", c, 2 * width, tail, 0, 3),
        command("copy", a, 0, width, 0, 3),
    )
    objects = tuple(TemporaryObject(name, SPACES[0], address, size) for name, address, size in (
        ("a", a, width), ("b", b, width), ("tail", c, tail),
    ))
    if manual:
        reload = 224
        calls = (*calls[:2], command("copy", a, 0, width, 0, 1), *calls[2:-1],
                 command("copy", 0, reload, width, 1, 0), command("copy", reload, 0, width, 0, 3))
        objects += (TemporaryObject("saved_a", SPACES[1], 0, width),
                    TemporaryObject("reloaded_a", SPACES[0], reload, width))
    kernel = Kernel(original.name, (TensorArg("input", (m, n), "i8", "read"),
                                    TensorArg("output", (m, n), "i8", "write")), calls)
    return dict(kernel=kernel, objects=objects, adapter=adapter, encoding=codec, limits=limits,
                selected_contracts=selected, private_ownership=True,
                bindings=PointwiseBindings((AddressWindow("input", 0, count),),
                                           AddressWindow("output", 0, count), ("little", "little"), True),
                fusion_qualification_identity="neutral-native-byte-increment-v1",
                spill_emitters={"vector_values": emit_copy},
                spill_qualification_identity="neutral-independent-device-spills-v1")


@pytest.mark.parametrize("m,n,width,word_format", [(1, 8, 3, "<BBBQQQ"), (1, 47, 16, ">BBBIII")])
def test_source_spill_repair_matches_legal_baseline_and_complete_native_state(
    tmp_path, native, m, n, width, word_format  # noqa: F811
):
    module, operation = source_program(source_text(m, n))
    results = []
    for manual in (True, False):
        path = tmp_path / str(manual)
        path.mkdir()
        kwargs = case(path, m=m, n=n, width=width, word_format=word_format, manual=manual)
        results.append(compile_source_pointwise(
            module, operation, budgets=ScheduleBudgets(max_fusions=0, eliminate_copies=False), **kwargs
        ))
    baseline, repaired = results
    assert baseline.images() == repaired.images()
    assert baseline.compiled.report["repair_rounds"] == 0
    assert repaired.compiled.report["repair_rounds"] == 1
    assert repaired.compiled.report["storage"]["spills"]["inserted_exact_copies"] == 2
    assert repaired.conformance["verified_fragments"] == 3
    length = 3 * width + 8
    data = bytearray((i * 71 + 253) % 256 for i in range(length))
    carried = bytearray([91] * length)
    for _ in range(3):
        expected = bytearray(carried)
        expected[:m * n] = bytes((value + 1) % 256 for value in data[:m * n])
        for result in results:
            current, output = bytearray(data), bytearray(carried)
            first, spill = bytearray([167] * 3 * width), bytearray([199] * 3 * width)
            counts = native(b"".join(result.images()), first, spill, current, output,
                            width=8 if word_format.endswith("Q") else 4,
                            big=word_format[0] == ">", separate=True)
            assert output == expected and current == data
            assert first[width:] == bytes([167] * 2 * width)
            assert spill[width:] == bytes([199] * 2 * width)
            assert counts[:3] == tuple(result.conformance[key] for key in (
                "actual_commands", "requested_read_bytes", "requested_write_bytes"))
            assert counts[:3] == (11, 11 * width - 3 * (3 * width - m * n),
                                  11 * width - 3 * (3 * width - m * n))
        data, carried = bytearray(expected), expected
    # Declared effects cannot establish the native implementation's correctness.
    faulty = bytearray([91] * length)
    native(b"".join(repaired.images()), bytearray(3 * width), bytearray(3 * width), data, faulty,
           width=8 if word_format.endswith("Q") else 4, big=word_format[0] == ">", separate=True,
           faulty_spill=1)
    assert faulty[:m * n] != bytes((value + 1) % 256 for value in data[:m * n])


@pytest.mark.parametrize("fault", ["source_offset", "destination_offset", "missing_tail", "tail_arithmetic",
                                  "overwrite", "shifted_arithmetic", "shifted_copy", "computed_prefix"])
def test_wrong_fragment_dataflow_cannot_escape_source_check(tmp_path, fault):
    module, operation = source_program(source_text())
    kwargs = case(tmp_path, register_slots=2)
    body = list(kwargs["kernel"].body)
    if fault == "source_offset":
        body[2] = command("copy", 0, 96, 16, 2, 0)
    elif fault == "destination_offset":
        body[4] = command("copy", 96, 0, 16, 0, 3)
    elif fault == "missing_tail":
        del body[7]
    elif fault == "tail_arithmetic":
        del body[6]
    elif fault == "overwrite":
        body.insert(-1, command("copy", 16, 32, 16, 2, 0))
    elif fault == "shifted_arithmetic":
        body[1] = command("increment", 32, 33, 16)
    elif fault == "shifted_copy":
        body.insert(2, command("copy", 32, 33, 16))
    else:
        body[4] = command("copy", 96, 16, 15, 0, 3)
    kwargs["kernel"] = replace(kwargs["kernel"], body=tuple(body))
    with pytest.raises(ValueError, match="source|undefined|fragment|shifted"):
        compile_source_pointwise(module, operation, **kwargs)


def test_unavailable_spills_leave_feasible_fragmented_program_legal(tmp_path):
    module, operation = source_program(source_text())
    for slots in (2, 1):
        directory = tmp_path / str(slots)
        directory.mkdir()
        kwargs = case(directory, register_slots=slots)
        kwargs["spill_qualification_identity"] = None
        if slots == 2:
            result = compile_source_pointwise(module, operation, **kwargs)
            assert result.conformance["actual_commands"] == 9
        else:
            with pytest.raises(StorageOptimizationUnavailable, match="missing_execution_qualification"):
                compile_source_pointwise(module, operation, **kwargs)


def test_issue_only_transfer_is_not_synchronous_source_initialization(tmp_path):
    module, operation = source_program(source_text())
    kwargs = case(tmp_path, register_slots=2)
    body = kwargs["kernel"].body
    kwargs["kernel"] = replace(kwargs["kernel"], body=(replace(body[0], instr="async_copy"), *body[1:]))
    with pytest.raises(UnsupportedElementwise, match="synchronous completion"):
        compile_source_pointwise(module, operation, **kwargs)


@pytest.mark.parametrize("count", [33, 48])
def test_one_element_tail_and_exact_capacity_boundary(tmp_path, count):
    module, operation = source_program(source_text(1, count))
    result = compile_source_pointwise(module, operation, **case(tmp_path, n=count))
    assert result.conformance["elements"] == count
    assert result.conformance["actual_commands"] == 11
    assert result.compiled.report["repair_rounds"] == 1


def test_exact_byte_copies_can_reassemble_one_computed_fragment(tmp_path):
    module, operation = source_program(source_text())
    kwargs = case(tmp_path, register_slots=2)
    body = kwargs["kernel"].body
    kwargs["kernel"] = replace(kwargs["kernel"], body=(
        *body[:4], command("copy", 96, 16, 7, 0, 3), command("copy", 103, 23, 9, 0, 3), *body[5:],
    ))
    result = compile_source_pointwise(
        module, operation, budgets=ScheduleBudgets(max_fusions=0, eliminate_copies=False), **kwargs
    )
    assert result.conformance["actual_commands"] == 10
    assert result.conformance["verified_fragments"] == 3


@pytest.mark.parametrize("start,size", [(96, 15), (97, 15)])
def test_arithmetic_cannot_interpret_prefix_or_suffix_of_computed_fragment(tmp_path, start, size):
    module, operation = source_program(source_text())
    kwargs = case(tmp_path, register_slots=2)
    body = list(kwargs["kernel"].body)
    body.insert(4, command("increment", start, start, size))
    kwargs["kernel"] = replace(kwargs["kernel"], body=tuple(body))
    with pytest.raises(ValueError, match="complete computed fragments"):
        compile_source_pointwise(module, operation, **kwargs)


@pytest.mark.parametrize("fault", ["drop_store", "short_store", "wrong_reload", "tail_arithmetic"])
def test_final_spill_bytes_are_reopened_against_original_source(tmp_path, monkeypatch, fault):
    from merlin.compile.scheduling import pointwise_source

    module, operation = source_program(source_text())
    kwargs = case(tmp_path)
    good = compile_source_pointwise(module, operation, **kwargs)
    body = list(kwargs["encoding"].decode_calls(b"".join(good.images())))
    if fault == "drop_store":
        del body[2]
    elif fault == "short_store":
        body[2] = command("copy", 0, 0, 15, 0, 1)
    elif fault == "wrong_reload":
        body[-2] = command("copy", 16, 0, 16, 1, 0)
    else:
        body[7] = command("copy", 0, 0, 15)
    image = kwargs["encoding"].encode(replace(good.compiled.kernel, body=tuple(body)))
    false = replace(good.compiled, program=replace(good.compiled.program, segments=(image,)))
    monkeypatch.setattr(pointwise_source, "compile_schedule", lambda *args, **kw: false)
    with pytest.raises(ValueError, match="undefined|source|fragment"):
        compile_source_pointwise(module, operation, **kwargs)


@pytest.mark.parametrize("changed", ["facts", "software", "source", "bytes", "library"])
def test_fragmented_result_rejects_stale_inputs(tmp_path, monkeypatch, changed):
    from merlin.compile.scheduling import pointwise_source

    module, operation = source_program(source_text())
    kwargs = case(tmp_path)
    result = compile_source_pointwise(module, operation, **kwargs)
    if changed in ("facts", "software"):
        selected = kwargs["selected_contracts"]
        path = selected.facts_path if changed == "facts" else selected.software_path
        path.write_bytes(path.read_bytes() + b"\n")
    elif changed == "source":
        other, _ = source_program(source_text().replace("constant 1:i8", "constant 2:i8"))
        result = replace(result, source=other)
    elif changed == "bytes":
        program = replace(result.compiled.program, segments=(bytes(len(result.images()[0])),))
        result = replace(result, compiled=replace(result.compiled, program=program))
    else:
        monkeypatch.setattr(pointwise_source, "shared_library_sources", lambda: {"modified": "0" * 64})
    with pytest.raises(ValueError, match="changed|replan|reselect"):
        result.images()


@pytest.mark.parametrize("misaligned", [False, True])
def test_wide_source_fragments_preserve_element_coordinates(tmp_path, execute, misaligned):  # noqa: F811
    from test_pointwise_source import case as wide_case
    from test_readout_model_fusion import command as vector_command

    # Reuse the independently executed i32 identity primitive, with an i32
    # destination rather than the usual narrowing epilogue. This exercises
    # byte-to-element coordinates, not just byte-sized source fragments.
    text = source_text(1, 15).replace("i8", "i32")
    text = text.replace("%one=arith.constant 1:i32", "").replace("%result=arith.addi %a,%one:i32", "")
    text = text.replace('"linalg.yield"(%result)', '"linalg.yield"(%a)')
    module, operation = source_program(text)
    kwargs = wide_case(tmp_path, m=1, n=15, separate=True)
    kwargs["bindings"] = PointwiseBindings((AddressWindow("x", 0, 60),), AddressWindow("y", 0, 60),
                                           ("little", "little"), True)
    body, objects = [], []
    for index, (offset, elements) in enumerate(((0, 8), (32, 7))):
        private = 3072 + 64 * index
        objects.append(TemporaryObject(f"stage_{index}", "stage", private, 4 * elements, 4))
        body += [vector_command("readout", m=1, n=elements, src=3, dst=1,
                                source=offset, destination=private),
                 vector_command("readout", m=1, n=elements, src=1, dst=9,
                                source=private, destination=offset)]
    if misaligned:
        body[0] = vector_command("readout", m=1, n=8, src=3, dst=1, source=1, destination=3072)
    kwargs["kernel"] = Kernel("wide_fragments", (TensorArg("x", (1, 15), "i32", "read"),
                                               TensorArg("y", (1, 15), "i32", "write")), tuple(body))
    kwargs["objects"] = tuple(objects)
    if misaligned:
        with pytest.raises(ValueError, match="alignment"):
            compile_source_pointwise(module, operation, **kwargs)
    else:
        result = compile_source_pointwise(module, operation, **kwargs)
        values = [-(1 << 31), (1 << 31) - 1, -1, 0, *range(11)]
        raw = b"".join(value.to_bytes(4, "little", signed=True) for value in values)
        output, counts = execute(result.images(), {"x": raw}, separate=True)
        assert output[:60] == raw and output[60:] == bytes([193] * (1032 - 60))
        assert counts == (4, 120, 120)
        assert result.conformance["verified_fragments"] == 2
