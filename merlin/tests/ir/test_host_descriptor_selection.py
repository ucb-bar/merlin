"""Diagnostic wiring controls for the ordinary selected descriptor caller.

Build/layout/process observations are explicit substitutes in this file. They
test refusal and byte transport, never native ABI, compiler or runtime authority.
Actual ordinary source/object/link execution belongs to separate native tests.
"""

import ctypes
import hashlib
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json
from merlin.llvmlower import host_descriptor_compile as C
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.descriptor_contract import DescriptorLimits, OriginalDescriptorSource, TensorStorageDeclaration
from merlin.llvmlower.descriptor_layout import DescriptorLayoutObservation
from merlin.llvmlower.descriptor_wrapper import parse
from merlin.llvmlower.host_descriptor_call import HostDescriptorBuffer, HostDescriptorTransport
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor

LIMITS = DescriptorLimits(200000, 200000, 64, 128, 8, 10000, 200000)
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def selection(tmp_path, shape=(3, 5), dtype="i32", outputs=1):
    typ = "tensor<" + "".join(str(size) + "x" for size in shape) + dtype + ">"
    returned = ", ".join(["%x"] * outputs)
    types = ", ".join([typ] * outputs)
    source = tmp_path / "original.mlir"
    source.write_text(f"module {{func.func @forward(%x: {typ}) -> ({types}) {{func.return {returned} : {types}}}}}")
    tensors = (CompileOnlyTensor("x", shape, dtype),) + tuple(
        CompileOnlyTensor(f"y{index}", shape, dtype) for index in range(outputs)
    )
    strides, count = [], 1
    for extent in reversed(shape):
        strides.append(count)
        count *= extent
    storage = tuple(
        TensorStorageDeclaration(
            tensor.name, "input" if ordinal == 0 else "output", 0, tuple(reversed(strides)), count, 8
        )
        for ordinal, tensor in enumerate(tensors)
    )
    original = OriginalDescriptorSource(
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "forward",
        "_mlir_ciface_forward",
        CompileOnlySourceAbi(tensors[:1], tensors[1:]),
        "mlir_ranked_memref_ciface_v1",
        storage,
        "disjoint",
    )
    return C.HostDescriptorSelection(original, LIMITS, tuple(ENVIRONMENT.items()), 30)


def module(selected):
    return parse(selected.source.path.read_bytes(), selected.limits, emitted=False)


def test_original_complete_body_returns_and_storage_are_checked_before_lowering(tmp_path):
    selected = selection(tmp_path, outputs=2)
    selected.require_module(module(selected))
    changed = selected.source.path.read_text().replace(
        "func.return %x, %x", "%e = tensor.empty() : tensor<3x5xi32>\nfunc.return %e, %x"
    )
    with pytest.raises(ValueError, match="differs from its original source"):
        selected.require_module(parse(changed.encode(), LIMITS, emitted=False))
    for update in (
        {"slot_aliasing": "unproved"},
        {"storage": selected.source.storage[:1]},
        {"original_abi": CompileOnlySourceAbi(selected.source.original_abi.inputs, ())},
    ):
        with pytest.raises(ValueError):
            replace(selected, source=replace(selected.source, **update)).verify()
    selected.source.path.write_bytes(selected.source.path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="source bytes changed"):
        selected.verify()


@pytest.mark.parametrize(
    "environment", [(), (("PATH", "/usr/bin"), ("PATH", "/bin")), (("A", True),), {"PATH": "/bin"}]
)
def test_no_environment_default_duplicate_or_scalar_alias(tmp_path, environment):
    with pytest.raises(ValueError, match="environment mapping"):
        replace(selection(tmp_path), environment=environment).verify()


def test_only_actual_calling_marker_may_differ_in_prepared_source(tmp_path):
    selected = selection(tmp_path)
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(
        selected.source.path.read_text().replace(" {func.return", " attributes {llvm.emit_c_interface} {func.return")
    )
    pin = {"path": str(prepared), "sha256": hashlib.sha256(prepared.read_bytes()).hexdigest()}
    assert C._prepared(selected, {"source": pin}).path == prepared
    prepared.write_text(
        prepared.read_text().replace("func.return %x", "%e = tensor.empty() : tensor<3x5xi32>\nfunc.return %e")
    )
    pin["sha256"] = hashlib.sha256(prepared.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="preprocessing changed"):
        C._prepared(selected, {"source": pin})


def build_records(tmp_path, selected):
    owner = tmp_path / "diagnostic-build"
    owner.mkdir()
    llvm, runtime, obj, runtime_obj, image = (
        owner / name for name in ("model.ll", "runtime.c", "model.o", "runtime.o", "model.so")
    )
    for ordinal, path in enumerate((llvm, runtime, obj, runtime_obj, image)):
        path.write_bytes(f"diagnostic substitute {ordinal}".encode())
    for stage, inputs, outputs in (
        ("object", (llvm,), (obj,)),
        ("runtime_object", (runtime,), (runtime_obj,)),
        ("link", (obj, runtime_obj), (image,)),
    ):
        # No native tool runs: these are expressly substituted observations.
        with I.observe(
            owner, stage=stage, argv=(sys.executable,), env=ENVIRONMENT, inputs=inputs, outputs=outputs
        ) as record:
            record.complete(subprocess.CompletedProcess((sys.executable,), 0, b"", b""))
    return owner, image, {"translated_llvm_ir": C._pin(llvm, selected.limits)}, obj


def test_join_requires_same_actual_object_runtime_link_environment_and_image(tmp_path):
    selected = selection(tmp_path)
    owner, image, product, obj = build_records(tmp_path, selected)
    record, rows = C.join_host_image(selection=selected, build_root=owner, product=product, image=image)
    assert len(rows) == 3 and I.verify(record)["outputs"] == [C._pin(obj, selected.limits)]
    other = owner / "other.so"
    other.write_bytes(image.read_bytes())
    with pytest.raises(ValueError, match="disconnected"):
        C.join_host_image(selection=selected, build_root=owner, product=product, image=other)
    with pytest.raises(ValueError, match="environment binding"):
        C.join_host_image(
            selection=replace(selected, environment=(("PATH", "/bin"),)), build_root=owner, product=product, image=image
        )
    saved = obj.read_bytes()
    obj.write_bytes(saved + b"changed")
    with pytest.raises(ValueError):
        C.join_host_image(selection=selected, build_root=owner, product=product, image=image)
    obj.write_bytes(saved)
    rows[-1].unlink()
    with pytest.raises(ValueError, match="build boundaries"):
        C.join_host_image(selection=selected, build_root=owner, product=product, image=image)


@pytest.mark.parametrize("defect", (None, "changed_raw", "changed_final", "missing_raw", "stale_raw_pin"))
def test_complete_raw_translation_copy_joins_actual_object_input_without_path_or_pin_shape_assumption(tmp_path, defect):
    selected = selection(tmp_path)
    owner, image, product, _ = build_records(tmp_path, selected)
    final = owner / "model.ll"
    raw = owner / "separately-retained-translation.ll"
    raw.write_bytes(final.read_bytes())
    product["translated_llvm_ir"] = {**C._pin(raw, selected.limits), "bytes": len(raw.read_bytes())}
    if defect == "changed_raw":
        raw.write_bytes(raw.read_bytes() + b"changed")
    elif defect == "changed_final":
        final.write_bytes(final.read_bytes() + b"changed")
    elif defect == "missing_raw":
        raw.unlink()
    elif defect == "stale_raw_pin":
        product["translated_llvm_ir"]["sha256"] = "0" * 64
    if defect is None:
        _, records = C.join_host_image(selection=selected, build_root=owner, product=product, image=image)
        assert len(records) == 3
    else:
        with pytest.raises((ValueError, OSError)):
            C.join_host_image(selection=selected, build_root=owner, product=product, image=image)


def test_compile_host_uses_selected_original_and_own_result_before_loading(tmp_path, monkeypatch):
    from merlin.llvmlower import kernel_backend, lower

    selected = selection(tmp_path)
    calls, result, transport = [], SimpleNamespace(host_so=tmp_path / "build/model.so"), object()

    def lowering(source, root, **kwargs):
        assert "forward" in source and root == tmp_path / "build"
        assert kwargs == {"targets": ("host",), "retain_llvm_dialect": True}
        root.mkdir()
        calls.append("lower")
        return result

    def observe(**kwargs):
        assert kwargs == {"selection": selected, "result": result}
        calls.append("observe")
        return transport

    def load(path, **kwargs):
        assert path == str(result.host_so) and kwargs["descriptor_transport"] is transport
        assert kwargs["name"] == "forward" and kwargs["scalar_result_dtype"] is None
        calls.append("load")
        return "diagnostic model"

    monkeypatch.setattr(lower, "lower_model", lowering)
    monkeypatch.setattr(C, "observe_host_descriptor_transport", observe)
    monkeypatch.setattr(HostModel, "load", load)
    assert (
        kernel_backend.compile_host(module(selected), tmp_path / "build", descriptor_selection=selected)
        == "diagnostic model"
    )
    assert calls == ["lower", "observe", "load"]
    changed = module(selected)
    changed.body.block.first_op.sym_name = StringAttr("changed")
    with pytest.raises(ValueError):
        kernel_backend.compile_host(changed, tmp_path / "absent", descriptor_selection=selected)
    assert calls == ["lower", "observe", "load"]


@pytest.mark.parametrize("selected_route", [False, True])
def test_compile_host_preserves_complete_operation_attributes_before_preprocessing(
    tmp_path, monkeypatch, selected_route
):
    from merlin.llvmlower import kernel_backend, lower
    from merlin.llvmlower.passes_xdsl import preprocess_text
    from merlin.xdsl_dialects._common import text

    selected = selection(tmp_path)
    original = parse(
        b"""module {
          func.func @forward(%x: tensor<3x5xi32>) -> tensor<3x5xi32> {
            %e = tensor.empty() : tensor<3x5xi32>
            %zero = arith.constant 0 : i32
            %predicate = arith.cmpi eq, %zero, %zero : i32
            %chosen = arith.select %predicate, %x, %e : tensor<3x5xi32>
            func.return %chosen : tensor<3x5xi32>
          }
        }""",
        LIMITS,
        emitted=False,
    )
    for ordinal, op in enumerate(original.walk()):
        op.attributes["source.boundary"] = StringAttr(f"original-{ordinal}")
    selected.source.path.write_text(text(original, generic=True))
    selected = replace(
        selected, source=replace(selected.source, sha256=hashlib.sha256(selected.source.path.read_bytes()).hexdigest())
    )
    observed = []

    class StopAfterSourceCheck(Exception):
        pass

    def lowering(source, root, **kwargs):
        reparsed = parse(source.encode(), LIMITS, emitted=False)
        assert text(reparsed, generic=True) == text(original, generic=True)
        prepared_text, _ = preprocess_text(source)
        prepared = tmp_path / "prepared.mlir"
        prepared.write_text(prepared_text)
        C._prepared(
            selected,
            {"source": {"path": str(prepared), "sha256": hashlib.sha256(prepared.read_bytes()).hexdigest()}},
        )
        observed.append(tuple(op.attributes["source.boundary"].data for op in reparsed.walk()))
        raise StopAfterSourceCheck

    monkeypatch.setattr(lower, "lower_model", lowering)
    with pytest.raises(StopAfterSourceCheck):
        kernel_backend.compile_host(
            original, tmp_path / "build", descriptor_selection=selected if selected_route else None
        )
    assert observed == [tuple(f"original-{ordinal}" for ordinal, _ in enumerate(original.walk()))]


def diagnostic_transport(tmp_path, monkeypatch, shape=(3, 5), dtype="i32", index_bits=32, outputs=1):
    selected = selection(tmp_path, shape, dtype, outputs)
    pointer_bytes = ctypes.sizeof(ctypes.c_void_p)
    rows = []
    for tensor, storage in zip(
        (*selected.source.original_abi.inputs, *selected.source.original_abi.outputs),
        selected.source.storage,
        strict=True,
    ):
        fields = [
            {"role": role, "type": "ptr", "offset_bytes": ordinal * pointer_bytes, "storage_bytes": pointer_bytes}
            for ordinal, role in enumerate(("allocated", "aligned"))
        ]
        roles = (
            ["offset"]
            + [f"size_{axis}" for axis in range(len(shape))]
            + [f"stride_{axis}" for axis in range(len(shape))]
        )
        fields.extend(
            {
                "role": role,
                "type": f"i{index_bits}",
                "offset_bytes": pointer_bytes * 2 + ordinal * (index_bits // 8),
                "storage_bytes": index_bits // 8,
            }
            for ordinal, role in enumerate(roles)
        )
        rows.append(
            {
                "name": tensor.name,
                "shape": list(shape),
                "element_bytes": int(dtype[1:]) // 8,
                "index_bits": index_bits,
                "fields": fields,
                "descriptor_bytes": pointer_bytes * 2 + len(roles) * (index_bits // 8),
                "descriptor_alignment": pointer_bytes,
                "explicit_load_alignment": pointer_bytes,
            }
        )
    data = {"pointer_bytes": pointer_bytes, "byte_order": sys.byteorder, "slots": rows}
    wrapper = SimpleNamespace(source=selected.source, limits=selected.limits, record=lambda: {})
    layout = DescriptorLayoutObservation(wrapper, tmp_path / "layout", (), (), canonical_json(data))
    transport = HostDescriptorTransport(selected, layout, tmp_path, tmp_path / "model.so", (), ())
    # Only the unavailable actual layout/build chain is substituted. Real
    # packing and ordinary call-argument checks still execute below.
    monkeypatch.setattr(DescriptorLayoutObservation, "verify", lambda self: self.record())
    monkeypatch.setattr(HostDescriptorTransport, "verify", lambda self: self.layout.record()["observation"])
    arguments = tuple(
        HostDescriptorBuffer(tensor, declaration, 4096 * (ordinal + 1), 4096 * (ordinal + 1))
        for ordinal, (tensor, declaration) in enumerate(
            zip(
                (*selected.source.original_abi.inputs, *selected.source.original_abi.outputs),
                selected.source.storage,
                strict=True,
            )
        )
    )
    return transport, arguments


@pytest.mark.parametrize("shape,dtype", [((), "i64"), ((3, 5), "i32")])
def test_complete_ordered_original_fields_use_observed_i32_indices(tmp_path, monkeypatch, shape, dtype):
    transport, arguments = diagnostic_transport(tmp_path, monkeypatch, shape, dtype, outputs=2)
    addresses, carriers, payloads = transport.prepare(arguments)
    assert len(addresses) == len(carriers) == len(payloads) == 3
    for ordinal, (address, payload, row) in enumerate(
        zip(addresses, payloads, transport.verify()["slots"], strict=True)
    ):
        assert address.value % row["descriptor_alignment"] == 0
        assert ctypes.string_at(address.value, len(payload)) == payload
        offset = row["fields"][2]
        assert (
            offset["storage_bytes"] == 4
            and len(payload) == 2 * ctypes.sizeof(ctypes.c_void_p) + (1 + 2 * len(shape)) * 4
        )
        assert (
            int.from_bytes(payload[: ctypes.sizeof(ctypes.c_void_p)], sys.byteorder)
            == arguments[ordinal].allocated_address
        )


@pytest.mark.parametrize(
    "defect",
    ("missing", "reordered", "dtype", "shape", "stride", "capacity", "alignment", "overlap", "scalar", "address"),
)
def test_changed_partial_or_aliased_caller_objects_refuse_before_descriptor_allocation(tmp_path, monkeypatch, defect):
    transport, arguments = diagnostic_transport(tmp_path, monkeypatch, outputs=2)
    arguments = list(arguments)
    if defect == "missing":
        arguments.pop()
    elif defect == "reordered":
        arguments[1:] = reversed(arguments[1:])
    elif defect in {"dtype", "shape"}:
        arguments[1] = replace(
            arguments[1], tensor=replace(arguments[1].tensor, **{defect: "i64" if defect == "dtype" else (3, 4)})
        )
    elif defect in {"stride", "capacity"}:
        field = "element_strides" if defect == "stride" else "capacity_elements"
        arguments[1] = replace(
            arguments[1], storage=replace(arguments[1].storage, **{field: (4, 1) if defect == "stride" else 14})
        )
    elif defect == "alignment":
        arguments[1] = replace(arguments[1], aligned_address=arguments[1].aligned_address + 1)
    elif defect == "overlap":
        arguments[1] = replace(
            arguments[1], allocated_address=arguments[0].allocated_address, aligned_address=arguments[0].aligned_address
        )
    elif defect == "scalar":
        arguments[1] = 3
    else:
        arguments[1] = replace(arguments[1], allocated_address=True)
    monkeypatch.setattr(
        ctypes, "create_string_buffer", lambda *args: pytest.fail("refusal happened after descriptor allocation")
    )
    with pytest.raises(ValueError):
        transport.prepare(arguments)


def test_constructed_transport_and_scalar_trampoline_paths_cannot_invoke(tmp_path):
    selected = selection(tmp_path)
    constructed = HostDescriptorTransport(selected, None, tmp_path, tmp_path / "missing.so", (), ())
    with pytest.raises(ValueError, match="not observed"):
        constructed.verify()
    with pytest.raises(ValueError, match="exact private image"):
        HostModel.load(str(constructed.image), descriptor_transport=constructed)


@pytest.mark.parametrize("defect", ("image", "entry", "wrapper", "scalar", "trampoline"))
def test_unsupported_selected_image_entry_scalar_or_trampoline_refuses_before_loading(tmp_path, monkeypatch, defect):
    from merlin.llvmlower.abi import PrivateHostImagePolicy

    transport, _ = diagnostic_transport(tmp_path, monkeypatch)
    options = {"name": "forward", "scalar_result_dtype": None, "n_args": None}
    image = transport.image
    if defect == "image":
        image = tmp_path / "different.so"
    elif defect == "entry":
        options["name"] = "different"
    elif defect == "wrapper":
        transport = replace(
            transport,
            selection=replace(
                transport.selection, source=replace(transport.selection.source, c_interface_symbol="different")
            ),
        )
    elif defect == "scalar":
        options["scalar_result_dtype"] = "i32"
    else:
        options["n_args"] = 3
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("unsupported selection loaded an image"))
    with pytest.raises(ValueError):
        HostModel.load(
            str(image), image_policy=PrivateHostImagePolicy(tmp_path), descriptor_transport=transport, **options
        )


def selected_model(tmp_path, monkeypatch):
    transport, arguments = diagnostic_transport(tmp_path, monkeypatch)
    function = SimpleNamespace(restype=None, argtypes=[ctypes.c_void_p] * len(arguments))
    model = HostModel(None, function, descriptor_transport=transport)
    model._descriptor_transport, model._descriptor_function = transport, function
    return model, arguments


@pytest.mark.parametrize("when", ("before", "during"))
@pytest.mark.parametrize("defect", ("drop", "replace", "function", "restype", "argtypes", "scalar", "trampoline"))
def test_actual_selected_model_route_checks_selection_and_entry_before_and_after_call(
    tmp_path, monkeypatch, when, defect
):
    model, arguments = selected_model(tmp_path, monkeypatch)
    calls = []

    def change():
        if defect in {"drop", "replace"}:
            model.descriptor_transport = None if defect == "drop" else object()
        elif defect == "function":
            model.fn = object()
        elif defect in {"restype", "argtypes"}:
            setattr(model.fn, defect, ctypes.c_int64 if defect == "restype" else [])
        else:
            setattr(
                model,
                "scalar_result_dtype" if defect == "scalar" else "trampoline",
                "i32" if defect == "scalar" else object(),
            )

    def invoke(self, function, actual):
        assert self is model._descriptor_transport and function is model._descriptor_function and actual is arguments
        calls.append("diagnostic invocation")
        if when == "during":
            change()
        return []

    monkeypatch.setattr(HostDescriptorTransport, "invoke", invoke)
    if when == "before":
        change()
    with pytest.raises(ValueError, match="function/ABI changed"):
        model(arguments)
    assert len(calls) == (1 if when == "during" else 0)


def test_absent_selection_keeps_ordinary_host_scalar_call_and_lowering_kwargs(tmp_path, monkeypatch):
    from merlin.llvmlower import kernel_backend, lower
    from merlin.llvmlower.abi import ScalarArg

    assert HostModel(None, lambda value: value.value + 3)([ScalarArg(7, "i32")]) == 10
    selected = selection(tmp_path)
    result = SimpleNamespace(host_so=tmp_path / "build/model.so")

    def lowering(source, root, **kwargs):
        assert kwargs == {"targets": ("host",), "retain_llvm_dialect": False}
        root.mkdir()
        return result

    def load(path, **kwargs):
        assert path == str(result.host_so)
        assert set(kwargs) == {"image_policy", "scalar_result_dtype"} and kwargs["scalar_result_dtype"] is None
        return "unchanged ordinary model"

    monkeypatch.setattr(lower, "lower_model", lowering)
    monkeypatch.setattr(HostModel, "load", load)
    assert kernel_backend.compile_host(module(selected), tmp_path / "build") == "unchanged ordinary model"
