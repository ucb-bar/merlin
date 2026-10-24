"""Whole immutable-parameter projection across prepared SSA and generated ABI."""

from __future__ import annotations

import json
import struct
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, SymbolRefAttr

from merlin.common import mlir_query
from merlin.llvmlower import c_runtime
from merlin.llvmlower.entry_weight_projection import (
    ArgumentBinding,
    ArgumentOwnership,
    EntryArgumentTable,
    GeneratedDispatchABI,
    apply_entry_weight_projection,
    compact_runtime_rows,
    derive_entry_argument_table,
    module_text,
    plan_entry_weight_projection,
    prepare_entry_weight_projection,
)

CONTRACT = GeneratedDispatchABI("forward", True, True, True, True, True, True)
MODEL = """builtin.module {
  func.func @forward(%dead: tensor<5xf32>, %x: tensor<5xf32>, %live: tensor<5xf32>,
                     %state: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
    %empty = tensor.empty() : tensor<5xf32>
    %unused = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]} ins(%dead : tensor<5xf32>) outs(%empty : tensor<5xf32>) {
      ^bb0(%a: f32, %b: f32):
        %c = arith.mulf %a, %a : f32
        linalg.yield %c : f32
    } -> tensor<5xf32>
    %init = tensor.empty() : tensor<5xf32>
    %result = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        ins(%x, %live : tensor<5xf32>, tensor<5xf32>) outs(%init : tensor<5xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %s = arith.addf %a, %b : f32
        linalg.yield %s : f32
    } -> tensor<5xf32>
    func.return %result, %state : tensor<5xf32>, tensor<5xf32>
  }
}
"""


def table():
    return EntryArgumentTable(
        tuple(
            ArgumentBinding(index, kind, (5,), "f32", name)
            for index, kind, name in (
                (0, ArgumentOwnership.IMMUTABLE_STORED_WEIGHT, "dead"),
                (1, ArgumentOwnership.RETAINED_CAPTURE, None),
                (2, ArgumentOwnership.IMMUTABLE_STORED_WEIGHT, "live"),
                (3, ArgumentOwnership.RETAINED_CAPTURE, None),
            )
        ),
        4,
        (),
        ("tensor<5xf32>", "tensor<5xf32>"),
    )


def plan(module=None, arguments=None, contract=CONTRACT):
    return plan_entry_weight_projection(
        module or mlir_query.parse(MODEL), entry_contract=contract, argument_table=arguments or table()
    )


def bundle(tmp_path, *, session=False):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text(MODEL)
    dead = np.arange(5, dtype=np.float32)
    live = np.array([1, 3, 5, 7, 9], np.float32)
    payload = dead.tobytes() + live.tobytes()
    header = {
        "dead": {"dtype": "F32", "shape": [5], "data_offsets": [0, 20]},
        "live": {"dtype": "F32", "shape": [5], "data_offsets": [20, 40]},
    }
    encoded = json.dumps(header).encode()
    (capture / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (capture / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": {"kind": "param", "weight": "dead"},
                "1": {"kind": "input", "name": "x"},
                "2": {"kind": "param", "weight": "live"},
                "3": {"kind": "input", "name": "state"},
            }
        )
    )
    (capture / "input_order.json").write_text(json.dumps({"x": 0, "state": 1}))
    np.savez(capture / "inputs.npz", in0=np.arange(5, dtype=np.float32), in1=np.zeros(5, np.float32))
    if session:
        import yaml

        np.savez(capture / "session_inputs.npz", frames=np.arange(15, dtype=np.float32).reshape(3, 5))
        np.savez(capture / "session_goldens.npz", output0=np.arange(15, dtype=np.float32).reshape(3, 5))
        (capture / "session_contract.yaml").write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "kind": "frames",
                    "paper_ready": False,
                    "stages": ["step"],
                    "inputs": "session_inputs.npz",
                    "states": [{"name": "state", "input_arg": 3, "output_index": 1}],
                    "streams": [{"name": "x", "input_arg": 1, "key": "frames"}],
                    "quality": {
                        "scope": "trajectory",
                        "golden": "session_goldens.npz",
                        "key": "output0",
                        "output_index": 0,
                    },
                }
            )
        )
    return capture


def test_prepared_pure_dce_removes_dead_chain_and_only_whole_weight():
    module = mlir_query.parse(MODEL)
    original = module_text(module)
    projection = plan(module)
    assert projection.removed_original_indices == (0,)
    assert projection.retained_original_indices == (1, 2, 3)
    assert projection.original_to_projected == (None, 0, 1, 2)
    assert dict(projection.erased_operations) == {"linalg.generic": 1, "tensor.empty": 1}
    selected = apply_entry_weight_projection(module, projection)
    assert module_text(module) == original
    selected.verify()
    forward = next(op for op in selected.body.block.ops if isinstance(op, func.FuncOp))
    assert len(forward.function_type.inputs) == 3 and len(forward.function_type.outputs) == 2


def test_unused_runtime_input_and_prepared_weight_are_preserved():
    module = mlir_query.parse(
        MODEL.replace("%dead: tensor<5xf32>", "%dead: tensor<5xf32>, %unused_input: tensor<5xf32>")
    )
    entries = list(table().entries)
    entries.insert(1, ArgumentBinding(1, ArgumentOwnership.RETAINED_CAPTURE, (5,), "f32"))
    arguments = EntryArgumentTable(
        tuple(replace(entry, original_index=i) for i, entry in enumerate(entries)), 5, (), table().result_types
    )
    assert plan(module, arguments).removed_original_indices == (0,)
    all_retained = replace(
        table(),
        entries=tuple(
            replace(entry, kind=ArgumentOwnership.PREPARED_WEIGHT)
            if entry.kind == ArgumentOwnership.IMMUTABLE_STORED_WEIGHT
            else entry
            for entry in table().entries
        ),
    )
    assert plan(arguments=all_retained).removed_original_indices == ()


def test_direct_multiple_callers_and_output_order_are_rewritten():
    source = (
        MODEL[:-2]
        + """
      func.func @caller(%w: tensor<5xf32>, %x: tensor<5xf32>, %s: tensor<5xf32>) -> tensor<5xf32> {
        %a, %b = func.call @forward(%w, %x, %w, %s) :
          (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
        %c, %d = func.call @forward(%w, %a, %w, %b) :
          (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
        func.return %c : tensor<5xf32>
      }
    }
    """
    )
    module = mlir_query.parse(source)
    projection = plan(module)
    selected = apply_entry_weight_projection(module, projection)
    assert projection.direct_calls_rewritten == 2
    calls = [op for op in selected.walk() if isinstance(op, func.CallOp)]
    assert [len(call.arguments) for call in calls] == [3, 3]
    assert [len(call.results) for call in calls] == [2, 2]
    selected.verify()


@pytest.mark.parametrize(
    "field",
    [
        "compiler_generated_dispatch_only",
        "entry_address_unobserved",
        "nontrapping_arithmetic",
        "arithmetic_flags_unobserved",
        "generated_weight_storage_readonly",
        "weight_addresses_unobserved",
    ],
)
def test_unknown_external_abi_or_arithmetic_effects_refuse(field):
    with pytest.raises(ValueError, match="explicit generated"):
        plan(contract=replace(CONTRACT, **{field: False}))
    with pytest.raises(ValueError, match="explicit generated"):
        plan(contract=replace(CONTRACT, **{field: "true"}))


@pytest.mark.parametrize(
    "attribute",
    [
        SymbolRefAttr("forward"),
        ArrayAttr([SymbolRefAttr("forward")]),
        DictionaryAttr({"address": SymbolRefAttr("forward")}),
    ],
)
def test_nested_entry_symbol_escape_refuses(attribute):
    module = mlir_query.parse(MODEL)
    module.attributes["escaped_entry"] = attribute
    with pytest.raises(ValueError, match="symbolic/address escape"):
        plan(module)


def test_unknown_effect_in_dead_tensor_body_is_not_erased():
    source = MODEL.replace(
        "%c = arith.mulf %a, %a : f32", '"unknown.observe"(%a) : (f32) -> ()\n        %c = arith.mulf %a, %a : f32'
    )
    projection = plan(mlir_query.parse(source))
    assert projection.removed_original_indices == ()
    assert "linalg.generic" not in dict(projection.erased_operations)


def test_unknown_call_with_weight_preserves_it():
    source = MODEL.replace(
        "%empty = tensor.empty()", "func.call @observe(%dead) : (tensor<5xf32>) -> ()\n    %empty = tensor.empty()"
    )
    source = source.replace("builtin.module {", "builtin.module {\n func.func private @observe(tensor<5xf32>)")
    assert plan(mlir_query.parse(source)).removed_original_indices == ()


def test_physical_allocator_effect_is_preserved_despite_unused_result():
    source = MODEL.replace(
        "%empty = tensor.empty()", "%allocation = memref.alloc() : memref<5xf32>\n    %empty = tensor.empty()"
    )
    module = mlir_query.parse(source)
    selected = apply_entry_weight_projection(module, plan(module))
    assert sum(operation.name == "memref.alloc" for operation in selected.walk()) == 1


def test_stale_module_table_type_and_remap_refuse():
    module = mlir_query.parse(MODEL)
    projection = plan(module)
    module.attributes["changed"] = StringAttr("yes")
    with pytest.raises(ValueError, match="changed after planning"):
        apply_entry_weight_projection(module, projection)
    with pytest.raises(ValueError, match="changed after planning"):
        apply_entry_weight_projection(mlir_query.parse(MODEL), replace(projection, retained_original_indices=(0, 1, 3)))
    with pytest.raises(ValueError, match="type does not match"):
        plan(arguments=replace(table(), entries=(replace(table().entries[0], shape=(6,)), *table().entries[1:])))
    with pytest.raises(ValueError, match="captured output ABI"):
        plan(arguments=replace(table(), result_types=("tensor<5xf32>",)))


def test_runtime_pack_preserves_retained_tails_aliases_and_bounds():
    projection = plan()
    rows = [
        ("MERLIN_WEIGHT", 0, 1, [5], 4, "f32"),
        ("MERLIN_INPUT", 1, 1, [5], 4, "f32"),
        ("MERLIN_WEIGHT", 20, 1, [5], 4, "f32"),
        ("MERLIN_INPUT", 3, 1, [5], 4, "f32"),
        ("MERLIN_OUTPUT", 0, 1, [5], 4, "f32"),
        ("MERLIN_OUTPUT", 1, 1, [5], 4, "f32"),
    ]
    selected, packed = compact_runtime_rows(rows, bytes(range(40)), projection)
    assert len(selected) == 5 and packed == bytes(range(20, 40))
    assert selected[1][1] == 0 and selected[-2:] == rows[-2:]
    with pytest.raises(ValueError, match="exceeds actual packed"):
        compact_runtime_rows(rows, bytes(39), projection)
    wrong = list(rows)
    wrong[2] = ("MERLIN_INPUT", 20, 1, [5], 4, "f32")
    with pytest.raises(ValueError, match="descriptor disagrees"):
        compact_runtime_rows(wrong, bytes(40), projection)


def test_generated_ciface_and_weight_pack_share_projection(tmp_path):
    capture = bundle(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    arguments = derive_entry_argument_table(capture, prepared)
    projection = plan(arguments=arguments)
    info = c_runtime.generate(
        capture,
        tmp_path / "cgen",
        capture / "inputs.npz",
        prepared_dir=prepared,
        entry_projection=projection,
        entry_projection_source=capture / "model.mlir",
    )
    assert info["n_args"] == 5 and info["weights_bytes"] == 20
    assert (tmp_path / "cgen/weights.bin").read_bytes() == np.array([1, 3, 5, 7, 9], np.float32).tobytes()
    assert "_mlir_ciface_forward(void*,void*,void*,void*,void*)" in (tmp_path / "cgen/model_call.c").read_text()
    assert "{(void*)merlin_in_1,0,(void*)merlin_in_3,0,0}" in (tmp_path / "cgen/model_io.h").read_text()
    assert capture.joinpath("weights.safetensors").stat().st_size > info["weights_bytes"]


def test_session_stream_and_state_indices_remap_together(tmp_path):
    capture = bundle(tmp_path, session=True)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    projection = plan(arguments=derive_entry_argument_table(capture, prepared))
    info = c_runtime.generate(
        capture,
        tmp_path / "cgen",
        capture / "inputs.npz",
        prepared_dir=prepared,
        entry_projection=projection,
        entry_projection_source=capture / "model.mlir",
    )
    io = (tmp_path / "cgen/model_io.h").read_text()
    assert "MERLIN_STATE_INPUT_ARGS[1] = {2}" in io
    assert "MERLIN_STATE_OUTPUT_INDICES[1] = {1}" in io
    assert "MERLIN_INPUT_PTR[0] = " in io
    assert "memcpy(merlin_in_3, merlin_initial_3, 20UL)" in io
    assert info["n_state_pairs"] == 1


def test_table_source_mutation_refuses_before_generation(tmp_path):
    capture = bundle(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    projection = plan(arguments=derive_entry_argument_table(capture, prepared))
    path = capture / "weights.safetensors"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source changed"):
        apply_entry_weight_projection(mlir_query.parse(MODEL), projection)
    with pytest.raises(ValueError, match="source identity disagrees"):
        c_runtime.generate(
            capture,
            tmp_path / "cgen",
            capture / "inputs.npz",
            prepared_dir=prepared,
            entry_projection=projection,
            entry_projection_source=capture / "model.mlir",
        )
    assert not (tmp_path / "cgen").exists()


@pytest.mark.parametrize("role", ["states", "streams"])
def test_session_owned_weight_cannot_gain_immutable_projection_permission(tmp_path, role):
    import yaml

    capture = bundle(tmp_path, session=True)
    path = capture / "session_contract.yaml"
    session = yaml.safe_load(path.read_text())
    session[role][0]["input_arg"] = 0
    path.write_text(yaml.safe_dump(session))
    with pytest.raises(ValueError, match="session-owned weight"):
        derive_entry_argument_table(capture, tmp_path)


def test_normal_session_contract_is_pinned_before_abi_generation(tmp_path):
    capture = bundle(tmp_path, session=True)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    source = prepared / "input.mlir"
    source.write_text(MODEL)
    arguments = derive_entry_argument_table(capture, prepared)
    projection = plan(mlir_query.parse(MODEL), arguments)
    assert str((capture / "session_contract.yaml").resolve()) in {pin.path for pin in arguments.source_files}
    path = capture / "session_contract.yaml"
    path.write_text(path.read_text() + "\n# changed after planning\n")
    with pytest.raises(ValueError, match="source identity"):
        c_runtime.generate(
            capture,
            tmp_path / "cgen",
            capture / "inputs.npz",
            prepared_dir=prepared,
            entry_projection=projection,
            entry_projection_source=source,
        )
    assert not (tmp_path / "cgen").exists()


@pytest.mark.parametrize("source", ["session", "hoist_plan", "hoist_values", "weights"])
def test_absent_ownership_or_argument_source_cannot_appear_after_plan(tmp_path, source):
    from merlin.llvmlower import quant_hoist

    capture = bundle(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    if source == "weights":
        manifest_path = capture / "weights.safetensors.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for index in ("0", "2"):
            manifest[index]["stub"] = True
        manifest_path.write_text(json.dumps(manifest))
        (capture / "weights.safetensors").unlink()
    arguments = derive_entry_argument_table(capture, prepared)
    projection = plan(mlir_query.parse(MODEL), arguments)
    path = {
        "session": capture / "session_contract.yaml",
        "weights": capture / "weights.safetensors",
        "hoist_plan": prepared / quant_hoist.PLAN_FILE,
        "hoist_values": prepared / quant_hoist.VALUES_FILE,
    }[source]
    assert any(
        identity.path == str(path.absolute()) and identity.present is False
        for identity in arguments.optional_source_files
    )
    path.write_bytes(b"created after planning")
    with pytest.raises(ValueError, match="optional entry argument source"):
        apply_entry_weight_projection(mlir_query.parse(MODEL), projection)


def test_default_seam_does_no_io_and_cgen_requires_source(tmp_path):
    absent = tmp_path / "absent.mlir"
    selected, projection = prepare_entry_weight_projection(absent, tmp_path / "absent", tmp_path / "capture", None)
    assert selected == absent and projection is None and not selected.exists()
    capture = bundle(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    projection = plan(arguments=derive_entry_argument_table(capture, prepared))
    with pytest.raises(ValueError, match="original prepared source"):
        c_runtime.generate(
            capture, tmp_path / "cgen", capture / "inputs.npz", prepared_dir=prepared, entry_projection=projection
        )


def test_unused_generated_zero_weight_stub_has_separate_proven_ownership(tmp_path):
    capture = bundle(tmp_path)
    manifest = capture / "weights.safetensors.manifest.json"
    metadata = json.loads(manifest.read_text())
    metadata["0"]["stub"] = True
    manifest.write_text(json.dumps(metadata))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    arguments = derive_entry_argument_table(capture, prepared)
    assert arguments.entries[0].kind == ArgumentOwnership.IMMUTABLE_GENERATED_ZERO_WEIGHT
    projection = plan(arguments=arguments)
    assert projection.removed_original_indices == (0,)
    info = c_runtime.generate(
        capture,
        tmp_path / "cgen",
        capture / "inputs.npz",
        prepared_dir=prepared,
        entry_projection=projection,
        entry_projection_source=capture / "model.mlir",
    )
    assert info["weights_bytes"] == 20
    assert (tmp_path / "cgen/weights.bin").read_bytes() == np.array([1, 3, 5, 7, 9], np.float32).tobytes()


def test_stub_metadata_never_promotes_runtime_input(tmp_path):
    capture = bundle(tmp_path)
    manifest = capture / "weights.safetensors.manifest.json"
    metadata = json.loads(manifest.read_text())
    metadata["0"] = {"kind": "input", "name": "extra", "stub": True}
    manifest.write_text(json.dumps(metadata))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    arguments = derive_entry_argument_table(capture, prepared)
    assert arguments.entries[0].kind == ArgumentOwnership.RETAINED_CAPTURE
    assert plan(arguments=arguments).removed_original_indices == ()


def test_dynamic_argument_and_malformed_ownership_refuse():
    with pytest.raises(ValueError, match="static argument shapes"):
        plan(arguments=replace(table(), entries=(replace(table().entries[0], shape=(-1,)), *table().entries[1:])))
    with pytest.raises(ValueError, match="unknown entry argument ownership"):
        plan(
            arguments=replace(
                table(), entries=(replace(table().entries[0], kind="immutable_stored_weight"), *table().entries[1:])
            )
        )


def test_trailing_quant_hoist_is_retained_and_cgen_compounds_original_remap(tmp_path):
    from merlin.llvmlower import quant_hoist

    capture = bundle(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    quant_hoist.write_plan(prepared, [quant_hoist.HoistedArg("packed", (5,), "i8")])
    quant_hoist.write_values(prepared, {"packed": np.array([-128, -1, 0, 1, 127], np.int8)})
    source = MODEL.replace("%state: tensor<5xf32>", "%state: tensor<5xf32>, %packed: tensor<5xi8>")
    path = prepared / "source.mlir"
    path.write_text(source)
    arguments = derive_entry_argument_table(capture, prepared)
    assert arguments.captured_arguments == 4 and len(arguments.entries) == 5
    assert arguments.entries[-1].kind == ArgumentOwnership.PREPARED_WEIGHT
    projection = plan(mlir_query.parse(path), arguments)
    assert projection.retained_original_indices == (1, 2, 3, 4)
    info = c_runtime.generate(
        capture,
        tmp_path / "cgen",
        capture / "inputs.npz",
        prepared_dir=prepared,
        entry_projection=projection,
        entry_projection_source=path,
    )
    assert info["n_args"] == 6 and info["n_quant_hoist"] == 1
    packed = (tmp_path / "cgen/weights.bin").read_bytes()
    assert packed[:20] == np.array([1, 3, 5, 7, 9], np.float32).tobytes()
    assert packed[64:] == np.array([-128, -1, 0, 1, 127], np.int8).tobytes()


def test_normal_upstream_rv64_and_native_ciface_keep_exact_outputs(tmp_path):
    from merlin.llvmlower import abi, codegen, toolchain
    from merlin.runtime.backends import spike, spike_model

    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")
    capture = bundle(tmp_path)
    work = tmp_path / "build"
    built = spike_model.build(
        capture,
        work,
        arena_mb=1,
        backend="scalar",
        host_vectorize=False,
        entry_weight_projection=CONTRACT,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    )
    assert built["entry_projection"]["removed_original_indices"] == [0]
    assert built["n_args"] == 5 and built["weights_bytes"] == 20
    record = json.loads((work / "compilation_recipe.json").read_text())
    assert record["status"] == "completed" and "entry_weight_projection" in record["preparation"]
    result = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
    np.testing.assert_array_equal(result["outputs"], np.array([1, 4, 7, 10, 13], np.float32))
    # Use the exact same upstream LLVM and normal native ABI helper. Both result
    # buffers, multiple invocations and readonly inputs are checked separately.
    shared = codegen.build_host_shared(work / "lower/model.ll", work / "native.so")
    native = abi.HostModel.load(str(shared))
    live = np.array([1, 3, 5, 7, 9], np.float32)
    for x, state in (
        (np.arange(5, dtype=np.float32), np.array([100, 101, 102, 103, 104], np.float32)),
        (np.array([-5, -1, 0, 3, 11], np.float32), np.array([-9, -7, -5, -3, -1], np.float32)),
    ):
        inputs = [x, live, state]
        before = [array.copy() for array in inputs]
        outputs = [np.full(5, -99, np.float32), np.full(5, -77, np.float32)]
        native([(array.ctypes.data, [5]) for array in (*inputs, *outputs)])
        np.testing.assert_array_equal(outputs[0].view(np.uint32), (x + live).view(np.uint32))
        np.testing.assert_array_equal(outputs[1].view(np.uint32), state.view(np.uint32))
        for actual, expected in zip(inputs, before, strict=True):
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_quant_inner_and_projection_compose_without_source_index_drift(tmp_path):
    from merlin.llvmlower import qinner

    capture = bundle(tmp_path)
    source = MODEL.replace(
        "%init = tensor.empty()", "%inner = tensor.empty() : tensor<5xf32>\n    %init = tensor.empty()"
    )
    source = source.replace("ins(%x, %live :", "ins(%x, %inner :")
    source = source.replace(
        "outs(%init : tensor<5xf32>) {", 'outs(%init : tensor<5xf32>) attrs = {prov.quant_inner_1 = "scale"} {'
    )
    (capture / "model.mlir").write_text(source)
    scale = np.array([2, 4, 6, 8, 10], np.float32)
    np.savez(capture / "extra.npz", **{"qinner::scale": scale})
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    module = mlir_query.parse(source).clone()
    lifted = qinner.lift(module)
    assert len(lifted) == 1
    path = prepared / "source.mlir"
    path.write_text(module_text(module))
    arguments = derive_entry_argument_table(capture, prepared)
    assert arguments.captured_arguments == 4 and len(arguments.entries) == 5
    projection = plan(module, arguments)
    assert projection.removed_original_indices == (0, 2)
    assert projection.retained_original_indices == (1, 3, 4)
    info = c_runtime.generate(
        capture,
        tmp_path / "cgen",
        capture / "inputs.npz",
        prepared_dir=prepared,
        entry_projection=projection,
        entry_projection_source=path,
    )
    assert info["n_args"] == 5 and info["n_qinner"] == 1
    assert (tmp_path / "cgen/weights.bin").read_bytes() == scale.tobytes()


@pytest.mark.parametrize(
    "cast,source_type", [("extsi", "i8"), ("extui", "i8"), ("trunci", "i64"), ("bitcast", "f32")]
)
@pytest.mark.parametrize("live", [False, True])
def test_value_cast_traits_do_not_keep_dead_tensor_components(cast, source_type, live):
    from merlin.llvmlower.entry_weight_projection import _pure_dce

    result = "tensor<2xi32>" if live else f"tensor<2x{source_type}>"
    returned = "%cast" if live else "%x"
    source = f'''builtin.module {{
      func.func @forward(%x: tensor<2x{source_type}>) -> {result} {{
        %empty=tensor.empty() : tensor<2xi32>
        %cast=linalg.generic {{
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]
        }} ins(%x : tensor<2x{source_type}>) outs(%empty : tensor<2xi32>) {{
        ^bb0(%a: {source_type},%unused: i32):
          %v=arith.{cast} %a : {source_type} to i32
          linalg.yield %v : i32
        }} -> tensor<2xi32>
        func.return {returned} : {result}
      }}
    }}'''
    module = mlir_query.parse(source)
    function = next(op for op in module.walk() if op.name == "func.func")
    _pure_dce(function)
    module.verify()
    assert any(op.name == "linalg.generic" for op in module.walk()) is live
    assert len(function.body.block.args) == 1
