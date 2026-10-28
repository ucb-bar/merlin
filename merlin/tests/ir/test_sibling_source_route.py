"""Original sibling source/ABI custody; target and numerical owners stay external."""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_source_region import sibling_source

from merlin.llvmlower import device_preparation, device_shim
from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.device_offload import build_arguments, load_sidecar
from merlin.llvmlower.sibling_source_route import SiblingSourceRoute, require_host, with_borrowed_bridge
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.xdsl_dialects._common import text


def route(tmp_path, monkeypatch, *, module=None):
    module = sibling_source() if module is None else module
    source = tmp_path / "original.mlir"
    source.write_text(text(module, generic=True))
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "compiler.py").write_text("# generic target package selected separately\n")
    selected = SiblingSourceRoute(
        hashlib.sha256(text(module, generic=True).encode()).hexdigest(),
        BorrowedBufferEffects("independent-original-implementation", True, True, True),
    )
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: device_shim.KernelAbi("logical", version=2))
    device = DeviceRouting("selected-target", str(package), "int8", "i32", granularity="group", sibling_source=selected)
    work = tmp_path / "build"
    work.mkdir()
    return module, source, selected, device, work


def test_original_host_and_multilateral_logical_sidecar_remain_bound(tmp_path, monkeypatch):
    module, source, selected, device, work = route(tmp_path, monkeypatch)
    before = text(module, generic=True)
    frozen = device_preparation.freeze_device_offload_selection(device)
    path = device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    host = frozen.sibling_host
    require_host(host, selected)
    assert text(module, generic=True) == before
    assert any(op.name == "arith.sitofp" for op in host.module.walk())
    assert not any(op.name == "arith.muli" for op in host.module.walk())
    assert host.module.body.block.first_op.function_type == module.body.block.first_op.function_type
    assert host.writer.result_arguments == (3, 4)
    cb = parse_interface_mlir(host.interface)
    assert [c["attributes"]["output_dtype"] for c in cb["commands"] if c["opcode"] == "COMMIT"] == ["i32", "i32"]
    prepared = device_preparation.prepare_device_route(device, path, work, selection=frozen)
    assert prepared == path
    args = build_arguments(load_sidecar(work), expected_granularity="group")
    assert set(args["signatures"]) == {"source_shared_core_device"}
    assert len(args["call_buffers"]["source_shared_core_device"]) == 5
    assert args["entries"]["source_shared_core_device"]["source_ir_sha256"] == selected.source_ir_sha256
    device_preparation.require_device_offload_selection(device, frozen)
    with pytest.raises(ValueError, match="more than once"):
        device_preparation.prepare_sibling_device_host(device, source, work, frozen)


@pytest.mark.parametrize("damage", ["source", "host", "interface", "sidecar", "selection"])
def test_changed_original_source_or_realized_products_refuse(tmp_path, monkeypatch, damage):
    module, source, selected, device, work = route(tmp_path, monkeypatch)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    host = frozen.sibling_host
    if damage == "source":
        source.write_text(source.read_text() + "\n// changed original bytes\n")
    elif damage == "host":
        host.module.body.block.first_op.sym_name = type(host.module.body.block.first_op.sym_name)("changed")
    elif damage == "interface":
        object.__setattr__(host, "interface", host.interface + "\n")
    elif damage == "sidecar":
        host.rewrite.entries[host.device_symbol]["core_ir_sha256"] = "0" * 64
    else:
        object.__setattr__(device, "sibling_source", replace(selected))
    with pytest.raises(ValueError, match="changed"):
        device_preparation.require_device_offload_selection(device, frozen)


def test_unknown_effect_nonzero_seed_and_opaque_routes_refuse(tmp_path, monkeypatch):
    for label, module in (("effect", sibling_source(unknown_effect=True)), ("seed", sibling_source(seed=1))):
        folder = tmp_path / label
        folder.mkdir()
        _, source, _, device, work = route(folder, monkeypatch, module=module)
        frozen = device_preparation.freeze_device_offload_selection(device)
        with pytest.raises(ValueError):
            device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    _, _, selected, device, _ = route(tmp_path, monkeypatch)
    for field in ("prepared_transform", "post_offload_transform", "catalog_builder"):
        with pytest.raises(ValueError, match="opaque"):
            replace(device, **{field: lambda *a: None})
    with pytest.raises(ValueError, match="typed selection"):
        replace(device, sibling_source=SimpleNamespace(**selected.record()))


def test_logical_borrowed_bridge_transports_all_results_without_tensor_arithmetic(tmp_path, monkeypatch):
    _, source, selected, device, work = route(tmp_path, monkeypatch)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    host = frozen.sibling_host
    buffers, reason = device_shim.logical_buffers(host.interface)
    assert not reason
    unit = device_shim.emit_logical_translation_unit(
        device.device, {host.device_symbol: (buffers, host.rewrite.call_buffers[host.device_symbol])},
        kernel_symbol_for=lambda _: "selected_device_kernel",
    )
    unit = with_borrowed_bridge(unit, selected, host)
    assert "void _mlir_ciface_source_shared_core_borrowed(" in unit.text
    assert "_mlir_ciface_source_shared_core_device(&result, d0, d1, d2, d3, d4);" in unit.text
    assert "selected_device_kernel((void *)p0_addr, (void *)p1_addr, (void *)p2_addr, (void *)p3_addr" in unit.text
    assert "malloc(" not in unit.text and "free(" not in unit.text
    assert unit.symbols == (host.device_symbol,)


def test_owned_model_entry_consumes_sibling_route_in_real_host_preparation(tmp_path, monkeypatch):
    from merlin.compile import baremetal_model, route_before_build
    from merlin.compile.owned_host_model import OwnedHostModelCompiler
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    module, source, selected, device, _ = route(tmp_path, monkeypatch)
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    (fixed / "fixed.txt").write_text("independent host")
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_bytes(source.read_bytes())
    service = OwnedHostModelCompiler.select(
        host_package=fixed, target=device.device, board_catalog=tmp_path / "boards",
        board="board", dts=tmp_path / "board.dts",
    )
    original_device = replace(device, sibling_source=None)
    monkeypatch.setattr(route_before_build, "plan_before_build", lambda *a, **kw: {
        "device_routing": original_device, "offload": {"groups": 2},
    })

    def build(**kwargs):
        assert kwargs["respect_captured_quantization_scope"] is True
        assert kwargs["device"].sibling_source is selected
        out = kwargs["output"]
        out.mkdir()
        frozen = device_preparation.freeze_device_offload_selection(kwargs["device"])
        final, _ = prepare_for_lowering(
            capture / "model.mlir", out, device=kwargs["device"], blocking=False,
            device_offload_invocation=frozen,
        )
        result = parse_mlir_file(final)
        result.verify()
        assert result.body.block.first_op.function_type == module.body.block.first_op.function_type
        assert any(op.name == "arith.sitofp" for op in result.walk())
        assert not any(op.name == "arith.muli" for op in result.walk())
        assert set(load_sidecar(out)["signatures"]) == {"source_shared_core_device"}
        device_preparation.require_device_offload_selection(kwargs["device"], frozen)
        return {"status": "ordinary_original_host_prepared", "raw_device_groups": 1}

    monkeypatch.setattr(baremetal_model, "compile_saved_model", build)
    assert service.compile(
        capture=capture, candidate_package=device.package_dir, datapath="int8", linked_elf_admission=object(),
        output=tmp_path / "owned_build", arena_mb=1, sibling_source=selected,
    )["raw_device_groups"] == 1


def test_complete_source_group_contract_refuses_an_uncovered_accelerator_group(tmp_path, monkeypatch):
    from merlin.llvmlower import device_offload

    module, source, _, device, work = route(tmp_path, monkeypatch)
    original = text(module, generic=True)
    device = replace(device, require_complete_groups=True)

    def independent_census(candidate, target, **kwargs):
        assert text(candidate, generic=True) == original
        assert kwargs["require_complete_groups"] is True
        return SimpleNamespace(moved=3)

    monkeypatch.setattr(device_offload, "rewrite_groups_to_device", independent_census)
    frozen = device_preparation.freeze_device_offload_selection(device)
    with pytest.raises(ValueError, match="every independently required"):
        device_preparation.prepare_sibling_device_host(device, source, work, frozen)


def test_wrong_original_source_identity_refuses_before_host_or_package_emission(tmp_path, monkeypatch):
    _, source, selected, device, work = route(tmp_path, monkeypatch)
    device = replace(device, sibling_source=replace(selected, source_ir_sha256="0" * 64))
    frozen = device_preparation.freeze_device_offload_selection(device)
    with pytest.raises(ValueError, match="original source SSA"):
        device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    assert not (work / "sibling_source").exists()


def test_actual_llvm_entry_reader_accepts_generic_form_and_refuses_partial_abi(tmp_path, monkeypatch):
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    from merlin.llvmlower.sibling_source_route import require_emission

    _, source, selected, device, work = route(tmp_path, monkeypatch)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    host = frozen.sibling_host
    for name, count in (("kernel_entry", 5), ("wrong", 5), ("kernel_entry", 4)):
        types = [llvm.LLVMPointerType()] * count
        body = Block(arg_types=types)
        body.add_op(llvm.ReturnOp())
        module = ModuleOp([llvm.FuncOp(name, llvm.LLVMFunctionType(types, llvm.LLVMVoidType()), body=Region(body))])
        artifact = text(module, generic=True)
        if name == "kernel_entry" and count == 5:
            require_emission(artifact, selected, host, entry_symbol="kernel_entry")
        else:
            with pytest.raises(ValueError, match="kernel entry"):
                require_emission(artifact, selected, host, entry_symbol="kernel_entry")


def composed_source_roster(module):
    from merlin_experiments.phase1.feedback.private_group_provenance import source_group_roster
    from test_source_region import sibling_roots
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    roots = sibling_roots(module)
    groups = []
    for index, root in enumerate(roots):
        root.attributes["prov.region_id"] = StringAttr("original_region_" + str(index))
        root.attributes["prov.source_node_ids"] = ArrayAttr([StringAttr("original_node_" + str(index))])
        groups.append(SimpleNamespace(index=index, root=root, members=(root,), placement="selected-target"))
    return source_group_roster(groups, [0, 1], host_placement="host", module=module)


def test_source_composition_join_derives_all_original_members_without_collapsing_denominator(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback.private_group_provenance import join_sibling_source_composition

    module = sibling_source()
    roster = composed_source_roster(module)
    _, source, _, device, work = route(tmp_path, monkeypatch, module=module)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    emitted = frozen.sibling_host.rewrite.to_sidecar()
    joined = join_sibling_source_composition(emitted, roster, module=module, target=device.device)
    assert joined["required_source_groups"] == [0, 1]
    assert joined["covered_source_groups"] == [0, 1]
    assert len(joined["members"]) == 2
    assert joined["raw_output_elements"] == 8 and joined["source_useful_macs"] == 16
    assert joined["original_source_denominator_unchanged"] is True
    assert "execution" in joined["scope"]


@pytest.mark.parametrize("damage", ["missing_member", "reordered_members", "wrong_pitch", "narrow_commit",
                                    "invented_source", "wrong_contraction", "uncovered_required", "physical_alias"])
def test_source_composition_join_refuses_changed_source_programs_and_coverage(tmp_path, monkeypatch, damage):
    from copy import deepcopy

    from merlin_experiments.phase1.feedback.private_group_provenance import join_sibling_source_composition

    module = sibling_source()
    roster = composed_source_roster(module)
    _, source, _, device, work = route(tmp_path, monkeypatch, module=module)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    emitted = deepcopy(frozen.sibling_host.rewrite.to_sidecar())
    symbol = frozen.sibling_host.device_symbol
    if damage == "missing_member":
        emitted["programs"][symbol]["commands"] = emitted["programs"][symbol]["commands"][:4]
    elif damage == "reordered_members":
        emitted["call_buffers"][symbol][-2:] = list(reversed(emitted["call_buffers"][symbol][-2:]))
    elif damage == "wrong_pitch":
        emitted["call_buffers"][symbol][1]["shape"] = [1, 4]
    elif damage == "narrow_commit":
        emitted["programs"][symbol]["commands"][2]["attributes"]["output_dtype"] = "i8"
    elif damage == "invented_source":
        roster["eligible_group_provenance"][0]["source_node_ids"] = ["other"]
    elif damage == "wrong_contraction":
        roster["source_group_contractions"][0]["reduction"] = [3]
    elif damage == "uncovered_required":
        roster["eligible_groups"].append(2)
        roster["eligible_group_provenance"].append({
            "source_group": 2, "source_region_id": "extra_region", "source_node_ids": ["extra_node"],
        })
    else:
        emitted["routed"].append(deepcopy(emitted["routed"][0]))
    with pytest.raises(ValueError):
        join_sibling_source_composition(emitted, roster, module=module, target=device.device)


def test_composed_audit_budgets_one_physical_object_without_inventing_source_groups(tmp_path, monkeypatch):
    from copy import deepcopy

    from merlin_experiments.phase1.feedback.private_device_audit import _composed_source_metrics
    from merlin_experiments.phase1.feedback.private_group_provenance import join_sibling_source_composition

    module = sibling_source()
    roster = composed_source_roster(module)
    _, source, _, device, work = route(tmp_path, monkeypatch, module=module)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    emitted = frozen.sibling_host.rewrite.to_sidecar()
    composition = join_sibling_source_composition(emitted, roster, module=module, target=device.device)
    sources, metrics = _composed_source_metrics(emitted["routed"], roster, composition)
    assert sources == {0: (0, 1)} and metrics == {0: {"elements": 8, "element_bytes": 4}}
    for damage in ("source_width", "raw_output_size", "physical_symbol", "missing_member"):
        changed = deepcopy(composition)
        if damage == "source_width":
            changed["members"][0]["geometry_m_k_n"][-1] = 3
        elif damage == "raw_output_size":
            changed["raw_output_elements"] = 4
        elif damage == "physical_symbol":
            changed["physical_symbol"] = "other"
        else:
            changed["members"].pop()
        with pytest.raises(ValueError):
            _composed_source_metrics(emitted["routed"], roster, changed)


def test_composition_join_preserves_original_eligibility_through_checked_weight_preparation(tmp_path, monkeypatch):
    import json
    import struct

    import numpy as np
    from merlin_experiments.phase1.feedback.private_group_provenance import join_sibling_source_composition

    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower import weight_prepack
    from merlin.llvmlower.compilation_recipe import CompilationRecipe

    original = sibling_source()
    roster = composed_source_roster(original)
    bundle = tmp_path / "capture"
    bundle.mkdir()
    (bundle / "model.mlir").write_text(text(original, generic=True))
    header = json.dumps({
        "b": {"dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]},
        "w": {"dtype": "I8", "shape": [2, 2], "data_offsets": [4, 8]},
    }).encode()
    header += b" " * ((-len(header)) % 8)
    payload = np.array([1, -2, 3, 4, -5, 6, 7, 8], dtype=np.int8).tobytes()
    (bundle / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (bundle / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "input", "dtype": "int8", "shape": [2, 2]},
        "1": {"kind": "param", "weight": "b", "dtype": "int8", "shape": [2, 2]},
        "2": {"kind": "buffer", "weight": "w", "dtype": "int8", "shape": [2, 2]},
    }))
    monkeypatch.setattr(weight_prepack, "_default_cache_root", lambda: tmp_path / "cache")
    from merlin.targetgen.owned_host_model_execution import _resolve_weight_preparation

    selected_prepack, selected_record = _resolve_weight_preparation(bundle, requested=True)
    assert selected_prepack is True and selected_record["immutable_arguments"] == [1, 2]
    preparation_work = tmp_path / "preparation"
    preparation_work.mkdir()
    compilation = CompilationRecipe(preparation_work, producer=Path(device_preparation.__file__))
    prepared = weight_prepack.prepare_build_bundle(
        bundle, preparation_work, features={weight_prepack.FEATURE}, compilation=compilation,
    )
    receipt = tmp_path / "preparation/bundle_preparation.json"
    prepared_module = parse_mlir_file(prepared / "model.mlir")
    assert text(original, generic=True) != text(prepared_module, generic=True)
    build = tmp_path / "prospective"
    build.mkdir()
    from merlin.compile.complete_group_routing import select_complete_group
    from merlin.llvmlower.prepacked_device_capture import rebind

    # Select the ORIGINAL source, then use the ordinary checked preparation
    # rebind. Layout derivation must not replace original owner authority.
    _, _, selected, device, _ = route(build, monkeypatch, module=original)
    device = replace(device, select=select_complete_group, require_complete_groups=True, capture=bundle)
    work = receipt.parent
    device = rebind(device, source=bundle, prepared=prepared, work=work)
    assert device.sibling_source is selected
    # Target eligibility is independently exercised by its owner; this fixture
    # isolates original/prepared source and ABI custody.
    device = replace(device, require_complete_groups=False)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, prepared / "model.mlir", work, frozen)
    device_preparation.retain_device_offload_source(frozen, prepared / "model.mlir", compilation=compilation)
    retained = compilation.record["preparation"]
    paths = [pin["path"] for pin in retained.values()]
    assert len(paths) == len(set(paths))
    original_pin = next(pin for pin in retained.values() if pin["path"] == str(prepared / "model.mlir"))
    assert (prepared / "model.mlir", original_pin["sha256"]) in frozen.sources
    saved_sha = original_pin["sha256"]
    original_pin["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="already retained preparation"):
        device_preparation.retain_device_offload_source(frozen, prepared / "model.mlir", compilation=compilation)
    original_pin["sha256"] = saved_sha
    emitted = frozen.sibling_host.rewrite.to_sidecar()
    assert emitted["model_sha256"] == selected.source_ir_sha256
    # The square fixture has the same abstract raw core before/after packing;
    # input-layout ancestry is recorded by the explicit checked receipt.
    joined = join_sibling_source_composition(
        emitted, roster, module=original, target=device.device, preparation_receipt=receipt,
    )
    assert joined["source_ir_sha256"] != joined["prepared_source_ir_sha256"]
    assert joined["covered_source_groups"] == joined["required_source_groups"] == [0, 1]
    assert joined["preparation"]["immutable_arguments"] == [1, 2]
    imported_owner = json.loads(json.dumps(emitted))
    imported_owner["model_sha256"] = joined["prepared_source_ir_sha256"]
    for entry in imported_owner["entries"].values():
        entry["source_ir_sha256"] = joined["prepared_source_ir_sha256"]
    with pytest.raises(ValueError, match="source and complete ABI"):
        join_sibling_source_composition(
            imported_owner, roster, module=original, target=device.device, preparation_receipt=receipt,
        )
    # A valid preparation from another original source cannot borrow eligibility.
    other = sibling_source(seed=1)
    with pytest.raises(ValueError, match="ancestry differs"):
        join_sibling_source_composition(
            emitted, roster, module=other, target=device.device, preparation_receipt=receipt,
        )
    # Original-value mutation invalidates the existing inverse-byte preparation proof.
    stored = prepared / "weights.safetensors"
    raw = stored.read_bytes()
    stored.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises((ValueError, weight_prepack.PrepackRefused)):
        join_sibling_source_composition(
            emitted, roster, module=original, target=device.device, preparation_receipt=receipt,
        )


def test_composition_preserves_original_node_order_after_canonical_eligibility_join(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback.private_group_provenance import join_sibling_source_composition
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    module = sibling_source()
    roster = composed_source_roster(module)
    root = next(op for op in module.walk() if getattr(op.attributes.get("prov.region_id"), "data", None)
                == "original_region_0")
    root.attributes["prov.source_node_ids"] = ArrayAttr([StringAttr("node_z"), StringAttr("node_a")])
    roster["eligible_group_provenance"][0]["source_node_ids"] = ["node_a", "node_z"]
    _, source, _, device, work = route(tmp_path, monkeypatch, module=module)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    joined = join_sibling_source_composition(
        frozen.sibling_host.rewrite.to_sidecar(), roster, module=module, target=device.device,
    )
    assert joined["members"][0]["source_node_ids"] == ["node_z", "node_a"]


@pytest.mark.parametrize("dense", [False, True])
def test_readonly_dense_offsets_compose_with_exact_original_borrowed_wrapper(tmp_path, monkeypatch, dense):
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StridedLayoutAttr

    from merlin.llvmlower.sibling_input_views import bind

    module = sibling_source()
    if not dense:
        function = module.body.block.first_op
        function.properties["arg_attrs"] = ArrayAttr([
            DictionaryAttr({"bufferization.buffer_layout": StridedLayoutAttr([3, 1], 0)}),
            DictionaryAttr({}), DictionaryAttr({}),
        ])
    _, source, selected, device, work = route(tmp_path, monkeypatch, module=module)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    host = frozen.sibling_host
    buffers, _ = device_shim.logical_buffers(host.interface)
    actual = host.module.clone()
    chosen, retained, _ = bind(actual, host, selected,
                              {host.device_symbol: (buffers, host.rewrite.call_buffers[host.device_symbol])})
    name = host.writer.symbol
    assert (0 in chosen.get(name, [])) is dense
    assert (0 in retained.get(name, [])) is not dense
    wrapper = next(o for o in actual.body.block.ops if getattr(o, "sym_name", None) and o.sym_name.data == name)
    assert all("bufferization.buffer_layout" not in tuple(wrapper.arg_attrs)[i].data
               for i in host.writer.result_arguments)
    actual.verify()
    require_host(host, selected)  # Binding changed only the independently parsed input, never frozen custody.


@pytest.mark.parametrize("damage", ["writer_access", "borrowed_access"])
def test_dense_wrapper_offsets_refuse_changed_original_access(tmp_path, monkeypatch, damage):
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

    from merlin.llvmlower.sibling_input_views import bind

    _, source, selected, device, work = route(tmp_path, monkeypatch)
    frozen = device_preparation.freeze_device_offload_selection(device)
    device_preparation.prepare_sibling_device_host(device, source, work, frozen)
    host = frozen.sibling_host
    buffers, _ = device_shim.logical_buffers(host.interface)
    actual = host.module.clone()
    name = host.writer.symbol if damage == "writer_access" else host.writer.borrowed_symbol
    function = next(op for op in actual.body.block.ops if getattr(op, "sym_name", None) and op.sym_name.data == name)
    attrs = list(function.arg_attrs)
    attrs[0] = DictionaryAttr({"bufferization.access": StringAttr("write")})
    function.properties["arg_attrs"] = ArrayAttr(attrs)
    with pytest.raises(ValueError, match="exact original borrowed writer"):
        bind(actual, host, selected, {host.device_symbol: (buffers, host.rewrite.call_buffers[host.device_symbol])})
