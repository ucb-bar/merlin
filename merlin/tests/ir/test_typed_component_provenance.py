"""Independent qualification rederives typed exits from original source."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback.private_component_provenance import join_components
from merlin_experiments.phase1.feedback.private_group_provenance import source_group_roster
from test_output_fusion import _path
from test_sibling_components import effects
from test_sibling_quantized_readout import shared_readout_source
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
from merlin.llvmlower import device_shim
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.sibling_component_route import build_components
from merlin.llvmlower.sibling_source_route import SiblingSourceRoute, _sha, _text
from merlin.xdsl_dialects.lowering import output_fusion as OF


def routed(tmp_path, monkeypatch, *, observer=False, heads=False, whole=False, captured=False):
    import json

    from merlin.compile import mesh
    from merlin.targetgen.corpus_spec import CorpusBinding

    if heads:
        import test_joint_source_route

        monkeypatch.setattr(test_joint_source_route, "multi_source",
                            lambda **kwargs: shared_readout_source(raw_observer=observer))
        module = test_joint_source_route.joint_source()
    else:
        module = shared_readout_source(raw_observer=observer)
    stored = {}
    if captured:
        import numpy as np

        function = module.body.block.first_op
        for op in tuple(function.body.block.ops):
            if op.name != "arith.constant" or not (op.result.name_hint or "").startswith("scale_v"):
                continue
            index = len(function.body.block.args)
            value = function.body.block.insert_arg(op.result.type, index)
            stored[index] = (f"scale_{index}", np.asarray(list(op.value.get_values()), dtype=np.float32))
            op.result.replace_all_uses_with(value)
            function.body.block.erase_op(op)
        function.update_function_type()
        module.verify()
    roots = []
    for op in module.walk():
        try:
            read_zero_seeded_integer_matmul(op)
        except ValueError:
            continue
        index = len(roots)
        op.attributes.update({"prov.region_id": StringAttr(f"original_{index}"),
                              "prov.source_node_ids": ArrayAttr([StringAttr(f"node_{index}")])})
        roots.append(op)
    groups = [SimpleNamespace(root=root, members=(root,), index=i, placement="selected")
              for i, root in enumerate(roots)]
    roster = source_group_roster(groups, list(range(len(groups))), host_placement="host", module=module)
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "compiler.py").write_text("# separately selected neutral consumer\n")
    (tmp_path / "model.mlir").write_text(_text(module))
    manifest = {str(i): {"kind": "input", "name": f"arg_{i}"}
                for i in range(len(module.body.block.first_op.function_type.inputs))}
    header, payload = {}, b""
    for index, (name, values) in stored.items():
        manifest[str(index)] = {"kind": "param", "name": name, "weight": name}
        raw = values.tobytes()
        header[name] = dict(dtype="F32", shape=list(values.shape), data_offsets=[len(payload), len(payload) + len(raw)])
        payload += raw
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "weights.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    monkeypatch.setattr(OF, "declared_path", lambda _: _path())
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: device_shim.KernelAbi("logical", version=2))
    monkeypatch.setattr(mesh, "_mesh_tile_binding",
                        lambda *args, **kwargs: CorpusBinding("neutral", 16, "int8", "i32", True, [], "exact_int"))
    if heads:
        from fake_quant_layer import Oracle

        from merlin.llvmlower import device_offload
        from merlin.llvmlower.head_source_route import HeadBatchSourceRoute

        selected = HeadBatchSourceRoute(_sha(_text(module)), effects(), shared_input_siblings=True, whole_batch=whole)
        ordinary = device_offload.rewrite_groups_to_device

        def rewrite(*args, **kwargs):
            return ordinary(*args, **{**kwargs, "oracle": Oracle(), "weight_args": ()})

        monkeypatch.setattr(device_offload, "rewrite_groups_to_device", rewrite)
    else:
        selected = SiblingSourceRoute(_sha(_text(module)), effects(), partial=True)
    device = DeviceRouting("neutral", str(package), "int8", "i32", granularity="group",
                           sibling_source=selected, capture=tmp_path, select=lambda _: True)
    if heads:
        from merlin.llvmlower.head_source_route import prepare_host

        _, host = prepare_host(device, selected, tmp_path / "model.mlir", tmp_path / "route")
    else:
        host = build_components(device, selected, module, original=module)
    return module, roster, host.rewrite.to_sidecar()


@pytest.mark.parametrize("observer", [False, True])
def test_independent_join_preserves_source_scales_bias_and_every_typed_observer(tmp_path, monkeypatch, observer):
    module, roster, emitted = routed(tmp_path, monkeypatch, observer=observer)
    before = _text(module)
    joined = join_components(emitted, roster, module=module, target="neutral", effects=effects(), capture=tmp_path)
    component, = joined["components"]
    symbol = component["physical_symbol"]
    commits = [c for c in emitted["programs"][symbol]["commands"] if c["opcode"] == "COMMIT"]
    assert [c["attributes"]["output_dtype"] for c in commits] == (["i32", "i8"] if observer else ["i8", "i8"])
    assert component["descriptor_arguments"] == (6 if observer else 7)
    assert component["covered_source_groups"] == [0, 1]
    assert joined["ordinary_joined_groups"] == {}
    assert _text(module) == before


@pytest.mark.parametrize("whole", [False, True])
def test_joint_head_join_preserves_typed_siblings_and_their_original_observer(tmp_path, monkeypatch, whole):
    from merlin_experiments.phase1.feedback.private_head_provenance import join_heads

    module, roster, emitted = routed(tmp_path, monkeypatch, observer=True, heads=True, whole=whole)
    joined = join_heads(emitted, roster, module=module, target="neutral", capture=tmp_path,
                        shared_input_siblings=True, effects=effects(), whole_batch=whole)
    head, siblings = joined["components"]
    assert head.get("whole_batch", False) is whole
    assert head["covered_source_groups"] == [0, 1]
    assert siblings["covered_source_groups"] == [2, 3]
    assert siblings["descriptor_arguments"] == 6
    assert joined["covered_source_groups"] == [0, 1, 2, 3]
    assert joined["ordinary_joined_groups"] == {}


def test_independent_join_reopens_current_capture_scales_instead_of_candidate_coefficients(tmp_path, monkeypatch):
    import numpy as np

    module, roster, emitted = routed(tmp_path, monkeypatch, captured=True)
    joined = join_components(emitted, roster, module=module, target="neutral", effects=effects(),
                             capture=tmp_path / "model.mlir")
    assert joined["components"][0]["descriptor_arguments"] == 7
    # Same source IR and emitted bytes, but the independently selected stored
    # scale changed. Candidate COMMIT coefficients cannot authorize the proof.
    path = tmp_path / "weights.safetensors"
    payload = bytearray(path.read_bytes())
    start = 8 + int.from_bytes(payload[:8], "little")
    payload[start:start + 4] = np.float32(0.5).tobytes()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="differ from original source"):
        join_components(emitted, roster, module=module, target="neutral", effects=effects(), capture=tmp_path)


@pytest.mark.parametrize("damage", ["scale", "bias", "bias_position", "output_dtype", "access", "undeclared"])
def test_typed_join_refuses_candidate_numerics_or_logical_abi_changes(tmp_path, monkeypatch, damage):
    module, roster, emitted = routed(tmp_path, monkeypatch)
    emitted = deepcopy(emitted)
    symbol, = emitted["programs"]
    commits = [c for c in emitted["programs"][symbol]["commands"] if c["opcode"] == "COMMIT"]
    if damage == "scale":
        commits[0]["attributes"]["acc_scale_by_column"][0] *= 2
    elif damage == "bias":
        commits[0]["attributes"]["bias"] = "bias_1"
    elif damage == "bias_position":
        emitted["call_buffers"][symbol][3:5] = reversed(emitted["call_buffers"][symbol][3:5])
    elif damage == "output_dtype":
        emitted["call_buffers"][symbol][-1]["dtype"] = "i32"
    elif damage == "access":
        emitted["arg_access"][symbol][3] = "write"
    else:
        monkeypatch.setattr(OF, "declared_path", lambda _: None)
    with pytest.raises(ValueError, match="differ from original source"):
        join_components(emitted, roster, module=module, target="neutral", effects=effects(), capture=tmp_path)


def test_borrowed_chain_uses_independently_proved_bias_descriptor_positions():
    from xdsl.context import Context
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import Builtin
    from xdsl.parser import Parser

    from merlin.targetgen.private_sibling_dispatch import borrowed_host_chain

    types = ",".join(["!llvm.ptr"] * 7)
    arguments = ",".join(f"%a{i}:!llvm.ptr" for i in range(7))
    values = ",".join(f"%a{i}" for i in range(7))
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(llvm.LLVM)
    module = Parser(context, f"""builtin.module {{
      llvm.func @_mlir_ciface_borrowed({types})
      llvm.func @borrowed({arguments}) {{
        llvm.call @_mlir_ciface_borrowed({values}):({types})->()
        llvm.return
      }}
      llvm.func @writer({arguments}) {{
        llvm.call @borrowed({values}):({types})->()
        llvm.return
      }}
      llvm.func @main({arguments}) {{
        llvm.call @writer({values}):({types})->()
        llvm.return
      }}
    }}""").parse_module()
    module.verify()
    main = next(op for op in module.body.block.ops if op.sym_name.data == "main")
    composition = dict(borrowed_entry_symbol="borrowed", host_entry_symbol="writer", members=[{}, {}],
                       descriptor_arguments=7, physical_symbol="device")
    result = borrowed_host_chain(module, main, composition, mandatory=lambda _fn, calls: bool(calls))
    assert result["descriptor_arguments"] == 7
    assert result["chain"] == ["main", "writer", "borrowed", "_mlir_ciface_borrowed"]
    with pytest.raises(ValueError, match="pointer-only multi-output ABI"):
        borrowed_host_chain(module, main, {**composition, "descriptor_arguments": 5},
                            mandatory=lambda _fn, calls: bool(calls))
