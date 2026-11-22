"""Graph requirements and provider claims preserve explicit numerical authority."""

from dataclasses import replace

import pytest

from merlin.compiler.backend.interface import (
    CompilerDescription,
    ImplementationFamily,
    SourceDomain,
    SourcePattern,
    TargetPipeline,
)
from merlin.compiler.quantization.policy_generation import (
    NumericalFamily,
    PolicyGenerationError,
    generate_policy,
)
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType


def graph(*, name="application", reverse=False, operation="contract", operand_policy="codes"):
    lhs = SemanticNode("lhs", "input", (), TensorType((3, 5), "i8", operand_policy), effect="input")
    rhs = SemanticNode("rhs", "input", (), TensorType((7, 5), "i8", "codes"), effect="input")
    result = SemanticNode(name, operation, ("rhs", "lhs") if reverse else ("lhs", "rhs"),
                          TensorType((3, 7), "bf16", "readout"), (("rounding", "rne"),))
    return KernelRequest((lhs, rhs, result), (name,), ("memory",),
                         (("lhs", "memory"), ("rhs", "memory")), "declared-device")


def description(*families):
    return CompilerDescription(
        "provider", SourceDomain("semantic", ("contract", "epilogue"), ("i8", "bf16")),
        (SourcePattern("contract-pattern", ("contract",), "lowering", 1),
         SourcePattern("epilogue-pattern", ("epilogue",), "lowering", 1)),
        families or (ImplementationFamily("contraction", ("contract-pattern",), "lowering"),
                     ImplementationFamily("epilogue", ("epilogue-pattern",), "lowering")),
        (TargetPipeline("lowering", "semantic", "device-ir", (("object", "executable"),),
                        "declared-abi", ("legalize",)),),
    )


def contract(**changes):
    return replace(NumericalFamily(
        "contraction", "contract-pattern", ("codes", "readout"), "ordered-readout",
        constraints=((("operands", "0", "shape"), "[3,5]"),),
        policy_fields=(("contraction", '{"policy":"ordered-readout"}'),),
    ), **changes)


def generate(source, declarations=None, compiler=None, **options):
    return generate_policy(source, compiler or description(), declarations or (contract(),),
                           original_source_sha256="a" * 64, preparation_sha256="b" * 64,
                           **options)


def test_shape_and_order_drive_selection_while_source_names_only_bind_results():
    selected = generate(graph())
    renamed = generate(graph(name="unrelated_model_projection"))
    assert selected["requirements"][0]["result"]["shape"] == [3, 7]
    assert [operand["shape"] for operand in selected["requirements"][0]["operands"]] == [[3, 5], [7, 5]]
    assert selected["requirements"][0]["inputs"] == ["lhs", "rhs"]
    assert selected["requirements"][0]["operand_count"] == 2
    assert selected["requirements"][0]["operands"][0]["producer"]["operation"] == "input"
    exclusion = ((("operand_count",), "2"), (("result", "dtype"), '"bf16"'))
    with pytest.raises(PolicyGenerationError):
        generate(graph(), (contract(excluded_constraints=exclusion),))
    # Exclusions are a conjunction, not independent prohibitions.
    generate(graph(), (contract(excluded_constraints=exclusion[:-1] + ((("result", "dtype"), '"f32"'),)),))
    generate(graph(), (contract(constraints=((("operand_count",), "2"),)),))
    with pytest.raises(PolicyGenerationError):
        generate(graph(), (contract(constraints=((("operand_count",), "3"),)),))
    assert [operand["effect"] for operand in selected["requirements"][0]["operands"]] == ["input", "input"]
    assert selected["compatibility_policy_fields"] == {"contraction": {"policy": "ordered-readout"}}
    assert renamed["selected_families"][0]["family"] == selected["selected_families"][0]["family"]
    assert renamed["graph_sha256"] != selected["graph_sha256"]
    assert renamed["source_sha256"] == "a" * 64
    source = graph()
    renamed_node = replace(source.nodes[-1], attrs=(*source.nodes[-1].attrs, ("prov.site_id", "other")))
    provenance = generate(replace(source, nodes=(*source.nodes[:-1], renamed_node)))
    assert provenance["requirements"] == selected["requirements"]
    assert provenance["graph_sha256"] != selected["graph_sha256"]
    with pytest.raises(PolicyGenerationError):
        generate(graph(reverse=True))


def test_every_compiled_operation_and_operand_policy_needs_a_declared_match():
    with pytest.raises(PolicyGenerationError):
        generate(graph(operation="undeclared-operation"))
    with pytest.raises(PolicyGenerationError):
        generate(graph(operand_policy="different-encoding"))
    # SourceDomain.types is an admission boundary even when a numerical row
    # happens to accept every policy string. Check all source value roles.
    compiler = description()
    compiler = replace(compiler, source_domain=replace(compiler.source_domain, types=("bf16",)))
    with pytest.raises(PolicyGenerationError, match="type is outside"):
        generate(graph(), compiler=compiler)
    source = graph()
    unused = SemanticNode("unused", "undeclared-operation", (source.outputs[0],), source.nodes[-1].type)
    with pytest.raises(PolicyGenerationError):
        generate(replace(source, nodes=source.nodes + (unused,)))


def test_precision_changes_need_authority_and_competing_claims_refuse():
    changed = contract(precision_change="selected-rounding-change")
    with pytest.raises(PolicyGenerationError):
        generate(graph(), (changed,))
    # Selecting an implementation is independent of numerical authorization.
    with pytest.raises(PolicyGenerationError):
        generate(graph(), (changed,), selected_family_ids=("contraction",))
    selected = generate(graph(), (changed,), authorized_precision_changes=("selected-rounding-change",))
    assert selected["selected_families"][0]["precision_change"] == "selected-rounding-change"
    alternative = ImplementationFamily("alternative", ("contract-pattern",), "lowering")
    compiler = description(*description().families, alternative)
    with pytest.raises(PolicyGenerationError):
        generate(graph(), (contract(), replace(contract(), family="alternative")), compiler)
    declared = contract(numerical_contract=(("readout", '{"rounding":"rne"}'),))
    selected = generate(graph(), (declared, replace(declared, family="alternative")), compiler,
                        selected_family_ids=("alternative",))
    assert selected["selected_family_ids"] == ["alternative"]
    assert selected["selected_families"][0]["family"] == "alternative"
    assert selected["selected_families"][0]["numerical_contract"] == {"readout": {"rounding": "rne"}}
    for identities in (("unknown",), (), ["alternative"], ("alternative", "alternative")):
        with pytest.raises(ValueError, match="selected families"):
            generate(graph(), compiler=compiler, selected_family_ids=identities)


def test_selected_families_cannot_publish_conflicting_authorization():
    source = graph()
    tail = SemanticNode("tail", "epilogue", source.outputs, source.nodes[-1].type)
    source = replace(source, nodes=source.nodes + (tail,), outputs=("tail",))
    epilogue = NumericalFamily("epilogue", "epilogue-pattern", ("readout",), "readout",
                               policy_fields=(("contraction", '"incompatible"'),))
    with pytest.raises(PolicyGenerationError, match="compatibility field"):
        generate(source, (contract(), epilogue))


def test_declarations_reject_name_routing_and_unbound_family_identities():
    with pytest.raises(ValueError):
        contract(constraints=((("node",), '"application"'),))
    with pytest.raises(ValueError):
        contract(constraints=((("attributes", "prov.site_id"), '"captured_projection"'),))
    with pytest.raises(ValueError):
        contract(policy_fields=(("mode", '{"value": 1}'),))
    with pytest.raises(ValueError):
        generate(graph(), (contract(family="missing-family"),))
    with pytest.raises(ValueError):
        contract(policy_fields=(["mode", "1"],))
    with pytest.raises(ValueError):
        contract(constraints=([("result", "dtype"), '"bf16"'],))
    for fields in ([('readout', '1')], (("readout", "1"), ("readout", "2")), (("readout", "NaN"),)):
        with pytest.raises(ValueError):
            contract(numerical_contract=fields)
    # Repeated indices are structural paths, not duplicate identity declarations.
    contract(constraints=((("index_maps", "0", "coefficients", "0", "0"), "1"),))


def policy_bundle(tmp_path):
    import json
    from hashlib import sha256

    export, fx, capture = (tmp_path / name for name in ("export", "fx", "capture"))
    for directory in (export, fx, capture):
        directory.mkdir()

    def write(directory, name, value):
        raw = json.dumps(value).encode()
        (directory / name).write_bytes(raw)
        return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    policy = write(export, "policy.json", {"schema": "example.execution.v1", "format_version": 1})
    manifest = write(export, "manifest.json", {
        "schema": "example.export.v1", "format_version": 1,
        "files": {"policy.json": policy}, "policy_sha256": policy["sha256"],
    })
    quantization = write(capture, "quantization-manifest.json", {
        "schema": "m2m.quantization_manifest.v2", "adapter_id": "selected-adapter",
        "contract_sha256": manifest["sha256"], "policy_sha256": policy["sha256"],
        "provider": {"saved_program": {"sha256": "c" * 64}},
    })
    write(capture, "capture_receipt.json", {
        "schema": "m2m.capture-receipt.v1", "artifacts": {"quantization-manifest.json": quantization},
    })
    write(fx, "fx_manifest.json", {
        "schema": "example.fx.v1", "provenance": {"export_manifest_sha256": manifest["sha256"]},
        "files": {"model.pt2": {"sha256": "c" * 64}},
    })
    return export, fx, capture


def authenticate(bundle):
    from merlin.compiler.frontend.policy_inputs import authenticate_policy_inputs

    return authenticate_policy_inputs(*bundle, export_manifest_schema="example.export.v1",
        export_policy_schema="example.execution.v1", fx_manifest_schema="example.fx.v1",
        adapter_id="selected-adapter", native_program_sha256_path=("provider", "saved_program", "sha256"))


def test_policy_inputs_join_export_capture_native_program_and_refuse_custody_drift(tmp_path, monkeypatch):
    import json

    bundle = policy_bundle(tmp_path)
    bound = authenticate(bundle)
    assert json.loads(bound.export_policy_json)["schema"] == "example.execution.v1"
    assert len(bound.capture_receipt_sha256) == 64
    assert len(bound.quantization_manifest_sha256) == 64
    assert len(bound.fx_manifest_sha256) == 64
    assert bound.native_program_sha256 == "c" * 64
    fx_manifest = bundle[1] / "fx_manifest.json"
    original_fx = fx_manifest.read_text()
    mismatched = json.loads(original_fx)
    mismatched["files"]["model.pt2"]["sha256"] = "d" * 64
    fx_manifest.write_text(json.dumps(mismatched))
    with pytest.raises(ValueError):
        authenticate(bundle)

    fx_manifest.write_text(original_fx)
    assert authenticate(bundle) == bound
    quantization = bundle[2] / "quantization-manifest.json"
    before = quantization.read_bytes()
    quantization.write_bytes(before + b" ")
    with pytest.raises(ValueError):
        authenticate(bundle)
    quantization.write_bytes(before)
    alias = tmp_path / "capture-alias"
    alias.symlink_to(bundle[2], target_is_directory=True)
    with pytest.raises(ValueError):
        authenticate((bundle[0], bundle[1], alias))
    _check_saved_program_signature(tmp_path, monkeypatch)


def _check_saved_program_signature(tmp_path, monkeypatch):
    import json
    import zipfile
    from copy import deepcopy
    from hashlib import sha256

    from merlin.compiler.frontend import saved_program_signature as reader

    signature = {
        "input_specs": [
            {"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "layer.weight"}},
            {"buffer": {"arg": {"name": "b_state"}, "buffer_name": "layer.state", "persistent": True}},
            {"user_input": {"arg": {"as_tensor": {"name": "x"}}}},
        ],
        "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "result"}}}}],
    }
    document = {"schema_version": {"major": 8, "minor": 17}, "graph_module": {"signature": signature}}
    archive = tmp_path / "saved-program.pt2"

    def write(value, *, second_model=False):
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            raw = value if isinstance(value, bytes) else json.dumps(value).encode()
            output.writestr("bundle/models/program.json", raw)
            if second_model:
                output.writestr("bundle/models/other.json", raw)
        return sha256(archive.read_bytes()).hexdigest()

    digest = write(document)
    checked = reader.read_saved_program_signature(archive, expected_sha256=digest)
    assert checked.buffers == (reader.SavedBufferSignature("b_state", "layer.state", False),)
    assert checked.buffer_mutations == checked.parameter_mutations == checked.user_input_mutations == ()
    assert checked.archive_sha256 == digest and checked.schema_version == (8, 17)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        reader.read_saved_program_signature(archive, expected_sha256="0" * 64)

    mutated = deepcopy(document)
    mutated["graph_module"]["signature"]["output_specs"].append(
        {"buffer_mutation": {"arg": {"name": "updated"}, "buffer_name": "layer.state"}}
    )
    changed = reader.read_saved_program_signature(archive, expected_sha256=write(mutated))
    assert changed.buffers == (reader.SavedBufferSignature("b_state", "layer.state", True),)
    assert changed.buffer_mutations == ("layer.state",)

    duplicate = deepcopy(document)
    duplicate["graph_module"]["signature"]["input_specs"].append(deepcopy(signature["input_specs"][1]))
    with pytest.raises(ValueError, match="duplicate"):
        reader.read_saved_program_signature(archive, expected_sha256=write(duplicate))
    unrecognized = deepcopy(document)
    unrecognized["graph_module"]["signature"]["output_specs"].append({"unreviewed_mutation": {"target": "layer.state"}})
    with pytest.raises(ValueError, match="unrecognized output signature variant"):
        reader.read_saved_program_signature(archive, expected_sha256=write(unrecognized))
    with pytest.raises(ValueError, match="unreviewed schema_version"):
        reader.read_saved_program_signature(
            archive, expected_sha256=write({**document, "schema_version": {"major": 8, "minor": 18}})
        )
    with pytest.raises(ValueError, match="exactly one models JSON member"):
        reader.read_saved_program_signature(archive, expected_sha256=write(document, second_model=True))
    duplicate_json = json.dumps(document).replace('"major": 8', '"major": 8, "major": 8').encode()
    with pytest.raises(ValueError, match="duplicate JSON field"):
        reader.read_saved_program_signature(archive, expected_sha256=write(duplicate_json))
    with pytest.raises(ValueError, match="byte budget"):
        reader.read_saved_program_signature(archive, expected_sha256=write(b" " * (16 * 1024 * 1024 + 1)))

    # Changing the already-open archive during inspection must revoke custody.
    original_read = reader._archive_signature

    def read_then_change(stream):
        result = original_read(stream)
        with archive.open("ab") as output:
            output.write(b"changed")
        return result

    digest = write(document)
    with monkeypatch.context() as patch:
        patch.setattr(reader, "_archive_signature", read_then_change)
        with pytest.raises(ValueError, match="changed while reading"):
            reader.read_saved_program_signature(archive, expected_sha256=digest)


def captured_policy_bundle(tmp_path, *, bind_quantization=True):
    import json
    from hashlib import sha256

    from corefixtures.compile_input_fixtures import SOURCE, _write_capture

    export, fx, capture = policy_bundle(tmp_path)
    quantization = (capture / "quantization-manifest.json").read_bytes()
    canonical = json.dumps(json.loads(quantization), sort_keys=True, separators=(",", ":"))
    digest = sha256(canonical.encode()).hexdigest() if bind_quantization else "0" * 64
    source = SOURCE.replace("func.return %a : tensor<2xf32>",
        '%r = "fixture.identity"(%a) : (tensor<2xf32>) -> tensor<2xf32>\n'
        '    func.return %r : tensor<2xf32>').replace('prov.weights_file =',
        f'prov.quantization_manifest_sha256 = "{digest}", prov.weights_file =')
    native = tmp_path / "native-capture"
    _write_capture(native, source=source)
    (native / "quantization-manifest.json").write_bytes(quantization)
    receipt = json.loads((native / "capture_receipt.json").read_text())
    receipt["artifacts"]["quantization-manifest.json"] = {
        "bytes": len(quantization), "sha256": sha256(quantization).hexdigest()}
    (native / "capture_receipt.json").write_text(json.dumps(receipt))
    return export, fx, native


def test_policy_staging_binds_receipt_and_ir_before_generating_claims(tmp_path):
    import json

    from merlin.compiler.frontend.policy_inputs import stage_policy_inputs

    bundle = captured_policy_bundle(tmp_path)
    bound = authenticate(bundle)
    with stage_policy_inputs(bound, bundle[2]) as (directory, staged):
        assert directory.is_dir()
        assert staged.precision_policy is None
    assert not directory.exists()
    from merlin.compiler.quantization.policy_generation import generate_capture_policy

    def translate(op, operands, results):
        return (SemanticNode(results[0], "contract", operands,
                             TensorType((2,), "f32", "raw-bit-pattern")),)

    compiler = description()
    compiler = replace(compiler, source_domain=replace(compiler.source_domain, types=("f32",)))
    declaration = contract(source_policies=("raw-bit-pattern",), constraints=())
    generated = generate_capture_policy(bound, bundle[2], compiler, (declaration,),
        target_identity="declared-device", source_operation_translators={"fixture.identity": translate})
    assert generated["source_sha256"] == staged.original_source_sha256
    assert generated["capture_program_sha256"] == staged.program_sha256
    preparation = generated["source_preparation"]
    assert generated["translated_graph_sha256"] == preparation["before_digest"]
    assert generated["graph_sha256"] == preparation["after_digest"]
    from merlin.compiler.quantization.policy_generation import _digest
    assert generated["preparation_sha256"] == _digest({
        "program_sha256": staged.program_sha256, "source_preparation": preparation,
        "authenticated_inputs": generated["authenticated_inputs"]})
    assert generated["authenticated_inputs"]["capture_receipt_sha256"] == bound.capture_receipt_sha256
    assert generated["authenticated_inputs"]["fx_manifest_sha256"] == bound.fx_manifest_sha256
    assert generated["selected_families"][0]["family"] == "contraction"
    assert generated["compatibility_policy_fields"] == {"contraction": {"policy": "ordered-readout"}}
    external = generate_capture_policy(bound, bundle[2], compiler, (declaration,),
        target_identity="declared-device", source_operation_translators={"fixture.identity": translate},
        bind_external_parameters=True)
    assert external["parameter_mode"] == "external"
    assert external["graph_sha256"] != generated["graph_sha256"]
    assert external["authenticated_inputs"] == generated["authenticated_inputs"]
    receipt_path = bundle[2] / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["changed_after_authentication"] = True
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt changed"):
        with stage_policy_inputs(bound, bundle[2]):
            pytest.fail("changed receipt admitted")
    rebound = authenticate(bundle)
    regenerated = generate_capture_policy(rebound, bundle[2], compiler, (declaration,),
        target_identity="declared-device", source_operation_translators={"fixture.identity": translate})
    assert regenerated["graph_sha256"] == generated["graph_sha256"]
    assert regenerated["preparation_sha256"] != generated["preparation_sha256"]
    assert regenerated["authenticated_inputs"]["capture_receipt_sha256"] == rebound.capture_receipt_sha256


def test_policy_staging_refuses_self_consistent_capture_with_wrong_export_join(tmp_path):
    from merlin.compiler.frontend.policy_inputs import stage_policy_inputs

    bundle = captured_policy_bundle(tmp_path, bind_quantization=False)
    bound = authenticate(bundle)
    with pytest.raises(ValueError, match="IR does not bind"):
        with stage_policy_inputs(bound, bundle[2]):
            pytest.fail("unbound captured IR admitted")


def test_signless_byte_staging_preserves_u8_storage_without_numeric_conversion(tmp_path):
    import json
    import struct
    from hashlib import sha256

    from corefixtures.compile_input_fixtures import _write_weighted_capture

    from merlin.compiler.frontend.inputs import (
        load_compile_inputs,
        load_lifted_argument_bytes,
        stage_compile_inputs,
    )

    capture, directory = tmp_path / "capture", tmp_path / "staged"
    _write_weighted_capture(capture)
    source = (capture / "model.mlir").read_text().replace(
        "%weight: tensor<2x2xf32>", "%weight: tensor<2x2xi8>")
    (capture / "model.mlir").write_text(source)
    trace = json.loads((capture / "frontend-trace.json").read_text())
    trace["mlir"]["sha256"] = sha256(source.encode()).hexdigest()
    (capture / "frontend-trace.json").write_text(json.dumps(trace))
    metadata = json.loads((capture / "meta.json").read_text())
    metadata["frontend_trace"]["sha256"] = sha256((capture / "frontend-trace.json").read_bytes()).hexdigest()
    (capture / "meta.json").write_text(json.dumps(metadata))
    manifest = json.loads((capture / "weights.safetensors.manifest.json").read_text())
    manifest["0"]["dtype"] = "uint8"
    (capture / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    payload = bytes((0, 127, 128, 255))
    header = json.dumps({"linear.weight": {"dtype": "U8", "shape": [2, 2], "data_offsets": [0, 4]}}).encode()
    (capture / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for name in receipt["artifacts"]:
        raw = (capture / name).read_bytes()
        receipt["artifacts"][name] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    receipt_path.write_text(json.dumps(receipt))
    stage_compile_inputs(capture, directory)
    staged = load_compile_inputs(directory)
    assert staged.argument_bindings[0]["dtype"] == "i8"
    assert load_lifted_argument_bytes(directory, staged, kinds=("param",), max_bytes=4) == {0: payload}
