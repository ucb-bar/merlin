"""Graph requirements and provider claims preserve explicit numerical authority."""

from dataclasses import replace

import pytest
from corefixtures.row_table import run_rows

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


def _shape_order(directory, patch):
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


def _declared_match(directory, patch):
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


def _precision_authority(directory, patch):
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


def _conflicting_fields(directory, patch):
    source = graph()
    tail = SemanticNode("tail", "epilogue", source.outputs, source.nodes[-1].type)
    source = replace(source, nodes=source.nodes + (tail,), outputs=("tail",))
    epilogue = NumericalFamily("epilogue", "epilogue-pattern", ("readout",), "readout",
                               policy_fields=(("contraction", '"incompatible"'),))
    with pytest.raises(PolicyGenerationError, match="compatibility field"):
        generate(source, (contract(), epilogue))


def _declaration_refusals(directory, patch):
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


def _selected_targets(directory, patch):
    # A device and a host description: each requirement claims a family of its own target.
    source = graph()
    widen = SemanticNode("widen", "widen", source.outputs, TensorType((3, 7), "f32", "readout"))
    source = replace(source, nodes=source.nodes + (widen,), outputs=("widen",))
    host = CompilerDescription(
        "host-provider", SourceDomain("semantic", ("widen",), ("bf16", "f32")),
        (SourcePattern("widen-pattern", ("widen",), "host-c", 1),),
        (ImplementationFamily("widen", ("widen-pattern",), "host-c"),),
        (TargetPipeline("host-c", "semantic", "c", (("c-source", "wrapper"),), "host-abi", ("emit",)),),
    )
    targets = (description(), host)
    rows = (contract(), NumericalFamily("widen", "widen-pattern", ("readout",), "widen-exact"))
    selected = generate(source, rows, targets, selected_family_ids=("contraction", "widen"))
    assert [(row["family"], row["pipeline"]) for row in selected["selected_families"]] == [
        ("contraction", "lowering"), ("widen", "host-c")]
    import hashlib
    import json
    records = json.dumps([row.record() for row in targets], sort_keys=True, separators=(",", ":"))
    assert selected["compiler_description_sha256"] == hashlib.sha256(records.encode()).hexdigest()
    assert generate(graph())["compiler_description_sha256"] != selected["compiler_description_sha256"]
    # Without the host description its operation and type are undeclared.
    with pytest.raises((ValueError, PolicyGenerationError)):
        generate(source, rows[:1], description())
    # A family cannot borrow another target's types: the device alone refuses i8 operands, and so does it beside
    # an unrelated target that declares i8.
    device = description()
    device = replace(device, source_domain=replace(device.source_domain, types=("bf16",)))
    with pytest.raises(PolicyGenerationError, match="type is outside"):
        generate(graph(), None, device)
    other = CompilerDescription(
        "other-provider", SourceDomain("semantic", ("load",), ("i8",)),
        (SourcePattern("load-pattern", ("load",), "other-c", 1),),
        (ImplementationFamily("load", ("load-pattern",), "other-c"),),
        (TargetPipeline("other-c", "semantic", "c", (("c-source", "wrapper"),), "other-abi", ("emit",)),),
    )
    with pytest.raises(PolicyGenerationError, match="requires one numerical family; found 0"):
        generate(graph(), None, (device, other))
    # Family and pattern identities must stay distinct across the selected targets, each on its own.
    same_family = replace(other, families=(ImplementationFamily("contraction", ("load-pattern",), "other-c"),))
    same_pattern = replace(other, patterns=(SourcePattern("contract-pattern", ("load",), "other-c", 1),),
                           families=(ImplementationFamily("load", ("contract-pattern",), "other-c"),))
    for repeated in (same_family, same_pattern):
        with pytest.raises(ValueError, match="repeat a family or pattern"):
            generate(graph(), None, (description(), repeated))


def _host_target(directory, patch):
    from merlin.compiler.backend.protocol import LoweringRecipe
    from merlin.compiler.lower.c import host_target

    compiler, rows = host_target.describe_compiler(), host_target.describe_numerical_families()
    assert compiler.provider == "host" and {row.family for row in rows} == {family.id for family in compiler.families}
    nodes = tuple(SemanticNode(key, "input", (), TensorType((4,), "f32", "raw-bit-pattern"), effect="input")
                  for key in ("x", "y"))
    add = SemanticNode("sum", "f32_add", ("x", "y"), TensorType((4,), "f32", "raw-bit-pattern"),
                       (("numeric_semantics", "ieee754-f32-add-rne-v1"),))
    source = KernelRequest((*nodes, add), ("sum",), ("memory",), (("x", "memory"), ("y", "memory")), "host")
    claim = generate(source, rows, compiler)["selected_families"][0]
    assert (claim["family"], claim["pipeline"], claim["precision_change"]) == ("host.f32-add", "host.c", None)
    assert claim["implementation_policy"] == "ieee754-f32-add-rne-v1"
    # A recipe names a declared family and pins the emitter sources it was made from.
    helper = directory / "emitter.py"
    helper.write_text("version = 1\n")
    patch.setattr(host_target, "_source_roots", lambda: (("host-c", directory, (".py",)),))
    recipe = host_target.lowering_recipe((("family", "host.f32-add"),))
    host_target.validate_recipe(recipe)
    with pytest.raises(ValueError, match="no declared host family"):
        host_target.lowering_recipe((("family", "host.undeclared"),))
    with pytest.raises(ValueError, match="differs from the declared host families"):
        host_target.validate_recipe(LoweringRecipe(recipe.implementation_sha256, recipe.pipeline,
                                                   (("family", "host.undeclared"),), recipe.artifacts, recipe.abi))
    helper.write_text("version = 2\n")
    with pytest.raises(ValueError, match="implementation sources changed"):
        host_target.validate_recipe(recipe)


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


def _authenticated_join(tmp_path, patch):
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


def _saved_program_reader(tmp_path, monkeypatch):
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
    monkeypatch.setattr(reader, "_archive_signature", read_then_change)
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


def _staging_binding(tmp_path, patch):
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


def _export_join(tmp_path, patch):
    from merlin.compiler.frontend.policy_inputs import stage_policy_inputs

    bundle = captured_policy_bundle(tmp_path, bind_quantization=False)
    bound = authenticate(bundle)
    with pytest.raises(ValueError, match="IR does not bind"):
        with stage_policy_inputs(bound, bundle[2]):
            pytest.fail("unbound captured IR admitted")


def test_generated_policy_admits_only_declared_claims(tmp_path, monkeypatch):
    run_rows(tmp_path, monkeypatch, {
        "shape_order": _shape_order,
        "declared_match": _declared_match,
        "precision_authority": _precision_authority,
        "conflicting_fields": _conflicting_fields,
        "declaration_refusals": _declaration_refusals,
        "selected_targets": _selected_targets,
        "host_target": _host_target,
    })


def test_policy_inputs_refuse_custody_drift(tmp_path, monkeypatch):
    run_rows(tmp_path, monkeypatch, {
        "authenticated_join": _authenticated_join,
        "saved_program_reader": _saved_program_reader,
        "staging_binding": _staging_binding,
        "export_join": _export_join,
    })
