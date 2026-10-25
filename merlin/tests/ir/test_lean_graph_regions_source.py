"""Retention-all qualification of the explicit frozen source operational bridge."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from xdsl.dialects import func

from merlin.common.paths import artifacts_dir, build_dir, repo_root
from merlin.verify import lean_graph_regions as base
from merlin.verify import lean_graph_regions_capture as producer
from merlin.verify import lean_graph_regions_mlir as compiler
from merlin.verify import lean_graph_regions_projection as projection
from merlin.verify import lean_graph_regions_source as source
from merlin.verify import lean_graph_regions_source_build as builder
from merlin.verify import lean_graph_regions_source_capture as variants


@pytest.fixture(scope="module")
def packet():
    name = os.environ.get("MERLIN_GRAPH_SOURCE_PROFILE", "development-" + uuid.uuid4().hex)
    output = artifacts_dir() / "verification/graph-regions-source" / name
    output.mkdir(parents=True, exist_ok=False)
    build = build_dir() / "graph-regions-source" / name
    builder.build(build, ["Merlin.GraphRegionsTensorWitness"])
    reference = Path("/home/jeremy/Projects/merlin-graph-proof-intake")
    oracle = Path("/home/jeremy/Projects/model2MLIR/.venv/bin/python")
    observed = producer.capture(reference, output / "observed")
    escaped = producer.capture(reference, output / "escaped", realize=True)
    native = producer.capture(reference, output / "native", observed=False, realize=True, suffix_observer=True)
    independent_input = output / "independent-input.mlir"
    independent_input.write_text(producer._source((2, 11), True).replace("i16", "i32"))
    independent = variants.capture(independent_input, reference, output / "independent")
    module = projection.parse_source(Path(native["source"]["path"]).read_text())
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    generics = [op for op in function.walk() if op.name == "linalg.generic"]
    for generic in generics:
        for op in tuple(generic.body.block.ops):
            if op.name == "arith.constant":
                generic.body.block.detach_op(op)
                function.body.block.insert_op_before(op, generics[0])
    module.verify()
    captured_input = output / "captured-input.mlir"
    captured_input.write_text(projection.ir_text(module))
    captured = variants.capture(captured_input, reference, output / "captured")
    scopes = dict(observed=observed, escaped=escaped, native=native, independent=independent, captured=captured)
    receipts = {}
    for kind, scope in scopes.items():
        directory = output / kind
        plan, fused, selection = None, None, None
        if kind != "captured":
            plan = directory / "plan.json"
            if kind in ("independent",):
                projected, _, _ = projection.project(projection.parse_source((directory / "source.mlir").read_text()))
                selection = [n["ordinal"] for n in projected["nodes"]][:3]
            else:
                selection = scope["selected_ordinals"]
            fused = directory / "fused-body.json"
            if not fused.exists():
                projected, _, _ = projection.project(projection.parse_source((directory / "source.mlir").read_text()))
                data = json.loads(plan.read_text())
                # Proposed logical composition is checked by Lean. This is NOT
                # fabricated evidence of an actual admitted native product.
                source.save(
                    fused,
                    dict(
                        body=[n for n in projected["nodes"] if n["ordinal"] in selection],
                        inputs=[data["values"][i] for i in data["inputs"]],
                        outputs=[data["values"][i] for i in data["outputs"]],
                    ),
                )
        origin = directory / ("PROBE.json" if kind in ("independent", "captured") else "CAPTURE.json")
        receipts[kind] = source.verify(
            directory / "source.mlir",
            directory / "retained-source.mlir",
            output / ("checked-" + kind),
            identity=build / "IDENTITY.json",
            oracle=oracle,
            plan=plan,
            fused=fused,
            selection=selection,
            witness=[3, 7],
            origin=origin,
        )
    return dict(output=output, identity=build / "IDENTITY.json", oracle=oracle, scopes=scopes, receipts=receipts)


@pytest.mark.parametrize("kind", ["observed", "escaped", "native", "independent", "captured"])
def test_complete_actual_source_execution(packet, kind):
    result = packet["receipts"][kind]
    assert result["source_accepted"] is True
    assert result["logical_program_correspondence"] is (kind != "captured")
    assert result["claims"]["native_implementation_contract_proved"] is False
    assert "witnessExecution" in result["theorem_audit"]
    assert "witnessPoints" in result["theorem_audit"]


def test_actual_production_domain_distinctions(packet):
    scopes = packet["scopes"]
    assert scopes["escaped"]["realization"]["status"] != "applied"
    assert scopes["native"]["realization"]["status"] == "applied"
    assert scopes["captured"]["connected_plan"]["status"] == "refused"
    assert scopes["captured"]["multioutput_source_reader"]["status"] == "accepted"
    assert scopes["independent"]["connected_plan"]["status"] == "accepted"
    producer.reproduce_native(scopes["native"], packet["output"] / "native-fresh-reproduction")


def mutation(text, fault):
    if fault == "changed-body":
        return text.replace('"arith.addi"', '"arith.subi"', 1)
    if fault == "overflow-flag":
        return text.replace("#arith.overflow<none>", "#arith.overflow<nsw>", 1)
    if fault == "cast-width":
        return text.replace("i8", "i4")
    if fault == "changed-map":
        return text.replace("-> (d0, d1)>", "-> (d1, d0)>")
    if fault == "reduction-iterator":
        return """builtin.module {func.func @forward(%x:tensor<3x5xi16>,%y:tensor<3x5xi16>) -> tensor<3xi16> {
          %e=tensor.empty():tensor<3xi16>
          %r="linalg.generic"(%x,%e) <{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,
            affine_map<(d0,d1)->(d0)>],iterator_types=[#linalg.iterator_type<parallel>,
            #linalg.iterator_type<reduction>],operandSegmentSizes=array<i32:1,1>}> ({
            ^bb0(%a:i16,%old:i16): %v=arith.addi %a,%old:i16
            "linalg.yield"(%v):(i16)->() }):(tensor<3x5xi16>,tensor<3xi16>)->tensor<3xi16>
          func.return %r:tensor<3xi16> }}"""
    module = projection.parse_source(text)
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    generics = [op for op in function.walk() if op.name == "linalg.generic"]
    term = function.body.block.last_op
    if fault in ("read-initializer", "yield-initializer"):
        generic = generics[0]
        if fault == "yield-initializer":
            generic.body.block.last_op.operands = (generic.body.block.args[-1],)
        else:
            op = generic.body.block.first_op
            op.operands = (op.operands[0], generic.body.block.args[-1])
    elif fault == "swap-input":
        op = generics[0]
        op.operands = (op.operands[1], op.operands[0], *op.operands[2:])
    elif fault == "swap-result":
        term.operands = (term.operands[0], term.operands[2], term.operands[1])
    elif fault == "swap-yield":
        generics[2].body.block.last_op.operands = (generics[2].body.block.args[0],)
    elif fault == "suffix-binding":
        op = generics[-1]
        op.operands = (generics[0].results[0], *op.operands[1:])
    elif fault in ("drop-second-output", "drop-external-output"):
        index = 1 if fault == "drop-second-output" else 0
        term.operands = tuple(value for i, value in enumerate(term.operands) if i != index)
        from xdsl.dialects.builtin import FunctionType

        function.properties["function_type"] = FunctionType.from_lists(
            function.function_type.inputs.data, [v.type for v in term.operands]
        )
    elif fault == "captured-literal":
        from xdsl.dialects.builtin import IntegerAttr, i16

        op = next(op for op in function.body.block.ops if op.name == "arith.constant")
        op.properties["value"] = IntegerAttr(-127, i16)
    else:
        raise ValueError(fault)
    module.verify()
    return projection.ir_text(module)


@pytest.mark.parametrize(
    "fault",
    [
        "read-initializer",
        "yield-initializer",
        "drop-second-output",
        "drop-external-output",
        "swap-input",
        "swap-result",
        "swap-yield",
        "changed-body",
        "cast-width",
        "changed-map",
        "reduction-iterator",
        "overflow-flag",
        "suffix-binding",
        "captured-literal",
    ],
)
def test_compiler_valid_mutations_rejected_by_lean(packet, fault):
    kind = "captured" if fault == "captured-literal" else "observed" if fault == "drop-external-output" else "native"
    original = packet["output"] / kind / "source.mlir"
    mutated = packet["output"] / (fault + ".mlir")
    mutated.write_text(mutation(original.read_text(), fault))
    static = fault in ("read-initializer", "yield-initializer")
    before = mutated if static else original
    result = source.verify(
        before,
        mutated,
        packet["output"] / ("refused-" + fault),
        identity=packet["identity"],
        oracle=packet["oracle"],
        expected=False,
        refusal="static" if static else "after",
    )
    assert result["source_accepted"] is False


@pytest.mark.parametrize("fault", ["semantic-key", "graph-binding", "capture-binding", "cast-model"])
def test_untrusted_model_and_graph_hints_are_not_proof_facts(packet, fault):
    kind = "captured" if fault == "capture-binding" else "native"
    checked = packet["output"] / ("checked-" + kind)
    model = json.loads((checked / "model.json").read_text())
    graph = json.loads((checked / "graph.json").read_text())
    generic = next(item for item in model["items"] if item["kind"] == "generic")
    if fault == "semantic-key":
        generic["key"] = "0" * 64
    elif fault == "graph-binding":
        graph["nodes"][1]["operands"] = graph["nodes"][0]["operands"][:1]
    elif fault == "capture-binding":
        generic = next(
            item for item in model["items"] if item["kind"] == "generic" and item["stage"]["body"]["captures"]
        )
        generic["stage"]["body"]["captures"][0]["literal"] += 1
    else:
        generic = next(
            item
            for item in model["items"]
            if item["kind"] == "generic"
            and any(i["code"]["kind"] == "cast" for i in item["stage"]["body"]["instructions"])
        )
        code = next(i["code"] for i in generic["stage"]["body"]["instructions"] if i["code"]["kind"] == "cast")
        code["output"] = code["input"]
    model_path, graph_path = packet["output"] / (fault + "-model.json"), packet["output"] / (fault + "-graph.json")
    source.save(model_path, model)
    source.save(graph_path, graph)
    result = source.verify(
        packet["output"] / kind / "source.mlir",
        packet["output"] / kind / "retained-source.mlir",
        packet["output"] / ("refused-" + fault),
        identity=packet["identity"],
        oracle=packet["oracle"],
        expected=False,
        refusal="static" if fault in ("capture-binding", "cast-model") else "graph",
        proposed=model_path,
        graph_input=graph_path,
    )
    assert result["source_accepted"] is False


@pytest.mark.parametrize("fault", ["second-output", "external-output", "second-consumer", "same-typed-input"])
def test_logical_realization_mutations(packet, fault):
    kind = "observed" if fault == "external-output" else "native"
    directory = packet["output"] / kind
    data = json.loads((directory / "fused-body.json").read_text())
    if fault in ("second-output", "external-output"):
        data["outputs"].pop(0 if fault == "external-output" else -1)
    elif fault == "second-consumer":
        data["body"].pop()
    else:
        data["body"][1]["operands"] = data["body"][0]["operands"][:1]
    path = packet["output"] / (fault + "-fused.json")
    source.save(path, data)
    receipt = packet["receipts"][kind]
    result = source.verify(
        directory / "source.mlir",
        directory / "retained-source.mlir",
        packet["output"] / ("refused-logical-" + fault),
        identity=packet["identity"],
        oracle=packet["oracle"],
        plan=directory / "plan.json",
        fused=path,
        selection=receipt["parameters"]["selection"],
        graph_expected=False,
    )
    assert result["source_accepted"] is True
    assert result["graph_accepted"] is False
    assert result["logical_program_correspondence"] is False


def test_compiler_invalid_ssa_is_a_separate_refusal(packet):
    text = (packet["output"] / "native/source.mlir").read_text()
    path = packet["output"] / "invalid-ssa.mlir"
    path.write_text(text.replace('"linalg.generic"(%r0,', '"linalg.generic"(%undefined,', 1))
    result = compiler.compiler_verify(path, packet["output"] / "invalid-ssa", python=packet["oracle"])
    assert result["oracle"]["valid"] is False


@pytest.mark.parametrize("role", ["before", "after", "proposed"])
@pytest.mark.parametrize("fault", ["substituted", "missing", "boolean-size"])
def test_strict_original_snapshot_bindings(packet, role, fault):
    result = copy.deepcopy(packet["receipts"]["native"])
    if role == "proposed":
        result = json.loads((packet["output"] / "refused-cast-model/EVIDENCE.json").read_text())
    if fault == "substituted":
        result["originals"][role] = base.file_ref(packet["output"] / "independent/source.mlir")
    elif fault == "missing":
        result["originals"][role]["path"] += ".missing"
    else:
        result["originals"][role]["bytes"] = True
    with pytest.raises((ValueError, OSError)):
        source.validate(result)
    source.save(packet["output"] / f"binding-refusal-{role}-{fault}.json", result)


@pytest.mark.parametrize("fault", ["stale-proof", "stale-audit", "stale-model", "tool", "host", "claim"])
def test_stale_and_mislabelled_evidence_is_refused(packet, fault):
    result = copy.deepcopy(packet["receipts"]["native"])
    if fault.startswith("stale-"):
        result[fault.removeprefix("stale-")]["sha256"] = "0" * 64
    elif fault == "tool":
        result["oracle"] = base.file_ref(Path(sys.executable))
        # A freshly recomputed reference to another tool cannot silently retain
        # the old exporter/authoritative verifier identity.
    elif fault == "host":
        result["host_sources"].pop("ast")
    else:
        result["claims"]["full_model_or_pipeline_verification"] = True
    with pytest.raises((ValueError, OSError)):
        source.validate(result)
    source.save(packet["output"] / (fault + "-refusal.json"), result)


def test_full_structural_import_parser_and_closure(packet):
    identity = json.loads(packet["identity"].read_text())
    assert len(identity["import_closure"]) > len(identity["project"])
    assert "MerlinVerify.Sem.Program" in identity["project"]
    assert "Merlin.GraphRegionsSourceMatcher" in identity["project"]
    source_path = packet["output"] / "nested-import.lean"
    source_path.write_text(
        "/- outer /- import False.Nested -/ still commented -/\n"
        "import\n Merlin.GraphRegionsSourceMatcher\n/- import False.Other -/\n"
        "import Merlin.GraphRegionsTensorWitness\n"
    )
    lean, env = source.identity_environment(packet["identity"])
    package_roots = [str(repo_root() / "out/build/lean/packages" / row["name"]) for row in identity["packages"]]
    env["LEAN_SRC_PATH"] = os.pathsep.join(
        [str(repo_root() / "src/merlin/verify/lean"), str(repo_root() / "merlin/lean"), *package_roots]
    )
    dependencies = builder.source_dependencies(lean, source_path, env, packet["output"] / "nested-import-process.json")
    names = {path.stem for path in dependencies}
    assert {"GraphRegionsSourceMatcher", "GraphRegionsTensorWitness"} <= names
    assert not {"Nested", "Other"} & names


@pytest.mark.parametrize("payload", ["proof", "audit"])
def test_well_formed_rehashed_payload_refusals(packet, payload):
    result = copy.deepcopy(packet["receipts"]["native"])
    path = packet["output"] / ("rehashed-" + payload + ".txt")
    path.write_bytes(base.fresh_ref(result[payload]).read_bytes() + b"\n-- substituted receipt payload\n")
    result[payload] = base.file_ref(path)
    source.save(packet["output"] / ("rehashed-" + payload + ".json"), result)
    with pytest.raises(ValueError, match="replayed operational bytes differ"):
        source.replay(result, packet["output"] / ("rehashed-" + payload + "-replay"), fresh_identity=packet["identity"])


def test_missing_import_closure_record_is_refused(packet):
    identity = json.loads(packet["identity"].read_text())
    missing = identity["project"]["MerlinVerify.Sem.Program"]["source"]["path"]
    identity["import_closure"].pop(missing)
    path = packet["output"] / "missing-import-identity.json"
    source.save(path, identity)
    with pytest.raises(ValueError, match="closure domains differ|dependency associations"):
        source.identity_environment(path)


@pytest.mark.parametrize(
    "fault", ["resolved-base-python", "different-configuration", "boolean-reference", "claim-type"]
)
def test_oracle_invocation_environment_and_claim_types(packet, fault):
    result = copy.deepcopy(packet["receipts"]["native"])
    if fault == "resolved-base-python":
        result["oracle_selection"]["invocation"] = result["oracle"]["path"]
    elif fault == "different-configuration":
        result["oracle_selection"]["configuration"] = base.file_ref(packet["output"] / "native/source.mlir")
    elif fault == "boolean-reference":
        result["oracle_selection"]["configuration"]["bytes"] = True
    else:
        result["claims"]["reviewed_contract"] = 0
    source.save(packet["output"] / (fault + "-refusal.json"), result)
    with pytest.raises((ValueError, OSError)):
        source.validate(result)


def test_one_fresh_process_original_and_refusal_batch(packet):
    names = [
        "checked-observed",
        "checked-native",
        "checked-captured",
        "refused-changed-body",
        "refused-logical-second-output",
    ]
    destination = packet["output"] / "fresh-process-batch"
    destination.mkdir()
    code = """import json,sys
from pathlib import Path
from merlin.verify import lean_graph_regions_source as source, lean_graph_regions_source_build as builder
destination=Path(sys.argv[1]); paths=[Path(value) for value in sys.argv[2:]]
builder.build(destination/'build',['Merlin.GraphRegionsTensorWitness'])
results=[]
for index,path in enumerate(paths):
 receipt=json.loads(path.read_text())
 repeated=source.replay(receipt,destination/f'replay-{index}',fresh_identity=destination/'build/IDENTITY.json')
 results.append(dict(original=source.base.file_ref(path),source_accepted=repeated['source_accepted'],
   logical_program_correspondence=repeated['logical_program_correspondence']))
source.save(destination/'RESULTS.json',results)
print(json.dumps(results))
"""
    command = [
        sys.executable,
        "-B",
        "-c",
        code,
        str(destination),
        *[str(packet["output"] / name / "EVIDENCE.json") for name in names],
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, cwd=repo_root())
    source.save(
        destination / "PROCESS.json",
        dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    assert result.returncode == 0, result.stderr
    results = json.loads(result.stdout)
    assert [(r["source_accepted"], r["logical_program_correspondence"]) for r in results] == [
        (True, True),
        (True, True),
        (True, False),
        (False, False),
        (True, False),
    ]
