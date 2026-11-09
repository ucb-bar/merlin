"""OOT-discovered overlapping regions, independently checked selected semantics."""

import importlib
import importlib.util
import json
import sys
import types
from dataclasses import replace

import pytest
import yaml

from merlin import api
from merlin.common.paths import repo_root
from merlin.compiler.pipeline import PlannedCompiler, RegionBackend
from merlin.compiler.planning.execution import select_execution
from merlin.compiler.planning.fusion import discover_fusion_candidates
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.offload_plan import bind_source
from merlin.system.topology import select_topology
from merlin.targetgen.target_registry import TargetInfo
from merlin.transforms.certificates import ir_text
from merlin.transforms.pipeline_state import PipelineContext


def _provider():
    # A real optional provider, never a fake emitter supplied by this test.
    root = repo_root().parent / "merlin-backends/gemmini-mlir/compiler"
    if not (root / "provider.py").is_file():
        pytest.skip("select the companion backend checkout for actual OOT integration")
    pytest.importorskip("z3")
    namespace = "gemmini_fusion_integration"
    package = types.ModuleType(namespace)
    package.__path__ = [str(root)]
    sys.modules[namespace] = package
    spec = importlib.util.spec_from_file_location(namespace + ".provider", root / "provider.py")
    provider = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = provider
    spec.loader.exec_module(provider)
    return provider


def _selected(tmp_path):
    # Independently authored target contract fixture. It is deliberately not
    # loaded from the editable compiler's source_groups_output_path.json.
    path = {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [
            {"kind": "preload", "dtype": "i32", "range": [-(1 << 31), (1 << 31) - 1], "rounding": "none"},
            {"kind": "accumulate", "dtype": "i32", "operand_scale": "none"},
        ],
        "readout": [
            {"kind": "activation", "function": "relu", "domain": "accumulator"},
            {"kind": "convert", "to": "f32", "rounding": "half_even"},
            {
                "kind": "scale",
                "dtype": "f32",
                "rounding": "half_even",
                "granularity": "per_command",
                "columns_per_scale": 16,
            },
            {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
            {"kind": "pool", "reductions": ["max"], "padding": "unknown"},
        ],
        "exits": {"accumulator": True, "requant": True},
    }
    contract = tmp_path / "target_contract.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "target": "fixture",
                "output_epilogue": {"output_path": path},
                "numerical_semantics": {"internal_arithmetic": {"mac_result_bits": 20, "signed_operand_bits": 8}},
            }
        )
    )
    target = TargetInfo(
        "fixture", "reference", tmp_path, contract, tmp_path / "dialect.yaml", tmp_path / "facts.json", "simulator"
    )
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.execution_topology.v1",
                "resources": [
                    {"id": "cpu0", "kind": "cpu", "features": [], "shared_group": "cpu0"},
                    {"id": "array0", "kind": "accelerator", "features": [], "shared_group": "array0"},
                ],
                "memory_domains": [{"id": "ram", "capacity_bytes": 100000, "alignment_bytes": 64}],
                "access": [
                    {
                        "resource": unit,
                        "domain": "ram",
                        "read": True,
                        "write": True,
                        "addressability": "shared",
                        "visibility": "coherent",
                        "completion": "synchronous",
                    }
                    for unit in ("cpu0", "array0")
                ],
            }
        )
    )
    return target, select_topology(hardware)


def test_bounded_oot_pages_preserve_overlapping_covers_and_pin_continuation(tmp_path, monkeypatch):
    from merlin.compiler.backend.discovery import ProposalBudget
    from merlin.transforms.certificates import ir_digest

    provider = _provider()
    text = (repo_root() / "merlin/tests/fixtures/fusion_requant.mlir").read_text()
    prefix, body = text.split("    %zero", 1)
    body = "    %zero" + body.split("    func.return", 1)[0]
    second = body.replace("%", "%second_").replace("%second_x", "%x").replace("%second_w", "%w")
    prefix = prefix.replace("-> tensor<2x2xi8> {", "-> (tensor<2x2xi8>, tensor<2x2xi8>) {")
    source = bind_source(
        parse_mlir_text(
            prefix + body + second + "    func.return %q, %second_q : tensor<2x2xi8>, tensor<2x2xi8>\n  }\n}"
        ),
        entry="forward",
        context=PipelineContext(),
        graph_mode="ssa",
    )
    target, topology = _selected(tmp_path)
    before = ir_digest(source.module)
    ordinary = provider.propose(source, resource="array0", memory_domain="ram")
    assert len(ordinary) == 4
    groups = importlib.import_module(provider.__package__ + ".backend.source_groups")
    original = groups._contraction_plans
    visits = []

    def scoped_match(module, *, roots=None):
        assert type(roots) is tuple and len(roots) == 1
        visits.append(roots[0])
        return original(module, roots=roots)

    monkeypatch.setattr(groups, "_contraction_plans", scoped_match)
    budget = ProposalBudget(max_candidates=16, max_match_visits=1, max_proof_sessions=3)
    options = dict(
        provider=provider,
        target=target,
        topology=topology,
        resource="array0",
        memory_domain="ram",
        cpu_resource="cpu0",
        validate_implementation=lambda *_: None,
        budget=budget,
        numerical_policy_sha256="a" * 64,
        preparation_sha256="b" * 64,
    )
    collected, reports, cursor = {}, [], None
    for _ in range(8):
        candidates, report = discover_fusion_candidates(source, **options, cursor=cursor)
        reports.append(report)
        assert report["cpu_alternatives"] == len(source.operations)
        work = report["discovery"]["work"]
        assert work["matcher_visits"] <= 1
        assert work["provider_proof_sessions"] + work["shared_admission_sessions"] <= 3
        for item in candidates:
            if item.proposal.provider == provider.name:
                collected[item.alternative.id] = item
        cursor = report["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("finite provider anchor traversal did not complete")
    assert len(visits) == 4 and len(set(visits)) == 4
    assert set(collected) == {p.alternative.id for p in ordinary}
    assert {dict(p.alternative.metadata)["pattern"] for p in ordinary} == {"dense", "dense-requant"}
    assert len(reports) == 4 and reports[0]["status"] == "budget_exhausted"
    assert reports[-1]["status"] == "complete"
    assert any(row["status"] == "no_match" for report in reports for row in report["discovery"]["provider_reasons"])
    assert reports[0]["discovery"]["work"]["returned_proposals"] == 2
    assert select_execution(
        source,
        tuple(collected.values()) + tuple(item for item in candidates if item.proposal.abi == "inlined"),
        preferred_providers=(provider.name,),
    ).selected
    assert ir_digest(source.module) == before
    json.dumps(reports)
    first_cursor = reports[0]["next_cursor"]
    for changes in (
        {"numerical_policy_sha256": "c" * 64},
        {"preparation_sha256": "c" * 64},
        {"budget": replace(budget, max_region_operations=17)},
    ):
        with pytest.raises(ValueError, match="cursor differs"):
            discover_fusion_candidates(source, **{**options, **changes}, cursor=first_cursor)
    with pytest.raises(ValueError, match="cursor differs"):
        discover_fusion_candidates(source, **options, cursor={**first_cursor, "offset": -1})
    with monkeypatch.context() as patch:
        identity = provider.discovery_identity()
        patch.setattr(provider, "discovery_identity", lambda: "0" * 64 if identity != "0" * 64 else "1" * 64)
        with pytest.raises(ValueError, match="cursor differs"):
            discover_fusion_candidates(source, **options, cursor=first_cursor)
    with pytest.raises(ValueError, match="distinct live"):
        original(source.module, roots=(source.operations[0], source.operations[0]))
    with pytest.raises(ValueError, match="distinct live"):
        original(source.module, roots=(source.operations[0].clone(),))
    limited, truncated = discover_fusion_candidates(
        source, **{**options, "budget": replace(budget, max_tensor_elements=1)}
    )
    assert all(item.proposal.abi == "inlined" for item in limited)
    assert truncated["status"] == "budget_exhausted"
    assert truncated["discovery"]["work"]["matcher_visits"] == 0
    for small in (replace(budget, max_candidates=1), replace(budget, max_proof_sessions=1)):
        with pytest.raises(ValueError, match="insufficient provider page budget"):
            discover_fusion_candidates(source, **{**options, "budget": small})
    single, single_report = discover_fusion_candidates(
        source, **{**options, "budget": replace(budget, max_region_operations=1, max_tensor_elements=(1 << 63) - 1)}
    )
    assert all(item.proposal.abi == "inlined" for item in single)
    assert single_report["discovery_context"]["budget"]["max_tensor_elements"] == (1 << 63) - 1
    assert single_report["status"] == "budget_exhausted"
    assert all(len(row["operation_ordinals"]) == 1 for row in single_report["discovery"]["candidates"])
    assert any(row["category"] == "region_operation_budget" for row in single_report["discovery"]["provider_reasons"])


def test_actual_oot_fused_and_bare_offers_survive_to_selection_and_lowering(tmp_path, monkeypatch):
    provider = _provider()
    fixture = repo_root() / "merlin/tests/fixtures/fusion_requant.mlir"
    source = bind_source(
        parse_mlir_text(fixture.read_text()), entry="forward", context=PipelineContext(), graph_mode="ssa"
    )
    target, topology = _selected(tmp_path)

    def independent_compute_check(current, proposal):
        # K=2 with signed-i8 values remains inside both the selected 20-bit
        # primitive and i32 source accumulator; no concrete sample inputs used.
        from merlin.compiler.ir.contraction_shapes import contraction_extents
        from merlin.targetgen.operation_numerics import integer_partial_sum_bound

        extents = contraction_extents(current.module)
        for index in proposal.alternative.nodes:
            root = current.operations[index]
            if id(root) in extents:
                proof = integer_partial_sum_bound(
                    target.load_contract()["numerical_semantics"],
                    reduction_extent=extents[id(root)][1],
                    lhs_values=[-128, 127],
                    rhs_values=[-128, 127],
                )
                assert proof["status"] == "proven_safe"

    candidates, report = discover_fusion_candidates(
        source,
        provider=provider,
        target=target,
        topology=topology,
        resource="array0",
        memory_domain="ram",
        cpu_resource="cpu0",
        validate_implementation=independent_compute_check,
    )
    offers = [item for item in candidates if item.proposal.provider == provider.name]
    assert {dict(item.alternative.metadata)["pattern"] for item in offers} == {"dense", "dense-requant"}
    assert set(offers[0].alternative.nodes) & set(offers[1].alternative.nodes)
    assert report["cpu_alternatives"] == len(source.operations)
    selected = select_execution(source, candidates, preferred_providers=(provider.name,))
    assert len(selected.selected) == 1
    assert dict(selected.selected[0].alternative.metadata)["pattern"] == "dense-requant"
    result = api.compile(
        ir_text(source.module),
        backend=PlannedCompiler(selected, {provider.name: RegionBackend(provider, "forward")}),
        output=tmp_path / "compiled",
    )
    assert any(artifact.kind == "llvm-dialect" for artifact in result.artifacts)
    assert result.deployment is None
    plan = json.loads((result.directory / "plan.json").read_text())
    assert plan["execution_verified"] is False
    assert result.metadata["execution_verified"] is False
    assert plan["selection"]["source_sha256"] == source.stamp.source_sha256
    # Check the real region manifest and product receipt, not a test emitter.
    from merlin.compiler.artifact import Compilation
    from merlin.compiler.backend.selected import LoweringTicket
    from merlin.compiler.partition.regions import partition_execution
    from merlin.compiler.request import CompileOptions, CompileRequest, Source

    row = plan["regions"][0]
    assert row["lowering_binding"] == "selected_offer"
    child_result = Compilation.load(result.directory / row["compilation"])
    partition = partition_execution(selected)
    region = partition.regions[0]
    child = CompileRequest(
        Source(region.request_source("forward").text, origin=row["lowering"]["request"][0][2]),
        child_result.directory,
        CompileOptions(),
    )
    ticket = LoweringTicket.bind(partition, region, "forward", child)
    observation = ticket.record()["source_observations"]
    assert observation["proposal"]["stage"] == "proposal"
    assert observation["partition"] == plan["source_observation"]
    assert child_result.metadata["selected_lowering"]["ticket_sha256"] == ticket.sha256
    ticket.admit(child, child_result)
    commands = json.loads(next(a.path for a in child_result.artifacts if a.role == "command-plan").read_text())
    assert commands["tensors"]["Y0"]["dtype"] == "i8"
    assert commands["source_group_proof"]["cut"] == "requant"
    assert commands["params"]["global_program_plan"]["source_sha256"] == child.source.sha256
    bad_ack = {**child_result.metadata["selected_lowering"], "ticket_sha256": "0" * 64}
    with pytest.raises(ValueError, match="acknowledgment/products"):
        ticket.admit(child, replace(child_result, metadata={"selected_lowering": bad_ack}))
    bad_product = replace(child_result.artifacts[0], sha256="0" * 64)
    with pytest.raises(ValueError, match="artifact bytes"):
        ticket.admit(child, replace(child_result, artifacts=(bad_product, *child_result.artifacts[1:])))
    incomplete = replace(child_result, artifacts=child_result.artifacts[:1])
    incomplete = replace(incomplete, metadata={"selected_lowering": ticket.acknowledgment(child, incomplete)})
    with pytest.raises(ValueError, match="artifact kinds/roles"):
        ticket.admit(child, incomplete)
    with pytest.raises(ValueError, match="ticket or request changed"):
        provider.validate_selected(
            replace(child, source=replace(child.source, text=child.source.text.replace("arith.mulf", "arith.addf"))),
            ticket,
        )
    other = replace(child, output=tmp_path / "other-region")
    swapped = LoweringTicket.bind(partition, region, "forward", other)
    with pytest.raises(ValueError, match="ticket or request changed"):
        provider.validate_selected(child, swapped)
    with monkeypatch.context() as patch:
        recipe = provider._recipe
        patch.setattr(provider, "_recipe", lambda *args: replace(recipe(*args), pipeline="changed-pipeline"))
        with pytest.raises(ValueError, match="implementation/pipeline/guards changed"):
            provider.validate_selected(child, ticket)
    # The forced selected route checks actual lowerer output as well as receipt
    # identity. Simulate a lowerer returning another precision after selection.
    program = importlib.import_module(provider.__package__ + ".backend.program")
    compiler = importlib.import_module(provider.__package__ + ".backend.compiler")
    with monkeypatch.context() as patch:
        import copy

        buffer = program._single_device_region(child.source.text)

        def switched_precision(value):
            changed = copy.deepcopy(value)
            changed["tensors"]["Y0"]["dtype"] = "i32"
            return changed, "unused"

        patch.setattr(compiler, "compile_buffer", switched_precision)
        with pytest.raises(ValueError, match="tensor representation changed"):
            program._compile(child.source.text, tmp_path, selected_buffer=buffer)
    with monkeypatch.context() as patch:
        patch.setattr(provider, "compile_selected", None)
        refused_output = tmp_path / "missing-selected-protocol"
        with pytest.raises(ValueError, match="explicit validate_selected/compile_selected support"):
            api.compile(
                ir_text(source.module),
                backend=PlannedCompiler(selected, {provider.name: RegionBackend(provider, "forward")}),
                output=refused_output,
            )
        assert not refused_output.exists()
    raw_candidates = tuple(
        item for item in candidates if item.proposal.abi == "inlined" or item.alternative.implementation == "dense"
    )
    raw = select_execution(source, raw_candidates, preferred_providers=(provider.name,))
    raw_result = api.compile(
        ir_text(source.module),
        backend=PlannedCompiler(raw, {provider.name: RegionBackend(provider, "forward")}),
        output=tmp_path / "raw-compiled",
    )
    raw_commands = json.loads(next(a.path for a in raw_result.artifacts if a.role == "region.command-plan").read_text())
    assert raw_commands["tensors"]["Y0"]["dtype"] == "i32"
    assert any(item.proposal.abi == "inlined" for item in raw.selected)
    assert not selected.plan.cycles.resolved
    cpu = select_execution(source, candidates, preferred_providers=("merlin.scalar",))
    assert all(item.proposal.abi == "inlined" for item in cpu.selected)
    # The direct device path cannot assign unrelated memory effects to its
    # device task merely because the returned value is matched successfully.
    effectful = fixture.read_text().replace(
        "    func.return %q",
        "    %scratch = memref.alloc() : memref<1xi8>\n    memref.dealloc %scratch : memref<1xi8>\n    func.return %q",
    )
    assert program._single_device_region(effectful) is None
    original_contract = target.contract_path.read_text()
    missing = yaml.safe_load(original_contract)
    missing.pop("output_epilogue")
    target.contract_path.write_text(yaml.safe_dump(missing))
    refused, refusal_report = discover_fusion_candidates(
        source,
        provider=provider,
        target=target,
        topology=topology,
        resource="array0",
        memory_domain="ram",
        cpu_resource="cpu0",
        validate_implementation=independent_compute_check,
    )
    assert any(row["status"] == "refused" and "no output path" in row["reason"] for row in refusal_report["offers"])
    assert any(row.get("category") == "missing_selected_output_path" for row in refusal_report["offers"])
    assert {
        dict(item.alternative.metadata)["pattern"] for item in refused if item.proposal.provider == provider.name
    } == {"dense"}
    # An edited selected hardware contract invalidates admitted offers, even
    # though the compiler still declares its original readout capability.
    target.contract_path.write_text("target: substituted\n")
    with pytest.raises(ValueError, match="contract changed"):
        selected.revalidate()


def test_public_stage_supplies_actual_policy_and_preparation_to_oot_discovery(tmp_path, monkeypatch):
    import hashlib

    from compile_input_fixtures import _write_capture

    from merlin.compiler.frontend.inputs import stage_compile_inputs
    from merlin.compiler.system import Component, SystemCompiler

    provider = _provider()
    source = (
        (repo_root() / "merlin/tests/fixtures/fusion_requant.mlir")
        .read_text()
        .replace("module {", 'module attributes {prov.weights_file = "origin/weights.safetensors"} {', 1)
    )
    capture, stage = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture, source=source, dtype="i8", input_shapes=((2, 2), (2, 2)), output_shapes=((2, 2),))
    policy = {
        "schema": "fixture.exact-i8.v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "arithmetic": "existing exact signed i8/i32 and ordered readout fixture",
    }
    selected = tmp_path / "precision.json"
    selected.write_text(json.dumps(policy))
    stage_compile_inputs(capture, stage, precision_policy=selected)
    staged = api.StagedProgram(stage)
    target, topology = _selected(tmp_path)
    contexts = []
    original = provider.propose_page

    def observe(current, context, *, cursor=None):
        # Execute the actual companion matcher/proof path, not a synthetic page.
        page = original(current, context, cursor=cursor)
        assert page.proposals
        contexts.append(context)
        raise ValueError("observed actual preparation in OOT discovery; lowering intentionally unattempted")

    monkeypatch.setattr(provider, "propose_page", observe)

    def admit_preparation(current, binding):
        binding.revalidate(current)
        assert binding.policy() == policy

    component = Component(
        RegionBackend(provider, "kernel"), "array0", "ram", lambda *_: None, target=target, cpu_resource="cpu0"
    )
    backend = SystemCompiler(
        topology=topology,
        components=(component,),
        preferred_providers=(provider.name,),
        validate_preparation=admit_preparation,
    )
    with pytest.raises(ValueError, match="observed actual preparation"):
        api.compile(staged, backend=backend, output=tmp_path / "out")
    assert len(contexts) == 1
    context = contexts[0]
    assert context.numerical_policy_sha256 == hashlib.sha256((stage / "precision-policy.json").read_bytes()).hexdigest()
    assert context.preparation_sha256 == hashlib.sha256((stage / "compile-inputs.json").read_bytes()).hexdigest()
    assert context.source_sha256 != policy["source_sha256"]  # Captured and normalized source are distinct identities.
    assert not (tmp_path / "out/compilation.json").exists()  # No lowering, image or execution claim.
