"""Exact source and original execution contract admission, without hardware claims."""

from dataclasses import replace

import pytest
import yaml
from test_offload_plan import example as legacy_example

from merlin.compiler.backend.sourceproposal import (
    SourceProposal,
    admit_original_cpu,
    admit_proposal,
    source_numerical_identity,
)
from merlin.llvmlower.offload_plan import bind_source
from merlin.system.topology import select_topology
from merlin.transforms.dense_ir import recognize_signed_integer_matmul


def example():
    legacy, alternatives, context = legacy_example()
    source = bind_source(legacy.module, entry=legacy.entry, context=context, graph_mode="ssa")
    alternatives = tuple(replace(source.cpu_alternatives[item.nodes[0]], id=item.id,
                                 implementation=item.implementation, placement=item.placement)
                         for item in alternatives)
    return source, alternatives, context


def topology(tmp_path, **edge_changes):
    document = {
        "schema": "merlin.execution_topology.v1",
        "resources": [
            {"id": "cpu0", "kind": "cpu", "features": ["rvv"], "shared_group": "cpu0"},
            {"id": "array0", "kind": "accelerator", "features": ["integer-matmul"], "shared_group": "array0"},
        ],
        "memory_domains": [{"id": "dram", "capacity_bytes": 4096, "alignment_bytes": 64}],
        "access": [dict(resource=resource, domain="dram", read=True, write=True,
                        addressability="shared", visibility="coherent", completion="synchronous")
                   for resource in ("cpu0", "array0")],
    }
    document["access"][1].update(edge_changes)
    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump(document))
    return select_topology(path)


def proposal(source, alternative):
    return SourceProposal(alternative, "test.checked.integer", "array0", "dram", "logical_tensor_v1",
                          source.stamp.source_sha256,
                          tuple(source.operation_ordinals[n] for n in alternative.nodes),
                          source_numerical_identity(source, alternative.nodes), ("integer-matmul",))


def original_check(source, candidate):
    # Independent source-semantic check for this unit fixture; no emitted kernel
    # or actual hardware is accepted/qualified by this test.
    for index in candidate.alternative.nodes:
        operation = source.operations[index]
        dtypes = tuple(str(v.type.element_type) for v in (*operation.inputs, *operation.outputs))
        recognize_signed_integer_matmul(operation, dtypes)


@pytest.mark.parametrize("mutation", ["source-bytes", "callback", "callback-code"])
def test_proposal_rechecks_shared_semantic_owner_after_original_callback(tmp_path, monkeypatch, mutation):
    from pathlib import Path

    from merlin.compiler.ir import semantics

    # Observe an exact copy as the module's selected source file. Live sources
    # remain untouched, and the real original source/checking algorithms run.
    observed = tmp_path / "semantics.py"
    observed.write_bytes(Path(semantics.__file__).read_bytes())
    monkeypatch.setattr(semantics, "__file__", str(observed))
    source, alternatives, _ = example()
    selected = topology(tmp_path)

    def check(bound, claim):
        original_check(bound, claim)
        if mutation == "source-bytes":
            observed.write_bytes(observed.read_bytes() + b"\n# changed during admission\n")
        elif mutation == "callback":
            monkeypatch.setattr(semantics, "region_semantic_attributes", lambda *args: ())
        else:
            def replacement(*args):
                return ()
            monkeypatch.setattr(semantics.region_semantic_attributes, "__code__", replacement.__code__)

    with pytest.raises(ValueError, match="shared source-region implementation"):
        admit_proposal(source, proposal(source, alternatives[0]), selected, validate_implementation=check)


def test_admitted_proposal_retains_observation_and_refuses_changed_bytes_before_callback(tmp_path, monkeypatch):
    from pathlib import Path

    from merlin.compiler.ir import semantics

    observed = tmp_path / "semantics.py"
    observed.write_bytes(Path(semantics.__file__).read_bytes())
    monkeypatch.setattr(semantics, "__file__", str(observed))
    source, alternatives, _ = example()
    calls = []
    def check(bound, claim):
        calls.append(claim.alternative.id)
        original_check(bound, claim)
    admitted = admit_proposal(source, proposal(source, alternatives[0]), topology(tmp_path),
                              validate_implementation=check)
    record = admitted.source_observation()
    assert any(row["path"] == str(observed) for row in record["files"])
    record["files"].clear()  # Public receipt mutations cannot reseal the live observation.
    observed.write_bytes(observed.read_bytes() + b"\n# changed after admission\n")
    with pytest.raises(ValueError, match="implementation bytes changed"):
        admitted.revalidate()
    assert len(calls) == 1


def test_exact_sites_shared_cpu_and_original_source_changes(tmp_path):
    source, alternatives, _ = example()
    selected = topology(tmp_path)
    admitted = admit_proposal(source, proposal(source, alternatives[1]), selected,
                              validate_implementation=original_check)
    assert admitted.alternative.nodes == (7,)
    assert admitted.proposal.operation_ordinals != proposal(source, alternatives[0]).operation_ordinals
    cpu = tuple(admit_original_cpu(source, n, selected, resource="cpu0", memory_domain="dram")
                for n in range(len(source.operations)))
    assert all(p.proposal.resource == "cpu0" and p.proposal.abi == "inlined" for p in cpu)
    assert selected.resource("cpu0").shared_group == "cpu0"
    # Scalar and vector implementation features do not create independent resources.
    vector = replace(proposal(source, alternatives[0]), resource="cpu0", required_features=("rvv",))
    admit_proposal(source, vector, selected, validate_implementation=original_check).revalidate()
    selected.path.write_text(selected.path.read_text() + "\n# changed selected evidence\n")
    with pytest.raises(ValueError, match="topology changed"):
        admitted.revalidate()


@pytest.mark.parametrize("change,reason", [
    ({"resource": "missing"}, "undeclared execution"),
    ({"required_features": ("missing",)}, "required implementation features"),
    ({"numerical_sha256": "changed"}, "numerical contract"),
    ({"operation_ordinals": (0,)}, "different original source"),
    ({"abi": "opaque"}, "boundary ABI"),
])
def test_proposal_refusals_precede_implementation_callback(tmp_path, change, reason):
    source, alternatives, _ = example()
    candidate = replace(proposal(source, alternatives[0]), **change)
    def forbidden(*args):
        pytest.fail("invalid original binding must refuse before implementation validation")
    with pytest.raises(ValueError, match=reason):
        admit_proposal(source, candidate, topology(tmp_path), validate_implementation=forbidden)


@pytest.mark.parametrize("edge", [
    {"read": False}, {"addressability": "separate"}, {"visibility": "explicit"}, {"completion": "unknown"},
])
def test_missing_memory_path_is_not_an_implicit_transfer(tmp_path, edge):
    source, alternatives, _ = example()
    with pytest.raises(ValueError, match="access|transfer, visibility or completion"):
        admit_proposal(source, proposal(source, alternatives[0]), topology(tmp_path, **edge),
                       validate_implementation=original_check)


def test_validator_source_and_capacity_cannot_change_after_admission(tmp_path):
    source, alternatives, _ = example()
    selected = topology(tmp_path)
    candidate = proposal(source, alternatives[0])
    with pytest.raises(ValueError, match="scratch exceeds"):
        admit_proposal(source, replace(candidate, alternative=replace(candidate.alternative, scratch_bytes=4097)),
                       selected, validate_implementation=original_check)
    admitted = admit_proposal(source, candidate, selected, validate_implementation=original_check)
    operation = source.operations[candidate.alternative.nodes[0]]
    from xdsl.dialects.builtin import StringAttr
    operation.attributes["prov.changed"] = StringAttr("changed")
    with pytest.raises(ValueError, match="changed|stale"):
        admitted.revalidate()


def test_original_checker_is_retained_and_provider_cannot_substitute_it(tmp_path):
    source, alternatives, _ = example()
    def checked(source, candidate):
        original_check(source, candidate)
    admitted = admit_proposal(source, proposal(source, alternatives[0]), topology(tmp_path),
                              validate_implementation=checked)
    def replacement(source, candidate):
        return None
    checked.__code__ = replacement.__code__
    with pytest.raises(ValueError, match="validator changed"):
        admitted.revalidate()


def test_memory_budget_requires_original_alignment_and_capacity(tmp_path):
    selected = topology(tmp_path)
    assert selected.memory("dram").alignment_bytes == 64
    document = yaml.safe_load(selected.path.read_text())
    document["memory_domains"][0].pop("alignment_bytes")
    selected.path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="exact explicitly typed fields"):
        select_topology(selected.path)
    document["memory_domains"][0]["alignment_bytes"] = 3
    selected.path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="power of two"):
        select_topology(selected.path)


@pytest.mark.parametrize("tensor", [False, True])
def test_logical_tensor_abi_checks_actual_ssa_boundaries_before_validator(tmp_path, tensor):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.offload_plan import bind_source
    from merlin.transforms.pipeline_state import PipelineContext
    from merlin.xdsl_dialects.lowering.global_plan import BufferRepresentation, ValueRepresentation
    from merlin.xdsl_dialects.lowering.global_plan_emission import _component_boundary

    # A zero-rank tensor and scalar share shape=[] in DispatchProgram. Only the
    # original SSA type can distinguish their invocation ABI.
    typ = "tensor<i32>" if tensor else "i32"
    if tensor:
        body = """%x = tensor.extract %a[] : tensor<i32>
          %y = tensor.extract %b[] : tensor<i32>
          %sum = arith.addi %x, %y : i32
          %result = tensor.from_elements %sum : tensor<i32>"""
    else:
        body = "%result = arith.addi %a, %b : i32"
    module = parse_mlir_text(f"""module {{func.func @forward(%a:{typ}, %b:{typ})->{typ} {{
      {body}
      func.return %result : {typ}
    }}}}""")
    source = bind_source(module, entry="forward", context=PipelineContext(), graph_mode="ssa")
    selected = topology(tmp_path)
    nodes = tuple(range(len(source.operations)))
    inputs, outputs = _component_boundary(source.program, nodes)
    order = {name: index for index, name in enumerate(source.program.buffers)}
    def reps(names):
        return tuple(BufferRepresentation(name, ValueRepresentation("tensor_ssa", "logical", "i32"))
                     for name in sorted(names, key=order.__getitem__))
    alternative = replace(source.cpu_alternatives[0], id="checked-chain", nodes=nodes,
                          inputs=reps(inputs), outputs=reps(outputs))
    candidate = replace(proposal(source, alternative), resource="cpu0", required_features=())
    called = []
    def independent_fixture_check(bound, offered):
        called.append(offered.operation_ordinals)
        assert any(bound.operations[n].name == "arith.addi" for n in nodes)
    if tensor:
        admit_proposal(source, candidate, selected,
                       validate_implementation=independent_fixture_check).revalidate()
        assert called
    else:
        with pytest.raises(ValueError, match="raw scalar boundaries"):
            admit_proposal(source, candidate, selected, validate_implementation=independent_fixture_check)
        assert not called
        # The exact original scalar program remains admissible inlined CPU IR.
        admit_original_cpu(source, 0, selected, resource="cpu0", memory_domain="dram").revalidate()


def test_legacy_outlined_graph_does_not_authorize_actual_ssa_boundaries(tmp_path):
    source, alternatives, _ = legacy_example()
    with pytest.raises(ValueError, match="actual SSA graph"):
        admit_proposal(source, proposal(source, alternatives[0]), topology(tmp_path),
                       validate_implementation=original_check)


def test_source_semantic_admission_does_not_authorize_builtin_scalar_fallback(tmp_path):
    from test_offload_plan import annotated_source, approved_source_options, check_annotated_source

    from merlin.compiler.backend.discovery import ProposalBudget
    from merlin.compiler.backend.scalaroffers import propose
    from merlin.compiler.components import _admit_cpu
    from merlin.transforms.pipeline_state import PipelineContext

    source = bind_source(annotated_source(), entry="forward", context=PipelineContext(), graph_mode="ssa",
                         validate_source=check_annotated_source, source_options=approved_source_options())
    selected = topology(tmp_path)
    # Unannotated initializer preservation inherits no unrelated operation's
    # numerical permission. Selecting the annotated ReLU still must refuse.
    original = admit_original_cpu(source, 0, selected, resource="cpu0", memory_domain="dram")
    assert original.alternative.nodes == (0,)
    with pytest.raises(ValueError, match="annotated source numerical obligations"):
        admit_original_cpu(source, 1, selected, resource="cpu0", memory_domain="dram")
    whole = propose(source, resource="cpu0", memory_domain="dram", budget=ProposalBudget())[0]
    with pytest.raises(ValueError, match="annotated source numerical obligations"):
        _admit_cpu(source, whole)
