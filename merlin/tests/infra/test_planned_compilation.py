"""Whole-roster compilation orchestration, not native backend qualification."""

import json

import pytest
from test_selected_execution import roster

from merlin import api
from merlin.compiler.artifact import Artifact, Compilation
from merlin.compiler.pipeline import PlannedCompiler, RegionBackend
from merlin.compiler.planning.execution import select_execution
from merlin.transforms.certificates import ir_text


class RecordingBackend:
    def __init__(self, name, events, *, reject=False, mutate=None):
        self.name, self.events, self.reject, self.mutate = name, events, reject, mutate

    def validate(self, request):
        from merlin.frontends.linalg_mlir import parse_mlir_text

        module = parse_mlir_text(request.source.text)
        assert module.body.block.first_op.sym_name.data == "kernel"
        self.events.append((self.name, "validate"))
        if self.reject:
            raise ValueError("fixture backend refuses region")

    def compile(self, request):
        self.events.append((self.name, "compile"))
        request.output.mkdir()
        path = request.output / "commands.bin"
        # Deliberately a transport fixture, not a numerical implementation.
        path.write_bytes(request.source.sha256.encode())
        if self.mutate:
            self.mutate()
        return Compilation(
            request.output,
            self.name,
            request.source.sha256,
            (Artifact.from_file(path, kind="device-binary", role="executable"),),
        )


def compiler(tmp_path, *, reject=False, mutate=None):
    source, _, cpu, device = roster(tmp_path)
    selected = select_execution(source, (*cpu, *device), preferred_providers=("vector", "array"))
    events = []
    backend = PlannedCompiler(
        selected,
        {
            "vector": RegionBackend(RecordingBackend("vector", events, mutate=mutate), "kernel"),
            "array": RegionBackend(RecordingBackend("array", events, reject=reject), "kernel"),
        },
    )
    return source, backend, events


def test_all_providers_compile_exact_regions_with_binary_products_and_no_image_claim(tmp_path):
    source, backend, events = compiler(tmp_path)
    result = api.compile(ir_text(source.module), backend=backend, output=tmp_path / "out")
    record = api.inspect(result.manifest)
    assert events[:2] == [("vector", "validate"), ("array", "validate")]
    assert [row for row in events if row[1] == "compile"] == [("vector", "compile"), ("array", "compile")]
    binaries = [item for item in result.artifacts if item.kind == "device-binary"]
    assert len(binaries) == 2 and all(item.role == "region.executable" for item in binaries)
    assert record["deployment"] is None and record["metadata"]["execution_verified"] is False
    report = json.loads((result.directory / "plan.json").read_text())
    assert len(report["regions"]) == 2 and report["stage"] == "compiled_regions"
    owners = {row["module"] for row in report["source_observation"]["files"]}
    assert {
        "merlin.compiler.ir.region",
        "merlin.compiler.ir.semantics",
        "merlin.compiler.partition.regions",
        "merlin.verification.partition",
    } <= owners
    assert report["regions"][0]["source_ordinals"] != report["regions"][1]["source_ordinals"]
    assert "region_abi_adaptation" in report["remaining"]
    binaries[0].path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes do not match"):
        api.inspect(result.manifest)


def test_preflight_refusal_source_mismatch_and_late_mutation_never_publish_success(tmp_path):
    source, backend, events = compiler(tmp_path, reject=True)
    with pytest.raises(ValueError, match="fixture backend refuses"):
        api.compile(ir_text(source.module), backend=backend, output=tmp_path / "refused")
    assert not (tmp_path / "refused").exists() and not any(kind == "compile" for _, kind in events)
    with pytest.raises(ValueError, match="differs from the original"):
        api.compile("module {func.func @other() {func.return}}", backend=backend, output=tmp_path / "wrong")
    other = tmp_path / "other"
    other.mkdir()
    source, backend, events = compiler(other)

    def change_options():
        backend.backends["array"].options.parameters["fixture.changed"] = 1

    backend.backends["vector"].backend.mutate = change_options
    with pytest.raises(ValueError, match="options changed"):
        api.compile(ir_text(source.module), backend=backend, output=other / "out")
    assert not (other / "out" / "compilation.json").exists()
    assert [(name, kind) for name, kind in events if kind == "compile"] == [("vector", "compile")]


def test_cpu_only_plan_retains_inlinable_source_without_fake_provider(tmp_path):
    source, _, cpu, _ = roster(tmp_path)
    backend = PlannedCompiler(select_execution(source, cpu), {})
    result = api.compile(ir_text(source.module), backend=backend, output=tmp_path / "out")
    assert {item.role for item in result.artifacts} == {"partitioned_source", "execution_plan"}
    report = json.loads((result.directory / "plan.json").read_text())
    assert report["regions"] == [] and report["execution_verified"] is False


def test_system_discovery_binds_the_actual_request_once_and_selects_an_explicit_cover(tmp_path):
    from merlin.compiler.backend.sourceproposal import SourceProposal, proposal_boundary, source_numerical_identity
    from merlin.compiler.system import Component, SystemCompiler
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, RegionAlternative

    original, topology, _, _ = roster(tmp_path)
    events, visits = [], []

    def offers(source, *, resource, memory_domain):
        source.revalidate()
        visits.append(source.stamp.source_sha256)
        nodes = tuple(range(len(source.operations)))
        inputs, outputs = proposal_boundary(source, nodes)
        alternative = RegionAlternative(
            "all", nodes, "fixture.transport", resource, CycleInterval.unknown("unmeasured fixture"), inputs, outputs
        )
        return (
            SourceProposal(
                alternative,
                "vector",
                resource,
                memory_domain,
                "logical_tensor_v1",
                source.stamp.source_sha256,
                source.operation_ordinals,
                source_numerical_identity(source, nodes),
            ),
        )

    def check(source, proposal):
        assert proposal.alternative.nodes == tuple(range(len(source.operations)))
        source.module.verify()

    component = Component(
        RegionBackend(RecordingBackend("vector", events), "kernel"), "cpu0", "dram", check, proposals=offers
    )

    def source_check(module, entry, options):
        from merlin.compiler.ir.semantics import source_semantic_attributes

        assert options == {"contract": "original-fixture-source"}
        module.verify()
        return source_semantic_attributes(module, entry)

    backend = SystemCompiler(
        topology=topology,
        components=(component,),
        preferred_providers=("vector",),
        validate_source=source_check,
        source_options={"contract": "original-fixture-source"},
    )
    result = api.compile(ir_text(original.module), backend=backend, output=tmp_path / "system")
    report = json.loads((result.directory / "system.json").read_text())
    assert result.backend == "merlin.system" and result.deployment is None
    assert len(visits) == 1 and report["realized_ir_sha256"] == visits[0]
    assert report["request_source_sha256"] == result.source_sha256
    assert report["selection"]["execution_order"][0]["provider"] == "vector"
    assert report["numerical_verification"] == "not_established"
    assert report["source_semantics"]["origin"] == "original_caller"
    assert report["source_semantics"]["attributes"] == []
    assert report["source_semantics"]["source_observation"]["stage"] == "semantics"
    assert report["discovery"][0]["report"]["source_observation"]["stage"] == "discovery"
    assert len([event for event in events if event[1] == "compile"]) == 1
    assert report["policy"]["cost_comparison"].startswith("not_performed")


@pytest.mark.parametrize("mutation", ["source-bytes", "callback"])
def test_discovery_observes_shared_owners_before_provider_callback(tmp_path, monkeypatch, mutation):
    from pathlib import Path

    from merlin.compiler.ir import region
    from merlin.compiler.system import Component, SystemCompiler

    copied = tmp_path / "region.py"
    copied.write_bytes(Path(region.__file__).read_bytes())
    monkeypatch.setattr(region, "__file__", str(copied))
    source, topology, _, _ = roster(tmp_path)
    events = []

    def offers(bound, **kwargs):
        if mutation == "source-bytes":
            copied.write_bytes(copied.read_bytes() + b"\n# provider changed shared owner\n")
        else:
            monkeypatch.setattr(region, "preview_region", lambda *args, **kwargs: None)
        return ()

    def check(*args):
        pytest.fail("changed shared owner must refuse before implementation admission")

    component = Component(
        RegionBackend(RecordingBackend("vector", events), "kernel"), "cpu0", "dram", check, proposals=offers
    )
    compiler = SystemCompiler(topology=topology, components=(component,), preferred_providers=("vector",))
    with pytest.raises(ValueError, match="shared source-region implementation"):
        api.compile(ir_text(source.module), backend=compiler, output=tmp_path / "out")
    assert not events and not (tmp_path / "out").exists()


def test_system_discovery_cannot_mutate_source_or_original_component_options(tmp_path):
    from xdsl.dialects.builtin import StringAttr

    from merlin.compiler.backend.sourceproposal import SourceProposal, proposal_boundary, source_numerical_identity
    from merlin.compiler.system import Component, SystemCompiler
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, RegionAlternative

    source, topology, _, _ = roster(tmp_path)
    events = []

    def check(bound, proposal):
        bound.module.verify()

    for mode in ("source", "options"):
        options = api.CompileOptions()

        def altered(bound, *, resource, memory_domain):
            nodes = tuple(range(len(bound.operations)))
            inputs, outputs = proposal_boundary(bound, nodes)
            variant = RegionAlternative(
                "all", nodes, "fixture.transport", resource, CycleInterval.unknown("unmeasured"), inputs, outputs
            )
            claim = SourceProposal(
                variant,
                "vector",
                resource,
                memory_domain,
                "logical_tensor_v1",
                bound.stamp.source_sha256,
                bound.operation_ordinals,
                source_numerical_identity(bound, nodes),
            )
            if mode == "source":
                bound.operations[0].attributes["prov.changed"] = StringAttr("changed")
            else:
                options.parameters["fixture.changed"] = True
            return (claim,)

        component = Component(
            RegionBackend(RecordingBackend("vector", events), "kernel", options),
            "cpu0",
            "dram",
            check,
            proposals=altered,
        )
        backend = SystemCompiler(topology=topology, components=(component,), preferred_providers=("vector",))
        with pytest.raises(ValueError, match="changed|stale"):
            api.compile(ir_text(source.module), backend=backend, output=tmp_path / mode)
        assert not (tmp_path / mode).exists()
    assert not events


def test_destination_initialization_replays_scalar_bits_and_rejects_changed_fill():
    from xdsl.dialects import arith, builtin, tensor
    from xdsl.dialects.linalg.ops import FillOp

    from merlin.compiler.transforms.initialization import normalize_destinations, verify_destinations
    from merlin.frontends.linalg_mlir import parse_mlir_text

    # The scalar's exact LLVM hex spelling is retained; no float conversion occurs.
    for literal in ("0x80000000", "0x7FC00001", "0x00000001"):
        module = parse_mlir_text(
            """module {func.func @f(%a:tensor<2x2xf32>) -> tensor<2x2xf32> {
 %v=arith.constant VALUE:f32
 %seed=tensor.splat %v : tensor<2x2xf32>
 %r=linalg.matmul ins(%a,%a:tensor<2x2xf32>,tensor<2x2xf32>)
 outs(%seed:tensor<2x2xf32>) -> tensor<2x2xf32>
 func.return %r:tensor<2x2xf32>
}}""".replace("VALUE", literal)
        )
        selected, records = normalize_destinations(module)
        assert len(records) == 1
        assert not any(type(op) is tensor.SplatOp for op in selected.walk())
        old_scalar = next(op.value for op in module.walk() if type(op) is arith.ConstantOp)
        assert next(op.value for op in selected.walk() if type(op) is arith.ConstantOp) == old_scalar
        fill = next(op for op in selected.walk() if type(op) is FillOp)
        fill.attributes["changed"] = builtin.UnitAttr()
        with pytest.raises(ValueError, match="arithmetic, effects or provenance"):
            verify_destinations(module, selected, records)
        # Any extra semantic attribute keeps the original splat untouched.
        source_splat = next(op for op in module.walk() if type(op) is tensor.SplatOp)
        source_splat.attributes["semantic"] = builtin.UnitAttr()
        unchanged, records = normalize_destinations(module)
        assert not records and module.is_structurally_equivalent(unchanged)
    module = parse_mlir_text("""module {func.func @f() -> tensor<2xf32> {
 %v=arith.constant 0.0:f32
 %s=tensor.splat %v:tensor<2xf32>
 func.return %s:tensor<2xf32>
}}""")
    unchanged, records = normalize_destinations(module)
    assert not records and module.is_structurally_equivalent(unchanged)

    module = parse_mlir_text("""module {func.func @f() -> tensor<2xf32> {
 %e="tensor.empty"() {prov.source_node_ids=["empty0"]}:() -> tensor<2xf32>
 func.return %e:tensor<2xf32>
}}""")
    selected, records = normalize_destinations(module)
    assert len(records) == 1 and "empty_metadata_ordinal" in records[0]
    verify_destinations(module, selected, records)
    empty = next(op for op in selected.walk() if type(op) is tensor.EmptyOp)
    empty.attributes["prov.changed"] = builtin.UnitAttr()
    with pytest.raises(ValueError, match="detached provenance or semantic"):
        verify_destinations(module, selected, records)
    old_empty = next(op for op in module.walk() if type(op) is tensor.EmptyOp)
    old_empty.attributes["semantic"] = builtin.UnitAttr()
    unchanged, records = normalize_destinations(module)
    assert not records and module.is_structurally_equivalent(unchanged)


def _prepared_system(tmp_path, policy, *, admission=True, alter=None, legacy_hook=False):
    from test_source_proposal_admission import topology

    from merlin.compiler.backend.sourceproposal import SourceProposal, proposal_boundary, source_numerical_identity
    from merlin.compiler.system import Component, SystemCompiler
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, RegionAlternative

    observed = []

    class IRBackend:
        name = "fixture.ir"

        def validate(self, request):
            from merlin.frontends.linalg_mlir import parse_mlir_text

            parse_mlir_text(request.source.text).verify()

        def compile(self, request):
            request.output.mkdir()
            path = request.output / "region.mlir"
            path.write_text(request.source.text)
            return Compilation(
                request.output,
                self.name,
                request.source.sha256,
                (Artifact.from_file(path, kind="mlir", role="intermediate"),),
            )

    def proposals(source, *, resource, memory_domain, preparation=None):
        assert preparation is not None and preparation.policy() == policy
        observed.append(preparation.record())
        nodes = tuple(range(len(source.operations)))
        inputs, outputs = proposal_boundary(source, nodes)
        alternative = RegionAlternative(
            "prepared",
            nodes,
            "fixture.identity-ir",
            resource,
            CycleInterval.unknown("IR transport, no native cost"),
            inputs,
            outputs,
        )
        if alter:
            alter()
        return (
            SourceProposal(
                alternative,
                "fixture.ir",
                resource,
                memory_domain,
                "logical_tensor_v1",
                source.stamp.source_sha256,
                source.operation_ordinals,
                source_numerical_identity(source, nodes),
            ),
        )

    def old_proposals(source, *, resource, memory_domain):
        raise AssertionError("unsupported preparation must refuse before invoking an old hook")

    def validate(source, binding):
        from merlin.frontends.linalg_mlir import parse_mlir_text

        binding.revalidate(source)
        assert binding.policy() == policy  # Original selected policy, not a provider's acceptance threshold.
        parse_mlir_text(source.text).verify()

    def check(source, proposal):
        source.module.verify()
        assert proposal.numerical_sha256 == source_numerical_identity(source, proposal.alternative.nodes)

    component = Component(
        RegionBackend(IRBackend(), "kernel"),
        "cpu0",
        "dram",
        check,
        proposals=old_proposals if legacy_hook else proposals,
    )
    import yaml

    from merlin.system.topology import select_topology

    hardware = topology(tmp_path)
    document = yaml.safe_load(hardware.path.read_bytes())
    document["memory_domains"][0]["capacity_bytes"] = 65536  # Explicit fixture resource, not a compiler default.
    hardware.path.write_text(yaml.safe_dump(document))
    selected = SystemCompiler(
        topology=select_topology(hardware.path),
        components=(component,),
        preferred_providers=("fixture.ir",),
        validate_preparation=validate if admission else None,
    )
    return selected, observed


@pytest.fixture
def staged_public_input(tmp_path):
    import hashlib

    from compile_input_fixtures import SOURCE, _write_capture

    from merlin.compiler.frontend.inputs import stage_compile_inputs

    capture, stage = tmp_path / "capture", tmp_path / "compiler"
    source = SOURCE.replace(
        "    func.return %a : tensor<2xf32>",
        "    %one = arith.constant dense<1.0> : tensor<2xf32>\n"
        "    %r = arith.addf %a, %one : tensor<2xf32>\n"
        "    func.return %r : tensor<2xf32>",
    )
    _write_capture(capture, source=source)
    policy = {
        "schema": "fixture.precision.v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "transformation": "identity",
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    stage_compile_inputs(capture, stage, precision_policy=path)
    return api.StagedProgram(stage), policy


def test_public_staged_preparation_reaches_component_discovery_and_excludes_evaluator(
    tmp_path, staged_public_input, monkeypatch, capsys
):
    import hashlib
    from pathlib import Path

    staged, policy = staged_public_input
    backend, observed = _prepared_system(tmp_path, policy)
    original = Path.read_bytes

    def compiler_only(path):
        if path.name in {"inputs.npz", "golden.npy", "inputs.json", "golden.json", "extra.npz"}:
            raise AssertionError("evaluator input entered ordinary compilation")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", compiler_only)
    result = api.compile(staged, backend=backend, output=tmp_path / "compiled")
    record = json.loads((result.directory / "system.json").read_text())
    expected = observed[0]
    assert len(observed) == 1 and record["preparation"] == expected
    assert (
        expected["numerical_policy_sha256"]
        == hashlib.sha256((staged.directory / "precision-policy.json").read_bytes()).hexdigest()
    )
    assert (
        expected["preparation_sha256"]
        == hashlib.sha256((staged.directory / "compile-inputs.json").read_bytes()).hexdigest()
    )
    assert expected["bound_source_sha256"] == result.source_sha256
    assert expected["captured_source_sha256"] == policy["source_sha256"]
    assert record["numerical_verification"] == "not_established" and result.deployment is None
    assert result.metadata["request"]["preparation"] == expected
    assert not any(path.name in {"inputs.npz", "golden.npy", "extra.npz"} for path in result.directory.rglob("*"))
    from types import SimpleNamespace

    from merlin.compiler.backend import service
    from merlin.service_cli import compile_main

    monkeypatch.setattr(service, "entry_points", lambda **kwargs: [SimpleNamespace(load=lambda: backend)])
    assert (
        compile_main(
            [
                str(staged.directory),
                "--input-kind=compiler-stage",
                "--backend",
                backend.name,
                "--output",
                str(tmp_path / "compiled-cli"),
            ]
        )
        == 0
    )
    cli = json.loads(capsys.readouterr().out)
    assert cli["metadata"]["request"]["preparation"] == expected
    assert cli["source_sha256"] == result.source_sha256 and observed[-1] == expected
    assert not any(
        path.name in {"inputs.npz", "golden.npy", "extra.npz"} for path in (tmp_path / "compiled-cli").rglob("*")
    )


@pytest.mark.parametrize(
    "changed",
    [
        "policy",
        "source",
        "directory",
        "policy_during_proposal",
        "no_admission",
        "old_hook",
    ],
)
def test_public_prepared_source_or_policy_substitution_and_unsupported_routes_refuse(
    tmp_path, staged_public_input, changed
):
    import hashlib

    staged, policy = staged_public_input

    def mutate(name):
        path = staged.directory / name
        if name == "precision-policy.json":
            document = json.loads(path.read_bytes())
            document["transformation"] = "different"
            path.write_text(json.dumps(document))
        else:
            path.write_text(path.read_text() + "\n")
        manifest_path = staged.directory / "compile-inputs.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["members"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    backend, observed = _prepared_system(
        tmp_path,
        policy,
        admission=changed != "no_admission",
        legacy_hook=changed == "old_hook",
        alter=(lambda: mutate("precision-policy.json")) if changed == "policy_during_proposal" else None,
    )
    if changed in {"policy", "source"}:
        mutate("precision-policy.json" if changed == "policy" else "program.mlir")
    elif changed == "directory":
        import shutil

        replacement = tmp_path / "substituted-stage"
        shutil.copytree(staged.directory, replacement)
        object.__setattr__(staged, "directory", replacement)
    with pytest.raises(ValueError, match="changed|preparation|stage"):
        api.compile(staged, backend=backend, output=tmp_path / "refused")
    assert not (tmp_path / "refused/compilation.json").exists()
    if changed != "policy_during_proposal":
        assert not observed


def test_real_recipe_prepared_int8_stage_plans_through_public_api_and_matches_independent_reference(tmp_path):
    import hashlib
    import os
    import subprocess
    from pathlib import Path

    import numpy as np

    from merlin.compiler.frontend.inputs import stage_compile_inputs
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.runtime import linalg_numpy as LN
    from merlin.targetgen import _m2m_capture_worker as worker
    from merlin.targetgen import quant_recipe as Q

    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("select the actual TorchAO capture interpreter and producer")
    loader = tmp_path / "loader.py"
    loader.write_text("""import torch
class Kernel(torch.nn.Module):
    def forward(self,a,b): return a @ b
def get_model_and_inputs():
    a=(torch.arange(64,dtype=torch.float32).reshape(1,64)%5)-2
    b=(torch.arange(128,dtype=torch.float32).reshape(64,2)%7)-3
    return Kernel().eval(),(a,b)
def get_calibration_inputs(model,inputs): return [inputs,inputs]
""")
    tensor = dict(dtype="int8", granularity="tensor", symmetric=True, block=None, mode="static")
    recipe = dict(
        schema=Q.SCHEMA,
        target="fixture",
        unit="matrix",
        status="derived",
        families=["contraction"],
        weight={**tensor, "quant_min": -127, "quant_max": 127},
        activation={**tensor, "quant_min": -128, "quant_max": 127},
        bias_domain="accumulator",
        software_numerical_engine="integer_reference",
        framework_capture_policy={"calibration_count": 2},
    )
    recipe["recipe_sha256"] = Q.digest(recipe)
    recipe = Q.with_dynamic_contractions(recipe, static_families=[])
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    capture = tmp_path / "capture"
    process = subprocess.run(
        [
            python,
            "-B",
            str(Path(worker.__file__)),
            "--m2m-dir",
            m2m,
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe_path),
            "--materialize-bundle",
            "--out",
            str(capture),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr[-7000:]
    source = (capture / "model.mlir").read_bytes()
    policy = {
        "schema": "fixture.recipe-preparation.v1",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "recipe": recipe,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    stage = tmp_path / "compiler"
    stage_compile_inputs(capture, stage, precision_policy=policy_path)
    backend, observed = _prepared_system(tmp_path, policy)
    result = api.compile(api.StagedProgram(stage), backend=backend, output=tmp_path / "compiled")
    actual_source = (result.directory / "frontend.mlir").read_text()
    with np.load(capture / "inputs.npz", allow_pickle=False) as arrays:
        arguments = [arrays[key] for key in sorted(arrays.files)]
    regions = [artifact for artifact in result.artifacts if artifact.role == "region.intermediate"]
    assert len(regions) == 1 and regions[0].kind == "mlir"
    # Evaluate the actual outlined/backend-emitted region, not just its input IR.
    actual = LN.evaluate(parse_mlir_text(regions[0].path.read_text()), arguments, function="kernel")[0]
    imported = LN.evaluate(parse_mlir_text(actual_source), arguments)[0]
    reference = np.load(capture / "golden.npy", allow_pickle=False)
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    np.testing.assert_array_equal(actual, imported)
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=2e-6)
    assert observed[0]["captured_source_sha256"] == policy["source_sha256"]
    assert "i8" in actual_source and "i32" in actual_source
    assert result.metadata["execution_verified"] is False  # IR/reference agreement is not device execution.


def test_public_staged_input_keeps_buffer_state_refusal(tmp_path):
    import hashlib

    from compile_input_fixtures import _write_weighted_capture

    from merlin.compiler.frontend.inputs import stage_compile_inputs

    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _write_weighted_capture(capture)
    manifest_path = capture / "weights.safetensors.manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["0"]["kind"] = "buffer"
    manifest_path.write_text(json.dumps(manifest))
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["artifacts"][manifest_path.name] = {
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "bytes": manifest_path.stat().st_size,
    }
    receipt_path.write_text(json.dumps(receipt))
    selected = tmp_path / "policy.json"
    selected.write_text(
        json.dumps(
            {
                "schema": "fixture.state.v1",
                "source_sha256": hashlib.sha256((capture / "model.mlir").read_bytes()).hexdigest(),
            }
        )
    )
    stage_compile_inputs(capture, stage, precision_policy=selected)
    events = []
    with pytest.raises(ValueError, match="buffer binding requires"):
        api.compile(api.StagedProgram(stage), backend=RecordingBackend("fixture", events), output=tmp_path / "out")
    assert not events and not (tmp_path / "out").exists()
