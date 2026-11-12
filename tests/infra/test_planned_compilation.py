import json

import pytest
from corefixtures.selected_execution import roster

from merlin import api
from merlin.compiler.artifact import Artifact, Compilation
from merlin.compiler.pipeline import PlannedCompiler, RegionBackend
from merlin.compiler.planning.execution import select_execution
from merlin.compiler.transforms.certificates import ir_text


class RecordingBackend:
    def __init__(self, name, events, *, reject=False, mutate=None):
        self.name, self.events, self.reject, self.mutate = (name, events, reject, mutate)

    def validate(self, request):
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        module = parse_mlir_text(request.source.text)
        assert module.body.block.first_op.sym_name.data == "kernel"
        self.events.append((self.name, "validate"))
        if self.reject:
            raise ValueError("fixture backend refuses region")

    def compile(self, request):
        self.events.append((self.name, "compile"))
        request.output.mkdir()
        path = request.output / "commands.bin"
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
    return (source, backend, events)


def test_all_providers_compile_exact_regions_with_binary_products_and_no_image_claim(tmp_path):
    source, backend, events = compiler(tmp_path)
    result = api.compile(ir_text(source.module), backend=backend, output=tmp_path / "out")
    record = api.inspect(result.manifest)
    assert events[:2] == [("vector", "validate"), ("array", "validate")]
    assert [row for row in events if row[1] == "compile"] == [("vector", "compile"), ("array", "compile")]
    binaries = [item for item in result.artifacts if item.kind == "device-binary"]
    assert len(binaries) == 2 and all((item.role == "region.executable" for item in binaries))
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


@pytest.fixture
def staged_public_input(tmp_path):
    import hashlib

    from corefixtures.compile_input_fixtures import SOURCE, _write_capture

    from merlin.compiler.frontend.inputs import stage_compile_inputs

    capture, stage = (tmp_path / "capture", tmp_path / "compiler")
    source = SOURCE.replace(
        "    func.return %a : tensor<2xf32>",
        "    %one = arith.constant dense<1.0> : tensor<2xf32>\n    %r = arith.addf %a, %one : tensor<2xf32>\n    func.return %r : tensor<2xf32>",
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
    return (api.StagedProgram(stage), policy)
