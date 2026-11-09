"""Fresh prepared-export evidence plus explicit caller consent freezes BUFFER bytes."""

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin import api
from merlin.compiler.frontend.inputs import load_compile_inputs, stage_compile_inputs
from merlin.compiler.frontend.staged import prepare
from merlin.compiler.frontend.state import validate_observation
from merlin.targetgen import _m2m_capture_worker as worker


@pytest.fixture(scope="module", params=("linear", "batchnorm"))
def readonly_capture(request, tmp_path_factory):
    python, producer = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not producer:
        pytest.skip("select the real pinned Torch/TorchAO/model2MLIR capture environment")
    directory = tmp_path_factory.mktemp("readonly-" + request.param)
    loader = directory / "loader.py"
    loader.write_text(
        """import torch
def get_model_and_inputs():
    torch.manual_seed(3)
    model = MODEL
    return model.eval(), (torch.arange(SIZE,dtype=torch.float32).reshape(SHAPE)/11,)
def get_calibration_inputs(model,inputs): return [inputs,inputs]
""".replace("MODEL", "torch.nn.Linear(64,2,bias=False)" if request.param == "linear" else "torch.nn.BatchNorm1d(4)")
        .replace("SIZE", "64" if request.param == "linear" else "8")
        .replace("SHAPE", "1,64" if request.param == "linear" else "2,4")
    )
    command = [
        python,
        "-B",
        str(Path(worker.__file__)),
        "--m2m-dir",
        producer,
        "--loader",
        str(loader),
        "--dtype",
        "int8" if request.param == "linear" else "fp32",
        "--materialize-bundle",
        "--observe-buffer-state",
        "--out",
        str(directory / "capture"),
    ]
    recipe = None
    if request.param == "linear":
        from merlin.targetgen import quant_recipe as Q

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
        path = directory / "recipe.json"
        path.write_text(json.dumps(recipe))
        command += ["--recipe", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stderr[-10000:]
    capture = directory / "capture"
    policy = {
        "schema": "fixture.readonly-preparation.v1",
        "recipe": recipe,
        "source_sha256": hashlib.sha256((capture / "model.mlir").read_bytes()).hexdigest(),
    }
    path = directory / "policy.json"
    path.write_text(json.dumps(policy))
    stage = directory / "compiler"
    stage_compile_inputs(capture, stage, precision_policy=path)
    return request.param, capture, stage, policy


def test_real_prepared_readonly_buffers_keep_roles_and_match_source_reference(readonly_capture, tmp_path):
    from test_planned_compilation import _prepared_system

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.runtime import linalg_numpy as LN

    kind, capture, stage, policy = readonly_capture
    staged = load_compile_inputs(stage)
    assert any(row["kind"] == "buffer" for row in staged.argument_bindings)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in stage.iterdir()}
    with pytest.raises(ValueError, match="buffer binding requires"):
        prepare(api.StagedProgram(stage), max_initialization_bytes=65536)
    selected = api.StagedProgram(stage, freeze_readonly_buffers=True)
    source, preparation = prepare(selected, max_initialization_bytes=65536)
    freeze = preparation.record()["buffer_freeze"]
    assert freeze["consent"] is True and all(row["kind"] == "buffer" for row in freeze["buffers"])
    initializers = preparation.bound.record["initializers"]
    assert all(row["binding"] == "frozen_snapshot" for row in initializers if row["kind"] == "buffer")
    if kind == "linear":
        assert any(row["target"] == "_frozen_param0" for row in freeze["buffers"])
        backend, observed = _prepared_system(tmp_path, policy)
        result = api.compile(selected, backend=backend, output=tmp_path / "compiled")
        legalization = observed[0]["quantization_legalization"]
        assert legalization["input_ir_sha256"] != legalization["output_ir_sha256"]
        assert "quant_ext.quantize_per_tensor" in staged.program and "quant_ext." not in source.text
        assert legalization["rewrites"][0]["source_attributes"]["prov.source_node_ids"]
        region = next(row for row in result.artifacts if row.role == "region.intermediate")
        with np.load(capture / "inputs.npz", allow_pickle=False) as arrays:
            inputs = [arrays[key] for key in sorted(arrays.files)]
        actual = LN.evaluate(parse_mlir_text(region.path.read_text()), inputs, function="kernel")[0]
        np.testing.assert_allclose(actual, np.load(capture / "golden.npy", allow_pickle=False), rtol=1e-5, atol=2e-6)
        assert observed[0]["buffer_freeze"] == freeze and result.metadata["execution_verified"] is False
    assert "buffer_freeze" in preparation.record() and source.kind == "mlir"
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in stage.iterdir()}
    object.__setattr__(selected, "freeze_readonly_buffers", False)
    with pytest.raises(ValueError, match="consent changed"):
        selected.revalidate()


@pytest.mark.parametrize("change", ("missing", "payload", "role", "alias", "effect", "prepared", "manifest"))
def test_observation_and_selected_buffer_changes_refuse(readonly_capture, tmp_path, change):
    import shutil

    _, _, original, _ = readonly_capture
    stage = tmp_path / "selected"
    shutil.copytree(original, stage)
    path = stage / "compile-inputs.json"
    manifest = json.loads(path.read_bytes())
    evidence = manifest["source"]["export_state"]
    row = next(item for item in evidence["inputs"] if item["kind"] == "BUFFER")
    if change == "missing":
        del manifest["source"]["export_state"]
    elif change == "payload":
        row["payload_sha256"] = "0" * 64
    elif change == "role":
        row["kind"] = "PARAMETER"
    elif change == "alias":
        next(item for item in evidence["inputs"] if item is not row)["storage_group"] = row["storage_group"]
    elif change == "effect":
        evidence["operator_effects"][0]["writes"] = ["self"]
    elif change == "prepared":
        evidence["prepared_graph_sha256"] = "0" * 64
    else:
        # Replacing archive bytes and resealing the ordinary stage does not
        # authorize replacing the producer-observed BUFFER snapshot.
        binding = load_compile_inputs(stage).argument_bindings[row["ordinal"]]
        archive = stage / binding.get("file", "weights.safetensors")
        with archive.open("r+b") as stream:
            size = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(size))
            stream.seek(8 + size + header[binding["weight"]]["data_offsets"][0])
            value = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([value[0] ^ 1]))
        manifest["members"][archive.name] = hashlib.sha256(archive.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="observation|snapshot|roles/targets|aliasing|schemas|prepared graph"):
        prepare(api.StagedProgram(stage, freeze_readonly_buffers=True), max_initialization_bytes=65536)


def test_host_observation_has_exact_closed_schema(readonly_capture):
    _, capture, _, _ = readonly_capture
    record = json.loads((capture / "meta.json").read_bytes())["export_state"]
    trace = json.loads((capture / "frontend-trace.json").read_bytes())
    assert validate_observation(record, trace=trace, source_sha256=record["source_sha256"]) == record
    changed = copy.deepcopy(record)
    changed["consent"] = True  # producer metadata cannot grant caller consent
    with pytest.raises(ValueError, match="schema"):
        validate_observation(changed, trace=trace, source_sha256=record["source_sha256"])


@pytest.mark.parametrize("readonly_capture", ("linear",), indirect=True)
@pytest.mark.parametrize("change", ("unknown", "property", "bound", "dtype"))
def test_marker_legalization_refuses_unknown_or_malformed_forms_without_editing_source(readonly_capture, change):
    from xdsl.dialects import builtin

    from merlin.compiler.quantization.legalize import legalize
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    _, _, stage, _ = readonly_capture
    module = parse_mlir_text(load_compile_inputs(stage).program)
    marker = next(op for op in module.walk() if isinstance(op, builtin.UnregisteredOp))
    if change == "unknown":
        marker.attributes["op_name__"] = builtin.StringAttr("quant_ext.unmodeled")
    elif change == "property":
        marker.properties["rounding"] = builtin.StringAttr("toward_zero")
    elif change == "bound":
        marker.properties["quant_max"] = builtin.IntegerAttr(128, builtin.i64)
    else:
        marker.properties["output_dtype"] = builtin.StringAttr("uint8")
    original = text(module, generic=True)
    with pytest.raises(ValueError, match="captured quantization"):
        legalize(module)
    assert text(module, generic=True) == original


def test_public_cli_freeze_option_uses_the_same_original_caller_contract(readonly_capture, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.service_cli import compile_main

    _, _, stage, _ = readonly_capture
    observed = []

    def compile(source, **kwargs):
        assert type(source) is api.StagedProgram and source.freeze_readonly_buffers is True
        prepared, binding = prepare(source, max_initialization_bytes=65536)
        observed.append(binding.record()["buffer_freeze"])
        assert prepared.kind == "mlir"
        return SimpleNamespace(manifest=tmp_path / "manifest.json", to_dict=lambda: {})

    monkeypatch.setattr(api, "compile", compile)
    assert (
        compile_main(
            [
                str(stage),
                "--input-kind",
                "compiler-stage",
                "--freeze-readonly-buffers",
                "--backend",
                "fixture.selected",
                "--output",
                str(tmp_path / "compiled"),
            ]
        )
        == 0
    )
    assert len(observed) == 1 and observed[0]["consent"] is True
    with pytest.raises(SystemExit):
        compile_main(
            [
                str(stage),
                "--input-kind",
                "mlir",
                "--freeze-readonly-buffers",
                "--backend",
                "fixture.selected",
                "--output",
                str(tmp_path / "refused"),
            ]
        )


def test_actual_producer_observer_refuses_mutation_unknown_effects_and_source_aliases(tmp_path):
    python, producer = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not producer:
        pytest.skip("select the actual pinned producer environment")
    from merlin.common.paths import module_source_path

    script = tmp_path / "observe.py"
    script.write_text("""import importlib.util, json, operator, sys
from pathlib import Path
import torch, m2m
spec=importlib.util.spec_from_file_location("observer",sys.argv[1]); observer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
root=Path(sys.argv[2])
class Readonly(torch.nn.Module):
    def __init__(self,alias=False):
        super().__init__(); data=torch.arange(4,dtype=torch.float32)
        self.register_buffer("b",data); self.register_buffer("c",data if alias else data.clone())
    def forward(self,x): return x+self.b+self.c
class Mutating(torch.nn.Module):
    def __init__(self): super().__init__(); self.register_buffer("counter",torch.zeros(4))
    def forward(self,x): self.counter.add_(1); return x+self.counter
def refuse(call,word):
    try: call()
    except observer.ExportStateUnavailable as error: assert word in str(error),str(error)
    else: raise AssertionError("unsupported state was accepted: "+word)
# Raw user outputs alone do not establish immutability: raw schema writes and
# decomposed mutation outputs are independently rejected by the observer.
raw=torch.export.export(Mutating(),(torch.ones(4),))
assert all(spec.kind.name=="USER_OUTPUT" for spec in raw.graph_signature.output_specs)
refuse(lambda:observer._effects(raw),"writes")
refuse(lambda:observer._outputs(raw.run_decompositions()),"non-user output")
for name,model in (("ok",Readonly()),("alias",Readonly(True)),("runtime_alias",Readonly()),("mutating",Mutating())):
    directory=root/name; directory.mkdir()
    inputs=(model.b,) if name=="runtime_alias" else (torch.ones(4),)
    result=m2m.convert(model.eval(),inputs,backend="fx_importer",level="linalg-on-tensors",
                       weights_path=str(directory/"weights.safetensors"),capture_trace=True)
    assert result.ok
    call=lambda:observer.observe_export_state(result,inputs,directory=directory)
    if name in ("alias","runtime_alias"): refuse(call,"aliases"); continue
    if name=="mutating": refuse(call,"non-user output"); continue
    observed=call(); assert any(item["kind"]=="BUFFER" for item in observed["inputs"])
    placeholders=[node for node in result.exported_program.graph.nodes if node.op=="placeholder"]
    old=placeholders[0].target; placeholders[0].target=placeholders[1].target
    refuse(call,"placeholder target")
    placeholders[0].target=old
    # A harmless observed live spelling is recorded, but duplicate placeholder
    # names or target/ordinal swaps cannot be normalized into correspondence.
    old_name=placeholders[0].name; placeholders[0].name=placeholders[1].name
    refuse(call,"placeholder rosters")
    placeholders[0].name=old_name
    result.exported_program.state_dict["b"].add_(1)
    refuse(call,"original bytes")
    result.exported_program.state_dict["b"].sub_(1)
    node=next(node for node in result.exported_program.graph.nodes if node.op=="call_function")
    target=node.target; node.target=torch.ops.aten.sub.Tensor
    refuse(call,"imported prepared graph")
    node.target=target
    node.target=operator.neg
    refuse(call,"unknown operator schema")
    result.exported_program=None
    refuse(call,"actual prepared export")
print("actual producer effect/alias/payload refusals passed")
""")
    result = subprocess.run(
        [python, "-B", str(script), str(module_source_path("merlin.compiler.frontend.torch.state")), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": producer},
    )
    assert result.returncode == 0, result.stderr[-10000:]
    assert "refusals passed" in result.stdout
