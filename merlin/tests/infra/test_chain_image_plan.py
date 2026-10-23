"""Exercise real models/input/capture planning and closed reference admission."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.model_check import chain_image_data as data
from merlin.model_check import chain_image_plan as planning
from merlin.model_check.chain_image_objects import pin


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def case(tmp_path, monkeypatch):
    stage, build = tmp_path / "stage", tmp_path / "build"
    stage.mkdir()
    (build / "cgen").mkdir(parents=True)
    (stage / "model.mlir").write_text(
        "module {func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {return %x : tensor<2xf32>}}"
    )
    save(stage / "input_order.json", {"x": 0})
    save(
        stage / "weights.safetensors.manifest.json",
        {"0": {"kind": "input", "name": "x", "shape": [2], "dtype": "float32"}},
    )
    values = np.asarray([1.0, 2.0], np.float32)
    np.savez(stage / "inputs.npz", in0=values)
    np.savez(tmp_path / "inputs.npz", value=values)
    (build / "cgen/model_gen.h").write_text("#define MERLIN_N_ARGS 2\n#define MERLIN_N_OUTPUTS 1\n")
    model = {
        "schema": "merlin.model_check.models.v1",
        "target": "fixture",
        "models": {
            "sample": {
                "workload": "public-fixture",
                "chain": {"kind": "single"},
                "programs": [{"name": "forward", "steps": 1, "inputs": [{"from": "input", "key": "value"}]}],
                "judged": {"kind": "logits", "outputs": [{"program": "forward", "output": 0}]},
            }
        },
    }
    models_path = tmp_path / "models.yaml"
    models_path.write_text(yaml.safe_dump(model))
    receipt = save(tmp_path / "receipt.json", {})
    recipe = save(tmp_path / "recipe.json", {})
    # Program reopening itself is independently covered by the real-object
    # namespace/link tests; this fixture tests the actual parser/planner route.
    monkeypatch.setattr(
        planning,
        "completed_program",
        lambda _: {
            "root": build,
            "receipt_pin": pin(receipt),
            "recipe_pin": pin(recipe),
            "receipt": {"inputs": {"capture": str(stage)}},
        },
    )
    selection = {
        "model": "sample",
        "models": pin(models_path),
        "inputs": pin(tmp_path / "inputs.npz"),
        "programs": {"forward": pin(receipt)},
        "transport": "all_element_comparison_v1",
    }
    work = tmp_path / "prepared"
    work.mkdir()
    plan = planning.prepare(selection, work)
    bundle = tmp_path / "bundle"
    (bundle / "configuration").mkdir(parents=True)
    (bundle / "configuration/models.yaml").write_bytes(models_path.read_bytes())
    case_path = bundle / "sample/visible/v00/int8.npz"
    case_path.parent.mkdir(parents=True)
    np.savez(case_path, out0=values)
    sums = bundle / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{pin(p)['sha256']}  {p.relative_to(bundle)}\n" for p in (bundle / "configuration/models.yaml", case_path)
        )
    )
    save(bundle / "MANIFEST.json", {"seal": {"sums": "SHA256SUMS", "sums_sha256": pin(sums)["sha256"]}})
    identity = {
        "models_config_sha256": pin(models_path)["sha256"],
        "bundle_manifest_sha256": pin(bundle / "MANIFEST.json")["sha256"],
        "bundle_sums_sha256": pin(sums)["sha256"],
    }
    expected = tmp_path / "forward.0.npz"
    np.savez(expected, out0=values)
    output = {
        "output": 0,
        "shape": [2],
        "dtype": "<f4",
        "element_type": "f32",
        "sha256_of_array_bytes": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    manifest = {
        "schema": "merlin.operator.full_declared_output_reference.v1",
        "status": "complete",
        "model": "sample",
        "input_id": "visible/v00",
        "input": selection["inputs"],
        "selected_identity": identity,
        "implementation": {"numeric_contract": {}},
        "judged_crossjoin": {"sealed_npz": pin(case_path), "bit_exact": True, "outputs": 1},
        "programs": [
            {
                "program": "forward",
                "stage": str(stage),
                "steps_declared": 1,
                "capture_program": {k: v for k, v in pin(stage / "model.mlir").items() if k != "bytes"},
                "steps": [{"step": 0, "outputs": [output], "outputs_file": pin(expected)}],
            }
        ],
    }
    reference = save(tmp_path / "reference.json", manifest)
    selection.update(reference=pin(reference), reference_identity=identity, bundle=str(bundle))
    return selection, plan, tmp_path, manifest


def test_actual_prepare_and_native_absent_policy_reference(case):
    selected, plan, root, _ = case
    assert plan["schedule"][0]["transfers"][0]["bytes"] == 8
    assert plan["input_blob"]["bytes"] == 8
    result = data.references(selected, plan, root)
    assert result["seal_binding"]["source"] == "sealed_int8"
    assert Path(result["blob"]["path"]).read_bytes() == np.asarray([1, 2], np.float32).tobytes()


@pytest.mark.parametrize("changed", ["portable", "sealed", "output", "shape", "input", "identity"])
def test_reference_does_not_transfer_other_policy_or_bytes(case, changed):
    selected, plan, root, manifest = case
    if changed == "portable":
        plan["host_math_policy"] = "portable_exp_trig"
    elif changed == "sealed":
        p = Path(manifest["judged_crossjoin"]["sealed_npz"]["path"])
        p.write_bytes(p.read_bytes() + b"changed")
    elif changed == "output":
        manifest["programs"][0]["steps"][0]["outputs"][0]["sha256_of_array_bytes"] = "0" * 64
    elif changed == "shape":
        manifest["programs"][0]["steps"][0]["outputs"][0]["shape"] = [1, 2]
    elif changed == "input":
        manifest["input"] = {**manifest["input"], "sha256": "0" * 64}
    else:
        selected["reference_identity"] = {"forged": True}
    if changed in {"output", "shape", "input"}:
        path = save(Path(selected["reference"]["path"]), manifest)
        selected["reference"] = pin(path)
    with pytest.raises(ValueError):
        data.references(selected, plan, root)


def test_stream_reference_and_guest_select_each_distinct_frame(case):
    from merlin.model_check import chain

    selected, _, root, _ = case
    path = Path(selected['models']['path'])
    declaration = yaml.safe_load(path.read_text())
    program = declaration['models']['sample']['programs'][0]
    program.update(steps=3, inputs=[{'from': 'stream', 'key': 'value', 'axis': 0, 'keepdims': False}])
    path.write_text(yaml.safe_dump(declaration))
    frames = np.asarray([[1, 2], [7, -3], [19, 5]], dtype=np.float32)
    inputs = root / 'frames.npz'
    np.savez(inputs, value=frames)
    selected = dict(selected, models=pin(path), inputs=pin(inputs))
    directory = root / 'stream-plan'
    directory.mkdir()
    plan = planning.prepare(selected, directory)
    blob = Path(plan['input_blob']['path']).read_bytes()
    model = plan['model']
    assert len(plan['schedule']) == 3
    for step, command in enumerate(plan['schedule']):
        transfer, = command['transfers']
        decoded = np.frombuffer(blob, dtype=np.float32, count=2, offset=transfer['source_offset'])
        expected = frames[step]
        np.testing.assert_array_equal(decoded, expected)
        reference = chain.bind(model.programs[0], model, {'value': frames}, step, chain.ChainRun())
        np.testing.assert_array_equal(reference['in0'], expected)


@pytest.mark.parametrize('axis,keepdims,expected', [(1, False, [2, 5]), (-1, True, [[2], [5]])])
def test_stream_axis_and_rank_are_explicit(axis, keepdims, expected):
    from merlin.model_check.chain import stream_input

    binding = {'from': 'stream', 'key': 'frames', 'axis': axis, 'keepdims': keepdims}
    actual = stream_input(binding, {'frames': np.array([[1, 2, 3], [4, 5, 6]])}, 1, 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('change', ['short', 'extra', 'axis', 'step', 'carry', 'rank_policy'])
def test_stream_never_silently_repeats_or_truncates(change):
    from merlin.model_check.chain import ChainError, stream_input
    from merlin.model_check.models import ModelsFileError

    binding = {'from': 'stream', 'key': 'frames', 'axis': 0, 'keepdims': False}
    frames, step = np.arange(6).reshape(3, 2), 1
    if change == 'short': frames = frames[:2]
    elif change == 'extra': frames = np.arange(8).reshape(4, 2)
    elif change == 'axis': binding['axis'] = 2
    elif change == 'step': step = 3
    elif change == 'carry': binding['carry'] = 0
    else: binding.pop('keepdims')
    with pytest.raises((ChainError, ModelsFileError)):
        stream_input(binding, {'frames': frames}, step, 3)
