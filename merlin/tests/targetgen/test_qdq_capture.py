"""Original scalar precision and typed source/IR bindings must survive Q/DQ capture."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import parse_mlir_text
from merlin.targetgen.qdq_capture import QDQCaptureError, check_source, scalar_scale


@pytest.mark.parametrize("value", [0.125, 1.0, 2.0**-126])
def test_exact_positive_f32_scales_are_supported(value):
    assert scalar_scale(value, quantize=True)["python_hex"] == value.hex()


@pytest.mark.parametrize(
    "value,reason",
    [
        (0.0023796888063332034, "loses precision"),  # retained native 50 versus 49 counterexample
        (0.0, "positive"),
        (-0.0, "positive"),
        (-1.0, "positive"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (2.0**-149, "reciprocal"),
        (1e100, "outside f32"),
        (True, "Python float"),
    ],
)
def test_incompatible_scales_are_refused(value, reason):
    with pytest.raises(QDQCaptureError, match=reason):
        scalar_scale(value, quantize=True)


def test_dequantization_does_not_invent_a_reciprocal_exception():
    assert scalar_scale(0.0, quantize=False)["f32_bits"] == "00000000"
    assert scalar_scale(-0.0, quantize=False)["f32_bits"] == "80000000"


def example():
    graphs = {}
    for stage in ("original", "prepared"):

        def node(index, op, dtype, args, target):
            identity = f"g:{stage}:root:n{index}"
            return {
                "id": identity,
                "op": op,
                "target": target,
                "args": args,
                "kwargs": {},
                "origin_node_ids": [f"g:original:root:n{index}"] if stage == "prepared" else [],
                "results": [{"id": identity + ":v0", "kind": "tensor", "shape": [2], "dtype": dtype}],
            }

        def ref(i):
            return {"node_id": f"g:{stage}:root:n{i}", "value_id": f"g:{stage}:root:n{i}:v0"}

        graphs[stage] = {
            "status": "complete",
            "nodes": [
                node(0, "placeholder", "float32", [], "x"),
                node(1, "placeholder", "float32", [], "other"),
                node(
                    2,
                    "call_function",
                    "int8",
                    [ref(0), 0.125, 0, -128, 127, {"kind": "dtype", "value": "torch.int8"}],
                    "quantized_decomposed.quantize_per_tensor.default",
                ),
                node(3, "output", "int8", [[ref(2)]], "output"),
            ],
        }
    program = """func.func @forward(%x: tensor<2xf32>, %other: tensor<2xf32>) -> tensor<2xi8> {
      %s = arith.constant 0.125 : f32
      %scale = tensor.splat %s : tensor<f32>
      %z = arith.constant 0 : i64
      %zero = tensor.splat %z : tensor<i64>
      %q = "quant_ext.quantize_per_tensor"(%x, %scale, %zero)
        <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}>
        {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2"],
         prov.aten = "quantized_decomposed.quantize_per_tensor.default"}
        : (tensor<2xf32>, tensor<f32>, tensor<i64>) -> tensor<2xi8>
      return %q : tensor<2xi8>
    }"""
    return {"graphs": graphs}, program


def test_original_scalar_call_and_actual_ssa_match():
    trace, program = example()
    sites = check_source(trace, parse_mlir_text(program))
    assert len(sites) == 1 and sites[0]["scale"]["f32_bits"] == "3e000000"


@pytest.mark.parametrize(
    "mutation",
    [
        "precision",
        "original_input",
        "prepared_input",
        "original_output",
        "zero",
        "scale",
        "bounds",
        "overload",
        "identity",
        "attribute",
    ],
)
def test_mutated_source_or_emitted_binding_is_refused(mutation):
    trace, program = example()
    if mutation == "precision":
        for graph in trace["graphs"].values():
            graph["nodes"][2]["args"][1] = 0.0023796888063332034
    elif mutation in {"original_input", "prepared_input"}:
        stage = mutation.split("_")[0]
        trace["graphs"][stage]["nodes"][2]["args"][0] = {
            "node_id": f"g:{stage}:root:n1",
            "value_id": f"g:{stage}:root:n1:v0",
        }
    elif mutation == "original_output":
        trace["graphs"]["original"]["nodes"][3]["args"] = [
            [{"node_id": "g:original:root:n0", "value_id": "g:original:root:n0:v0"}]
        ]
    else:
        substitutions = {
            "zero": ("arith.constant 0 : i64", "arith.constant 1 : i64"),
            "scale": ("arith.constant 0.125", "arith.constant 0.25"),
            "bounds": ("quant_max = 127", "quant_max = 126"),
            "overload": ("per_tensor.default", "per_tensor.tensor"),
            "identity": ('["g:original:root:n2"]', '["g:original:root:n0"]'),
            "attribute": ("{prov.source_node_ids", '{unreviewed = "yes", prov.source_node_ids'),
        }
        before, after = substitutions[mutation]
        assert before in program
        program = program.replace(before, after)
    with pytest.raises(QDQCaptureError):
        check_source(trace, parse_mlir_text(program))


def test_same_typed_tensor_inputs_cannot_be_swapped_in_ir():
    trace, program = example()
    with pytest.raises(QDQCaptureError, match="data operand"):
        check_source(trace, parse_mlir_text(program.replace('per_tensor"(%x,', 'per_tensor"(%other,')))


def test_missing_original_graph_cannot_qualify_a_historical_capture():
    trace, program = example()
    trace["graphs"].pop("original")
    with pytest.raises(QDQCaptureError, match="complete original"):
        check_source(trace, parse_mlir_text(program))


def test_serialized_receipt_status_is_not_an_input_to_source_check():
    trace, program = example()
    forged = copy.deepcopy(trace)
    forged.update(status="verified", source_closure_verified=True)
    forged["graphs"]["original"]["nodes"][2]["args"][1] = 0.25
    with pytest.raises(QDQCaptureError, match="arguments changed"):
        check_source(json.loads(json.dumps(forged)), parse_mlir_text(program))


@pytest.mark.parametrize("mutation", [None, "dirty", "untracked", "foreign_import"])
def test_executed_producer_inventory_is_checked_beyond_a_pin_label(tmp_path, monkeypatch, mutation):
    from merlin.integrations import model2mlir
    from merlin.targetgen.qdq_capture_identity import producer_identity

    producer = tmp_path / "producer"
    (producer / "m2m").mkdir(parents=True)
    (producer / "m2m/__init__.py").write_text("")
    helper = producer / "m2m/helper.py"
    helper.write_text("value = 1\n")
    command = ["git", "-C", str(producer), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*command, "init", "-q"], check=True)
    subprocess.run([*command, "add", "."], check=True)
    subprocess.run([*command, "commit", "-qm", "original"], check=True)
    monkeypatch.setitem(sys.modules, "m2m", SimpleNamespace(__file__=str(producer / "m2m/__init__.py")))
    monkeypatch.setattr(model2mlir, "capture_pin", lambda root: {"ok": True})
    if mutation == "dirty":
        helper.write_text("value = 2\n")
    elif mutation == "untracked":
        (producer / "m2m/shadow.py").write_text("")
    elif mutation == "foreign_import":
        monkeypatch.setitem(sys.modules, "m2m.helper", SimpleNamespace(__file__=str(tmp_path / "foreign.py")))
    if mutation is None:
        assert "m2m/helper.py" in producer_identity(producer)["sources"]
    else:
        with pytest.raises(ValueError, match="differs|foreign"):
            producer_identity(producer)


@pytest.mark.parametrize("mutation", [None, "wheel", "installed_payload", "import_owner"])
def test_torch_identity_checks_archive_and_installed_bytes(tmp_path, monkeypatch, mutation):
    from merlin.targetgen import qdq_capture_identity as identity

    installed = tmp_path / "site-packages"
    (installed / "torch").mkdir(parents=True)
    owner = installed / "torch/__init__.py"
    owner.write_bytes(b"# original Torch source\n")
    archive = tmp_path / "torch.whl"
    with zipfile.ZipFile(archive, "w") as wheel:
        wheel.writestr("torch/__init__.py", owner.read_bytes())
        wheel.writestr("torch-2.10.0+cpu.dist-info/RECORD", b"not used as payload authority")
    requirements = tmp_path / "build_tools/toolchains/lean-torch-requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text(f"torch @ selected-wheel#sha256={hashlib.sha256(archive.read_bytes()).hexdigest()}\n")
    torch = SimpleNamespace(
        __file__=str(owner), __version__="2.10.0+cpu", version=SimpleNamespace(git_version="source")
    )
    distribution = SimpleNamespace(version="2.10.0+cpu", locate_file=lambda name: installed / name)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(identity, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(identity.importlib.metadata, "distribution", lambda name: distribution)
    if mutation == "wheel":
        archive.write_bytes(b"different wheel")
    elif mutation == "installed_payload":
        owner.write_bytes(b"# changed source, same version metadata\n")
    elif mutation == "import_owner":
        torch.__file__ = str(tmp_path / "foreign.py")
    if mutation is None:
        assert "torch/__init__.py" in identity.torch_identity(archive)["installed_members"]
    else:
        with pytest.raises(ValueError, match="differs"):
            identity.torch_identity(archive)


def test_native_capture_and_worker_refuse_lossy_scale(tmp_path):
    selections = {
        name: os.environ.get(name)
        for name in (
            "MERLIN_QDQ_CAPTURE_PYTHON",
            "MERLIN_QDQ_PRODUCER",
            "MERLIN_QDQ_TORCH_WHEEL",
            "MERLIN_QDQ_CAPTURE_DEPS",
            "MERLIN_COMPILER_PYTHON",
            "MERLIN_LEAN_SDK_BIN",
        )
    }
    missing = [name for name, value in selections.items() if not value or not Path(value).exists()]
    if missing:
        message = f"explicit Q/DQ native tools missing: {missing}"
        if os.environ.get("MERLIN_LEAN_REQUIRED") == "1":
            pytest.fail(message)
        pytest.skip(message)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(repo_root() / "src"),
                selections["MERLIN_QDQ_PRODUCER"],
                selections["MERLIN_QDQ_CAPTURE_DEPS"],
            )
        ),
    }
    python = selections["MERLIN_QDQ_CAPTURE_PYTHON"]
    script = repo_root() / "build_tools/scripts/qualify_qdq_capture.py"
    result = subprocess.run(
        [
            python,
            "-B",
            str(script),
            "capture",
            "--output",
            str(tmp_path / "native"),
            "--producer",
            selections["MERLIN_QDQ_PRODUCER"],
            "--torch-wheel",
            selections["MERLIN_QDQ_TORCH_WHEEL"],
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [
            python,
            "-B",
            str(script),
            "replay-source",
            "--output",
            str(tmp_path / "native"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The command-line compiler selection must also reach textual materialization,
    # even when no matching compiler environment selection is inherited.
    lower_env = {**os.environ, "PYTHONPATH": str(repo_root() / "src")}
    lower_env.pop("MERLIN_COMPILER_PYTHON", None)
    lower_env.pop("MERLIN_LEAN_SDK_BIN", None)
    for mode in ("lower", "replay-lowering"):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(script),
                mode,
                "--output",
                str(tmp_path / "native"),
                "--compiler-python",
                selections["MERLIN_COMPILER_PYTHON"],
                "--sdk",
                selections["MERLIN_LEAN_SDK_BIN"],
            ],
            env=lower_env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    for scale, expected in (("0.125", "matched"), ("0.0023796888063332034", "unsupported")):
        loader = tmp_path / f"loader-{expected}.py"
        loader.write_text(f"""import torch
import torch.ao.quantization.fx._decomposed
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(x, {scale}, 0, -128, 127, torch.int8)
def get_model_and_inputs():
    return Model(), (torch.tensor([0.11779459565877914]),)
""")
        output = tmp_path / f"worker-{expected}"
        result = subprocess.run(
            [
                python,
                "-B",
                "-m",
                "merlin.targetgen._m2m_capture_worker",
                "--loader",
                str(loader),
                "--out",
                str(output),
                "--dtype",
                "fp32",
                "--already-quantized",
                "--qualify-qdq",
                "--m2m-dir",
                selections["MERLIN_QDQ_PRODUCER"],
                "--qdq-torch-wheel",
                selections["MERLIN_QDQ_TORCH_WHEEL"],
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        diagnostic = (
            (output / "qdq-source-check.json").read_text()
            if (output / "qdq-source-check.json").is_file()
            else "no Q/DQ source diagnostic"
        )
        assert result.returncode == (0 if expected == "matched" else 3), result.stdout + result.stderr + diagnostic
        report = json.loads((output / "qdq-source-check.json").read_bytes())
        assert report["status"] == expected
        assert json.loads((output / "meta.json").read_bytes())["ok"] == (expected == "matched")
        if expected == "unsupported":
            assert "loses precision" in report["reason"]
            assert (output / "qdq-source.mlir").is_file()
