"""One joined capture/ABI/lineage check over an actual recurrent three-stage model."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.session_bundle import load
from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen.application_inventory import verify_capture_receipt
from merlin.targetgen.quant_recipe import digest as recipe_digest

LOADER = """
import torch
from torch import nn
from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session

class Prefix(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
    def forward(self, x):
        return self.layer(x)

class Step(nn.Module):
    def forward(self, context, state):
        if hasattr(self, "layer"):
            context = self.layer(context)
        result = context + state
        return result, result

class Final(nn.Module):
    def forward(self, state):
        return state[:, :2]

class Capture:
    def external_runtime_session(self):
        prefix, step, final = Prefix().eval(), Step().eval(), Final().eval()
        value = torch.randn(1, 4)
        with torch.no_grad():
            context = prefix(value)
        state = torch.zeros_like(context)
        child = dict(kind="generic_recurrent", paper_ready=False, steps=2,
                     stages=["step"],
                     stage_schedule=[dict(name="step", steps=2, execution="compiled_recurrent", timed=True)],
                     states=[dict(name="state", input_index=1, output_index=1)],
                     streams=[], quality=dict(output_index=0),
                     provenance=dict(synthetic_inputs=True, full_checkpoint=False))
        programs = (ExternalRuntimeProgram("prefix", prefix, (value,), 1),
                    ExternalRuntimeProgram("step", step, (context, state), 2, child),
                    ExternalRuntimeProgram("final", final, (context * 2,), 1))
        metadata = dict(kind="generic_recurrent", paper_ready=False,
                        stages=[p.name for p in programs], quality_program="step", states=["state"],
                        stage_schedule=[dict(name=p.name, steps=p.steps, execution="compiled", timed=True)
                                        for p in programs],
                        provenance=dict(synthetic_inputs=True, full_checkpoint=False),
                        bindings=[dict(name="context", **{"from":dict(program="prefix", output_index=0),
                                       "to":dict(program="step", input_index=0)}),
                                  dict(name="result", **{"from":dict(program="step", output_index=0),
                                       "to":dict(program="final", input_index=0)})])
        return make_external_runtime_session(version=2, programs=programs, metadata=metadata)

def get_model_and_inputs():
    return Capture(), ()
"""


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["fp32", "int8"])
@pytest.mark.parametrize("variant", ["plain", "shared", "mixed", "paper", "mixed_staged"])
def test_worker_captures_the_entire_declared_session_with_owned_sidecars(tmp_path, dtype, variant):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    paper = variant == "paper"
    staged = variant == "mixed_staged"
    if staged and dtype != "int8":
        pytest.skip("FP32 staging before a recipe is the int8 deployment path")
    shared = variant in {"shared", "paper"}
    text = LOADER
    if shared:
        text = text.replace("value = torch.randn", "step.layer = prefix.layer\n        value = torch.randn")
    if paper:
        # A paper-ready recurrent stage that the recipe quantizes: the writer only ever sees the
        # quantized program, so the worker must supply the pre-quantization reference trajectory.
        text = text.replace("paper_ready=False", "paper_ready=True")
    elif variant in {"mixed", "mixed_staged"}:
        text = text.replace(
            "self.layer = nn.Linear(4, 4)",
            "self.layer = nn.Linear(4, 4)\n        self.bf16_layer = nn.Linear(4, 4).bfloat16()",
        )
        text = text.replace("return self.layer(x)", "return self.layer(x) + self.bf16_layer(x.bfloat16()).float()")
    loader.write_text(text)
    out = tmp_path / "capture"
    recipe_args = []
    if dtype == "int8":
        tensor = dict(dtype="int8", granularity="tensor", symmetric=True, block=None, mode="static")
        recipe = dict(
            schema="quant_recipe_v1",
            target="t",
            unit="matrix",
            status="derived",
            weight={**tensor, "quant_min": -127, "quant_max": 127},
            activation={**tensor, "quant_min": -128, "quant_max": 127},
            families=["contraction"],
            bias_domain="accumulator",
            software_numerical_engine="integer_reference",
        )
        recipe["recipe_sha256"] = recipe_digest(recipe)
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(recipe))
        recipe_args = ["--recipe", str(recipe_path)]
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            dtype,
            *recipe_args,
            *(["--stage-fp32"] if staged else []),
            "--seed",
            "7",
            "--materialize-bundle",
            "--out",
            "capture",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    session = load(out)  # Checks ordered stage/weight-prefixed ABI and carried-state bindings.
    assert session.program_names == ("prefix", "step", "final")
    assert len(session.bindings) == 2
    receipt = json.loads((out / "session-receipt.json").read_bytes())
    assert receipt["agentic"] is False
    assert receipt["determinism"]["seed"] == 7
    if dtype == "int8":
        assert receipt["recipe_sha256"] == recipe["recipe_sha256"]
        deployments = {row["name"]: row.get("numeric_deployment") for row in receipt["programs"]}
        prefix = deployments["prefix"]
        assert prefix["form"] == ("fp32_staged_then_recipe_w8a8" if staged else "recipe_w8a8")
        if staged:
            # The deployment states the exact-source FP32 retyping it quantized after.
            staging = prefix["precision_staging"]
            assert staging["target_dtype"] == "torch.float32" and staging["audit_status"] == "complete"
            assert staging["non_target_floating_values"] == 0 and staging["retyped_dtype_decisions"] >= 1
        assert prefix["contractions"]["linear"]["integerized"] == prefix["integer_contractions"] >= 1
        # The mixed variant's bf16 layer is selected but kept as dequantized float, and says why.
        kept = 1 if variant == "mixed" else 0
        assert prefix["contractions"]["linear"]["selected_kept_float"] == kept
        assert sum(prefix["selected_kept_float_reasons"].values()) == kept
        if kept:
            assert prefix["selected_kept_float_reasons"] == {"preserve_float_qdq:linear:torch.bfloat16": 1}
        assert [row["precision_selection"] for row in receipt["programs"]] == [
            "recipe",
            "recipe" if shared else "no_recipe_work",
            "no_recipe_work",
        ]
    if paper:
        import numpy as np
        import yaml

        step = out / "stages" / "step"
        contract = yaml.safe_load((step / "session_contract.yaml").read_text())
        assert contract["paper_ready"] is True
        assert contract["quality"]["reference"] == "eager_fp32"
        with np.load(step / "session_quality_fp32.npz") as quality, np.load(step / "session_goldens.npz") as golden:
            reference, observed = quality[contract["quality"]["key"]], golden[contract["correctness"]["key"]]
        assert reference.shape == observed.shape and np.isfinite(reference).all()
        if dtype == "int8":
            # Independently generated: the float trajectory, not the quantized program's own golden.
            assert not np.array_equal(reference, observed)
        else:
            np.testing.assert_array_equal(reference, observed)
    for program in session.programs:
        stage = program.bundle
        assert verify_capture_receipt(stage / "model.mlir")["status"] == "verified_materialized"
        assert f'prov.weights_file = "{stage / "weights.safetensors"}"' in (stage / "model.mlir").read_text()
        assert json.loads((stage / "frontend-trace.json").read_bytes())["status"] == "complete"
        metadata = json.loads((stage / "meta.json").read_bytes())
        assert metadata["input_abi"]
        assert len(metadata["output_abi"]) == (2 if program.name == "step" else 1)
        assert all(row["dtype"] == "f32" for row in metadata["output_abi"])
        assert metadata["ok"] and metadata["opaque"] == 0
        assert metadata["loader_provenance"]["synthetic_inputs"] is True
        assert metadata["framework_catalog"]["status"] == "available"
        if dtype == "int8" and (program.name == "prefix" or (shared and program.name == "step")):
            assert metadata["recipe_sha256"] == recipe["recipe_sha256"]
            assert metadata["integerization_receipt"]["golden_agreement"]["status"] == "passed"
            assert metadata["integerization_receipt"]["golden_agreement"]["reference"] == "pt2e_integer"
            assert metadata["integerization_receipt"]["exported_integer_mm_count"] > 0
            if variant == "mixed":
                partition = metadata["integerization_receipt"]["precision_decision_counts"]
                assert partition == {"integerized_i32": 1, "preserve_float_qdq": 1, "unresolved": 0}
                executed = metadata["integerization_receipt"]["golden_agreement"]["executed_contractions"]
                assert executed["total"] == 1
                assert executed["selected"] == executed["observed"] == 2
        elif dtype == "int8":
            assert metadata["dtype"] == "fp32"
            assert metadata["recipe_sha256"] is None
