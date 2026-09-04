"""Multi-program sessions validate cross-stage ABIs and generate model-agnostic C glue."""
from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from merlin.llvmlower import session_bundle


def _program(root, name: str, *, child_steps: int | None = None):
    bundle = root / "stages" / name
    bundle.mkdir(parents=True)
    (bundle / "model.mlir").write_text(
        "module { func.func @forward(%arg0: tensor<2xf32>) -> tensor<2xf32> { "
        "return %arg0 : tensor<2xf32> } }\n", encoding="utf-8")
    (bundle / "inputs.npz").write_bytes(b"inputs")
    (bundle / "weights.safetensors").write_bytes(b"weights")
    (bundle / "weights.safetensors.manifest.json").write_text("{}\n", encoding="utf-8")
    if child_steps is not None:
        (bundle / "session_contract.yaml").write_text(yaml.safe_dump({
            "version": 1, "kind": "autoregressive_decode", "paper_ready": True,
            "stages": [name], "steps": child_steps,
            "states": [{"name": "state", "input_arg": 0, "output_index": 0}],
            "streams": [], "quality": {"scope": "trajectory"},
        }), encoding="utf-8")
    return bundle


def _root(tmp_path, *, mismatch: bool = False):
    root = tmp_path / "session"
    _program(root, "prefill")
    _program(root, "decode", child_steps=3)
    contract = {
        "version": 2, "kind": "autoregressive_decode", "paper_ready": True,
        "stages": ["prefill", "decode"],
        "stage_schedule": [
            {"name": "prefill", "steps": 1, "execution": "compiled", "timed": True},
            {"name": "decode", "steps": 3, "execution": "compiled_recurrent", "timed": True},
        ],
        "programs": [
            {"name": "prefill", "bundle": "stages/prefill", "steps": 1},
            {"name": "decode", "bundle": "stages/decode", "steps": 3},
        ],
        "bindings": [{
            "name": "state", "from": {"program": "prefill", "output_index": 0},
            "to": {"program": "decode", "input_arg": 0},
        }],
        "states": [{"name": "state"}], "streams": [],
        "quality": {"scope": "trajectory", "program": "decode"},
    }
    (root / "session_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")
    return root


def _mock_abis(monkeypatch, *, target_shape=None):
    monkeypatch.setattr(session_bundle, "parse_forward_signature", lambda path: [
        (target_shape or [2], "f32")])
    monkeypatch.setattr(session_bundle, "forward_signature", lambda path: (
        [([2], "f32")], [([2], "f32")]))


def test_load_validates_stage_order_state_routes_and_abis(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _mock_abis(monkeypatch)
    session = session_bundle.load(root)
    assert session.program_names == ("prefill", "decode")
    assert session.bindings[0].source_program == "prefill"
    assert session.quality_program == "decode"

    _mock_abis(monkeypatch, target_shape=[3])
    with pytest.raises(ValueError, match="ABI mismatch"):
        session_bundle.load(root)


def test_load_rejects_eager_primary_stage_and_escaping_bundle(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _mock_abis(monkeypatch)
    contract_path = root / "session_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text())
    contract["stage_schedule"][0]["execution"] = "eager_reference_initial_state"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="timed compiled code"):
        session_bundle.load(root)

    contract["stage_schedule"][0]["execution"] = "compiled"
    contract["programs"][0]["bundle"] = "../outside"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="contained by the session bundle"):
        session_bundle.load(root)


def test_generate_emits_unique_entrypoints_and_compilable_scheduler(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _mock_abis(monkeypatch)

    def fake_generate(_bundle, out, _inputs, *, ciface_name, invoke_name):
        out.mkdir(parents=True)
        (out / "weights.bin").write_bytes(b"same-weights")
        (out / "model_gen.h").write_text("/* fixture */\n", encoding="utf-8")
        (out / "model_io.h").write_text("/* fixture */\n", encoding="utf-8")
        (out / "model_call.c").write_text(
            f"void {invoke_name}(void **d) {{ (void)d; }}\n", encoding="utf-8")
        return {"n_args": 2, "n_outputs": 1, "ciface_name": ciface_name,
                "invoke_name": invoke_name}

    monkeypatch.setattr(session_bundle.c_runtime, "generate", fake_generate)
    out = tmp_path / "generated"
    manifest = session_bundle.generate(root, out)
    assert [row["entrypoint"] for row in manifest["programs"]] == [
        "merlin_stage_0_prefill", "merlin_stage_1_decode"]
    assert manifest["programs"][0]["invoke"] == "merlin_invoke"
    assert manifest["programs"][1]["invoke"] == "merlin_stage_1_invoke"
    assert "@merlin_stage_0_prefill" in (
        out / "stage_0_prefill" / "model.renamed.mlir").read_text()
    assert "@forward" not in (out / "stage_1_decode" / "model.renamed.mlir").read_text()
    recorded = json.loads((out / "session_build.json").read_text())
    assert recorded["bindings"][0]["target_program"] == "decode"

    fixture = out / "fixture.c"
    fixture.write_text('''
#include <stddef.h>
static float data[2];
void merlin_stage_0_reset(void) {} void merlin_stage_1_reset(void) {}
int merlin_stage_0_run(const void *w,long s,int v){return !w+s+v;}
int merlin_stage_1_run(const void *w,long s,int v){return !w+s+v;}
void *merlin_stage_0_input(int x){(void)x;return data;}
void *merlin_stage_1_input(int x){(void)x;return data;}
size_t merlin_stage_0_input_bytes(int x){(void)x;return sizeof(data);}
size_t merlin_stage_1_input_bytes(int x){(void)x;return sizeof(data);}
void *merlin_stage_0_output(int x){(void)x;return data;}
void *merlin_stage_1_output(int x){(void)x;return data;}
size_t merlin_stage_0_output_bytes(int x){(void)x;return sizeof(data);}
size_t merlin_stage_1_output_bytes(int x){(void)x;return sizeof(data);}
long merlin_stage_0_quality_steps(void){return 0;}
long merlin_stage_1_quality_steps(void){return 3;}
long merlin_stage_0_quality_min_cos_ppm(void){return 0;}
long merlin_stage_1_quality_min_cos_ppm(void){return 1000000;}
long merlin_stage_0_quality_max_rel_ppm(void){return 0;}
long merlin_stage_1_quality_max_rel_ppm(void){return 0;}
long merlin_stage_0_quality_top1(void){return 0;}
long merlin_stage_1_quality_top1(void){return 3;}
long merlin_stage_0_correctness_steps(void){return 0;}
long merlin_stage_1_correctness_steps(void){return 3;}
long merlin_stage_0_correctness_min_cos_ppm(void){return 0;}
long merlin_stage_1_correctness_min_cos_ppm(void){return 1000000;}
long merlin_stage_0_correctness_max_rel_ppm(void){return 0;}
long merlin_stage_1_correctness_max_rel_ppm(void){return 0;}
long merlin_stage_0_correctness_top1(void){return 0;}
long merlin_stage_1_correctness_top1(void){return 3;}
''', encoding="utf-8")
    proc = subprocess.run([
        "cc", "-std=c11", f"-I{out}", "-fsyntax-only", str(out / "merlin_session.c"),
        str(fixture),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_c_runtime_can_name_a_stage_entrypoint(monkeypatch, tmp_path):
    # Lock the generated trampoline contract without requiring a real safetensors capture.
    source = session_bundle.rename_forward("func.func @forward()", "merlin_stage_2_action")
    assert source == "func.func @merlin_stage_2_action()"
    with pytest.raises(ValueError, match="no @forward"):
        session_bundle.rename_forward("func.func @other()", "merlin_stage_2_action")
