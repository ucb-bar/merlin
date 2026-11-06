"""Offline tests for MLIR inventory and the explicit-noise inference boundary."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parent


def load_local(name):
    spec = importlib.util.spec_from_file_location(f"mtdit_test_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_named_matmul():
    source = b'''module {
      func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>,
                         %c: tensor<2x4xf32>) -> tensor<2x4xf32> {
        %r = linalg.matmul ins(%a, %b: tensor<2x3xf32>, tensor<3x4xf32>)
          outs(%c: tensor<2x4xf32>) -> tensor<2x4xf32>
        return %r : tensor<2x4xf32>
      }
    }'''
    report = load_local("inventory").inventory(source)
    assert report["verified"] is True
    assert report["op_counts"]["linalg.matmul"] == 1
    # Named Linalg operations also contain scalar regions.
    assert report["op_counts"]["arith.mulf"] == 1
    assert report["op_counts"]["arith.addf"] == 1
    assert report["tensor_dtype_occurrences"] == {"f32": 5}
    occurrences = {
        tuple(row["shape"]): row["count"]
        for row in report["tensor_shape_dtype_occurrences"]
    }
    assert occurrences == {(2, 3): 1, (3, 4): 1, (2, 4): 3}
    assert len(report["contractions"]) == 1
    assert report["contractions"][0]["recognition"] == "named"
    match = report["classification_by_op_kind"]["linalg.matmul"]
    assert match["geometric_family_candidates"] == ["projection_gemm"]
    assert match["f32_direct_support_no_quantization"] == "NOT_SUPPORTED"


def test_inventory_refuses_invalid_mlir():
    with pytest.raises(Exception, match="Unable to parse|failed|error"):
        load_local("inventory").inventory(b"module { invalid }")


def test_inference_runs_every_step_and_slices_explicit_noise(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("MTDIT_TORCHVISION_SCHEMA_COMPAT", "1")
    loader = load_local("loader")

    def forbid_construction(*args, **kwargs):
        raise AssertionError("test must not construct a real policy or CLIP encoder")

    monkeypatch.setattr(loader, "MultiTaskDiTPolicy", forbid_construction)
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", forbid_construction)
    calls, batches = [], []
    conditioning = torch.tensor([[0.5], [1.5]])

    class Scheduler:
        def set_timesteps(self, steps):
            assert steps == 3
            self.timesteps = torch.tensor([8, 4, 0])

        def step(self, prediction, timestep, sample):
            calls.append(("step", timestep, sample.clone()))
            return SimpleNamespace(prev_sample=sample - prediction / 4)

    class Encoder:
        def encode(self, batch):
            batches.append(batch)
            return conditioning

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.objective = SimpleNamespace(noise_scheduler=Scheduler(), num_inference_steps=3)
            self.observation_encoder = Encoder()
            self.config = SimpleNamespace(n_obs_steps=3, n_action_steps=2)

        def noise_predictor(self, sample, timesteps, *, conditioning_vec):
            assert conditioning_vec is conditioning
            assert timesteps.dtype == torch.long
            assert timesteps.device == sample.device
            assert timesteps.shape == (2,)
            assert torch.equal(timesteps, timesteps[:1].expand(2))
            calls.append(("predict", int(timesteps[0]), sample.clone()))
            return torch.ones_like(sample) * (timesteps[:, None, None] + 1)

    model = loader.Inference(Policy())
    assert model.timesteps == (8, 4, 0)
    images, state = torch.zeros(2, 3, 1, 3, 2, 2), torch.zeros(2, 3, 2)
    tokens, mask = torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long)
    noise = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2)
    original_noise = noise.clone()
    actual = model(images, state, tokens, mask, noise)
    assert [(kind, timestep) for kind, timestep, _ in calls] == [
        (kind, timestep) for timestep in (8, 4, 0) for kind in ("predict", "step")
    ]
    torch.testing.assert_close(calls[0][2], original_noise, rtol=0, atol=0)
    # n_obs_steps - 1 selects rows 2 and 3 after all three DDIM updates.
    torch.testing.assert_close(actual, (original_noise - 15 / 4)[:, 2:4], rtol=0, atol=0)
    assert actual.shape == (2, 2, 2)
    torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
    assert len(batches) == 1
    for key, value in ((loader.OBS_IMAGES, images), (loader.OBS_STATE, state),
                       (loader.OBS_LANGUAGE_TOKENS, tokens),
                       (loader.OBS_LANGUAGE_ATTENTION_MASK, mask)):
        assert batches[0][key] is value


def test_static_scheduler_lookup_matches_unmodified_ddim(monkeypatch):
    """Changing scalar lookup addressing must preserve diffusion numerics."""
    monkeypatch.setenv("MTDIT_TORCHVISION_SCHEMA_COMPAT", "1")
    from diffusers import DDIMScheduler
    loader = load_local("loader")

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(n_obs_steps=2, n_action_steps=3)
            self.objective = SimpleNamespace(
                noise_scheduler=DDIMScheduler(num_train_timesteps=100, prediction_type="epsilon"),
                num_inference_steps=10,
            )
            self.observation_encoder = SimpleNamespace(encode=lambda batch: batch[loader.OBS_STATE])

        def noise_predictor(self, sample, timesteps, *, conditioning_vec):
            return sample.sin() * 0.1 + conditioning_vec[:, :1, :1] * 0.01

    policy = Policy()
    original = loader.Inference(policy)
    frozen = loader.Inference(policy, static_scheduler=True)
    generator = torch.Generator().manual_seed(319)
    noise = torch.randn(2, 8, 4, generator=generator)
    state = torch.randn(2, 2, 4, generator=generator)
    inputs = (torch.zeros(2, 2, 1, 3, 2, 2), state,
              torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.long), noise)
    torch.testing.assert_close(frozen(*inputs), original(*inputs), atol=0, rtol=0)
    assert isinstance(policy.objective.noise_scheduler.alphas_cumprod, torch.Tensor)


def test_unresolved_census_counts_mangled_symbols():
    source = '''builtin.module {
      func.func private @aten_select_int(tensor<100xf32>) -> tensor<f32>
      func.func @forward(%table: tensor<100xf32>) -> tensor<f32> {
        %value = func.call @aten_select_int(%table) : (tensor<100xf32>) -> tensor<f32>
        func.return %value : tensor<f32>
      }
    }'''
    assert load_local("capture").unresolved_calls(source) == {"aten_select_int": 1}
